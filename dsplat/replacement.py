"""Fixed-budget support replacement.

Both policies install a new support only if its refitted objective is lower.
utility replaces the least useful atom by the best certificate candidate. merged pools many candidates with
the whole support, ranks every atom by the cost of deleting it from one pooled
fit, and keeps the budget's cheapest-to-lose complement.
"""

from dataclasses import replace

import torch

from .certificate import certificate
from .parameters import periodic_delta
from .varpro import (
    Fit,
    accept,
    fit_coefficients,
    rendered_loss,
    row_chunk,
    screen_bar,
    shifted_cholesky,
    varpro,
)


def _feasible(state, pool, owner):
    """The pool as feasible atoms; fields the pool lacks copy slot owner."""
    count = len(pool["centers"])
    result = {}
    for key, spec in state.parameters.items():
        reference = state.params[key][owner].expand(count, *state.params[key].shape[1:])
        result[key] = spec.project(pool.get(key, reference), reference)
    return result


# --- utility: one slot per step --------------------------------------------------------------


def utility(gram, coefficients):
    """Half relative amplitude, half unregularized projection efficiency of every atom.

    For a singular Gram a column is redundant exactly when its coordinate takes part
    in the nullspace; the other columns use G+ instead of G^-1.
    """
    chol, info = torch.linalg.cholesky_ex(gram)
    if int(info) == 0:
        inverse_diag = torch.cholesky_inverse(chol).diag().real
        efficiency = (1 / (inverse_diag * gram.diag().real).clamp(min=1)).sqrt()
    else:
        values, vectors = torch.linalg.eigh(gram)
        eps = torch.finfo(values.dtype).eps
        active = values > values.abs().max() * len(values) * eps
        inv = torch.where(active, values.clamp(min=torch.finfo(values.dtype).tiny).reciprocal(), 0)
        inverse_diag = (vectors.abs().square() * inv[None]).sum(1)
        null_weight = vectors[:, ~active].abs().square().sum(1)
        efficiency = (1 / (inverse_diag * gram.diag().real).clamp(min=1)).sqrt()
        efficiency = torch.where(null_weight > len(values) * eps * 10, 0, efficiency)
        efficiency = torch.where(gram.diag().real > 0, efficiency, 0)
    magnitude = coefficients.abs()
    return 0.5 * magnitude / magnitude.max().clamp(min=1e-30) + 0.5 * efficiency


@torch.no_grad()
def replace_one(state, config, pool, screen=None):
    """Try the best pool candidate in the least useful slot; the state and atoms replaced."""
    owner = int(utility(state.gram, state.coefficients).argmin())
    candidate = _feasible(state, pool, owner)
    residual = state.target - state.prediction
    scores = certificate(state.model, candidate, residual, config.coefficients)
    if config.min_separation > 0:
        centers, period = state.params["centers"], state.parameters["centers"].period
        for start in range(0, len(scores), 256):
            delta = periodic_delta(
                candidate["centers"][start : start + 256, None] - centers[None], period
            )
            near = delta.norm(dim=-1).min(1).values < config.min_separation
            scores[start : start + 256][near] = -torch.inf
    scores[~torch.isfinite(scores)] = -torch.inf
    index = int(scores.argmax())
    if not scores[index] > 0:
        return state, 0
    proposal = {key: value.clone() for key, value in state.params.items()}
    for key in proposal:
        proposal[key][owner] = candidate[key][index]
    accepted = accept(state, proposal, config, screen_bar(screen, state.params, config))
    return (state, 0) if accepted is None else (accepted, 1)


# --- merged: pool candidates, fit once, prune back to the budget -----------------------------


def regularized_inverse(h, floor):
    """Inverse of h + floor * estimated lambda_max(h) * I in double precision, or None.

    The shift lifts every eigenvalue above the level where a Gram accumulated from
    single-precision kernels stops being meaningful, so the inverse ranks unknowns
    without amplifying that rounding.
    """
    factor = shifted_cholesky(h, floor)
    return None if factor is None else torch.cholesky_inverse(factor)


def backward_elimination(inverse, rhs, count, limited=None, limit=0):
    """Remove count unknowns one at a time from the least squares with this inverse.

    With W the inverse of the current system and b = W q, deleting unknown j and
    refitting the rest raises the objective by |b_j|^2 / W_jj. Each step deletes the
    cheapest unknown and continues with the exact inverse of the remaining system,
    W - w w^H / W_jj with w = W[:, j], kept as rank-one corrections so one step costs
    O(n * steps). At most limit of the unknowns marked in limited are removed.
    Returns the removed indices in order.
    """
    size = len(inverse)
    b = inverse @ rhs.to(inverse.dtype)
    diagonal = inverse.diagonal().real.clone()
    alive = torch.ones(size, device=inverse.device, dtype=torch.bool)
    updates = inverse.new_zeros((size, count))
    removed = torch.zeros(count, device=inverse.device, dtype=torch.long)
    spent = torch.zeros((), device=inverse.device, dtype=torch.long)
    tiny = torch.finfo(diagonal.dtype).tiny
    for step in range(count):
        cost = b.abs().square() / diagonal.clamp(min=tiny)
        blocked = ~alive if limited is None else ~alive | (limited & (spent >= limit))
        index = cost.masked_fill(blocked, torch.inf).argmin()
        column = inverse[:, index] - updates[:, :step] @ updates[index, :step].conj()
        pivot = column[index].real.clamp(min=tiny)
        b = b - column * (b[index] / pivot)
        scaled = column / pivot.sqrt()
        updates[:, step] = scaled
        diagonal = diagonal - scaled.abs().square()
        # A removed unknown stays masked; a unit diagonal only keeps its cost finite.
        alive[index] = False
        diagonal[index] = 1.0
        b[index] = 0
        removed[step] = index
        if limited is not None:
            spent = spent + limited[index]
    return removed


def _distinct(scores, points, occupied, period, separation, width):
    """Best-scoring candidates colliding neither with the support nor with each other.

    A collision is a distance below separation, or exactly zero: an atom pooled
    twice would make the pooled system singular for no gain.
    """

    def distance(a, b):
        return periodic_delta(a[:, None] - b[None], period).norm(dim=-1)

    order = scores.argsort(descending=True, stable=True)
    order = order[torch.isfinite(scores[order])]
    kept, found = [], 0
    # Blocks bound the pairwise distances; later blocks also avoid earlier picks.
    for start in range(0, len(order), 1024):
        if found >= width:
            break
        ids = order[start : start + 1024]
        block = points[ids]
        near = distance(block, occupied).min(1).values
        free = (near >= separation) & (near > 0)
        for previous in kept:
            near = distance(block, points[previous]).min(1).values
            free &= (near >= separation) & (near > 0)
        inside = distance(block, block)
        clash = ((inside < separation) | (inside == 0)).triu(1)
        if bool(clash[free][:, free].any()):
            # Greedy by score: a candidate yields only to a kept better one.
            for i in free.nonzero().flatten().tolist():
                if bool((clash[:i, i] & free[:i]).any()):
                    free[i] = False
        kept.append(ids[free])
        found += int(free.sum())
    return torch.cat(kept)[:width] if kept else order[:0]


def pooled_statistics(state, pool, config):
    """Gram and right-hand side over the support followed by the pooled new atoms.

    Only the new columns are streamed; the support block is the state's.
    """
    model, count = state.model, len(state.coefficients)
    total, added = len(pool["centers"]), len(pool["centers"]) - count
    gram = state.gram.new_empty((total, total))
    rhs = state.rhs.new_empty(total)
    gram[:count, :count], rhs[:count] = state.gram, state.rhs
    cross = state.gram.new_zeros((total, added))
    right = state.rhs.new_zeros(added)
    real = config.real
    for a, y in model.blocks(pool, state.target, real, row_chunk(model, total, config)):
        new = a[:, count:]
        cross.addmm_(a.H, new)
        right.addmv_(new.H, y)
    cross /= model.size
    gram[:, count:], gram[count:, :] = cross, cross.H
    gram[count:, count:] = (cross[count:] + cross[count:].H) * 0.5
    rhs[count:] = right / model.size
    return gram, rhs


@torch.no_grad()
def merge_support(state, config, pool, scores=None):
    """Pool the best candidates with the support, fit once, prune back to the budget.

    One regularized fit over all old and new atoms ranks them by the cost of deleting
    them; the cheapest leave until the budget is met, and surviving new atoms take the
    slots of old atoms that left. The pooled fit only ranks: the resulting support
    needs a lower rendered objective, and a rejected support is retried with half as
    many old atoms allowed to leave. scores rank the pool (default: its certificates).
    Returns the state and the number of atoms replaced.
    """
    centers = state.params["centers"]
    count = len(centers)
    candidate = _feasible(state, pool, 0)
    if scores is None:
        residual = state.target - state.prediction
        scores = certificate(state.model, candidate, residual, config.coefficients)
    chosen = _distinct(
        scores, candidate["centers"], centers, state.parameters["centers"].period,
        config.min_separation, config.batch,
    )  # fmt: skip
    if not len(chosen):
        return state, 0
    pool = {key: torch.cat((value, candidate[key][chosen])) for key, value in state.params.items()}
    gram, rhs = pooled_statistics(state, pool, config)
    real = config.real
    h = (gram.real if real else gram) + config.ridge * torch.eye(
        len(gram), dtype=gram.dtype, device=gram.device
    )
    # Regularized at the level where the solver truncates the spectrum, so the
    # order does not follow rounding of the Gram.
    inverse = regularized_inverse(h, len(h) * torch.finfo(h.real.dtype).eps)
    if inverse is None:
        return state, 0
    q = rhs.real if real else rhs
    old = torch.arange(len(q), device=q.device) < count
    limit = None
    while True:
        removed = backward_elimination(
            inverse, q, len(chosen), None if limit is None else old, limit or 0
        )
        gone = torch.zeros(len(q), device=q.device, dtype=torch.bool)
        gone[removed] = True
        owners = gone[:count].nonzero().flatten()
        if not len(owners):
            return state, 0
        ids = torch.arange(count, device=q.device)
        ids[owners] = (~gone[count:]).nonzero().flatten() + count
        trial = {key: value[ids] for key, value in pool.items()}
        # New atoms copied slot 0's frozen components; they take those of the slot
        # they fill, and then their pooled columns no longer apply.
        moved = False
        for key, spec in state.parameters.items():
            if not spec.mask(trial[key]).all():
                fixed = spec.project(trial[key][owners], state.params[key][owners])
                moved = moved or not torch.equal(fixed, trial[key][owners])
                trial[key][owners] = fixed
        if moved:
            fit = varpro(state.model, trial, state.target, config)
        else:
            trial_gram, trial_rhs = gram[ids[:, None], ids[None, :]], rhs[ids]
            b = fit_coefficients(trial_gram, trial_rhs, config)
            fit = Fit(b, *rendered_loss(state.model, trial, b, state.target, config.ridge),
                      trial_gram, trial_rhs)  # fmt: skip
        if fit.objective < state.objective:
            return replace(state, params=trial, **fit._asdict()), len(owners)
        limit = len(owners) // 2
        if limit < 1:
            return state, 0
