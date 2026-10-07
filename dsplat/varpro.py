"""Variable projection: streamed normal equations, coefficient elimination and acceptance.

The objective is mean(|A b - y|^2) + ridge |b|^2. Complex coefficients use the full
Gram G = A^H A / M; signed real coefficients use Re(G) and Re(q). A new geometry is
accepted only if its refitted objective is lower.
"""

from dataclasses import replace
from typing import NamedTuple

import torch

from .kernels.cublas import syrk
from .kernels.reduce import energy


class Fit(NamedTuple):
    """VarPro solution of one geometry; the field names match DSFWState."""

    coefficients: torch.Tensor
    prediction: torch.Tensor
    objective: float
    gram: torch.Tensor  # Re(A^H A) / M for real coefficients, A^H A / M otherwise
    rhs: torch.Tensor


POWER_STEPS = 24  # power iterations for the largest eigenvalue of a Gram
SHIFT_RETRIES = 4


def largest_eigenvalue(h):
    """Power-iteration estimate of the largest eigenvalue of a positive semidefinite h."""
    # A fixed non-constant start keeps the estimate reproducible without a generator.
    vector = (
        (torch.arange(len(h), device=h.device, dtype=torch.float64) * 0.6180339887) % 1 + 0.5
    ).to(h.dtype)
    for _ in range(POWER_STEPS):
        vector = h @ vector
        top = vector.norm()
        vector = vector / top.clamp(min=torch.finfo(top.dtype).tiny)
    return top


def shifted_cholesky(h, floor):
    """Cholesky factor of h + shift I in double precision, or None.

    The shift starts at floor times a power-iteration estimate of lambda_max(h)
    and grows while factorization fails, at most SHIFT_RETRIES times. Double precision keeps the
    rounding of the factorization below the shift, so the first shift normally holds.
    """
    matrix = h.to(torch.complex128 if h.is_complex() else torch.float64, copy=True)
    shift = floor * largest_eigenvalue(matrix)
    for _ in range(SHIFT_RETRIES):
        matrix.diagonal().add_(shift)
        factor, info = torch.linalg.cholesky_ex(matrix)
        if int(info) == 0:
            return factor
        shift = shift * 9  # added to the shifts already on the diagonal: 1, 10, 91, ... x
    return None


def solve(h, rhs):
    """Hermitian solve; a singular system is shifted to where torch.linalg.pinv would truncate.

    A support with a surfel that no aperture sees (a zero column) or two atoms at one
    position has a singular Gram, and Cholesky fails. The shift len(h) * eps *
    lambda_max is the eigenvalue level the pseudoinverse drops; it changes the
    solution only in near-null directions, which do not change the prediction. An
    eigendecomposition would be exact there, but on a complex Gram of thousands of
    atoms the GPU eigensolver fails and LAPACK then takes half a minute per solve.
    """
    q = rhs[:, None] if rhs.ndim == 1 else rhs
    chol, info = torch.linalg.cholesky_ex(h)
    if int(info) != 0:
        chol = shifted_cholesky(h, len(h) * torch.finfo(h.real.dtype).eps)
    if chol is None:  # every column is zero: nothing to fit
        return torch.zeros_like(rhs)
    x = torch.cholesky_solve(q.to(chol.dtype), chol).to(q.dtype)
    return x[:, 0] if rhs.ndim == 1 else x


FOLD = 8  # float32 blocks summed before each float64 fold of the Gram


def row_chunk(model, columns, config):
    """Observation rows per streamed block: bounded elements, whole apertures."""
    chunk = min(config.row_chunk, max(1, config.dictionary_elements // columns))
    return max(model.row_alignment, chunk - chunk % model.row_alignment)


def normal_equations(model, params, target, config):
    """G = A^H A / M and q = A^H y / M, streamed over projected row blocks.

    Real blocks accumulate the upper triangle in float32 by SYRK and fold into a
    float64 total every FOLD blocks.
    """
    count = len(params["centers"])
    chunk = row_chunk(model, count, config)
    real = config.real
    stream = model.blocks(params, target, real, chunk)
    if not real:
        gram = target.new_zeros((count, count))
        rhs = target.new_zeros(count)
        for a, y in stream:
            gram.addmm_(a.H, a)
            rhs.addmv_(a.H, y)
        return (gram + gram.H) / (2 * model.size), rhs / model.size
    like = target.real
    part = like.new_zeros((count, count))
    total = torch.zeros((count, count), dtype=torch.float64, device=like.device)
    rhs = torch.zeros(count, dtype=torch.float64, device=like.device)
    for index, (a, y) in enumerate(stream):
        syrk(a.contiguous(), part)
        rhs += a.T @ y
        if index % FOLD == FOLD - 1:
            total += part
            part.zero_()
    total += part
    gram = (torch.triu(total) + torch.triu(total, 1).T) / model.size
    return gram.float(), (rhs / model.size).float()


def fit_coefficients(gram, rhs, config):
    """Minimizer of b^H G b - 2 Re(b^H q) + ridge |b|^2 in the configured domain."""
    if config.real:
        gram, rhs = gram.real, rhs.real
    return solve(
        gram + config.ridge * torch.eye(len(gram), dtype=gram.dtype, device=gram.device), rhs
    )


def rendered_loss(model, params, coefficients, target, ridge):
    """Prediction and the complete objective for the given coefficients."""
    prediction = model.render(params, coefficients)
    loss = energy(prediction, target) / model.size + ridge * coefficients.abs().square().sum()
    return prediction, float(loss)


@torch.no_grad()
def varpro(model, params, target, config):
    """Coefficients, prediction, objective, Gram and right-hand side of one geometry."""
    gram, rhs = normal_equations(model, params, target, config)
    b = fit_coefficients(gram, rhs, config)
    return Fit(b, *rendered_loss(model, params, b, target, config.ridge), gram, rhs)


@torch.no_grad()
def exchange_varpro(state, proposal, slot, config):
    """VarPro after one atom changed: only its Gram column and right-hand side are new."""
    model, target = state.model, state.target
    atom = {k: v[slot : slot + 1] for k, v in proposal.items()}
    column = model.render(atom, target.real.new_ones(1))
    cross = model.correlate(proposal, column)[0] / model.size
    value = (column.conj() * target).sum() / model.size
    if config.real:
        cross, value = cross.real, value.real
    gram, rhs = state.gram.clone(), state.rhs.clone()
    gram[:, slot] = cross
    gram[slot, :] = cross.conj()
    gram[slot, slot] = cross[slot].real
    rhs[slot] = value
    b = fit_coefficients(gram, rhs, config)
    return Fit(b, *rendered_loss(model, proposal, b, target, config.ridge), gram, rhs)


def screen_bar(screen, params, config):
    """A (model, target) aperture subset with the subset objective of params, or None.

    The bar is computed once per state, so every proposal from that state is
    screened against the same refit.
    """
    if screen is None:
        return None
    model, target = screen
    return model, target, varpro(model, params, target, config).objective


@torch.no_grad()
def accept(state, proposal, config, screen=None):
    """The state with proposal installed if its refitted objective is lower, else None.

    A single changed atom refits only its Gram column. screen, from screen_bar on
    state.params, first rejects a proposal that is clearly worse on its aperture
    subset; ambiguous small differences go to the full objective, but a subset can
    still miss a real improvement.
    """
    changed = torch.zeros(len(state.coefficients), dtype=torch.bool, device=state.target.device)
    for key, value in proposal.items():
        changed |= (value != state.params[key]).reshape(len(changed), -1).any(-1)
    slots = changed.nonzero().flatten().tolist()
    if not slots:
        return None
    if screen is not None:
        model, target, bar = screen
        if varpro(model, proposal, target, config).objective > bar + 1e-4 * state.power:
            return None
    if len(slots) == 1:
        fit = exchange_varpro(state, proposal, slots[0], config)
    else:
        fit = varpro(state.model, proposal, state.target, config)
    if not fit.objective < state.objective:
        return None
    return replace(state, params=proposal, **fit._asdict())


def pcg(operator, rhs, diagonal, *, iterations=32, tolerance=1e-3):
    """Jacobi-preconditioned conjugate gradients from zero."""
    x = torch.zeros_like(rhs)
    norm = rhs.norm()
    r = rhs.clone()
    z = r / diagonal
    p = z.clone()
    rz = r @ z
    for _ in range(iterations):
        hp = operator(p)
        curvature = p @ hp
        if not curvature > 0:
            break
        alpha = rz / curvature
        x += alpha * p
        r -= alpha * hp
        if r.norm() <= tolerance * norm:
            break
        z = r / diagonal
        next_rz = r @ z
        p = z + (next_rz / rz) * p
        rz = next_rz
    return x
