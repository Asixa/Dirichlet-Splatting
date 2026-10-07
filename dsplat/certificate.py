"""Residual certificate: normalized correlations, candidate grids, peaks and continuous refinement."""

from dataclasses import dataclass
from math import ceil

import torch

from .kernels import search


def certificate(model, params, residual, mode, *, gradient=False, normals=False):
    """Squared normalized certificate |s|^2 / (|a|^2 |r|^2) per atom, and its gradient.

    s = a^H r for complex coefficients and Re(a^H r) for real ones. It ranks insertions;
    it is not a guaranteed decrease after a refit. With gradient, also returns the
    explicit gradient by centers, and by normals when normals is set.
    """
    if gradient:
        c, n, dc, dn = model.correlate_gradient(params, residual, include_normals=normals)
    else:
        c, n = model.correlate(params, residual)
    # Small per-candidate algebra in double keeps the squared norm floor representable.
    c, n = c.to(torch.complex128), n.double().clamp(min=1e-60)
    power = residual.abs().square().sum().double().clamp(min=1e-30)
    s = c.real if mode == "real" else c
    value = s.abs().square() / n / power
    if not gradient:
        return value.float()
    dc, dn = dc.to(torch.complex128), torch.where((n > 1e-60)[:, None], dn.double(), 0)
    ds = 2 * s[:, None] * dc.real if mode == "real" else 2 * (c.conj()[:, None] * dc).real
    return value.float(), ((ds - value[:, None] * power * dn) / n[:, None] / power).float()


@dataclass(frozen=True)
class CertificateSearchConfig:
    peaks: int = 1024  # separated coarse peaks refined per search
    separation: float = 0.0005  # center units, for coarse peak selection
    iterations: int = 40  # quasi-Newton ascent steps
    trust_radius: float = 1.0  # times each parameter's scale
    backtracks: int = 8


def candidate_grid(lower, upper, spacing, device="cuda"):
    """Regular grid including both box corners; the actual pitch never exceeds spacing."""
    axes = [
        torch.linspace(a, b, 1 + ceil((b - a) / spacing), device=device)
        for a, b in zip(lower, upper, strict=True)
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(axes))


PEAK_POOL = 64  # candidates kept per wanted peak before suppression


def select_peaks(scores, centers, count, separation):
    """Greedy nonmaximum suppression: best-first indices at least separation apart."""
    if separation == 0:
        order = scores.argsort(descending=True, stable=True)[:count]
        return order[torch.isfinite(scores[order])]
    wanted = min(count, len(scores))
    if len(scores) > PEAK_POOL * wanted:
        # Every candidate at or above a score level is picked or suppressed before any
        # below it, so when the pool already yields every peak the rest cannot change
        # the result. Index order keeps the tie-breaking.
        level = scores.topk(PEAK_POOL * wanted).values[-1]
        pool = (scores >= level).nonzero().flatten()
        chosen = search.peaks(scores[pool], centers[pool], wanted, separation)
        if len(chosen) == wanted:
            return pool[chosen]
    return search.peaks(scores, centers, count, separation)


def fibonacci_sphere(count, device="cuda"):
    """Nearly uniform unit vectors [count, 3]."""
    j = torch.arange(count, device=device, dtype=torch.float32)
    z = 1 - 2 * (j + 0.5) / count
    angle = j * (torch.pi * (3 - 5**0.5))
    radius = (1 - z.square()).sqrt()
    return torch.stack((radius * angle.cos(), radius * angle.sin(), z), -1)


@torch.no_grad()
def orient_candidates(model, params, residual, count, mode, table=None, chunk=8192):
    """Give every position the best normal of its own and a count-normal sphere bank.

    Returns the candidates with the chosen normals and their squared certificates.
    """
    centers = params["centers"]
    bank = fibonacci_sphere(count, centers.device)
    power = residual.abs().square().sum().double().clamp(min=1e-30)
    scores, normals = [], []
    for start in range(0, len(centers), chunk):
        block = {k: v[start : start + chunk] for k, v in params.items()}
        c, n = model.correlate_orientations(block, residual, bank, table)
        # Double keeps the 1e-60 floor of a back-facing atom's zero norm representable.
        c, n = c.to(torch.complex128), n.double()
        s = c.real if mode == "real" else c
        best, ids = (s.abs().square() / n.clamp(min=1e-60) / power).max(1)
        normal = block["normals"].clone()
        normal[ids > 0] = bank[ids[ids > 0] - 1]
        scores.append(best.float())
        normals.append(normal)
    return {**params, "normals": torch.cat(normals)}, torch.cat(scores)


@torch.no_grad()
def search_certificate(
    model, residual, candidates, center_spec, config, mode, *, normal_spec=None,
    initial_scores=None, full_model=None, full_residual=None,
):  # fmt: skip
    """Refine separated peaks in xyz (and unit normals) by batched quasi-Newton ascent.

    Each candidate backtracks independently inside the parameter constraints. The
    coarse peaks are kept beside the refined ones, and all are reranked on
    full_model and full_residual when given. Returns parameters and scores, best first.
    """
    reference = candidates["centers"]
    feasible = dict(candidates, centers=center_spec.project(reference, reference))
    specs = {"centers": center_spec}
    if normal_spec is not None:
        specs["normals"] = normal_spec
        feasible["normals"] = normal_spec.project(candidates["normals"], candidates["normals"])
    masks = {key: spec.mask(feasible[key]) for key, spec in specs.items()}
    controls = reference.new_zeros((4, 6))  # scale, mask, lower, upper per coordinate
    controls[0, :3] = center_spec.scale
    controls[1, :3] = masks["centers"]
    controls[2, :3] = reference.new_tensor(
        -torch.inf if center_spec.lower is None else center_spec.lower
    )
    controls[3, :3] = reference.new_tensor(
        torch.inf if center_spec.upper is None else center_spec.upper
    )
    if normal_spec is not None:
        controls[0, 3:], controls[1, 3:] = normal_spec.scale, 1

    def ascent(params, gradient):
        """Negative gradient in scaled tangent coordinates, the quantity minimized."""
        parts = [
            spec.tangent(gradient[:, 3 * i : 3 * i + 3], params[key]) * masks[key] * spec.scale
            for i, (key, spec) in enumerate(specs.items())
        ]
        return -torch.cat(parts, -1)

    def score(params, gradient=False):
        return certificate(
            model, params, residual, mode, gradient=gradient, normals="normals" in specs
        )

    coarse_scores = score(feasible) if initial_scores is None else initial_scores
    ids = select_peaks(coarse_scores, feasible["centers"], config.peaks, config.separation)
    coarse = {k: v[ids] for k, v in feasible.items()}
    params = {k: v.clone() for k, v in coarse.items()}
    value, grad = score(params, True)
    grad = ascent(params, grad)
    count, dim = grad.shape
    eye = torch.eye(dim, device=reference.device).expand(count, dim, dim)
    inverse = eye.clone()
    for _ in range(config.iterations):
        direction = search.direction(
            inverse, grad, params["normals"], controls[1], config.trust_radius
        )
        old = {k: v.clone() for k, v in params.items()}
        accepted = torch.zeros(count, dtype=torch.bool, device=reference.device)
        for backtrack in range(config.backtracks):
            centers, normals = search.trial(
                old["centers"], old["normals"], direction, controls, 0.5**backtrack
            )
            trial = dict(old, centers=centers, normals=normals)
            pending = torch.where(~accepted)[0]
            trial_value = score({k: v[pending] for k, v in trial.items()})
            keep = torch.zeros_like(accepted)
            keep[pending] = torch.isfinite(trial_value) & (trial_value > value[pending])
            for key in specs:
                params[key][keep] = trial[key][keep]
            accepted |= keep
            if accepted.all():
                break
        if not accepted.any():
            break
        new_value, new_grad = score(params, True)
        new_grad = ascent(params, new_grad)
        # Damped BFGS update of the inverse Hessian, per candidate.
        step = torch.cat([(params[k] - old[k]) / spec.scale for k, spec in specs.items()], -1)
        change = new_grad - grad
        sy = (step * change).sum(-1)
        valid = accepted & (sy > 1e-8 * step.norm(dim=-1) * change.norm(dim=-1)) & (sy > 1e-20)
        rho = torch.where(valid, 1 / sy.clamp(min=1e-30), 0)
        left = eye - rho[:, None, None] * step[:, :, None] * change[:, None, :]
        updated = (
            left @ inverse @ left.transpose(-1, -2)
            + rho[:, None, None] * step[:, :, None] * step[:, None, :]
        )
        inverse = torch.where(valid[:, None, None], updated, inverse)
        value, grad = new_value, new_grad
    combined = {k: torch.cat((coarse[k], params[k])) for k in params}
    if full_model is None:
        scores = torch.cat((coarse_scores[ids], value))
    else:
        scores = certificate(full_model, combined, full_residual, mode)
    order = scores.argsort(descending=True, stable=True)
    return {k: v[order] for k, v in combined.items()}, scores[order]
