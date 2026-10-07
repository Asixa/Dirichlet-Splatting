"""Spatial candidate search: hierarchical 3D grids and continuous certificate refinement.

Each step, before replacement, this builds a fresh candidate set inside the box of
the centers' bounds: a range-gated coarse grid plus cached peaks, finer cubes around
separated peaks, quasi-Newton refinement on a subset of apertures, then on all of
them. Trainable unit normals use a sphere bank and joint refinement; frozen fields
copy slot 0. The model provides the range gate, the FFT range table and the
normal-bank scores (FMCWModel).
"""

from dataclasses import dataclass, field, replace
from math import ceil, prod

import torch

from .certificate import (
    CertificateSearchConfig,
    candidate_grid,
    certificate,
    orient_candidates,
    search_certificate,
    select_peaks,
)


@dataclass(frozen=True)
class SpatialSearchConfig:
    coarse_spacing: float = 0.002  # metres
    # (radius, pitch) of each finer level; every level samples a cube around every peak.
    levels: tuple[tuple[float, float], ...] = ((0.002, 0.0005), (0.0005, 0.000125))
    proposal_apertures: int = 384  # apertures scoring the grids
    refine_apertures: int = 768  # apertures of the first continuous refinement
    max_candidates: int = 2_000_000  # points per grid level
    range_power: float = 0.03  # range gate threshold relative to each profile's peak
    range_margin_cells: float = 1.5  # range gate dilation in range resolutions
    range_fraction: float = 0.7  # share of apertures a candidate must hit
    ungated_after: int = 2  # failed steps after which the gate is dropped
    normal_candidates: int = 24  # sphere-bank normals tried per position; 0 keeps the slot's
    joint_normals: bool = True  # refine trainable normals with the positions
    fft_oversample: int = 8  # FFT-interpolated proposal scores; 0 is exact
    cache_peaks: int = 256  # peak locations carried to the next step
    full_peaks: int = 64  # peaks refined again on every aperture
    full_iterations: int = 8
    continuous: CertificateSearchConfig = field(default_factory=CertificateSearchConfig)


@torch.no_grad()
def propose(state, config, search):
    """Freshly refined candidates, their scores best first, and the peaks for the next step.

    After failed steps (state.search_failures) the coarse grid gets finer and shifts
    by a fraction of a cell, more peaks are kept and the range gate loosens.
    """
    centers = state.params["centers"]
    spec, normal_spec = state.parameters["centers"], state.parameters["normals"]
    free_normals = bool(normal_spec.mask(state.params["normals"]).any())
    refine_normals = normal_spec if free_normals and search.joint_normals else None
    failures, mode = state.search_failures, config.coefficients
    residual = state.target - state.prediction
    ids, model = state.model.sample_apertures(
        search.proposal_apertures, config.seed + state.iteration
    )
    r = residual[ids]
    # The FFT table only serves the normal bank; without it the scores stay exact.
    banked = free_normals and search.normal_candidates
    table = (
        model.range_table(r, search.fft_oversample) if banked and search.fft_oversample else None
    )

    def score(candidates):
        if banked:
            oriented, scores = orient_candidates(
                model, candidates, r, search.normal_candidates, mode, table
            )
            candidates["normals"] = oriented["normals"]
            return scores
        return certificate(model, candidates, r, mode)

    def atoms(points, normals=None):
        """Feasible atoms at these points; other fields copy slot 0."""
        result = {
            key: value[:1].expand(len(points), *value.shape[1:]).clone()
            for key, value in state.params.items()
        }
        result["centers"] = spec.project(points, result["centers"])
        if normals is not None:
            result["normals"] = normal_spec.project(normals, result["normals"])
        return result

    peaks = min(search.continuous.peaks * (1 + failures), 4 * search.continuous.peaks)
    separation = search.continuous.separation
    lower = centers.new_tensor(spec.lower).expand(3).clone()
    upper = centers.new_tensor(spec.upper).expand(3).clone()
    frozen = ~spec.mask(centers)
    lower[frozen] = upper[frozen] = centers[0, frozen]

    spacing = search.coarse_spacing / (1 + min(failures, 2))
    while (
        prod(1 + ceil(float(b - a) / spacing) for a, b in zip(lower, upper, strict=True))
        > search.max_candidates
    ):
        spacing *= 1.1
    points = candidate_grid(lower.tolist(), upper.tolist(), spacing)
    if failures:
        shift = points.new_tensor([0.37, 0.61, 0.23]) * spacing * ((failures % 3) + 1) / 3
        points = spec.project(points + shift, points)
    if failures < search.ungated_after:
        keep = model.range_gate(
            points,
            state.target[ids],
            relative_power=search.range_power / (1 + failures),
            margin_cells=search.range_margin_cells * (1 + failures),
            min_fraction=search.range_fraction,
        )
        if keep.any():  # an empty measurement gate never stops the search
            points = points[keep]
    cached = 0
    if state.peak_cache is not None:
        cached = min(len(state.peak_cache["centers"]), search.max_candidates - len(points))
        points = torch.cat((points, state.peak_cache["centers"][:cached]))
    candidates = atoms(points)
    if cached:
        normals = state.peak_cache["normals"][:cached]
        candidates["normals"][-cached:] = normal_spec.project(
            normals, candidates["normals"][-cached:]
        )

    scores = score(candidates)
    selected = select_peaks(scores, candidates["centers"], peaks, separation)
    candidates = {k: v[selected] for k, v in candidates.items()}
    for radius, pitch in search.levels:
        offset_lower = [-radius if active else 0 for active in ~frozen]
        offset_upper = [radius if active else 0 for active in ~frozen]
        while (
            prod(1 + ceil((b - a) / pitch) for a, b in zip(offset_lower, offset_upper, strict=True))
            > search.max_candidates
        ):
            pitch *= 1.1
        offsets = candidate_grid(offset_lower, offset_upper, pitch)
        # Highest-ranked regions keep priority within the hard per-level budget.
        candidates = {
            k: v[: max(1, search.max_candidates // len(offsets))] for k, v in candidates.items()
        }
        points = (candidates["centers"][:, None] + offsets[None]).reshape(-1, 3)
        candidates = atoms(points, candidates["normals"].repeat_interleave(len(offsets), dim=0))
        scores = score(candidates)
        selected = select_peaks(scores, candidates["centers"], peaks, min(separation, radius / 2))
        candidates = {k: v[selected] for k, v in candidates.items()}
    # FFT-interpolated scores only propose; the refinement rescores exactly.
    initial = scores[selected] if table is None else None
    if search.refine_apertures > search.proposal_apertures:
        refine_ids, model = state.model.sample_apertures(
            search.refine_apertures, config.seed + state.iteration + 1000003
        )
        r, initial = residual[refine_ids], None
    params, scores = search_certificate(
        model, r, candidates, spec, replace(search.continuous, peaks=peaks, separation=0), mode,
        normal_spec=refine_normals, initial_scores=initial,
        full_model=state.model, full_residual=residual,
    )  # fmt: skip
    if search.full_iterations:
        full = select_peaks(
            scores, params["centers"], search.full_peaks, min(separation, spec.scale / 2)
        )
        refined, refined_scores = search_certificate(
            state.model, residual, {k: v[full] for k, v in params.items()}, spec,
            replace(search.continuous, peaks=search.full_peaks, iterations=search.full_iterations, separation=0),
            mode, normal_spec=refine_normals, initial_scores=scores[full],
        )  # fmt: skip
        # The screened finalists stay in the pool beside the fully refined ones.
        scores = torch.cat((refined_scores, scores))
        order = scores.argsort(descending=True, stable=True)
        params = {k: torch.cat((v, params[k]))[order] for k, v in refined.items()}
        scores = scores[order]
    if not search.cache_peaks:
        return params, scores, None
    cache = select_peaks(scores, params["centers"], search.cache_peaks, separation)
    return params, scores, {k: v[cache].clone() for k, v in params.items()}
