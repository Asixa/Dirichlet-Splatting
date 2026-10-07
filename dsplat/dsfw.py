"""DSFW: replace support, refine geometry, accept on the full objective, repeat.

Every function takes a state and returns a new one; tensors are shared, never
written in place. The defaults are the fast 3D configuration: signed-real
coefficients, merged replacement pooling up to 1024 candidates per step, and joint
Gauss-Newton built on 1024 apertures that backs off while it gains less than the
replacement.
"""

from dataclasses import dataclass, replace

import torch

from .parameters import ParameterConfig
from .refinement import dense_slide, joint_slide, trainable
from .replacement import merge_support, replace_one
from .spatial import propose
from .varpro import varpro

LM_WAIT_LIMIT = 8  # longest run of skipped geometry steps; see DSFWConfig.lm_gain_ratio


@dataclass(frozen=True)
class DSFWConfig:
    iterations: int = 80
    ridge: float = 0.0  # on the mean-square data term
    coefficients: str = "real"  # "real" (signed) or "complex"
    replacement: str = "merged"  # "merged" or "utility" (one slot per step)
    batch: int = 1024  # merged: candidates pooled with the support per step
    min_separation: float = 0.0  # replacement candidates' distance from the support
    lm_solver: str = "joint_cg"  # "joint_cg" or "dense"
    lm_steps: int = 1
    # Skip the geometry step while it lowers the objective by less than this fraction
    # of the replacement's gain: for one step, then two, doubling up to LM_WAIT_LIMIT,
    # until it pays again. Zero runs it every step.
    lm_gain_ratio: float = 1.0
    lm_rows: int = 4096  # dense: sampled observation rows
    # Seeded apertures (the first axis of an FMCW scan) of the joint_cg proposal system,
    # and of a subset check that rejects clearly worse proposals before the full
    # evaluation. 0, at least the aperture count, or a SpectralModel uses everything.
    joint_apertures: int = 1024
    screen_apertures: int = 1024
    cg_iterations: int = 16
    cg_tolerance: float = 1e-3
    cg_probes: int = 2
    damping: float = 1e-3
    trust_radius: float = 1.5  # dimensionless; times each ParameterConfig.scale
    row_chunk: int = 65536
    dictionary_elements: int = 32_000_000  # observations x atoms per streamed block
    seed: int = 0

    @property
    def real(self):
        """Signed-real coefficients; a misspelt mode raises KeyError."""
        return {"real": True, "complex": False}[self.coefficients]


@dataclass(frozen=True)
class DSFWState:
    model: object
    target: torch.Tensor
    params: dict
    candidates: dict  # the replacement pool without a spatial search
    parameters: dict  # name -> ParameterConfig
    coefficients: torch.Tensor
    prediction: torch.Tensor
    objective: float
    gram: torch.Tensor  # Re(A^H A) / M for real coefficients, A^H A / M otherwise
    rhs: torch.Tensor
    power: float  # mean |target|^2
    iteration: int = 0
    history: tuple = ()
    peak_cache: dict | None = None  # spatial search peaks carried to the next step
    search_failures: int = 0
    lm_skip: int = 0  # steps left before the geometry step runs again
    lm_wait: int = 0  # length of the current run of skipped steps


def _record(state, config, **extra):
    """State with a history entry appended; the data term of the objective gives the NMSE."""
    data = state.objective - config.ridge * float(state.coefficients.abs().square().sum())
    # A zero target fits exactly with zero coefficients; its NMSE is then 0.
    nmse = data / max(state.power, torch.finfo(torch.float32).tiny)
    entry = dict(iteration=state.iteration, objective=state.objective, nmse=nmse)
    return replace(state, history=(*state.history, dict(entry, **extra)))


@torch.no_grad()
def initialize(model, target, params, config, *, candidates=None, parameters=None):
    """Project params onto their constraints and fit the coefficients.

    candidates (default: params) are the replacement pool without a spatial search;
    fields they omit copy slot 0 (merged) or the replaced slot (utility).
    """
    # Fields without a configuration stay fixed.
    specs = {key: (parameters or {}).get(key, ParameterConfig(trainable=False)) for key in params}
    params = {
        key: specs[key].project(value.detach().clone(), value.detach())
        for key, value in params.items()
    }
    model = model.prepare(specs)
    candidates = (
        params if candidates is None else {k: v.detach().clone() for k, v in candidates.items()}
    )
    state = DSFWState(
        model, target, params, candidates, specs,
        **varpro(model, params, target, config)._asdict(),
        power=float(target.abs().square().mean()),
    )  # fmt: skip
    return _record(state, config)


@torch.no_grad()
def step(state, config, search=None):
    """One DSFW iteration; state.history[-1] describes it.

    With a SpatialSearchConfig the replacement pool is a fresh spatial search.
    """
    before = state.objective
    pool, scores = state.candidates, None
    if search is not None:
        pool, scores, cache = propose(state, config, search)
        state = replace(state, peak_cache=cache)
    screen = None
    if 0 < config.screen_apertures < state.model.apertures:
        ids, model = state.model.sample_apertures(
            config.screen_apertures, config.seed + state.iteration + 2000003
        )
        screen = model, state.target[ids]
    if {"merged": True, "utility": False}[config.replacement]:
        state, replaced = merge_support(state, config, pool, scores)
    else:
        state, replaced = replace_one(state, config, pool, screen)
    if search is not None:
        state = replace(state, search_failures=0 if replaced else state.search_failures + 1)
    middle = state.objective
    slides, skip, wait = 0, state.lm_skip, state.lm_wait
    if trainable(state.params, state.parameters) and skip == 0:
        slide = {"joint_cg": joint_slide, "dense": dense_slide}[config.lm_solver]
        state, slides = slide(state, config, screen)
        if config.lm_gain_ratio > 0:
            # The wait doubles while the step keeps paying too little and ends as
            # soon as it pays again, so a late-stage step is never starved.
            poor = middle - state.objective < config.lm_gain_ratio * (before - middle)
            wait = min(LM_WAIT_LIMIT, max(1, 2 * wait)) if poor else 0
            skip = wait
    elif skip:
        skip -= 1
    state = replace(state, iteration=state.iteration + 1, lm_skip=skip, lm_wait=wait)
    return _record(state, config, replaced=replaced, lm_accepted=slides)


def fit(
    model,
    target,
    params,
    config=None,
    search=None,
    *,
    candidates=None,
    parameters=None,
    callback=None,
):
    """initialize, then config.iterations steps; returns the final state.

    callback receives each step's history entry.
    """
    config = config or DSFWConfig()
    state = initialize(model, target, params, config, candidates=candidates, parameters=parameters)
    for _ in range(config.iterations):
        state = step(state, config, search)
        if callback is not None:
            callback(state.history[-1])
    return state
