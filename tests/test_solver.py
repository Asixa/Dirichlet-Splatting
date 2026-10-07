"""VarPro, replacement and geometry refinement against dense linear algebra."""

import itertools

import pytest
import torch
from conftest import close, fmcw_specs, surfels

from dsplat import DSFWConfig, ParameterConfig, SpectralGrid, SpectralModel, initialize, step
from dsplat.refinement import dense_slide, joint_system, reduced_system
from dsplat.replacement import (
    backward_elimination,
    merge_support,
    pooled_statistics,
    regularized_inverse,
    replace_one,
    utility,
)
from dsplat.varpro import (
    accept,
    exchange_varpro,
    normal_equations,
    pcg,
    screen_bar,
    solve,
    varpro,
)


def test_streamed_normal_equations_match_dense_products(fmcw):
    model, params, _ = fmcw
    model = model.prepare(fmcw_specs())
    y = model.scan.data.flatten()
    a = model.design(params, slice(0, model.size)).to(torch.complex128)
    m = model.size
    # Two-aperture blocks give several float64 folds of the float32 Gram.
    config = DSFWConfig(row_chunk=2 * model.shape[1])
    gram, rhs = normal_equations(model, params, model.scan.data, config)
    close(gram, (a.H @ a).real / m, 1e-4)
    close(rhs, (a.H @ y.to(torch.complex128)).real / m, 1e-4)
    gram, rhs = normal_equations(model, params, model.scan.data, DSFWConfig(coefficients="complex"))
    close(gram, a.H @ a / m, 1e-4)
    close(rhs, a.H @ y.to(torch.complex128) / m, 1e-4)


def test_varpro_minimizes_the_ridge_objective(spectral):
    model, params, b = spectral
    target = model.render(params, b)
    trial = {"centers": params["centers"] + 0.3}
    config = DSFWConfig(coefficients="complex", ridge=1e-3)
    coefficients, prediction, loss, _, _ = varpro(model, trial, target, config)
    a = model.design(trial, slice(0, model.size)).to(torch.complex128)
    m = model.size
    h = a.H @ a / m + 1e-3 * torch.eye(3, device="cuda")
    expected = torch.linalg.solve(h, a.H @ target.flatten().to(torch.complex128) / m)
    close(coefficients, expected, 1e-4)
    rendered = (
        prediction - target
    ).abs().square().mean() + 1e-3 * coefficients.abs().square().sum()
    assert abs(loss - float(rendered)) <= 1e-5 * float(rendered)


def test_solve_of_a_singular_system_predicts_like_the_pseudoinverse():
    basis = torch.randn(6, 3, device="cuda", dtype=torch.float64)
    h = basis @ basis.T
    q = basis @ torch.randn(3, device="cuda", dtype=torch.float64)
    x = solve(h, q)
    # The shift moves x only along the null space, which the prediction basis^T x
    # does not see.
    close(basis.T @ x, basis.T @ (torch.linalg.pinv(h, hermitian=True) @ q), 1e-8)
    close(h @ x, q, 1e-8)


def test_one_column_exchange_equals_a_full_refit(fmcw):
    model, params, b = fmcw
    config = DSFWConfig()
    state = initialize(
        model,
        model.scan.data,
        {**params, "centers": params["centers"] + 1e-4},
        config,
        parameters=fmcw_specs(),
    )
    proposal = {k: v.clone() for k, v in state.params.items()}
    proposal["centers"][2] = params["centers"][2]
    fast = exchange_varpro(state, proposal, 2, config)
    full = varpro(model, proposal, model.scan.data, config)
    close(fast.coefficients, full.coefficients, 1e-3)
    assert abs(fast.objective - full.objective) <= 1e-3 * full.objective


def test_backward_elimination_follows_brute_force_refits():
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(20, 8, generator=generator, dtype=torch.float64)
    h, q = x.T @ x, x.T @ torch.randn(20, generator=generator, dtype=torch.float64)

    def objective(keep):
        keep = list(keep)
        return float(-q[keep] @ torch.linalg.solve(h[keep][:, keep], q[keep]))

    alive, expected = set(range(8)), []
    for _ in range(3):
        best = min(alive, key=lambda j: objective(alive - {j}))
        expected.append(best)
        alive.remove(best)
    assert backward_elimination(torch.linalg.inv(h), q, 3).tolist() == expected


def test_regularized_inverse_shifts_by_a_fraction_of_the_largest_eigenvalue():
    # A separated, known top eigenvalue makes the exact-shift comparison meaningful.
    h = torch.diag(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 10.0], dtype=torch.float64))
    inverse = regularized_inverse(h, 1e-3)
    shifted = h + 0.01 * torch.eye(6, dtype=torch.float64)
    close(inverse, torch.linalg.inv(shifted), 1e-6)


@pytest.mark.parametrize("seed", [1, 3, 4, 28])
def test_regularized_inverse_has_a_positive_scalar_shift_and_scales_with_the_gram(seed):
    # These seeds failed the old exact-eigenvalue assertion. Power iteration is
    # approximate: verify the resulting regularization and its scale covariance.
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(30, 6, dtype=torch.float64, generator=generator)
    h = x.T @ x
    inverse = regularized_inverse(h, 1e-3)
    difference = torch.linalg.inv(inverse) - h
    shift = difference.diag().mean()
    assert 0 < shift <= 1e-3 * torch.linalg.eigvalsh(h)[-1] * (1 + 1e-10)
    torch.testing.assert_close(difference, shift * torch.eye(6, dtype=h.dtype), atol=1e-12, rtol=1e-9)
    torch.testing.assert_close(regularized_inverse(7 * h, 1e-3) * 7, inverse)


def test_utility_marks_a_duplicated_column_redundant():
    x = torch.randn(40, 4, dtype=torch.float64)
    x = torch.cat((x, x[:, :1]), 1)
    efficiency = 2 * utility(x.T @ x, torch.zeros(5, dtype=torch.float64))
    assert efficiency[0] < 1e-6 and efficiency[4] < 1e-6 and (efficiency[1:4] > 0.1).all()


def test_merged_replacement_restores_displaced_surfels(fmcw):
    model, params, _ = fmcw
    moved = surfels(5, seed=7)
    config = DSFWConfig(batch=5, screen_apertures=0)
    state = initialize(
        model, model.scan.data, moved, config, candidates=params, parameters=fmcw_specs()
    )
    after, replaced = merge_support(state, config, state.candidates)
    assert replaced > 0 and after.objective < state.objective and len(after.coefficients) == 5


def test_utility_replacement_recovers_a_misplaced_tone(spectral):
    model, params, b = spectral
    target = model.render(params, b)
    wrong = params["centers"].clone()
    wrong[1] = torch.tensor([20.0, 2.0], device="cuda")
    config = DSFWConfig(coefficients="complex", replacement="utility")
    state = initialize(
        model, target, {"centers": wrong}, config,
        candidates={"centers": model.grid.bins()}, parameters={"centers": ParameterConfig(period=model.shape)},
    )  # fmt: skip
    after, replaced = replace_one(state, config, state.candidates)
    assert replaced == 1 and after.objective < 0.1 * state.objective


def test_reduced_gradient_is_half_the_derivative_of_the_projected_objective(spectral):
    model, params, b = spectral
    target = model.render(params, b)
    specs = {"centers": ParameterConfig(scale=0.5)}
    config = DSFWConfig(coefficients="complex", ridge=1e-4)
    rows = torch.arange(model.size, device="cuda")
    trial = {"centers": params["centers"] + torch.tensor([[0.2, -0.1]], device="cuda")}
    _, gradient = reduced_system(model, trial, target, specs, config, rows)

    def projected(centers):
        a = model.design({"centers": centers}, rows).to(torch.complex128)
        y = target.flatten().to(torch.complex128)
        coefficients = torch.linalg.solve(
            a.H @ a / len(a) + 1e-4 * torch.eye(3, device="cuda"), a.H @ y / len(a)
        )
        return float(
            (a @ coefficients - y).abs().square().mean() + 1e-4 * coefficients.abs().square().sum()
        )

    direction = torch.randn_like(trial["centers"])
    h = 1e-2
    numeric = (
        projected(trial["centers"] + h * 0.5 * direction)
        - projected(trial["centers"] - h * 0.5 * direction)
    ) / (2 * h)
    assert abs(2 * float(gradient @ direction.flatten()) - numeric) <= 2e-2 * abs(numeric)


def test_joint_system_is_symmetric_and_pcg_matches_a_dense_solve(spectral):
    model, params, b = spectral
    target = model.render(params, b)
    trial = {"centers": params["centers"] + 0.1}
    specs = {"centers": ParameterConfig()}
    config = DSFWConfig(coefficients="complex", ridge=1e-4)
    matvec, gradient, diagonal, _ = joint_system(model, trial, b, target, specs, config, 0)
    columns = torch.eye(len(gradient), device="cuda")
    dense = torch.stack([matvec(column) for column in columns], 1).double()
    close(dense, dense.T, 1e-4)
    assert torch.linalg.eigvalsh((dense + dense.T) / 2)[0] > 0
    solution = pcg(matvec, -gradient, diagonal, iterations=200, tolerance=1e-7)
    close(solution, torch.linalg.solve(dense, -gradient.double()), 1e-3)


def test_screened_steps_never_raise_the_objective(fmcw):
    model, params, _ = fmcw
    # 16 of the 64 apertures, so the joint system and the screen both use subsets.
    config = DSFWConfig(batch=5, lm_gain_ratio=0, joint_apertures=16, screen_apertures=16)
    state = initialize(
        model,
        model.scan.data,
        surfels(5, seed=9),
        config,
        candidates=params,
        parameters=fmcw_specs(),
    )
    for _ in range(4):
        state = step(state, config)
    objectives = [entry["objective"] for entry in state.history]
    assert all(b <= a for a, b in itertools.pairwise(objectives)) and objectives[-1] < objectives[0]


def test_screened_acceptance_keeps_better_and_rejects_worse_proposals(fmcw):
    model, params, _ = fmcw
    config = DSFWConfig()
    start = {**params, "centers": params["centers"] + 1e-4}
    state = initialize(model, model.scan.data, start, config, parameters=fmcw_specs())
    ids, subset = state.model.sample_apertures(16, 0)
    screen = screen_bar((subset, state.target[ids]), state.params, config)
    worse = {**start, "centers": start["centers"] + 0.005}
    assert accept(state, worse, config, screen) is None
    assert accept(state, params, config, screen).objective < state.objective


def test_unconfigured_parameters_stay_fixed(fmcw):
    model, params, _ = fmcw
    specs = fmcw_specs()
    del specs["areas"]
    config = DSFWConfig(batch=5, lm_gain_ratio=0, screen_apertures=0)
    state = initialize(
        model, model.scan.data, surfels(5, seed=3), config, candidates=params, parameters=specs
    )
    assert torch.equal(step(state, config).params["areas"], state.params["areas"])


def test_pooled_statistics_match_a_full_stream(fmcw):
    model, params, _ = fmcw
    config = DSFWConfig()
    state = initialize(model, model.scan.data, surfels(5, seed=2), config, parameters=fmcw_specs())
    pool = {k: torch.cat((v, params[k])) for k, v in state.params.items()}
    gram, rhs = pooled_statistics(state, pool, config)
    expected_gram, expected_rhs = normal_equations(model, pool, model.scan.data, config)
    close(gram, expected_gram, 1e-4)
    close(rhs, expected_rhs, 1e-4)


def test_complex_exchange_equals_a_full_refit(spectral):
    model, params, b = spectral
    target = model.render(params, b)
    config = DSFWConfig(coefficients="complex", ridge=1e-6)
    state = initialize(model, target, {"centers": params["centers"] + 0.2}, config)
    proposal = {"centers": state.params["centers"].clone()}
    proposal["centers"][1] = params["centers"][1]
    fast = exchange_varpro(state, proposal, 1, config)
    full = varpro(model, proposal, target, config)
    close(fast.coefficients, full.coefficients, 1e-3)


def test_streamed_spectral_correlation_matches_the_dictionary(spectral):
    model, params, b = spectral
    residual = model.render(params, b)
    candidates = {"centers": torch.rand(50, 2, device="cuda") * 20}
    corr, norm = model.correlate(candidates, residual)
    a = model.design(candidates, slice(0, model.size))
    close(corr, a.H @ residual.flatten(), 1e-4)
    close(norm, a.abs().square().sum(0), 1e-4)


def test_dense_lm_moves_tones_back(spectral):
    model, params, b = spectral
    target = model.render(params, b)
    config = DSFWConfig(
        coefficients="complex",
        lm_solver="dense",
        lm_steps=5,
        lm_rows=model.size,
        screen_apertures=0,
    )
    specs = {"centers": ParameterConfig(period=model.shape, scale=0.5)}
    state = initialize(
        model, target, {"centers": params["centers"] + 0.15}, config, parameters=specs
    )
    after, accepted = dense_slide(state, config)
    assert accepted > 0 and after.objective < 1e-3 * state.objective


def test_geometry_step_backs_off_while_it_gains_little(spectral):
    model, params, b = spectral
    target = model.render(params, b)
    wrong = params["centers"].clone()
    wrong[1] = torch.tensor([20.0, 2.0], device="cuda")
    config = DSFWConfig(
        coefficients="complex",
        replacement="utility",
        lm_solver="dense",
        lm_rows=model.size,
        lm_gain_ratio=1e9,
    )
    specs = {"centers": ParameterConfig(period=model.shape, scale=0.5)}
    candidates = {"centers": model.grid.bins()}
    state = initialize(
        model, target, {"centers": wrong}, config, candidates=candidates, parameters=specs
    )
    state = step(state, config)  # replacement gains far more than the geometry step
    assert state.lm_wait == 1 and state.lm_skip == 1
    state = step(state, config)
    assert state.history[-1]["lm_accepted"] == 0 and state.lm_skip == 0


def test_default_config_runs_on_a_spectral_grid_wider_than_the_aperture_subsets():
    grid = SpectralGrid((2048,))
    model = SpectralModel(grid)
    target = model.render(
        {"centers": torch.tensor([[300.4]], device="cuda")}, torch.ones(1, device="cuda")
    )
    config = DSFWConfig(coefficients="complex")
    specs = {"centers": ParameterConfig(period=grid.shape)}
    start = {"centers": torch.tensor([[900.0]], device="cuda")}
    state = initialize(
        model, target, start, config, candidates={"centers": grid.bins()}, parameters=specs
    )
    assert step(state, config).objective < 0.05 * state.objective


def test_merged_replacement_keeps_each_slot_frozen_components(spectral):
    model, params, b = spectral
    target = model.render(params, b)
    config = DSFWConfig(coefficients="complex", batch=8, screen_apertures=0)
    specs = {"centers": ParameterConfig(trainable=(True, False), period=model.shape)}
    start = {"centers": torch.tensor([[9.0, 1.0], [14.0, 5.0], [3.0, 11.0]], device="cuda")}
    state = initialize(model, target, start, config, candidates={"centers": model.grid.bins()},
                       parameters=specs)  # fmt: skip
    after, replaced = merge_support(state, config, state.candidates)
    assert replaced > 0
    assert torch.equal(after.params["centers"][:, 1], start["centers"][:, 1])
    assert after.objective < state.objective


def test_unit_vectors_with_a_frozen_component_stay_unit():
    spec = ParameterConfig(trainable=(True, True, False), normalize=True)
    reference = torch.nn.functional.normalize(torch.tensor([[0.3, 0.4, -0.6]]), dim=1)
    value = spec.project(torch.tensor([[2.0, -1.0, 5.0]]), reference)
    assert value[0, 2] == reference[0, 2]
    torch.testing.assert_close(value.norm(dim=1), torch.ones(1))


def test_unit_vector_tangent_matches_the_projection_with_a_frozen_component():
    spec = ParameterConfig(trainable=(True, False, True), normalize=True)
    generator = torch.Generator().manual_seed(3)
    value = torch.nn.functional.normalize(
        torch.randn(4, 3, generator=generator, dtype=torch.float64), dim=1
    )
    for axis in (0, 2):
        step = torch.zeros_like(value)
        step[:, axis] = 1e-6
        numeric = (spec.project(value + step, value) - spec.project(value - step, value)) / 2e-6
        torch.testing.assert_close(spec.tangent(step / 1e-6, value), numeric, atol=1e-7, rtol=0)


def test_a_free_part_of_zero_length_takes_the_reference_direction():
    spec = ParameterConfig(trainable=(True, False, True), normalize=True)
    reference = torch.tensor([[0.6, 0.8, 0.0]])
    torch.testing.assert_close(spec.project(torch.tensor([[0.0, 0.8, 0.0]]), reference), reference)


def test_a_zero_target_fits_with_zero_nmse(spectral):
    model, params, _ = spectral
    target = torch.zeros(model.shape, dtype=torch.complex64, device="cuda")
    state = initialize(model, target, params, DSFWConfig(coefficients="complex"))
    assert state.history[-1]["nmse"] == 0


def test_unit_vector_with_no_free_radius_has_zero_derivatives():
    spec = ParameterConfig(trainable=(True, False, True), normalize=True)
    value = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    direction = torch.tensor([[1.0, 0.0, 2.0]], dtype=torch.float64)
    numeric = (
        spec.project(value + 1e-6 * direction, value)
        - spec.project(value - 1e-6 * direction, value)
    ) / 2e-6
    torch.testing.assert_close(numeric, torch.zeros_like(value))
    torch.testing.assert_close(spec.tangent(direction, value), numeric)
    jacobian = torch.eye(3, dtype=value.dtype).reshape(3, 1, 3)
    torch.testing.assert_close(
        spec.differential(jacobian, value), torch.zeros(3, 1, 2, dtype=value.dtype)
    )


@pytest.mark.parametrize("trainable", [True, (True, False, True)])
def test_normalizing_without_any_reference_direction_raises(trainable):
    spec = ParameterConfig(trainable=trainable, normalize=True)
    value = torch.zeros(1, 3)
    with pytest.raises(ValueError, match="reference direction"):
        spec.project(value, value)
