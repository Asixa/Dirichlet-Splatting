"""Surfel scattering, rendering and derivatives against float64 oracles."""

import torch
from conftest import aperture_grid, close, surfels
from reference.oracle import fmcw_columns
from reference.physics import render_scene as render_scene_oracle

from dsplat import FMCWConfig, FMCWModel, surfel
from dsplat.groundtruth import synthesize


def test_render_matches_independent_adc_synthesis():
    config = FMCWConfig(pad_factor=4)
    positions = aperture_grid(6)
    params = surfels(4)
    reflectivity = torch.rand(4, device="cuda") + 0.5
    rendered = surfel.render(config, positions, params, reflectivity)
    amplitudes = reflectivity * params["areas"].sqrt()
    synthesized = synthesize(positions, params["centers"], params["normals"], amplitudes, config)
    close(rendered, synthesized.data, 1e-3)  # float32 carrier phase at 0.3 m


def test_columns_and_jacobian_match_the_oracle(fmcw):
    model, params, b = fmcw
    config, positions = model.sensor
    columns = fmcw_columns(positions, params["centers"], params["normals"], config)
    scale = params["areas"].sqrt()
    # float32 carrier phase at 0.3 m limits agreement to about 2e-4.
    close(
        model.design(params, slice(0, model.size)), (columns * scale).reshape(model.size, -1), 1e-3
    )
    close(model.render(params, b), (columns * scale) @ b.to(torch.complex128), 1e-3)
    rows, atoms = torch.arange(model.size, device="cuda"), torch.arange(5, device="cuda")
    jacobian = model.jacobian(params, rows, atoms, ["centers", "normals"])
    h = 1e-6
    double = {k: v.double() for k, v in params.items()}
    for key in ("centers", "normals"):
        for axis in range(3):
            shift = torch.zeros_like(double[key])
            shift[:, axis] = h
            plus = {**double, key: double[key] + shift}
            minus = {**double, key: double[key] - shift}
            numeric = (
                fmcw_columns(positions, plus["centers"], plus["normals"], config)
                - fmcw_columns(positions, minus["centers"], minus["normals"], config)
            ) / (2 * h)
            close(jacobian[key][..., axis], (numeric * scale).reshape(model.size, -1), 2e-3)


def test_fresnel_material_matches_the_oracle(fmcw):
    model, params, b = fmcw
    config, positions = model.sensor
    columns = fmcw_columns(positions, params["centers"], params["normals"], config, fresnel_n=1.6)
    expected = (columns * params["areas"].sqrt().double()) @ b.to(torch.complex128)
    close(FMCWModel(model.scan, fresnel_n=1.6).render(params, b), expected, 1e-3)


def test_jvp_vjp_and_autograd_are_consistent(fmcw):
    model, params, b = fmcw
    b = b.to(torch.complex64) * torch.exp(1j * torch.arange(5, device="cuda"))
    tangent = {
        "centers": torch.randn_like(params["centers"]) * 1e-4,
        "normals": torch.randn_like(params["normals"]) * 1e-2,
        "areas": torch.randn_like(params["areas"]) * 1e-6,
    }
    db = torch.randn_like(b)
    jvp = model.jvp(params, b, tangent, db)
    config, positions = model.sensor

    def oracle(p, coefficients):
        columns = fmcw_columns(positions, p["centers"], p["normals"], config) * p["areas"].sqrt()
        return columns @ coefficients.to(torch.complex128)

    h = 1e-3
    double = {k: v.double() for k, v in params.items()}
    plus = {k: v + h * tangent[k].double() for k, v in double.items()}
    minus = {k: v - h * tangent[k].double() for k, v in double.items()}
    close(jvp, (oracle(plus, b + h * db) - oracle(minus, b - h * db)) / (2 * h), 2e-3)
    upstream = torch.randn(model.shape, dtype=torch.complex64, device="cuda")
    grads, gb = model.vjp(params, b, upstream)
    left = (upstream.conj() * jvp).real.sum()
    right = sum((grads[k] * v).sum() for k, v in tangent.items()) + (gb.conj() * db).real.sum()
    torch.testing.assert_close(left, right, rtol=1e-3, atol=1e-9)
    leaves = {k: v.clone().requires_grad_() for k, v in params.items()}
    b_leaf = b.clone().requires_grad_()
    (upstream.conj() * model.render(leaves, b_leaf)).real.sum().backward()
    for key in params:
        close(leaves[key].grad, grads[key], 1e-4)
    close(b_leaf.grad, gb, 1e-4)


def test_correlations_and_their_gradient(fmcw):
    model, params, b = fmcw
    residual = model.scan.data
    a = model.design(params, slice(0, model.size))
    corr, norm = model.correlate(params, residual)
    close(corr, a.H @ residual.flatten(), 1e-4)
    close(norm, a.abs().square().sum(0), 1e-4)
    c, n, dc, dn = model.correlate_gradient(params, residual, include_normals=True)
    close(c, corr, 1e-4)
    close(n, norm, 1e-4)
    # The independent Jacobian kernel gives d(a^H r) = da^H r and d|a|^2 = 2 Re(da^H a).
    rows, atoms = torch.arange(model.size, device="cuda"), torch.arange(5, device="cuda")
    jacobian = model.jacobian(params, rows, atoms, ["centers", "normals"])
    da = torch.cat((jacobian["centers"], jacobian["normals"]), -1)
    close(dc, (da.conj() * residual.flatten()[:, None, None]).sum(0), 1e-3)
    close(dn, 2 * (da.conj() * a[..., None]).sum(0).real, 1e-3)


def test_thirteen_dof_composition_matches_the_double_precision_oracle():
    sensor = surfel.Sensor(shape=(32, 8, 8))
    generator = torch.Generator().manual_seed(3)
    positions = torch.tensor([0.0, 0.0, 1.0]) + 0.1 * torch.randn(4, 3, generator=generator)
    direction = torch.tensor([0.0, 0.0, -1.0]) + 0.2 * torch.randn(4, 3, generator=generator)
    normals = torch.nn.functional.normalize(direction, dim=1)
    areas = torch.rand(4, generator=generator) + 0.5
    reflectivity = torch.randn(4, generator=generator) + 0j
    velocities = 0.05 * torch.randn(4, 3, generator=generator)
    impedance = torch.full((4,), 0.3 + 0.01j)
    values = (positions, normals, areas, reflectivity, velocities, impedance)
    expected = render_scene_oracle(
        *(v.to(torch.complex128 if v.is_complex() else torch.float64) for v in values), sensor
    )
    actual = surfel.render_scene(*(v.cuda() for v in values), sensor)
    close(actual, expected, 1e-3)  # float32 carrier phase of a 1 m range
