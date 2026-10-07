"""Local geometry refinement: dense reduced-VarPro LM and matrix-free joint Gauss-Newton.

Both propose moves of every trainable atom and keep one only if its refitted
objective is lower (varpro.accept). Steps are dimensionless: each
ParameterConfig.scale sets its unit.
"""

import torch

from .varpro import accept, fit_coefficients, pcg, screen_bar, solve


def trainable(params, specs):
    return [(key, spec) for key, spec in specs.items() if spec.mask(params[key]).any()]


# --- dense: reduced VarPro on sampled rows ---------------------------------------------------


@torch.no_grad()
def reduced_system(model, params, target, specs, config, rows):
    """Gauss-Newton normal matrix and gradient of the ridge-VarPro objective on some rows.

    Every coefficient is eliminated: J = dA b + A db with the exact coefficient
    differential db, including the dA^H r term. Unit-vector retractions and parameter
    scales are chained before elimination.
    """
    a = model.design(params, rows)
    y = target.flatten()[rows]
    m = len(a)
    gram, rhs = a.H @ a / m, a.H @ y / m
    b = fit_coefficients(gram, rhs, config)
    residual = a @ b.to(a.dtype) - y
    layout = trainable(params, specs)
    atoms = torch.arange(len(b), device=a.device)
    raw = model.jacobian(params, rows, atoms, [key for key, _ in layout])
    da = torch.cat([spec.differential(raw[key], params[key]) for key, spec in layout], -1)
    real = config.real
    h = (gram.real if real else gram) + config.ridge * torch.eye(
        len(b), dtype=b.dtype, device=a.device
    )
    j = (da * b[None, :, None]).reshape(m, -1)
    cross = a.H @ j / m
    columns = torch.arange(j.shape[1], device=a.device)
    cross[atoms.repeat_interleave(da.shape[-1]), columns] += (
        da.conj() * residual[:, None, None]
    ).sum(0).flatten() / m
    db = -solve(h, cross.real if real else cross)
    reduced = j + a @ db.to(a.dtype)
    normal = (reduced.H @ reduced / m + config.ridge * db.H @ db).real
    gradient = (reduced.H @ residual / m + config.ridge * db.H @ b).real
    return normal, gradient


def dense_slide(state, config, screen=None):
    """Damped LM steps on lm_rows sampled rows with a trust radius per atom.

    Returns the state and the number of accepted steps.
    """
    layout = trainable(state.params, state.parameters)
    rows = state.model.sample_rows(
        config.lm_rows, config.seed + state.iteration, state.target.device
    )
    accepted = 0
    for _ in range(config.lm_steps):
        normal, gradient = reduced_system(
            state.model, state.params, state.target, state.parameters, config, rows
        )
        scale = normal.diag().clamp(min=normal.diag().max() * 1e-6 + 1e-30)
        delta = solve(normal + config.damping * torch.diag(scale), -gradient).reshape(
            len(state.coefficients), -1
        )
        delta *= (config.trust_radius / delta.norm(dim=1, keepdim=True).clamp(min=1e-30)).clamp(
            max=1
        )
        if not torch.isfinite(delta).all():
            break
        bar = screen_bar(screen, state.params, config)
        for fraction in (1.0, 0.5, 0.25):
            proposal = {key: value.clone() for key, value in state.params.items()}
            offset = 0
            for key, spec in layout:
                value = proposal[key]
                matrix = value[:, None] if value.ndim == 1 else value.clone()
                mask = spec.mask(value)
                width = int(mask.sum())
                matrix[:, mask] += fraction * spec.scale * delta[:, offset : offset + width]
                proposal[key] = spec.project(
                    matrix[:, 0] if value.ndim == 1 else matrix, state.params[key]
                )
                offset += width
            trial = accept(state, proposal, config, bar)
            if trial is not None:
                state, accepted = trial, accepted + 1
                break
        else:
            break
    return state, accepted


# --- joint: matrix-free Gauss-Newton over geometry and coefficients --------------------------


def joint_system(model, params, coefficients, target, specs, config, seed):
    """Scaled Gauss-Newton system of all trainable geometry and every coefficient.

    Returns matvec, gradient, a damped Jacobi diagonal and proposal(vector, fraction).
    Products use the model's exact JVP and VJP; Rademacher probes estimate only the
    diagonal. Coefficients are scaled by their magnitude, floored at a tenth of the
    RMS magnitude.
    """
    b = coefficients.to(target.dtype)
    complex_mode = not config.real
    count = len(b)
    layout, offset = [], 0
    for key, spec in trainable(params, specs):
        mask = spec.mask(params[key])
        layout.append((key, mask, slice(offset, offset + count * int(mask.sum()))))
        offset += count * int(mask.sum())
    geometry = offset
    size = geometry + (2 * count if complex_mode else count)
    amplitude = b.abs()
    coefficient_scale = amplitude.clamp(min=amplitude.square().mean().sqrt() * 0.1 + 1e-20)
    power = target.abs().square().mean().clamp(min=1e-30)
    normalization = target.numel() * power
    ridge = config.ridge / power
    ridge_diagonal = amplitude.new_zeros(size)
    ridge_gradient = amplitude.new_zeros(size)
    if complex_mode:
        ridge_diagonal[geometry:] = (ridge * coefficient_scale.square()).repeat_interleave(2)
        ridge_gradient[geometry:] = torch.view_as_real(b * (ridge * coefficient_scale)).flatten()
    else:
        ridge_diagonal[geometry:] = ridge * coefficient_scale.square()
        ridge_gradient[geometry:] = ridge * coefficient_scale * b.real

    def unpack(vector):
        tangent = {key: torch.zeros_like(value) for key, value in params.items()}
        for key, mask, section in layout:
            value = tangent[key]
            matrix = value[:, None] if value.ndim == 1 else value
            matrix[:, mask] = vector[section].reshape(count, -1) * specs[key].scale
            tangent[key] = specs[key].tangent(value, params[key])
        tail = vector[geometry:]
        if complex_mode:
            return tangent, torch.complex(tail[0::2], tail[1::2]) * coefficient_scale
        return tangent, (tail * coefficient_scale).to(b.dtype)

    def adjoint(upstream):
        grads, coefficient = model.vjp(params, b, upstream)
        result = amplitude.new_zeros(size)
        for key, mask, section in layout:
            value = specs[key].tangent(grads[key], params[key])
            matrix = value[:, None] if value.ndim == 1 else value
            result[section] = (matrix[:, mask] * specs[key].scale).flatten()
        tail = coefficient * coefficient_scale
        result[geometry:] = torch.view_as_real(tail).flatten() if complex_mode else tail.real
        return result

    gradient = adjoint(model.render(params, b) - target) / normalization + ridge_gradient
    generator = torch.Generator(device=b.device).manual_seed(seed)
    diagonal = torch.zeros_like(gradient)
    for _ in range(config.cg_probes):
        signs = torch.randint(0, 2, (*target.shape, 2), device=b.device, generator=generator)
        probe = torch.view_as_complex((2 * signs - 1).to(target.real.dtype))
        diagonal += adjoint(probe).square() / normalization
    diagonal = diagonal / config.cg_probes + ridge_diagonal
    diagonal = diagonal.clamp(min=diagonal.max().clamp(min=1e-30) * 1e-6)
    damping = config.damping * diagonal

    def matvec(vector):
        return (
            adjoint(model.jvp(params, b, *unpack(vector))) / normalization
            + (ridge_diagonal + damping) * vector
        )

    def proposal(vector, fraction):
        # One scale for every atom keeps the joint direction; the radial part of a
        # unit-normal update is a null direction and must not shrink the step.
        tangent, _ = unpack(vector)
        lengths = amplitude.new_zeros(count)
        for key, mask, _ in layout:
            matrix = tangent[key][:, None] if tangent[key].ndim == 1 else tangent[key]
            lengths += (matrix[:, mask] / specs[key].scale).square().sum(1)
        scale = (config.trust_radius / lengths.sqrt().max().clamp(min=1e-30)).clamp(max=1)
        return {
            key: specs[key].project(value + tangent[key] * (scale * fraction), value)
            for key, value in params.items()
        }

    return matvec, gradient, diagonal + damping, proposal


@torch.no_grad()
def joint_slide(state, config, screen=None):
    """PCG steps of the joint system, built on joint_apertures sampled apertures.

    Sampling affects proposals only; acceptance still uses every fitting observation.
    Returns the state and the number of accepted steps.
    """
    model, target = state.model, state.target
    if 0 < config.joint_apertures < model.apertures:
        ids, model = model.sample_apertures(config.joint_apertures, config.seed + state.iteration)
        target = target[ids]
    accepted = 0
    for inner in range(config.lm_steps):
        matvec, gradient, diagonal, proposal = joint_system(
            model, state.params, state.coefficients, target, state.parameters, config,
            config.seed + state.iteration + inner,
        )  # fmt: skip
        delta = pcg(
            matvec,
            -gradient,
            diagonal,
            iterations=config.cg_iterations,
            tolerance=config.cg_tolerance,
        )
        if not (torch.isfinite(delta).all() and delta @ gradient < 0):
            break
        bar = screen_bar(screen, state.params, config)
        for fraction in (1.0, 0.5, 0.25, 0.125):
            trial = accept(state, proposal(delta, fraction), config, bar)
            if trial is not None:
                state, accepted = trial, accepted + 1
                break
        else:
            break
    return state, accepted
