"""Bindings of search.slang: range gating, separated peaks and quasi-Newton steps."""

import math

import torch

from . import blocks, load


def range_gate(points, positions, intervals, spacing, min_fraction):
    """Points whose spherical range bin lies in a marked interval at min_fraction of apertures."""
    output = torch.empty(len(points), device=points.device, dtype=torch.int32)
    load("search").candidate_range_gate(
        points=points.contiguous(),
        positions=positions.contiguous(),
        intervals=intervals.contiguous(),
        output=output,
        count=len(points),
        apertures=len(positions),
        bins=intervals.shape[1],
        spacing=spacing,
        required=math.ceil(min_fraction * len(positions) - 1e-7),
    ).launchRaw(**blocks(len(points)))
    return output.bool()


def peaks(scores, centers, count, separation):
    """Greedy best-first indices at least separation apart, in one block."""
    output = torch.full((min(count, len(scores)),), -1, device=centers.device, dtype=torch.int32)
    load("search").candidate_peaks(
        scores=scores.float().clone(),
        centers=centers.contiguous(),
        output=output,
        count=len(scores),
        peaks=len(output),
        dimension=centers.shape[1],
        separation=separation,
    ).launchRaw(blockSize=(256, 1, 1), gridSize=(1, 1, 1))
    return output[output >= 0].long()


def direction(inverse, gradient, normals, mask, radius):
    """Quasi-Newton descent per candidate: tangent to unit normals, inside the trust radius."""
    count, width = gradient.shape
    output = torch.empty_like(gradient)
    load("search").candidate_direction(
        inverse=inverse.contiguous(),
        gradient=gradient.contiguous(),
        normals=normals.contiguous(),
        mask=mask,
        output=output,
        count=count,
        width=width,
        radius=radius,
    ).launchRaw(**blocks(count, 128))
    return output


def trial(centers, normals, direction, controls, fraction):
    """Step fraction along direction, clamped to the box and renormalized; frozen axes kept.

    controls rows: per-coordinate scale, trainable mask, lower and upper bounds.
    """
    count, width = direction.shape
    out_centers, out_normals = torch.empty_like(centers), torch.empty_like(normals)
    load("search").candidate_trial(
        centers=centers.contiguous(),
        normals=normals.contiguous(),
        direction=direction,
        controls=controls,
        out_centers=out_centers,
        out_normals=out_normals,
        count=count,
        width=width,
        fraction=fraction,
    ).launchRaw(**blocks(count, 128))
    return out_centers, out_normals
