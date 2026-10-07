"""Deterministic scenes: view directions, shape outlines, planar targets and mesh samples."""

import numpy as np
import torch


def view_axes(count=3, geometry="orthogonal", cone_deg=25.0):
    """Look directions toward the y-up object; no aperture center is below it."""
    if geometry == "orthogonal":
        axes = np.array(
            [[0, 0, 1], [1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.float32
        )
        return axes[:count]
    angle = np.deg2rad(cone_deg)
    ring = [
        (np.sin(angle) * np.cos(t), -np.sin(angle) * np.sin(t), np.cos(angle))
        for t in np.linspace(0, np.pi, count - 1)
    ]
    return np.asarray([(0, 0, 1), *ring], dtype=np.float32)


STAR = [
    (np.sin(2 * np.pi * k / 10) * (1 if k % 2 == 0 else 0.38),
     np.cos(2 * np.pi * k / 10) * (1 if k % 2 == 0 else 0.38))
    for k in range(11)
]  # fmt: skip
OUTLINES = {
    "star": [STAR],
    "square": [[(-0.8, -0.8), (0.8, -0.8), (0.8, 0.8), (-0.8, 0.8), (-0.8, -0.8)]],
    "letter_A": [[(-0.7, -0.9), (0.0, 0.9), (0.7, -0.9)], [(-0.4, -0.15), (0.4, -0.15)]],
}


def outline(name, count):
    """count points evenly spaced by arc length from the first vertex, in [-1, 1]^2."""
    segments = [
        (np.array(a), np.array(b))
        for stroke in OUTLINES[name]
        for a, b in zip(stroke[:-1], stroke[1:], strict=True)
    ]
    lengths = np.array([np.linalg.norm(b - a) for a, b in segments])
    ends = np.cumsum(lengths)
    points = []
    for s in np.arange(count) * ends[-1] / count:
        i = int(np.searchsorted(ends, s))
        a, b = segments[i]
        points.append(a + (b - a) * (1 - (ends[i] - s) / lengths[i]))
    return np.stack(points)


def plane_scene(name="letter_A", side=24, extent=0.04, z=0.35, *, device="cpu"):
    axis = torch.linspace(-extent, extent, side, device=device)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    u, v = x / extent, y / extent
    if name == "square":
        mask = (u.abs() < 0.7) & (v.abs() < 0.7)
    elif name == "letter_A":
        mask = (((u.abs() - (0.65 - 0.4 * (v + 0.7))).abs() < 0.13) & (v > -0.7) & (v < 0.8)) | (
            (v.abs() < 0.10) & (u.abs() < 0.42)
        )
    elif name == "letter_T":
        mask = ((v > 0.5) & (v < 0.8) & (u.abs() < 0.8)) | (
            (u.abs() < 0.13) & (v > -0.8) & (v < 0.6)
        )
    elif name == "letter_H":
        mask = (((u.abs() - 0.55).abs() < 0.12) & (v.abs() < 0.8)) | (
            (v.abs() < 0.12) & (u.abs() < 0.55)
        )
    elif name == "two_slabs":
        mask = ((u.abs() - 0.45).abs() < 0.2) & (v.abs() < 0.75)
    else:  # grid
        mask = (
            (
                (torch.sin(3 * torch.pi * u).abs() < 0.35)
                | (torch.sin(3 * torch.pi * v).abs() < 0.35)
            )
            & (u.abs() < 0.8)
            & (v.abs() < 0.8)
        )
    points = torch.stack([x[mask], y[mask], torch.full_like(x[mask], z)], -1)
    normals = torch.zeros_like(points)
    normals[:, 2] = -1
    weights = torch.full((len(points),), 2 * extent / (side - 1), device=device)
    return points, normals, weights, mask


def sample_surface(mesh, count, seed=0, *, sources=None):
    """count area-uniform surfels on a trimesh mesh, on the surface sources see if given.

    Sources are antenna positions; visible means facing one of them with no mesh in
    between. This only selects the reference surfels: the renderer has no occlusion.
    Returns centers, unit normals and amplitudes sqrt(area per surfel), float32 arrays.
    """
    from .visibility import visible_surface

    rng = np.random.default_rng(seed)
    points, normals, tested, accepted = [], [], 0, 0
    while sum(len(p) for p in points) < count:
        number = count if sources is None else max(1024, 2 * count)
        faces = rng.choice(len(mesh.faces), number, p=mesh.area_faces / mesh.area)
        uv = rng.random((number, 2))
        r = np.sqrt(uv[:, 0])
        bary = np.stack([1 - r, r * (1 - uv[:, 1]), r * uv[:, 1]], -1)
        batch = (mesh.triangles[faces] * bary[:, :, None]).sum(1).astype(np.float32)
        normal = mesh.face_normals[faces].astype(np.float32)
        keep = (
            np.ones(number, dtype=bool)
            if sources is None
            else visible_surface(mesh, batch, normal, sources)
        )
        tested += number
        accepted += int(keep.sum())
        points.append(batch[keep])
        normals.append(normal[keep])
        if tested >= max(100_000, count * 100) and accepted < count:
            raise ValueError("too little visible surface for the supplied antenna locations")
    area = mesh.area * accepted / tested / count
    return (
        np.concatenate(points)[:count],
        np.concatenate(normals)[:count],
        np.full(count, np.sqrt(area), dtype=np.float32),
    )
