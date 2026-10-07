"""Deterministic visibility tests used only for synthetic scene generation."""

import numpy as np


def aperture_visibility_sources(positions, views, samples_per_axis=3):
    """Actual antenna locations on a small regular subset of each raster view.

    Visibility from these rays is a conservative subset of visibility from the
    entire array. The returned locations are saved with the synthetic truth.
    """
    positions = np.asarray(positions)
    per_view = len(positions) // views
    side = int(np.sqrt(per_view))
    ids = np.unique(np.linspace(0, side - 1, min(samples_per_axis, side)).round().astype(int))
    local = (ids[:, None] * side + ids[None]).ravel()
    return np.concatenate([positions[v * per_view + local] for v in range(views)])


def visible_surface(mesh, points, normals, sources, *, ray_chunk=256):
    """Front-facing and unobstructed to at least one supplied antenna location."""
    points, normals, sources = (np.asarray(x, dtype=np.float64) for x in (points, normals, sources))
    visible = np.zeros(len(points), dtype=bool)
    epsilon = max(float(np.linalg.norm(mesh.extents)) * 1e-6, 1e-10)
    for source in sources:
        delta = source - points
        facing = (delta * normals).sum(1) > 0
        ids = np.flatnonzero(~visible & facing)
        if not len(ids):
            continue
        # Trace from the antenna toward the sampled surface. Starting just
        # above a millimetre-scale face can produce a behind-origin self-hit
        # under trimesh's absolute ray tolerance; sensor-origin rays avoid it.
        # Trimesh materializes candidate ray/triangle pairs. Bound the rays per
        # query so denser GT sampling does not multiply its entire working set.
        for start in range(0, len(ids), ray_chunk):
            selected = ids[start : start + ray_chunk]
            origins = np.broadcast_to(source, (len(selected), 3))
            direction = points[selected] - origins
            length = np.linalg.norm(direction, axis=1)
            direction /= length[:, None]
            hits, rays, _ = mesh.ray.intersects_location(origins, direction, multiple_hits=False)
            blocked = np.ones(len(selected), dtype=bool)
            if len(rays):
                distance = ((hits - origins[rays]) * direction[rays]).sum(1)
                blocked[rays] = np.abs(distance - length[rays]) > epsilon
            visible[selected[~blocked]] = True
    return visible
