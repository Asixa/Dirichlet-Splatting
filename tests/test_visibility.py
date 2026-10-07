import numpy as np
import pytest
import trimesh

from examples.scenes import sample_surface
from examples.visibility import aperture_visibility_sources, visible_surface


@pytest.mark.parametrize("ray_chunk", [1, 7, 256])
def test_chunked_visibility_bounds_queries_and_preserves_occlusion(monkeypatch, ray_chunk):
    low = trimesh.creation.box(extents=(2, 2, 1))
    high = trimesh.creation.box(extents=(1, 1, 0.2))
    high.apply_translation((0, 0, 1.5))
    mesh = trimesh.util.concatenate([low, high])
    points = np.tile([[0, 0, 0.5], [0.8, 0, 0.5], [0, 0, -0.5], [0, 0, 1.6]], (73, 1))
    normals = np.tile([[0, 0, 1], [0, 0, 1], [0, 0, -1], [0, 0, 1]], (73, 1))
    original = mesh.ray.intersects_location
    sizes = []

    def bounded(origins, direction, **kwargs):
        sizes.append(len(origins))
        return original(origins, direction, **kwargs)

    monkeypatch.setattr(mesh.ray, "intersects_location", bounded)
    actual = visible_surface(mesh, points, normals, [[0, 0, 5], [0, 0, -5]], ray_chunk=ray_chunk)
    np.testing.assert_array_equal(actual, np.tile([False, True, True, True], 73))
    assert sizes and max(sizes) <= ray_chunk


def test_visible_surface_rejects_backfaces_and_occluded_frontfaces():
    low = trimesh.creation.box(extents=(2, 2, 1))
    high = trimesh.creation.box(extents=(1, 1, 0.2))
    high.apply_translation((0, 0, 1.5))
    mesh = trimesh.util.concatenate([low, high])
    points = np.array([[0, 0, 0.5], [0.8, 0, 0.5], [0, 0, -0.5], [0, 0, 1.6]])
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, -1], [0, 0, 1]])
    assert visible_surface(mesh, points, normals, [[0, 0, 5]]).tolist() == [
        False,
        True,
        False,
        True,
    ]
    assert visible_surface(mesh, points, normals, [[0, 0, 5], [0, 0, -5]]).tolist() == [
        False,
        True,
        True,
        True,
    ]


def test_visible_sampling_keeps_exact_count_and_is_deterministic():
    mesh = trimesh.creation.box()
    sources = [[0, 0, 5]]
    first = sample_surface(mesh, 40, 7, sources=sources)
    second = sample_surface(mesh, 40, 7, sources=sources)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    assert len(first[0]) == 40
    assert (first[1][:, 2] == 1).all()
    assert visible_surface(mesh, first[0], first[1], sources).all()
    # Only the top face is visible: a sixth of the area, shared by 40 surfels.
    np.testing.assert_allclose(first[2] ** 2 * 40, mesh.area / 6, rtol=0.2)


def test_visibility_sources_are_actual_antennas():
    positions = np.arange(3 * 16 * 3).reshape(-1, 3)
    selected = aperture_visibility_sources(positions, 3)
    assert selected.shape == (27, 3)
    assert all(any(np.array_equal(s, p) for p in positions) for s in selected)


def test_millimetre_mesh_does_not_self_occlude_under_absolute_ray_tolerance():
    mesh = trimesh.creation.box(extents=(0.06, 0.06, 0.06))
    rotation = trimesh.transformations.rotation_matrix(0.6, [0, 1, 0])
    mesh.apply_transform(rotation)
    points = np.array([[0, 0, 0.03], [0.01, 0.005, 0.03]]) @ rotation[:3, :3].T
    normals = np.array([[0, 0, 1], [0, 0, 1]]) @ rotation[:3, :3].T
    assert visible_surface(mesh, points, normals, [[0, 0, 0.35]]).all()
