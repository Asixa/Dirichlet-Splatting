"""Experiment metrics and portable point-cloud export."""

import numpy as np
from scipy.spatial import cKDTree


def geometry_metrics(points, reference, threshold_m=0.002):
    forward = cKDTree(reference).query(points)[0]
    backward = cKDTree(points).query(reference)[0]
    precision = float(np.mean(forward < threshold_m))
    recall = float(np.mean(backward < threshold_m))
    return dict(
        surface_median_mm=float(np.median(forward) * 1000),
        surface_mean_mm=float(forward.mean() * 1000),
        chamfer_mm=float((forward.mean() + backward.mean()) * 500),
        precision_2mm=precision,
        recall_2mm=recall,
        fscore_2mm=2 * precision * recall / max(precision + recall, 1e-30),
    )


def write_ply(path, positions, normals=None):
    """Portable ASCII PLY, positions in metres."""
    with open(path, "w", encoding="ascii") as file:
        file.write(f"ply\nformat ascii 1.0\nelement vertex {len(positions)}\n")
        for name in ("x", "y", "z") + (("nx", "ny", "nz") if normals is not None else ()):
            file.write(f"property float {name}\n")
        file.write("end_header\n")
        values = positions if normals is None else np.concatenate([positions, normals], axis=1)
        np.savetxt(file, values, fmt="%.9g")
