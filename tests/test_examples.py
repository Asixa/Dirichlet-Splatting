"""The three experiments at their smoke presets."""

import itertools
import json
from argparse import Namespace

import pytest

from examples import bunny, raw2d, signal2d


def run(module, tmp_path, **extra):
    args = Namespace(
        experiment=module.__name__,
        preset="smoke",
        seed=0,
        iterations=None,
        ridge=None,
        output=str(tmp_path),
        **extra,
    )
    path = module.run(args)
    history = json.loads((path / "history.json").read_text())
    assert all(b["objective"] <= a["objective"] for a, b in itertools.pairwise(history))
    return json.loads((path / "metrics.json").read_text())


@pytest.mark.parametrize("window", ["rect", "hann"])
def test_signal2d_recovers_every_tone(tmp_path, window):
    metrics = run(signal2d, tmp_path, window=window, scene="star")
    assert metrics["nmse"] < 1e-6 and metrics["center_rmse_bins"] < 1e-2


def test_raw2d_fits_the_planar_scene(tmp_path):
    metrics = run(raw2d, tmp_path, scene="letter_A", initialization="migration")
    assert metrics["nmse"] < 1e-4


def test_bunny_reduces_the_error_from_a_random_start(tmp_path):
    metrics = run(bunny, tmp_path, surfels=None)
    assert metrics["nmse"] < 0.7 and metrics["precision_2mm"] > 0.5
