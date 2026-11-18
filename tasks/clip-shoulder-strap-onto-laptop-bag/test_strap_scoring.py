"""Offline checks of the strap scorer (no GPU, no Isaac Sim): hooks clipped, open, beside the ring, twisted.

    pytest .    (in an environment with yam_arena installed)
"""
import math
from pathlib import Path

import numpy as np
import pytest
from yam_arena.task_file import load

TASK = load(next(p for p in Path(__file__).parent.glob("*.py") if p.name.replace("-", "_").startswith("clip_")))
T = TASK.module
K = T.K
layout = T.strap_layout


def test_start_scores_zero():
    assert T.score(layout(which=())) == 0.0
    assert not T.success(layout(which=()))


@pytest.mark.parametrize("ring_deg", [0.0, -20.0, 30.0])
def test_both_clipped_untwisted_is_full(ring_deg):
    assert T.score(layout(ring_deg=ring_deg)) == 1.0


def test_hooks_on_each_others_rings_still_count():
    assert T.score(layout(swap=True)) == 1.0


def test_half_twist_is_075():
    assert T.score(layout(twist=180.0)) == 0.75
    assert T.score(layout(twist=-200.0)) == 0.75
    assert T.score(layout(twist=60.0)) == 1.0


def test_one_ring_is_half():
    assert T.score(layout(which=("left",))) == 0.5
    assert T.score(layout(which=("right",))) == 0.5
    assert T.score(layout(gate=(-25.0, 0.0))) == 0.5


def test_gate_jammed_open_on_wire_is_quarter():
    assert T.score(layout(gate=(-25.0, -25.0))) == 0.25
    assert T.score(layout(which=("right",), gate=(0.0, -30.0))) == 0.25


@pytest.mark.parametrize("wire", [(0.0485, 0.020), (0.0485, 0.030), (0.070, 0.0), (0.010, 0.0)])
def test_hook_beside_the_wire_is_zero(wire):
    assert T.score(layout(wire=wire)) == 0.0


def test_non_finite_is_zero():
    o = layout()
    poses = dict(o["strap"].poses)
    p, R = poses["web_05"]
    poses["web_05"] = (np.full(3, np.nan), R)
    o["strap"] = K.strap_state_from(T.STRAP, poses)
    assert T.score(o) == 0.0
