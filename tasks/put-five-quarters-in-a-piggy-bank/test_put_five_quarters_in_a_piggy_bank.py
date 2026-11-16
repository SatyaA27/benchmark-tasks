"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest put-five-quarters-in-a-piggy-bank    (in an environment with yam_arena installed)
"""

import math
from pathlib import Path

import numpy as np
import pytest
from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.modules.pop("yam_parts", None)
import yam_parts
sys.modules.pop("yam_joints", None)
import yam_joints
sys.modules.pop("yam_mesh", None)
import yam_mesh

TASK = load(Path(__file__).with_name("put_five_quarters_in_a_piggy_bank.py"))
T = TASK.module
OBJ = {o.name: o for o in T.OBJECTS}
RUN_LIMIT_STEPS = 2400


def _rot(yaw_deg, tilt_deg, axis):
    y, t = math.radians(yaw_deg), math.radians(tilt_deg)
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    c, s = math.cos(t), math.sin(t)
    rt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) if axis == "x" else np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return rz @ rt


def _half(o, r):
    if hasattr(o, "size"):
        return np.abs(r) @ (np.asarray(o.size) / 2)
    if hasattr(o, "height"):
        a = r[:, 2]
        return o.height / 2 * np.abs(a) + o.radius * np.sqrt(np.clip(1 - a * a, 0, 1))
    return np.full(3, o.radius)


def place(name, pos, yaw_deg=0.0, tilt_deg=0.0, tilt_axis="x", vel=(0.0, 0.0, 0.0), quat=None):
    """The ObjectState of `name` with its centre at `pos`, turned and tipped; its bounding box follows the turn."""
    r = (_rot(yaw_deg, tilt_deg, tilt_axis) if quat is None else
         ik.quat_xyzw_to_matrix(np.asarray(quat, float) / np.linalg.norm(quat)))
    if isinstance(OBJ[name], yam_parts.Part):
        origin = np.asarray(pos, float)
        low, high = OBJ[name].aabb(origin, r)
        return yam_parts.PartState((low + high) / 2, np.asarray(ik.matrix_to_quat_xyzw(r), float),
                                   np.asarray(vel, float), np.zeros(3), low, high, origin)
    pos, half = np.asarray(pos, float), _half(OBJ[name], r)
    q = ik.matrix_to_quat_xyzw(r)
    return ObjectState(pos, np.asarray(q, float), np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def layout(poses):
    """Every object's state: the loose ones as given, the fixed ones where the task file puts them."""
    out = {o.name: place(o.name, o.pos, math.degrees(o.yaw)) for o in T.OBJECTS if o.mass is None}
    out.update({name: place(name, **p) for name, p in poses.items()})
    return out


START_POSES = {
    'piggy_bank': {'pos': (0.32, 0.2, 0.7475), 'yaw_deg': 70},
    'coin1': {'pos': (0.235, 0.06, 0.7484)},
    'coin2': {'pos': (0.19, 0.095, 0.7484)},
    'coin3': {'pos': (0.13, 0.15, 0.7484)},
    'coin4': {'pos': (0.2, 0.02, 0.7484)},
    'coin5': {'pos': (0.12, 0.07, 0.7484)},
}
GOAL_POSES = {
    'piggy_bank': {'pos': (0.32, 0.2, 0.7475), 'yaw_deg': 70},
    'coin1': {'pos': (0.3077, 0.1662, 0.7868)},
    'coin2': {'pos': (0.3138, 0.1831, 0.7812)},
    'coin3': {'pos': (0.32, 0.2, 0.7794)},
    'coin4': {'pos': (0.3262, 0.2169, 0.7812)},
    'coin5': {'pos': (0.3323, 0.2338, 0.7868)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'coin1', 'coin2', 'coin4', 'piggy_bank', 'coin3', 'coin5'}
    assert TASK.docs and TASK.docs.strip()


def test_the_start_scores_nothing():
    assert T.score(START) == 0.0
    assert not T.success(START)


def test_the_goal_scores_full_and_succeeds():
    assert T.score(GOAL) == 1.0
    assert T.success(GOAL)

# (name, the layout it starts from, what's moved, the score it should get)
CASES = [
    ('untouched start, all five quarters flat on the table', START_POSES, {
    }, 0),
    ('one quarter in the bank', START_POSES, {
        'coin1': {'pos': (0.3077, 0.1662, 0.7868)},
    }, 0.2),
    ('two quarters in the bank', START_POSES, {
        'coin1': {'pos': (0.3077, 0.1662, 0.7868)},
        'coin2': {'pos': (0.3138, 0.1831, 0.7812)},
    }, 0.4),
    ('three quarters in the bank', GOAL_POSES, {
        'coin4': {'pos': (0.2, 0.02, 0.7484)},
        'coin5': {'pos': (0.12, 0.07, 0.7484)},
    }, 0.6),
    ('four in, the fifth still on the table', GOAL_POSES, {
        'coin5': {'pos': (0.12, 0.07, 0.7484)},
    }, 0.8),
    ('four in, the fifth stuck halfway down the slot on edge', GOAL_POSES, {
        'coin5': {'pos': (0.32, 0.2, 0.9235), 'yaw_deg': 70, 'tilt_deg': 90, 'tilt_axis': 'x'},
    }, 0.8),
    ("four in, the fifth on edge in the slot with only 2 mm still above the wall's inner face", GOAL_POSES, {
        'coin5': {'pos': (0.32, 0.2, 0.9084), 'yaw_deg': 70, 'tilt_deg': 90, 'tilt_axis': 'x'},
    }, 0.8),
    ('four in, the fifth just through the slot (whole coin below the inner mouth), falling', GOAL_POSES, {
        'coin5': {'pos': (0.32, 0.2, 0.9044), 'yaw_deg': 70, 'tilt_deg': 90, 'tilt_axis': 'x', 'vel': (0.0, 0.0, -0.5)},
    }, 1),
    ('all five in, the bank nudged 20 mm along the table and turned 5 degrees', GOAL_POSES, {
        'piggy_bank': {'pos': (0.34, 0.2, 0.7475), 'yaw_deg': 75},
        'coin1': {'pos': (0.3277, 0.1662, 0.7868)},
        'coin2': {'pos': (0.3338, 0.1831, 0.7812)},
        'coin3': {'pos': (0.34, 0.2, 0.7794)},
        'coin4': {'pos': (0.3462, 0.2169, 0.7812)},
        'coin5': {'pos': (0.3523, 0.2338, 0.7868)},
    }, 1),
    ('all five in, the bank rocked 15 degrees onto two feet (under the 20 degree limit)', GOAL_POSES, {
        'piggy_bank': {'pos': (0.2957, 0.2089, 0.7509), 'yaw_deg': 70, 'tilt_deg': 15, 'tilt_axis': 'x'},
        'coin1': {'pos': (0.2929, 0.1716, 0.7888)},
        'coin2': {'pos': (0.2977, 0.189, 0.7834)},
        'coin3': {'pos': (0.3034, 0.206, 0.7817)},
        'coin4': {'pos': (0.31, 0.2228, 0.7834)},
        'coin5': {'pos': (0.3175, 0.2392, 0.7888)},
    }, 1),
    ('the bank laid on its side (slot upright) with all five posted in', GOAL_POSES, {
        'piggy_bank': {'pos': (0.226, 0.2342, 0.8335), 'yaw_deg': 70, 'tilt_deg': 90, 'tilt_axis': 'x'},
        'coin1': {'pos': (0.2506, 0.1869, 0.8335)},
        'coin2': {'pos': (0.2515, 0.2058, 0.8335)},
        'coin3': {'pos': (0.256, 0.2233, 0.8335)},
        'coin4': {'pos': (0.2638, 0.2396, 0.8335)},
        'coin5': {'pos': (0.2752, 0.2546, 0.8335)},
    }, 0),  # laying the bank down turns the slot upright so flat coins could be pushed in without standing them on edge; a ceramic bank knocked over can also break
    ('all five in, the bank tipped 25 degrees and still leaning', GOAL_POSES, {
        'piggy_bank': {'pos': (0.2803, 0.2145, 0.7569), 'yaw_deg': 70, 'tilt_deg': 25, 'tilt_axis': 'x'},
        'coin1': {'pos': (0.2836, 0.1749, 0.7925)},
        'coin2': {'pos': (0.2875, 0.1927, 0.7874)},
        'coin3': {'pos': (0.2929, 0.2099, 0.7858)},
        'coin4': {'pos': (0.2998, 0.2265, 0.7874)},
        'coin5': {'pos': (0.3082, 0.2426, 0.7925)},
    }, 0),  # tipping the bank past 20 degrees counts as knocking it over
    ("five quarters lying flat on the bank's back beside the slot", START_POSES, {
        'coin1': {'pos': (0.335, 0.1945, 0.9244)},
        'coin2': {'pos': (0.305, 0.2055, 0.9244)},
    }, 0),  # a coin dropped flat lands across the 5 mm slot and stays on the back
    ("all five stacked flat on the table against the bank's belly", START_POSES, {
        'coin1': {'pos': (0.32, 0.2, 0.7484)},
        'coin2': {'pos': (0.32, 0.2, 0.7501)},
        'coin3': {'pos': (0.32, 0.2, 0.7519)},
    }, 0),  # under the bank is not in it (the hollow starts 29 mm up)
    ('the bank knocked off the table', START_POSES, {
        'piggy_bank': {'pos': (0.32, 0.8, 0), 'yaw_deg': 70},
    }, 0),
]


@pytest.mark.parametrize(("name", "base", "moved", "expected"), CASES, ids=[c[0] for c in CASES])
def test_named_layouts(name, base, moved, expected):
    objects = layout({**base, **moved})
    assert T.score(objects) == pytest.approx(expected), name
    assert bool(T.success(objects)) == (expected == 1.0), name


def test_the_demo_is_reachable_within_the_joint_limits():
    plan = TASK.demo()
    assert set(plan) <= {"left", "right"}
    for side, waypoints in plan.items():
        arm = ArmPlan(waypoints, station.BASES[side])
        steps = round(arm.duration * 20) + 5
        actions = np.array([arm.action7((k + 1) / 20)[0] for k in range(steps)])
        assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, side
        assert np.all(actions[:, :6] >= ik.LOWER - 1e-6) and np.all(actions[:, :6] <= ik.UPPER + 1e-6), side


def test_the_demo_fits_in_the_run_limit():
    longest = max(ArmPlan(w, station.BASES[s]).duration for s, w in TASK.demo().items())
    assert (longest + TASK.success_hold_s) * 20 < RUN_LIMIT_STEPS * 0.6
