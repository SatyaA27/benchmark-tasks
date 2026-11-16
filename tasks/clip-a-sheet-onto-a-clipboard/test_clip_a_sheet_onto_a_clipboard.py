"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest clip-a-sheet-onto-a-clipboard    (in an environment with yam_arena installed)
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
from dataclasses import replace

TASK = load(Path(__file__).with_name("clip_a_sheet_onto_a_clipboard.py"))
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


def place(name, pos, yaw_deg=0.0, tilt_deg=0.0, tilt_axis="x", vel=(0.0, 0.0, 0.0), quat=None, joints=None):
    """The ObjectState of `name` with its centre at `pos`, turned and tipped; its bounding box follows the turn."""
    r = (_rot(yaw_deg, tilt_deg, tilt_axis) if quat is None else
         ik.quat_xyzw_to_matrix(np.asarray(quat, float) / np.linalg.norm(quat)))
    if isinstance(OBJ[name], yam_joints.Mechanism):
        q = {link.name: float((joints or {}).get(link.name, 0.0)) *
             (math.pi / 180 if isinstance(link.joint, yam_joints.Hinge) else 1.0)
             for link in OBJ[name].links}
        state = OBJ[name].offline_state(pos, math.radians(yaw_deg), q, R=r)
        return replace(state, lin_vel=np.asarray(vel, float))
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
    'clipboard': {'pos': (0.2, 0.09, 0.7475), 'joints': {'lever': 0.0}},
    'sheet': {'pos': (0.2, 0.0381, 0.7507)},
}
GOAL_POSES = {
    'clipboard': {'pos': (0.2, 0.09, 0.7475), 'joints': {'lever': -0.45794409325613933}},
    'sheet': {'pos': (0.2, 0.083, 0.7507)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {}


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'clipboard', 'sheet'}
    assert TASK.docs and TASK.docs.strip()


def test_the_start_scores_nothing():
    assert T.score(START) == 0.0
    assert not T.success(START)


def test_the_goal_scores_full_and_succeeds(monkeypatch):
    module_values = {}
    for name, expr in GOAL_MODULE.items():
        module_values[name] = eval(expr, {**vars(T), **module_values, "math": math, "__builtins__": {}})
    for name, value in module_values.items():
        monkeypatch.setattr(T, name, value)
    assert T.score(GOAL) == 1.0
    assert T.success(GOAL)


def test_success_waits_for_the_objects_to_settle(monkeypatch):
    module_values = {}
    for name, expr in GOAL_MODULE.items():
        module_values[name] = eval(expr, {**vars(T), **module_values, "math": math, "__builtins__": {}})
    for name, value in module_values.items():
        monkeypatch.setattr(T, name, value)
    moving = layout({n: {**p, "vel": (0.05, 0.0, 0.0)} for n, p in GOAL_POSES.items()})
    assert not T.success(moving)

# (name, the layout it starts from, what's moved, the score it should get, module overrides)
CASES = [
    ('untouched start (card loose on the board, 45 mm short of the stop)', START_POSES, {
    }, 0, {}),
    ('clipped on the stop, square and centred, hands clear, still (the tilt check passes)', GOAL_POSES, {
    }, 1, {}),
    ('clipped, square, but 6 mm short of the stop (12 mm under the jaw)', GOAL_POSES, {
        'sheet': {'pos': (0.2, 0.077, 0.7507)},
    }, 0.75, {}),
    ('clipped, 1.5 mm short of the stop (inside the 2 mm band)', GOAL_POSES, {
        'sheet': {'pos': (0.2, 0.0815, 0.7507)},
    }, 1, {}),
    ("clipped on the stop but shifted 9 mm sideways (2.5 mm over the board's long edge)", GOAL_POSES, {
        'sheet': {'pos': (0.209, 0.083, 0.7507)},
    }, 0.75, {}),
    ('clipped 8 mm short of the stop and turned 3 degrees (skewed, one end further in)', GOAL_POSES, {
        'sheet': {'pos': (0.2, 0.0751, 0.7507), 'yaw_deg': 3},
    }, 0.5, {}),
    ('clipped on the stop, square, but the board slid 20 mm on the bench and turned 4 degrees (still fine)', GOAL_POSES, {
        'clipboard': {'pos': (0.22, 0.09, 0.7475), 'yaw_deg': 4, 'joints': {'lever': -0.45794409325613933}},
        'sheet': {'pos': (0.2205, 0.0831, 0.7507), 'yaw_deg': 4},
    }, 1, {}),
    ('clip let down before the card was under it (card 10 mm short of the lip, jaw on the board)', START_POSES, {
        'sheet': {'pos': (0.2, 0.0461, 0.7507)},
    }, 0, {}),  # the jaw rests on the board, so nothing holds the card; tilted, it slides off
    ("card pushed against the closed clip (its top edge at the lip's face, jaw on the board)", START_POSES, {
        'sheet': {'pos': (0.2, 0.065, 0.7507)},
    }, 0, {}),  # the lip's square front face stops the card; it is not under the jaw
    ('card only 3 mm under the lip (the lip on its very edge)', GOAL_POSES, {
        'sheet': {'pos': (0.2, 0.068, 0.7507)},
    }, 0, {}),  # under 5 mm of bite the card is not held, it pulls out
    ('card on the stop but the clip held open (lever 15 degrees open)', GOAL_POSES, {
        'clipboard': {'pos': (0.2, 0.09, 0.7475), 'joints': {'lever': -14.999999999999998}},
    }, 0, {}),  # the jaw is not resting on the card
    ('card on the stop, jaw on it, but the left fingertips still on the tab', GOAL_POSES, {
    }, 0, {'TEST_GRIPPERS': '{"left": (np.array([BX - 0.035, BY + CURL_C[0], TOP + CURL_C[1] + CURL_R + 0.013]), 0.0), "right": (np.array([0.32, -0.30, 1.0]), 1.0)}'}),  # not let go
    ('clipped but the tilt check failed (the card slid more than 3 mm on the board)', GOAL_POSES, {
    }, 0, {'TEST_TILT': 'False'}),  # the clamp did not hold it
    ('the whole clipboard lifted 30 mm off the bench (held up, not lying on the bench)', GOAL_POSES, {
        'clipboard': {'pos': (0.2, 0.09, 0.7775), 'joints': {'lever': -0.45794409325613933}},
        'sheet': {'pos': (0.2, 0.083, 0.7807)},
    }, 0, {}),  # the check starts from the board lying flat on the bench
    ('card on the stop and clipped but still moving (sliding at 20 mm/s)', GOAL_POSES, {
        'sheet': {'pos': (0.2, 0.083, 0.7507), 'vel': (0.0, 0.02, 0.0)},
    }, 0, {}),  # not still
]


@pytest.mark.parametrize(("name", "base", "moved", "expected", "module"), CASES, ids=[c[0] for c in CASES])
def test_named_layouts(name, base, moved, expected, module, monkeypatch):
    module_values = {}
    for name, expr in module.items():
        module_values[name] = eval(expr, {**vars(T), **module_values, "math": math, "__builtins__": {}})
    for name, value in module_values.items():
        monkeypatch.setattr(T, name, value)
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
