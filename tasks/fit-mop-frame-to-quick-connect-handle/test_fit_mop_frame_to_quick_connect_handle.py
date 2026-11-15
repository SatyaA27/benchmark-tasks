"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest fit-mop-frame-to-quick-connect-handle    (in an environment with yam_arena installed)
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

TASK = load(Path(__file__).with_name("fit_mop_frame_to_quick_connect_handle.py"))
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
    'frame': {'pos': (0.28, -0.24, 0.7475), 'yaw_deg': 180, 'joints': {'cradle': 0.0, 'arm': 0.0}},
    'handle': {'pos': (0.28, -0.042, 0.7675), 'yaw_deg': 90, 'joints': {'button': 0.0}},
}
GOAL_POSES = {
    'frame': {'pos': (0.28, -0.24, 0.7475), 'yaw_deg': 180, 'joints': {'cradle': 0.0, 'arm': -12.0}},
    'handle': {'pos': (0.28, -0.16, 0.7675), 'yaw_deg': 90, 'joints': {'button': 0.0}},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {}


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'handle', 'frame'}
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
    ('untouched start, the socket folded and turned aside, the handle 45 mm short', START_POSES, {
    }, 0, {}),
    ('clicked, still pinched in both hands', GOAL_POSES, {
    }, 0.5, {'TEST_GRIPPERS': '{"left": (np.array([MOUTH_SQ[0], MOUTH_SQ[1] - SEATED_DEPTH + GRIP_X, MOUTH_SQ[2]]), 0.0), "right": (np.array([MOUTH_SQ[0], MOUTH_SQ[1] - MOUTH_S + BAR_GRIP_S, MOUTH_SQ[2]]), 0.0)}'}),
    ("clicked, hands opened but only 20 mm above the tube and the socket's arm", GOAL_POSES, {
    }, 0.5, {'TEST_GRIPPERS': '{"left": (np.array([MOUTH_SQ[0], MOUTH_SQ[1] - SEATED_DEPTH + GRIP_X, MOUTH_SQ[2] + 0.020]), 1.0), "right": (np.array([MOUTH_SQ[0], MOUTH_SQ[1] - MOUTH_S + BAR_GRIP_S, MOUTH_SQ[2] + 0.020]), 1.0)}'}),
    ("clicked, hands 60 mm above the tube and the socket's arm", GOAL_POSES, {
    }, 1, {'TEST_GRIPPERS': '{"left": (np.array([MOUTH_SQ[0], MOUTH_SQ[1] - SEATED_DEPTH + GRIP_X, MOUTH_SQ[2] + 0.060]), 1.0), "right": (np.array([MOUTH_SQ[0], MOUTH_SQ[1] - MOUTH_S + BAR_GRIP_S, MOUTH_SQ[2] + 0.060]), 1.0)}'}),
    ('clicked, the handle still rolling at 20 mm/s', GOAL_POSES, {
        'handle': {'pos': (0.28, -0.16, 0.7675), 'yaw_deg': 90, 'vel': (0.02, 0.0, 0.0), 'joints': {'button': 0.0}},
    }, 0.5, {}),
    ('lug 1 mm short of the notch floor, the button out (a 1.5 mm gauge does not pass)', GOAL_POSES, {
        'handle': {'pos': (0.28, -0.159, 0.7675), 'yaw_deg': 90, 'joints': {'button': 0.0}},
    }, 1, {}),
    ('pushed 6 mm short, the button still pressed inside the bore, let go', GOAL_POSES, {
        'handle': {'pos': (0.28, -0.154, 0.7675), 'yaw_deg': 90, 'joints': {'button': -0.0027}},
    }, 0.25, {}),
    ('pushed home but the button not out (held pressed), let go', GOAL_POSES, {
        'handle': {'pos': (0.28, -0.16, 0.7675), 'yaw_deg': 90, 'joints': {'button': -0.0027}},
    }, 0.25, {}),
    ('tip only 20 mm into the socket, let go', GOAL_POSES, {
        'handle': {'pos': (0.28, -0.107, 0.7675), 'yaw_deg': 90, 'joints': {'button': 0.0}},
    }, 0, {}),
    ('tip stopped on the mouth of a socket left turned aside (handle pushed straight)', START_POSES, {
        'handle': {'pos': (0.28, -0.087, 0.7675), 'yaw_deg': 90, 'joints': {'button': 0.0}},
    }, 0, {}),
    ('handle at the seated place but the socket turned aside 12 degrees (tip beside the socket)', START_POSES, {
        'handle': {'pos': (0.28, -0.16, 0.7675), 'yaw_deg': 90, 'joints': {'button': 0.0}},
    }, 0, {}),
    ('clicked, but the joined mop knocked off the bench', GOAL_POSES, {
        'frame': {'pos': (0.28, -0.24, 0.4475), 'yaw_deg': 180, 'joints': {'cradle': 0.0, 'arm': -12.0}},
        'handle': {'pos': (0.28, -0.16, 0.4675), 'yaw_deg': 90, 'joints': {'button': 0.0}},
    }, 0, {}),
    ("handle dropped off the bench's far edge", START_POSES, {
        'handle': {'pos': (0.28, 0.8, 0.5675), 'yaw_deg': 90, 'joints': {'button': 0.0}},
    }, 0, {}),
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
