"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest crank-megger-armoured-cable    (in an environment with yam_arena installed)
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
from dataclasses import replace

TASK = load(Path(__file__).with_name("crank_megger_armoured_cable.py"))
T = TASK.module
OBJ = {o.name: o for o in T.OBJECTS}
RUN_LIMIT_STEPS = 1200


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
    'tester': {'pos': (0.3, -0.0875, 0.7475), 'joints': {'crank': 0.0, 'knob': 0.0}},
    'board': {'pos': (0.3, 0, 0.7475), 'joints': {'flag0': 0.0, 'flag1': 0.0, 'flag2': 0.0}},
    'prod': {'pos': (0.24, 0.245, 0.7615)},
}
GOAL_POSES = {
    'tester': {'pos': (0.3, -0.0875, 0.7475), 'joints': {'crank': 0.0, 'knob': 0.0}},
    'board': {'pos': (0.3, 0, 0.7475), 'joints': {'flag0': 0.0, 'flag1': 45.0, 'flag2': 0.0}},
    'prod': {'pos': (0.24, 0.245, 0.7615)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {'TEST_TESTED': '(0, 1, 2)', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'prod'}
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
    ('all three cores tested, no flag tipped yet', START_POSES, {
    }, 0.5, {'TEST_TESTED': '(0, 1, 2)'}),
    ('two cores tested', START_POSES, {
    }, 0.25, {'TEST_TESTED': '(0, 1)'}),
    ('only the leaking core tested', START_POSES, {
    }, 0.25, {'TEST_TESTED': '(1,)'}),
    ("all tested and the leaking core's flag tipped, prod still held 50 mm over its rest", GOAL_POSES, {
        'prod': {'pos': (0.24, 0.245, 0.8115)},
    }, 0.75, {'TEST_TESTED': '(0, 1, 2)', 'TEST_GRIPPERS': '{"left": (np.array(PROD_START) + [0, 0, 0.050 + GRIP_Z], 0.0), "right": (PARK["right"], 1.0)}'}),
    ("all tested, flag tipped, prod on its rest, the open hand 30 mm over the flag's top", GOAL_POSES, {
    }, 0.75, {'TEST_TESTED': '(0, 1, 2)', 'TEST_GRIPPERS': '{"left": (np.array([BOARD_POS[0] + FLAG_X + 0.026, STUD_Y[1], Z0 + FLAG_Z + 0.032 + 0.030]), 1.0), "right": (PARK["right"], 1.0)}'}),
    ('all tested, flag tipped, right hand still on the crank handle', GOAL_POSES, {
    }, 0.75, {'TEST_TESTED': '(0, 1, 2)', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (knob_world(0.0), 0.0)}'}),
    ('all tested, flag tipped, prod lying on the bench beside its rest', GOAL_POSES, {
        'prod': {'pos': (0.16, 0.245, 0.7565), 'tilt_deg': 90, 'tilt_axis': 'y'},
    }, 0.75, {'TEST_TESTED': '(0, 1, 2)', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ("all tested, the leaking core's flag tipped only 20 degrees (not over)", GOAL_POSES, {
        'board': {'pos': (0.3, 0, 0.7475), 'joints': {'flag0': 0.0, 'flag1': 20.0, 'flag2': 0.0}},
    }, 0.5, {'TEST_TESTED': '(0, 1, 2)', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ("all tested, the leaking core's flag tipped just past 25 degrees", GOAL_POSES, {
        'board': {'pos': (0.3, 0, 0.7475), 'joints': {'flag0': 0.0, 'flag1': 26.0, 'flag2': 0.0}},
    }, 1, {'TEST_TESTED': '(0, 1, 2)', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('brown core leaking (trial seed mod 3 = 0), its flag tipped', GOAL_POSES, {
        'board': {'pos': (0.3, 0, 0.7475), 'joints': {'flag0': 45.0, 'flag1': 0.0, 'flag2': 0.0}},
    }, 1, {'TEST_FAULTY': '0', 'TEST_TESTED': '(0, 1, 2)', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('grey core leaking (trial seed mod 3 = 2), its flag tipped', GOAL_POSES, {
        'board': {'pos': (0.3, 0, 0.7475), 'joints': {'flag0': 0.0, 'flag1': 0.0, 'flag2': 45.0}},
    }, 1, {'TEST_FAULTY': '2', 'TEST_TESTED': '(0, 1, 2)', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('all tested, flag tipped, prod in its rest leaning 16 degrees against the rim (as the cup holds it)', GOAL_POSES, {
        'prod': {'pos': (0.232, 0.245, 0.7643), 'tilt_deg': 16, 'tilt_axis': 'y'},
    }, 1, {'TEST_TESTED': '(0, 1, 2)', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('all tested, flag tipped, prod leaning 25 degrees with its guard still low (more than the cup allows)', GOAL_POSES, {
        'prod': {'pos': (0.232, 0.245, 0.7643), 'tilt_deg': 25, 'tilt_axis': 'y'},
    }, 0.75, {'TEST_TESTED': '(0, 1, 2)', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ("the leaking core's flag tipped without testing anything", GOAL_POSES, {
    }, 0, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),  # a guess; nothing was tested
    ('right flag, but it was tipped before all three cores were tested', GOAL_POSES, {
    }, 0, {'TEST_TESTED': '(0, 1, 2)', 'TEST_EARLY': 'True', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),  # the instruction says test all three first; a flag moved early is a guess and stays a zero
    ("all tested, a healthy core's flag tipped instead", GOAL_POSES, {
        'board': {'pos': (0.3, 0, 0.7475), 'joints': {'flag0': 45.0, 'flag1': 0.0, 'flag2': 0.0}},
    }, 0, {'TEST_TESTED': '(0, 1, 2)', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),  # a good core marked faulty
    ('all tested, every flag tipped', GOAL_POSES, {
        'board': {'pos': (0.3, 0, 0.7475), 'joints': {'flag0': 45.0, 'flag1': 45.0, 'flag2': 45.0}},
    }, 0, {'TEST_TESTED': '(0, 1, 2)', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),  # marking every core hedges instead of reading the needle
    ('the middle flag tipped when the grey core is the one leaking', GOAL_POSES, {
    }, 0, {'TEST_FAULTY': '2', 'TEST_TESTED': '(0, 1, 2)', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),  # the flag is scored against the trial's configured leak, so a fixed answer fails
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
