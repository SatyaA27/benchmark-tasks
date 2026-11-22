"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest fit-the-last-shapes-of-a-perfection-game-before-the-tray-pops    (in an environment with yam_arena installed)
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

TASK = load(Path(__file__).with_name("fit_the_last_shapes_of_a_perfection_game_before_the_tray_pops.py"))
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
        q = {link.name: float((joints or {}).get(link.name, getattr(link.joint, "initial", 0.0))) *
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
    'game': {'pos': (0.2, -0.045, 0.7475), 'joints': {'tray': 0.0, 'switch': 0.0, 'dial': -119.99999999999999}},
    'p_square': {'pos': (0.22, 0.185, 0.7475), 'yaw_deg': 25},
    'p_circle': {'pos': (0.13, 0.185, 0.7475)},
    'p_hotdog': {'pos': (0.17, -0.3, 0.7475), 'yaw_deg': -55},
    'p_cross': {'pos': (0.27, -0.3, 0.7475), 'yaw_deg': 30},
}
GOAL_POSES = {
    'game': {'pos': (0.2, -0.045, 0.7475), 'joints': {'tray': 0.0, 'switch': 0.0, 'dial': -119.99999999999999}},
    'p_square': {'pos': (0.158, 0.084, 0.7935)},
    'p_circle': {'pos': (0.2, 0.084, 0.7935)},
    'p_hotdog': {'pos': (0.158, -0.084, 0.7935), 'yaw_deg': 180},
    'p_cross': {'pos': (0.2, -0.084, 0.7935), 'yaw_deg': 90},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {'TEST_CLOCK': '(True, False, False)'}


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'p_square', 'game', 'p_circle', 'p_hotdog', 'p_cross'}
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

# (name, the layout it starts from, what's moved, the score it should get, module overrides)
CASES = [
    ('untouched start, the clock still running', START_POSES, {
    }, 0, {}),
    ('three in their holes when the switch was moved to STOP, the cross still on the desk', GOAL_POSES, {
        'p_cross': {'pos': (0.27, -0.3, 0.7475), 'yaw_deg': 30},
    }, 0.75, {'TEST_CLOCK': '(True, False, False)'}),
    ('two in when stopped, the square resting on top of its own hole (not down in it)', GOAL_POSES, {
        'p_square': {'pos': (0.158, 0.084, 0.7995)},
        'p_cross': {'pos': (0.27, -0.3, 0.7475), 'yaw_deg': 30},
    }, 0.5, {'TEST_CLOCK': '(True, False, False)'}),
    ('one in when stopped', START_POSES, {
        'p_circle': {'pos': (0.2, 0.084, 0.7935), 'yaw_deg': 40},
    }, 0.25, {'TEST_CLOCK': '(True, False, False)'}),
    ('all four in but the switch never moved, so the clock ran out and the tray popped', GOAL_POSES, {
    }, 0.25, {'TEST_CLOCK': '(False, True, False)'}),
    ('all four in, the clock still running (not stopped yet)', GOAL_POSES, {
    }, 0, {'TEST_CLOCK': '(False, False, False)'}),
    ('switch moved to STOP with no piece fitted', START_POSES, {
    }, 0, {'TEST_CLOCK': '(True, False, False)'}),  # no shape was in its hole when the clock was stopped
    ("square lying on the circle's hole, the switch at STOP", START_POSES, {
        'p_square': {'pos': (0.2, 0.084, 0.7995)},
    }, 0, {'TEST_CLOCK': '(True, False, False)'}),  # a shape on the wrong hole is not seated
    ('clock ran out with three in', GOAL_POSES, {
        'p_cross': {'pos': (0.27, -0.3, 0.7475), 'yaw_deg': 30},
    }, 0, {'TEST_CLOCK': '(False, True, False)'}),  # the tray popped before all four were in
    ('all four in but the tray held down when it popped', GOAL_POSES, {
    }, 0, {'TEST_CLOCK': '(False, True, True)'}),  # the tray was held down against its spring
    ('all four in and stopped but the unit shoved 50 mm', GOAL_POSES, {
        'game': {'pos': (0.25, -0.045, 0.7475), 'joints': {'tray': 0.0, 'switch': 0.0, 'dial': -119.99999999999999}},
    }, 0, {'TEST_CLOCK': '(True, False, False)'}),  # the game unit was shoved off its place
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
