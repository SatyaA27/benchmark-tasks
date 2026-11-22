"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest load-a-bucking-toy-mule-gently    (in an environment with yam_arena installed)
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

TASK = load(Path(__file__).with_name("load_a_bucking_toy_mule_gently.py"))
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
    'mule': {'pos': (0.175, 0, 0.7475)},
    'rump': {'pos': (0.175, 0, 0.7475)},
    'saddle': {'pos': (0.15, 0, 0.8915)},
    'rope': {'pos': (0.255, -0.15, 0.7475)},
    'canteen': {'pos': (0.105, -0.175, 0.7475)},
    'pan': {'pos': (0.1, 0.175, 0.7475)},
}
GOAL_POSES = {
    'mule': {'pos': (0.175, 0, 0.7475)},
    'rump': {'pos': (0.175, 0, 0.7475)},
    'saddle': {'pos': (0.15, 0, 0.8915)},
    'rope': {'pos': (0.186, 0, 0.8975)},
    'canteen': {'pos': (0.11, -0.026, 0.8975)},
    'pan': {'pos': (0.11, 0.026, 0.8975)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {}


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'pan', 'rope', 'rump', 'mule', 'saddle', 'canteen'}
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
    ('untouched start, all gear on the desk', START_POSES, {
    }, 0, {}),
    ('rope and canteen hung, pan still on the desk', START_POSES, {
        'rope': {'pos': (0.186, 0, 0.8975)},
        'canteen': {'pos': (0.11, -0.026, 0.8975)},
    }, 0.6, {}),
    ('only the pan hung', START_POSES, {
        'pan': {'pos': (0.11, 0.026, 0.8975)},
    }, 0.3, {}),
    ('canteen on the horn instead (any hanger counts), rope and pan hung', START_POSES, {
        'rope': {'pos': (0.11, -0.026, 0.8975)},
        'canteen': {'pos': (0.186, 0, 0.8975)},
    }, 0.6, {}),
    ('pan lying on the seat beside its hanger, not over it', START_POSES, {
        'rope': {'pos': (0.186, 0, 0.8975)},
        'pan': {'pos': (0.11, 0.038, 0.8975)},
    }, 0.3, {}),
    ('all three hung but the mule bucked on the way', GOAL_POSES, {
    }, 0, {'TEST_BUCKED': 'True'}),  # the mule bucked
    ('all three hung but the hand is still on the rope', GOAL_POSES, {
    }, 0.6, {'TEST_GRIPPERS': "{'right': [[XM + SADDLE_X + 0.036, 0.0, TOP + SEAT_Z + SEAT_T + 0.004], [XM + SADDLE_X + 0.036, 0.018, TOP + SEAT_Z + SEAT_T + 0.004]]}"}),  # a piece still in the hand is not hanging
    ('all three hung but the mule shoved 50 mm', GOAL_POSES, {
        'mule': {'pos': (0.225, 0, 0.7475)},
    }, 0, {}),  # the mule was shoved off its place
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
