"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest push-a-wooden-toy-train-round-a-bend    (in an environment with yam_arena installed)
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

TASK = load(Path(__file__).with_name("push_a_wooden_toy_train_round_a_bend.py"))
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
    'engine': {'pos': (0.1456, -0.3439, 0.7565), 'yaw_deg': -86.284, 'quat': (0.0, 0.0, -0.6838099934219222, 0.7296601214924047)},
    'wagon_1': {'pos': (0.1559, -0.2629, 0.7565), 'yaw_deg': -108.189, 'quat': (0.0, 0.0, -0.8099876799298878, 0.5864468930447134)},
    'wagon_2': {'pos': (0.1956, -0.1916, 0.7565), 'yaw_deg': -130.095, 'quat': (0.0, 0.0, 0.9066568369946447, -0.4218689132074871)},
    'box_wagon': {'pos': (0.2591, -0.1403, 0.7565), 'yaw_deg': -152},
}
GOAL_POSES = {
    'engine': {'pos': (0.3143, -0.54, 0.7565), 'yaw_deg': -12.284},
    'wagon_1': {'pos': (0.2392, -0.5077, 0.7565), 'yaw_deg': -34.189},
    'wagon_2': {'pos': (0.1817, -0.4499, 0.7565), 'yaw_deg': -56.095},
    'box_wagon': {'pos': (0.1498, -0.3747, 0.7565), 'yaw_deg': -78},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {}


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'engine', 'wagon_1', 'box_wagon', 'wagon_2'}
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
    ('not moved', START_POSES, {
        'engine': {'pos': (0.1456, -0.3439, 0.7565), 'yaw_deg': -86.284},
        'wagon_1': {'pos': (0.1559, -0.2629, 0.7565), 'yaw_deg': -108.189},
        'wagon_2': {'pos': (0.1956, -0.1916, 0.7565), 'yaw_deg': -130.095},
    }, 0, {}),
    ('pushed 5 cm', START_POSES, {
        'engine': {'pos': (0.1546, -0.393, 0.7565), 'yaw_deg': -72.952},
        'wagon_1': {'pos': (0.1459, -0.3118, 0.7565), 'yaw_deg': -94.857},
        'wagon_2': {'pos': (0.1681, -0.2332, 0.7565), 'yaw_deg': -116.762},
        'box_wagon': {'pos': (0.2181, -0.1686, 0.7565), 'yaw_deg': -138.668},
    }, 0, {}),
    ('pushed 10 cm, coupled', START_POSES, {
        'engine': {'pos': (0.1766, -0.4419, 0.7565), 'yaw_deg': -58.62},
        'wagon_1': {'pos': (0.148, -0.3654, 0.7565), 'yaw_deg': -80.525},
        'wagon_2': {'pos': (0.1502, -0.2838, 0.7565), 'yaw_deg': -102.43},
        'box_wagon': {'pos': (0.1826, -0.2088, 0.7565), 'yaw_deg': -124.336},
    }, 0.25, {}),
    ('pushed 10 cm, box wagon left behind', START_POSES, {
        'engine': {'pos': (0.1766, -0.4419, 0.7565), 'yaw_deg': -58.62},
        'wagon_1': {'pos': (0.148, -0.3654, 0.7565), 'yaw_deg': -80.525},
        'wagon_2': {'pos': (0.1502, -0.2838, 0.7565), 'yaw_deg': -102.43},
        'box_wagon': {'pos': (0.1978, -0.189, 0.7565), 'yaw_deg': -131.002},
    }, 0.25, {}),
    ('past half way, coupled, in the grooves', START_POSES, {
        'engine': {'pos': (0.1947, -0.4673, 0.7565), 'yaw_deg': -50.284},
        'wagon_1': {'pos': (0.1554, -0.3957, 0.7565), 'yaw_deg': -72.189},
        'wagon_2': {'pos': (0.1457, -0.3147, 0.7565), 'yaw_deg': -94.095},
        'box_wagon': {'pos': (0.1669, -0.2358, 0.7565), 'yaw_deg': -116},
    }, 0.5, {}),
    ('past half way, coupled, box wagon on the rail tops', START_POSES, {
        'engine': {'pos': (0.1947, -0.4673, 0.7565), 'yaw_deg': -50.284},
        'wagon_1': {'pos': (0.1554, -0.3957, 0.7565), 'yaw_deg': -72.189},
        'wagon_2': {'pos': (0.1457, -0.3147, 0.7565), 'yaw_deg': -94.095},
        'box_wagon': {'pos': (0.1669, -0.2358, 0.76), 'yaw_deg': -116},
    }, 0.5, {}),
    ('past half way, a coupling open', START_POSES, {
        'engine': {'pos': (0.1947, -0.4673, 0.7565), 'yaw_deg': -50.284},
        'wagon_1': {'pos': (0.1554, -0.3957, 0.7565), 'yaw_deg': -72.189},
        'wagon_2': {'pos': (0.148, -0.2948, 0.7565), 'yaw_deg': -99.428},
        'box_wagon': {'pos': (0.1765, -0.2183, 0.7565), 'yaw_deg': -121.333},
    }, 0.25, {}),
    ('short of the finish by 2 degrees', START_POSES, {
        'engine': {'pos': (0.2926, -0.534, 0.7565), 'yaw_deg': -18.284},
        'wagon_1': {'pos': (0.2213, -0.4941, 0.7565), 'yaw_deg': -40.189},
        'wagon_2': {'pos': (0.1701, -0.4306, 0.7565), 'yaw_deg': -62.095},
        'box_wagon': {'pos': (0.1463, -0.3525, 0.7565), 'yaw_deg': -84},
    }, 0.5, {}),
    ('past the finish, box wagon on the rail tops', START_POSES, {
        'engine': {'pos': (0.3143, -0.54, 0.7565), 'yaw_deg': -12.284},
        'wagon_1': {'pos': (0.2392, -0.5077, 0.7565), 'yaw_deg': -34.189},
        'wagon_2': {'pos': (0.1817, -0.4499, 0.7565), 'yaw_deg': -56.095},
        'box_wagon': {'pos': (0.1498, -0.3747, 0.76), 'yaw_deg': -78},
    }, 0.75, {}),
    ('past the finish, engine 3 mm off its grooves', START_POSES, {
        'engine': {'pos': (0.3149, -0.537, 0.7565), 'yaw_deg': -12.284},
        'wagon_1': {'pos': (0.2392, -0.5077, 0.7565), 'yaw_deg': -34.189},
        'wagon_2': {'pos': (0.1817, -0.4499, 0.7565), 'yaw_deg': -56.095},
        'box_wagon': {'pos': (0.1498, -0.3747, 0.7565), 'yaw_deg': -78},
    }, 0.75, {}),
    ("past the finish, hand still pushing the box wagon's back face", START_POSES, {
        'engine': {'pos': (0.3143, -0.54, 0.7565), 'yaw_deg': -12.284},
        'wagon_1': {'pos': (0.2392, -0.5077, 0.7565), 'yaw_deg': -34.189},
        'wagon_2': {'pos': (0.1817, -0.4499, 0.7565), 'yaw_deg': -56.095},
        'box_wagon': {'pos': (0.1498, -0.3747, 0.7565), 'yaw_deg': -78},
    }, 0.75, {'TEST_GRIPPERS': "{'right': (np.array([0.14098, -0.3331, 0.7945]), 0.0, 0.20944)}"}),
    ("past half way, coupled, hand pushing the box wagon's back face", START_POSES, {
        'engine': {'pos': (0.1947, -0.4673, 0.7565), 'yaw_deg': -50.284},
        'wagon_1': {'pos': (0.1554, -0.3957, 0.7565), 'yaw_deg': -72.189},
        'wagon_2': {'pos': (0.1457, -0.3147, 0.7565), 'yaw_deg': -94.095},
        'box_wagon': {'pos': (0.1669, -0.2358, 0.7565), 'yaw_deg': -116},
    }, 0.5, {'TEST_GRIPPERS': "{'right': (np.array([0.1855, -0.19761, 0.7945]), 0.0, -0.45379)}"}),
    ('past the finish, hand parked clear (as the demo ends)', START_POSES, {
        'engine': {'pos': (0.3143, -0.54, 0.7565), 'yaw_deg': -12.284},
        'wagon_1': {'pos': (0.2392, -0.5077, 0.7565), 'yaw_deg': -34.189},
        'wagon_2': {'pos': (0.1817, -0.4499, 0.7565), 'yaw_deg': -56.095},
        'box_wagon': {'pos': (0.1498, -0.3747, 0.7565), 'yaw_deg': -78},
    }, 1, {'TEST_GRIPPERS': "{'right': (np.array([0.12, -0.62, 0.8975]), 1.0, 0.0)}"}),
    ('past the finish, the box wagon uncoupled 20 mm behind', START_POSES, {
        'engine': {'pos': (0.3143, -0.54, 0.7565), 'yaw_deg': -12.284},
        'wagon_1': {'pos': (0.2392, -0.5077, 0.7565), 'yaw_deg': -34.189},
        'wagon_2': {'pos': (0.1817, -0.4499, 0.7565), 'yaw_deg': -56.095},
        'box_wagon': {'pos': (0.1466, -0.3549, 0.7565), 'yaw_deg': -83.333},
    }, 0.5, {}),
    ('pushed too fast: the front two ran on and uncoupled', START_POSES, {
        'engine': {'pos': (0.3739, -0.5444, 0.7565), 'yaw_deg': 3.716},
        'wagon_1': {'pos': (0.2929, -0.5341, 0.7565), 'yaw_deg': -18.189},
        'wagon_2': {'pos': (0.2001, -0.4735, 0.7565), 'yaw_deg': -48.094},
        'box_wagon': {'pos': (0.1581, -0.4035, 0.7565), 'yaw_deg': -69.999},
    }, 0.5, {}),
    ('train carried through the air to the finish', START_POSES, {
        'engine': {'pos': (0.3143, -0.54, 0.7565), 'yaw_deg': -12.284},
        'wagon_1': {'pos': (0.2392, -0.5077, 0.7565), 'yaw_deg': -34.189},
        'wagon_2': {'pos': (0.1817, -0.4499, 0.7565), 'yaw_deg': -56.095},
        'box_wagon': {'pos': (0.1498, -0.3747, 0.7565), 'yaw_deg': -78},
    }, 0, {'TEST_LIFTED': 'True'}),  # a vehicle was lifted clear of the track on the way
    ('train held 30 mm over the track past the finish', START_POSES, {
        'engine': {'pos': (0.3143, -0.54, 0.7865), 'yaw_deg': -12.284},
        'wagon_1': {'pos': (0.2392, -0.5077, 0.7865), 'yaw_deg': -34.189},
        'wagon_2': {'pos': (0.1817, -0.4499, 0.7865), 'yaw_deg': -56.095},
        'box_wagon': {'pos': (0.1498, -0.3747, 0.7865), 'yaw_deg': -78},
    }, 0, {}),  # vehicles in the air
    ('engine alone carried round, wagons not moved', START_POSES, {
        'engine': {'pos': (0.3143, -0.54, 0.7565), 'yaw_deg': -12.284},
        'wagon_1': {'pos': (0.1559, -0.2629, 0.7565), 'yaw_deg': -108.189},
        'wagon_2': {'pos': (0.1956, -0.1916, 0.7565), 'yaw_deg': -130.095},
    }, 0, {'TEST_LIFTED': 'True'}),  # lifted
    ('engine knocked off the track past the finish', START_POSES, {
        'engine': {'pos': (0.3058, -0.579, 0.7565), 'yaw_deg': -12.284},
        'wagon_1': {'pos': (0.2392, -0.5077, 0.7565), 'yaw_deg': -34.189},
        'wagon_2': {'pos': (0.1817, -0.4499, 0.7565), 'yaw_deg': -56.095},
        'box_wagon': {'pos': (0.1498, -0.3747, 0.7565), 'yaw_deg': -78},
    }, 0, {}),  # the engine is off its grooves
    ('whole train shoved 15 mm off the track past the finish, coupled', START_POSES, {
        'engine': {'pos': (0.3111, -0.5546, 0.7565), 'yaw_deg': -12.284},
        'wagon_1': {'pos': (0.2308, -0.5202, 0.7565), 'yaw_deg': -34.189},
        'wagon_2': {'pos': (0.1692, -0.4582, 0.7565), 'yaw_deg': -56.095},
        'box_wagon': {'pos': (0.1351, -0.3778, 0.7565), 'yaw_deg': -78},
    }, 0, {}),  # off the track: no vehicle in its grooves
    ("engine alone pushed off the track to the finish's angle, outside the circle", START_POSES, {
        'engine': {'pos': (0.2888, -0.6572, 0.7565), 'yaw_deg': -12.284},
        'wagon_1': {'pos': (0.1559, -0.2629, 0.7565), 'yaw_deg': -108.189},
        'wagon_2': {'pos': (0.1956, -0.1916, 0.7565), 'yaw_deg': -130.095},
    }, 0, {}),  # the engine is off the track
    ('train towed round the whole bend by a pinch on the engine, still held', START_POSES, {
        'engine': {'pos': (0.3143, -0.54, 0.7565), 'yaw_deg': -12.284},
        'wagon_1': {'pos': (0.2392, -0.5077, 0.7565), 'yaw_deg': -34.189},
        'wagon_2': {'pos': (0.1817, -0.4499, 0.7565), 'yaw_deg': -56.095},
        'box_wagon': {'pos': (0.1498, -0.3747, 0.7565), 'yaw_deg': -78},
    }, 0, {'TEST_GRIPPERS': "{'right': (np.array([0.29278, -0.53528, 0.7995]), 0.3727, 6.06879)}"}),  # a hand grips the engine
    ('train towed round by the engine, then let go and hands clear', START_POSES, {
        'engine': {'pos': (0.3143, -0.54, 0.7565), 'yaw_deg': -12.284},
        'wagon_1': {'pos': (0.2392, -0.5077, 0.7565), 'yaw_deg': -34.189},
        'wagon_2': {'pos': (0.1817, -0.4499, 0.7565), 'yaw_deg': -56.095},
        'box_wagon': {'pos': (0.1498, -0.3747, 0.7565), 'yaw_deg': -78},
    }, 0, {'TEST_HANDLED': 'True'}),  # a hand gripped the engine earlier in the run
    ('train pushed round by the side of a load wagon', START_POSES, {
        'engine': {'pos': (0.3143, -0.54, 0.7565), 'yaw_deg': -12.284},
        'wagon_1': {'pos': (0.2392, -0.5077, 0.7565), 'yaw_deg': -34.189},
        'wagon_2': {'pos': (0.1817, -0.4499, 0.7565), 'yaw_deg': -56.095},
        'box_wagon': {'pos': (0.1498, -0.3747, 0.7565), 'yaw_deg': -78},
    }, 0, {'TEST_GRIPPERS': "{'right': (np.array([0.22661, -0.52635, 0.7945]), 0.0, 5.68647)}"}),  # a hand touches a load wagon
    ('train pushed round by the side of the last wagon', START_POSES, {
        'engine': {'pos': (0.3143, -0.54, 0.7565), 'yaw_deg': -12.284},
        'wagon_1': {'pos': (0.2392, -0.5077, 0.7565), 'yaw_deg': -34.189},
        'wagon_2': {'pos': (0.1817, -0.4499, 0.7565), 'yaw_deg': -56.095},
        'box_wagon': {'pos': (0.1498, -0.3747, 0.7565), 'yaw_deg': -78},
    }, 0, {'TEST_GRIPPERS': "{'right': (np.array([0.1217, -0.36021, 0.7945]), 0.0, 4.92183)}"}),  # a hand touches the last wagon other than on its back face
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
