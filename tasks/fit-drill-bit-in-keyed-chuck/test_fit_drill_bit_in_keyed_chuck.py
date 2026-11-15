"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest fit-drill-bit-in-keyed-chuck    (in an environment with yam_arena installed)
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

TASK = load(Path(__file__).with_name("fit_drill_bit_in_keyed_chuck.py"))
T = TASK.module
OBJ = {o.name: o for o in T.OBJECTS}
RUN_LIMIT_STEPS = 4200


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
    'drill': {'pos': (0.24, -0.1, 0.7825)},
    'bit': {'pos': (0.24, -0.207, 0.7825)},
    'key': {'pos': (0.12, -0.17, 0.7595)},
}
GOAL_POSES = {
    'drill': {'pos': (0.24, -0.1, 0.7825)},
    'bit': {'pos': (0.24, -0.076, 0.7825)},
    'key': {'pos': (0.12, -0.17, 0.7595)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {'TEST_MODEL': '{"gripped": True, "tightness": 0.34, "engaged": False, "opening": 0.006}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'drill', 'bit', 'key'}
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
    ('untouched start', START_POSES, {
    }, 0, {}),
    ('bit pushed 24 mm into the open jaws, key never used (jaws not closed)', GOAL_POSES, {
    }, 0.25, {'TEST_MODEL': '{"gripped": False, "tightness": 0.0, "engaged": False, "opening": 0.0075}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('jaws closed on the shank (snug) but not tightened, key out', GOAL_POSES, {
    }, 0.5, {'TEST_MODEL': '{"gripped": True, "tightness": 0.12, "engaged": False, "opening": 0.006}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('tight and centred, key in its rest, but the right hand still on the bit 40 mm out from the nose', GOAL_POSES, {
    }, 0.5, {'TEST_MODEL': '{"gripped": True, "tightness": 0.34, "engaged": False, "opening": 0.006}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (np.array([X_AX, Y_NOSE - 0.040, TOP + Z_AX]), 0.0)}'}),
    ('tight, but only 15 mm of shank in the jaws', GOAL_POSES, {
        'bit': {'pos': (0.24, -0.085, 0.7825)},
    }, 0.25, {'TEST_MODEL': '{"gripped": True, "tightness": 0.34, "engaged": False, "opening": 0.006}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('tight; the key laid on the bench beside the drill (not in a hole)', GOAL_POSES, {
        'key': {'pos': (0.16, -0.08, 0.7515), 'tilt_deg': 90, 'tilt_axis': 'y'},
    }, 1, {'TEST_MODEL': '{"gripped": True, "tightness": 0.34, "engaged": False, "opening": 0.006}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('tight, key out; the drill slid 60 mm along the bench and turned 30 degrees on it', GOAL_POSES, {
        'drill': {'pos': (0.3, -0.1, 0.7825), 'yaw_deg': 30},
        'bit': {'pos': (0.288, -0.0792, 0.7825), 'yaw_deg': 30},
    }, 1, {'TEST_MODEL': '{"gripped": True, "tightness": 0.34, "engaged": False, "opening": 0.006}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ("tight, key out; the drill turned over onto its other side (rolled half a turn about the chuck's axis)", GOAL_POSES, {
        'drill': {'pos': (0.24, -0.1, 0.7825), 'yaw_deg': 180, 'tilt_deg': 180, 'tilt_axis': 'y', 'quat': (0.0, 1.0, 0.0, 6.123233995736766e-17)},
    }, 1, {'TEST_MODEL': '{"gripped": True, "tightness": 0.34, "engaged": False, "opening": 0.006}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('tight, key out; the drill lifted 40 mm and tipped 20 degrees in the left hand', GOAL_POSES, {
        'drill': {'pos': (0.24, -0.1, 0.8225), 'tilt_deg': 20, 'tilt_axis': 'x'},
        'bit': {'pos': (0.24, -0.0775, 0.8307), 'tilt_deg': 20, 'tilt_axis': 'x'},
    }, 0.5, {'TEST_MODEL': '{"gripped": True, "tightness": 0.34, "engaged": False, "opening": 0.006}', 'TEST_GRIPPERS': '{"left": (np.array([X_AX - 0.080, Y_NOSE + 0.100, TOP + Z_AX + 0.040]), 0.0), "right": (PARK["right"], 1.0)}'}),
    ('tightened, then the key left in the chuck (pilot in the top hole)', GOAL_POSES, {
        'key': {'pos': (0.24, -0.063, 0.7925)},
    }, 0, {'TEST_MODEL': '{"gripped": True, "tightness": 0.34, "engaged": True, "opening": 0.006}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),  # the maker's warning - a key left in a chuck is flung when the drill starts
    ('key standing in the hole but not meshed (lifted 1.5 mm)', GOAL_POSES, {
        'key': {'pos': (0.24, -0.063, 0.794)},
    }, 0, {'TEST_MODEL': '{"gripped": True, "tightness": 0.34, "engaged": False, "opening": 0.006}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),  # still in the chuck
    ('bit put in flutes first (shank sticking out)', GOAL_POSES, {
        'bit': {'pos': (0.24, -0.141, 0.7825), 'yaw_deg': 180},
    }, 0, {'TEST_MODEL': '{"gripped": False, "tightness": 0.0, "engaged": False, "opening": 0.0075}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),  # the bit is backwards; nothing is fitted
    ("drill pushed off the bench's right edge onto the floor, bit tight in it, key out", GOAL_POSES, {
        'drill': {'pos': (0.24, -0.75, 0.0225)},
        'bit': {'pos': (0.24, -0.726, 0.0225)},
    }, 0, {'TEST_MODEL': '{"gripped": True, "tightness": 0.34, "engaged": False, "opening": 0.006}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),  # the drill fell off the bench
    ("drill held out past the bench's right edge, bit tight in it, key out", GOAL_POSES, {
        'drill': {'pos': (0.24, -0.82, 0.8475)},
        'bit': {'pos': (0.24, -0.796, 0.8475)},
    }, 0, {'TEST_MODEL': '{"gripped": True, "tightness": 0.34, "engaged": False, "opening": 0.006}', 'TEST_GRIPPERS': '{"left": (np.array([X_AX - 0.080, -0.720, TOP + 0.100]), 0.0), "right": (PARK["right"], 1.0)}'}),  # the drill has left the workspace (its centre of mass is outside the bench top)
    ('bit laid on the bench beside the chuck, parallel to it and level with the jaws (not in them), key in its rest', START_POSES, {
        'bit': {'pos': (0.2, -0.08, 0.7505)},
    }, 0, {'TEST_MODEL': '{"gripped": False, "tightness": 0.0, "engaged": False, "opening": 0.0075}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),  # the shank is 51 mm off the chuck's axis, not between the jaws
    ('bit left on its rest, key in its rest', START_POSES, {
    }, 0, {}),  # nothing done
    ("clamped crooked, the bit 2 degrees off the chuck's axis (2.7 mm out 50 mm in front)", GOAL_POSES, {
        'bit': {'pos': (0.24, -0.076, 0.7825), 'tilt_deg': 2, 'tilt_axis': 'x'},
    }, 0, {'TEST_MODEL': '{"gripped": True, "tightness": 0.34, "engaged": False, "opening": 0.006}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),  # clamped off the chuck's axis, the bit would wobble and snap
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
