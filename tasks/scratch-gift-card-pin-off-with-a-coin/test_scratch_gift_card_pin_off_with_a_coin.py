"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest scratch-gift-card-pin-off-with-a-coin    (in an environment with yam_arena installed)
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

TASK = load(Path(__file__).with_name("scratch_gift_card_pin_off_with_a_coin.py"))
T = TASK.module
OBJ = {o.name: o for o in T.OBJECTS}
RUN_LIMIT_STEPS = 3600


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
    'card': {'pos': (0.2, -0.0186, 0.7479)},
    'quarter': {'pos': (0.13, -0.175, 0.7484), 'yaw_deg': -5.625},
}
GOAL_POSES = {
    'card': {'pos': (0.2, -0.0186, 0.7479)},
    'quarter': {'pos': (0.2, -0.3, 0.947), 'yaw_deg': -5.625},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {'TEST_REC': 'rec_with(15)'}


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'card', 'quarter'}
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
    ('untouched start, sticker whole', START_POSES, {
    }, 0, {}),
    ('14 characters uncovered, none worn', GOAL_POSES, {
    }, 0.75, {'TEST_REC': 'rec_with(14)'}),
    ('13 characters uncovered', GOAL_POSES, {
    }, 0.75, {'TEST_REC': 'rec_with(13)'}),
    ('12 characters uncovered', GOAL_POSES, {
    }, 0.5, {'TEST_REC': 'rec_with(12)'}),
    ('9 characters uncovered', GOAL_POSES, {
    }, 0.5, {'TEST_REC': 'rec_with(9)'}),
    ('8 characters uncovered', GOAL_POSES, {
    }, 0.25, {'TEST_REC': 'rec_with(8)'}),
    ('4 characters uncovered', GOAL_POSES, {
    }, 0.25, {'TEST_REC': 'rec_with(4)'}),
    ('3 characters uncovered', GOAL_POSES, {
    }, 0, {'TEST_REC': 'rec_with(3)'}),
    ('all uncovered but one 0.5 mm fleck of silver left on one character (still reads)', GOAL_POSES, {
    }, 1, {'TEST_REC': 'rec_with(15, left=[5])'}),
    ('all uncovered but two 0.5 mm cells of silver left on one character (that one does not read)', GOAL_POSES, {
    }, 0.75, {'TEST_REC': 'rec_with(15, left=[5, 5])'}),
    ('the middle characters uncovered, both ends still silver', GOAL_POSES, {
    }, 0.5, {'TEST_REC': 'rec_with((3, 4, 5, 6, 7, 8, 9, 10, 11))'}),
    ('all uncovered, every character passed over exactly three times', GOAL_POSES, {
    }, 1, {'TEST_REC': 'rec_with(15, passes=3)'}),
    ('all uncovered, one character passed over a fourth time (worn)', GOAL_POSES, {
    }, 0, {'TEST_REC': 'rec_with(15, passes=[3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])'}),
    ('all uncovered, one character pressed over 6 N (worn)', GOAL_POSES, {
    }, 0, {'TEST_REC': 'rec_with(15, hard=[7])'}),
    ('10 uncovered but one of them worn', GOAL_POSES, {
    }, 0, {'TEST_REC': 'rec_with(10, hard=[2])'}),
    ('all uncovered, the right hand still resting 1 cm over the card', GOAL_POSES, {
    }, 0.75, {'TEST_REC': 'rec_with(15)', 'TEST_GRIPPERS': "{'right': (over_card(0.010), 0.0), 'left': (np.array([0.32, 0.30, TOP + 0.26]), 1.0)}"}),
    ('all uncovered, hands 5 cm clear', GOAL_POSES, {
    }, 1, {'TEST_REC': 'rec_with(15)', 'TEST_GRIPPERS': "{'right': (over_card(0.050), 0.0), 'left': (np.array([0.32, 0.30, TOP + 0.26]), 1.0)}"}),
    ('all uncovered, the left hand still pressing the hanger', GOAL_POSES, {
    }, 0.75, {'TEST_REC': 'rec_with(15)', 'TEST_GRIPPERS': "{'right': (over_card(0.050), 0.0), 'left': (pin_target('press')[0], 0.0)}"}),
    ('all uncovered, card slid 4 cm', GOAL_POSES, {
        'card': {'pos': (0.24, -0.0186, 0.7479)},
    }, 1, {'TEST_REC': 'rec_with(15)'}),
    ('all uncovered, card slid 6 cm', GOAL_POSES, {
        'card': {'pos': (0.2, -0.0786, 0.7479)},
    }, 0, {'TEST_REC': 'rec_with(15)'}),
    ('all uncovered, card turned 25 degrees', GOAL_POSES, {
        'card': {'pos': (0.2, -0.0186, 0.7479), 'yaw_deg': 25},
    }, 1, {'TEST_REC': 'rec_with(15)'}),
    ('all uncovered, card turned 35 degrees', GOAL_POSES, {
        'card': {'pos': (0.2, -0.0186, 0.7479), 'yaw_deg': -35},
    }, 0, {'TEST_REC': 'rec_with(15)'}),
    ('all uncovered, card flipped face down', GOAL_POSES, {
        'card': {'pos': (0.2, -0.0186, 0.7479), 'tilt_deg': 180, 'tilt_axis': 'x'},
    }, 0, {'TEST_REC': 'rec_with(15)'}),
    ('card dragged off toward the arm, sticker untouched', START_POSES, {
        'card': {'pos': (0.2, -0.1386, 0.7479)},
    }, 0, {'TEST_REC': 'rec_with(15)'}),  # moved 12 cm, so 0 even with a clean record
    ('quarter laid on the sticker, nothing scratched', START_POSES, {
        'quarter': {'pos': (0.2, -0.09, 0.7491), 'yaw_deg': -5.625},
    }, 0, {}),  # a coin lying flat scratches nothing
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
