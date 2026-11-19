"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest build-block-marble-run-and-roll-marble-to-cup    (in an environment with yam_arena installed)
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

TASK = load(Path(__file__).with_name("build_block_marble_run_and_roll_marble_to_cup.py"))
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
    'support_near': {'pos': (0.16, -0.0375, 0.768)},
    'support_far': {'pos': (0.16, 0.1175, 0.768)},
    'start_beam': {'pos': (0.16, 0.04, 0.8085)},
    'catch_cup': {'pos': (0.16, 0.06, 0.748)},
    'turn_down': {'pos': (0.25, 0.095, 0.768), 'yaw_deg': 90},
    'turn_across': {'pos': (0.25, 0.17, 0.768), 'yaw_deg': 90},
    'straight_block': {'pos': (0.25, 0.245, 0.768)},
    'marble': {'pos': (0.15, 0.2, 0.7555)},
}
GOAL_POSES = {
    'support_near': {'pos': (0.16, -0.0375, 0.7675)},
    'support_far': {'pos': (0.16, 0.1175, 0.7675)},
    'start_beam': {'pos': (0.16, 0.04, 0.8075)},
    'catch_cup': {'pos': (0.16, 0.06, 0.7475)},
    'turn_down': {'pos': (0.16, 0.06, 0.8475)},
    'turn_across': {'pos': (0.16, 0.104, 0.8475)},
    'straight_block': {'pos': (0.16, 0.104, 0.8875)},
    'marble': {'pos': (0.16, 0.06, 0.756)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {'TEST_RUN': "'arrived'"}


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'catch_cup', 'start_beam', 'straight_block', 'marble', 'turn_down', 'support_far', 'support_near', 'turn_across'}
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
    ('built, marble not yet run (still on the table)', START_POSES, {
        'turn_down': {'pos': (0.16, 0.06, 0.8475)},
        'turn_across': {'pos': (0.16, 0.104, 0.8475)},
        'straight_block': {'pos': (0.16, 0.104, 0.8875)},
    }, 0.75, {}),
    ('built, marble stuck inside the turn-across block (run still going at the end)', GOAL_POSES, {
        'marble': {'pos': (0.16, 0.092, 0.8435)},
    }, 0.75, {'TEST_RUN': "'running'"}),
    ('only the turn-down block placed (joint 1)', START_POSES, {
        'turn_down': {'pos': (0.16, 0.06, 0.8475)},
    }, 0.25, {}),
    ('turn-down and turn-across joined, straight block still in its row', START_POSES, {
        'turn_down': {'pos': (0.16, 0.06, 0.8475)},
        'turn_across': {'pos': (0.16, 0.104, 0.8475)},
    }, 0.5, {}),
    ("turn-across block on the beam's far end with the straight block on it, turn-down block not placed (joint 3)", START_POSES, {
        'turn_across': {'pos': (0.16, 0.095, 0.8475)},
        'straight_block': {'pos': (0.16, 0.095, 0.8875)},
    }, 0.25, {}),
    ('built with harmless differences (blocks 3 mm off, straight block turned 30 deg and upside down)', GOAL_POSES, {
        'turn_down': {'pos': (0.163, 0.061, 0.8475), 'yaw_deg': 4},
        'turn_across': {'pos': (0.161, 0.105, 0.8475), 'yaw_deg': -3},
        'straight_block': {'pos': (0.162, 0.102, 0.8875), 'yaw_deg': 30, 'tilt_deg': 180, 'tilt_axis': 'x', 'quat': (0.9659258358036645, 0.25881900959356163, 0.0, 0.0)},
    }, 1, {'TEST_RUN': "'arrived'"}),
    ("mirrored build (turn-across block on the beam's right of the turn-down block, both turned round)", GOAL_POSES, {
        'turn_down': {'pos': (0.16, 0.06, 0.8475), 'yaw_deg': 180},
        'turn_across': {'pos': (0.16, 0.016, 0.8475), 'yaw_deg': 180},
        'straight_block': {'pos': (0.16, 0.016, 0.8875)},
    }, 1, {'TEST_RUN': "'arrived'"}),
    ('built, marble carried or dropped straight into the cup (never ran from the top hole)', GOAL_POSES, {
    }, 0.75, {}),  # no run was seen from the top hole, so the marble earns nothing
    ('built, marble left the run at a joint and then fell into the cup', GOAL_POSES, {
    }, 0.75, {'TEST_RUN': "'left'"}),  # the marble came out of the blocks on the way
    ('nothing built, marble in the cup', START_POSES, {
        'marble': {'pos': (0.16, 0.06, 0.756)},
    }, 0, {'TEST_RUN': "'arrived'"}),  # no joint made
    ('turn-down block back to front (side hole facing away from the turn-across block)', GOAL_POSES, {
        'turn_down': {'pos': (0.16, 0.06, 0.8475), 'yaw_deg': 180},
    }, 0.5, {'TEST_RUN': "'arrived'"}),  # joint 2 is not made (and the marble could not pass)
    ('turn-across block turned a quarter turn (side hole facing forward)', GOAL_POSES, {
        'turn_across': {'pos': (0.16, 0.104, 0.8475), 'yaw_deg': 90},
    }, 0.5, {}),  # its side hole does not face the turn-down block's
    ('turn-across block 9 mm sideways off the joint', GOAL_POSES, {
        'turn_across': {'pos': (0.169, 0.104, 0.8475)},
        'straight_block': {'pos': (0.169, 0.104, 0.8875)},
    }, 0.5, {'TEST_RUN': "'arrived'"}),  # the side holes are 9 mm apart across the joint
    ('turn-across block 10 mm away from the turn-down block (gap)', GOAL_POSES, {
        'turn_across': {'pos': (0.16, 0.11, 0.8475)},
        'straight_block': {'pos': (0.16, 0.11, 0.8875)},
    }, 0.5, {'TEST_RUN': "'arrived'"}),  # the faces are more than 6 mm apart
    ('straight block hovering 4 mm over the turn-across block (still held)', GOAL_POSES, {
        'straight_block': {'pos': (0.16, 0.104, 0.8915)},
    }, 0.5, {'TEST_RUN': "'arrived'"}),  # not sitting on the block below
    ('straight block still moving', GOAL_POSES, {
        'straight_block': {'pos': (0.16, 0.104, 0.8875), 'vel': (0.2, 0.0, 0.0)},
    }, 0.5, {'TEST_RUN': "'arrived'"}),  # not still
    ('turn-down block tipped on its side on the beam', GOAL_POSES, {
        'turn_down': {'pos': (0.16, 0.06, 0.8475), 'tilt_deg': 90, 'tilt_axis': 'x'},
    }, 0.25, {'TEST_RUN': "'arrived'"}),  # its holes no longer meet the beam's or the turn-across block's
    ('start beam knocked off its cubes onto the table', GOAL_POSES, {
        'start_beam': {'pos': (0.16, 0.04, 0.7675)},
    }, 0, {'TEST_RUN': "'arrived'"}),  # the start structure is broken, so no joint counts
    ('blocks stacked on the table, not on the beam', START_POSES, {
        'turn_down': {'pos': (0.25, 0.1, 0.7675)},
        'turn_across': {'pos': (0.25, 0.144, 0.7675)},
        'straight_block': {'pos': (0.25, 0.144, 0.8075)},
    }, 0, {}),  # the joints only count on the start beam
    ('marble pushed off the table', START_POSES, {
        'marble': {'pos': (0.4, 0.7, 0.5475)},
    }, 0, {}),  # nothing built
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
