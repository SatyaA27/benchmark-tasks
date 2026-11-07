"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest change-pipette-tip-hands-free    (in an environment with yam_arena installed)
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

TASK = load(Path(__file__).with_name("change_pipette_tip_hands_free.py"))
T = TASK.module
OBJ = {o.name: o for o in T.OBJECTS}
RUN_LIMIT_STEPS = 1400


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
    'pipette': {'pos': (0.26, 0.18, 0.7635), 'joints': {'sleeve': 0.0, 'ball_xp': 0.0, 'ball_xn': 0.0, 'ball_yp': 0.0, 'ball_yn': 0.0}},
    'tip_used': {'pos': (0.26, 0.18, 0.7635)},
    'tip_a': {'pos': (0.2, 0.105, 0.7635)},
    'tip_b': {'pos': (0.13, 0.185, 0.7635)},
}
GOAL_POSES = {
    'pipette': {'pos': (0.26, 0.18, 0.7635), 'joints': {'sleeve': 0.0, 'ball_xp': -0.0015, 'ball_xn': 0.0015, 'ball_yp': -0.0015, 'ball_yn': 0.0015}},
    'tip_used': {'pos': (0.14, 0.025, 0.7645), 'tilt_deg': 90, 'tilt_axis': 'y'},
    'tip_a': {'pos': (0.26, 0.18, 0.7635)},
    'tip_b': {'pos': (0.13, 0.185, 0.7635)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {}


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'tip_used', 'tip_b', 'tip_a', 'pipette'}
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
    ('still held 10 mm above stand fork', GOAL_POSES, {
        'pipette': {'pos': (0.26, 0.18, 0.7735), 'joints': {'sleeve': 0.0, 'ball_xp': -0.0015, 'ball_xn': 0.0015, 'ball_yp': -0.0015, 'ball_yn': 0.0015}},
        'tip_a': {'pos': (0.26, 0.18, 0.7735)},
    }, 0.5, {}),
    ('half done carried 30 mm above park', GOAL_POSES, {
        'pipette': {'pos': (0.26, 0.18, 0.7935), 'joints': {'sleeve': 0.0, 'ball_xp': -0.0015, 'ball_xn': 0.0015, 'ball_yp': -0.0015, 'ball_yn': 0.0015}},
        'tip_a': {'pos': (0.26, 0.18, 0.7935)},
    }, 0.5, {}),
    ('parked but moving at .2 m/s', GOAL_POSES, {
        'pipette': {'pos': (0.26, 0.18, 0.7635), 'vel': (0.2, 0.0, 0.0), 'joints': {'sleeve': 0.0, 'ball_xp': -0.0015, 'ball_xn': 0.0015, 'ball_yp': -0.0015, 'ball_yn': 0.0015}},
    }, 0.75, {}),
    ('fresh tip halfway pressed 2.5 mm short of seat', GOAL_POSES, {
        'tip_a': {'pos': (0.26, 0.18, 0.761)},
    }, 0.25, {}),  # Offline capture-zone geometry earns disposal only; a dynamic loss of a fresh tip would be sticky zero.
    ('used tip disposed but both fresh tips still in rack', GOAL_POSES, {
        'tip_a': {'pos': (0.2, 0.105, 0.7635)},
    }, 0.25, {}),
    ('stand bearing 2 mm solver penetration', GOAL_POSES, {
        'pipette': {'pos': (0.26, 0.18, 0.7616), 'joints': {'sleeve': 0.0, 'ball_xp': -0.0015, 'ball_xn': 0.0015, 'ball_yp': -0.0015, 'ball_yn': 0.0015}},
        'tip_a': {'pos': (0.26, 0.18, 0.7616)},
    }, 1, {}),
    ('pipette resting 5.8 mm deeper in the slot, sleeve flange against the stand back wall', GOAL_POSES, {
        'pipette': {'pos': (0.26, 0.1858, 0.7635), 'joints': {'sleeve': 0.0, 'ball_xp': -0.0015, 'ball_xn': 0.0015, 'ball_yp': -0.0015, 'ball_yn': 0.0015}},
        'tip_a': {'pos': (0.26, 0.1858, 0.7635)},
    }, 1, {}),  # The deepest reachable park; the 56 mm sleeve flange meets the back wall's inner face, 34 mm behind the stand centre, at +6 mm, and the 12 mm shoulder bar rests flat on both rails.
    ('red team: pipette resting 12 mm toward the stand fork slot mouth', GOAL_POSES, {
        'pipette': {'pos': (0.26, 0.168, 0.7635), 'joints': {'sleeve': 0.0, 'ball_xp': -0.0015, 'ball_xn': 0.0015, 'ball_yp': -0.0015, 'ball_yn': 0.0015}},
        'tip_a': {'pos': (0.26, 0.168, 0.7635)},
    }, 1, {}),  # The shoulder bar (y -0.018 to -0.006 from the slot centre) still sits fully on both rails, which run to -0.028; the pipette is stably parked in its stand.
    ('red team: pipette resting 3.5 mm sideways between the rails', GOAL_POSES, {
        'pipette': {'pos': (0.2635, 0.18, 0.7635), 'joints': {'sleeve': 0.0, 'ball_xp': -0.0015, 'ball_xn': 0.0015, 'ball_yp': -0.0015, 'ball_yn': 0.0015}},
        'tip_a': {'pos': (0.2635, 0.18, 0.7635)},
    }, 1, {}),  # The 44 mm shoulder bar still overlaps both rails (1.5 mm on one, 10 mm on the other) with its CoM between them; it is parked.
    ("red team: used tip leaning on the bin's internal fork plate, stem end on the bin floor", GOAL_POSES, {
        'tip_used': {'pos': (0.1641, 0.025, 0.7608), 'tilt_deg': 60, 'tilt_axis': 'y'},
    }, 1, {}),  # The tip lies wholly inside the bin's footprint, its collar corner on the floor and its mouth against the eject fork's own vertical plate 43 mm from the eject axis; it was ejected into the bin and is disposed of.
    ('unused fresh tip standing in wrong rack slot', GOAL_POSES, {
        'tip_b': {'pos': (0.13, 0.105, 0.7635)},
    }, 0, {}),  # Static layouts require every fresh tip in its own hole, seated, or engaged; the unused fresh tip is in an empty hole belonging to neither fresh tip.
    ('used tip standing in empty rack slot', GOAL_POSES, {
        'tip_used': {'pos': (0.2, 0.185, 0.7635)},
    }, 0, {}),  # The rack supports the collar, but the used tip is outside the bin.
    ('pipette propped 8 degrees on stand fork edge', GOAL_POSES, {
        'pipette': {'pos': (0.26, 0.1981, 0.7656), 'tilt_deg': 8, 'tilt_axis': 'x', 'joints': {'sleeve': 0.0, 'ball_xp': -0.0015, 'ball_xn': 0.0015, 'ball_yp': -0.0015, 'ball_yn': 0.0015}},
        'tip_a': {'pos': (0.26, 0.1981, 0.7656), 'tilt_deg': 8, 'tilt_axis': 'x'},
    }, 0.5, {}),  # The shoulder's lower edge bears on the fork top and the fresh tip remains seated, but 8 degrees fails the stand's under-5-degree tilt requirement.
    ('used tip propped 30 degrees against inner bin wall above rim', GOAL_POSES, {
        'tip_used': {'pos': (0.14, 0.001, 0.764), 'tilt_deg': 30, 'tilt_axis': 'x'},
    }, 0, {}),  # The lower stem touches the bin floor and the funnel bears on the south rim; above the rim the mouth leans out over the wall to y=-0.034, 11 mm outside the interior, so the tip hangs over the edge and fails in_bin.
    ('pipette yawed 45 degrees in stand, 12 mm toward the slot mouth', GOAL_POSES, {
        'pipette': {'pos': (0.26, 0.168, 0.7635), 'yaw_deg': 45, 'joints': {'sleeve': 0.0, 'ball_xp': -0.0015, 'ball_xn': 0.0015, 'ball_yp': -0.0015, 'ball_yn': 0.0015}},
        'tip_a': {'pos': (0.26, 0.168, 0.7635), 'yaw_deg': 45},
    }, 1, {}),  # At 45 degrees the 44 x 12 mm shoulder bar spans 39.6 mm along x, reaching 2.8 mm onto each rail; 12 mm toward the mouth keeps the 56 mm sleeve flange's corner clear of the back wall.
    ('pipette yawed 90 degrees at fork height', GOAL_POSES, {
        'pipette': {'pos': (0.26, 0.18, 0.7635), 'yaw_deg': 90, 'joints': {'sleeve': 0.0, 'ball_xp': -0.0015, 'ball_xn': 0.0015, 'ball_yp': -0.0015, 'ball_yn': 0.0015}},
        'tip_a': {'pos': (0.26, 0.18, 0.7635), 'yaw_deg': 90},
    }, 0.5, {}),  # Turned 90 degrees, the bar spans only 12 mm along x and sits wholly inside the 34 mm rail gap; nothing supports the pipette at fork height, so it is not parked.
    ('pipette 6 mm sideways, shoulder bar on one rail only', GOAL_POSES, {
        'pipette': {'pos': (0.266, 0.18, 0.7635), 'joints': {'sleeve': 0.0, 'ball_xp': -0.0015, 'ball_xn': 0.0015, 'ball_yp': -0.0015, 'ball_yn': 0.0015}},
        'tip_a': {'pos': (0.266, 0.18, 0.7635)},
    }, 0.5, {}),  # The bar's far end stops 1 mm short of the far rail, so the pipette bears on one rail with its centre of mass outside that support and would tip.
    ('used tip lying in bin pointing opposite demo landing pose', GOAL_POSES, {
        'tip_used': {'pos': (0.14, 0.025, 0.7645), 'yaw_deg': 180, 'tilt_deg': 90, 'tilt_axis': 'y'},
    }, 1, {}),  # Reversing the horizontal tip keeps its entire box inside the bin; disposal scoring has no yaw requirement.
    ('fresh tip nudged 4 mm and tilted 8 deg in its hole, rest perfect', GOAL_POSES, {
        'tip_b': {'pos': (0.134, 0.185, 0.7635), 'tilt_deg': 8, 'tilt_axis': 'y'},
    }, 0.5, {}),  # In-hole prevents contamination, but 4 mm exceeds the unchanged 3 mm standing tolerance for the spare. Restoring it permits 1.0 without a sticky zero.
    ('pipette lying on table with fresh tip seated', GOAL_POSES, {
        'pipette': {'pos': (0.29, 0.1, 0.7755), 'tilt_deg': 90, 'tilt_axis': 'y', 'joints': {'sleeve': 0.0, 'ball_xp': -0.0015, 'ball_xn': 0.0015, 'ball_yp': -0.0015, 'ball_yn': 0.0015}},
        'tip_a': {'pos': (0.29, 0.1, 0.7755), 'tilt_deg': 90, 'tilt_axis': 'y'},
    }, 0, {}),
    ('used tip resting on bin rim', GOAL_POSES, {
        'tip_used': {'pos': (0.14, 0.025, 0.7915), 'tilt_deg': 90, 'tilt_axis': 'y'},
    }, 0, {}),
    ('used tip half outside bin', GOAL_POSES, {
        'tip_used': {'pos': (0.255, 0.025, 0.7645), 'tilt_deg': 90, 'tilt_axis': 'y'},
    }, 0, {}),
    ('used tip on eject fork plate', GOAL_POSES, {
        'tip_used': {'pos': (0.17, 0.0535, 0.8595), 'tilt_deg': 90, 'tilt_axis': 'y'},
    }, 0, {}),
    ('used tip still on nozzle and pipette parked', START_POSES, {
    }, 0, {}),
    ('swapped used seated fresh in bin', START_POSES, {
        'tip_a': {'pos': (0.14, 0.025, 0.7645), 'tilt_deg': 90, 'tilt_axis': 'y'},
    }, 0, {}),
    ('other fresh tip knocked out onto table', GOAL_POSES, {
        'tip_b': {'pos': (0.3, 0.2, 0.7615), 'tilt_deg': 90, 'tilt_axis': 'y'},
    }, 0, {}),
    ('start layout', START_POSES, {
    }, 0, {}),
    ('moving used tip inside bin', GOAL_POSES, {
        'tip_used': {'pos': (0.14, 0.025, 0.7645), 'tilt_deg': 90, 'tilt_axis': 'y', 'vel': (0.2, 0.0, 0.0)},
    }, 0, {}),
    ('fresh tip tilted 12 degrees relative to nozzle at park', GOAL_POSES, {
        'tip_a': {'pos': (0.26, 0.18, 0.7635), 'tilt_deg': 12, 'tilt_axis': 'x'},
    }, 0.25, {}),
    ('sticky violation gripper touched a tip', GOAL_POSES, {
    }, 0, {'TEST_STICKY': "('gripper touched a tip',)"}),
    ('sticky violation removed other than by the ejector', GOAL_POSES, {
    }, 0, {'TEST_STICKY': "('removed other than by the ejector',)"}),
    ('sticky violation a fresh tip was dropped or knocked over', GOAL_POSES, {
    }, 0, {'TEST_STICKY': "('a fresh tip was dropped or knocked over',)"}),
    ('sticky violation pipette dropped', GOAL_POSES, {
    }, 0, {'TEST_STICKY': "('pipette dropped',)"}),
    ('red team: pipette yawed 45 degrees at the slot mouth, one shoulder corner past the rail end', GOAL_POSES, {
        'pipette': {'pos': (0.26, 0.158, 0.7635), 'yaw_deg': 45, 'joints': {'sleeve': 0.0, 'ball_xp': -0.0015, 'ball_xn': 0.0015, 'ball_yp': -0.0015, 'ball_yn': 0.0015}},
        'tip_a': {'pos': (0.26, 0.158, 0.7635), 'yaw_deg': 45},
    }, 0.5, {}),  # At 45 degrees the shoulder bar's only overlap with the -x rail is the corner triangle at y -0.036 to -0.031 from the stand centre, beyond the rail end at -0.028; that side has no support, so the pipette is held by one rail and is not parked.
    ('red team: pipette yawed 50 degrees at the slot mouth, one shoulder corner past the rail end', GOAL_POSES, {
        'pipette': {'pos': (0.26, 0.158, 0.7635), 'yaw_deg': 50, 'joints': {'sleeve': 0.0, 'ball_xp': -0.0015, 'ball_xn': 0.0015, 'ball_yp': -0.0015, 'ball_yn': 0.0015}},
        'tip_a': {'pos': (0.26, 0.158, 0.7635), 'yaw_deg': 50},
    }, 0.5, {}),  # At 50 degrees the bar reaches the -x rail only with its corner at y -0.035 from the stand centre, 7 mm past the 28 mm rail end; nothing supports that side and the pipette would tip off the other rail.
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
