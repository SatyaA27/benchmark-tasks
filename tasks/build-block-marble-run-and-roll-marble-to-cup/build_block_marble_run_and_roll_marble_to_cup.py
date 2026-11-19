"""Wooden block marble run: stack three drilled beech blocks on the start beam so their bores join up, then put the
glass marble in at the top so that it rolls down through every block into the catch cup.

On the table (metres; x forward from the arm bases, y to the robot's left, z up; table top z = TOP):
- The start structure, already built: two plain 40 mm beech cubes (supports) standing on the table at
  y = BEAM_Y -+ 0.0775, a 160 x 40 x 40 mm drilled beech beam lying across them along y (centre (BEAM_X, BEAM_Y),
  its vertical 24 mm bore at y = BEAM_Y + 0.020), and a stainless 1.5 oz sauce cup standing under that bore.
- Three loose drilled 40 mm beech cubes in a row at x = 0.25: the turn-down block (holes in one side and the bottom),
  the turn-across block (holes in the top and one side; both side holes face sideways at the start, each turned a
  different way), and the straight block (a straight bore from top to bottom).
- One 15 mm blue glass marble lying on the table at MARBLE_XY.
Goal layout: the turn-down block on the beam over the beam's hole with its side hole facing the robot's left; the
turn-across block beside it on the beam, its side hole against the turn-down block's side hole; the straight block on
top of the turn-across block. The marble, let go at the straight block's top hole, falls down, turns sideways, turns
down again and drops through the beam into the cup.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from channel_blocks import (BEAM_HOLE, BEAM_LEN, CUP_BASE, CUP_BOTTOM_R, CUP_H, CUP_TOP_R, HOLES, MODULE, TURN_R,  # noqa: E402
                            beam, cup, plain, straight, turn)
from yam_parts import Part, ball  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
RUN_LIMIT_STEPS = 2400  # 120 s at 20 Hz
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
PROBE = False       # GPU-only probe copy: no straight block; the marble goes in at the turn-across block
DIAG = False        # GPU-only copies print the marble's path
DEMO_WRONG = False  # wrong-way copy: the turn-down block is set down back to front (side hole facing away)

# ------------------------------------------------------------------------------------------------ layout
H = MODULE / 2
BEAM_X, BEAM_Y = 0.160, 0.040
SUPPORT_DY = 0.0775
GAP = 0.004                   # planned gap between the two side faces that meet (the hole joint)
MARBLE_R = 0.0075             # 15 mm glass marble (maker)
MARBLE_MASS = 0.0044          # 4/3 pi r^3 x 2.5 g/cm3 (estimate)
MARBLE_XY = (0.150, 0.200)
START = {
    "turn_down": ((0.250, 0.095), math.pi / 2),    # side hole facing the robot (-x)
    "turn_across": ((0.250, 0.170), math.pi / 2),  # side hole facing away (+x)
    "straight_block": ((0.250, 0.245), 0.0),
}
KIND = {"start_beam": "beam", "turn_down": "turn_up", "turn_across": "turn_down", "straight_block": "straight"}
BLUE = (0.10, 0.22, 0.85)
STEEL = (0.45, 0.50, 0.56)

OBJECTS = [
    plain("support_near", (BEAM_X, BEAM_Y - SUPPORT_DY, TOP + H + 0.0005)),
    plain("support_far", (BEAM_X, BEAM_Y + SUPPORT_DY, TOP + H + 0.0005)),
    beam("start_beam", (BEAM_X, BEAM_Y, TOP + MODULE + H + 0.001)),
    cup("catch_cup", (BEAM_X, BEAM_Y + BEAM_HOLE), TOP + 0.0005, color=STEEL),
    turn("turn_down", (*START["turn_down"][0], TOP + H + 0.0005), yaw=START["turn_down"][1], outlet_down=True),
    turn("turn_across", (*START["turn_across"][0], TOP + H + 0.0005), yaw=START["turn_across"][1]),
    Part("marble", (*MARBLE_XY, TOP + MARBLE_R + 0.0005), (ball((0, 0, 0), MARBLE_R),), mass=MARBLE_MASS,
         color=BLUE, friction=(0.4, 0.3), contact_offset=0.0005),
]
if not PROBE:
    OBJECTS.insert(-1, straight("straight_block", (*START["straight_block"][0], TOP + H + 0.0005),
                                yaw=START["straight_block"][1]))
BLOCKS = ["turn_down", "turn_across"] + ([] if PROBE else ["straight_block"])
TOP_BLOCK = BLOCKS[-1]

# ------------------------------------------------------------------------------------------------ scoring numbers
JOINT_TOL = 0.006     # two holes that meet: centres within 6 mm of each other across the joint (ruler)
GAP_MAX = 0.006       # the two side faces that meet: at most 6 mm apart (ruler)
SEAT_TOL = 0.003      # a block sits on the one below: its hole face within 3 mm of the face below
FACE_DEG = 20.0       # a hole faces the hole it meets within 20 degrees
TILT_MAX = 5.0        # blocks sit flat: tilted at most 5 degrees
BEAM_HEIGHT = (MODULE + H, 0.004)   # the start beam still lies on its supports: centre 60 +- 4 mm above the table
CH_SLACK = 0.007      # marble centre within 7 mm of a bore's centre line counts as inside the bore
CH_EXT = 0.008        # ... and up to 8 mm past a hole face (the joint between two blocks)
INLET_DEPTH = 0.008   # a run starts when the marble centre is 8 mm or more inside the top block's top hole
CUP_MARGIN = 0.004    # in the cup: centre below the rim and at least 4 mm inside the wall
RELEASE_CLEAR = 0.030
STILL_S = 1.0
MARBLE_STILL = 0.020  # the marble lies in the cup moving under 2 cm/s ...
MARBLE_SPIN = 3.0     # ... (rolling at 2 cm/s is 2.7 rad/s; a ball on the cup's flat base keeps rolling slowly)
TEST_GRIPPERS = None
TEST_RUN = None       # offline tests only: the run record a layout stands for ('idle', 'running', 'arrived', 'left')


# ------------------------------------------------------------------------------------------------ run context
def _run_context():
    context = {"env": None, "t": None, "step": 0, "episode": 0}
    try:
        from yam_arena import inspect_embodiment as _ie
    except ImportError:
        return context
    cls = _ie.YamArenaEmbodiment
    if hasattr(cls, "_rc_run"):
        return cls._rc_run
    original_step, original_reset = cls._task_step, cls.reset

    def task_step(self, *args, **kwargs):
        context["env"], context["t"] = self._env, self._t
        context["step"] += 1
        return original_step(self, *args, **kwargs)

    def reset(self, *args, **kwargs):
        result = original_reset(self, *args, **kwargs)
        context["episode"] += 1
        context["t"] = 0.0
        context["env"] = self._env
        return result

    task_step._rc_wrapped = reset._rc_wrapped = True
    cls._rc_run = context
    cls._task_step, cls.reset = task_step, reset
    return context


_RUN = _run_context()
_MEMORY_EPISODE = None
_TIMERS = {}
_TRACK = {"step": None, "state": "idle"}


def grippers():
    if TEST_GRIPPERS is not None:
        return TEST_GRIPPERS
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    out = {}
    for arm in ("left", "right"):
        art = env.unwrapped.scene[arm + "_robot"]
        arm_ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        finger_ids = art.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        joints = art.data.joint_pos
        q = getattr(joints, "torch", joints)[0]
        if hasattr(q, "detach"):
            q = q.detach().cpu().numpy()
        q = np.asarray(q)
        tcp = ik.tcp_pose(q[arm_ids], station.BASES[arm])[0]
        opening = float(np.clip(np.mean(q[finger_ids]) / FINGER_OPEN, 0.0, 1.0))
        out[arm] = (tcp, opening)
    return out


def released(state, clear=RELEASE_CLEAR):
    """Every gripper TCP at least `clear` from the object's box."""
    hands = grippers()
    if hands is None:
        return True  # offline arm poses are unknown
    low, high = np.asarray(state.low), np.asarray(state.high)
    return all(np.linalg.norm(np.maximum(np.maximum(low - tcp, tcp - high), 0.0)) >= clear
               for tcp, _ in hands.values())


def _episode_memory():
    global _MEMORY_EPISODE
    if _MEMORY_EPISODE != _RUN["episode"]:
        _TIMERS.clear()
        _TRACK.update(step=None, state="idle")
        _MEMORY_EPISODE = _RUN["episode"]


def held_for(key, cond):
    """Elapsed time over consecutive observed control steps; repeat calls are harmless."""
    _episode_memory()
    t, step = _RUN["t"], _RUN["step"]
    if t is None:
        return math.inf if cond else 0.0
    previous_step, start = _TIMERS.get(key, (None, None))
    if not cond:
        _TIMERS[key] = (step, None)
        return 0.0
    if start is None or step not in (previous_step, previous_step + 1) or t < start:
        start = t
    _TIMERS[key] = (step, start)
    return float(t - start)


# ------------------------------------------------------------------------------------------------ geometry
def hole(state, name, index):
    """(centre, outward normal) in world of hole `index` of block `name` (see channel_blocks.HOLES)."""
    c, n = HOLES[KIND[name]][index]
    return state.to_world(c), state.rotation() @ np.asarray(n, float)


def in_bore(kind, q, slack=CH_SLACK, ext=CH_EXT):
    """Is the marble centre q (block frame) inside the block's bore, or just past one of its holes?"""
    x, y, z = (float(v) for v in q)
    if kind == "straight":
        return math.hypot(x, y) <= slack and abs(z) <= H + ext
    if kind == "beam":
        return math.hypot(x, y - BEAM_HOLE) <= slack and abs(z) <= H + ext
    if kind == "turn_up":  # the same block turned over about x
        y, z = -y, -z
    if math.hypot(x, y) <= slack and H <= z <= H + ext:
        return True
    if math.hypot(x, z) <= slack and -H - ext <= y <= -H:
        return True
    if max(abs(x), abs(y), abs(z)) > H:
        return False
    qy, qz = y + H, z - H        # from the bend's edge
    th = min(max(math.atan2(-qz, qy), 0.0), math.pi / 2)
    return math.sqrt(x * x + (qy - TURN_R * math.cos(th)) ** 2 + (qz + TURN_R * math.sin(th)) ** 2) <= slack


def cup_inside(objects, p, margin=CUP_MARGIN):
    c = objects["catch_cup"]
    q = c.to_local(p)
    if not (0.0 <= q[2] <= CUP_H):
        return False
    r = CUP_BOTTOM_R + (CUP_TOP_R - CUP_BOTTOM_R) * q[2] / CUP_H
    return math.hypot(q[0], q[1]) <= r - margin and c.tilt_deg() <= 30.0


def _in_run(objects, p):
    """Inside a bore of the run, in the drop from the beam's bottom hole to the cup, or in the cup."""
    for name in ["start_beam"] + BLOCKS:
        if in_bore(KIND[name], objects[name].to_local(p)):
            return True
    if cup_inside(objects, p, margin=-0.003):
        return True
    beam_hole, _ = hole(objects["start_beam"], "start_beam", 1)
    cup_top = objects["catch_cup"].to_world((0, 0, CUP_H))
    return (math.hypot(p[0] - beam_hole[0], p[1] - beam_hole[1]) <= 2 * H * 0.6
            and cup_top[2] - 0.005 <= p[2] <= beam_hole[2] + CH_EXT)


def top_hole(state, name):
    """The block's hole that faces up (centre, normal), in world."""
    return max((hole(state, name, i) for i in (0, 1)), key=lambda h: h[1][2])


def _at_inlet(objects, p):
    """The marble is inside the top block's top hole, at least INLET_DEPTH in: no gripper can hold it there."""
    st = objects[TOP_BLOCK]
    c, n = top_hole(st, TOP_BLOCK)
    depth = -float(np.dot(np.asarray(p) - c, n))
    return depth >= INLET_DEPTH and n[2] >= math.cos(math.radians(TILT_MAX)) and in_bore(KIND[TOP_BLOCK],
                                                                                         st.to_local(p))


def marble_run(objects):
    """The latest run of the marble, tracked every control step: 'idle' (no run yet), 'running' (it went in at the
    top hole and has stayed inside the run since), 'arrived' (that run reached the cup without ever leaving the
    run) or 'left' (it came out of the run). A new run starts whenever it is inside the top hole again. Offline a
    single state cannot show a run, so it is always 'idle'."""
    if TEST_RUN is not None:
        return TEST_RUN
    if _RUN["t"] is None:
        return "idle"
    _episode_memory()
    if _TRACK["step"] == _RUN["step"]:
        return _TRACK["state"]
    _TRACK["step"] = _RUN["step"]
    p = np.asarray(objects["marble"].origin, float)
    old = _TRACK["state"]
    if _at_inlet(objects, p):
        new = "running"
    elif old in ("running", "arrived"):
        new = "arrived" if cup_inside(objects, p) else ("running" if old == "running" and _in_run(objects, p)
                                                      else "left")
    else:
        new = old
    _TRACK["state"] = new
    if DIAG and (new != old or _RUN["step"] % 10 == 0):
        print("MARBLE_DIAG", _RUN["episode"], round(_RUN["t"] or 0.0, 2), "p_mm",
              np.round((p - [0, 0, TOP]) * 1000, 1).tolist(), old, "->", new,
              "joints", joints(objects), flush=True)
    return new


def _flat(st):
    """Sits square: its z axis within TILT_MAX of vertical (either way up; the holes decide which way is right)."""
    t = st.tilt_deg()
    return min(t, 180.0 - t) <= TILT_MAX


def joints(objects):
    """Which of the three hole joints are made: each needs both blocks flat, still and let go, the turn blocks
    standing on the beam, and the start beam still on its supports."""
    beam_st = objects["start_beam"]
    if not (_flat(beam_st) and abs(beam_st.origin[2] - TOP - BEAM_HEIGHT[0]) <= BEAM_HEIGHT[1]):
        return {"beam_turn_down": False, "turn_down_turn_across": False, "turn_across_straight": False}
    ok = {n: _flat(objects[n]) and objects[n].resting() and released(objects[n]) for n in BLOCKS}

    def on_beam(name):  # stands on the beam's top face (centre over the beam, bottom within SEAT_TOL of its top)
        q = beam_st.to_local(objects[name].origin)
        return abs(q[0]) <= H and abs(q[1]) <= BEAM_LEN / 2 and abs(q[2] - MODULE) <= SEAT_TOL
    up = np.array([0.0, 0.0, 1.0])
    cos_face = math.cos(math.radians(FACE_DEG))

    def stacked(lower, lower_hole, upper, upper_hole):
        c0, n0 = lower_hole
        c1, n1 = upper_hole
        return (ok[upper] and (lower == "start_beam" or ok[lower]) and float(n0 @ up) >= cos_face
                and float(n1 @ up) <= -cos_face and math.hypot(*(c1 - c0)[:2]) <= JOINT_TOL
                and abs(float(c1[2] - c0[2])) <= SEAT_TOL)

    out = {}
    out["beam_turn_down"] = stacked("start_beam", hole(beam_st, "start_beam", 0),
                                    "turn_down", hole(objects["turn_down"], "turn_down", 1))
    c0, n0 = hole(objects["turn_down"], "turn_down", 0)
    c1, n1 = hole(objects["turn_across"], "turn_across", 1)
    gap = float((c1 - c0) @ n0)            # face to face, along the turn-down block's side hole
    side = float(np.linalg.norm(c1 - c0 - gap * n0))
    out["turn_down_turn_across"] = bool(ok["turn_down"] and ok["turn_across"] and float(n0 @ n1) <= -cos_face
                                    and -SEAT_TOL <= gap <= GAP_MAX and side <= JOINT_TOL
                                    and on_beam("turn_down") and on_beam("turn_across"))
    if PROBE:
        out["turn_across_straight"] = False
    else:
        st = objects["straight_block"]
        lower = [hole(st, "straight_block", i) for i in (0, 1)]
        low = min(lower, key=lambda h: h[0][2])   # the straight block is the same either way up
        out["turn_across_straight"] = on_beam("turn_across") and stacked(
            "turn_across", hole(objects["turn_across"], "turn_across", 0), "straight_block", low)
    return out


def score(objects):
    """1.0 built and the marble ran through into the cup; 0.75 built; 0.5 two joints; 0.25 one; 0 none."""
    run = marble_run(objects)  # tracked on every call, before any return
    m = objects["marble"]
    p = np.asarray(m.origin, float)
    in_cup = run == "arrived" and cup_inside(objects, p)
    still = held_for("marble_still", in_cup and m.resting(MARBLE_STILL, MARBLE_SPIN) and released(m)) >= STILL_S
    made = sum(joints(objects).values())
    need = 2 if PROBE else 3
    if made == need:
        return 1.0 if still else (0.0 if PROBE else 0.75)
    if PROBE:
        return 0.0
    return {0: 0.0, 1: 0.25, 2: 0.5}[made]


def success(objects):
    return score(objects) == 1.0


# ------------------------------------------------------------------------------------------------ demo
_READ = {"episode": None, "values": {}}
TRAVEL_Z = TOP + 0.165
PICK_OPEN, BLOCK_PINCH, BLOCK_LET_GO = -0.030, -0.0185, -0.027   # pads 60 mm, 37 mm (3 mm inside 40), 54 mm
BALL_OPEN, BALL_PINCH, BALL_LET_GO = -0.014, -0.006, -0.014      # pads 28 mm, 12 mm (3 mm inside 15), 28 mm
GRIP_DZ = 0.010        # TCP 10 mm above a block's centre: the pads cover its upper 23 mm
BALL_GRIP_DZ = 0.009   # TCP 9 mm above the marble's centre (fingertips 3.5 mm over the table)
FEED_ABOVE = 0.040     # marble centre 40 mm over the top hole before it goes down
FEED_AT = MARBLE_R + 0.0015  # let go with the marble 1.5 mm over the top face
PARK = {"left": (0.120, 0.300, TOP + 0.150), "right": (0.140, -0.290, TOP + 0.150)}


def _state(name):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        return arena_scene.object_states(env, [name])[name]
    except Exception as e:  # noqa: BLE001
        print(f"MARBLE_READ {name} failed: {e!r}", flush=True)
        return None


def _read(key, fn):
    if _READ["episode"] != _RUN["episode"]:
        _READ.update(episode=_RUN["episode"], values={})
    if key not in _READ["values"]:
        _READ["values"][key] = np.asarray(fn(), float)
        if DIAG:
            print("MARBLE_PLAN", key, np.round((_READ["values"][key] - [0, 0, TOP]) * 1000, 1).tolist(), flush=True)
    return _READ["values"][key]


def _nominal(name):
    """Start pose (origin, yaw) of a loose object, or its planned goal pose when the simulator is not running."""
    for spec in OBJECTS:
        if spec.name == name:
            return np.asarray(spec.pos, float), float(spec.yaw)
    raise KeyError(name)


class _Pose:
    def __init__(self, origin, yaw):
        self.origin, self.yaw = np.asarray(origin, float), float(yaw)

    def rotation(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def to_world(self, p):
        return self.origin + self.rotation() @ np.asarray(p, float)


def _pose(name, planned=None):
    st = _state(name)
    if st is not None:
        return st
    if planned is not None:
        return planned
    return _Pose(*_nominal(name))


def goal_pose(name, planned=None, live=True):
    """Where block `name` goes, from where the block it meets actually lies (live), or from the planned poses."""
    planned = planned or {}
    pose = (lambda n: _pose(n, planned.get(n))) if live else (lambda n: planned.get(n) or _Pose(*_nominal(n)))
    if name == "turn_down":
        b = pose("start_beam")
        return _Pose(b.to_world((0.0, BEAM_HOLE, MODULE)), math.pi if DEMO_WRONG else 0.0)
    if name == "turn_across":
        if DEMO_WRONG:  # set where the photo shows it, from the beam, not from the back-to-front block
            b = pose("start_beam")
            return _Pose(b.to_world((0.0, BEAM_HOLE + MODULE + GAP, MODULE)), 0.0)
        d = pose("turn_down")
        return _Pose(d.to_world((0.0, MODULE + GAP, 0.0)), 0.0)
    a = pose("turn_across")
    return _Pose(a.to_world((0.0, 0.0, MODULE)), 0.0)


def _planned_goals():
    planned = {}
    for name in BLOCKS:
        planned[name] = goal_pose(name, planned, live=False)
    return planned


class LiveWaypoint(Waypoint):
    """A waypoint whose target is worked out from the simulator state when it starts (planned values offline)."""

    def __init__(self, duration, fn, finger, label, key, yaw=0.0):
        self.duration, self.fn, self.finger, self.label, self.key, self.yaw = duration, fn, finger, label, key, yaw

    @property
    def pos(self):
        return tuple(float(v) for v in _read(self.key, self.fn))


def _origin(name):
    st = _state(name)
    return np.asarray(st.origin, float) if st is not None else _nominal(name)[0]


def _hand_left():
    hands = grippers()
    return None if hands is None else np.asarray(hands["left"][0], float)


def demo():
    wp = []
    planned = _planned_goals()

    def live(t, fn, finger, label, key=None, yaw=0.0):
        wp.append(LiveWaypoint(t, fn, finger, label, key or f"{len(wp)}:{label}", yaw))

    for name in BLOCKS:
        pick_yaw = 0.0  # every block starts with two solid side faces across world y
        if name == "straight_block":
            place_yaw = 0.0
        elif DEMO_WRONG and name == "turn_down":
            place_yaw = math.pi / 2       # the block turns +90 degrees: back to front
        else:
            place_yaw = -math.pi / 2      # the block turns -90 degrees: side hole towards its goal direction
        above = 0.045 if name == "turn_across" else 0.040
        src = lambda name=name: _read(name + " source", lambda: _origin(name))  # noqa: E731
        dst = lambda name=name: _read(name + " goal", lambda: (  # noqa: E731
            goal_pose(name) if _RUN["env"] is not None else planned[name]).origin)

        def hover(name=name, dst=dst, above=above):
            # grasp offset (hand minus block, sag included) read after the carry
            hand = _hand_left()
            off = hand - _origin(name) if hand is not None else np.array([0.0, 0.0, GRIP_DZ])
            return dst() + off + [0.0, 0.0, above]

        def lower(name=name, dst=dst, above=above):
            # the block's own error at the hover is taken off the hand target, then it goes down 2 mm short
            cmd = _read(name + " hover", hover)
            if _hand_left() is None:
                return cmd + [0.0, 0.0, 0.002 - above]
            err = dst() + [0.0, 0.0, above] - _origin(name)
            return cmd + err + [0.0, 0.0, 0.002 - above]

        live(2.0, lambda src=src: np.r_[src()[:2], TRAVEL_Z], PICK_OPEN, f"over the {name}", yaw=pick_yaw)
        live(2.0, lambda src=src: src() + [0, 0, GRIP_DZ], PICK_OPEN, f"fingers down at the {name}", yaw=pick_yaw)
        live(0.8, lambda src=src: src() + [0, 0, GRIP_DZ], BLOCK_PINCH, f"pinch the {name}", yaw=pick_yaw)
        live(1.5, lambda src=src: np.r_[src()[:2], TRAVEL_Z], BLOCK_PINCH, f"lift the {name}", yaw=pick_yaw)
        live(2.5, lambda dst=dst, above=above: np.r_[dst()[:2], max(TRAVEL_Z, dst()[2] + GRIP_DZ + above)],
             BLOCK_PINCH, f"carry the {name} over its place", yaw=place_yaw)
        live(1.5, hover, BLOCK_PINCH, f"hold the {name} over its place", key=name + " hover", yaw=place_yaw)
        live(1.8, lower, BLOCK_PINCH, f"lower the {name} onto its place", key=name + " lower", yaw=place_yaw)
        live(0.6, lower, BLOCK_LET_GO, f"let go of the {name}", key=name + " lower", yaw=place_yaw)
        live(1.2, lambda name=name, lower=lower: _read(name + " lower", lower) + [0, 0, 0.050], BLOCK_LET_GO,
             f"lift the fingers off the {name}", yaw=place_yaw)

    ball = lambda: _read("marble source", lambda: _origin("marble"))  # noqa: E731
    live(2.0, lambda: np.r_[ball()[:2], TRAVEL_Z], BALL_OPEN, "over the marble")
    live(2.0, lambda: ball() + [0, 0, BALL_GRIP_DZ], BALL_OPEN, "fingers down at the marble")
    live(0.8, lambda: ball() + [0, 0, BALL_GRIP_DZ], BALL_PINCH, "pinch the marble")
    live(1.5, lambda: np.r_[ball()[:2], TRAVEL_Z + 0.020], BALL_PINCH, "lift the marble")

    def inlet():
        c, _ = top_hole(_pose(TOP_BLOCK) if _RUN["env"] is not None else planned[TOP_BLOCK], TOP_BLOCK)
        return np.asarray(c, float)

    def feed_hover():
        hand = _hand_left()
        off = hand - _origin("marble") if hand is not None else np.array([0.0, 0.0, BALL_GRIP_DZ])
        return _read("inlet", inlet) + off + [0.0, 0.0, FEED_ABOVE]

    def feed():
        cmd = _read("feed hover", feed_hover)
        if _hand_left() is None:
            return cmd + [0.0, 0.0, FEED_AT - FEED_ABOVE]
        err = _read("inlet", inlet) + [0.0, 0.0, FEED_ABOVE] - _origin("marble")
        return cmd + err + [0.0, 0.0, FEED_AT - FEED_ABOVE]

    live(2.5, lambda: _read("inlet", inlet) + [0, 0, FEED_ABOVE + BALL_GRIP_DZ], BALL_PINCH,
         "carry the marble over the top hole")
    live(1.5, feed_hover, BALL_PINCH, "hold it over the top hole", key="feed hover")
    live(1.5, feed, BALL_PINCH, "lower it to the top hole", key="feed")
    live(0.6, feed, BALL_LET_GO, "let go of the marble", key="feed")
    live(1.0, lambda: _read("feed", feed) + [0, 0, 0.025], BALL_LET_GO, "lift the fingers clear")
    live(2.0, lambda: np.array(PARK["left"]), BALL_LET_GO, "park")
    live(6.0, lambda: np.array(PARK["left"]), BALL_LET_GO, "watch the marble run")
    total = sum(w.duration for w in wp)
    return {"left": wp, "right": [Waypoint(2.0, PARK["right"], FINGER_OPEN, "park"),
                                  Waypoint(total - 2.0, PARK["right"], FINGER_OPEN, "wait")]}
