"""A stock clerk's cycle count with a desk tally counter.

On the robot's left: a blue polypropylene parts bin holding eleven loose M16 steel hex nuts (zinc plated) in one
layer on a 4 x 3 grid with one slot empty, and nearer the robot an empty white polypropylene tray. On the right: a
chrome desk tally counter bolted to the bench, its window facing the robot. It still shows the last count, 0038.
The counter has a push key on top (one full stroke adds one) and a knurled reset knob on its right side (a twist
of about 70 degrees to its stop returns the units and tens wheels to 0). The count is right when all eleven nuts
are in the tray and the wheels read 0011, counted from a 0000 the reset showed. Metres; zero jitter.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import Part, box  # noqa: E402
from yam_mesh import Mesh, MeshPart, _bored_extrusion  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

# The counter module binds the kit's classes when it is imported. A loader that reloads the kit (offline checkers
# do) needs the counter rebuilt on the same classes, so drop a cached copy bound to an earlier load.
_cached = sys.modules.get("tally_counter")
if _cached is not None and (getattr(_cached, "Mesh", None) is not Mesh or getattr(_cached, "box", None) is not box):
    del sys.modules["tally_counter"]
import tally_counter  # noqa: E402

SCENE = "station"
# Release-and-stillness block, verbatim from hang_holder_on_keyhole_studs.py (see release-stillness.md).
# The task module must define: import math; import numpy as np; from yam_arena.constants import FINGER_OPEN;
# and SCENE ("station" or "tabletop"). Paste after the imports.

# Release and stillness
# Released means every gripper TCP is at least RELEASE_CLEAR from the object's box.
# Still means below the task's linear and angular speed limits for STILL_S seconds.
# Evaluate every sticky condition and timer at the top of score() on every call,
# before any early return, so interrupted conditions cannot retain elapsed time.
RELEASE_CLEAR = 0.030
STILL_S = 1.0
TEST_GRIPPERS = None


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
_EVER = {}


def grippers():
    if TEST_GRIPPERS is not None:
        return TEST_GRIPPERS
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, LEFT_BASE_POS, RIGHT_BASE_POS

    bases = station.BASES if SCENE == "station" else {"left": LEFT_BASE_POS, "right": RIGHT_BASE_POS}
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
        tcp = ik.tcp_pose(q[arm_ids], bases[arm])[0]
        opening = float(np.clip(np.mean(q[finger_ids]) / FINGER_OPEN, 0.0, 1.0))
        out[arm] = (tcp, opening)
    return out


def released(state, clear=RELEASE_CLEAR):
    hands = grippers()
    if hands is None:
        return True  # Offline arm poses are unknown, so release is assumed.
    low, high = np.asarray(state.low), np.asarray(state.high)
    return all(np.linalg.norm(np.maximum(np.maximum(low - tcp, tcp - high), 0.0)) >= clear
               for tcp, _ in hands.values())


def _episode_memory():
    global _MEMORY_EPISODE
    if _MEMORY_EPISODE != _RUN["episode"]:
        _TIMERS.clear()
        _EVER.clear()
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


def ever(key, cond):
    """Remember a satisfied condition until the next episode; offline each call stands alone."""
    if _RUN["t"] is None: return bool(cond)
    _episode_memory()
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]


# ---------------------------------------------------------------------------------------------------------------
# Scene
DEMO_PACE = 1.0
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
RUN_LIMIT_STEPS = 2400

# Desk tally counter, bolted to the bench, window facing the robot (-x).
COUNTER_XY = (0.240, -0.120)
COUNTER = tally_counter.make_counter((*COUNTER_XY, TOP), 0.0)
START_VALUE = tally_counter.START_VALUE  # 38: the wheels still show the last count, 0038

# DIN 934 M16 hex nut, zinc-plated steel: 24 mm across flats (faces +-y), 27.7 mm across corners (+-x).
NUT_COUNT = 11
NUT_FLATS, NUT_THICK, NUT_BORE = 0.024, 0.013, 0.016
NUT_CORNERS = NUT_FLATS / math.cos(math.pi / 6)
NUT_MASS = 0.0333
NUT_COLOR = (0.63, 0.65, 0.68)
NUT_GAP = 0.0003  # start height above the bin floor

# Bin and tray: polypropylene trays, 3 mm walls and floor, 163 mm along x by 171 mm along y outside.
TRAY_SIZE = (0.163, 0.171)
WALL = 0.003
BIN_XY, BIN_HEIGHT, BIN_COLOR = (0.262, 0.225), 0.033, (0.12, 0.33, 0.78)  # open upper fingers pass 5 mm over the walls
TRAY_XY, TRAY_HEIGHT, TRAY_COLOR = (0.098, 0.115), 0.025, (0.93, 0.93, 0.90)
FLOOR_TOP = TOP + WALL
PITCH_X, PITCH_Y = 0.036, 0.048  # 4 slots along x, 3 along y; the 36 mm wide open pads clear each neighbour by 4 mm
EMPTY_SLOT = (2, 1)  # (column, row) of the bin slot that starts empty


def grid(center):
    """The 12 slot centres (x, y), row by row from the robot's left, columns from the robot outward."""
    return [(center[0] + (i - 1.5) * PITCH_X, center[1] + (1 - j) * PITCH_Y) for j in range(3) for i in range(4)]


BIN_SLOTS = [xy for k, xy in enumerate(grid(BIN_XY)) if k != EMPTY_SLOT[1] * 4 + EMPTY_SLOT[0]]
TRAY_SLOTS = grid(TRAY_XY)[:NUT_COUNT]  # nut k goes to tray slot k; the last tray slot stays empty
NUT_NAMES = tuple(f"nut_{k:02d}" for k in range(1, NUT_COUNT + 1))


def tray_part(name, center, height, color):
    length, width = TRAY_SIZE
    shapes = [box((0, 0, WALL / 2), (length, width, WALL))]
    shapes += [box((s * (length - WALL) / 2, 0, height / 2), (WALL, width, height)) for s in (-1, 1)]
    shapes += [box((0, s * (width - WALL) / 2, height / 2), (length - 2 * WALL, WALL, height)) for s in (-1, 1)]
    return Part(name, (*center, TOP), shapes, color=color)


def nut_mesh():
    angles = np.arange(6) * math.pi / 3  # corners on +-x, flats facing +-y
    outline = (NUT_CORNERS / 2) * np.column_stack((np.cos(angles), np.sin(angles)))
    points, faces = _bored_extrusion(outline, NUT_THICK, NUT_BORE / 2)
    # The convex hull (the solid hex) is the collider; the bore is visual only.
    return Mesh("hex_nut", points, faces, collision="convex", color=NUT_COLOR,
                proxy=(box((0, 0, 0), (NUT_CORNERS, NUT_FLATS, NUT_THICK)),))


_NUT_MESH = nut_mesh()
NUTS = [MeshPart(name, (*xy, FLOOR_TOP + NUT_THICK / 2 + NUT_GAP), mass=NUT_MASS, color=NUT_COLOR, grip=0,
                 meshes=(_NUT_MESH,)) for name, xy in zip(NUT_NAMES, BIN_SLOTS)]
BIN = tray_part("bin", BIN_XY, BIN_HEIGHT, BIN_COLOR)
TRAY = tray_part("tray", TRAY_XY, TRAY_HEIGHT, TRAY_COLOR)
OBJECTS = [COUNTER, BIN, TRAY, *NUTS]

# ---------------------------------------------------------------------------------------------------------------
# Scorer
DROP_HEIGHT = 0.020  # a nut centre below table + 20 mm outside both trays, or below table - 20 mm, is dropped
POS_STILL, ROT_STILL_DEG = 0.001, 2.0  # nut pose spread over the last STILL_S
NUT_SPEED = 0.01  # m/s
WHEEL_REST_DEG, WHEEL_STILL_DEG = 2.0, 0.3
KEY_REST, KEY_STILL = 0.0005, 0.0002
KNOB_REST_TOL_DEG = 2.0
KNOB_REST_DEG = COUNTER._link("knob").joint.lower  # the spring holds the knob on its lower stop
READ_STEADY_S = 0.25  # a reading is taken once both wheels have shown it, aligned, for this long
TEST_HISTORY = None  # [(reading, nuts in tray), ...] seen before this state; only whether a 0000 was among them counts
TEST_LATCHED = None  # the latched reading when the wheels show no valid reading


def _inside(xy, center, half):
    return bool(abs(xy[0] - center[0]) <= half[0] and abs(xy[1] - center[1]) <= half[1])


INNER_HALF = (TRAY_SIZE[0] / 2 - WALL, TRAY_SIZE[1] / 2 - WALL)
OUTER_HALF = (TRAY_SIZE[0] / 2, TRAY_SIZE[1] / 2)


STACK_CAP = 0.050  # tray count: a nut centre up to 50 mm above the table (stacks of four, on edge on a flat nut)


def in_container(state, center, height):
    """Nut centre inside the inner wall rectangle, between the floor top and `height` above the table."""
    c = np.asarray(state.origin, float)
    return _inside(c, center, INNER_HALF) and FLOOR_TOP <= c[2] <= TOP + height


def in_tray(state):
    """Over the tray (held or resting); resting_in_tray() decides which of these are in the tray."""
    return in_container(state, TRAY_XY, STACK_CAP)


SEAT_TOL = 0.003  # N: a nut's lowest point at most 3 mm above the tray floor top, or above a supporting nut's top
PEN_TOL = 0.002  # resting contact may sink 2 mm into the floor


def resting_in_tray(nuts):
    """N: nuts over the tray (in_tray) that rest on the tray floor, or on a counted nut, built up layer by layer.

    On the floor: lowest point within -2/+3 mm of the floor top. On a nut: centre (x, y) over that nut's bounding
    box, lowest point no lower than that nut's mid-height and at most 3 mm above its top. A nut held over the tray
    with nothing under it is not counted.
    """
    over = [s for s in nuts if in_tray(s)]
    low = {id(s): float(s.low[2]) for s in over}
    counted = [s for s in over if FLOOR_TOP - PEN_TOL <= low[id(s)] <= FLOOR_TOP + SEAT_TOL]
    grown = True
    while grown:
        grown = False
        for s in over:
            if any(s is c for c in counted):
                continue
            c_xy = np.asarray(s.origin, float)[:2]
            for b in counted:
                b_lo, b_hi = np.asarray(b.low, float), np.asarray(b.high, float)
                if (np.all(c_xy >= b_lo[:2]) and np.all(c_xy <= b_hi[:2])
                        and (b_lo[2] + b_hi[2]) / 2 <= low[id(s)] <= b_hi[2] + SEAT_TOL):
                    counted.append(s)
                    grown = True
                    break
    return len(counted)


def in_bin(state):
    return in_container(state, BIN_XY, BIN_HEIGHT)


def dropped(state):
    c = np.asarray(state.origin, float)
    outside = not (_inside(c, TRAY_XY, OUTER_HALF) or _inside(c, BIN_XY, OUTER_HALF))
    return bool((outside and c[2] < TOP + DROP_HEIGHT) or c[2] < TOP - DROP_HEIGHT)


_STATE = {"episode": None, "step": None, "latched": START_VALUE, "candidate": None, "since": None,
          "history": {}, "result": None}


def _fresh_episode():
    if _STATE["episode"] != _RUN["episode"]:
        _STATE.update(episode=_RUN["episode"], step=None, latched=START_VALUE if TEST_LATCHED is None else TEST_LATCHED,
                      candidate=None, since=None, history={}, result=None)


def latched_reading(value):
    """The last reading the wheels showed steadily (both aligned on it for READ_STEADY_S); 38 at the start."""
    if _RUN["t"] is None:  # a single offline layout is taken as steady
        if value is not None:
            return value
        return START_VALUE if TEST_LATCHED is None else TEST_LATCHED
    _fresh_episode()
    t = _RUN["t"]
    if value is None:
        _STATE["candidate"] = _STATE["since"] = None
    elif value != _STATE["candidate"]:
        _STATE["candidate"], _STATE["since"] = value, t
    if value is not None and t - _STATE["since"] >= READ_STEADY_S - 1e-9:
        _STATE["latched"] = value
    return _STATE["latched"]


def _rotation_deg(a, b):
    return math.degrees(math.acos(float(np.clip((np.trace(a.T @ b) - 1) / 2, -1.0, 1.0))))


def _windows_still(nuts, counter, wheel_q):
    """Pose and angle windows over the last STILL_S (SDF contacts make reported joint speeds chatter)."""
    if _RUN["t"] is None:
        return True  # a single offline layout cannot show motion over time; speeds are checked separately
    t, step, history = _RUN["t"], _RUN["step"], _STATE["history"]
    sample = {"nuts": [(np.array(s.origin, float), s.rotation()) for s in nuts],
              "wheels": np.degrees([wheel_q["units"], wheel_q["tens"]]), "key": float(counter.joints["key"])}
    samples = history.setdefault("samples", [])
    if samples and (step != samples[-1][0] + 1 or t <= samples[-1][1]):
        samples.clear()  # a missed step or a clock rewind needs a fresh window
    samples.append((step, t, sample))
    while len(samples) > 1 and samples[1][1] <= t - STILL_S + 1e-9:
        samples.pop(0)
    if samples[0][1] > t - STILL_S + 1e-9:
        return False  # less than a full window observed
    now = sample
    for _, _, old in samples:
        for (p, r), (p0, r0) in zip(now["nuts"], old["nuts"]):
            if np.linalg.norm(p - p0) > POS_STILL or _rotation_deg(r, r0) > ROT_STILL_DEG:
                return False
    wheels = np.array([old["wheels"] for _, _, old in samples])
    keys = np.array([old["key"] for _, _, old in samples])
    return bool(np.all(np.ptp(wheels, axis=0) <= WHEEL_STILL_DEG) and np.ptp(keys) <= KEY_STILL)


def _history_zeroed(pairs):
    """Whether a list of earlier (reading, nuts in tray) pairs includes a steady 0000."""
    return any(r == 0 for r, _ in pairs)


def _evaluate(objects):
    counter = objects["counter"]
    nuts = [objects[n] for n in NUT_NAMES]
    value, info = tally_counter.reading(counter)
    latched = latched_reading(value)
    # Remembered conditions, evaluated on every call before anything else decides the score: a steady 0000 at some
    # moment (the count starts from zero) and a dropped nut. Beyond these the end state alone grades the count; how
    # clicks and placements interleaved (either order, several nuts then their clicks, a re-zero and recount) does
    # not matter. Offline, a single layout without a history is taken as a count that started from 0000.
    offline_zeroed = _RUN["t"] is None and TEST_HISTORY is None
    zeroed = ever("zeroed", _history_zeroed(TEST_HISTORY or ()) or latched == 0 or offline_zeroed)
    n_tray = resting_in_tray(nuts)  # the credit tiers count only nuts resting in the tray
    fell = ever("dropped", any(dropped(s) for s in nuts))
    still = _windows_still(nuts, counter, info.get("q_rad", {"units": math.nan, "tens": math.nan}))
    errors = info.get("error_deg", {})
    at_rest = bool(still and info.get("aligned", False)
                   and all(errors.get(n, math.inf) <= WHEEL_REST_DEG for n in ("units", "tens"))
                   and float(counter.joints["key"]) >= -KEY_REST
                   and abs(math.degrees(float(counter.joints["knob"])) - KNOB_REST_DEG) <= KNOB_REST_TOL_DEG
                   and all(np.linalg.norm(s.lin_vel) < NUT_SPEED for s in nuts))
    hands_off = (all(released(s) for s in nuts) and released(counter.link_view("key"))
                 and released(counter.link_view("knob")))
    if fell or not zeroed:
        return 0.0  # no credit for a count that never started from a steady 0000
    if n_tray == NUT_COUNT:
        if latched == NUT_COUNT:
            return 1.0 if (value == NUT_COUNT and at_rest and hands_off) else 0.8
        return 0.25 if abs(latched - NUT_COUNT) == 1 else 0.0
    if 1 <= n_tray < NUT_COUNT and latched == n_tray:
        return round(0.4 * n_tray / NUT_COUNT, 3)  # an accurate but unfinished count, to 3 decimals
    return 0.0


def score(objects):
    if _RUN["t"] is not None:
        _fresh_episode()
        if _STATE["step"] == _RUN["step"] and _STATE["result"] is not None:
            return _STATE["result"]  # success() and score() share one control step
        _STATE["step"] = _RUN["step"]
        _STATE["result"] = _evaluate(objects)
        return _STATE["result"]
    return _evaluate(objects)


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------------------------------------------------------
# Demo: open loop from the known start layout, time and the robot's own state. It reads no object or joint state.
class PitchedYaw(float):
    """A grip yaw (rad) that also carries a pitch (rad) about world +y through the TCP."""

    def __new__(cls, yaw, pitch=0.0):
        value = super().__new__(cls, yaw)
        value.pitch = float(pitch)
        return value

    def __repr__(self):
        return f"PitchedYaw({float(self)!r}, pitch={self.pitch!r})"


def _install_pitch():
    """Let a PitchedYaw tilt the straight-down grip; a plain float yaw is unchanged."""
    from yam_arena import demo as arena_demo
    from yam_arena.urdf import axis_angle_matrix

    original = arena_demo.grip_rotation
    if getattr(original, "_pitched_yaw", False):
        return

    def grip_rotation(yaw=0.0):
        rotation = original(float(yaw))
        pitch = float(getattr(yaw, "pitch", 0.0))
        return rotation if pitch == 0.0 else axis_angle_matrix(np.array([0.0, 1.0, 0.0]), pitch) @ rotation

    grip_rotation._pitched_yaw = True
    grip_rotation.__wrapped__ = original
    arena_demo.grip_rotation = grip_rotation


_install_pitch()

PAD_TIP = 0.013  # the pad tips are 13 mm past the TCP
RIDE_LOW = 0.0035  # a gripped part rides this much low in the pads
NUT_OPEN = -0.020  # pads 40 mm apart around a 24 mm nut
KNOB_OPEN = -0.010  # pads 20 mm apart around the 13 mm knob: the lower open pad stays clear of the base plate
PICK_Z = FLOOR_TOP + 0.015  # pad tips 2 mm above the bin floor
PLACE_Z = FLOOR_TOP + 0.0015 + 0.015 + RIDE_LOW  # nut bottom 1.5 mm above the tray floor
TRAVEL_Z = TOP + 0.080  # carried nut bottom 31 mm above the bin wall tops
KNOB_TCP = np.array([COUNTER_XY[0] + tally_counter.KNOB_TCP[0], COUNTER_XY[1] + tally_counter.KNOB_TCP[1],
                     TOP + tally_counter.KNOB_TCP[2]])
KNOB_PITCH = (math.radians(-37.5), math.radians(37.5))  # 75 degrees: the knob's 70 degree stop plus 5 degrees of slip
KNOB_LET_GO = math.radians(-30.0)  # turned back by hand to about 5 degrees before letting go
KNOB_ENTRY = 0.045  # the pinch slides onto the knob along the pitched approach axis
KNOB_BACKOFF = 0.040  # opened pads slide off the knob end along -y
KEY_XY = np.array(COUNTER_XY)
GRIP_YAW = math.pi / 2  # pads close along x for the knob pinch and the key press
PAD_BLOCK_HALF = (0.010, 0.018)  # closed pad block at yaw 90: half sizes along x and y


def _case_top_under_pads():
    """Highest counter collider (key and pawl excluded) under the closed pad block, from the counter's geometry."""
    labels = COUNTER.shape_labels()
    shapes = COUNTER.world_shapes_at(np.array([*COUNTER_XY, TOP]), np.eye(3), COUNTER.initial_q())
    top = TOP
    for label, (kind, center, rot, dims) in zip(labels, shapes):
        if label in ("key", "pawl"):
            continue  # the cap is what the pads press; the pawl rides down with it
        half = (np.asarray(dims) / 2 if kind == "box" else
                np.array((dims[0], dims[0], dims[1] / 2)) if kind == "cyl" else np.full(3, dims[0]))
        extent = np.abs(rot) @ half
        low, high = center - extent, center + extent
        if (low[0] < KEY_XY[0] + PAD_BLOCK_HALF[0] and high[0] > KEY_XY[0] - PAD_BLOCK_HALF[0]
                and low[1] < KEY_XY[1] + PAD_BLOCK_HALF[1] and high[1] > KEY_XY[1] - PAD_BLOCK_HALF[1]):
            top = max(top, float(high[2]))
    return top


CASE_TOP_UNDER_PADS = _case_top_under_pads()
KEY_TOP = TOP + tally_counter.KEY_CAP_TOP
PRESS_TIP_Z = max(KEY_TOP - tally_counter.KEY_STROKE - 0.005, CASE_TOP_UNDER_PADS + 0.0005)
KEY_HOVER_Z = KEY_TOP + 0.015 + PAD_TIP  # closed pad tips 15 mm above the cap
PRESS_Z = PRESS_TIP_Z + PAD_TIP
PRESS_DELAY = 0.6  # bottom of each press after the matching nut is let go
SPRING_BACK = 0.5  # the opened knob returns to its stop before nut 1 enters the tray
TAIL = 2.5  # stillness after both arms are clear: 1 s windows plus the 1 s success hold, with margin
RIGHT_PARK = (0.160, -0.240, TOP + 0.120)
LEFT_PARK = (BIN_XY[0], BIN_XY[1], TOP + 0.120)


def demo():
    plans, clock = {"left": [], "right": []}, {"left": 0.0, "right": 0.0}
    marks = {"release": []}

    def add(arm, label, dt, pos, finger, yaw=0.0, keep=False):
        dt = dt if keep else dt * DEMO_PACE
        plans[arm].append(Waypoint(dt, tuple(float(v) for v in pos), finger, label, yaw))
        clock[arm] += dt

    def hold_until(arm, t, label):
        last = plans[arm][-1]
        if t - clock[arm] > 1e-9:
            add(arm, label, t - clock[arm], last.pos, last.finger, last.yaw, keep=True)

    # Right arm, part 1: zero the counter with one twist of the reset knob. The whole pinch (approach, descent onto
    # the knob with 3.5 mm of pad clearance, grip, twist, return, opening) keeps its nominal timing at any pace.
    pinch = lambda pitch: PitchedYaw(GRIP_YAW, pitch)  # noqa: E731
    approach = np.array([-math.sin(KNOB_PITCH[0]), 0.0, -math.cos(KNOB_PITCH[0])])
    entry = KNOB_TCP - KNOB_ENTRY * approach
    backoff = KNOB_TCP - (0.0, KNOB_BACKOFF, 0.0)
    add("right", "knob approach", 2.0, entry, KNOB_OPEN, pinch(KNOB_PITCH[0]), keep=True)
    add("right", "knob settle", 0.4, entry, KNOB_OPEN, pinch(KNOB_PITCH[0]), keep=True)
    add("right", "knob descend", 1.2, KNOB_TCP, KNOB_OPEN, pinch(KNOB_PITCH[0]), keep=True)
    add("right", "knob still", 0.3, KNOB_TCP, KNOB_OPEN, pinch(KNOB_PITCH[0]), keep=True)
    add("right", "knob close", 0.5, KNOB_TCP, FINGER_CLOSED, pinch(KNOB_PITCH[0]), keep=True)
    add("right", "knob grip", 0.2, KNOB_TCP, FINGER_CLOSED, pinch(KNOB_PITCH[0]), keep=True)
    add("right", "knob twist", 3.0, KNOB_TCP, FINGER_CLOSED, pinch(KNOB_PITCH[1]), keep=True)
    add("right", "knob hold", 0.3, KNOB_TCP, FINGER_CLOSED, pinch(KNOB_PITCH[1]), keep=True)
    add("right", "knob return", 0.8, KNOB_TCP, FINGER_CLOSED, pinch(KNOB_LET_GO), keep=True)
    add("right", "knob open", 0.5, KNOB_TCP, KNOB_OPEN, pinch(KNOB_LET_GO), keep=True)
    knob_clear = clock["right"] + SPRING_BACK
    add("right", "knob back off", 0.8, backoff, KNOB_OPEN, pinch(KNOB_LET_GO))
    add("right", "knob rise", 1.0, (backoff[0], backoff[1], KEY_HOVER_Z), FINGER_CLOSED, pinch(0.0))
    add("right", "key hover", 1.2, (*KEY_XY, KEY_HOVER_Z), FINGER_CLOSED, pinch(0.0))

    # Left arm: nut k from bin slot k to tray slot k; nut 1 waits above the tray until the knob is back.
    for k, (src, dst) in enumerate(zip(BIN_SLOTS, TRAY_SLOTS), start=1):
        tag = f"nut {k:02d}"
        add("left", tag + " to bin", 1.6 if k == 1 else 1.0, (*src, TRAVEL_Z), NUT_OPEN)
        add("left", tag + " descend", 0.8, (*src, PICK_Z), NUT_OPEN)
        add("left", tag + " close", 0.5, (*src, PICK_Z), FINGER_CLOSED)
        add("left", tag + " lift", 0.7, (*src, TRAVEL_Z), FINGER_CLOSED)
        add("left", tag + " carry", 1.0, (*dst, TRAVEL_Z), FINGER_CLOSED)
        if k == 1:
            hold_until("left", knob_clear, tag + " wait for zero")
        add("left", tag + " lower", 0.6, (*dst, PLACE_Z), FINGER_CLOSED)
        add("left", tag + " release", 0.4, (*dst, PLACE_Z), NUT_OPEN, keep=True)
        marks["release"].append(clock["left"])
        add("left", tag + " rise", 0.5, (*dst, TRAVEL_Z), NUT_OPEN)
    add("left", "park", 1.0, LEFT_PARK, NUT_OPEN)

    # Right arm, part 2: one full key stroke per nut, each just after that nut is let go in the tray.
    for released_at in marks["release"]:
        lower = 0.5 * DEMO_PACE
        hold_until("right", released_at + PRESS_DELAY - lower, "click wait")
        add("right", "click press", 0.5, (*KEY_XY, PRESS_Z), FINGER_CLOSED, pinch(0.0))
        add("right", "click hold", 0.3, (*KEY_XY, PRESS_Z), FINGER_CLOSED, pinch(0.0), keep=True)
        add("right", "click rise", 0.5, (*KEY_XY, KEY_HOVER_Z), FINGER_CLOSED, pinch(0.0))
    add("right", "park", 1.0, RIGHT_PARK, FINGER_CLOSED, pinch(0.0))

    end = max(clock.values()) + TAIL
    for arm in ("left", "right"):
        hold_until(arm, end, "stillness tail")
    return plans
