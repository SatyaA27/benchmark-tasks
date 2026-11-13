"""A stock clerk's count of a heap of loose parts with a desk tally counter.

On the robot's left: a blue polypropylene parts bin holding eleven loose M16 steel hex nuts (zinc plated) in one
overlapping heap, lying flat in three layers (six on the floor, four across pairs of those, one on top), and nearer
the robot an empty white polypropylene tray. On the right: a chrome desk tally counter bolted to the bench, its window
facing the robot. It still shows the last count, 0038. The counter has a push key on top (one full stroke adds one)
and a knurled reset knob on its right side (a twist of about 70 degrees to its stop returns the units and tens wheels
to 0). The nuts are lifted out of the heap one at a time, each with one click, so the count is right when all eleven
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
RUN_LIMIT_STEPS = 3600

# Desk tally counter, bolted to the bench, window facing the robot (-x).
COUNTER_XY = (0.240, -0.120)
COUNTER = tally_counter.make_counter((*COUNTER_XY, TOP), 0.0)
START_VALUE = tally_counter.START_VALUE  # 38: the wheels still show the last count, 0038

# DIN 934 M16 hex nut, zinc-plated steel: 24 mm across flats, 27.7 mm across corners (local +-x at yaw 0).
NUT_COUNT = 11
NUT_FLATS, NUT_THICK, NUT_BORE = 0.024, 0.013, 0.016
NUT_CORNERS = NUT_FLATS / math.cos(math.pi / 6)
NUT_MASS = 0.0333
NUT_COLOR = (0.63, 0.65, 0.68)
NUT_GAP = 0.0003  # start gap under each nut (above the bin floor or the nuts it rests on)

# Bin and tray: polypropylene trays, 3 mm walls and floor, 163 mm along x by 171 mm along y outside.
TRAY_SIZE = (0.163, 0.171)
WALL = 0.003
BIN_XY, BIN_HEIGHT, BIN_COLOR = (0.262, 0.225), 0.033, (0.12, 0.33, 0.78)  # open upper fingers pass 5 mm over the walls
TRAY_XY, TRAY_HEIGHT, TRAY_COLOR = (0.098, 0.115), 0.025, (0.93, 0.93, 0.90)
FLOOR_TOP = TOP + WALL
PITCH_X, PITCH_Y = 0.036, 0.048  # tray slots: 4 along x, 3 along y


def grid(center):
    """The 12 tray slot centres (x, y), row by row from the robot's left, columns from the robot outward."""
    return [(center[0] + (i - 1.5) * PITCH_X, center[1] + (1 - j) * PITCH_Y) for j in range(3) for i in range(4)]


# The heap, as tipped out of a supplier's bag and set from this table: (x mm, y mm, layer, yaw deg). Every nut lies
# flat. Layer 0 is on the bin floor; each layer-1 nut lies across two layer-0 nuts; the layer-2 nut lies across two
# layer-1 nuts. Yaw 0 puts two corners on +-x (flats facing +-y).
HEAP = (
    (218, 197, 0, 4), (256, 195, 0, 52), (294, 199, 0, 10), (220, 249, 0, -66), (258, 250, 0, 7), (295, 247, 0, 57),
    (237, 196, 1, 57), (275, 198, 1, -52), (239, 249, 1, 6), (276, 248, 1, -9),
    (256, 198, 2, 52),
)
TRAY_SLOTS = grid(TRAY_XY)[:NUT_COUNT]  # the k-th nut lifted goes to tray slot k; the last tray slot stays empty
NUT_NAMES = tuple(f"nut_{k:02d}" for k in range(1, NUT_COUNT + 1))


def heap_pose(entry):
    x, y, layer, yaw = entry
    z = FLOOR_TOP + layer * NUT_THICK + NUT_THICK / 2 + (layer + 1) * NUT_GAP
    return (x / 1000, y / 1000, z), math.radians(yaw)


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
NUTS = [MeshPart(name, heap_pose(entry)[0], mass=NUT_MASS, color=NUT_COLOR, grip=0, yaw=heap_pose(entry)[1],
                 meshes=(_NUT_MESH,)) for name, entry in zip(NUT_NAMES, HEAP)]
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
TEST_HISTORY = None  # [(reading, M) or (reading, M, nuts lifted), ...] seen before this state (offline states only)
TEST_LATCHED = None  # the latched reading when the wheels show no valid reading


def _inside(xy, center, half):
    return bool(abs(xy[0] - center[0]) <= half[0] and abs(xy[1] - center[1]) <= half[1])


INNER_HALF = (TRAY_SIZE[0] / 2 - WALL, TRAY_SIZE[1] / 2 - WALL)
OUTER_HALF = (TRAY_SIZE[0] / 2, TRAY_SIZE[1] / 2)


STACK_CAP = 0.050  # in a container: a nut centre up to 50 mm above the table (the heap's top nut is at 35.5 mm)


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
    return in_container(state, BIN_XY, STACK_CAP)


def lifted(state):
    """Out of the heap and not in the tray: in neither container (in the gripper, or anywhere else)."""
    return not (in_bin(state) or in_tray(state))


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


STEP_TOL = 2  # out of step: the latched reading and M differ by this much or more after the first steady 0000


def _history_flags(rows):
    """From earlier (reading, M[, lifted]) rows: whether a steady 0000 came, out of step after it, two lifted."""
    zeroed = out = two = False
    for row in rows:
        reading, moved = row[0], row[1]
        zeroed = zeroed or reading == 0
        out = out or (zeroed and abs(reading - moved) >= STEP_TOL)
        two = two or (len(row) > 2 and row[2] >= 2)
    return zeroed, out, two


def _evaluate(objects):
    counter = objects["counter"]
    nuts = [objects[n] for n in NUT_NAMES]
    value, info = tally_counter.reading(counter)
    latched = latched_reading(value)
    # Remembered conditions, evaluated on every call before anything else decides the score: a steady 0000 at some
    # moment (the count starts from zero), a dropped nut, two nuts lifted out of the heap at once, and the count out
    # of step (after the first steady 0000, the latched reading and M, the nuts out of the bin, differ by 2 or more:
    # one click goes with each nut, before or after it is moved). Offline, a single layout without a history is
    # taken as a count that started from 0000 and stayed in step.
    h_zeroed, h_out, h_two = _history_flags(TEST_HISTORY or ())
    offline_zeroed = _RUN["t"] is None and TEST_HISTORY is None
    zeroed = ever("zeroed", h_zeroed or latched == 0 or offline_zeroed)
    moved = sum(not in_bin(s) for s in nuts)  # M
    out_of_step = ever("out_of_step", h_out or (zeroed and abs(latched - moved) >= STEP_TOL))
    two_lifted = ever("two_lifted", h_two or sum(lifted(s) for s in nuts) >= 2)
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
    if fell or not zeroed or out_of_step or two_lifted:
        return 0.0  # a sticky fault, or a count that never started from a steady 0000
    if n_tray == NUT_COUNT:
        if latched == NUT_COUNT:
            return 1.0 if (value == NUT_COUNT and at_rest and hands_off) else 0.8
        return 0.25 if abs(latched - NUT_COUNT) == 1 else 0.0
    if 1 <= n_tray < NUT_COUNT and latched == n_tray:
        return round(0.4 * n_tray / NUT_COUNT, 3)  # an accurate but unfinished count, to 3 decimals
    return 0.0


_LIVE = {"episode": None, "step": None, "objects": None}  # the latest object states the scorer saw (for the demo)


def score(objects):
    if _RUN["t"] is not None:
        _LIVE.update(episode=_RUN["episode"], step=_RUN["step"], objects=objects)
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
# Demo: timed waypoints. The left arm plans each pick from the nut poses the scorer last saw (the review rules allow
# a scripted demo to read simulator state): when a pick's first move starts it takes the highest uncovered nut with
# the most room round the pads, and a grip across a pair of its flats. Offline, the start layout stands in.
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
TIP_ABOVE_FLOOR = 0.002  # pad tips 2 mm above the lowest point of a nut on the bin floor
TIP_ABOVE_NUT = 0.003  # and 3 mm above the lowest point of a nut lying on others (clear of the nuts under it)
PLACE_GAP = 0.0015  # carried nut bottom 1.5 mm above the tray floor at release
TRAVEL_Z = TOP + 0.080  # carried nut bottom about 60 mm above the table: over the bin walls and the heap's top
HEAP_HOVER = (0.256, 0.223)  # first approach: over the middle of the heap
PAD_HALF_T = 0.005  # fingertip thickness across the closing direction: 10 mm (the offline gripper model)
FINGER_HALF_W = 0.014  # tapered fingertip half-width along the pads, tip to 5 mm behind the TCP
FINGER_REACH = 0.030  # height above the pad tips that the fingertips sweep while gripping
YAW_LIMIT = math.radians(60.0)  # grip yaws the left arm turns to over the bin
ROOM_ENOUGH = 0.006  # beyond this much room round the open fingertips, the smaller grip yaw wins
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
PRESS_S = 0.8  # each press, hover to bottom, at any pace: the pads meet the cap at about 45 mm/s (see README)
SPRING_BACK = 0.5  # the opened knob returns to its stop before nut 1 enters the tray
TAIL = 2.5  # stillness after both arms are clear: 1 s windows plus the 1 s success hold, with margin
RIGHT_PARK = (0.160, -0.240, TOP + 0.120)
LEFT_PARK = (BIN_XY[0], BIN_XY[1], TOP + 0.120)


class LiveWaypoint(Waypoint):
    """A waypoint whose position and yaw are callables, resolved the first time the plan asks (when the move starts)."""

    def __init__(self, duration, pos, finger, label="", yaw=0.0):
        self.duration, self.finger, self.label = duration, finger, label
        self._pos, self._yaw = pos, yaw

    @property
    def pos(self):
        return tuple(float(v) for v in (self._pos() if callable(self._pos) else self._pos))

    @property
    def yaw(self):
        return self._yaw() if callable(self._yaw) else self._yaw


def _poly_hex(xy, yaw):
    a = yaw + np.arange(6) * math.pi / 3
    return np.column_stack((xy[0] + NUT_CORNERS / 2 * np.cos(a), xy[1] + NUT_CORNERS / 2 * np.sin(a)))


def _poly_rect(center, u, v, half_u, half_v):
    c = np.asarray(center, float)
    return np.array([c + su * half_u * u + sv * half_v * v for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def _poly_gap(P, Q):
    """Distance between two convex polygons (2D); negative (the least penetration) when they overlap."""
    sep = -math.inf
    for A in (P, Q):
        for i in range(len(A)):
            e = A[(i + 1) % len(A)] - A[i]
            n = np.array((-e[1], e[0])) / np.linalg.norm(e)
            p, q = P @ n, Q @ n
            sep = max(sep, float(q.min() - p.max()), float(p.min() - q.max()))
    if sep <= 0:
        return sep
    best = math.inf
    for A, B in ((P, Q), (Q, P)):
        for pt in A:
            for i in range(len(B)):
                a, b = B[i], B[(i + 1) % len(B)]
                t = float(np.clip(np.dot(pt - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0))
                best = min(best, float(np.linalg.norm(pt - (a + t * (b - a)))))
    return best


def _nut_pose(state):
    """(centre, yaw of the local x axis, lowest point z, highest point z) of a nut state."""
    c = np.asarray(state.origin, float)
    R = state.rotation()
    return c, math.atan2(R[1, 0], R[0, 0]), float(np.asarray(state.low)[2]), float(np.asarray(state.high)[2])


def plan_pick(poses, room=None):
    """Choose the next nut and grip from {name: (centre, yaw, low, high)} of the nuts still in the bin.

    Uncovered nuts only (no other nut above and over them). Grip yaws: across one of the three pairs of flats, turned
    at most 60 degrees. Room: the least gap from the two open fingertips to the other nuts they could touch and to
    the bin's inner walls. The highest layer goes first, then the most room. Returns (name, yaw, tip z, room)."""
    best = None
    for name, (c, yaw, low, high) in poses.items():
        hexa = _poly_hex(c, yaw)
        if any(o[0][2] > c[2] + 0.005 and _poly_gap(hexa, _poly_hex(o[0], o[1])) < 0
               for other, o in poses.items() if other != name):
            continue  # covered
        on_floor = low < FLOOR_TOP + 0.004
        tip = low + (TIP_ABOVE_FLOOR if on_floor else TIP_ABOVE_NUT)
        for k in range(6):
            grip = (yaw + k * math.pi / 3 + math.pi / 2) % math.pi - math.pi / 2
            if abs(grip) > YAW_LIMIT + 1e-9:
                continue
            u = np.array((-math.sin(grip), math.cos(grip)))  # the pads close along u
            v = np.array((math.cos(grip), math.sin(grip)))
            pads = [_poly_rect(c[:2] + side * (-NUT_OPEN + PAD_HALF_T) * u, u, v, PAD_HALF_T, FINGER_HALF_W)
                    for side in (-1, 1)]
            gap = math.inf
            for other, (oc, oyaw, olow, ohigh) in poses.items():
                if other != name and ohigh > tip + 0.0005 and olow < tip + FINGER_REACH:
                    gap = min(gap, *(_poly_gap(p, _poly_hex(oc, oyaw)) for p in pads))
            for p in pads:
                gap = min(gap, float(np.min(INNER_HALF[0] - np.abs(p[:, 0] - BIN_XY[0]))),
                          float(np.min(INNER_HALF[1] - np.abs(p[:, 1] - BIN_XY[1]))))
            key = (round(c[2] / 0.006), min(gap, ROOM_ENOUGH), -abs(grip), gap)
            if best is None or key > best[0]:
                best = (key, name, grip, tip, gap)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


class HeapPicks:
    """Pick k: planned once, when its first move starts, from the scorer's latest nut poses (offline: the start
    layout, nuts already picked removed). Positions are TCP targets."""

    def __init__(self):
        self.plans, self.taken = {}, []

    def _poses(self):
        objects = _LIVE["objects"] if _LIVE["episode"] == _RUN["episode"] and _RUN["t"] is not None else None
        if objects is not None:
            return {n: _nut_pose(objects[n]) for n in NUT_NAMES if in_bin(objects[n])}
        poses = {}
        for spec in NUTS:
            if spec.name not in self.taken:
                c = np.asarray(spec.pos, float)
                poses[spec.name] = (c, spec.yaw, c[2] - NUT_THICK / 2, c[2] + NUT_THICK / 2)
        return poses

    def plan(self, k):
        for j in range(1, k + 1):
            if j not in self.plans:
                poses = self._poses()
                chosen = plan_pick(poses)
                if chosen is None:  # nothing left: a pick in place over the heap's middle
                    self.plans[j] = {"name": None, "xy": HEAP_HOVER, "yaw": 0.0, "tcp_low": FLOOR_TOP + 0.015}
                    continue
                name, yaw, tip, room = chosen
                self.taken.append(name)
                self.plans[j] = {"name": name, "xy": tuple(poses[name][0][:2]), "yaw": yaw, "tcp_low": tip + PAD_TIP,
                                 "grip_below": tip + PAD_TIP - poses[name][2], "room": room}
        return self.plans[k]

    def slot(self, k):
        return _Slot(self, k)


class _Slot:
    def __init__(self, picks, k):
        self.picks, self.k = picks, k

    def yaw(self):
        return lambda: self.picks.plan(self.k)["yaw"]

    def at(self, where, dst=None):
        def resolve():
            p = self.picks.plan(self.k)
            if where == "hover":
                return (*p["xy"], TRAVEL_Z)
            if where == "pick":
                return (*p["xy"], p["tcp_low"])
            if where == "carry":
                return (*dst, TRAVEL_Z)
            # place: the carried nut's bottom PLACE_GAP above the tray floor; it rides RIDE_LOW low in the pads
            return (*dst, FLOOR_TOP + PLACE_GAP + p.get("grip_below", 0.015) + RIDE_LOW)
        return resolve


def demo():
    plans, clock = {"left": [], "right": []}, {"left": 0.0, "right": 0.0}
    marks = {"release": []}

    def add(arm, label, dt, pos, finger, yaw=0.0, keep=False):
        dt = dt if keep else dt * DEMO_PACE
        if callable(pos) or callable(yaw):
            plans[arm].append(LiveWaypoint(dt, pos, finger, label, yaw))
        else:
            plans[arm].append(Waypoint(dt, tuple(float(v) for v in pos), finger, label, yaw))
        clock[arm] += dt

    def hold_until(arm, t, label):
        last = plans[arm][-1]
        if t - clock[arm] > 1e-9:
            # A live waypoint is held through its callables: reading its position here would plan that pick when
            # demo() is built (before the episode's first scored step), not when its first move starts.
            pos, yaw = (last._pos, last._yaw) if isinstance(last, LiveWaypoint) else (last.pos, last.yaw)
            add(arm, label, t - clock[arm], pos, last.finger, yaw, keep=True)

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

    # Left arm: the k-th nut lifted out of the heap goes to tray slot k; nut 1 waits above the tray until the knob is
    # back. Each pick is planned when its first move starts, from the nut poses the scorer last saw.
    picks = HeapPicks()
    for k, dst in enumerate(TRAY_SLOTS, start=1):
        tag, pick = f"nut {k:02d}", picks.slot(k)
        if k == 1:
            add("left", tag + " to heap", 1.6, (*HEAP_HOVER, TRAVEL_Z), NUT_OPEN)
            add("left", tag + " over", 0.6, pick.at("hover"), NUT_OPEN, pick.yaw())
        else:
            add("left", tag + " to bin", 1.0, pick.at("hover"), NUT_OPEN, pick.yaw())
        add("left", tag + " descend", 0.8, pick.at("pick"), NUT_OPEN, pick.yaw())
        add("left", tag + " close", 0.5, pick.at("pick"), FINGER_CLOSED, pick.yaw())
        add("left", tag + " lift", 0.7, pick.at("hover"), FINGER_CLOSED, pick.yaw())
        # Turned back to yaw 0 on the way, so the nut lands flats to the pads in its tray slot.
        add("left", tag + " carry", 1.0, pick.at("carry", dst), FINGER_CLOSED, 0.0)
        if k == 1:
            hold_until("left", knob_clear, tag + " wait for zero")
        add("left", tag + " lower", 0.6, pick.at("place", dst), FINGER_CLOSED, 0.0)
        add("left", tag + " release", 0.4, pick.at("place", dst), NUT_OPEN, 0.0, keep=True)
        marks["release"].append(clock["left"])
        add("left", tag + " rise", 0.5, pick.at("carry", dst), NUT_OPEN, 0.0)
    add("left", "park", 1.0, LEFT_PARK, NUT_OPEN)

    # Right arm, part 2: one full key stroke per nut, each just after that nut is let go in the tray.
    for released_at in marks["release"]:
        hold_until("right", released_at + PRESS_DELAY - PRESS_S, "click wait")
        add("right", "click press", PRESS_S, (*KEY_XY, PRESS_Z), FINGER_CLOSED, pinch(0.0), keep=True)
        add("right", "click hold", 0.3, (*KEY_XY, PRESS_Z), FINGER_CLOSED, pinch(0.0), keep=True)
        add("right", "click rise", 0.5, (*KEY_XY, KEY_HOVER_Z), FINGER_CLOSED, pinch(0.0))
    add("right", "park", 1.0, RIGHT_PARK, FINGER_CLOSED, pinch(0.0))

    end = max(clock.values()) + TAIL
    for arm in ("left", "right"):
        hold_until(arm, end, "stillness tail")
    return plans
