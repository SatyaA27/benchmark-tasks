"""Send a pull-back toy car through a gate: press it down, drag it back to wind it, and let it go.

On a grey self-healing cutting mat (Fiskars, 610 x 914 x 1.5 mm) lies a die-cast pull-back toy car, a KiNSMART 2017 Ford GT at
1:38 (127 mm long, 54 mm wide at the body, 32 mm tall, 109 g, rubber tyres), on the robot's right, its nose turned
20 degrees off the lane toward the far side. The lane runs along y at x = 210 mm, in front of both arms: a strip of
blue painter's tape across the mat is the start line (near edge at y = -200 mm); 320 mm further on, two wooden unit
blocks (35 x 70 x 140 mm, standing on end, taped down) make a gate 120 mm wide centred on the lane at y = +120 mm; a
foam yoga block (229 x 152 x 102 mm, lying flat) is the backstop at the end of the bay, from y = +320 mm. The car's
motor winds only while its rear wheels roll backwards, so the car must be pressed onto the mat while it is dragged
back; let go, it drives itself forward. Module pullback_kit (reusable) builds the car and its motor. Metres; x forward
from the arm bases, y left, z up.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pullback_kit as PK  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_parts import Part, box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
RUN_LIMIT_STEPS = 1200  # 60 s at 20 Hz
SUCCESS_HOLD_S = 0.5
MM = 0.001

# Layout (metres; x forward, y left)
MAT = (0.610, 0.914, 0.0015)  # Fiskars 24 x 36 x 0.06 in
MAT_XY = (0.3825, 0.040)
FLOOR = TOP + MAT[2]  # the mat's top
LANE_X = 0.210
LINE_Y = -0.200  # start line: the tape's near edge (the car's nose must be short of it when let go)
TAPE = 0.024
GATE_Y = 0.120  # gate centre line
GATE_GAP = 0.120  # between the posts' inner faces
POST = (0.070, 0.035, 0.140)  # a unit block on end: 70 across the lane, 35 along it, 140 tall
BLOCK = (0.229, 0.152, 0.102)  # yoga block lying flat, long side across the lane
BACK_Y = 0.320  # the block's near face
POST_X = (LANE_X - GATE_GAP / 2 - POST[0] / 2, LANE_X + GATE_GAP / 2 + POST[0] / 2)
CAR_START = (0.225, -0.285)  # middle of the wheelbase at the start
CAR_HEADING = 70.0  # degrees from +x; the lane points along +y (90)
AXLE_Z = FLOOR + PK.R + 0.0005

LANE = Part("lane", (0, 0, 0), [
    box((MAT_XY[0], MAT_XY[1], TOP + MAT[2] / 2), MAT, color=(0.36, 0.38, 0.40)),
    box((MAT_XY[0], LINE_Y + TAPE / 2, FLOOR + 0.00015), (MAT[0] - 0.02, TAPE, 0.0003), color=(0.15, 0.40, 0.85)),
    box((POST_X[0], GATE_Y, FLOOR + POST[2] / 2), POST, color=(0.80, 0.62, 0.36)),
    box((POST_X[1], GATE_Y, FLOOR + POST[2] / 2), POST, color=(0.80, 0.62, 0.36)),
    box((LANE_X, BACK_Y + BLOCK[1] / 2, FLOOR + BLOCK[2] / 2), BLOCK, color=(0.45, 0.25, 0.65))],
    color=(0.36, 0.38, 0.40), friction=(0.9, 0.8))
CAR = PK.car("car", (CAR_START[0], CAR_START[1], AXLE_Z), CAR_HEADING)
OBJECTS = [LANE, CAR]
PRESSED_PARTS = ("car",)  # the hand presses the car onto the mat while it drags it back

# Scoring (README, Scoring): what a person checks on the overhead video and with a tape measure
PULLED = 0.020  # m the car must roll back on the mat in the hand's last hold before it is let go
STILL = 0.05  # m/s: the car is still when let go (not thrown)
SELF_MIN = 0.010  # m it must then move by itself (it was wound)
SELF_RUN = 0.300  # m driven by itself for 0.5
TOUCH_GAP = 0.002  # m: a finger this close to the body counts as touching
REST_SPEED = 0.01  # m/s
REST_S = 0.5
UPRIGHT_DEG = 30.0
TEST_TRACK = None  # offline tests: a dict standing in for a run's history
TEST_HANDS = None  # offline tests: {"left"/"right": (tcp, R, gap)} in place of the arms' joint readings
DIAG = False  # GPU diagnostic copies print CAR_DIAG lines
DEMO_PRESS = 0.010  # m the hand's target goes below the gripping height to press the car down
DEMO_NOPRESS = False  # wrong-way copy: drag the car back without pressing it down


def _run_context():
    context = {"env": None, "t": None, "step": 0, "episode": 0}
    try:
        from yam_arena import inspect_embodiment as _ie
    except ImportError:
        return context
    cls = _ie.YamArenaEmbodiment
    if hasattr(cls, "_rc_run_pullback"):
        return cls._rc_run_pullback
    original_step, original_reset = cls._task_step, cls.reset

    def task_step(self, *args, **kwargs):
        context["env"], context["t"] = self._env, self._t
        context["step"] += 1
        return original_step(self, *args, **kwargs)

    def reset(self, *args, **kwargs):
        PK.restore(None)
        result = original_reset(self, *args, **kwargs)
        PK.restore(None)
        context["episode"] += 1
        context["t"] = 0.0
        context["env"] = self._env
        try:
            PK.install(self._env)
        except Exception as e:  # noqa: BLE001
            print(f"PULLBACK_HOOK failed {e!r}", flush=True)
        return result

    cls._rc_run_pullback = context
    cls._task_step, cls.reset = task_step, reset
    return context


PK.spawn_settings("car")
_RUN = _run_context()
_TRACK = {}
_LAST = {"hands": {}}
_START = {"xy": np.array(CAR_START, float), "heading": math.radians(CAR_HEADING)}  # read in the first second


# ---------------------------------------------------------------- geometry
def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _quat_rot(q):
    x, y, z, w = np.asarray(q, float)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def car_boxes(origin, rot, grow=0.0):
    """The body's two boxes (centre, rotation, half sizes) in the world, from the car's origin and rotation."""
    zb = (PK.CLEARANCE + PK.BODY_TOP) / 2 - PK.R
    zc = (PK.BODY_TOP + PK.HEIGHT) / 2 - PK.R
    out = []
    for c, size in (((0, 0, zb), (PK.LENGTH, PK.WIDTH, PK.BODY_TOP - PK.CLEARANCE)),
                    ((PK.CABIN_X, 0, zc), (PK.CABIN[0], PK.CABIN[1], PK.HEIGHT - PK.BODY_TOP))):
        out.append((np.asarray(origin) + rot @ np.asarray(c), rot, np.asarray(size) / 2 + grow))
    return out


def finger_boxes(tcp, rot, gap):
    """Each finger as two boxes in the TCP frame (x closes, z points to the tips), from the harness's finger hulls."""
    out = []
    for sign in (-1.0, 1.0):
        for (x0, x1), hy, (z0, z1) in (((gap / 2, gap / 2 + 0.007), 0.0135, (-0.003, 0.013)),
                                         ((gap / 2, gap / 2 + 0.010), 0.020, (-0.020, -0.003))):
            c = np.array([sign * (x0 + x1) / 2, 0.0, (z0 + z1) / 2])
            out.append((np.asarray(tcp) + rot @ c, rot, np.array([(x1 - x0) / 2, hy, (z1 - z0) / 2])))
    return out


def _overlap(a, b):
    """Separating-axis test of two oriented boxes (centre, rotation, half sizes)."""
    ca, ra, ha = a
    cb, rb, hb = b
    d = cb - ca
    axes = [ra[:, i] for i in range(3)] + [rb[:, i] for i in range(3)]
    axes += [np.cross(ra[:, i], rb[:, j]) for i in range(3) for j in range(3)]
    for ax in axes:
        n = np.linalg.norm(ax)
        if n < 1e-9:
            continue
        ax = ax / n
        pa = sum(ha[i] * abs(ra[:, i] @ ax) for i in range(3))
        pb = sum(hb[i] * abs(rb[:, i] @ ax) for i in range(3))
        if abs(d @ ax) > pa + pb:
            return False
    return True


def hands():
    """Per arm (TCP position, TCP rotation, finger gap), from the arms' joint readings."""
    if TEST_HANDS is not None:
        return TEST_HANDS
    env = _RUN["env"]
    if env is None:
        return {}
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    out = {}
    for arm in ("left", "right"):
        art = env.unwrapped.scene[arm + "_robot"]
        arm_ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        finger_ids = art.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        q = PK._np(art.data.joint_pos)[0]
        tcp, rot = ik.tcp_pose(q[arm_ids], station.BASES[arm])
        out[arm] = (np.asarray(tcp), np.asarray(rot), float(-np.sum(q[finger_ids])))
    return out


def touching(car_state, hand_poses):
    boxes = car_boxes(np.asarray(car_state.origin, float), _quat_rot(car_state.quat_xyzw), TOUCH_GAP)
    return any(_overlap(f, c) for tcp, rot, gap in hand_poses.values() for f in finger_boxes(tcp, rot, gap)
               for c in boxes)


def _corners(origin, rot):
    pts = []
    for c, r, h in car_boxes(origin, rot)[:1]:
        for sx in (-1, 1):
            for sy in (-1, 1):
                pts.append(c + r @ np.array([sx * h[0], sy * h[1], 0.0]))
    return np.array(pts)


def nose_y(origin, rot):
    return float(_corners(origin, rot)[:, 1].max())


def hits_post(origin, rot):
    """The body (grown 1 mm) overlaps a gate post."""
    posts = [(np.array([x, GATE_Y, FLOOR + POST[2] / 2]), np.eye(3), np.array(POST) / 2) for x in POST_X]
    return any(_overlap(c, p) for c in car_boxes(origin, rot, 0.001) for p in posts)


def _heading(rot):
    return math.atan2(rot[1, 0], rot[0, 0])


def _tilt_deg(rot):
    return math.degrees(math.acos(max(-1.0, min(1.0, rot[2, 2]))))


# ---------------------------------------------------------------- the trial's history
def _fresh():
    return {"episode": _RUN["episode"], "step": -1, "t": 0.0, "origin": None, "touch": False, "spell_start_y": None,
            "spell_max_y": None, "pulled": 0.0, "released": False, "release": None, "valid": False, "why": "",
            "self_run": 0.0, "through": False, "post": False, "rest_since": None, "done": False, "after_touch": False,
            "max_y": None}


def track(car_state, hand_poses):
    """Follow the car once per control step: holds, the release, the run by itself, the gate and the rest."""
    k = _TRACK
    if k.get("episode") != _RUN["episode"] or _RUN["t"] < k.get("t", 0.0) - 1e-9:
        k.clear()
        k.update(_fresh())
    if _RUN["step"] == k["step"]:
        return k
    origin = np.asarray(car_state.origin, float)
    rot = _quat_rot(car_state.quat_xyzw)
    vel = np.asarray(car_state.lin_vel, float)
    t = _RUN["t"]
    on_mat = origin[2] < AXLE_Z + 0.004 and _tilt_deg(rot) < 10
    touch = touching(car_state, hand_poses)
    prev = k["origin"]
    if touch:
        if k["released"] and not k["done"]:
            k["after_touch"] = True  # touched again while it was running: pushed or caught
        if not k["touch"]:
            k["spell_max_y"] = origin[1] if on_mat else None
        if on_mat:
            k["spell_max_y"] = origin[1] if k["spell_max_y"] is None else max(k["spell_max_y"], origin[1])
            k["pulled"] = k["spell_max_y"] - origin[1]
        else:
            k["spell_max_y"], k["pulled"] = None, 0.0
        k["last_touch"] = (origin.copy(), rot.copy(), vel.copy(), on_mat)
    elif k["touch"] and not k["released"]:
        o, r, v, m = k["last_touch"]
        speed = float(np.linalg.norm(v[:2]))
        k["release"] = {"y": float(o[1]), "nose": nose_y(o, r), "on_mat": m, "speed": speed, "pulled": k["pulled"]}
        k["valid"] = m and speed < STILL and k["pulled"] >= PULLED and nose_y(o, r) < LINE_Y
        k["why"] = ("" if k["valid"] else "not on the mat" if not m else "moving when let go" if speed >= STILL
                    else "not pulled back" if k["pulled"] < PULLED else "nose past the start line")
        k["released"] = True
        k["max_y"] = float(o[1])
    if k["released"] and not k["done"] and not touch:
        k["max_y"] = max(k["max_y"], float(origin[1]))
        k["self_run"] = k["max_y"] - k["release"]["y"]
        if prev is not None:
            for s in np.linspace(0.0, 1.0, 6):  # the body between this step and the last (it moves ~3 cm a step)
                o = prev + s * (origin - prev)
                if hits_post(o, rot):
                    k["post"] = True
        if (prev is not None and prev[1] < GATE_Y <= origin[1]
                and LANE_X - GATE_GAP / 2 < origin[0] < LANE_X + GATE_GAP / 2):
            k["through"] = True
        speed = float(np.linalg.norm(vel[:2]))
        if speed < REST_SPEED:
            k["rest_since"] = t if k["rest_since"] is None else k["rest_since"]
            if t - k["rest_since"] >= REST_S and k["self_run"] >= SELF_MIN:
                k["done"] = True
        else:
            k["rest_since"] = None
        if k["self_run"] < SELF_MIN and k["rest_since"] is not None and t - k["rest_since"] >= 1.0:
            k["released"], k["release"] = False, None  # it did not move by itself: not an attempt; the hand may retry
    if k["done"] and touch:
        k["after_touch"] = True
    k["touch"] = touch
    k.update(step=_RUN["step"], t=t, origin=origin)
    return k


def score_terms(objects):
    c = objects["car"]
    origin = np.asarray(c.origin, float)
    rot = _quat_rot(c.quat_xyzw)
    if _RUN["t"] is None:  # offline: one layout; TEST_TRACK stands in for the history
        k = {"released": False, "valid": False, "why": "", "self_run": 0.0, "through": False, "post": False,
             "done": False, "after_touch": False, **(TEST_TRACK or {})}
    else:
        hp = hands()
        k = track(c, hp)
        _LAST["hands"] = hp
        if _RUN["t"] < 1.0:
            _START["xy"], _START["heading"] = origin[:2].copy(), _heading(rot)
    in_bay = (GATE_Y + POST[1] / 2 < origin[1] < BACK_Y + 0.01
              and abs(origin[0] - LANE_X) < BLOCK[0] / 2 and _tilt_deg(rot) < UPRIGHT_DEG)
    terms = {"released": bool(k["released"]), "valid": bool(k["valid"]), "why": k["why"],
             "self_run": float(k["self_run"]), "through": bool(k["through"]), "post": bool(k["post"]),
             "done": bool(k["done"]), "after_touch": bool(k["after_touch"]), "in_bay": bool(in_bay)}
    if DIAG and _RUN["t"] is not None:
        st = c.joints
        print("CAR_DIAG t=%.2f o=(%.4f,%.4f,%.4f) hd=%.1f v=%.3f q=(%.2f,%.2f) W=%.4f pk=%.4f rb=%.4f touch=%d "
              "pulled=%.4f %s" % (_RUN["t"], origin[0], origin[1], origin[2], math.degrees(_heading(rot)),
                                  float(np.linalg.norm(np.asarray(c.lin_vel)[:2])), st["rear"], st["front"],
                                  PK.MOTOR.wind, PK.MOTOR.peak, PK.MOTOR.rolled_back, int(k.get("touch", False)),
                                  k.get("pulled", 0.0), {a: (round(b, 4) if isinstance(b, float) else b)
                                                         for a, b in terms.items()}), flush=True)
        r = _LAST["hands"].get("right")
        if r is not None and _RUN["step"] % 2 == 0:
            print("HAND_DIAG t=%.2f tcp=(%.4f,%.4f,%.4f) gap=%.4f" % (_RUN["t"], *r[0], r[2]), flush=True)
    return terms


def score(objects):
    """First rule that applies, in this order (README, Scoring)."""
    t = score_terms(objects)
    if not t["released"] or not t["valid"] or t["after_touch"] or t["self_run"] < SELF_MIN:
        return 0.0
    if t["through"]:
        return 1.0 if (t["done"] and not t["post"] and t["in_bay"]) else 0.75
    if t["self_run"] >= SELF_RUN:
        return 0.5
    return 0.25


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- scripted demonstration
GRIP_BACK = 0.008  # the fingers close this far behind the middle of the wheelbase (more of the press on the rear)
TIP_UP = 0.010  # fingertips this far above the mat while gripping
GRIP_Z = FLOOR + TIP_UP + 0.0128  # TCP height (the pad tips are 12.8 mm past the TCP)
SET_Y = -0.270  # the car's middle when set down on the lane: its nose 6.5 mm short of the tape
PULL = 0.065  # m dragged back
SQUEEZE = -(PK.WIDTH / 2 - 0.0015)  # finger joint target: jaws 3 mm inside the 54 mm body
OPEN = FINGER_OPEN  # 94 mm


class ReadWaypoint(Waypoint):
    """A waypoint whose position and grip yaw follow the car's start pose (read in the first second)."""

    def __init__(self, duration, finger, label, dz=0.0, lift=0.0):
        self.duration, self.finger, self.label, self.dz, self.lift = duration, finger, label, dz, lift

    @property
    def pos(self):
        h = _START["heading"]
        xy = _START["xy"] - GRIP_BACK * np.array([math.cos(h), math.sin(h)])
        return (float(xy[0]), float(xy[1]), GRIP_Z + self.dz + self.lift)

    @property
    def yaw(self):
        return _START["heading"]


def demo():
    W, RW = Waypoint, ReadWaypoint
    lane = math.pi / 2
    gx, gy = LANE_X, SET_Y - GRIP_BACK
    press = 0.0 if DEMO_NOPRESS else DEMO_PRESS
    right = [RW(1.8, OPEN, "above the car", dz=0.060),
             RW(1.0, OPEN, "down beside its doors"),
             RW(0.6, SQUEEZE, "close across the body"),
             RW(0.5, SQUEEZE, "lift it a little", lift=0.015),
             W(1.4, (gx, gy, GRIP_Z + 0.015), SQUEEZE, "turn it to face the gate", lane),
             W(0.6, (gx, gy, GRIP_Z), SQUEEZE, "set it down behind the start line", lane),
             W(0.5, (gx, gy, GRIP_Z - press), SQUEEZE, "press it onto the mat", lane),
             W(1.6, (gx, gy - PULL, GRIP_Z - press), SQUEEZE, "drag it back, wheels rolling", lane),
             W(0.3, (gx, gy - PULL, GRIP_Z - press), SQUEEZE, "hold", lane),
             W(0.3, (gx, gy - PULL, GRIP_Z - press), OPEN, "let go, still pressing", lane),
             W(0.4, (gx, gy - PULL, GRIP_Z + 0.060), OPEN, "fingers straight up", lane),
             W(1.5, (0.120, -0.300, FLOOR + 0.150), OPEN, "away to the right", lane),
             W(8.0, (0.120, -0.300, FLOOR + 0.150), OPEN, "watch it run", lane)]
    return {"left": [], "right": right}
