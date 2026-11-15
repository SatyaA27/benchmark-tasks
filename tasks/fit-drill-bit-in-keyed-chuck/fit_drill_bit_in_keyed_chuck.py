"""Fit a 6 mm drill bit in a keyed drill chuck.

An unplugged corded drill (Bosch GBM 10 RE, keyed-chuck version: 225 mm long, 200 mm high, 1.3 kg) lies on its side
on the bench in front of the arms, its chuck pointing to the robot's right, its handle toward the robot. The chuck
(10 mm keyed chuck, 32 mm across) is opened to 7.5 mm, one of its key holes facing up. On the right: a 6 mm HSS
twist drill (93 mm long) lies across two 32 mm wooden blocks, its shank end toward the chuck; behind it the chuck key
(T-handle, 110 mm bar) hangs by its bar in a slotted wooden key rest. Metres; x forward from the arm bases, y left,
z up.
"""
import math
import pathlib
import sys
from collections import deque

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import Part, ball, box, cyl  # noqa: E402
import chuck_runtime as CR  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TEST_GRIPPERS = None
TEST_MODEL = None  # offline fixtures: a dict of model readouts (the chuck model runs only in the simulator)
DIAG = False  # GPU diagnostic copies print CHUCK_DIAG lines
DEMO_PACE = 1.0
DEMO_LEAVE_KEY = False  # the wrong way, for a check run: tighten, then leave the key in the chuck
DEMO_NO_HOLD = False  # a check run: the left hand does not hold the drill
DEMO_SHALLOW = False  # the wrong way, for a check run: push the shank only 10 mm in, then tighten as usual
SUCCESS_HOLD_S, SCORE_OVER = 1.0, "end"
RUN_LIMIT_STEPS = 4200  # 210 s at 20 Hz


def _run_context():
    context = {"env": None, "t": None, "step": 0, "episode": 0}
    try:
        from yam_arena import inspect_embodiment as _ie
    except ImportError:
        return context
    cls = _ie.YamArenaEmbodiment
    if hasattr(cls, "_rc_run_chuck"):
        return cls._rc_run_chuck
    original_step, original_reset = cls._task_step, cls.reset

    def task_step(self, *args, **kwargs):
        context["env"], context["t"] = self._env, self._t
        context["step"] += 1
        return original_step(self, *args, **kwargs)

    def reset(self, *args, **kwargs):
        CR.MODEL.clear()
        result = original_reset(self, *args, **kwargs)
        CR.MODEL.clear()  # whatever the settle steps did, every trial starts with the chuck open at 7.5 mm
        context["episode"] += 1
        context["t"] = 0.0
        context["env"] = self._env
        try:
            CR.install(self._env)
        except Exception as e:  # noqa: BLE001
            print(f"CHUCK_HOOK failed: {e!r}", flush=True)
        return result

    cls._rc_run_chuck = context
    cls._task_step, cls.reset = task_step, reset
    return context


_RUN = _run_context()
_MEMORY_EPISODE = None
_TIMERS = {}


def grippers():
    """{arm: (tcp position, opening 0..1)} or None offline."""
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
        tcp, rot = ik.tcp_pose(q[arm_ids], station.BASES[arm])
        opening = float(np.clip(np.mean(q[finger_ids]) / FINGER_OPEN, 0.0, 1.0))
        out[arm] = (tcp, opening)
        _HAND_ROT[arm] = rot
    return out


_HAND_ROT = {}  # each hand's TCP rotation at the last grippers() call (diagnostics)


def _hand_yaw(arm):
    """Degrees the hand is turned about the vertical from a yaw-0 straight-down grip (diagnostics)."""
    rot = _HAND_ROT.get(arm)
    if rot is None:
        return float("nan")
    from yam_arena.demo import grip_rotation
    m = rot @ grip_rotation(0.0).T
    return math.degrees(math.atan2(m[1, 0], m[0, 0]))


def _episode_memory():
    global _MEMORY_EPISODE
    if _MEMORY_EPISODE != _RUN["episode"]:
        _TIMERS.clear()
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


TOP = TABLE_TOP[SCENE]
MM = 0.001

# ---------------------------------------------------------------- colours
BOSCH_BLUE, GRIP_BLACK, CHUCK_STEEL = (0.05, 0.27, 0.50), (0.10, 0.10, 0.11), (0.66, 0.67, 0.69)
JAW_STEEL, GEAR_DARK, HSS_GREY, HSS_DARK = (0.50, 0.51, 0.53), (0.30, 0.31, 0.33), (0.62, 0.63, 0.65), (0.25, 0.25, 0.27)
KEY_STEEL, WOOD = (0.58, 0.59, 0.61), (0.72, 0.56, 0.36)

# ---------------------------------------------------------------- the drill (free, 1.3 kg; never lifted)
# Drill frame: origin at the chuck's nose face on its axis; +y along the axis into the body (world +y), +x from the
# handle toward the motor housing's top (world +x, away from the robot), +z up (the drill lies on its left side).
X_AX, Y_NOSE, Z_AX = 0.240, -0.100, 0.035  # chuck axis 35 mm above the bench (half the housing's 70 mm)
DRILL_MASS = 1.3  # Bosch (EPTA 01:2014)
HOLE_Y, HOLE_TOP = 0.037, 0.0165  # key hole: 37 mm behind the nose face, mouth on the body's 33 mm diameter
HOLE_W, HOLE_FLOOR = 0.0064, 0.0095  # pilot hole (6.0 mm pilot), 7 mm deep
SLEEVE_R, SLEEVE_LEN = 0.016, 0.030
BODY_R, BODY_END = 0.0165, 0.056
JAW_R, JAW_LEN = 0.002, 0.025
OPEN_0 = 0.0075
STOP_Y = 0.034  # the bore's end: a fully inserted shank stops 34 mm in
HOUSING = (0.031, 0.070, 0.225, 0.035)  # half-height (x), front y, rear y, half-width (z)
HANDLE = (-0.169, -0.031, 0.130, 0.175, 0.020)  # x0, x1, y0, y1, half-thickness (z)
FOOT = (-0.169, -0.149, 0.035)  # the handle's foot (cord exit): x0, x1, half-width (z)


def _ring(n, radius, r_cyl, y0, y1, color, skip=(), split=None):
    """n cylinders along y at `radius` about the chuck axis (a knurled sleeve or ribbed body); `split` = (ya, yb)
    leaves a gap between ya and yb in those whose angle (degrees) is in `skip`."""
    out = []
    for k in range(n):
        a = 360.0 * k / n
        x, z = radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a))
        spans = [(y0, y1)]
        if split and any(abs(((a - s + 180) % 360) - 180) < 1e-6 for s in skip):
            spans = [(y0, split[0]), (split[1], y1)]
        for ya, yb in spans:
            out.append(cyl((x, (ya + yb) / 2, z), r_cyl, yb - ya, "y", color))
    return out


def _drill_shapes():
    s = []
    # sleeve (knurled, 32 mm), its ring gear (last 4 mm), the jaws inside
    s += _ring(10, 0.012, 0.004, 0.0, SLEEVE_LEN - 0.004, CHUCK_STEEL)
    s += _ring(10, 0.012, 0.004, SLEEVE_LEN - 0.004, SLEEVE_LEN, GEAR_DARK)
    rc = OPEN_0 / 2 + JAW_R
    for a in CR.JAW_ANGLES:
        x, z = rc * math.cos(math.radians(a)), rc * math.sin(math.radians(a))
        s.append(cyl((x, 0.0015 + (JAW_LEN - 0.0015) / 2, z), JAW_R, JAW_LEN - 0.0015, "y", JAW_STEEL))
        s.append(ball((x, 0.0015, z), JAW_R, JAW_STEEL))  # the jaws' rounded tips lead a shank in
    # body (33 mm) with the key hole on top: 12 ribs, those over the hole cut away; walls each side of the hole
    hy0, hy1 = HOLE_Y - HOLE_W / 2, HOLE_Y + HOLE_W / 2
    s += _ring(12, 0.011, 0.0055, SLEEVE_LEN, BODY_END, CHUCK_STEEL, skip=(60.0, 90.0, 120.0), split=(hy0, hy1))
    for sg in (1, -1):
        s.append(box((sg * (HOLE_W / 2 + 0.0029), HOLE_Y, (HOLE_FLOOR + HOLE_TOP) / 2),
                     (0.0058, HOLE_W, HOLE_TOP - HOLE_FLOOR), CHUCK_STEEL))
    s.append(cyl((0.0, (STOP_Y + BODY_END) / 2, 0.0), HOLE_FLOOR, BODY_END - STOP_Y, "y", CHUCK_STEEL))  # core
    # spindle collar (43 mm), gear head, motor housing, handle, trigger, foot
    s.append(cyl((0.0, 0.063, 0.0), 0.0215, 0.014, "y", GRIP_BLACK))
    s.append(cyl((0.0, 0.080, 0.0), 0.028, 0.020, "y", BOSCH_BLUE))
    hx, y0, y1, hz = HOUSING
    s.append(box((0.0, (y0 + y1) / 2, 0.0), (2 * hx, y1 - y0, 2 * hz), BOSCH_BLUE))
    x0, x1, gy0, gy1, gz = HANDLE
    s.append(box(((x0 + x1) / 2, (gy0 + gy1) / 2, 0.0), (x1 - x0, gy1 - gy0, 2 * gz), GRIP_BLACK))
    s.append(box((-0.043, 0.124, 0.0), (0.024, 0.012, 0.016), GRIP_BLACK))  # trigger
    fx0, fx1, fz = FOOT
    s.append(box(((fx0 + fx1) / 2, (gy0 + gy1) / 2, 0.0), (fx1 - fx0, gy1 - gy0, 2 * fz), GRIP_BLACK))
    return s


DRILL_START = np.array([X_AX, Y_NOSE, TOP + Z_AX])
_DRILL_SHAPES = _drill_shapes()
DRILL = Part("drill", tuple(DRILL_START), _DRILL_SHAPES, mass=DRILL_MASS, color=BOSCH_BLUE, friction=(0.5, 0.4),
             contact_offset=0.001, grip=len(_DRILL_SHAPES) - 1)  # held (never lifted) by the handle's foot

# ---------------------------------------------------------------- the bit (free): DIN 338 HSS, 6 mm x 93 mm
# Bit frame: origin at the shank's end, on its axis; the flutes run toward -y (the bit's axis, tip to shank, is +y).
BIT_D, BIT_LEN, FLUTE_LEN = 0.006, 0.093, 0.057
SHANK_LEN = BIT_LEN - FLUTE_LEN
BIT_MASS = 0.014
REST_X, REST_Z = X_AX, 0.032  # bit rest blocks' tops (the bit's axis 35 mm up, level with the chuck's)
BIT_START = np.array([REST_X, -0.207, TOP + REST_Z + BIT_D / 2])
BIT = Part("bit", tuple(BIT_START),
           [cyl((0.0, -SHANK_LEN / 2, 0.0), BIT_D / 2, SHANK_LEN, "y", HSS_GREY),
            cyl((0.0, -SHANK_LEN - FLUTE_LEN / 2, 0.0), BIT_D / 2, FLUTE_LEN, "y", HSS_DARK)],
           mass=BIT_MASS, color=HSS_GREY, friction=(0.5, 0.4), contact_offset=0.001, grip=1)
BIT_REST = Part("bit_rest", (0.0, 0.0, TOP),
                [box((REST_X, -0.290, REST_Z / 2), (0.040, 0.020, REST_Z), WOOD),
                 box((REST_X, -0.225, REST_Z / 2), (0.040, 0.020, REST_Z), WOOD)],
                mass=None, color=WOOD, friction=(0.5, 0.4))

# ---------------------------------------------------------------- the chuck key (free) and its rest (fixed)
# Key frame: origin at the pilot's tip on the key's axis, +z up the key, the T-bar along +x.
PILOT_R, PINION_R, STEM_R = 0.003, 0.006, 0.003
PINION_Z0, PINION_Z1, BAR_Z = 0.0065, 0.0115, 0.040
BAR = (0.110, 0.008, 0.008)
KEY_MASS = 0.050  # estimate
RACK_X, RACK_Y, RACK_TOP = 0.120, -0.170, 0.048
KEY_START = np.array([RACK_X, RACK_Y, TOP + RACK_TOP - (BAR_Z - BAR[2] / 2)])
KEY = Part("key", tuple(KEY_START),
           [ball((0.0, 0.0, PILOT_R), PILOT_R, KEY_STEEL),
            cyl((0.0, 0.0, (PILOT_R + PINION_Z0) / 2), PILOT_R, PINION_Z0 - PILOT_R, "z", KEY_STEEL),
            cyl((0.0, 0.0, (PINION_Z0 + PINION_Z1) / 2), PINION_R, PINION_Z1 - PINION_Z0, "z", GEAR_DARK),
            cyl((0.0, 0.0, (PINION_Z1 + BAR_Z - BAR[2] / 2) / 2), STEM_R, BAR_Z - BAR[2] / 2 - PINION_Z1, "z",
                KEY_STEEL),
            box((0.0, 0.0, BAR_Z), BAR, KEY_STEEL)],
           mass=KEY_MASS, color=KEY_STEEL, friction=(0.5, 0.4), contact_offset=0.001, grip=4)
_SX, _SY = 0.025, 0.030  # the rest's slot: 50 x 60 mm
RACK = Part("key_rest", (RACK_X, RACK_Y, TOP),
            [box((-(_SX + 0.0175), 0.0, RACK_TOP / 2), (0.035, 0.080, RACK_TOP), WOOD),
             box((_SX + 0.0175, 0.0, RACK_TOP / 2), (0.035, 0.080, RACK_TOP), WOOD),
             box((0.0, -(_SY + 0.005), RACK_TOP / 2), (2 * _SX, 0.010, RACK_TOP), WOOD),
             box((0.0, _SY + 0.005, RACK_TOP / 2), (2 * _SX, 0.010, RACK_TOP), WOOD)],
            mass=None, color=WOOD, friction=(0.5, 0.4))

OBJECTS = [DRILL, BIT, KEY, BIT_REST, RACK]

# hand the geometry to the model
CR.HOLE = np.array((0.0, HOLE_Y, HOLE_TOP))
CR.HOLE_N = np.array((0.0, 0.0, 1.0))
CR.NOSE = np.zeros(3)
CR.AXIS = np.array((0.0, 1.0, 0.0))
CR.JAW_LEN, CR.JAW_R, CR.OPEN_0 = JAW_LEN, JAW_R, OPEN_0
CR.PINION_Z, CR.BAR_Z, CR.BIT_R, CR.BIT_LEN = PINION_Z0, BAR_Z, BIT_D / 2, BIT_LEN
CR.DIAG = DIAG

# ---------------------------------------------------------------- scoring
CAPTURE_MIN = 0.020  # at least 20 mm of shank between the jaws
CR.ALIGN_MIN = CAPTURE_MIN
RUNOUT_AT, RUNOUT_MAX = 0.050, 0.001  # the bit's axis within 1 mm of the chuck's, 50 mm in front of the nose
SECURE_NM = 0.30  # chuck tightened to at least 0.30 N m at the key
KEY_CLEAR = 0.030  # key out: its pilot tip at least 30 mm from the hole's mouth
RELEASE_CLEAR = 0.030
# The workspace is the bench top (the station's 1.10 x 1.30 m bench): at the end the drill's centre of mass must be
# over it and not below its surface (fallen off). Lifting, tipping or moving the drill over the bench is allowed.
BENCH_X, BENCH_Y = (-0.190, 0.910), (-0.650, 0.650)
DRILL_COM = DRILL.volume_centroid()  # drill frame; its colliders at uniform density, as the simulator takes it
STILL_WINDOW, STILL_MOVE = 0.5, 0.0010


def pose(st):
    return np.asarray(st.origin, float), np.asarray(st.rotation(), float)


def world(st, local):
    o, rot = pose(st)
    return o + np.asarray(local, float) @ rot.T


def frames_of(objects):
    out = {}
    for n in ("drill", "bit", "key"):
        st = objects[n]
        o, R = pose(st)
        out[n] = (o, R, np.asarray(st.lin_vel, float), np.asarray(st.ang_vel, float))
    return out


def bit_terms(objects):
    """Depth of the shank end past the nose face (m), its radial offset there, runout 50 mm out."""
    fr = frames_of(objects)
    (pd, Rd, _, _), (pb, Rb, _, _) = fr["drill"], fr["bit"]
    c = Rd[:, 1]
    depth = float((pb - pd) @ c)
    r = pb - pd - ((pb - pd) @ c) * c
    return {"depth": depth, "end_off": float(np.linalg.norm(r)), "runout": CR.MODEL.runout(fr, RUNOUT_AT),
            "angle": math.degrees(math.acos(float(np.clip(Rb[:, 1] @ c, -1, 1))))}


def key_terms(objects):
    d, k = objects["drill"], objects["key"]
    hole = world(d, (0.0, HOLE_Y, HOLE_TOP))
    tip = np.asarray(k.origin, float)
    return {"tip_from_hole": float(np.linalg.norm(tip - hole))}


def drill_terms(objects):
    o, R = pose(objects["drill"])
    tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))
    move = float(np.linalg.norm((o - DRILL_START)[:2]))
    com = world(objects["drill"], DRILL_COM)
    on_bench = bool(BENCH_X[0] <= com[0] <= BENCH_X[1] and BENCH_Y[0] <= com[1] <= BENCH_Y[1] and com[2] >= TOP)
    return {"tilt": tilt, "move": move, "z": float(o[2] - DRILL_START[2]), "on_bench": on_bench}


def model_terms():
    if TEST_MODEL is not None:
        return dict(TEST_MODEL)
    m = CR.MODEL
    return {"gripped": m.gripped, "tightness": m.tightness(), "engaged": m.engaged, "opening": m.d()}


def released(objects, clear=RELEASE_CLEAR):
    hands = grippers()
    if hands is None:
        return True
    boxes = []
    for n in ("drill", "bit", "key"):
        st = objects[n]
        boxes.append((np.asarray(st.low, float), np.asarray(st.high, float)))
    for tcp, _ in hands.values():
        for low, high in boxes:
            if float(np.linalg.norm(np.maximum(np.maximum(low - tcp, tcp - high), 0.0))) < clear:
                return False
    return True


_WINDOW = {"episode": None, "hist": deque()}


def at_rest(objects):
    """Drill and bit moved under 1 mm over the last 0.5 s (on the simulator); offline: velocities."""
    d, b = objects["drill"], objects["bit"]
    t = _RUN["t"]
    if t is None:
        return all(np.linalg.norm(s.lin_vel) < 0.01 and np.linalg.norm(s.ang_vel) < 0.5 for s in (d, b))
    hist = _WINDOW["hist"]
    if _WINDOW["episode"] != _RUN["episode"] or (hist and t < hist[-1][0]):
        hist.clear()
        _WINDOW["episode"] = _RUN["episode"]
    ref = np.vstack([world(d, [(0, 0, 0), (0, 0.2, 0)]), world(b, [(0, 0, 0), (0, -BIT_LEN, 0)])])
    if not hist or t > hist[-1][0] + 1e-9:
        hist.append((t, ref))
    while len(hist) > 1 and hist[1][0] <= t - STILL_WINDOW + 1e-9:
        hist.popleft()
    if t - hist[0][0] < STILL_WINDOW - 0.051:
        return False
    return all(float(np.max(np.linalg.norm(r - ref, axis=1))) <= STILL_MOVE for _, r in hist)


def score_terms(objects):
    b, k, dr, m = bit_terms(objects), key_terms(objects), drill_terms(objects), model_terms()
    on_bench = dr["on_bench"]
    key_out = k["tip_from_hole"] >= KEY_CLEAR and not m["engaged"]
    in_chuck = (0.005 <= b["depth"] <= STOP_Y + 0.002 and b["end_off"] <= CR.OPEN_MAX / 2  # shank end between jaws
                and b["angle"] <= 15.0)
    captured = b["depth"] >= CAPTURE_MIN
    centred = b["runout"] <= RUNOUT_MAX
    gripped = bool(m["gripped"])
    secure = m["tightness"] >= SECURE_NM - 1e-9
    free = released(objects)
    rest = at_rest(objects)
    full = gripped and captured and centred and secure and key_out and free and rest and on_bench
    full_s = held_for("bit gripped, centred, secure; key out; hands clear; still", full)
    if DIAG:
        _diag(objects, b, k, dr, m, free, rest)
    return {"bit": b, "key": k, "drill": dr, "model": m, "on_bench": on_bench, "key_out": key_out,
            "in_chuck": in_chuck, "captured": captured, "centred": centred, "gripped": gripped, "secure": secure,
            "released": free, "at_rest": rest, "full_s": full_s}


def score(objects):
    t = score_terms(objects)
    if not t["on_bench"] or not t["key_out"] or not t["in_chuck"]:
        return 0.0  # the drill fallen off the bench or out past its edge; the key left in; no bit in the chuck
    if t["gripped"] and not t["centred"]:
        return 0.0  # clamped off the chuck's axis (crooked, or shallow and tipped): it would wobble and snap
    tight = t["gripped"] and t["captured"] and t["centred"]
    if tight and t["secure"] and t["released"] and t["full_s"] + 1e-10 >= SUCCESS_HOLD_S:
        return 1.0
    if tight:
        return 0.5
    return 0.25


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


def _diag(objects, b, k, dr, m, free, rest):
    step = _RUN["step"]
    if _RUN["t"] is None or step == _DIAG["step"] or step % 4:
        return
    _DIAG["step"] = step
    M = CR.MODEL
    o, R = pose(objects["drill"])
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    ko = np.asarray(objects["key"].origin, float)
    hands = grippers() or {}
    tcp = " ".join(f"{a[0]}=({','.join(f'{v * 1000:.1f}' for v in hands[a][0])};{hands[a][1]:.2f})" for a in hands)
    print(f"CHUCK_DIAG t={_RUN['t']:.2f} eng={int(M.engaged)} held={int(M.held)} th={M.theta / (2 * math.pi):.3f}t "
          f"psi={M.psi / (2 * math.pi):.3f}t d={M.d() * 1000:.2f} tq={M.torque:.3f} pk={M.peak:.3f} "
          f"lift={M.lift:.1f} gap={M.gap * 1000:.2f} rad={M.radial * 1000:.2f} tilt={M.tilt:.1f} "
          f"grip={int(M.gripped)} tight={M.tightness():.3f} cap={int(M.captured)} depth={b['depth'] * 1000:.1f} "
          f"run={b['runout'] * 1000:.2f} ang={b['angle']:.1f} key_tip=({','.join(f'{v * 1000:.1f}' for v in ko)}) "
          f"kfh={k['tip_from_hole'] * 1000:.1f} drill_d=({','.join(f'{v * 1000:.1f}' for v in o - DRILL_START)}) "
          f"yaw={yaw:.2f} tilt={dr['tilt']:.2f} free={int(free)} rest={int(rest)} {tcp} ry={_hand_yaw('right'):.1f} "
          f"bar_yaw={_yaw_of(pose(objects['key'])[1][:, 0]):.1f}", flush=True)


# ---------------------------------------------------------------- demo
# Left hand: pinches the drill's handle foot across its 45 mm (front and back faces) and holds it from before the bit
# goes in until the key is out. Right hand: picks the bit across its flutes, lines its shank up with the chuck (two
# reads), pushes it 30 mm into the open jaws and lets go (it lies in the lower two jaws); picks the key by the middle
# of its T-bar, carries it over the top hole, lines the pilot up (two reads), lowers it in and presses it down (pinion
# meshed); then strokes: turn the key clockwise (wrist roll, seen from above) up to 180 degrees, ease the press, open,
# roll back, read the bar and close on it again; until the chuck is tightened to 0.35 N m (read from the model's
# state, as a demo may); lifts the key out and hangs it back in its rest.
RIGHT, LEFT = "right", "left"
OPEN_BIT = -0.0105  # pads ~21 mm apart round the 6 mm bit
OPEN_KEY = -0.0080  # pads ~16 mm apart round the 8 mm bar (fits the rest's slot)
OPEN_REGRIP = -0.0100  # pads ~20 mm apart round the 8 mm bar: off it for the roll back, back over it
REGRIP_RISE = 0.035  # pad tips 18 mm over the bar's top before the roll back
OPEN_FOOT = -0.0330  # pads ~66 mm apart round the 45 mm foot
PRESS = 0.014  # TCP commanded 14 mm below the seated bar while turning (~6 N on the key)
YAW_HI, YAW_LO = 140.0, -40.0  # degrees: the stroke window at the hole (joint 6 at -95 .. +85)
STROKE_MAX = 205.0  # the whole window, 165 down to YAW_LO: each stroke ends at YAW_LO (the start cannot creep up)
N_STROKES = 9
GOAL_NM = 0.34
BIT_GRIP_FROM_END = 0.0505  # TCP on the flutes, 50.5 mm from the shank end
INSERT_DEPTH = 0.024
PARK = {LEFT: np.array([0.15, 0.30, TOP + 0.16]), RIGHT: np.array([0.15, -0.30, TOP + 0.16])}
_READS = {"episode": None, "values": {}}


def _reads():
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    return _READS["values"]


def _now(names):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        st = arena_scene.object_states(env, list(names))
        return {n: pose(st[n]) for n in names}
    except Exception as e:  # noqa: BLE001
        print(f"CHUCK_READ failed: {e!r}", flush=True)
        return None


def _rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def _yaw_of(v):
    return math.degrees(math.atan2(v[1], v[0]))


def _near(yaw, ref):
    """yaw + k*180 closest to ref (the T-bar and the bit pinch look the same turned 180 degrees)."""
    return yaw + 180.0 * round((ref - yaw) / 180.0)


def _planned_drill():
    return DRILL_START.copy(), np.eye(3)


def _drill_now():
    v = _reads()
    if "drill0" not in v:
        now = _now(["drill"])
        v["drill0"] = now["drill"] if now else _planned_drill()
        if now:
            o, R = now["drill"]
            print(f"CHUCK_READ drill0 t={_RUN['t']} off_mm={tuple(round(float(x) * 1000, 2) for x in o - DRILL_START)}"
                  f" yaw={_yaw_of(R[:, 0]):.2f}", flush=True)
    return v["drill0"]


def _hole_world():
    o, R = _drill_now()
    return o + R @ np.array((0.0, HOLE_Y, HOLE_TOP)), R[:, 2]


def _nose_world():
    o, R = _drill_now()
    return o, R[:, 1]


def _lag(arm, cmd):
    """Where the hand still has to go to reach its command (commanded minus actual TCP): a part rigid in the hand
    will move this much more, so a read taken while the arm trails corrects only the rest."""
    hands = grippers()
    if hands is None or cmd is None:
        return np.zeros(3)
    return np.asarray(cmd, float) - np.asarray(hands[arm][0], float)


def _cached(key, fn):
    v = _reads()
    if key not in v:
        v[key] = fn()
    return v[key]


# -- right hand: the bit
def _bit0():
    def fn():
        now = _now(["bit"])
        if now is None:
            return BIT_START.copy(), np.eye(3)
        o, R = now["bit"]
        print(f"CHUCK_READ bit0 t={_RUN['t']} off_mm={tuple(round(float(x) * 1000, 2) for x in o - BIT_START)}",
              flush=True)
        return o, R
    return _cached("bit0", fn)


def _bit_pick(lift=0.0, above=0.0):
    o, R = _bit0()
    u = R[:, 1]
    p = o - BIT_GRIP_FROM_END * u
    yaw = _near(_yaw_of(u), 90.0)  # fingers close across the bit
    return p + np.array((0.0, 0.0, lift + above)), yaw


def _bit_line_up(key, short, refine=None):
    """Where the hand goes so the shank end sits `short` in front of the nose on the chuck's axis: the bit in the hand
    is read (actual pose) and the hand moved by the error, turned about the shank end to line the bit's yaw up."""
    def fn():
        prior = _reads().get(refine) if refine else None
        base_p, base_yaw = prior if prior else _bit_carry_cmd()
        cur = _now(["drill"])
        nose, c = (cur["drill"][0], cur["drill"][1][:, 1]) if cur else _nose_world()
        target = nose - short * c
        now = _now(["bit"])
        if now is None:
            return target - BIT_GRIP_FROM_END * c, base_yaw
        o, R = now["bit"]
        lag = _lag(RIGHT, base_p)
        o = o + lag
        u = R[:, 1]
        dyaw = _yaw_of(c) - _yaw_of(u)
        dyaw = (dyaw + 180.0) % 360.0 - 180.0
        dyaw = float(np.clip(dyaw, -8.0, 8.0))
        p = target + _rz(dyaw) @ (np.asarray(base_p) - o)
        err = target - o
        print(f"CHUCK_READ {key} t={_RUN['t']:.2f} end_err_mm={tuple(round(float(x) * 1000, 2) for x in err)} "
              f"dyaw={dyaw:.2f} lag_mm={tuple(round(float(x) * 1000, 2) for x in lag)}", flush=True)
        return p, base_yaw + dyaw
    return _cached(key, fn)


def _bit_carry_cmd():
    """Planned TCP with the shank end 25 mm in front of the nose on its axis (before any read)."""
    nose, c = _nose_world()
    o, R = _bit0()
    _, yaw = _bit_pick()
    return nose - 0.025 * c - BIT_GRIP_FROM_END * c, yaw


# -- right hand: the key
def _key0():
    def fn():
        now = _now(["key"])
        if now is None:
            return KEY_START.copy(), np.eye(3)
        o, R = now["key"]
        print(f"CHUCK_READ key0 t={_RUN['t']:.2f} off_mm={tuple(round(float(x) * 1000, 2) for x in o - KEY_START)} "
              f"bar_yaw={_yaw_of(R[:, 0]):.2f} tilt={math.degrees(math.acos(np.clip(R[2, 2], -1, 1))):.2f}", flush=True)
        return o, R
    return _cached("key0", fn)


def _key_pick(dz=0.0):
    o, R = _key0()
    bar = o + R[:, 2] * BAR_Z
    yaw = _near(_yaw_of(R[:, 0]), 0.0)  # pads close across the bar (yaw 0 closes along y)
    return bar + np.array((0.0, 0.0, dz)), yaw


def _key_over_hole(key, h, refine=None):
    """TCP so the pilot's tip is h above its seated place on the hole's axis (key read in the hand)."""
    def fn():
        cur = _now(["drill"])
        if cur:
            od, Rd = cur["drill"]
            hole, n = od + Rd @ np.array((0.0, HOLE_Y, HOLE_TOP)), Rd[:, 2]
        else:
            hole, n = _hole_world()
        seated_tip = hole - PINION_Z0 * n
        prior = _reads().get(refine) if refine else None
        if prior:
            base_p, yaw = prior
        else:
            _, yaw = _key_pick()
            base_p = seated_tip + n * (h + BAR_Z)
        now = _now(["key"])
        target = seated_tip + n * h
        if now is None:
            return target + n * BAR_Z, yaw
        o, R = now["key"]
        lag = _lag(RIGHT, base_p)
        err = target - (o + lag)
        print(f"CHUCK_READ {key} t={_RUN['t']:.2f} tip_err_mm={tuple(round(float(x) * 1000, 2) for x in err)} "
              f"tilt={math.degrees(math.acos(np.clip(R[:, 2] @ n, -1, 1))):.2f} "
              f"lag_mm={tuple(round(float(x) * 1000, 2) for x in lag)}", flush=True)
        return np.asarray(base_p) + err, yaw
    return _cached(key, fn)


def _seated_bar(refine):
    """The TCP target once seated (the last line-up's x, y; the bar's seated height), and its yaw."""
    p, yaw = _reads().get(refine) or _key_over_hole(refine, 0.009)
    hole, n = _hole_world()
    bar = hole - PINION_Z0 * n + BAR_Z * n
    return np.array((p[0], p[1], bar[2])), yaw


def _stroke(i):
    """Stroke i: (start yaw, end yaw, TCP xy at the bar) worked out when it starts, from the bar's pose and the model's
    state: turn at most 180 degrees, and no further than the chuck needs to reach GOAL_NM."""
    def fn():
        M = CR.MODEL
        if i == 0:
            p, y0 = _seated_bar("pilot2")
        else:
            now = _now(["key"])
            hole, n = _hole_world()
            if now is None:
                p, y0 = _seated_bar("pilot2")
                y0 = YAW_HI
            else:
                o, R = now["key"]
                p = o + R[:, 2] * BAR_Z
                p = np.array((p[0], p[1], (hole - PINION_Z0 * n + BAR_Z * n)[2]))
                y0 = _near(_yaw_of(R[:, 0]), YAW_HI)
                if y0 > YAW_HI + 25.0:
                    y0 -= 180.0
        tc = M.theta_c if M.theta_c is not None else CR.theta_at(2 * CR.BIT_R)
        goal = tc + (GOAL_NM - CR.T_FREE) / CR.K_TIGHT + GOAL_NM / CR.K_MESH
        need = math.degrees(goal - max(M.psi, M.theta)) if _RUN["env"] is not None else STROKE_MAX
        turn = float(np.clip(min(need, y0 - YAW_LO), 0.0, STROKE_MAX))
        print(f"CHUCK_READ stroke{i} t={_RUN['t']} y0={y0:.1f} turn={turn:.1f} need={need:.1f} "
              f"theta={M.theta / (2 * math.pi):.3f}t tight={M.tightness():.3f} eng={int(M.engaged)}", flush=True)
        return np.asarray(p, float), y0, y0 - turn
    return _cached(f"stroke{i}", fn)


class LW(Waypoint):
    """A waypoint whose target is worked out when the arm first reaches it (so it can use a state read)."""

    def __init__(self, duration, fn, finger, label, keep=False):
        self.duration = duration * (1.0 if keep else DEMO_PACE)
        self._fn, self.finger, self.label = fn, finger, label

    @property
    def pos(self):
        return tuple(float(v) for v in self._fn()[0])

    @property
    def yaw(self):
        return math.radians(float(self._fn()[1]))


def _right_plan():
    C = FINGER_CLOSED
    W = []

    def add(d, fn, f, label, keep=False):
        W.append(LW(d, fn, f, label, keep))

    up = np.array((0.0, 0.0, 1.0))
    # the bit
    add(2.5, lambda: _bit_pick(above=0.040), OPEN_BIT, "over the bit")
    add(1.5, lambda: _bit_pick(), OPEN_BIT, "pads down round the flutes")
    add(0.8, lambda: _bit_pick(), C, "pinch the bit", True)
    add(0.3, lambda: _bit_pick(), C, "settle")
    add(1.0, lambda: _bit_pick(lift=0.015), C, "lift off the rest")
    add(2.5, lambda: (_bit_carry_cmd()[0] + up * 0.015, _bit_carry_cmd()[1]), C, "carry to the chuck")
    add(1.0, lambda: _bit_carry_cmd(), C, "down to the axis, 25 mm short")
    add(0.6, lambda: _bit_carry_cmd(), C, "hold")
    add(1.2, lambda: _bit_line_up("nose1", 0.010), C, "line up 10 mm short (read)")
    add(0.6, lambda: _bit_line_up("nose1", 0.010), C, "hold")
    add(0.8, lambda: _bit_line_up("nose2", 0.004, "nose1"), C, "line up 4 mm short (read)")
    add(0.5, lambda: _bit_line_up("nose2", 0.004, "nose1"), C, "hold")
    add(0.6, lambda: _bit_line_up("nose3", 0.003, "nose2"), C, "line up 3 mm short (read)")
    add(0.4, lambda: _bit_line_up("nose3", 0.003, "nose2"), C, "hold")

    def pushed():
        p, yaw = _bit_line_up("nose3", 0.003, "nose2")
        _, c = _nose_world()
        return np.asarray(p) + c * (0.003 + (0.010 if DEMO_SHALLOW else INSERT_DEPTH)), yaw
    add(3.0, pushed, C, "push the shank 24 mm into the jaws")
    add(0.3, pushed, C, "hold")
    add(0.6, pushed, OPEN_BIT, "let go of the bit", True)
    add(1.0, lambda: (pushed()[0] + up * 0.040, pushed()[1]), OPEN_BIT, "rise")
    # the key
    add(2.5, lambda: _key_pick(0.040), OPEN_KEY, "over the key")
    add(1.5, lambda: _key_pick(0.002), OPEN_KEY, "pads down round the bar")
    add(0.8, lambda: _key_pick(0.002), C, "pinch the bar", True)
    add(0.3, lambda: _key_pick(0.002), C, "settle")
    add(1.5, lambda: _key_pick(0.055), C, "lift the key out of its rest")

    def over(h):
        hole, n = _hole_world()
        _, yaw = _key_pick()
        return hole - PINION_Z0 * n + n * (h + BAR_Z), yaw
    add(3.0, lambda: over(0.030), C, "carry over the top hole")
    add(1.0, lambda: over(0.015), C, "down, pilot 15 mm up")
    add(0.6, lambda: over(0.015), C, "hold")
    add(1.0, lambda: _key_over_hole("pilot1", 0.015), C, "line up the pilot (read)")
    add(0.6, lambda: _key_over_hole("pilot1", 0.015), C, "hold")
    add(1.0, lambda: _key_over_hole("pilot2", 0.009, "pilot1"), C, "down, tip 2.5 mm over the mouth (read)")
    add(0.4, lambda: _key_over_hole("pilot2", 0.009, "pilot1"), C, "hold")

    def seat(press):
        p, yaw = _seated_bar("pilot2")
        return p - up * press, yaw
    add(1.5, lambda: seat(0.0), C, "lower the pilot in, pinion on the ring gear")
    add(0.6, lambda: seat(PRESS), C, "press the key in")
    # strokes
    for i in range(N_STROKES):
        def st(frac, press=PRESS, i=i):
            p, y0, y1 = _stroke(i)
            return p - up * press, y0 + frac * (y1 - y0)
        if i > 0:
            add(0.5, lambda st=st: st(0.0, -0.025), OPEN_REGRIP, f"stroke {i + 1}: over the bar (read)")
            add(0.3, lambda st=st: st(0.0, -0.025), OPEN_REGRIP, f"stroke {i + 1}: hold over it")
            add(0.5, lambda st=st: st(0.0, 0.0), OPEN_REGRIP, f"stroke {i + 1}: down round the bar")
            add(0.5, lambda st=st: st(0.0, 0.0), C, f"stroke {i + 1}: close on the bar", True)
            add(0.4, lambda st=st: st(0.0), C, f"stroke {i + 1}: press")
        add(0.9, lambda st=st: st(1 / 3), C, f"stroke {i + 1}: turn (1)")
        add(0.9, lambda st=st: st(2 / 3), C, f"stroke {i + 1}: turn (2)")
        add(0.9, lambda st=st: st(1.0), C, f"stroke {i + 1}: turn (3)")
        add(0.3, lambda st=st: st(1.0), C, f"stroke {i + 1}: hold")
        if i < N_STROKES - 1:
            add(0.3, lambda st=st: st(1.0, 0.0), C, f"stroke {i + 1}: ease the press")
            add(0.5, lambda st=st: st(1.0, 0.0), OPEN_REGRIP, f"stroke {i + 1}: open", True)
            add(0.6, lambda st=st: st(1.0, -REGRIP_RISE), OPEN_REGRIP, f"stroke {i + 1}: rise off the bar")
            add(0.2, lambda st=st: st(1.0, -REGRIP_RISE), OPEN_REGRIP, f"stroke {i + 1}: hold clear")

            def back(frac, i=i):
                p, y0, y1 = _stroke(i)
                yb = y1 + 180.0 if y1 + 180.0 <= YAW_HI + 25.0 else y1
                return p + up * REGRIP_RISE, y1 + frac * (yb - y1)
            add(0.5, lambda b=back: b(1 / 3), OPEN_REGRIP, f"stroke {i + 1}: roll back (1)")
            add(0.5, lambda b=back: b(2 / 3), OPEN_REGRIP, f"stroke {i + 1}: roll back (2)")
            add(0.5, lambda b=back: b(1.0), OPEN_REGRIP, f"stroke {i + 1}: roll back (3)")
            add(0.4, lambda b=back: b(1.0), OPEN_REGRIP, f"stroke {i + 1}: hold (the wrist catches up)")
    # out and back to its rest
    last = N_STROKES - 1

    def lifted(h):
        p, y0, y1 = _stroke(last)
        return p + up * h, y1
    add(0.5, lambda: lifted(0.0), C, "ease the press")
    if DEMO_LEAVE_KEY:
        add(0.6, lambda: lifted(0.0), OPEN_KEY, "let go, key left in", True)
        add(1.5, lambda: lifted(0.060), OPEN_KEY, "rise")
        return W
    add(1.5, lambda: lifted(0.045), C, "lift the key out")
    rest_bar = np.array((RACK_X, RACK_Y, TOP + RACK_TOP + BAR[2] / 2 + 0.001))
    add(1.0, lambda: (lifted(0.045)[0], 0.0), C, "turn the bar along x")
    add(3.0, lambda: (rest_bar + up * 0.050, 0.0), C, "carry to the rest")
    add(1.5, lambda: (rest_bar, 0.0), C, "lower into the rest")
    add(0.6, lambda: (rest_bar, 0.0), OPEN_KEY, "let go", True)
    add(1.0, lambda: (rest_bar + up * 0.050, 0.0), OPEN_KEY, "rise")
    return W


def _left_plan(t_hold_until):
    def foot(dz=0.0):
        o, R = _drill_now()
        x0, x1, gy0, gy1, _ = HANDLE
        p = o + R @ np.array((FOOT[0] + 0.012, (gy0 + gy1) / 2, 0.0))
        return p + np.array((0.0, 0.0, dz)), _near(_yaw_of(R[:, 1]) - 90.0, 0.0)
    if DEMO_NO_HOLD:
        return [Waypoint(3.0, tuple(float(v) for v in PARK[LEFT]), OPEN_FOOT, "park", 0.0)]
    W = [LW(3.0, lambda: foot(0.050), OPEN_FOOT, "over the handle's foot"),
         LW(2.0, lambda: foot(), OPEN_FOOT, "pads down round the foot"),
         LW(0.8, lambda: foot(), FINGER_CLOSED, "pinch the foot", True)]
    used = sum(w.duration for w in W)
    W.append(LW(max(0.5, t_hold_until - used), lambda: foot(), FINGER_CLOSED, "hold the drill"))
    W += [LW(0.8, lambda: foot(), OPEN_FOOT, "let go", True),
          LW(1.2, lambda: foot(0.060), OPEN_FOOT, "rise")]
    return W


def demo():
    right = _right_plan()
    t, t_out = 0.0, None
    for w in right:
        t += w.duration
        if w.label in ("lift the key out", "let go, key left in"):
            t_out = t
    left = _left_plan(t_out if t_out is not None else t)
    out = {}
    for arm, wps in ((RIGHT, right), (LEFT, left)):
        wps = list(wps)
        wps.append(Waypoint(2.0 * DEMO_PACE, tuple(float(v) for v in PARK[arm]), OPEN_KEY if arm == RIGHT else
                            OPEN_FOOT, "park", 0.0))
        wps.append(Waypoint(3.0, tuple(float(v) for v in PARK[arm]), OPEN_KEY if arm == RIGHT else OPEN_FOOT,
                            "hold clear", 0.0))
        out[arm] = wps
    return out
