"""Open the brush door of a Roomba i7 robot vacuum lying upside down on the bench: pinch the door's green spring
catch, then swing the door up about its hinge until it stands open.

An iRobot Roomba i7 (339 mm across, 3.4 kg) lies upside down on the bench, switched off, as its owner's guide shows
for cleaning the brushes, its front (caster) toward the robot. Its two green rubber brush rollers lie side by side
across the cleaning head. At the rollers' end on the robot's left, a narrow black brush door (an arm 16 mm wide and
94 mm long) is hinged on a pin at its front end and covers the rollers' bearing ends. Its rear end is held down by a
spring catch: two green buttons stick out of either side of the door's end, in finger recesses, and hook under the
recesses' floors. Squeezing both buttons in frees the door. Metres; x forward from the arm bases, y left, z up.
"""
import math
import pathlib
import sys
from collections import deque

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import box, cyl  # noqa: E402
from yam_joints import Hinge, Link, Slide  # noqa: E402
from yam_linkage import LinkageMechanism  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
RELEASE_CLEAR = 0.030
STILL_S = 1.0
TEST_GRIPPERS = None
DIAG = False  # GPU diagnostic copies print VAC_DIAG lines


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


def grippers():
    if TEST_GRIPPERS is not None:
        return TEST_GRIPPERS
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN

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
SUCCESS_HOLD_S, SCORE_OVER = 1.0, "end"
RUN_LIMIT_STEPS = 1200  # 60 s at 20 Hz
DEMO_PACE = 1.0  # scales every demo duration except the fingers' close and open
DEMO_WRONG = False  # the wrong way, for a check run: hold the door's end without squeezing the catch, then lift

MM = 0.001
BODY_BLACK, PLATE_GREY, BUMPER = (0.06, 0.06, 0.07), (0.16, 0.16, 0.17), (0.11, 0.11, 0.12)
DOOR_BLACK, CATCH_GREEN = (0.04, 0.04, 0.045), (0.40, 0.85, 0.12)
ROLLER_LIGHT, ROLLER_DARK, BEARING_GREY = (0.47, 0.78, 0.30), (0.12, 0.42, 0.20), (0.62, 0.63, 0.64)
WHEEL_GREY, BIN_GREY, HUB_GREEN = (0.30, 0.31, 0.32), (0.22, 0.22, 0.23), (0.30, 0.72, 0.30)

# ---------------------------------------------------------------- the vacuum (part frame: origin at the middle of
# its footprint on the bench, upside down; its front (caster) toward -x, i.e. toward the robot)
R_BODY = 0.1695  # 339 mm across (iRobot)
MASS = 3.4  # kg (iRobot)
Z_PLATE = 0.080  # its underside, now facing up, 80 mm above the bench (92.2 mm tall on its wheels; estimate)
LAYER = 0.030  # the underside's moulded layer, in which the cleaning head, wheel wells and recesses sit

# Cleaning head: two rollers across the robot (along y), 25 mm across, 184 mm long, side by side
ROLLER_R, ROLLER_HALF = 0.0125, 0.092
ROLLER_X = (-0.0095, 0.0175)  # front (dark green) and rear (light green) rollers' axes
ROLLER_Z = Z_PLATE - 0.0145  # their tops 2 mm under the underside
TROUGH_X, TROUGH_Y = (-0.032, 0.031), 0.094

# Brush door: a black arm at the rollers' +y end, hinged at its front end on a pin across it (axis y)
DOOR_Y = (0.094, 0.110)
DOOR_YC = 0.102
DOOR_X = (-0.036, 0.058)
DOOR_T = 0.014  # its top flush with the underside
HINGE = (-0.028, DOOR_YC, Z_PLATE - DOOR_T / 2)
DOOR_MASS = 0.012  # estimate
OPEN_STOP_DEG = 100.0  # estimate: it stands just past upright when fully open
HINGE_STICTION = (0.010, 0.008)  # N·m, static and moving (estimate): it stays where it is put

# The catch: a green button each side of the door's end, sprung outward; below each, a tongue hooks under the
# floor of its finger recess. Squeezed 3.5 mm in (flush with the door's sides), both tongues clear.
PINCH_X = 0.047
BTN_X = (0.040, 0.054)
BTN_PROUD = 0.0035
BTN_TOP, BTN_BOT = Z_PLATE + 0.003, Z_PLATE - 0.012
TONGUE_Z = (Z_PLATE - 0.0215, Z_PLATE - 0.0185)
TONGUE_OUT = 0.0065  # the tongue reaches 6.5 mm past the door's side; the recess floor starts 3.5 mm past it
SPRING_K, SPRING_PRELOAD = 400.0, 0.005  # N/m and its preload (2.0 N out, 3.4 N squeezed flush; estimates)
BTN_MASS = 0.002
POCKET_X = (0.031, 0.070)
POCKET_W = 0.018  # each finger recess 18 mm wide beside the door's end
SLAB_Z = (Z_PLATE - 0.018, Z_PLATE - 0.016)  # the recess floor (2 mm), the tongue under it
CAVITY = (0.036, 0.058, Z_PLATE - 0.024)  # x range and floor height of the space the tongues live in
POCKET_IN, POCKET_OUT = (DOOR_Y[0] - POCKET_W, DOOR_Y[0]), (DOOR_Y[1], DOOR_Y[1] + POCKET_W)

WHEEL_R, WHEEL_W, WHEEL_Y, WHEEL_PROUD = 0.036, 0.022, 0.143, 0.015
CASTER = (-0.140, 0.0)
SIDE_BRUSH = (-0.102, -0.111)
BIN_X = (0.080, 0.150)


def _b(x0, x1, y0, y1, z0, z1, color=PLATE_GREY):
    return box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color)


def _subtract(interval, cuts):
    out = [interval]
    for c0, c1 in sorted(cuts):
        nxt = []
        for a, b in out:
            if c1 <= a or c0 >= b:
                nxt.append((a, b))
                continue
            if c0 > a:
                nxt.append((a, c0))
            if c1 < b:
                nxt.append((c1, b))
        out = nxt
    return [(a, b) for a, b in out if b - a > 1e-6]


HOLES = [  # cut through the whole underside layer: (x0, x1, y0, y1)
    (TROUGH_X[0], TROUGH_X[1], -TROUGH_Y, TROUGH_Y),  # the roller trough
    (DOOR_X[0] - 0.002, POCKET_X[1], DOOR_Y[0], DOOR_Y[1]),  # the door's channel
    (POCKET_X[0], POCKET_X[1], POCKET_IN[0], POCKET_IN[1]),  # finger recesses
    (POCKET_X[0], POCKET_X[1], POCKET_OUT[0], POCKET_OUT[1]),
    (-0.040, 0.040, WHEEL_Y - 0.013, WHEEL_Y + 0.013),  # wheel wells
    (-0.040, 0.040, -WHEEL_Y - 0.013, -WHEEL_Y + 0.013),
    (CASTER[0] - 0.012, CASTER[0] + 0.012, -0.012, 0.012),  # caster well
]


def _layer_boxes(step=0.008):
    """The underside layer as boxes: the round outline in 8 mm strips, less the holes."""
    z0, z1 = Z_PLATE - LAYER, Z_PLATE
    xs = set(np.round(np.arange(-R_BODY, R_BODY + 1e-9, step), 6).tolist()) | {R_BODY}
    for h in HOLES:
        xs |= {round(h[0], 6), round(h[1], 6)}
    xs = sorted(x for x in xs if -R_BODY <= x <= R_BODY)
    out = []
    for a, b in zip(xs[:-1], xs[1:]):
        w2 = R_BODY ** 2 - max(a * a, b * b)
        if w2 <= 0:
            continue
        w = math.sqrt(w2)
        cuts = [(h[2], h[3]) for h in HOLES if h[0] <= a + 1e-9 and h[1] >= b - 1e-9]
        for y0, y1 in _subtract((-w, w), cuts):
            out.append(_b(a, b, y0, y1, z0, z1))
    return out


def _base_shapes():
    zl = Z_PLATE - LAYER
    s = [cyl((0.0, 0.0, zl / 2), R_BODY, zl, color=BODY_BLACK)]  # top cover and bumper (on the bench)
    s.append(cyl((0.0, 0.0, zl - 0.004), R_BODY + 0.0005, 0.008, color=BUMPER))  # bumper band
    s += _layer_boxes()
    # floors inside the holes: under the door (along its length), and the latch end's recesses
    s.append(_b(DOOR_X[0] - 0.002, POCKET_X[0], DOOR_Y[0], DOOR_Y[1], zl, Z_PLATE - DOOR_T))
    y_lo, y_hi = POCKET_IN[0], POCKET_OUT[1]
    s.append(_b(POCKET_X[0], POCKET_X[1], y_lo, y_hi, zl, CAVITY[2]))
    cav_y = (DOOR_Y[0] - 0.0085, DOOR_Y[1] + 0.0085)
    s += [_b(POCKET_X[0], POCKET_X[1], y_lo, cav_y[0], CAVITY[2], SLAB_Z[0]),
          _b(POCKET_X[0], POCKET_X[1], cav_y[1], y_hi, CAVITY[2], SLAB_Z[0]),
          _b(POCKET_X[0], CAVITY[0], cav_y[0], cav_y[1], CAVITY[2], SLAB_Z[0]),
          _b(CAVITY[1], POCKET_X[1], cav_y[0], cav_y[1], CAVITY[2], SLAB_Z[0])]
    s += [_b(POCKET_X[0], POCKET_X[1], y_lo, DOOR_Y[0] - BTN_PROUD, SLAB_Z[0], SLAB_Z[1]),  # recess floors
          _b(POCKET_X[0], POCKET_X[1], DOOR_Y[1] + BTN_PROUD, y_hi, SLAB_Z[0], SLAB_Z[1])]
    # rollers, their grey bearing ends, wheels, caster, side-brush hub, bin
    for x, c in zip(ROLLER_X, (ROLLER_DARK, ROLLER_LIGHT)):
        s.append(cyl((x, 0.0, ROLLER_Z), ROLLER_R, 2 * ROLLER_HALF - 0.008, axis="y", color=c))
        for sgn in (1, -1):
            s.append(cyl((x, sgn * (ROLLER_HALF - 0.002), ROLLER_Z), ROLLER_R - 0.002, 0.004, axis="y",
                         color=BEARING_GREY))
    for sgn in (1, -1):
        s.append(cyl((0.0, sgn * WHEEL_Y, Z_PLATE + WHEEL_PROUD - WHEEL_R), WHEEL_R, WHEEL_W, axis="y",
                     color=WHEEL_GREY))
    s.append(cyl((CASTER[0], CASTER[1], Z_PLATE - 0.006), 0.012, 0.016, axis="y", color=WHEEL_GREY))
    s.append(cyl((SIDE_BRUSH[0], SIDE_BRUSH[1], Z_PLATE + 0.003), 0.013, 0.006, color=HUB_GREEN))
    s.append(_b(BIN_X[0], BIN_X[1], -0.100, 0.100, Z_PLATE, Z_PLATE + 0.001, BIN_GREY))
    return s


def _door_shapes():
    return [_b(DOOR_X[0], DOOR_X[1], DOOR_Y[0], DOOR_Y[1], Z_PLATE - DOOR_T, Z_PLATE, DOOR_BLACK),
            cyl(HINGE, 0.003, DOOR_Y[1] - DOOR_Y[0] + 0.002, axis="y", color=BEARING_GREY)]


def _button_shapes(sgn):
    """sgn +1: the button on the door's outer (+y) side; -1: the inner side."""
    side = DOOR_Y[1] if sgn > 0 else DOOR_Y[0]

    def yb(a, b):  # y range from offsets measured outward from the door's side
        lo, hi = sorted((side + sgn * a, side + sgn * b))
        return lo, hi

    return [_b(BTN_X[0], BTN_X[1], *yb(-0.004, BTN_PROUD), BTN_BOT, BTN_TOP, CATCH_GREEN),  # button
            _b(BTN_X[0], BTN_X[1], *yb(0.0005, 0.003), TONGUE_Z[0], BTN_BOT, CATCH_GREEN),  # stem
            _b(BTN_X[0], BTN_X[1], *yb(0.003, TONGUE_OUT), TONGUE_Z[0], TONGUE_Z[1], CATCH_GREEN)]  # tongue


def _vacuum(pos):
    door = Link("door", _door_shapes(),
                Hinge("y", HINGE, -OPEN_STOP_DEG, 0.0, damping=2e-4, armature=1e-5,
                      static_friction=HINGE_STICTION[0], dynamic_friction=HINGE_STICTION[1]),
                mass=DOOR_MASS, color=DOOR_BLACK, friction=(0.6, 0.5))
    anchor = lambda side: (PINCH_X, side, Z_PLATE - 0.006)  # noqa: E731
    out_btn = Link("catch_out", _button_shapes(1),
                   Slide("y", anchor(DOOR_Y[1]), -BTN_PROUD, 0.0, spring=SPRING_K, rest=SPRING_PRELOAD,
                         damping=2.0, armature=0.01),
                   mass=BTN_MASS, parent="door", color=CATCH_GREEN, friction=(0.8, 0.6))
    in_btn = Link("catch_in", _button_shapes(-1),
                  Slide("y", anchor(DOOR_Y[0]), 0.0, BTN_PROUD, spring=SPRING_K, rest=-SPRING_PRELOAD,
                        damping=2.0, armature=0.01),
                  mass=BTN_MASS, parent="door", color=CATCH_GREEN, friction=(0.8, 0.6))
    # The body is fixed where it lies (mass=None): 3.4 kg on the bench, which the arms' few newtons cannot shift or
    # tip. (A free base also made PhysX pick the door, the most-jointed body, as the articulation's root link.)
    return LinkageMechanism("vacuum", pos, _base_shapes(), [door, out_btn, in_btn], mass=None, color=PLATE_GREY,
                     friction=(0.6, 0.5), contact_offset=0.0005, self_collide=True)


START = np.array([0.233, 0.050, TOP])  # the vacuum's footprint centre on the bench
VACUUM = _vacuum(tuple(float(v) for v in START))
OBJECTS = [VACUUM]

# ---------------------------------------------------------------- scoring
OPEN_DEG, PART_DEG, AJAR_DEG = 80.0, 30.0, 5.0
MOVED_MAX, TILT_MAX_DEG = 0.040, 10.0
STILL_WINDOW, STILL_DEG, STILL_MOVE = 1.0, 0.5, 0.002


def door_deg(vac):
    return -vac.angle_deg("door")


def vacuum_terms(vac):
    o = np.asarray(vac.origin, float)
    rot = np.asarray(vac.rotation(), float)
    tilt = math.degrees(math.acos(float(np.clip(rot[2, 2], -1.0, 1.0))))
    moved = float(np.linalg.norm(o[:2] - START[:2]))
    return {"moved": moved, "tilt": tilt, "ok": bool(moved <= MOVED_MAX and tilt <= TILT_MAX_DEG
                                                     and abs(o[2] - TOP) <= 0.02)}


def released_door(vac, clear=RELEASE_CLEAR):
    """Both grippers' centre points at least `clear` outside the door's and catch buttons' boxes."""
    hands = grippers()
    if hands is None:
        return True
    boxes = [vac.link_box(n) for n in ("door", "catch_out", "catch_in")]
    for tcp, _ in hands.values():
        p = np.asarray(tcp, float)
        for low, high in boxes:
            if float(np.linalg.norm(np.maximum(np.maximum(low - p, p - high), 0.0))) < clear:
                return False
    return True


_WINDOW = {"episode": None, "hist": deque()}


def at_rest(vac):
    """The door's angle within 0.5 deg and the vacuum within 2 mm over the last second."""
    t = _RUN["t"]
    if t is None:
        return bool(all(abs(v) < 0.01 for v in vac.joint_vel.values()) and np.linalg.norm(vac.lin_vel) < 0.01)
    hist = _WINDOW["hist"]
    if _WINDOW["episode"] != _RUN["episode"] or (hist and t < hist[-1][0]):
        hist.clear()
        _WINDOW["episode"] = _RUN["episode"]
    ref = (door_deg(vac), np.asarray(vac.origin, float))
    if not hist or t > hist[-1][0] + 1e-9:
        hist.append((t, ref))
    while len(hist) > 1 and hist[1][0] <= t - STILL_WINDOW + 1e-9:
        hist.popleft()
    if t - hist[0][0] < STILL_WINDOW - 0.051:
        return False
    return all(abs(a - ref[0]) <= STILL_DEG and float(np.linalg.norm(o - ref[1])) <= STILL_MOVE for _, (a, o) in hist)


def score_terms(objects):
    vac = objects["vacuum"]
    deg = door_deg(vac)
    v = vacuum_terms(vac)
    free = released_door(vac)
    rest = at_rest(vac)
    full_s = held_for("door open, released, still", v["ok"] and deg >= OPEN_DEG and free and rest)
    if DIAG:
        _diag(vac, deg, v, free, rest)
    return {"deg": deg, "vacuum": v, "released": free, "at_rest": rest, "full_s": full_s}


def score(objects):
    t = score_terms(objects)
    if not t["vacuum"]["ok"]:
        return 0.0
    if t["deg"] >= OPEN_DEG and t["full_s"] + 1e-10 >= STILL_S:
        return 1.0
    if t["deg"] >= PART_DEG:
        return 0.5
    if t["deg"] >= AJAR_DEG:
        return 0.25
    return 0.0


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


def _diag(vac, deg, v, free, rest):
    step = _RUN["step"]
    if step == _DIAG["step"] or step % 4:
        return
    _DIAG["step"] = step
    o = np.asarray(vac.origin, float)
    hands = grippers() or {}
    tcp = " ".join(f"{a[0]}=({','.join(f'{x * 1000:.1f}' for x in hands[a][0])};{hands[a][1]:.3f})" for a in hands)
    print(f"VAC_DIAG t={_RUN['t']:.2f} door={deg:.2f} cout={vac.joints['catch_out'] * 1000:.2f} "
          f"cin={vac.joints['catch_in'] * 1000:.2f} vac=({','.join(f'{x * 1000:.1f}' for x in (o - START))}) "
          f"tilt={v['tilt']:.2f} free={int(free)} rest={int(rest)} {tcp} F={_hand_forces()}", flush=True)


def _hand_forces():
    """Each hand's force (N, world x, y, z) from the joints' PD error: J^T F = K (q_target - q). Diagnostic only."""
    env = _RUN["env"]
    if env is None:
        return "-"
    try:
        from yam_arena import ik, station
        from yam_arena.constants import ARM_JOINTS, SHOULDER, WRIST
        k = np.array([SHOULDER["stiffness"]] * 3 + [WRIST["stiffness"]] * 3)
        out = []
        for arm in ("left", "right"):
            art = env.unwrapped.scene[arm + "_robot"]
            ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
            q = art.data.joint_pos[0][ids].detach().cpu().numpy().astype(float)
            qt = art.data.joint_pos_target[0][ids].detach().cpu().numpy().astype(float)
            tau = k * (qt - q)
            p0, r0 = ik.tcp_pose(q, station.BASES[arm])
            jac = np.zeros((6, 6))
            for j in range(6):
                dq = q.copy()
                dq[j] += 1e-5
                p1, r1 = ik.tcp_pose(dq, station.BASES[arm])
                ax, a = _axis_angle(r1 @ r0.T)
                jac[:3, j] = (p1 - p0) / 1e-5
                jac[3:, j] = ax * a / 1e-5
            w = np.linalg.lstsq(jac.T, tau, rcond=None)[0]
            out.append(f"{arm[0]}({w[0]:.1f},{w[1]:.1f},{w[2]:.1f})")
        return " ".join(out)
    except Exception as e:  # noqa: BLE001
        return f"err:{type(e).__name__}"


# ---------------------------------------------------------------- demo
# Left hand: fingers closing across the door's end (along y), approach tilted 15 deg toward the robot, pads straddling
# the door's end in the two finger recesses. Squeeze (both buttons in, tongues clear), then turn the hand with the
# door about its hinge to 95 deg, let go and slide off its end. The right hand stays parked.
ALPHA_DEG = 15.0
SWING_DEG = 95.0
TIP_Z = Z_PLATE - 0.011  # pad tips' centre, 5 mm above the recess floors
GAP_OPEN, GAP_PINCH, GAP_LIGHT = 0.029, 0.012, 0.0235  # pad gaps (m): open, squeezing (4 mm into the door), touching
PARK = {"left": np.array([0.150, 0.300, TOP + 0.200]), "right": np.array([0.150, -0.300, TOP + 0.200])}
READ_MAX, READ_MAX_DEG = 0.025, 8.0
FINAL_HOLD_S = 3.0
_READ = {"episode": None, "value": None}


def finger(gap):
    return -gap / 2.0


def _rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _axis_angle(rot):
    c = (np.trace(rot) - 1.0) / 2.0
    a = math.acos(float(np.clip(c, -1.0, 1.0)))
    if a < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]) / (2 * math.sin(a))
    return axis, a


def _aa_matrix(axis, a):
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(a) * k + (1.0 - math.cos(a)) * (k @ k)


def _grip0():
    from yam_arena.demo import grip_rotation
    return grip_rotation(0.0)


def hand_pose(deg, back=0.0, along=0.0):
    """Left TCP (part frame) and rotation pinching the door's end with the door open `deg`; `back` moves the hand
    back along its approach, `along` along the door's length (toward its end)."""
    r0 = _rot_y(ALPHA_DEG) @ _grip0()
    tip = np.array([PINCH_X, DOOR_YC, TIP_Z])
    tcp0 = tip - 0.013 * r0[:, 2]
    rd = _rot_y(-deg)
    h = np.asarray(HINGE, float)
    p = h + rd @ (tcp0 - h)
    rot = rd @ r0
    p = p - back * rot[:, 2] + along * (rd @ np.array([1.0, 0.0, 0.0]))
    return p, rot


def _vacuum_now():
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        st = arena_scene.object_states(env, ["vacuum"])["vacuum"]
        return np.asarray(st.origin, float), np.asarray(st.rotation(), float)
    except Exception as e:  # noqa: BLE001
        print(f"VAC_READ failed: {e!r}", flush=True)
        return None


def _read():
    """At the start: where the vacuum actually rests (seeded resets shift it), at most 25 mm and 8 deg."""
    if _READ["episode"] != _RUN["episode"] or _READ["value"] is None:
        now = _vacuum_now()
        if now is None:
            value = (START.copy(), np.eye(3))
        else:
            o, rot = now
            move = o - START
            n = float(np.linalg.norm(move))
            if n > READ_MAX:
                move *= READ_MAX / n
            axis, a = _axis_angle(rot)
            value = (START + move, _aa_matrix(axis, min(a, math.radians(READ_MAX_DEG))))
            print(f"VAC_READ t={_RUN['t']} start_off_mm={tuple(round(float(v) * 1000, 2) for v in o - START)} "
                  f"rot_deg={math.degrees(a):.2f}", flush=True)
        _READ["episode"], _READ["value"] = _RUN["episode"], value
    return _READ["value"]


class Grip(float):
    def __new__(cls, yaw, matrix=None):
        value = super().__new__(cls, yaw)
        value.grip_matrix = None if matrix is None else np.array(matrix, float)
        return value


def _install_grip_matrix():
    from yam_arena import demo as arena_demo
    original = arena_demo.grip_rotation
    if getattr(original, "_grip_matrix", False):
        return

    def grip_rotation(yaw=0.0):
        matrix = getattr(yaw, "grip_matrix", None)
        return matrix.copy() if matrix is not None else original(float(yaw))

    grip_rotation._grip_matrix = True
    grip_rotation.__wrapped__ = original
    arena_demo.grip_rotation = grip_rotation


_install_grip_matrix()


class VacWaypoint(Waypoint):
    """A hand waypoint in the vacuum's frame, placed where the vacuum was read to rest when played."""

    def __init__(self, duration, local_pos, local_rot, finger_target, label, keep=False):
        self.duration = duration * (1.0 if keep else DEMO_PACE)
        self.local_pos, self.local_rot = np.asarray(local_pos, float), np.asarray(local_rot, float)
        self.finger, self.label = finger_target, label

    @property
    def pos(self):
        o, rot = _read()
        return tuple(float(v) for v in o + rot @ self.local_pos)

    @property
    def yaw(self):
        _, rot = _read()
        return Grip(0.0, rot @ self.local_rot)


def demo_plan():
    """{arm: [(label, seconds, local pos, local rot, finger, keep)]}"""
    O, P, L, C = finger(GAP_OPEN), finger(GAP_PINCH), finger(GAP_LIGHT), FINGER_CLOSED
    squeeze = L if DEMO_WRONG else P
    g0 = _grip0()
    left = []
    p, r = hand_pose(0.0, back=0.045)
    left.append(("over the door's end", 3.0, p, r, O, False))
    p, r = hand_pose(0.0, back=0.012)
    left.append(("pads down into the recesses", 1.5, p, r, O, False))
    p, r = hand_pose(0.0)
    left.append(("pads beside the catch buttons", 1.0, p, r, O, False))
    left.append(("squeeze the catch", 1.0, p, r, squeeze, True))
    left.append(("hold the squeeze", 0.5, p, r, squeeze, False))
    steps = [7.5 * k for k in range(1, 13)] + [SWING_DEG]
    for deg in steps:
        p, r = hand_pose(deg)
        left.append((f"swing the door up to {deg:.1f} deg", 0.4, p, r, squeeze, False))
    left.append(("hold open", 0.5, p, r, squeeze, False))
    left.append(("let go", 1.0, p, r, O, True))
    p2, r2 = hand_pose(SWING_DEG, along=0.032)
    left.append(("slide off the door's end", 1.2, p2, r2, O, False))
    left.append(("rise clear", 1.0, p2 + np.array([0.0, 0.0, 0.030]), r2, O, False))
    t_left = sum(row[1] for row in left)
    right = [("stay parked", t_left, PARK["right"] - START, g0, FINGER_OPEN, False)]
    return {"left": left, "right": right}


def demo():
    plan = demo_plan()
    out = {}
    for arm, rows in plan.items():
        wps = [VacWaypoint(secs, p, r, f, label, keep) for label, secs, p, r, f, keep in rows]
        f_park = finger(GAP_OPEN) if arm == "left" else FINGER_OPEN
        wps.append(Waypoint(2.0 * DEMO_PACE, tuple(float(v) for v in PARK[arm]), f_park, "park", 0.0))
        wps.append(Waypoint(FINAL_HOLD_S, tuple(float(v) for v in PARK[arm]), f_park, "hold clear", 0.0))
        out[arm] = wps
    return out
