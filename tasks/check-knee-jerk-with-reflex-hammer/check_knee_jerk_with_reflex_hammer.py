"""Test a knee-jerk reflex: tap a life-size leg trainer's patellar tendon with a Taylor reflex hammer, briskly and just
below the kneecap, so the lower leg kicks; then stand the hammer back in its cup.

A life-size adult right-leg trainer (cut at mid-thigh, its thigh stub clamped, the knee hinged) lies supine on the
bench with its knee resting on an 80 mm foam roll, knee bent about 19 deg, heel on the bench: the standard supine
position for the knee jerk (knee partly flexed and supported, heel resting lightly). It lies diagonally, its foot away
to the right, so the robot stands at the patient's right side. A 190 mm Taylor (tomahawk) percussion hammer stands
head-down in a desk cup to the right, its handle end up. Metres; world x forward from the arm bases, y left, z up.

The reflex is a rule in code (no muscle is simulated): a tap of the hammer's head on the patellar tendon band (just
below the kneecap), landing at 0.6 m/s or more and lifting off within 0.25 s, while the hammer is held, makes the
knee extend; a faster tap kicks harder (10 deg at 1.5 m/s and above). A slow push, a press that stays on the tendon,
or a tap on the kneecap, the tibial tubercle or the shin gives no kick, as on a patient.
"""
import math
import pathlib
import sys
from collections import deque

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import Part, ball, box, cyl  # noqa: E402
from yam_joints import Hinge, Link, Mechanism  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S, SCORE_OVER = 1.0, "end"
RUN_LIMIT_STEPS = 1200  # 60 s at 20 Hz
TEST_GRIPPERS = None
TEST_MODEL = None  # offline fixtures: a dict of reflex-model readouts (the model runs only in the simulator)
DIAG = False  # GPU diagnostic copies print REFLEX_DIAG lines
DEMO_PACE = 1.0
DEMO_SLOW = False  # the wrong way, for a check run: bring the head down onto the tendon slowly (a push, 0.1 m/s)
DEMO_PATELLA = False  # the wrong way, for a check run: the same brisk tap, aimed 40 mm up the leg (onto the kneecap)
LAG_S = (0.065, 0.065, 0.065, 0.15, 0.15, 0.15)  # feed-forward of each arm joint's tracking lag in the swing (s)

MM = 0.001
SKIN = (0.86, 0.70, 0.58)
CUP_BLUE, ROLL_BLUE, CLAMP_GREY = (0.20, 0.36, 0.62), (0.25, 0.42, 0.62), (0.45, 0.46, 0.48)
HEAD_RED, CHROME = (0.78, 0.10, 0.12), (0.55, 0.57, 0.60)
TENDON_TONE, PATELLA_TONE = (0.90, 0.78, 0.68), (0.88, 0.74, 0.62)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


# ---------------------------------------------------------------- the leg trainer (fixed thigh, hinged knee)
# Leg frame: origin at the knee's joint centre; x distal (toward the foot) along the bench, y toward the patient's
# left (medial for this right leg), z up. Shin shapes are authored with the shin level (q = 0); it rests turned
# REST_DEG down about y (heel on the bench). Knee extension is negative q.
LEG_YAW = math.radians(-60.0)  # the leg's distal direction points 60 deg right of straight ahead
KNEE_XY = (0.338, -0.062)
KNEE_H = 0.150  # knee joint centre above the bench
R_LEG = rot_z(LEG_YAW)
KNEE = np.array((KNEE_XY[0], KNEE_XY[1], TOP + KNEE_H))

THIGH_STUB = 0.30  # mid-thigh cut, 300 mm proximal of the knee centre
HIP_H = 0.090  # the thigh's axis would reach the bench-side hip at this height (430 mm proximal)
THIGH_SLOPE = math.atan2(KNEE_H - HIP_H, 0.43)  # 8.0 deg: thigh rising toward the knee

TENDON_X = (0.012, 0.057)  # patellar tendon on the shin (level frame): from the kneecap's lower pole to the tubercle
TENDON_HALF_W = 0.014  # 28 mm wide
TENDON_TOP = 0.045  # its front surface above the knee axis (level frame)
TENDON_T = 0.010
TUBERCLE = (0.064, 0.0, 0.039, 0.012)  # x, y, z, radius (level frame): top 51 mm, 6 mm proud of the tendon
PATELLA = (-0.012, 0.0, 0.037, 0.025)  # kneecap on the thigh (leg frame): front 62 mm above the knee axis
BAND_X = (0.017, 0.047)  # the scored strike band on the tendon (level shin frame): 30 mm along it ...
BAND_HALF_W = 0.015  # ... and 30 mm across

SHIN_SPHERES = [(0.020, 0.040), (0.060, 0.042), (0.100, 0.047), (0.140, 0.052), (0.180, 0.052), (0.220, 0.047),
                (0.260, 0.042), (0.300, 0.037), (0.340, 0.032), (0.380, 0.029), (0.415, 0.030)]  # (x, radius)
HEEL = ((0.445, 0.0, -0.033), (0.060, 0.064, 0.066))  # heel pad (box, level frame): bottom 66 mm below the knee axis
FOOT_BOX = ((0.432, 0.0, 0.075), (0.050, 0.090, 0.200))  # foot upright (supine): sole toward +x, toes up
THIGH_SPHERES = [(0.000, 0.047), (0.035, 0.050), (0.070, 0.054), (0.110, 0.057), (0.150, 0.060), (0.190, 0.062),
                 (0.230, 0.064), (0.270, 0.066)]  # (distance proximal along the thigh axis, radius)
SHIN_MASS = 2.0  # lower leg and foot of the trainer (estimate)
KNEE_DAMPING = 0.3  # N m s/rad passive knee damping (estimate)


def thigh_point(s):
    return np.array((-s * math.cos(THIGH_SLOPE), 0.0, -s * math.sin(THIGH_SLOPE)))


def _rest_deg():
    """The shin's rest angle: the heel pad's far bottom edge just touching the bench."""
    (cx, _, cz), (sx, _, sz) = HEEL
    x, z = cx + sx / 2, cz - sz / 2
    best = None
    for k in range(4001):
        a = math.radians(k * 0.01)
        zz = -x * math.sin(a) + z * math.cos(a)
        err = abs(zz - (-KNEE_H + 0.0005))
        if best is None or err < best[0]:
            best = (err, k * 0.01)
    return best[1]


REST_DEG = _rest_deg()


def _leg_shapes():
    base = []
    for s, r in THIGH_SPHERES:
        base.append(ball(tuple(thigh_point(s)), r, SKIN))
    base.append(ball(PATELLA[:3], PATELLA[3], PATELLA_TONE))
    end = thigh_point(THIGH_STUB)
    base.append(box((end[0] + 0.015, 0.0, (end[2] - KNEE_H) / 2), (0.060, 0.110, end[2] + KNEE_H), CLAMP_GREY))
    shin = [ball((x, 0.0, 0.0), r, SKIN) for x, r in SHIN_SPHERES]
    shin.append(box(((TENDON_X[0] + TENDON_X[1]) / 2, 0.0, TENDON_TOP - TENDON_T / 2),
                    (TENDON_X[1] - TENDON_X[0], 2 * TENDON_HALF_W, TENDON_T), TENDON_TONE))
    shin.append(ball(TUBERCLE[:3], TUBERCLE[3], SKIN))
    shin.append(box(*HEEL, SKIN))
    shin.append(box(*FOOT_BOX, SKIN))
    return base, shin


_BASE_SHAPES, _SHIN_SHAPES = _leg_shapes()
LEG = Mechanism(
    "leg", tuple(KNEE), _BASE_SHAPES,
    [Link("shin", _SHIN_SHAPES,
          Hinge("y", (0.0, 0.0, 0.0), -45.0, 45.0, damping=KNEE_DAMPING, initial=REST_DEG, armature=0.002),
          mass=SHIN_MASS, friction=(0.8, 0.7))],
    mass=None, color=SKIN, yaw=LEG_YAW, friction=(0.8, 0.7), contact_offset=0.002)

ROLL_X = -0.090  # the foam roll under the distal thigh (leg frame)
ROLL_R = 0.040
ROLL = Part("roll", tuple(KNEE + R_LEG @ np.array((ROLL_X, 0.0, -KNEE_H + ROLL_R))),
            [cyl((0.0, 0.0, 0.0), ROLL_R, 0.250, axis="y", color=ROLL_BLUE)], mass=None, color=ROLL_BLUE,
            yaw=LEG_YAW, friction=(0.9, 0.8))

# ---------------------------------------------------------------- the instrument cup and the hammer
CUP_XY = (0.215, -0.400)
CUP_R_IN, CUP_WALL, CUP_H, CUP_FLOOR = 0.040, 0.003, 0.060, 0.004  # 80 mm inside, 60 mm tall, rigid plastic


def _cup_shapes():
    sh = [cyl((0.0, 0.0, CUP_FLOOR / 2), CUP_R_IN + CUP_WALL, CUP_FLOOR, color=CUP_BLUE)]
    n = 16
    r_mid = CUP_R_IN + CUP_WALL / 2
    side = 2 * (CUP_R_IN + CUP_WALL) * math.tan(math.pi / n) + 0.001
    for k in range(n):
        a = 2 * math.pi * k / n
        c = (r_mid * math.cos(a), r_mid * math.sin(a), CUP_H / 2)
        # axis-aligned boxes: the wall is built from thin slabs, each as wide as the facet in x or y
        if abs(math.cos(a)) > abs(math.sin(a)):
            sh.append(box(c, (CUP_WALL, side, CUP_H), CUP_BLUE))
        else:
            sh.append(box(c, (side, CUP_WALL, CUP_H), CUP_BLUE))
    return sh


CUP = Part("cup", (CUP_XY[0], CUP_XY[1], TOP), _cup_shapes(), mass=None, color=CUP_BLUE, friction=(0.5, 0.4))
CUP_RIM = TOP + CUP_H

# Hammer frame (as it stands in the cup, head down): origin at the bottom of the head on the handle's line; z up the
# handle toward its tip; the flat handle lies in the x-z plane (its faces normal to y); the head's broad end points -x
# (toward the robot), its apex +x.
HAM_LEN = 0.190
STRIP = (0.012, 0.003, 0.165)  # flat chrome-plated handle: width (x), thickness (y), length (z)
HEAD_LEN = 0.028  # the head along the handle
HEAD_T = 0.013  # head thickness
HEAD_HALF = 0.025  # broad end 25 mm one side of the handle, apex 25 mm the other
BROAD_FACE = np.array((-HEAD_HALF, 0.0, HEAD_LEN / 2))  # centre of the broad end's striking face
BROAD_N = np.array((-1.0, 0.0, 0.0))  # its outward normal
GRIP_U = 0.035  # grip centre, measured down the handle from its tip
TCP_IN_HAMMER = np.array((0.0, 0.0, HAM_LEN - GRIP_U))
HAM_MASS = 0.070
HAM_POS = np.array((CUP_XY[0], CUP_XY[1], TOP + CUP_FLOOR + 0.0005))
HAMMER = Part("hammer", tuple(HAM_POS), [
    box((0.0, 0.0, HAM_LEN - STRIP[2] / 2), STRIP, CHROME),
    box((-HEAD_HALF / 2, 0.0, HEAD_LEN / 2), (HEAD_HALF, HEAD_T, HEAD_LEN), HEAD_RED),
    box((HEAD_HALF / 2, 0.0, 0.014), (HEAD_HALF, HEAD_T, 0.020), HEAD_RED),
], mass=HAM_MASS, color=HEAD_RED, friction=(0.7, 0.6), contact_offset=0.0015)

OBJECTS = [LEG, ROLL, CUP, HAMMER]

# sample points on the hammer (hammer frame) for the reflex model
_FACE_PTS = [BROAD_FACE + np.array((0.0, dy, dz)) for dz in (-0.012, 0.0, 0.012) for dy in (-0.006, 0.0, 0.006)]
_APEX_PTS = [np.array((HEAD_HALF, dy, HEAD_LEN / 2 + dz)) for dz in (-0.005, 0.005) for dy in (-0.006, 0.006)]
_HANDLE_PTS = [np.array((dx, 0.0, z)) for z in np.linspace(0.03, 0.19, 9) for dx in (-0.006, 0.006)]
SAMPLE_PTS = np.array(_FACE_PTS + _APEX_PTS + _HANDLE_PTS)
SAMPLE_KIND = ["face"] * len(_FACE_PTS) + ["apex"] * len(_APEX_PTS) + ["handle"] * len(_HANDLE_PTS)
FACE_CENTRE_IDX = 4
HAM_LOW = np.array((-HEAD_HALF, -HEAD_T / 2, 0.0))
HAM_HIGH = np.array((HEAD_HALF, HEAD_T / 2, HAM_LEN))

# ---------------------------------------------------------------- the reflex rule
V_MIN = 0.6  # m/s: slower landings give no kick (Tham et al. 2013: ~0.4 m/s gave 0.2 deg, ~0.8 m/s about 2 deg)
V_FULL = 1.5  # m/s: the kick saturates at KICK_MAX_DEG
KICK_MAX_DEG = 10.0
KICK_PASS_DEG = 5.0  # a clear knee jerk (about 40 mm of heel lift): needs a landing of about 1.05 m/s
KICK_WEAK_DEG = 1.0  # a weak jerk (about 8 mm of heel lift): a landing of about 0.69 m/s
T_TAP = 0.25  # s: the head must lift off the tendon within this time of landing (a tap, not a press)
LATENCY = 0.06  # s from landing to the kick
CONTACT_D, RELEASE_D = 0.0030, 0.0050  # m: head-to-skin distance counted as touching / lifted off
KICK_W_PER_SQRT_DEG = 1.08  # rad/s of knee extension speed per sqrt(deg) of target kick (GPU: 1.0 gave 8.5 of 10)
PHYSICS_DT = 1.0 / 240.0
GRIP_SHUT = 0.30  # a gripper holds the hammer only with its fingers under 28 mm apart (fully open: 94 mm)


def kick_target_deg(speed):
    if speed < V_MIN:
        return 0.0
    return float(min(KICK_MAX_DEG, KICK_MAX_DEG * (speed - V_MIN) / (V_FULL - V_MIN)))


def _shin_rot(q):
    return R_LEG @ rot_y(q)


def _sphere_d(p, c, r):
    v = p - c
    d = float(np.linalg.norm(v))
    return d - r, (v / d if d > 1e-9 else np.array((0.0, 0.0, 1.0)))


def _box_d(p, c, half):
    d = np.abs(p - c) - half
    out = float(np.linalg.norm(np.maximum(d, 0.0))) + min(float(np.max(d)), 0.0)
    i = int(np.argmax(d))
    n = np.zeros(3)
    n[i] = 1.0 if p[i] >= c[i] else -1.0
    return out, n


_SHIN_BALLS = np.array([(TUBERCLE[0], TUBERCLE[1], TUBERCLE[2], TUBERCLE[3])] +
                       [(x, 0.0, 0.0, r) for x, r in SHIN_SPHERES[:6]])
_SHIN_BALL_ZONE = ["tubercle"] + ["shin"] * 6
_THIGH_BALLS = np.array([(*PATELLA[:3], PATELLA[3])] + [(*thigh_point(s), r) for s, r in THIGH_SPHERES[:5]])
_THIGH_BALL_ZONE = ["kneecap"] + ["thigh"] * 5
_T_C = np.array(((TENDON_X[0] + TENDON_X[1]) / 2, 0.0, TENDON_TOP - TENDON_T / 2))
_T_H = np.array(((TENDON_X[1] - TENDON_X[0]) / 2, TENDON_HALF_W, TENDON_T / 2))


def _balls(p, balls):
    v = p[:, None, :] - balls[None, :, :3]
    d = np.linalg.norm(v, axis=2)
    return d - balls[None, :, 3], v / np.maximum(d, 1e-9)[:, :, None]


def leg_distances(pts_world, q):
    """For each world point: (distance to the nearest leg surface, zone, outward normal in the world, shin-frame point).
    Zones: tendon, kneecap, tubercle, shin, thigh."""
    Rs = _shin_rot(q)
    pts_world = np.asarray(pts_world, float)
    pl = (pts_world - KNEE) @ R_LEG  # leg frame
    ps = (pts_world - KNEE) @ Rs  # level shin frame
    dd = np.abs(ps - _T_C) - _T_H
    d_t = np.linalg.norm(np.maximum(dd, 0.0), axis=1) + np.minimum(dd.max(axis=1), 0.0)
    d_s, n_s = _balls(ps, _SHIN_BALLS)
    d_l, n_l = _balls(pl, _THIGH_BALLS)
    out = []
    for i in range(len(pts_world)):
        js, jl = int(np.argmin(d_s[i])), int(np.argmin(d_l[i]))
        cands = [(d_t[i], 0), (d_s[i, js], 1), (d_l[i, jl], 2)]
        d, k = min(cands, key=lambda c: c[0])
        if k == 0:
            j = int(np.argmax(dd[i]))
            n = np.zeros(3)
            n[j] = 1.0 if ps[i, j] >= _T_C[j] else -1.0
            out.append((float(d), "tendon", Rs @ n, ps[i]))
        elif k == 1:
            out.append((float(d), _SHIN_BALL_ZONE[js], Rs @ n_s[i, js], ps[i]))
        else:
            out.append((float(d), _THIGH_BALL_ZONE[jl], R_LEG @ n_l[i, jl], ps[i]))
    return out


def in_band(p_shin):
    return bool(BAND_X[0] <= p_shin[0] <= BAND_X[1] and abs(p_shin[1]) <= BAND_HALF_W)


class Reflex:
    """The knee-jerk rule, run before every physics substep (240 Hz)."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.t = 0.0
        self.contact = None  # dict while the hammer touches the leg
        self.prev = None  # (point world positions, point velocities) of the last substep
        self.taps = []  # every touch, classified when it ends (or when it outlasts T_TAP)
        self.kick_at = None  # (time, extension speed)
        self.fired = []  # (time, target deg)
        self.q_rest = math.radians(REST_DEG)
        self.max_ext_deg = 0.0
        self.ext_after_fire = 0.0
        self.lift_seen = False
        self.max_head_speed = 0.0
        self.diag_n = 0

    # ---- readouts for the scorer
    def readout(self):
        good = [tp for tp in self.taps if tp["valid"]]
        band = [tp for tp in self.taps if tp["band"] and tp["head"] and tp["held"]]
        knee = [tp for tp in self.taps if tp["head"] and tp["zone"] in ("tendon", "kneecap", "tubercle", "shin", "thigh")]
        best = max((tp["kick_deg"] for tp in good), default=0.0)
        return {"jerk_deg": best, "band_touch": bool(band), "knee_touch": bool(knee), "taps": len(self.taps),
                "measured_ext_deg": self.ext_after_fire}

    def update(self, ham_pos, ham_rot, ham_v, ham_w, q, hands_fn):
        """Advance one substep; returns a knee joint velocity to write now, or None."""
        self.t += PHYSICS_DT
        ext = math.degrees(self.q_rest - q)
        self.max_ext_deg = max(self.max_ext_deg, ext)
        if self.fired and self.t - self.fired[-1][0] < 0.8:
            self.ext_after_fire = max(self.ext_after_fire, ext)
        cmd = None
        if self.kick_at is not None and self.t >= self.kick_at[0]:
            cmd = -self.kick_at[1]
            self.fired.append((self.t, self.kick_at[2]))
            self.kick_at = None
        pts = ham_pos + SAMPLE_PTS @ ham_rot.T
        vel = ham_v + np.cross(ham_w, pts - ham_pos)
        self.max_head_speed = max(self.max_head_speed, float(np.linalg.norm(vel[FACE_CENTRE_IDX])))
        if float(np.linalg.norm(pts[FACE_CENTRE_IDX] - KNEE)) > 0.30 and self.contact is None:
            self.prev = (pts, vel)
            return cmd
        dist = leg_distances(pts, q)
        i = int(np.argmin([d[0] for d in dist]))
        if DIAG and dist[i][0] < 0.08 and self.diag_n < 400:
            self.diag_n += 1
            fc = (pts[FACE_CENTRE_IDX] - KNEE) @ _shin_rot(q)
            print(f"REFLEX_SUB t={self.t:.4f} d={dist[i][0] * 1000:.1f} zone={dist[i][1]} kind={SAMPLE_KIND[i]} "
                  f"face_shin_mm={np.round(fc * 1000, 1)} v={np.round(vel[FACE_CENTRE_IDX], 3)} "
                  f"knee_deg={math.degrees(q):.2f}", flush=True)
        dmin, zone, n, p_shin = dist[i]
        if self.contact is None:
            if dmin <= CONTACT_D:
                pv = self.prev[1][i] if self.prev is not None else vel[i]
                speed = float(-pv @ n)
                held = _held(ham_pos, ham_rot, hands_fn)
                fc = (pts[FACE_CENTRE_IDX] - KNEE) @ _shin_rot(q)  # the face centre, where the tap lands
                self.contact = {"t0": self.t, "zone": zone, "kind": SAMPLE_KIND[i], "speed": speed,
                                "p_shin": fc, "p_touch": p_shin.copy(), "held": held, "far": 0, "resolved": False}
                if DIAG:
                    print(f"REFLEX_TOUCH t={self.t:.4f} zone={zone} kind={SAMPLE_KIND[i]} speed={speed:.3f} "
                          f"|v|={float(np.linalg.norm(pv)):.3f} face_shin_mm={np.round(fc * 1000, 1)} "
                          f"touch_shin_mm={np.round(p_shin * 1000, 1)} held={held}", flush=True)
        else:
            c = self.contact
            if dmin > RELEASE_D:
                c["far"] += 1
            else:
                c["far"] = 0
            dur = self.t - c["t0"]
            if not c["resolved"] and dur > T_TAP:
                self._resolve(c, dur, pressed=True)
            if c["far"] >= 2:
                if not c["resolved"]:
                    self._resolve(c, dur - PHYSICS_DT, pressed=False)
                self.contact = None
        self.prev = (pts, vel)
        return cmd

    def _resolve(self, c, dur, pressed):
        c["resolved"] = True
        head = c["kind"] in ("face", "apex")
        band = c["zone"] == "tendon" and in_band(c["p_shin"])
        target = kick_target_deg(c["speed"]) if (head and band and c["held"] and not pressed) else 0.0
        tap = {"t": c["t0"], "zone": c["zone"], "kind": c["kind"], "speed": c["speed"], "dur": dur,
               "pressed": pressed, "held": c["held"], "head": head, "band": band,
               "x_mm": 1000 * float(c["p_shin"][0]), "y_mm": 1000 * float(c["p_shin"][1]),
               "kick_deg": target, "valid": target > 0.0}
        self.taps.append(tap)
        if target > 0.0:
            w = KICK_W_PER_SQRT_DEG * math.sqrt(target)
            self.kick_at = (max(c["t0"] + LATENCY, self.t), w, target)
        if DIAG:
            print("REFLEX_TAP " + " ".join(f"{k}={(round(v, 4) if isinstance(v, float) else v)}" for k, v in tap.items()),
                  flush=True)


MODEL = Reflex()


def _held(ham_pos, ham_rot, hands_fn):
    hands = hands_fn() if hands_fn is not None else None
    if not hands:
        return True
    for tcp, opening in hands.values():
        p = ham_rot.T @ (np.asarray(tcp, float) - ham_pos)
        if np.all(p >= HAM_LOW - 0.004) and np.all(p <= HAM_HIGH + 0.004) and opening < GRIP_SHUT:
            return True
    return False


def _np(x):
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x, dtype=np.float64)


def _quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                     (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def _install(environment):
    """Wrap the leg's write_data_to_sim (Isaac Lab calls it before every physics substep): run the reflex rule there
    and, when it fires, write the knee's extension speed."""
    import torch
    base = environment.unwrapped
    leg = base.scene.articulations["leg"]
    ham = base.scene.rigid_objects["hammer"]
    if getattr(leg, "_reflex_hooked", False):
        return
    orig = leg.write_data_to_sim
    origin = _np(base.scene.env_origins)[0][:3]
    jid = list(leg.joint_names).index("shin_joint")
    state = {"err": None}

    def hands():
        return _hands_from(base)

    def write():
        orig()
        try:
            pose = _np(ham.data.root_link_pose_w)[0]
            vel = _np(ham.data.root_link_vel_w)[0]
            q = float(_np(leg.data.joint_pos)[0][jid])
            cmd = MODEL.update(pose[:3] - origin, _quat_matrix(pose[3:7]), vel[:3], vel[3:6], q, hands)
            if cmd is not None:
                v = torch.tensor([[float(cmd)]], dtype=torch.float32, device=leg.device)
                leg.write_joint_velocity_to_sim_index(velocity=v, joint_ids=[jid])
                if DIAG:
                    print(f"REFLEX_KICK t={MODEL.t:.3f} w={cmd:.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            if state["err"] is None:
                state["err"] = repr(e)
                print(f"REFLEX_HOOK error: {e!r}", flush=True)

    leg.write_data_to_sim = write
    leg._reflex_hooked = True


def _restore_leg(env):
    try:
        import torch
        leg = env.unwrapped.scene.articulations["leg"]
        jid = list(leg.joint_names).index("shin_joint")
        q = torch.tensor([[math.radians(REST_DEG)]], dtype=torch.float32, device=leg.device)
        leg.write_joint_position_to_sim_index(position=q, joint_ids=[jid])
        leg.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=[jid])
    except Exception as e:  # noqa: BLE001
        print(f"REFLEX_RESET failed: {e!r}", flush=True)


def _hands_from(base):
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
    out = {}
    for arm in ("left", "right"):
        art = base.scene[arm + "_robot"]
        arm_ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        finger_ids = art.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        q = _np(art.data.joint_pos)[0]
        tcp = ik.tcp_pose(q[arm_ids], station.BASES[arm])[0]
        out[arm] = (tcp, float(np.clip(np.mean(q[finger_ids]) / FINGER_OPEN, 0.0, 1.0)))
    return out


def _run_context():
    context = {"env": None, "t": None, "step": 0, "episode": 0}
    try:
        from yam_arena import inspect_embodiment as _ie
    except ImportError:
        return context
    cls = _ie.YamArenaEmbodiment
    if hasattr(cls, "_rc_run_reflex"):
        return cls._rc_run_reflex
    original_step, original_reset = cls._task_step, cls.reset

    def task_step(self, *args, **kwargs):
        context["env"], context["t"] = self._env, self._t
        context["step"] += 1
        return original_step(self, *args, **kwargs)

    def reset(self, *args, **kwargs):
        env = getattr(self, "_env", None)
        if env is not None:
            _restore_leg(env)
        MODEL.clear()
        result = original_reset(self, *args, **kwargs)
        _restore_leg(self._env)
        MODEL.clear()  # whatever the settle steps did, every trial starts with no taps
        context["episode"] += 1
        context["t"] = 0.0
        context["env"] = self._env
        try:
            _install(self._env)
        except Exception as e:  # noqa: BLE001
            print(f"REFLEX_HOOK install failed: {e!r}", flush=True)
        return result

    cls._rc_run_reflex = context
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
    return _hands_from(env.unwrapped)


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
    if _RUN["t"] is None:
        return bool(cond)
    _episode_memory()
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]


# ---------------------------------------------------------------- scoring
RELEASE_CLEAR = 0.030
STILL_S = 1.0
LIFT_MIN = 0.020  # picked up: the hammer's lowest point 20 mm above the cup's rim, while held
REST_TOL = 0.005  # lying on the bench: lowest point within 5 mm of the bench top


def hammer_pose(h):
    return np.asarray(h.origin, float), np.asarray(h.rotation(), float)


def hammer_corners(h):
    o, rot = hammer_pose(h)
    c = np.array([[x, y, z] for x in (HAM_LOW[0], HAM_HIGH[0]) for y in (HAM_LOW[1], HAM_HIGH[1])
                  for z in (HAM_LOW[2], HAM_HIGH[2])])
    return o + c @ rot.T


def released(h, clear=RELEASE_CLEAR):
    hands = grippers()
    if hands is None:
        return True
    o, rot = hammer_pose(h)
    for tcp, _ in hands.values():
        p = rot.T @ (np.asarray(tcp, float) - o)
        if float(np.linalg.norm(np.maximum(np.maximum(HAM_LOW - p, p - HAM_HIGH), 0.0))) < clear:
            return False
    return True


def holding(h):
    hands = grippers()
    if hands is None:
        return False
    o, rot = hammer_pose(h)
    for tcp, opening in hands.values():
        p = rot.T @ (np.asarray(tcp, float) - o)
        if np.all(p >= HAM_LOW - 0.004) and np.all(p <= HAM_HIGH + 0.004) and opening < GRIP_SHUT:
            return True
    return False


def in_cup(h):
    """The head inside the cup: its lowest point below the rim and within the cup's inner circle."""
    c = hammer_corners(h)
    low = c[np.argmin(c[:, 2])]
    return bool(low[2] < CUP_RIM and math.hypot(low[0] - CUP_XY[0], low[1] - CUP_XY[1]) <= CUP_R_IN)


def put_down(h):
    """At rest and either back in its cup or lying on the bench (lowest point within 5 mm of the bench top)."""
    low = float(np.min(hammer_corners(h)[:, 2]))
    still = bool(np.linalg.norm(h.lin_vel) < 0.01 and np.linalg.norm(h.ang_vel) < 0.5)
    return bool(still and (in_cup(h) or abs(low - TOP) <= REST_TOL))


def reflex_readout():
    if TEST_MODEL is not None:
        return dict(TEST_MODEL)
    if _RUN["t"] is None:
        return {"jerk_deg": 0.0, "band_touch": False, "knee_touch": False, "taps": 0, "measured_ext_deg": 0.0}
    return MODEL.readout()


def score_terms(objects):
    h = objects["hammer"]
    rd = reflex_readout()
    low = float(np.min(hammer_corners(h)[:, 2]))
    held = holding(h)
    picked = ever("hammer picked up (head above the cup's rim, held)", held and low >= CUP_RIM + LIFT_MIN)
    dropped = ever("hammer off the bench", low < TOP - 0.020)
    jerk = rd["jerk_deg"] >= KICK_PASS_DEG
    down = put_down(h) and released(h)
    down_s = held_for("hammer laid down, released, still", down)
    if DIAG:
        _diag(h, rd, held, low)
    return {"readout": rd, "picked": picked, "dropped": dropped, "jerk": jerk, "down": down, "down_s": down_s}


def score(objects):
    t = score_terms(objects)
    rd = t["readout"]
    if t["dropped"]:
        return 0.0
    if t["jerk"] and t["down"] and t["down_s"] + 1e-10 >= STILL_S:
        return 1.0
    if t["jerk"]:
        return 0.75
    if rd["jerk_deg"] >= KICK_WEAK_DEG:
        return 0.5
    if rd["band_touch"]:
        return 0.25
    return 0.0


def success(objects):
    return score(objects) >= 1.0


_DIAG = {"step": -1}


def _diag(h, rd, held, low):
    step = _RUN["step"]
    if step == _DIAG["step"] or step % 5:
        return
    _DIAG["step"] = step
    o, rot = hammer_pose(h)
    face = o + rot @ BROAD_FACE
    print(f"REFLEX_DIAG step={step} t={_RUN['t']:.2f} face=({face[0]:.4f},{face[1]:.4f},{face[2]:.4f}) "
          f"held={held} low={low - TOP:.4f} jerk={rd['jerk_deg']:.2f} band={rd['band_touch']} "
          f"ext={MODEL.max_ext_deg:.2f} vmax={MODEL.max_head_speed:.2f} taps={rd['taps']}", flush=True)


# ---------------------------------------------------------------- scripted demo
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

PRE_OPEN = -0.017  # fingers 34 mm apart before they close across the handle's 12 mm width
BASES = {"left": np.array((0.0, 0.305, 0.75)), "right": np.array((0.0, -0.305, 0.75))}


def hand_in_hammer_nominal():
    """Nominal TCP pose in the hammer frame: on the handle's line GRIP_U below its tip, approach down the handle toward
    the head, fingers closing across the handle's width (on its narrow edges), so the closing axis lies along the
    swing's direction and the pads' normal forces, not friction alone, keep the head from turning in the pinch."""
    R = np.column_stack(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)))  # x_g, y_g, z_g in hammer coordinates
    return TCP_IN_HAMMER.copy(), R


def _hammer_now():
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        st = arena_scene.object_states(env, ["hammer"])["hammer"]
        return np.asarray(st.origin, float), np.asarray(st.rotation(), float)
    except Exception as e:  # noqa: BLE001
        print(f"REFLEX_READ failed: {e!r}", flush=True)
        return None


def _q_now(arm):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena.constants import ARM_JOINTS
        art = env.unwrapped.scene[arm + "_robot"]
        ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        return _np(art.data.joint_pos)[0][ids]
    except Exception as e:  # noqa: BLE001
        print(f"REFLEX_Q failed: {e!r}", flush=True)
        return None


def _rot_axis(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array(((0, -axis[2], axis[1]), (axis[2], 0, -axis[0]), (-axis[1], axis[0], 0)))
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


STRIKE_TILT = math.radians(-10.0)  # at impact the handle dips 10 deg toward the head (hand above the head)
V_PLAN = 1.35  # planned head speed at impact (m/s)
T_ACC, T_CONST, T_OVER = 0.20, 0.15, 0.10  # swing: speed-up, steady approach, planned run past the landing (s)
AIM_OFFSET = np.zeros(3)  # world correction of the aim point (from probe telemetry)
LIFT_Z = 0.065  # lift the hammer this far straight up out of the cup (head 9.5 mm over the rim)


def strike_point():
    """Middle of the scored band on the tendon's front surface, at rest (world)."""
    p = np.array(((BAND_X[0] + BAND_X[1]) / 2, 0.0, TENDON_TOP))
    if DEMO_PATELLA:
        p = p + np.array((-0.040, 0.0, 0.020))
    return KNEE + _shin_rot(math.radians(REST_DEG)) @ p + AIM_OFFSET


def hammer_at_impact():
    """(origin, rotation) of the hammer as its broad face lands on the strike point: handle across the knee from the
    lateral side (along the leg frame's y, the right arm's radial direction), tilted STRIKE_TILT."""
    y_l = R_LEG[:, 1]
    up = np.array((0.0, 0.0, 1.0))
    h = math.cos(STRIKE_TILT) * y_l + math.sin(STRIKE_TILT) * up  # handle direction, tip -> head
    n_f = math.sin(STRIKE_TILT) * y_l - math.cos(STRIKE_TILT) * up  # broad face's outward normal (down)
    z_h, x_h = -h, -n_f
    R = np.column_stack((x_h, np.cross(z_h, x_h), z_h))
    return strike_point() - R @ BROAD_FACE, R


def tcp_for_hammer(o, R, hand):
    p_h, R_h = hand
    return o + R @ p_h, R @ R_h


def _ik(p, R, seeds):
    from yam_arena import ik
    best = None
    for q0 in seeds:
        q, pe, re = ik.solve(p, ik.matrix_to_quat_xyzw(R), np.asarray(q0, float), base_pos=BASES["right"], iters=400)
        if best is None or pe + 0.05 * re < best[1] + 0.05 * best[2]:
            best = (q, pe, re)
    return best


class Plan:
    """Lazily worked-out right-arm motion: reads the hammer where it stands and in the hand, then fixes the swing."""

    def __init__(self):
        self.episode = None
        self.cache = {}

    def reset_if_new(self):
        if self.episode != _RUN["episode"]:
            self.episode = _RUN["episode"]
            self.cache = {}

    def start(self):
        """The hammer where it stands (read at the first approach)."""
        self.reset_if_new()
        if "start" not in self.cache:
            got = _hammer_now()
            nominal = (HAM_POS.copy(), np.eye(3))
            if got is not None and float(np.linalg.norm(got[0] - nominal[0])) < 0.03:
                self.cache["start"] = got
                if DIAG:
                    print(f"REFLEX_READ start: hammer at {np.round(got[0] * 1000, 1)} mm, tilt "
                          f"{math.degrees(math.acos(min(1.0, got[1][2, 2]))):.1f} deg", flush=True)
            else:
                self.cache["start"] = nominal
        return self.cache["start"]

    def hand(self):
        """The TCP in the hammer frame, read once the hammer is lifted (nominal offline or if the read is odd)."""
        self.reset_if_new()
        if "hand" not in self.cache:
            nominal = hand_in_hammer_nominal()
            got = None
            ham, q = _hammer_now(), _q_now("right")
            if ham is not None and q is not None:
                (o, R), (p, Rt) = ham, _fk(q)
                got = (R.T @ (p - o), R.T @ Rt)
                d = float(np.linalg.norm(got[0] - nominal[0]))
                print(f"REFLEX_READ in hand: TCP in hammer {np.round(got[0] * 1000, 1)} mm (nominal "
                      f"{np.round(nominal[0] * 1000, 1)}), off {d * 1000:.1f} mm", flush=True)
                if d > 0.030:
                    got = None
            self.cache["hand"] = got or nominal
        return self.cache["hand"]

    def lift_q(self):
        self.reset_if_new()
        if "lift_q" not in self.cache:
            p, R = _pick_pose(LIFT_Z)
            q = _q_now("right")
            seeds = ([q] if q is not None else []) + [[-0.4, 1.2, 1.0, -0.6, 0.0, 0.0], [-0.4, 1.0, 1.0, -0.6, 0.0, 0.0]]
            self.cache["lift_q"] = _ik(p, R, seeds)[0]
        return self.cache["lift_q"]

    def swing(self):
        self.reset_if_new()
        if "swing" not in self.cache:
            self.cache["swing"] = plan_swing(self.hand())
            sw = self.cache["swing"]
            if DIAG:
                print(f"REFLEX_PLAN q_imp {np.round(sw['q_imp'], 3)} ik {sw['ik_mm']:.2f} mm {sw['ik_deg']:.2f} deg "
                      f"w {np.round(sw['w'], 2)} n_pre {sw['n_pre']} aim {np.round(strike_point(), 4)}", flush=True)
        return self.cache["swing"]

    def hand_after(self):
        """The grip read again after the tap (the head's landing can turn the hammer in the pinch)."""
        self.reset_if_new()
        if "hand_after" not in self.cache:
            got = None
            ham, q = _hammer_now(), _q_now("right")
            if ham is not None and q is not None:
                (o, R), (p, Rt) = ham, _fk(q)
                got = (R.T @ (p - o), R.T @ Rt)
                print(f"REFLEX_READ after tap: TCP in hammer {np.round(got[0] * 1000, 1)} mm", flush=True)
            self.cache["hand_after"] = got or self.hand()
        return self.cache["hand_after"]

    def hand_cup(self):
        """The grip read once more with the hammer hanging upright over its cup, just before it is lowered in."""
        self.reset_if_new()
        if "hand_cup" not in self.cache:
            got = None
            ham, q = _hammer_now(), _q_now("right")
            if ham is not None and q is not None:
                (o, R), (p, Rt) = ham, _fk(q)
                got = (R.T @ (p - o), R.T @ Rt)
                if DIAG:
                    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))
                    print(f"REFLEX_READ over the cup: TCP in hammer {np.round(got[0] * 1000, 1)} mm, hammer at "
                          f"{np.round(o * 1000, 1)} mm, tilt {tilt:.1f} deg", flush=True)
            self.cache["hand_cup"] = got or self.hand_after()
        return self.cache["hand_cup"]

    def place_q(self):
        """Joints holding the hammer upright over its cup (the grip as read after the tap), for the way back."""
        self.reset_if_new()
        if "place_q" not in self.cache:
            p, R = _place_pose(LIFT_Z, self.hand_after())
            self.cache["place_q"] = _ik(p, R, [self.lift_q()])[0]
        return self.cache["place_q"]


PLAN = Plan()


def _head_jac(q, hand):
    def face(qq):
        p, R = _fk(qq)
        p_h, R_h = hand
        Rham = R @ R_h.T
        return p - Rham @ p_h + Rham @ BROAD_FACE
    s0 = face(q)
    J = np.zeros((3, 6))
    for i in range(6):
        dq = np.array(q, float)
        dq[i] += 1e-6
        J[:, i] = (face(dq) - s0) / 1e-6
    return J


CAP = np.array((0.287979, 0.1832595, 0.1570795, 0.1631885, 0.15708, 0.20944))  # runner's per-tick joint caps (rad)


def _rates(q, hand, speed):
    """Rates for the pitch joints (2, 3, 4) giving `speed` m/s of head speed along the face's normal, its sideways part
    under 30 %, using the smallest fraction of the runner's per-tick caps."""
    cap = CAP / 0.05
    J = _head_jac(q, hand)
    _, R = hammer_at_impact()
    n = R @ BROAD_N
    fr = np.linspace(-1.0, 1.0, 61)
    a = np.stack(np.meshgrid(fr, fr, fr, indexing="ij"), -1).reshape(-1, 3)
    w = np.zeros((len(a), 6))
    w[:, 1:4] = a * cap[1:4]
    v = w @ J.T
    along = v @ n
    side = np.linalg.norm(v - np.outer(along, n), axis=1)
    ok = (along >= speed) & (side <= 0.3 * along)
    if not ok.any():
        k = int(np.argmax(np.where(side <= 0.3 * along, along, -1)))
        return w[k]
    m = np.abs(a).max(axis=1)
    return w[int(np.argmin(np.where(ok, m, 9.0)))]


def plan_swing(hand):
    o, R = hammer_at_impact()
    p, Rt = tcp_for_hammer(o, R, hand)
    best = _ik(p, Rt, ([0.53, 0.55, 0.55, -0.25, 0.0, 0.0], [0.53, 0.67, 0.69, -0.37, 0.0, 0.0],
                       [0.5, 1.0, 1.0, -0.6, 0.0, 0.0]))
    q_imp = best[0]
    speed = 0.10 if DEMO_SLOW else V_PLAN
    w = _rates(q_imp, hand, speed)
    dt = 0.05
    t_acc = T_ACC if not DEMO_SLOW else 0.3
    t_const = T_CONST if not DEMO_SLOW else 1.5
    lag = np.asarray(LAG_S) if not DEMO_SLOW else np.zeros(6)
    n_pre = int(round((t_acc + t_const) / dt))
    n_post = int(round(T_OVER / dt))
    ticks = []
    for k in range(-n_pre, n_post + 1):
        t = k * dt
        tau = t + t_acc + t_const  # time since the swing began
        if tau < t_acc:
            s = tau / t_acc
            disp = t_acc * (s ** 3 - s ** 4 / 2) - (t_acc / 2 + t_const)  # integral of a smoothstep speed ramp
            vel = s * s * (3 - 2 * s)
        else:
            disp, vel = t, 1.0
        ticks.append(q_imp + w * disp + lag * w * vel)
    q_up = q_imp + w * (-(t_acc / 2 + t_const))
    return {"q_imp": q_imp, "w": w, "ticks": ticks, "n_pre": n_pre, "q_up": q_up, "q_back": q_imp - 0.25 * w,
            "ik_mm": best[1] * 1000, "ik_deg": math.degrees(best[2])}


def _fk(q):
    from yam_arena import ik
    return ik.tcp_pose(np.asarray(q, float), BASES["right"])


class HandWaypoint(Waypoint):
    """A right-hand waypoint whose pose is worked out when it is first played (and then kept for the episode)."""

    def __init__(self, duration, fn, finger, label):
        self.duration, self.fn, self.finger, self.label = duration, fn, finger, label
        self._key = None
        self._val = None

    def _get(self):
        key = _RUN["episode"]
        if self._val is None or self._key != key:
            self._val = self.fn()
            self._key = key
        return self._val

    @property
    def pos(self):
        return tuple(float(v) for v in self._get()[0])

    @property
    def yaw(self):
        return Grip(0.0, self._get()[1])


def _pick_pose(dz=0.0):
    o, R = PLAN.start()
    return tcp_for_hammer(o + np.array((0.0, 0.0, dz)), R, hand_in_hammer_nominal())


def _place_pose(dz=0.0, hand=None):
    """The hand holding the hammer upright over the cup's centre, head bottom `dz` above the cup floor (the grip as read
    over the cup unless `hand` is given)."""
    o = np.array((CUP_XY[0], CUP_XY[1], TOP + CUP_FLOOR + 0.002 + dz))
    return tcp_for_hammer(o, np.eye(3), hand if hand is not None else PLAN.hand_cup())


ASIDE = np.array((0.0, -0.045, 0.0))  # after letting go, the open hand moves this far aside (clear of the handle)


def _aside_pose():
    p, R = _place_pose(0.060)
    return p + ASIDE, R


def _joint_path(fa, fb, s):
    def fn():
        qa, qb = np.asarray(fa(), float), np.asarray(fb(), float)
        u = s * s * (3 - 2 * s)
        return _fk(qa + u * (qb - qa))
    return fn


def _swing_tick(i):
    def fn():
        sw = PLAN.swing()
        if MODEL.contact is not None or MODEL.taps:
            if DIAG and "retract" not in PLAN.cache:
                PLAN.cache["retract"] = True
                print(f"REFLEX_RETRACT at tick {i} t={_RUN['t']}", flush=True)
            return _fk(sw["q_back"])
        q = sw["ticks"][min(i, len(sw["ticks"]) - 1)]
        if DIAG:
            qa = _q_now("right")
            print(f"REFLEX_TICK i={i} t={_RUN['t']} cmd={np.round(q, 3)} act={None if qa is None else np.round(qa, 3)}",
                  flush=True)
            ham = _hammer_now()
            if i == 0 and ham is not None and qa is not None:
                (o, R), (p, Rt) = ham, _fk(qa)
                p_in, R_in = R.T @ (p - o), R.T @ Rt
                R_nom = PLAN.hand()[1]
                ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R_nom.T @ R_in) - 1) / 2))))
                print(f"REFLEX_GRIP cocked: TCP in hammer {np.round(p_in * 1000, 1)} mm, turned {ang:.1f} deg from "
                      f"the read grip", flush=True)
        return _fk(q)
    return fn


THIGH_REST_S = 0.20  # the left hand rests on the thigh 200 mm above the knee


def left_steady_pose(lift=0.0):
    pt = KNEE + R_LEG @ thigh_point(THIGH_REST_S)
    top = pt[2] + 0.062
    return (float(pt[0]), float(pt[1]), float(top + 0.013 - 0.002 + lift)), LEG_YAW


def demo():
    P = DEMO_PACE
    O = PRE_OPEN
    C = FINGER_CLOSED
    n_swing = len(plan_swing(hand_in_hammer_nominal())["ticks"])
    n_tr = int(round(3.0 * P / 0.05))
    right = [
        HandWaypoint(2.5 * P, lambda: _pick_pose(0.060), O, "hand over the hammer's handle"),
        HandWaypoint(1.5 * P, lambda: _pick_pose(0.0), O, "fingers round the handle's end"),
        HandWaypoint(0.8, lambda: _pick_pose(0.0), C, "pinch the handle"),
        HandWaypoint(1.5 * P, lambda: _pick_pose(LIFT_Z), C, "lift the hammer out of the cup"),
        HandWaypoint(0.5, lambda: _pick_pose(LIFT_Z), C, "hold"),
    ]
    right += [HandWaypoint(0.05, _joint_path(PLAN.lift_q, lambda: PLAN.swing()["ticks"][0], (k + 1) / n_tr), C,
                           "raise the hammer over the knee") for k in range(n_tr)]
    right += [HandWaypoint(1.0, lambda: _fk(PLAN.swing()["ticks"][0]), C, "hold, cocked")]
    right += [HandWaypoint(0.05, _swing_tick(i), C, f"swing {i}") for i in range(n_swing)]
    right += [HandWaypoint(1.0, lambda: _fk(PLAN.swing()["q_back"]), C, "hold up after the tap")]
    right += [HandWaypoint(0.05, _joint_path(lambda: PLAN.swing()["q_back"], PLAN.place_q, (k + 1) / n_tr), C,
                           "carry the hammer back over its cup") for k in range(n_tr)]
    right += [
        HandWaypoint(0.6, lambda: _fk(PLAN.place_q()), C, "hold it upright over the cup"),
        HandWaypoint(1.5 * P, lambda: _place_pose(0.010), C, "lower it head first into the cup"),
        HandWaypoint(0.8, lambda: _place_pose(0.010), O, "let go"),
        HandWaypoint(1.2 * P, lambda: _place_pose(0.060), O, "rise clear of the handle"),
        HandWaypoint(1.0 * P, _aside_pose, O, "move the open hand aside"),
        HandWaypoint(2.0, _aside_pose, O, "hold clear"),
    ]
    pos_up, yaw = left_steady_pose(0.025)
    pos_on, _ = left_steady_pose(0.0)
    t_strike_end = 2.5 * P + 1.5 * P + 0.8 + 1.5 * P + 0.5 + n_tr * 0.05 + 1.0 + n_swing * 0.05 + 1.0
    left = [
        Waypoint(3.0 * P, pos_up, C, "over the thigh", yaw),
        Waypoint(1.5 * P, pos_on, C, "fingers resting on the thigh", yaw),
        Waypoint(max(1.0, t_strike_end - 4.5 * P), pos_on, C, "steady the thigh", yaw),
        Waypoint(1.5 * P, pos_up, C, "lift off the thigh", yaw),
    ]
    t_right = sum(w.duration for w in right)
    left.append(Waypoint(max(1.0, t_right - sum(w.duration for w in left)), pos_up, C, "hold clear", yaw))
    return {"right": right, "left": left}
