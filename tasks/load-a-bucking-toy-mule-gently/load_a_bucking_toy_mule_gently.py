"""The bucking-mule game: a spring-loaded toy mule stands on its base with a saddle on its blanket; the saddle has a
horn at the front and two hangers at the back. Three pieces of the game's gear lie on the desk: a coiled rope and a
canteen on the robot's right, a frying pan on its left. The robot hangs all three on the saddle, the rope over the
horn, the canteen and the pan over the back hangers, setting each down so gently that the mule does not buck.

The mule's catch is the force law in bucking_kit.py: a hold of the hindquarters that gives under load and lets go
past an estimated threshold, after which the hindquarters kick up and the saddle is struck upward. Load reaches it only
through contact; nothing snaps a piece onto a hanger.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bucking_kit as bk  # noqa: E402
from yam_parts import Part, ball, box, cyl  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TEST_GRIPPERS = None       # offline tests: {arm: [world points of the gripper]}
TEST_BUCKED = False        # offline tests: pretend the mule has bucked
DIAG = False
DEMO_MODE = ""             # GPU-only copies: "press" pushes each piece down onto the saddle before letting go,
                           # "seq" the probe sequence
DEMO_SEQ = ("", "", "", "press")


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


def _episode_memory():
    global _MEMORY_EPISODE
    if _MEMORY_EPISODE != _RUN["episode"]:
        _TIMERS.clear()
        _EVER.clear()
        _MEMORY_EPISODE = _RUN["episode"]


def ever(key, cond):
    if _RUN["t"] is None:
        return bool(cond)
    _episode_memory()
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]


# ------------------------------------------------------------------ the mule (forward = +x, away from the robot)
TOP = TABLE_TOP[SCENE]
MM = 0.001
# Size from retail listings: mule 22 cm long, 8.5 cm wide, 23 cm tall. Body split, base and masses are estimates.
XM, YM = 0.175, 0.0                       # mule origin: middle of its base, on the desk
BASE_L, BASE_W, BASE_T = 0.200, 0.090, 0.012
BAR_Z0, BAR_Z1 = 0.090, 0.140             # barrel (body) bottom and top
BAR_W = 0.070
BLANKET_T = 0.004
SEAT_Z = BAR_Z1 + BLANKET_T               # the saddle's underside rests here (144 mm)
SADDLE_X = -0.025                         # saddle centre along the back (mule frame)
SEAT_T = 0.006
SEAT_L, SEAT_W = 0.100, 0.066
HANG_R, HANG_H = 0.003, 0.018             # hangers: 6 mm posts, 18 mm above the seat
HANGERS = {"horn": (0.036, 0.0), "left": (-0.040, 0.026), "right": (-0.040, -0.026)}   # saddle frame
M_MULE, M_RUMP, M_SADDLE = 0.30, 0.050, 0.025
COAT, DARK, HOOF = (0.60, 0.46, 0.34), (0.22, 0.16, 0.12), (0.15, 0.13, 0.12)
SAND, YELLOW, RED, BRASS = (0.88, 0.76, 0.50), (0.97, 0.80, 0.12), (0.85, 0.12, 0.10), (0.80, 0.66, 0.25)
ORANGE, BLUE, GREEN, WHITE = (0.98, 0.55, 0.10), (0.15, 0.42, 0.92), (0.20, 0.72, 0.28), (0.95, 0.95, 0.95)


def _mule():
    s = [
        box((0.0, 0.0, BASE_T / 2), (BASE_L, BASE_W, BASE_T), SAND),
        cyl((0.050, 0.020, (BASE_T + BAR_Z0 + 0.002) / 2), 0.009, BAR_Z0 + 0.002 - BASE_T, "z", COAT),
        cyl((0.050, -0.020, (BASE_T + BAR_Z0 + 0.002) / 2), 0.009, BAR_Z0 + 0.002 - BASE_T, "z", COAT),
        box((0.050, 0.020, BASE_T + 0.006), (0.020, 0.020, 0.012), HOOF),
        box((0.050, -0.020, BASE_T + 0.006), (0.020, 0.020, 0.012), HOOF),
        box((0.040, 0.0, (BAR_Z0 + 0.136) / 2), (0.050, 0.064, 0.136 - BAR_Z0), COAT),           # chest
        box((0.072, 0.0, 0.160), (0.030, 0.034, 0.060), COAT),                                    # neck
        box((0.064, 0.0, 0.165), (0.008, 0.010, 0.060), DARK),                                    # mane
        box((0.092, 0.0, 0.205), (0.060, 0.036, 0.034), COAT),                                    # head
        box((0.118, 0.0, 0.198), (0.010, 0.034, 0.022), DARK),                                    # muzzle
        box((0.072, 0.011, 0.229), (0.010, 0.007, 0.014), COAT),                                  # ears
        box((0.072, -0.011, 0.229), (0.010, 0.007, 0.014), COAT),
        ball((0.100, 0.0185, 0.212), 0.003, (0.05, 0.05, 0.05)),                                               # eyes
        ball((0.100, -0.0185, 0.212), 0.003, (0.05, 0.05, 0.05)),
    ]
    return Part("mule", (XM, YM, TOP), s, mass=M_MULE, color=COAT, friction=(0.8, 0.7), contact_offset=0.001)


def _rump():
    # The hindquarters with the blanket on top: the sprung part, at the mule's own origin when set. Hind legs end
    # 5 mm above the base (in the toy they do not carry the body).
    s = [
        box((-0.040, 0.0, (BAR_Z0 + BAR_Z1) / 2), (0.104, BAR_W, BAR_Z1 - BAR_Z0), COAT),       # barrel
        box((-0.035, 0.0, BAR_Z1 + BLANKET_T / 2), (0.100, 0.088, BLANKET_T), YELLOW),            # blanket
        cyl((-0.075, 0.020, (BASE_T + 0.005 + BAR_Z0 + 0.002) / 2), 0.010, BAR_Z0 + 0.002 - BASE_T - 0.005, "z", COAT),
        cyl((-0.075, -0.020, (BASE_T + 0.005 + BAR_Z0 + 0.002) / 2), 0.010, BAR_Z0 + 0.002 - BASE_T - 0.005, "z", COAT),
        box((-0.097, 0.0, 0.112), (0.010, 0.010, 0.040), DARK),                                   # tail
    ]
    return Part("rump", (XM, YM, TOP), s, mass=M_RUMP, color=COAT, friction=(0.8, 0.7), contact_offset=0.001)


def _saddle():
    # Origin: middle of the seat's underside.
    s = [box((0.0, 0.0, SEAT_T / 2), (SEAT_L, SEAT_W, SEAT_T), RED),
         box((0.0, 0.048, -0.012), (0.060, 0.004, 0.036), RED),
         box((0.0, -0.048, -0.012), (0.060, 0.004, 0.036), RED),
         box((0.0, 0.0405, SEAT_T / 2), (0.060, 0.015, SEAT_T), RED),
         box((0.0, -0.0405, SEAT_T / 2), (0.060, 0.015, SEAT_T), RED)]
    for hx, hy in HANGERS.values():
        s.append(cyl((hx, hy, SEAT_T + HANG_H / 2), HANG_R, HANG_H, "z", BRASS))
    return Part("saddle", (XM + SADDLE_X, YM, TOP + SEAT_Z), s, mass=M_SADDLE, color=RED, friction=(0.8, 0.7),
                contact_offset=0.001)


def ring(center, r_in, r_out, h, color):
    """A flat ring (axis z) as small squares of side (r_out - r_in) round its mid-circle (the kit's boxes do not
    turn); the squares' corners narrow the hole a little: see hole_free()."""
    rm, w = (r_in + r_out) / 2, r_out - r_in
    n = max(8, math.ceil(2 * math.pi * rm * 1.1 / w))
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        out.append(box((center[0] + rm * math.cos(a), center[1] + rm * math.sin(a), center[2]), (w, w, h), color))
    return out


def hole_free(r_in, r_out):
    """Smallest clear radius of a ring() hole (a square's corner pointing at the centre)."""
    rm, w = (r_in + r_out) / 2, r_out - r_in
    return rm - w / math.sqrt(2)


# Pieces: origin at the centre of the hanging hole, on the piece's underside; body forward (+x) of the hole.
# Sizes and masses are estimates (ruler and kitchen scale on the real pieces).
PIECES = {
    "rope":    dict(hole_r=0.010, out_r=0.018, h=0.008, body=(0.0, 0.0), body_r=0.018, mass=0.008, color=ORANGE,
                    hanger="horn"),
    "canteen": dict(hole_r=0.008, out_r=0.012, h=0.012, body=(0.029, 0.0), body_r=0.017, mass=0.009, color=GREEN,
                    hanger="right"),
    "pan":     dict(hole_r=0.007, out_r=0.011, h=0.006, body=(0.035, 0.0), body_r=0.018, mass=0.006, color=BLUE,
                    hanger="left"),
}
START = {"rope": (0.255, -0.150), "canteen": (0.105, -0.175), "pan": (0.100, 0.175)}


def _piece(name):
    p = PIECES[name]
    if name == "rope":
        s = ring((0.0, 0.0, p["h"] / 2), p["hole_r"], p["out_r"], p["h"], ORANGE)
    elif name == "canteen":
        s = ring((0.0, 0.0, 0.002), p["hole_r"], p["out_r"], 0.004, DARK)                  # strap loop
        bx = p["body"][0]
        s += [cyl((bx, 0.0, p["h"] / 2), p["body_r"], p["h"], "z", GREEN),
              cyl((bx + p["body_r"] + 0.003, 0.0, p["h"] / 2), 0.004, 0.006, "x", DARK)]           # cap
    else:
        s = ring((0.0, 0.0, 0.002), p["hole_r"], p["out_r"], 0.004, BLUE)                  # handle end
        bx = p["body"][0]
        s += [box(((p["out_r"] + bx - p["body_r"]) / 2 - 0.001, 0.0, 0.002), (bx - p["body_r"] - p["out_r"] + 0.004,
                  0.009, 0.004), BLUE),                                                             # handle
              cyl((bx, 0.0, p["h"] / 2), p["body_r"], p["h"], "z", BLUE),                          # pan
              cyl((bx, 0.0, p["h"] + 0.0003), p["body_r"] - 0.003, 0.0006, "z", (0.32, 0.34, 0.40))]   # inside
    x, y = START[name]
    # grip: the shape the pads close on (rope: the coil's side, 14 mm off its centre; others: the body)
    grip = (len(s) // 4) if name == "rope" else (len(s) - 2 if name == "canteen" else len(s) - 2)
    return Part(name, (x, y, TOP), s, mass=p["mass"], color=p["color"], friction=(0.9, 0.8), contact_offset=0.001,
                grip=grip)


MULE, RUMP, SADDLE = _mule(), _rump(), _saddle()
OBJECTS = [MULE, RUMP, SADDLE] + [_piece(n) for n in ("rope", "canteen", "pan")]
CATCH = bk.Catch("mule", "rump", "saddle", rump_com=RUMP.volume_centroid(), mule_com=MULE.volume_centroid(),
                 seat=(SADDLE_X, 0.0, SEAT_Z), m_rump=M_RUMP, m_saddle=M_SADDLE)
bk.install(CATCH, saddle_local=(SADDLE_X, 0.0, SEAT_Z))


# ------------------------------------------------------------------ what a person checks
HAND_CLEAR = 0.030         # every gripper point at least this far from a piece's body centre
TIP_DEG, MOVED_M = 20.0, 0.030


def gripper_points():
    """{arm: (k, 3) world points}: tool point, the two pad centres and the two fingertips, from the arm's joints."""
    if TEST_GRIPPERS is not None:
        return {k: np.asarray(v, float) for k, v in TEST_GRIPPERS.items()}
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
        q = getattr(art.data.joint_pos, "torch", art.data.joint_pos)[0]
        q = np.asarray(q.detach().cpu().numpy() if hasattr(q, "detach") else q, float)
        p, R = ik.tcp_pose(q[arm_ids], station.BASES[arm])
        half = abs(float(np.mean(q[finger_ids])))
        x, z = R[:, 0], R[:, 2]
        out[arm] = np.array([p, p + half * x, p - half * x, p + half * x + 0.013 * z, p - half * x + 0.013 * z])
    return out


def on_hanger(piece_state, saddle_state, name):
    """The piece hangs on a saddle hanger: the post passes through its hole and the hole sits on the seat or the
    post (between 5 mm under the seat top and the post's top), the piece lying within 30 degrees of flat."""
    p = PIECES[name]
    hole = saddle_state.to_local(np.asarray(piece_state.origin, float))
    if piece_state.tilt_deg() > 30.0:
        return None
    for hname, (hx, hy) in HANGERS.items():
        d = math.hypot(hole[0] - hx, hole[1] - hy)
        if d <= hole_free(p["hole_r"], p["out_r"]) - HANG_R + 0.001 and SEAT_T - 0.005 <= hole[2] <= SEAT_T + HANG_H:
            return hname
    return None


def status(objects):
    sad, mule = objects["saddle"], objects["mule"]
    pts = gripper_points()
    hung = {}
    for name in PIECES:
        st = objects[name]
        h = on_hanger(st, sad, name)
        clear = True
        if pts:
            body = st.to_world((PIECES[name]["body"][0], 0.0, PIECES[name]["h"] / 2))
            for arr in pts.values():
                if float(np.min(np.linalg.norm(arr - body, axis=1))) < HAND_CLEAR:
                    clear = False
        hung[name] = (h, clear)
    if not hasattr(status, "_start") or status._start[0] != _RUN["episode"] or _RUN["t"] is None:
        status._start = (_RUN["episode"], np.asarray(mule.origin, float)[:2].copy())
    start = np.array([XM, YM]) if _RUN["t"] is None else status._start[1]
    moved = float(np.linalg.norm(np.asarray(mule.origin, float)[:2] - start))
    saddle_on = sad.tilt_deg() < 20.0 and abs(float(sad.origin[2]) - (float(mule.origin[2]) + SEAT_Z)) < 0.015
    return dict(hung=hung, moved=moved, tilt=mule.tilt_deg(), saddle_on=saddle_on)


def count_hung(st):
    if not st["saddle_on"]:
        return 0
    return sum(1 for h, clear in st["hung"].values() if h is not None and clear)


def score(objects):
    st = status(objects)
    bucked = ever("mule bucked", bk.tripped() or TEST_BUCKED)
    knocked = ever("mule knocked over or shoved", st["tilt"] > TIP_DEG or st["moved"] > MOVED_M)
    n = count_hung(st)
    if DIAG and _RUN["t"] is not None and _RUN["step"] % 5 == 0:
        print(f"BKLOG {_RUN['step']} {_RUN['t']:.2f} n={n} hung={st['hung']} saddle_on={st['saddle_on']} "
              f"moved_mm={st['moved'] / MM:.1f} tilt={st['tilt']:.1f} give_win_mm={bk.take_window():.2f} "
              f"{ {k: (round(v, 2) if isinstance(v, float) else v) for k, v in bk.last().items()} }", flush=True)
    if bucked or knocked:
        return 0.0
    return {3: 1.0, 2: 0.6, 1: 0.3}.get(n, 0.0)


def success(objects):
    return score(objects) >= 1.0


# ------------------------------------------------------------------ scripted demo
# Each piece: a top-down pinch across its body (pads closing along y), lifted over the hanger, its hole read and
# lined up over the post, lowered slowly, read again, let go about 2 mm above the seat; the hand opens only a little,
# rises straight up and leaves. The left hand hangs the pan; the right hand the canteen, then the rope.
SQUEEZE = 0.0025
TIP_BELOW = 0.015          # TCP above the piece's underside when pinched (fingertips end 2 mm above it)
CARRY_Z = 0.200            # TCP height (above the desk) while carrying
HOVER = 0.012              # piece underside this far above the post's top at the first read
LOW = 0.008                # piece underside this far above the seat at the second read
LET_GO = 0.002             # let go this far above the seat
PRESS = -0.008             # "press" mode: pushed this far into the seat before letting go
FIX_MAX = 0.010
PARK = {"left": np.array([0.12, 0.22, TOP + 0.16]), "right": np.array([0.12, -0.24, TOP + 0.16])}
_READS = {}


def _mode():
    if DEMO_MODE == "seq":
        ep = max(_RUN["episode"], 1)
        return DEMO_SEQ[(ep - 1) % len(DEMO_SEQ)]
    return DEMO_MODE


def _states(names):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        return arena_scene.object_states(env, list(names))
    except Exception as e:  # noqa: BLE001
        print(f"BK_READ failed: {e!r}", flush=True)
        return None


def _reads():
    if _READS.get("episode") != _RUN["episode"]:
        _READS.clear()
        _READS["episode"] = _RUN["episode"]
    return _READS


def _start(name):
    """(piece hole position on the desk, hanger post base (x, y, seat top z)) as found when first needed."""
    v = _reads()
    key = "start-" + name
    if key not in v:
        x, y = START[name]
        hole = np.array([x, y, TOP])
        hx, hy = HANGERS[PIECES[name]["hanger"]]
        post = np.array([XM + SADDLE_X + hx, YM + hy, TOP + SEAT_Z + SEAT_T])
        s = _states([name, "saddle"])
        if s is not None:
            o = np.asarray(s[name].origin, float)
            if np.linalg.norm(o[:2] - hole[:2]) < 0.03:
                hole = o
            pw = s["saddle"].to_world((hx, hy, SEAT_T))
            if np.linalg.norm(pw - post) < 0.03:
                post = pw
            print(f"BK_READ {name} hole=({hole[0]:.4f},{hole[1]:.4f},{hole[2] - TOP:.4f}) post=({post[0]:.4f},"
                  f"{post[1]:.4f},{post[2] - TOP:.4f}) mode={_mode()!r}", flush=True)
        v[key] = (hole, post)
    return v[key]


def _grip_offset(name):
    p = PIECES[name]
    return np.array([p["body"][0], p["body"][1], TIP_BELOW])


def _fix(name, key, target, prior):
    """Correction read once at `key`: where the hole should be minus where it is, added to the prior correction."""
    v = _reads()
    k = name + "-" + key
    if k not in v:
        corr = np.array(prior, float)
        s = _states([name])
        if s is not None and _RUN["t"] is not None:
            err = target - np.asarray(s[name].origin, float)
            n = float(np.linalg.norm(err))
            if n > FIX_MAX:
                err *= FIX_MAX / n
            corr = corr + err
            print(f"BK_FIX {name} {key} t={_RUN['t']:.2f} err_mm={np.round(err / MM, 1).tolist()} "
                  f"corr_mm={np.round(corr / MM, 1).tolist()} give_win_mm={bk.take_window():.2f}", flush=True)
        v[k] = corr
    return v[k]


class At(Waypoint):
    """A waypoint placed when first played: `where` gives the TCP position."""

    def __init__(self, duration, where, finger, label):
        self.duration, self.finger, self.label, self.yaw = duration, finger, label, 0.0
        self._where = where

    @property
    def pos(self):
        return tuple(float(x) for x in self._where())


def hang(arm, name):
    p = PIECES[name]
    half = p["body_r"]
    pinch, release = -(half - SQUEEZE), -(half + 0.004)
    g = _grip_offset(name)
    hole0 = lambda: _start(name)[0]                                          # noqa: E731
    post = lambda: _start(name)[1]                                           # noqa: E731
    hover_t = lambda: post() + np.array([0.0, 0.0, HANG_H + HOVER])          # noqa: E731
    low_t = lambda: post() + np.array([0.0, 0.0, LOW])                       # noqa: E731
    c1 = lambda: _fix(name, "hover", hover_t(), np.zeros(3))                 # noqa: E731
    c2 = lambda: _fix(name, "low", low_t(), c1())                            # noqa: E731
    final = lambda: post() + np.array([0.0, 0.0, PRESS if _mode() == "press" else LET_GO]) + g + c2()  # noqa: E731
    carry = lambda: np.array([hover_t()[0] + g[0], hover_t()[1] + g[1], TOP + CARRY_Z])  # noqa: E731
    lab = f"{arm}: {name}"
    side = 1.0 if arm == "left" else -1.0
    return [
        At(1.5, lambda: hole0() + g + np.array([0.0, 0.0, 0.06]), FINGER_OPEN, f"{lab}: above it, pads open"),
        At(1.0, lambda: hole0() + g, FINGER_OPEN, f"{lab}: pads either side of its body"),
        At(0.6, lambda: hole0() + g, pinch, f"{lab}: pinch"),
        At(1.0, lambda: np.array([hole0()[0] + g[0], hole0()[1] + g[1], TOP + CARRY_Z]), pinch, f"{lab}: lift"),
        At(1.8, carry, pinch, f"{lab}: carry over its hanger"),
        At(1.0, lambda: hover_t() + g, pinch, f"{lab}: hole over the post"),
        At(0.6, lambda: hover_t() + g + c1(), pinch, f"{lab}: line the hole up (first read)"),
        At(1.6, lambda: low_t() + g + c1(), pinch, f"{lab}: lower slowly over the post"),
        At(0.6, lambda: low_t() + g + c2(), pinch, f"{lab}: just above the seat (second read)"),
        At(0.8, final, pinch, f"{lab}: down to 2 mm above the seat"),
        At(0.5, final, release, f"{lab}: let go, pads open a little"),
        At(0.3, final, release, f"{lab}: hand still"),
        At(0.8, lambda: final() + np.array([0.0, 0.0, 0.025]), release, f"{lab}: straight up"),
        At(1.0, lambda: np.array([final()[0], side * 0.10, final()[2] + 0.025]), release, f"{lab}: out sideways"),
        Waypoint(1.0, tuple(PARK[arm]), FINGER_OPEN, f"{lab}: away"),
    ]


def _dur(ws):
    return sum(w.duration for w in ws)


def demo():
    r1, l1, r2 = hang("right", "canteen"), hang("left", "pan"), hang("right", "rope")
    right = r1 + [Waypoint(_dur(l1), tuple(PARK["right"]), FINGER_OPEN, "right: wait")] + r2
    left = [Waypoint(_dur(r1), tuple(PARK["left"]), FINGER_OPEN, "left: wait")] + l1
    tail = 3.0
    left.append(Waypoint(_dur(right) - _dur(left) + tail, tuple(PARK["left"]), FINGER_OPEN, "left: parked"))
    right.append(Waypoint(tail, tuple(PARK["right"]), FINGER_OPEN, "stillness tail"))
    return {"left": left, "right": right}


def demo_duration():
    return _dur(demo()["right"])
