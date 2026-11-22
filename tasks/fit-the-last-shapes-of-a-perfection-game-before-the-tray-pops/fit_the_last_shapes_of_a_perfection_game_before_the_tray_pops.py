"""The end of a game of Perfection against its clock. The game unit stands on the desk with its timer running: 21 of
its 25 yellow shapes already sit in the blue pop-up tray, and the last four (a square, a circle, a cross and a "hot
dog") lie on the desk beside the unit, two on each side. The robot fits each shape into its own hole, turned to fit,
and moves the timer's switch to STOP before the clock runs out; if it runs out first, the tray pops up and throws the
pieces out.

The unit is an articulated mechanism (yam_joints): a free base, the tray on a vertical slide (locked down; at time-out
its spring is released), the switch on a horizontal slide with a detent, and the timer dial on a hinge. The clock,
the switch rule and the pop are the state machine in perfection_kit.py; the tray's spring and the dial's drive are
forces and torques between the unit's own bodies, applied each physics step. A piece counts only where it physically
sits in its own hole; the clock stops only when the switch is physically moved to STOP.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import perfection_kit as pk  # noqa: E402
from yam_joints import Hinge, Link, Mechanism, Slide  # noqa: E402
from yam_parts import Part, box, cyl  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TEST_CLOCK = None          # offline tests: (stopped, popped, tray_held) as the game saw them
TEST_GRIPPERS = None
DIAG = False
DEMO_MODE = ""             # GPU-only copies: "swap" tries each piece in the other hole on its side, then stops the
                           # clock; "swapnostop" the same without the switch (the clock runs out); "seq" the probe
DEMO_SEQ = ("", "", "", "swapnostop")


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
_EVER = {}


def _episode_memory():
    global _MEMORY_EPISODE
    if _MEMORY_EPISODE != _RUN["episode"]:
        _EVER.clear()
        _MEMORY_EPISODE = _RUN["episode"]


def ever(key, cond):
    if _RUN["t"] is None:
        return bool(cond)
    _episode_memory()
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]


# ------------------------------------------------------------------ the game unit (x forward, away from the robot)
TOP = TABLE_TOP[SCENE]
MM = 0.001
# Unit 13.11 x 10.51 x 2.99 in (333 x 267 x 76 mm, retail listing), long side across the robot (y). Everything inside
# that outline (tray, pitch, heights, the timer housing's layout) is an estimate.
UX, UY = 0.200, -0.045                    # unit origin: middle of its underside, on the desk
LX, LY, LH = 0.267, 0.333, 0.076
RIM = 0.055                               # top of the unit's moulded frame round the tray
TRAY_C = (0.0, 0.045)                     # tray centre (unit frame)
TRAY_S = 0.225                            # tray side
PITCH = 0.042                             # 5 x 5 cells
TRAY_TOP = 0.052                          # tray top above the desk (locked down)
FLOOR = TRAY_TOP - pk.THICK               # pocket floor: a seated piece's underside
TRAY_BOT = 0.036
HOUSING_Y = (-LY / 2, TRAY_C[1] - TRAY_S / 2 - 0.006)     # the timer housing: the strip of the unit right of the tray
DIAL_C = (0.040, -0.118)                  # dial centre (unit frame)
DIAL_R = 0.034
SW_C = (-0.070, -0.118)                   # switch knob at START (unit frame); it slides +x to STOP
SW_TRAVEL = 0.012
SW_SIZE = (0.010, 0.014, 0.012)           # knob x, y, height above the housing
SW_STATIC, SW_DYNAMIC = 2.0, 1.5          # N: the switch's detent / slide friction
M_UNIT, M_TRAY, M_SWITCH, M_DIAL = 0.60, 0.12, 0.004, 0.020
RED, RED2, BLUE, BLUE2 = (0.80, 0.10, 0.08), (0.55, 0.06, 0.05), (0.10, 0.30, 0.82), (0.07, 0.20, 0.60)
YELLOW, WHITE, DARK = (0.98, 0.80, 0.08), (0.88, 0.82, 0.55), (0.08, 0.08, 0.09)


def cell_xy(row, col):
    """Cell centre (unit frame): row 0 nearest the robot, col 0 on the robot's right."""
    return (TRAY_C[0] + (row - 2) * PITCH, TRAY_C[1] + (col - 2) * PITCH)


# The four empty holes (row, col, shape, hole yaw deg) and the four loose pieces (shape, world x, y, yaw deg).
HOLES = {"square": (1, 4, 0.0), "circle": (2, 4, 0.0), "hotdog": (1, 0, 0.0), "cross": (2, 0, 0.0)}
PIECES0 = {"circle": (0.130, 0.185, 0.0), "square": (0.220, 0.185, 25.0),
           "cross": (0.270, -0.300, 30.0), "hotdog": (0.170, -0.300, -55.0)}
ARM_OF = {"square": "left", "circle": "left", "hotdog": "right", "cross": "right"}
GRIP_YAW0 = {"left": 0.0, "right": 0.0}
# The 21 shapes already in (decoration of the tray's top: a yellow outline flush with the tray and its knob).
SEATED_DECOR = ["circle", "square", "cross", "hotdog", "square", "cross", "circle", "hotdog", "cross", "circle",
                "hotdog", "square", "circle", "cross", "square", "hotdog", "circle", "square", "cross", "hotdog",
                "circle"]


def _shape_tops(shape, cx, cy, z0, z1, color, scale=1.0):
    out = []
    for c in pk.SHAPES[shape][0]:
        if c[0] == "box":
            _, x, y, sx, sy = c
            out.append(box((cx + x * scale, cy + y * scale, (z0 + z1) / 2), (sx * scale, sy * scale, z1 - z0), color))
        else:
            _, x, y, r = c
            out.append(cyl((cx + x * scale, cy + y * scale, (z0 + z1) / 2), r * scale, z1 - z0, "z", color))
    return out


def _tray_shapes():
    tx, ty = TRAY_C
    s = [box((tx, ty, (TRAY_BOT + FLOOR) / 2), (TRAY_S, TRAY_S, FLOOR - TRAY_BOT), BLUE2)]
    # border of the tray round the 5 x 5 cells
    b = (TRAY_S - 5 * PITCH) / 2
    for sgn in (-1, 1):
        s.append(box((tx + sgn * (TRAY_S - b) / 2, ty, (FLOOR + TRAY_TOP) / 2), (b, TRAY_S, TRAY_TOP - FLOOR), BLUE))
        s.append(box((tx, ty + sgn * (TRAY_S - b) / 2, (FLOOR + TRAY_TOP) / 2), (5 * PITCH, b, TRAY_TOP - FLOOR), BLUE))
    holes = {(r, c): shape for shape, (r, c, _) in HOLES.items()}
    k = 0
    for row in range(5):
        for col in range(5):
            cx, cy = cell_xy(row, col)
            if (row, col) in holes:
                for wx, wy, sx, sy in pk.hole_strips(holes[row, col], PITCH):
                    s.append(box((cx + wx, cy + wy, (FLOOR + TRAY_TOP) / 2), (sx, sy, TRAY_TOP - FLOOR), BLUE))
                continue
            s.append(box((cx, cy, (FLOOR + TRAY_TOP - 0.0004) / 2), (PITCH, PITCH, TRAY_TOP - 0.0004 - FLOOR), BLUE))
            shape = SEATED_DECOR[k % len(SEATED_DECOR)]
            k += 1
            s.extend(_shape_tops(shape, cx, cy, TRAY_TOP - 0.0006, TRAY_TOP, YELLOW))
            s.append(cyl((cx, cy, TRAY_TOP + pk.KNOB_H / 2), pk.KNOB_R, pk.KNOB_H, "z", YELLOW))
    return s


def _unit():
    tx, ty = TRAY_C
    hy0, hy1 = HOUSING_Y
    s = [box((0.0, ty, TRAY_BOT / 2), (LX, TRAY_S + 0.012, TRAY_BOT), RED),                  # body under the tray
         box((0.0, (hy0 + hy1) / 2, LH / 2), (LX, hy1 - hy0, LH), RED)]                      # timer housing
    # the frame round the tray (up to the rim)
    for sgn in (-1, 1):
        s.append(box((sgn * (LX / 2 - (LX - TRAY_S) / 4), ty, (TRAY_BOT + RIM) / 2),
                     ((LX - TRAY_S) / 2, TRAY_S + 0.012, RIM - TRAY_BOT), RED))
    s.append(box((0.0, LY / 2 - 0.003, (TRAY_BOT + RIM) / 2), (TRAY_S, 0.006, RIM - TRAY_BOT), RED))
    for fx in (-1, 1):
        for fy in (-1, 1):
            s.append(cyl((fx * 0.11, fy * 0.14, 0.0015), 0.010, 0.003, "z", DARK))            # feet (decor)
    # dial face with its 0-60 marks, the switch slot and its START / STOP marks
    dx, dy = DIAL_C
    s.append(cyl((dx, dy, LH + 0.0005), DIAL_R + 0.004, 0.001, "z", WHITE))
    for k in range(12):
        a = -2 * math.pi * k / 12
        s.append(box((dx + (DIAL_R + 0.001) * math.cos(a), dy + (DIAL_R + 0.001) * math.sin(a), LH + 0.0012),
                     (0.003, 0.003, 0.0004), DARK))
    sx, sy = SW_C
    s.append(box((sx + SW_TRAVEL / 2, sy, LH + 0.0003), (SW_TRAVEL + SW_SIZE[0] + 0.006, 0.006, 0.0006), DARK))
    s.append(box((sx - SW_SIZE[0] / 2 - 0.006, sy, LH + 0.0004), (0.004, 0.012, 0.0008), (0.10, 0.65, 0.20)))  # START
    s.append(box((sx + SW_TRAVEL + SW_SIZE[0] / 2 + 0.006, sy, LH + 0.0004), (0.004, 0.012, 0.0008), RED2))  # STOP
    links = [
        Link("tray", _tray_shapes(), Slide("z", (tx, ty, TRAY_BOT), 0.0, pk.POP_STROKE, damping=0.5),
             M_TRAY, color=BLUE, friction=(0.6, 0.5)),
        Link("switch", [box((sx, sy, LH + SW_SIZE[2] / 2), SW_SIZE, YELLOW)],
             Slide("x", (sx, sy, LH), 0.0, SW_TRAVEL, damping=0.5, static_friction=SW_STATIC,
                   dynamic_friction=SW_DYNAMIC), M_SWITCH, color=YELLOW, friction=(0.9, 0.8)),
        Link("dial", [cyl((dx, dy, LH + 0.003), DIAL_R, 0.004, "z", WHITE),
                      box((dx + DIAL_R * 0.45, dy, LH + 0.0055), (DIAL_R * 0.9, 0.004, 0.001), RED),
                      cyl((dx, dy, LH + 0.006), 0.006, 0.002, "z", RED2)],
             Hinge("z", (dx, dy, LH), -360.0, 10.0, damping=0.002, initial=-360.0 * _dial0()), M_DIAL,
             color=WHITE),
    ]
    return Mechanism("game", (UX, UY, TOP), s, links, mass=M_UNIT, color=RED, friction=(1.0, 0.9),
                     contact_offset=0.001)


def _dial0():
    return (60.0 - pk.T_LEFT) / 60.0


def piece_mass(shape):
    """ABS (1,050 kg/m3), solid: 3.4-5.6 g (estimate; kitchen scale)."""
    area = 0.0
    for c in pk.SHAPES[shape][0]:
        area += c[3] * c[4] if c[0] == "box" else math.pi * c[3] ** 2
    if shape == "cross":
        area -= 0.012 * 0.012
    if shape == "hotdog":
        area -= math.pi * 0.008 ** 2
    return 1050.0 * (area * pk.THICK + math.pi * pk.KNOB_R ** 2 * pk.KNOB_H)


def _piece(shape):
    x, y, yaw = PIECES0[shape]
    s = _shape_tops(shape, 0.0, 0.0, 0.0, pk.THICK, YELLOW)
    s.append(cyl((0.0, 0.0, pk.THICK + pk.KNOB_H / 2), pk.KNOB_R, pk.KNOB_H, "z", YELLOW))
    return Part("p_" + shape, (x, y, TOP), s, mass=piece_mass(shape), color=YELLOW, yaw=math.radians(yaw), friction=(0.6, 0.5),
                contact_offset=0.001, grip=len(s) - 1)


HOLE_W = {s: (UX + cell_xy(r, c)[0], UY + cell_xy(r, c)[1]) for s, (r, c, _) in HOLES.items()}   # world, unit at rest
UNIT = _unit()
PIECES = {shape: _piece(shape) for shape in HOLES}
OBJECTS = [UNIT] + list(PIECES.values())
CLOCK = pk.Clock()


# ------------------------------------------------------------------ what a person checks
SEAT_XY, SEAT_Z, SEAT_TILT, SEAT_YAW = 0.0015, 0.0015, 5.0, 5.0
MOVED_M, TILT_DEG = 0.030, 10.0
HELD_RISE, HELD_S = 0.015, 1.0            # after the pop, the tray must be up 15 mm within 1 s, or it was held down


def hole_world(game_state, shape):
    """(world position of the hole's floor centre, hole yaw deg in the world) from the tray link's reported pose."""
    row, col, yaw = HOLES[shape]
    cx, cy = cell_xy(row, col)
    tx, ty = TRAY_C
    p = game_state.link_to_world("tray", np.array([cx, cy, FLOOR]))
    R = game_state.link_rotation("tray")
    return p, math.degrees(math.atan2(R[1, 0], R[0, 0])) + yaw


def piece_yaw_deg(st):
    R = st.rotation()
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def seat_error(objects, shape):
    """(dxy m, dz m, tilt deg, yaw deg) of a piece against its own hole, in the tray's frame."""
    g, p = objects["game"], objects["p_" + shape]
    hp, hyaw = hole_world(g, shape)
    R = g.link_rotation("tray")
    d = R.T @ (np.asarray(p.origin, float) - hp)
    tilt = math.degrees(math.acos(float(np.clip((R.T @ p.rotation())[2, 2], -1.0, 1.0))))
    return float(np.hypot(d[0], d[1])), float(d[2]), tilt, abs(pk.yaw_error_deg(piece_yaw_deg(p), hyaw, shape))


def seated(objects, shape):
    dxy, dz, tilt, yaw = seat_error(objects, shape)
    return dxy <= SEAT_XY and abs(dz) <= SEAT_Z and tilt <= SEAT_TILT and yaw <= SEAT_YAW


def seated_set(objects):
    return {s for s in HOLES if seated(objects, s)}


# ------------------------------------------------------------------ the tray's spring and the dial's drive (GPU only)
_HOOK = {"err": None, "n": 0}
DIAL_K = 0.02                              # N m/rad toward the dial's position for the clock's time


def install_hooks(env, name="game", log=print):
    """Wrap the unit articulation's write_data_to_sim (before every physics substep): once the clock has popped, the
    tray's spring pushes the tray up and the unit's base down (pk.pop_force); the dial is turned toward the clock's
    time with an equal and opposite torque on the base. Nothing here acts on the robot or the pieces."""
    if env is None:
        return False
    try:
        art = env.unwrapped.scene.articulations[name]
    except Exception as e:  # noqa: BLE001
        if _HOOK["err"] is None:
            _HOOK["err"] = repr(e)
            log(f"PF_HOOK no articulation: {e!r}")
        return False
    if getattr(art, "_pf_hooked", False):
        return True
    import torch
    bodies = list(art.body_names)
    joints = list(art.joint_names)
    ib, it, idl = (bodies.index(n) for n in ("base", "tray", "dial"))
    jt, jd = joints.index("tray_joint"), joints.index("dial_joint")
    original = art.write_data_to_sim

    def _np(x):
        x = getattr(x, "torch", x)
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x, float)

    def write_data(*args, **kwargs):
        try:
            pose = _np(art.data.body_link_pose_w)[0]
            q = _np(art.data.joint_pos)[0]
            x, y, z, w = pose[ib, 3:7]
            zb = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])
            F = np.zeros((3, 3))
            T = np.zeros((3, 3))
            if CLOCK.popped:
                f = pk.pop_force(float(q[jt])) * zb
                F[1], F[0] = f, -f
            target = -2 * math.pi * CLOCK.dial_used_s() / 60.0
            tau = DIAL_K * (target - float(q[jd]))
            T[2], T[0] = tau * zb, -tau * zb
            ids = [ib, it, idl]
            Ft = torch.tensor(F[None], dtype=torch.float32, device=art.device)
            Tt = torch.tensor(T[None], dtype=torch.float32, device=art.device)
            art.permanent_wrench_composer.set_forces_and_torques_index(forces=Ft, torques=Tt, body_ids=ids,
                                                                       is_global=True)
            if _HOOK["n"] < 2:
                _HOOK["n"] += 1
                log(f"PF_HOOK ok tray_q={q[jt]:.4f} dial={q[jd]:.3f} popped={CLOCK.popped}")
        except Exception as e:  # noqa: BLE001
            if _HOOK["err"] is None:
                _HOOK["err"] = repr(e)
                log(f"PF_HOOK error: {e!r}")
        return original(*args, **kwargs)

    art.write_data_to_sim = write_data
    art._pf_hooked = True
    log(f"PF_HOOK installed on {name}: bodies {bodies} joints {joints}")
    return True


_GAME_EP = {"episode": None, "t": None}


def play(objects):
    """Advance the clock one control step (once per step, whatever calls it) and return it."""
    if _RUN["t"] is None:
        return CLOCK
    if _GAME_EP["episode"] != _RUN["episode"]:
        CLOCK.reset()
        g = objects["game"]
        _GAME_EP.update(episode=_RUN["episode"], t=None, origin0=np.asarray(g.origin, float).copy(), tray_max=0.0)
        install_hooks(_RUN["env"])
    t = float(_RUN["t"])
    if _GAME_EP["t"] is not None and t <= _GAME_EP["t"]:
        return CLOCK
    _GAME_EP["t"] = t
    g = objects["game"]
    CLOCK.update(t, float(g.joints["switch"]), seated=lambda: seated_set(objects))
    if CLOCK.popped:
        _GAME_EP["tray_max"] = max(_GAME_EP["tray_max"], float(g.joints["tray"]))
    return CLOCK


def status(objects):
    g = objects["game"]
    clock = play(objects)
    now = seated_set(objects)
    if _RUN["t"] is None:
        stopped, popped, held = TEST_CLOCK or (False, False, False)
        at_stop = now if stopped else set()
        at_pop = now if popped else set()
        origin0 = np.array([UX, UY, TOP])
    else:
        stopped, popped = clock.stopped, clock.popped
        at_stop, at_pop = set(clock.seated_at_stop), set(clock.seated_at_pop)
        held = popped and _RUN["t"] - clock.t_pop >= HELD_S and _GAME_EP["tray_max"] < HELD_RISE
        origin0 = _GAME_EP["origin0"]
    moved = float(np.linalg.norm(np.asarray(g.origin, float)[:2] - origin0[:2]))
    return dict(stopped=stopped, popped=popped, held=held, now=now, at_stop=at_stop, at_pop=at_pop, moved=moved,
                tilt=g.tilt_deg())


def count_tier(n):
    return {4: 1.0, 3: 0.75, 2: 0.5, 1: 0.25}.get(n, 0.0)


POP_ALL_IN = 0.25


def score(objects):
    st = status(objects)
    shoved = ever("game unit shoved or tipped", st["moved"] > MOVED_M or st["tilt"] > TILT_DEG)
    held = ever("tray held down when it popped", st["held"])
    if DIAG and _RUN["t"] is not None and _RUN["step"] % 10 == 0:
        errs = {s: tuple(round(v, 1) for v in (e[0] / MM, e[1] / MM, e[2], e[3]))
                for s, e in ((s, seat_error(objects, s)) for s in HOLES)}
        g = objects["game"]
        print(f"PFLOG {_RUN['step']} {_RUN['t']:.2f} left={CLOCK.t_left:.2f} stopped={CLOCK.stopped} "
              f"popped={CLOCK.popped} sw_mm={g.joints['switch'] / MM:.1f} tray_mm={g.joints['tray'] / MM:.1f} "
              f"dial={math.degrees(g.joints['dial']):.0f} seated={sorted(st['now'])} err(dxy,dz,tilt,yaw)={errs} "
              f"moved_mm={st['moved'] / MM:.1f} hook_err={_HOOK['err']} events={CLOCK.events[-2:]} {_hand_diag()}", flush=True)
    if shoved or held:
        return 0.0
    if st["popped"]:
        return POP_ALL_IN if len(st["at_pop"]) == len(HOLES) else 0.0
    if not st["stopped"]:
        return 0.0
    return count_tier(len(st["at_stop"] & st["now"]))


def _hand_diag():
    try:
        out = []
        for arm, p in gripper_points().items():
            out.append(f"{arm}=({p[0][0]:.3f},{p[0][1]:.3f},{p[0][2] - TOP:.3f}) gap={np.linalg.norm(p[1] - p[2]) / MM:.1f}")
        return " ".join(out)
    except Exception as e:  # noqa: BLE001
        return f"hand?{e!r}"


def success(objects):
    return score(objects) >= 1.0


# ------------------------------------------------------------------ scripted demo
# Each arm fits the two pieces on its side (left: square and circle; right: hot dog and cross), one at a time: a
# top-down pinch on the piece's knob, carried over its own hole turned to fit; at a hover 3 mm above the tray the hand
# re-reads where the piece really is and corrects its aim twice at 16 mm above the tray and three times at 1 mm,
# then lowers the piece into the hole and lets go. Then the right hand pushes the timer's switch to STOP with its closed fingertips. The demo reads the pieces'
# and the unit's poses (as a person sees them), never the game's clock.
KNOB_HALF_GAP = 0.0025     # commanded half gap on the 8 mm knob (3 mm inside it)
OPEN_HALF = 0.010          # opened only 20 mm: the neighbours' knobs are 38 mm away
TIP_ABOVE = 0.0015         # fingertips this far above the piece's top when pinching
TCP_PAST_TIP = 0.013
GRASP_DZ = pk.THICK + TIP_ABOVE + TCP_PAST_TIP          # TCP above the piece's underside in the pinch
CARRY = 0.040              # piece underside this far above the tray top while carried (knobs are 11 mm)
HOVER = 0.001              # last aims with the underside 1 mm above the tray top
GAIN = 0.8                 # each aim corrects this share of the error read (the arm still lags a little)
AIM_H = 0.016              # first aims with the underside 16 mm above the tray, clear of the neighbours' 11 mm knobs
RELEASE = 0.001            # let go with the underside this far above the pocket floor
PARK = {"left": np.array([0.10, 0.25, TOP + 0.15]), "right": np.array([0.10, -0.25, TOP + 0.15])}
RIGHT_DELAY = 4.0
ORDER = {"left": ["square", "circle"], "right": ["cross", "hotdog"]}   # never both hands at the same row
SWAP = {"square": "circle", "circle": "square", "hotdog": "cross", "cross": "hotdog"}
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
        print(f"PF_READ failed: {e!r}", flush=True)
        return None


def _reads():
    if _READS.get("episode") != _RUN["episode"]:
        _READS.clear()
        _READS["episode"] = _RUN["episode"]
    return _READS


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


def _piece_now(shape):
    """(position of the piece's underside centre, yaw deg), read now; the planned layout offline."""
    s = _states(["p_" + shape])
    if s is None:
        x, y, yaw = PIECES0[shape]
        return np.array([x, y, TOP]), yaw
    st = s["p_" + shape]
    return np.asarray(st.origin, float), piece_yaw_deg(st)


def _hole_now(shape):
    s = _states(["game"])
    if s is None:
        row, col, yaw = HOLES[shape]
        cx, cy = cell_xy(row, col)
        return np.array([UX + cx, UY + cy, TOP + FLOOR]), yaw
    return hole_world(s["game"], shape)


def _tcp_now(arm):
    pts = gripper_points()
    return None if not pts else np.asarray(pts[arm][0], float)


class At(Waypoint):
    """A waypoint whose TCP position and grip yaw are fixed when it is first played."""

    def __init__(self, duration, where, finger, label, yaw=None):
        self.duration, self.finger, self.label = duration, finger, label
        self._where, self._yaw_fn = where, yaw
        self._pos = self._yaw = None
        self._episode = None

    def _fix(self):
        if self._pos is None or self._episode != _RUN["episode"]:
            self._episode = _RUN["episode"]
            self._pos = tuple(float(v) for v in self._where())
            self._yaw = float(self._yaw_fn()) if self._yaw_fn is not None else _reads().get(self._arm_key(), 0.0)

    def _arm_key(self):
        return "yaw-" + self.label.split(":")[0]

    @property
    def pos(self):
        self._fix()
        return self._pos

    @property
    def yaw(self):
        self._fix()
        return self._yaw


def _piece_plan(arm, shape):
    """Waypoints for one piece: pick `shape`, fit it into its own hole ("swap" copies: the other hole on its side)."""
    v = _reads
    key = f"{arm}-{shape}"
    into = lambda: SWAP[shape] if _mode().startswith("swap") else shape  # noqa: E731
    pinch = -KNOB_HALF_GAP
    open_ = -OPEN_HALF

    def start():
        d = v()
        if key + "-p0" not in d:
            p, yaw = _piece_now(shape)
            d[key + "-p0"], d[key + "-yaw0"] = p, yaw
        return d[key + "-p0"]

    def above_piece():
        p = start()
        return np.array([p[0], p[1], TOP + TRAY_TOP + CARRY + GRASP_DZ])

    def at_piece():
        return start() + np.array([0.0, 0.0, GRASP_DZ])

    def lifted():
        p = start()
        return np.array([p[0], p[1], TOP + TRAY_TOP + CARRY + GRASP_DZ])

    def grip_yaw0():
        v()["yaw-" + arm] = GRIP_YAW0[arm]
        return GRIP_YAW0[arm]

    def turn():
        """Grip yaw (rad) that turns the held piece square to its target hole (smallest turn for its symmetry)."""
        d = v()
        _, hyaw = _hole_now(into())
        err = pk.yaw_error_deg(d[key + "-yaw0"], hyaw, shape)
        sym = pk.SHAPES[shape][1]
        ks = range(-2, 3) if sym > 0 else (0,)
        y = min((GRIP_YAW0[arm] - math.radians(err - k * sym) for k in ks), key=abs)
        d["yaw-" + arm] = y
        return y

    def over_hole():
        hp, _ = _hole_now(into())
        d = v()
        d[key + "-h"] = CARRY
        d[key + "-cmd"] = np.array([hp[0], hp[1], TOP + TRAY_TOP + CARRY + GRASP_DZ])
        return d[key + "-cmd"]

    def hover(h):
        def f():
            d = v()
            c = d[key + "-cmd"].copy()
            c[2] += (TOP + TRAY_TOP + h) - (d.get(key + "-h", CARRY) + TOP + TRAY_TOP)
            d[key + "-h"] = h
            d[key + "-cmd"] = c
            return c
        return f

    def correct(k):
        def f():
            d = v()
            h = d.get(key + "-h", HOVER)
            hp, hyaw = _hole_now(into())
            p, pyaw = _piece_now(shape)
            c = d[key + "-cmd"].copy()
            tcp = _tcp_now(arm)
            if _RUN["env"] is not None:
                c[:2] += GAIN * (hp - p)[:2]
                c[2] += (TOP + TRAY_TOP + h) - p[2]
                err = pk.yaw_error_deg(pyaw, hyaw, shape)
                d["yaw-" + arm] = d.get("yaw-" + arm, 0.0) - GAIN * math.radians(err)
                if DIAG:
                    off = None if tcp is None else np.round((p - tcp) / MM, 1).tolist()
                    print(f"PF_AIM {arm} {shape}->{into()} k={k} h_mm={h / MM:.0f} dxy_mm="
                          f"{np.round((hp - p)[:2] / MM, 2).tolist()} yaw_err={err:.2f} piece_z_mm="
                          f"{(p[2] - TOP) / MM:.1f} piece_minus_tcp_mm={off}", flush=True)
            d[key + "-cmd"] = c
            return c
        return f

    def down():
        d = v()
        c = d[key + "-cmd"].copy()
        c[2] -= d.get(key + "-h", HOVER) + pk.THICK - RELEASE
        return c

    def up():
        d = v()
        c = d[key + "-cmd"].copy()
        c[2] = TOP + TRAY_TOP + CARRY + GRASP_DZ
        return c

    yaw_now = lambda: v().get("yaw-" + arm, 0.0)  # noqa: E731
    lab = f"{arm}: {shape}"
    return [
        At(1.2, above_piece, open_, f"{lab}: over the piece", grip_yaw0),
        At(0.8, at_piece, open_, f"{lab}: pads either side of the knob", yaw_now),
        At(0.5, at_piece, pinch, f"{lab}: pinch the knob", yaw_now),
        At(0.2, at_piece, pinch, f"{lab}: hold the pinch", yaw_now),
        At(0.5, lifted, pinch, f"{lab}: lift", yaw_now),
        At(1.0, over_hole, pinch, f"{lab}: over its hole, turned to fit", turn),
        At(0.5, hover(AIM_H), pinch, f"{lab}: down to 16 mm above the tray", yaw_now),
        At(0.6, correct(1), pinch, f"{lab}: aim 1", yaw_now),
        At(0.6, correct(2), pinch, f"{lab}: aim 2", yaw_now),
        At(0.5, hover(HOVER), pinch, f"{lab}: down to 1 mm above the tray", yaw_now),
        At(0.7, correct(3), pinch, f"{lab}: aim 3", yaw_now),
        At(0.7, correct(4), pinch, f"{lab}: aim 4", yaw_now),
        At(0.7, correct(5), pinch, f"{lab}: aim 5", yaw_now),
        At(0.6, down, pinch, f"{lab}: lower into the hole", yaw_now),
        At(0.3, down, open_, f"{lab}: let go", yaw_now),
        At(0.4, up, open_, f"{lab}: lift clear", yaw_now),
    ]


SW_TIP_Z = 0.004           # fingertips this far above the housing's top when pushing the switch
SW_LEAD = 0.015            # commanded this far past the switch's end stop (the arm's spring gives the push)


def _switch_plan():
    def knob():
        s = _states(["game"])
        if s is None:
            return np.array([UX + SW_C[0], UY + SW_C[1], TOP + LH])
        g = s["game"]
        return g.link_to_world("switch", np.array([SW_C[0], SW_C[1], LH]))

    def behind(dz):
        def f():
            k = knob()
            return k + np.array([-(SW_SIZE[0] / 2 + 0.0075 + 0.004), 0.0, SW_TIP_Z + TCP_PAST_TIP + dz])
        return f

    def pushed():
        k = _reads().setdefault("sw0", knob())
        lead = 0.0 if _mode() == "swapnostop" else SW_TRAVEL + SW_LEAD
        return k + np.array([lead - (SW_SIZE[0] / 2 + 0.0075 + 0.004), 0.0, SW_TIP_Z + TCP_PAST_TIP])

    z0 = lambda: 0.0  # noqa: E731
    return [
        At(1.0, behind(0.03), 0.0, "right: switch: over the switch, behind it", z0),
        At(0.5, behind(0.0), 0.0, "right: switch: fingertips down behind the knob", z0),
        At(0.8, pushed, 0.0, "right: switch: push it to STOP", z0),
        At(0.4, lambda: pushed() + np.array([0.0, 0.0, 0.04]), 0.0, "right: switch: lift clear", z0),
    ]


def demo():
    plans = {}
    for arm in ("left", "right"):
        wps = []
        if arm == "right":    # half a cycle behind the left hand: the two never reach in over the tray together
            first = PIECES0[ORDER["right"][0]]
            wps.append(At(RIGHT_DELAY, lambda: np.array([first[0], first[1], TOP + TRAY_TOP + CARRY + GRASP_DZ + 0.03]),
                          -OPEN_HALF, "right: wait half a cycle above its first shape", lambda: 0.0))
        for shape in ORDER[arm]:
            wps += _piece_plan(arm, shape)
        plans[arm] = wps
    tl = sum(w.duration for w in plans["left"])
    tr = sum(w.duration for w in plans["right"])
    wait = max(0.0, tl - tr)
    hold_r = plans["right"][-1]
    plans["right"].append(At(wait + 0.2, lambda: np.asarray(hold_r.pos), -OPEN_HALF, "right: wait for the left hand",
                             lambda: 0.0))
    plans["right"] += _switch_plan()
    tr = sum(w.duration for w in plans["right"])
    plans["left"].append(At(1.0, lambda: PARK["left"], -OPEN_HALF, "left: park", lambda: 0.0))
    plans["left"].append(Waypoint(max(0.5, tr - tl - 1.0), tuple(PARK["left"]), -OPEN_HALF, "left: parked"))
    plans["right"].append(At(1.0, lambda: PARK["right"], 0.0, "right: park", lambda: 0.0))
    # both arms stay parked until 6 s after the clock would have run out, so the run shows the clock stopped (or the pop)
    for arm in ("left", "right"):
        rest = pk.T_LEFT + 6.0 - sum(w.duration for w in plans[arm])
        if rest > 0:
            plans[arm].append(Waypoint(rest, tuple(PARK[arm]), -OPEN_HALF if arm == "left" else 0.0, f"{arm}: parked"))
    return plans


def demo_duration():
    d = demo()
    return max(sum(w.duration for w in d["left"]), sum(w.duration for w in d["right"]))
