"""Buy a gumball from a countertop coin gumball machine: drop a quarter into the coin slot, turn the handle a full turn
(two half turns with a re-grip), then lift the exit door so the gumball rolls out onto the table.

The machine (a VEVOR 16-inch class 25-cent gumball machine, 215 x 200 x 410 mm) and its parts are modelled in
gumball_machine.py. Metres; x forward from the arm bases, y left, z up. The machine stands on the table (fixed) with its
coin mechanism facing the robot; the quarter lies on a rubber mat to its right.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gumball_machine as GM  # noqa: E402
from yam_parts import Part, box  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
DIAG = False  # GPU diagnostic copies print GB_DIAG lines
DEMO_WAY = "full"  # "full"; "nocoin": wrong way, the handle is turned without putting the coin in
DEMO_PACE = 1.0
PROBE_OFFSET = None  # probe copies: per-episode offsets [(coin pick dx, dy, bar grip dy, dz) mm, ...]
TEST_GRIPPERS = None

TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S, SCORE_OVER = 1.0, "end"
RUN_LIMIT_STEPS = 2400  # 120 s at 20 Hz
MM = 0.001


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

# ---------------------------------------------------------------- layout
MACHINE_XY = np.array([0.332, -0.060])
MAT_XY = np.array([0.150, -0.175])
MAT_SIZE, MAT_T, MAT_MASS = 0.110, 0.002, 0.035
RUBBER = (0.08, 0.08, 0.09)

MACHINE = GM.machine("gumball_machine", (MACHINE_XY[0], MACHINE_XY[1], TOP))
_MPOS = np.array(MACHINE.pos, float)
COIN = GM.quarter("quarter", (MAT_XY[0], MAT_XY[1], TOP + MAT_T + GM.COIN_T * MM / 2))
GUM_1 = GM.gumball("gumball_1", tuple(_MPOS + GM.pocket_ball_local()), GM.GUM_COLORS[2])
GUM_2 = GM.gumball("gumball_2", tuple(_MPOS + GM.tube_ball_local()), GM.GUM_COLORS[3])
MAT = Part("mat", (MAT_XY[0], MAT_XY[1], TOP + MAT_T / 2), (box((0, 0, 0), (MAT_SIZE, MAT_SIZE, MAT_T)),),
           mass=MAT_MASS, color=RUBBER, friction=(1.0, 0.9), contact_offset=0.001)
OBJECTS = [MACHINE, COIN, GUM_1, GUM_2, MAT]

# ---------------------------------------------------------------- scoring
UPRIGHT_DEG = 10.0
QUARTER_TURN = 90.0


def _tilt_deg(state):
    R = state.rotation()
    return math.degrees(math.acos(float(np.clip(R[2, 2], -1.0, 1.0))))


def score_terms(objects):
    m, coin = objects["gumball_machine"], objects["quarter"]
    balls = [objects["gumball_1"], objects["gumball_2"]]
    c = np.asarray(coin.origin, float)
    places = [GM.ball_place(m, b.origin) for b in balls]
    out = [b for b, pl in zip(balls, places) if pl != "held"]
    t = {
        "upright": _tilt_deg(m) < UPRIGHT_DEG,
        "coin_in": GM.coin_inside(m, c),
        "seated": GM.coin_in_channel(m, c),
        "turn_deg": GM.dial_turn_deg(m),
        "places": places,
        "n_out": len(out),
        "through_door": bool(len(out) == 1 and places.count("free") == 1),
    }
    if DIAG:
        _diag(objects, t)
    return t


def score(objects):
    t = score_terms(objects)
    if not t["upright"] or not t["coin_in"] or t["n_out"] > 1:
        return 0.0  # tipped, or no coin paid in, or more than one gumball out: no purchase
    if t["n_out"] == 1:
        return 1.0 if t["through_door"] else 0.75
    if t["turn_deg"] >= QUARTER_TURN:
        return 0.5
    return 0.25


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- demo (right arm; the left arm stays at rest)
# Coin: the hand comes straight down over the quarter on the mat, pads across its diameter, pinches its rim 1.25 mm
# inside each side, lifts it and pitches level (pointing at the machine), so the coin hangs at the fingertips' ends,
# face to the machine. It reads where the coin really is in its hand, brings it over the slot, lowers it until the
# fingertips are just over the coin mechanism's top (the coin's lower 3-4 mm in the slot's mouth) and lets go: the
# quarter drops edge-down into the dial's coin channel. Handle: the level hand, turned so its fingers close top and
# bottom on the T-bar, grips it and turns the wrist half a turn clockwise; opens, backs off, turns back, grips again
# and turns the second half. Door: the hand, almost level (5 degrees down) with its fingers closed, slides its
# fingertip in under the door's lip, presses up and lifts it along its arc to ~55 degrees; the gumball rolls out
# under it onto the table.
PAD_TIP = 0.013
OPEN_WIDE = -0.017  # pads 34 mm apart
COIN_PINCH = -(GM.COIN_R * 2 - 2.5) / 2 * MM  # pads sent 2.5 mm inside the quarter's 24.26 mm diameter
COIN_OPEN = -0.016
BAR_OPEN = -0.013
BAR_PINCH = -(2 * GM.BAR[3] - 2.5) / 2 * MM  # 2.5 mm inside the bar's 13 mm height
CLOSED = 0.0
TIP_HALF_W = 0.0077  # half the pads' width over the first 10 mm (gripper profile)
COIN_OVER_MOUTH = 0.006
MOUTH_TOP = GM.PLATE_Z1 * MM
FINGER_CLEAR = 0.0009  # fingertips this far over the mechanism's top at let-go
PARK = np.array([0.18, -0.30, TOP + 0.20])
FINAL_HOLD_S = 2.5
DOOR_OPEN_DEG = 55.0
DOOR_UNDER = 0.0025  # the fingertip's top edge this far under the hem while it slides in
DOOR_PRESS = 0.003  # the lift targets 3 mm higher than the hem (measured sag of the hand at the door: 3 mm)
HEM_ON_TIP = 0.005  # the hem is planned 5 mm up the fingertip from its axis (its top edge is at 7.7 mm: it lifts 2.3 mm more)
_READS = {"episode": None, "values": {}}


def _rot(axis, a):
    a_ = np.asarray(axis, float)
    K = np.array([[0, -a_[2], a_[1]], [a_[2], 0, -a_[0]], [-a_[1], a_[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def _down():
    from yam_arena.demo import grip_rotation
    return getattr(grip_rotation, "__wrapped__", grip_rotation)(0.0)  # top-down, fingers closing along world y


def hand_rot(pitch_deg=0.0, roll_deg=0.0):
    """pitch 0: straight down; -90: level, pointing +x (at the machine); then rolled about world x (clockwise from
    the front is positive)."""
    return _rot((1, 0, 0), math.radians(roll_deg)) @ _rot((0, 1, 0), math.radians(pitch_deg)) @ _down()


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
    def grip_rotation(yaw=0.0):  # noqa: E306
        matrix = getattr(yaw, "grip_matrix", None)
        return matrix.copy() if matrix is not None else original(float(yaw))
    grip_rotation._grip_matrix = True
    grip_rotation.__wrapped__ = original
    arena_demo.grip_rotation = grip_rotation


_install_grip_matrix()


def _objects_now(names):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        return arena_scene.object_states(env, list(names))
    except Exception as e:  # noqa: BLE001
        print(f"GB_READ failed: {e!r}", flush=True)
        return None


def _values():
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    return _READS["values"]


def _mm(p):
    return tuple(round(float(x) * 1000, 1) for x in p)


def machine_point(p_local):
    return _MPOS + np.asarray(p_local, float)


def machine_now():
    """The machine's position read once per episode (it is fixed to the table; read anyway, as a robot would look)."""
    v = _values()
    if "machine" not in v:
        st = _objects_now(["gumball_machine"])
        v["machine"] = _MPOS.copy() if st is None else np.asarray(st["gumball_machine"].origin, float)
        if st is not None:
            print(f"GB_READ machine t={_RUN['t']:.2f} pos={_mm(v['machine'])}", flush=True)
    return v["machine"]


def mpoint(x_mm, y_mm, z_mm):
    return machine_now() + np.array([x_mm, y_mm, z_mm]) * MM


def coin_start():
    v = _values()
    if "coin0" not in v:
        st = _objects_now(["quarter"])
        v["coin0"] = np.asarray(COIN.pos, float) if st is None else np.asarray(st["quarter"].origin, float)
        if st is not None:
            print(f"GB_READ coin0 t={_RUN['t']:.2f} pos={_mm(v['coin0'])}", flush=True)
    return v["coin0"]


def coin_in_hand(tcp_cmd):
    """Coin centre minus the commanded hand centre, read once while the hand holds still (offline: the nominal)."""
    v = _values()
    if "coin_off" not in v:
        st = _objects_now(["quarter"])
        nominal = np.array([PAD_TIP - GM.COIN_T * MM / 2 + 0.0005, 0.0, 0.0])
        if st is None:
            v["coin_off"] = nominal
        else:
            off = np.asarray(st["quarter"].origin, float) - np.asarray(tcp_cmd, float)
            print(f"GB_READ coin_off t={_RUN['t']:.2f} off_mm={_mm(off)}", flush=True)
            v["coin_off"] = off if np.linalg.norm(off - nominal) < 0.015 else nominal
    return v["coin_off"]


def dial_now(tag):
    v = _values()
    key = f"dial_{tag}"
    if key not in v:
        st = _objects_now(["gumball_machine"])
        v[key] = None if st is None else GM.dial_turn_deg(st["gumball_machine"])
        if st is not None:
            print(f"GB_READ {key} t={_RUN['t']:.2f} turn_deg={v[key]:.1f}", flush=True)
    return v[key]


LIFT_Z = TOP + 0.080
CARRY = None


def _probe_off():
    """This episode's probe offsets in metres (coin pick dx, dy; bar grip dy, dz); zeros in the task itself."""
    if not PROBE_OFFSET:
        return np.zeros(4)
    return np.asarray(PROBE_OFFSET[(max(_RUN["episode"], 1) - 1) % len(PROBE_OFFSET)], float) * MM


def coin_targets(kind):
    c0 = coin_start() + np.r_[_probe_off()[:2], 0.0]
    if kind == "over":
        return np.array([c0[0], c0[1], LIFT_Z]), hand_rot()
    pad_z = TOP + MAT_T + PAD_TIP
    if kind == "low":
        return np.array([c0[0], c0[1], pad_z + 0.008]), hand_rot()
    if kind == "pick":
        return np.array([c0[0], c0[1], pad_z + 0.0005]), hand_rot()
    if kind == "lift":
        return np.array([c0[0], c0[1], LIFT_Z]), hand_rot()
    level = hand_rot(-90.0)
    want = mpoint(GM.SLIT_X, 0.0, 0.0)
    if kind == "swing_mid":
        a = np.array([c0[0], c0[1], LIFT_Z])
        b = np.array([want[0] - 0.045, want[1], TOP + 0.225])
        return (a + b) / 2, hand_rot(-45.0)
    if kind == "swing":
        return np.array([want[0] - 0.045, want[1], TOP + 0.225]), level
    if kind == "front":
        return np.array([want[0] - 0.035, want[1], TOP + 0.215]), level
    off = coin_in_hand(_TARGETS.get("front", coin_targets("front")[0]))
    tcp_low_z = MOUTH_TOP + FINGER_CLEAR + TIP_HALF_W + TOP
    coin_low = np.array([want[0], want[1], tcp_low_z + off[2]])
    if kind == "above":
        return coin_low + np.array([0.0, 0.0, 0.014]) - off, level
    if kind == "in":
        tcp = coin_low - off
        tcp[2] = tcp_low_z
        return tcp, level
    if kind == "away":
        tcp = coin_low - off
        return tcp + np.array([-0.040, 0.0, 0.012]), level
    raise ValueError(kind)


BAR_TCP_X = GM.BAR[0] + 0.0  # the pads' tips 2 mm short of the plate: TCP at the bar's front face


def bar_targets(kind, roll):
    tcp = mpoint(BAR_TCP_X, 0.0, GM.AX_Z) + np.r_[0.0, _probe_off()[2:]]
    if kind == "back":
        tcp = tcp + np.array([-0.032, 0.0, 0.0])
    elif kind == "far":
        tcp = tcp + np.array([-0.080, 0.0, 0.010])
    return tcp, hand_rot(-90.0, roll)


DOOR_PITCH = -80.0


def door_targets(kind, open_deg=0.0):
    """Hand centre so that the fingertip's upper front edge sits under the door hem's corner (open by open_deg)."""
    rot = hand_rot(DOOR_PITCH)
    a = rot[:, 2]  # approach
    w = -rot[:, 1]  # the pads' width direction (up and forward here)
    hx, hz = GM.DOOR_HINGE
    corner = np.array([GM.DOOR_TAB[0] + 5.5, 0.0, GM.DOOR_TAB[2]])  # under the hem, 5.5 mm in from its front
    r = corner - np.array([hx, 0.0, hz])
    ph = math.radians(open_deg)
    rr = np.array([r[0] * math.cos(ph) + r[2] * math.sin(ph), 0.0, -r[0] * math.sin(ph) + r[2] * math.cos(ph)])
    p = np.array([hx, 0.0, hz]) + rr
    y_corner = GM.DOOR_Y - 5.0  # fingertip at the door's +y corner, clear of the ball's path
    contact = mpoint(p[0], y_corner, p[2])
    tcp = contact - a * PAD_TIP - w * HEM_ON_TIP
    if kind == "lift":
        tcp = tcp + np.array([0.0, 0.0, DOOR_PRESS])  # the arm sags ~3 mm here: aim higher so the tip presses the hem
    below = contact - a * PAD_TIP - w * TIP_HALF_W + np.array([0.0, 0.0, -DOOR_UNDER])  # tip's top edge under the hem
    if kind == "under":
        tcp = below
    elif kind == "front":
        tcp = below + np.array([-0.030, 0.0, 0.0])
    elif kind == "approach":
        tcp = tcp + np.array([-0.030, 0.0, 0.070])
    elif kind == "away":
        tcp = tcp + np.array([-0.040, 0.0, 0.030])
    return tcp, rot


_TARGETS = {}


class GBWaypoint(Waypoint):
    """A right-hand waypoint worked out when played (reads happen at its start)."""

    def __init__(self, duration, fn, finger, label, keep=False, tag=None):
        self.duration = duration * (1.0 if keep else DEMO_PACE)
        self.fn, self.finger, self.label, self.tag = fn, finger, label, tag

    def _t(self):
        p, R = self.fn()
        if self.tag:
            _TARGETS[self.tag] = p
        return p, R

    @property
    def pos(self):
        return tuple(float(x) for x in self._t()[0])

    @property
    def yaw(self):
        return Grip(0.0, self._t()[1])


def _turn(rolls, label):
    W = GBWaypoint
    out = []
    for r in rolls:
        out.append(W(0.9, lambda r=r: bar_targets("on", r), BAR_PINCH, f"{label}: wrist at {r:+.0f} deg"))
    return out


def demo():
    W = GBWaypoint
    right = []
    if DEMO_WAY == "full":
        right += [
            W(1.6, lambda: coin_targets("over"), OPEN_WIDE, "over the quarter"),
            W(1.0, lambda: coin_targets("low"), OPEN_WIDE, "pad tips 8 mm over the mat"),
            W(0.6, lambda: coin_targets("pick"), OPEN_WIDE, "pads either side of the quarter"),
            W(0.8, lambda: coin_targets("pick"), COIN_PINCH, "pinch its rim across the diameter", keep=True),
            W(0.3, lambda: coin_targets("pick"), COIN_PINCH, "settle"),
            W(1.0, lambda: coin_targets("lift"), COIN_PINCH, "lift it"),
            W(1.2, lambda: coin_targets("swing_mid"), COIN_PINCH, "pitch the hand toward level"),
            W(1.2, lambda: coin_targets("swing"), COIN_PINCH, "level: the coin hangs face to the machine"),
            W(1.0, lambda: coin_targets("front"), COIN_PINCH, "in front of the slot", tag="front"),
            W(0.5, lambda: coin_targets("front"), COIN_PINCH, "hold"),
            W(1.2, lambda: coin_targets("above"), COIN_PINCH, "coin over the slot"),
            W(0.4, lambda: coin_targets("above"), COIN_PINCH, "hold"),
            W(1.0, lambda: coin_targets("in"), COIN_PINCH, "lower it into the slot's mouth"),
            W(0.4, lambda: coin_targets("in"), COIN_PINCH, "hold"),
            W(0.6, lambda: coin_targets("in"), COIN_OPEN, "let go: the quarter drops in", keep=True),
            W(0.8, lambda: coin_targets("in"), COIN_OPEN, "wait"),
            W(1.0, lambda: coin_targets("away"), COIN_OPEN, "back away"),
        ]
    right += [
        W(1.6, lambda: bar_targets("back", -90.0), BAR_OPEN, "in front of the handle, fingers top and bottom"),
        W(1.0, lambda: bar_targets("on", -90.0), BAR_OPEN, "fingers over the T-bar"),
        W(0.6, lambda: bar_targets("on", -90.0), BAR_PINCH, "grip the T-bar", keep=True),
    ]
    right += _turn((-45.0, 0.0, 45.0, 90.0), "first half turn")
    right += [
        W(0.6, lambda: bar_targets("on", 90.0), BAR_OPEN, "let go", keep=True),
        W(0.8, lambda: bar_targets("back", 90.0), BAR_OPEN, "back off"),
        W(0.8, lambda: bar_targets("back", 30.0), BAR_OPEN, "turn the wrist back"),
        W(0.8, lambda: bar_targets("back", -30.0), BAR_OPEN, "turn the wrist back"),
        W(0.8, lambda: bar_targets("back", -90.0), BAR_OPEN, "turn the wrist back"),
        W(1.0, lambda: bar_targets("on", -90.0), BAR_OPEN, "fingers over the T-bar again"),
        W(0.6, lambda: bar_targets("on", -90.0), BAR_PINCH, "grip it again", keep=True),
    ]
    right += _turn((-45.0, 0.0, 45.0, 90.0), "second half turn")
    right += [
        W(0.6, lambda: bar_targets("on", 90.0), BAR_OPEN, "let go", keep=True),
        W(0.8, lambda: bar_targets("back", 90.0), BAR_OPEN, "back off"),
    ]
    if DEMO_WAY in ("full", "nocoin"):  # the wrong way lifts the door too: nothing comes out
        right += [
            W(1.6, lambda: door_targets("approach"), CLOSED, "fingers closed, hand almost level, in front of the door"),
            W(1.2, lambda: door_targets("front"), CLOSED, "in front of the door's hem"),
            W(1.0, lambda: door_targets("under"), CLOSED, "fingertip under the hem's corner"),
            W(0.5, lambda: door_targets("lift", 0.0), CLOSED, "press up under the hem"),
        ]
        for k, ang in enumerate((8.0, 20.0, 32.0, 44.0, DOOR_OPEN_DEG)):
            right.append(W(0.8, lambda ang=ang: door_targets("lift", ang), CLOSED, f"lift the door to {ang:.0f} deg"))
        right += [
            W(2.5, lambda: door_targets("lift", DOOR_OPEN_DEG), CLOSED, "hold it open: the gumball rolls out"),
            W(1.0, lambda: door_targets("lift", 20.0), CLOSED, "let the door down"),
            W(0.8, lambda: door_targets("away"), CLOSED, "away from the door"),
        ]
    right += [
        Waypoint(1.5 * DEMO_PACE, tuple(float(x) for x in PARK), OPEN_WIDE, "park", 0.0),
        Waypoint(FINAL_HOLD_S, tuple(float(x) for x in PARK), OPEN_WIDE, "hold clear", 0.0),
    ]
    return {"right": right}


# ---------------------------------------------------------------- offline test layouts (states.yaml)
DIAL_HOME_DEG = GM.DIAL_HOME


def coin_seated_world():
    return [float(x) for x in _MPOS + GM.coin_seated_local()]


def coin_box_world():
    return [float(x) for x in _MPOS + np.array([-0.080, 0.0, 0.086])]


def ball_behind_door_world():
    return [float(x) for x in _MPOS + np.array([-0.0735, 0.0, (GM.FLOOR_Z_FRONT + 2 + GM.GUM_R) * MM])]


def ball_on_table_world():
    """Rolled out of the door onto the table, 50 mm in front of the machine's base."""
    return [float(MACHINE_XY[0]) - 0.138, float(MACHINE_XY[1]), TOP + GM.GUM_R * MM]


def ball_rolled_away_world():
    """Rolled out and away across the table, toward the robot and to the right."""
    return [float(MACHINE_XY[0]) - 0.200, float(MACHINE_XY[1]) - 0.080, TOP + GM.GUM_R * MM]


def coin_edge_quat():
    """Standing on edge, face along x (authored flat, turned 90 degrees about y)."""
    return [0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)]


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


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
        out[arm] = (tcp, float(np.mean(q[finger_ids])))
    return out


def _diag(objects, t):
    step = _RUN["step"]
    if step == _DIAG["step"] or step % 4:
        return
    _DIAG["step"] = step
    m = objects["gumball_machine"]
    j = m.joints
    hands = grippers() or {}
    h = " ".join(f"{a}=({','.join(f'{x * 1000:.1f}' for x in hands[a][0])};{hands[a][1] * 1000:.1f})" for a in hands)
    c = m.to_local(np.asarray(objects["quarter"].origin, float))
    b = [m.to_local(np.asarray(objects[n].origin, float)) for n in ("gumball_1", "gumball_2")]
    print(f"GB_DIAG t={_RUN['t'] or 0.0:.2f} turn={t['turn_deg']:.1f} pin_mm={j['gate'] * 1000:.1f} "
          f"pawl={math.degrees(j['pawl']):.1f} door={math.degrees(j['door']):.1f} coin_m=({c[0]*1000:.1f},"
          f"{c[1]*1000:.1f},{c[2]*1000:.1f}) b1=({b[0][0]*1000:.0f},{b[0][1]*1000:.0f},{b[0][2]*1000:.0f}) "
          f"b2=({b[1][0]*1000:.0f},{b[1][1]*1000:.0f},{b[1][2]*1000:.0f}) places={t['places']} "
          f"seated={t['seated']} coin_in={t['coin_in']} tilt={_tilt_deg(m):.1f} {h}", flush=True)
