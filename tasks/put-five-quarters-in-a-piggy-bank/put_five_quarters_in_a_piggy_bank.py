"""Put five quarters in a piggy bank: pinch each flat coin off the table by its edge, turn it on edge and drop it
through the slot in the pig's back.

A glazed ceramic piggy bank (Child to Cherish Large Piggy Bank: 203 mm long, 178 mm wide, 178 mm high, 0.91 kg,
38 mm coin slot along the top of its back, rubber stopper underneath) stands loose on its four feet at the front left
of the table, its face toward the arms. Five US quarters (24.26 mm across, 1.75 mm thick, 5.67 g) lie flat on the
table between the bank and the arms. Metres; x forward from the arm bases, y left, z up.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import yam_mesh  # noqa: E402
from yam_mesh import Mesh, MeshPart  # noqa: E402
from yam_parts import ball, cyl  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
DIAG = False  # GPU diagnostic copies print PIG_DIAG lines
DEMO_WAY = "edge"  # "edge": each coin turned on edge over the slot; "flat": wrong way, coins carried flat and dropped
DEMO_PACE = 1.0
TEST_GRIPPERS = None

TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S, SCORE_OVER = 1.0, "end"
RUN_LIMIT_STEPS = 2400  # 120 s at 20 Hz


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
_MEMORY = {"episode": None, "ever": {}}


def ever(key, cond):
    """Remember a satisfied condition until the next episode; offline each call stands alone."""
    if _RUN["t"] is None:
        return bool(cond)
    if _MEMORY["episode"] != _RUN["episode"]:
        _MEMORY["episode"], _MEMORY["ever"] = _RUN["episode"], {}
    _MEMORY["ever"][key] = bool(_MEMORY["ever"].get(key, False) or cond)
    return _MEMORY["ever"][key]


# ---------------------------------------------------------------- the coins: US quarters (US Mint figures)
COIN_R, COIN_T, COIN_MASS = 0.02426 / 2, 0.00175, 0.00567
COIN_SILVER = (0.66, 0.67, 0.70)
COIN_FRICTION = (0.40, 0.35)  # cupronickel on wood, glaze and the pads (estimate)
COIN_XY = [(0.235, 0.060), (0.190, 0.095), (0.130, 0.150), (0.200, 0.020), (0.120, 0.070)]  # the order they go in
COIN_SIDES = 32  # collider: a 32-sided prism (64 corners, the most a PhysX convex hull keeps), square-edged like a coin
COIN_RC = COIN_R * math.sqrt(2 * math.pi / (COIN_SIDES * math.sin(2 * math.pi / COIN_SIDES)))  # same area as the disc


def _coin(i, xy):
    a = np.arange(COIN_SIDES) * 2 * math.pi / COIN_SIDES
    disc = yam_mesh.prism(np.column_stack((COIN_RC * np.cos(a), COIN_RC * np.sin(a))), COIN_T, name="disc",
                          collision="convex", color=COIN_SILVER, proxy=(cyl((0.0, 0.0, 0.0), COIN_R, COIN_T, "z"),))
    return MeshPart(f"coin{i + 1}", (xy[0], xy[1], TOP + COIN_T / 2), (), mass=COIN_MASS, color=COIN_SILVER,
                    friction=COIN_FRICTION, contact_offset=0.001, meshes=(disc,))


COINS = [_coin(i, xy) for i, xy in enumerate(COIN_XY)]
COIN_NAMES = [c.name for c in COINS]

# ---------------------------------------------------------------- the piggy bank (bank frame: origin on the table under
# the body's centre, x along the spine toward the snout, y across, z up)
A_OUT, B_OUT, C_OUT, WALL = 0.086, 0.086, 0.076, 0.005  # body ellipsoid semi-axes and ceramic wall (estimates)
ZC = 0.100  # body centre above the table (legs 24 mm under the belly)
SLOT_LEN, SLOT_W = 0.038, 0.005  # 1.5 in slot (maker); 5 mm width (estimate)
SLOT_TOP = ZC + C_OUT  # 0.176: the slot's mouth on top of the back
A_IN, B_IN, C_IN = A_OUT - WALL, B_OUT - WALL, C_OUT - WALL
PIG_MASS = 0.91  # 2.0 lb item weight (retailer listing)
PIG_FRICTION = (0.50, 0.45)  # glazed ceramic on wood and coins on glaze (estimate)
PIG_PINK, SNOUT_PINK, RUBBER, EYE = (0.95, 0.66, 0.73), (0.93, 0.57, 0.65), (0.22, 0.22, 0.23), (0.08, 0.06, 0.06)
PIG_XY = np.array([0.320, 0.200])
APPROACH = math.radians(-20.0)  # the hand points this way (world heading) across the slot when it posts a coin
PIG_YAW = APPROACH + math.pi / 2  # the bank's snout points forward-left; its -y (across the slot) is the approach


def _pig_shell():
    from pig_mesh import shell
    p, f = shell(A_OUT, B_OUT, C_OUT, WALL, SLOT_LEN, SLOT_W)
    return Mesh("shell", p + np.array([0.0, 0.0, ZC]), f, collision="sdf", sdf_resolution=256, color=PIG_PINK)


def _ear(sign):
    """A triangular ear flap 6 mm thick standing on the head (convex)."""
    x0, x1 = 0.050, 0.056
    tri = [(sign * 0.024, 0.146), (sign * 0.058, 0.146), (sign * 0.046, 0.178)]
    pts = [(x, y, z) for x in (x0, x1) for (y, z) in tri]
    p = np.array(pts, float)
    f = np.array([(0, 2, 1), (3, 4, 5), (0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4), (2, 0, 3), (2, 3, 5)], int)
    if yam_mesh.signed_volume((p, f)) < 0:
        f = f[:, ::-1]
    return Mesh("ear_l" if sign > 0 else "ear_r", p, f, collision="convex", color=PIG_PINK)


def _leg_top(x, y):
    return ZC - C_OUT * math.sqrt(max(0.0, 1.0 - (x / A_OUT) ** 2 - (y / B_OUT) ** 2)) + 0.002


def _pig_shapes():
    legs = [cyl((sx * 0.046, sy * 0.046, _leg_top(0.046, 0.046) / 2), 0.015, _leg_top(0.046, 0.046), "z", PIG_PINK)
            for sx in (1, -1) for sy in (1, -1)]
    return legs + [
        cyl((0.090, 0.0, ZC - 0.012), 0.024, 0.030, "x", SNOUT_PINK),  # snout, 105 mm from the body's centre
        ball((0.105, 0.008, ZC - 0.012), 0.0035, RUBBER), ball((0.105, -0.008, ZC - 0.012), 0.0035, RUBBER),  # nostrils
        ball((0.068, 0.030, ZC + 0.032), 0.0045, EYE), ball((0.068, -0.030, ZC + 0.032), 0.0045, EYE),
        ball((-0.090, 0.0, ZC + 0.012), 0.009, PIG_PINK),  # tail
        cyl((0.0, 0.0, ZC - C_OUT + 0.001), 0.021, 0.004, "z", RUBBER),  # rubber stopper in the belly
    ]


PIG = MeshPart("piggy_bank", (float(PIG_XY[0]), float(PIG_XY[1]), TOP), _pig_shapes(), mass=PIG_MASS,
               color=PIG_PINK, yaw=PIG_YAW, friction=PIG_FRICTION, contact_offset=0.006,
               meshes=(_pig_shell(), _ear(1), _ear(-1)))
OBJECTS = [PIG, *COINS]


def _glazed_look():
    """The kit gives meshes a metallic look; the bank is glazed ceramic, so rebind its meshes to a glossy non-metal."""
    original = yam_mesh._author_mesh
    if getattr(original, "_glazed", False):
        return

    def author(stage, parent, mesh, anchor, color, contact_offset):
        original(stage, parent, mesh, anchor, color, contact_offset)
        if mesh.name not in ("shell", "ear_l", "ear_r"):  # the coins keep the kit's metal look
            return
        import isaaclab.sim as sim_utils
        from yam_arena.station import linear_rgb
        path = f"{parent}/mesh_{mesh.name}"
        look_path = f"{parent}/mesh_{mesh.name}_glaze"
        look = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color if mesh.color is None else mesh.color),
                                           roughness=0.25, metallic=0.0)
        look.func(look_path, look)
        sim_utils.bind_visual_material(path, look_path, stage=stage)

    author._glazed = True
    yam_mesh._author_mesh = author


_glazed_look()

# ---------------------------------------------------------------- scoring
TIP_MAX_DEG = 20.0  # the bank tipped further than this at any time (knocked over, or laid down to post coins flat): 0
OFF_TABLE = 0.05


def pose(st):
    return np.asarray(st.origin, float), np.asarray(st.rotation(), float)


def coin_in_bank(pig, coin):
    """The coin's centre inside the bank's hollow and the whole coin below the slot's inner mouth."""
    o, rot = pose(pig)
    u, v, w = rot.T @ (np.asarray(coin.origin, float) - o) - np.array([0.0, 0.0, ZC])
    inside = (u / A_IN) ** 2 + (v / B_IN) ** 2 + (w / C_IN) ** 2 < 1.0
    below = w < C_IN - COIN_R - 0.001
    return bool(inside and below)


def score_terms(objects):
    pig = objects["piggy_bank"]
    tilt = pig.tilt_deg()
    tipped = ever("bank tipped past 20 deg", tilt > TIP_MAX_DEG)
    off = ever("bank off the table", float(pig.origin[2]) < TOP - OFF_TABLE)
    inside = [n for n in COIN_NAMES if coin_in_bank(pig, objects[n])]
    if DIAG:
        _diag(objects, inside, tilt)
    return {"inside": inside, "tilt": tilt, "tipped": tipped, "off": off}


def score(objects):
    t = score_terms(objects)
    if t["tipped"] or t["off"]:
        return 0.0
    return round(0.2 * len(t["inside"]), 6)


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- demo (left arm alone; the right arm stays at rest)
# Each coin: the hand comes over it pointing straight down, fingers closing along the bank's spine, the coin's centre
# 4.5 mm off the pads' middle. It stops with the pad tips 8 mm up, reads where the hand really is, corrects sideways and
# lowers the tips to 0.5 mm over the table beside the coin, so the pads close on the coin's 1.75 mm rim, across its
# diameter. Lift, swing the hand up to level, pointing across the bank's back at the slot, so the coin turns on edge in
# the pinch; hold it over the slot, read the coin and the bank, lower to just above the slot, read again, lower until
# the coin's lower edge is 2 mm into the slot, let go (it drops in) and back away.
PAD_TIP = 0.013  # pad tips past the hand's centre point
TIP_Z = 0.0005  # pad tips this far above the table when they close on the rim (the hand is sent exactly here)
TIP_LOW = 0.008  # ... and this far up when the hand stops to read itself before going down
GRASP_DY, GRASP_DZ = 0.0045, PAD_TIP + TIP_Z - COIN_T / 2  # coin centre in the hand frame (pad-width axis, fingers)
GRIP_OPEN = -0.020  # pads 40 mm apart
GRIP_PINCH = -0.0105  # pads commanded 21 mm apart: ~1.6 mm past the rim each side, ~8 N
HOVER_Z = TOP + 0.060
LIFT_Z = TOP + 0.060
PRE_ABOVE = 0.025  # coin's lower edge this far above the slot's mouth while the hand reads
NEAR_ABOVE = 0.008
INTO_SLOT = 0.002
RISE_Z = TOP + 0.200
WAIT_BACK = 0.050  # the turned coin waits this far back from the slot before moving over it
CORR_MAX, CORR_MAX_DEG, HAND_MAX = 0.015, 8.0, 0.004
PARK = np.array([0.20, 0.30, TOP + 0.20])
FINAL_HOLD_S = 1.0
_READS = {"episode": None, "values": {}}


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _aa(axis, a):
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(a) * k + (1.0 - math.cos(a)) * (k @ k)


def _down():
    from yam_arena.demo import grip_rotation
    return np.asarray(getattr(grip_rotation, "__wrapped__", grip_rotation)(0.0), float)


def hand_rotations(bank_yaw):
    """(pick, level) hand rotations: fingers closing along the bank's spine; level points along the bank's +y."""
    head = bank_yaw - math.pi / 2  # heading of the bank's -y (across the slot, away from the hand)
    pick = _rz(head) @ _down()
    if DEMO_WAY == "flat":
        return pick, pick
    return pick, _rz(head) @ _ry(-math.pi / 2) @ _down()


def _objects_now(names):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        return arena_scene.object_states(env, list(names))
    except Exception as e:  # noqa: BLE001
        print(f"PIG_READ failed: {e!r}", flush=True)
        return None


def _values():
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    return _READS["values"]


def read_bank():
    v = _values()
    if "bank" not in v:
        st = _objects_now(["piggy_bank"])
        if st is None:
            v["bank"] = (np.array([PIG_XY[0], PIG_XY[1], TOP]), PIG_YAW)
        else:
            o, r = pose(st["piggy_bank"])
            yaw = math.atan2(r[1, 0], r[0, 0])
            v["bank"] = (o, yaw)
            print(f"PIG_READ bank t={_RUN['t']:.2f} off_mm={tuple(round(float(x) * 1000, 1) for x in o[:2] - PIG_XY)} "
                  f"yaw_err_deg={math.degrees(yaw - PIG_YAW):.2f} tilt={st['piggy_bank'].tilt_deg():.2f}", flush=True)
    return v["bank"]


def read_coin(i):
    key = f"coin{i}"
    v = _values()
    if key not in v:
        st = _objects_now([COIN_NAMES[i]])
        if st is None:
            v[key] = np.array([COIN_XY[i][0], COIN_XY[i][1], TOP + COIN_T / 2])
        else:
            c = np.asarray(st[COIN_NAMES[i]].origin, float)
            v[key] = c
            print(f"PIG_READ {key} t={_RUN['t']:.2f} off_mm={tuple(round(float(x) * 1000, 1) for x in c[:2] - COIN_XY[i])} "
                  f"z_mm={(c[2] - TOP) * 1000:.2f} tilt={st[COIN_NAMES[i]].tilt_deg():.2f}", flush=True)
    return v[key]


def read_hand(i, commanded):
    """Where the left hand's centre point really is, minus where it was sent (clamped), read once per coin."""
    key = f"hand{i}"
    v = _values()
    if key not in v:
        hands = grippers() if _RUN["env"] is not None else None
        if not hands:
            v[key] = np.zeros(3)
        else:
            err = np.asarray(hands["left"][0], float) - np.asarray(commanded, float)
            v[key] = np.clip(err, -HAND_MAX, HAND_MAX)
            print(f"PIG_READ {key} t={_RUN['t']:.2f} hand_err_mm={tuple(round(float(x) * 1000, 2) for x in err)}",
                  flush=True)
    return v[key]


def slot_mouth():
    o, yaw = read_bank()
    return o + _rz(yaw) @ np.array([0.0, 0.0, SLOT_TOP]), -_rz(yaw)[:, 1]


def ideal_coin(height):
    """The coin's centre wanted with its lower edge `height` above the slot's mouth (negative: into the slot)."""
    mouth, normal = slot_mouth()
    return mouth + np.array([0.0, 0.0, COIN_R + height]), normal


def correction(i, stage):
    """(R, t) applied to coin i's insertion targets, so the coin as actually held lands on its planned line over the
    slot: x -> dR (x - c_act) + c_act + (c_plan - c_act), chained after the previous stage's correction."""
    key = f"ins{i}{stage}"
    v = _values()
    if key in v:
        return v[key]
    prior = correction(i, "a") if stage == "b" else (np.eye(3), np.zeros(3))
    c_plan, n_plan = ideal_coin(PRE_ABOVE if stage == "a" else NEAR_ABOVE)
    st = _objects_now([COIN_NAMES[i]])
    if st is None:
        v[key] = prior
        return prior
    coin = st[COIN_NAMES[i]]
    c_act, n_act = np.asarray(coin.origin, float), np.asarray(coin.rotation(), float)[:, 2]
    if float(np.linalg.norm(c_act - c_plan)) > 0.040:  # the coin is not in the hand: go on as planned, no correction
        print(f"PIG_READ {key} t={_RUN['t']:.2f} coin not in hand, no correction", flush=True)
        v[key] = prior
        return prior
    if float(np.dot(n_act, n_plan)) < 0:
        n_act = -n_act
    axis = np.cross(n_act, n_plan)
    s = float(np.linalg.norm(axis))
    a = math.atan2(s, float(np.dot(n_act, n_plan)))
    dR = _aa(axis, min(a, math.radians(CORR_MAX_DEG))) if s > 1e-9 else np.eye(3)
    move = c_plan - c_act
    n = float(np.linalg.norm(move))
    if n > CORR_MAX:
        move *= CORR_MAX / n
    Rp, tp = prior
    v[key] = (dR @ Rp, dR @ (tp - c_act) + c_act + move)
    print(f"PIG_READ {key} t={_RUN['t']:.2f} coin_err_mm={tuple(round(float(x) * 1000, 2) for x in c_act - c_plan)} "
          f"face_err_deg={math.degrees(a):.2f} coin_tilt={coin.tilt_deg():.2f}", flush=True)
    return v[key]


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


def coin_hand_target(i, kind, corr=None):
    """(hand centre point, hand rotation) for coin i at one stage of its plan."""
    _, yaw = read_bank()
    pick_r, level_r = hand_rotations(yaw)
    local = np.array([0.0, GRASP_DY, GRASP_DZ])
    if kind in ("above", "rise"):  # high over the coin, the hand half turned toward level
        c = read_coin(i)
        tcp = np.array([c[0], c[1], TOP + COIN_T / 2]) - pick_r @ local
        half = pick_r if DEMO_WAY == "flat" else _rz(yaw - math.pi / 2) @ _ry(-math.pi / 4) @ _down()
        return np.array([tcp[0], tcp[1], RISE_Z]), half
    if kind in ("over", "low", "pick", "lift"):
        c = read_coin(i)
        tcp = np.array([c[0], c[1], TOP + COIN_T / 2]) - pick_r @ local
        if kind == "over":
            return np.array([tcp[0], tcp[1], HOVER_Z]), pick_r
        if kind == "lift":
            return np.array([tcp[0], tcp[1], LIFT_Z]), pick_r
        low = np.array([tcp[0], tcp[1], TOP + PAD_TIP + TIP_LOW])
        if kind == "low":
            return low, pick_r
        err = read_hand(i, low) * np.array([1.0, 1.0, 0.0])  # sideways only: the height is sent as is
        return np.array([tcp[0], tcp[1], TOP + PAD_TIP + TIP_Z]) - err, pick_r
    if DEMO_WAY == "flat":  # wrong way: the coin carried flat, let go with its centre 25 mm over the slot
        coin = slot_mouth()[0] + np.array([0.0, 0.0, PRE_ABOVE + COIN_T / 2])
        tcp = coin - pick_r @ local
        return (tcp - 0.040 * slot_mouth()[1] if kind == "back" else tcp), pick_r
    height = {"wait": PRE_ABOVE, "pre": PRE_ABOVE, "near": NEAR_ABOVE, "in": -INTO_SLOT, "back": -INTO_SLOT}[kind]
    coin, normal = ideal_coin(height)
    if kind == "wait":  # level, coin on edge, held back from the slot across the back
        coin = coin - WAIT_BACK * normal
    R, t = correction(i, corr) if corr else (np.eye(3), np.zeros(3))
    coin_c, rot = R @ coin + t, R @ level_r
    tcp = coin_c - rot @ local
    if kind == "back":
        tcp = tcp - 0.025 * normal + np.array([0.0, 0.0, 0.040])
    return tcp, rot


class CoinWaypoint(Waypoint):
    """A left-hand waypoint worked out when it is played (reads are taken the first time a waypoint needs them, which
    is when it starts: the hand is then at the previous waypoint)."""

    def __init__(self, duration, i, kind, finger, label, keep=False, corr=None):
        self.duration = duration * (1.0 if keep else DEMO_PACE)
        self.i, self.kind, self.finger, self.label, self.corr = i, kind, finger, label, corr

    @property
    def pos(self):
        return tuple(float(x) for x in coin_hand_target(self.i, self.kind, self.corr)[0])

    @property
    def yaw(self):
        return Grip(0.0, coin_hand_target(self.i, self.kind, self.corr)[1])


def coin_plan(i):
    O, P = GRIP_OPEN, GRIP_PINCH
    n = i + 1
    return [
        CoinWaypoint(1.2, i, "above", O, f"coin {n}: high over it"),
        CoinWaypoint(1.2, i, "over", O, f"coin {n}: down over it"),
        CoinWaypoint(1.0, i, "low", O, f"coin {n}: pads down to 8 mm over the table"),
        CoinWaypoint(0.25, i, "low", O, f"coin {n}: hold low"),
        CoinWaypoint(0.6, i, "pick", O, f"coin {n}: pad tips to the table round it"),
        CoinWaypoint(0.8, i, "pick", P, f"coin {n}: pinch the rim", keep=True),
        CoinWaypoint(0.25, i, "pick", P, f"coin {n}: settle"),
        CoinWaypoint(0.8, i, "lift", P, f"coin {n}: lift"),
        CoinWaypoint(1.2, i, "rise", P, f"coin {n}: rise, turning it"),
        CoinWaypoint(1.4, i, "wait", P, f"coin {n}: on edge, behind the slot"),
        CoinWaypoint(0.8, i, "pre", P, f"coin {n}: over the slot"),
        CoinWaypoint(0.5, i, "pre", P, f"coin {n}: hold over the slot"),
        CoinWaypoint(1.0, i, "near", P, f"coin {n}: lower to just over the slot", corr="a"),
        CoinWaypoint(0.4, i, "near", P, f"coin {n}: hold near", corr="a"),
        CoinWaypoint(0.8, i, "in", P, f"coin {n}: lower edge into the slot", corr="b"),
        CoinWaypoint(0.6, i, "in", O, f"coin {n}: let go", keep=True, corr="b"),
        CoinWaypoint(0.8, i, "back", O, f"coin {n}: back away", corr="b"),
    ]


def demo():
    left = []
    for i in range(len(COINS)):
        left += coin_plan(i)
    left.append(Waypoint(1.5 * DEMO_PACE, tuple(float(x) for x in PARK), GRIP_OPEN, "park", 0.0))
    left.append(Waypoint(FINAL_HOLD_S, tuple(float(x) for x in PARK), GRIP_OPEN, "hold clear", 0.0))
    return {"left": left}


# ---------------------------------------------------------------- offline test layouts (states.yaml)
def _quat(rot):
    from yam_arena.ik import matrix_to_quat_xyzw
    return [float(v) for v in matrix_to_quat_xyzw(np.asarray(rot, float))]


BANK_O = np.array([PIG_XY[0], PIG_XY[1], TOP])
BANK_R = _rz(PIG_YAW)


def bank_point(u, v, w):
    """World point from bank coordinates (u along the spine to the snout, v across, w up from the table)."""
    return [float(x) for x in BANK_O + BANK_R @ np.array([u, v, w])]


def lying_inside(k):
    """Coin k lying flat on the bottom of the bank's hollow, side by side along the spine."""
    u = -0.036 + 0.018 * k
    w = ZC - C_IN * math.sqrt(1.0 - (u / A_IN) ** 2) + COIN_T / 2 + 0.002
    return bank_point(u, 0.0, w)


def on_edge_quat():
    """A coin standing on edge in line with the slot (its faces toward the bank's sides)."""
    return _quat(BANK_R @ np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]))


def tipped_bank(deg):
    """The bank turned `deg` about its spine (lying on its side at 90), resting on the table."""
    a = math.radians(deg)
    roll = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])
    rot = BANK_R @ roll
    centre = BANK_O + np.array([0.0, 0.0, ZC])
    origin = centre - rot @ np.array([0.0, 0.0, ZC])
    if deg >= 60:
        origin[2] += (B_OUT - ZC) * 1.0  # resting on its flank
    return [float(x) for x in origin], _quat(rot)


def inside_tipped(k, deg):
    """Coin k inside the tipped bank (same hollow position as lying_inside(k), carried with the bank)."""
    o, q = tipped_bank(deg)
    a = math.radians(deg)
    roll = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])
    local = BANK_R.T @ (np.array(lying_inside(k)) - BANK_O)
    return [float(x) for x in np.array(o) + BANK_R @ roll @ local]


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


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
        out[arm] = (tcp, float(np.mean(q[finger_ids])))
    return out


def _diag(objects, inside, tilt):
    step = _RUN["step"]
    if step == _DIAG["step"] or step % 4:
        return
    _DIAG["step"] = step
    pig = objects["piggy_bank"]
    o, rot = pose(pig)
    cs = []
    for n in COIN_NAMES:
        c = objects[n]
        u, v, w = rot.T @ (np.asarray(c.origin, float) - o) - np.array([0.0, 0.0, SLOT_TOP])
        cs.append(f"{n[-1]}({u * 1000:.1f},{v * 1000:.1f},{w * 1000:.1f};{c.tilt_deg():.0f};"
                  f"{np.linalg.norm(c.lin_vel) * 1000:.0f})")
    hands = grippers() or {}
    h = " ".join(f"{a[0]}=({','.join(f'{x * 1000:.1f}' for x in hands[a][0])};{hands[a][1] * 1000:.1f})" for a in hands)
    print(f"PIG_DIAG t={_RUN['t'] or 0.0:.2f} pig=({(o[0] - PIG_XY[0]) * 1000:.1f},{(o[1] - PIG_XY[1]) * 1000:.1f},"
          f"{(o[2] - TOP) * 1000:.1f};tilt {tilt:.2f}) in={len(inside)} coins(slot frame mm;tilt;v) {' '.join(cs)} {h}",
          flush=True)
