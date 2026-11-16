"""Scratch the silver PIN coating off a gift card with a quarter held on edge, without wearing the code away, while the
other hand pins the card's hanger.

A one-piece white PVC hanging gift card (3.375 x 6.625 in = 85.7 x 168.3 mm, 30 mil = 0.76 mm, about 15 g) lies back
up on the table: a CR80 tear-off card (85.7 x 54.0 mm) at the end nearer the right arm and, joined to it along a
perforated tear line, the 85.7 x 114.3 mm hanger with its sombrero hang slot. On the card's back a 15-character code
(XXXXX-XXXXX-XXXXX, 46 mm long, characters 2.5 mm tall) is covered by a silver scratch-off panel 9.5 x 63.5 mm, 8 mm in
from the card's free edge. A US quarter (24.26 mm across, 1.75 mm thick, 5.67 g) lies flat on the table in front of
the right arm. The right hand scratches with the quarter; the left hand presses the hanger's far end flat, 130 mm
beyond the code line. Metres; x forward from the arm bases, y left, z up.
"""
import functools
import importlib.abc
import importlib.machinery
import importlib.util
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import yam_mesh  # noqa: E402
import yam_parts  # noqa: E402
from yam_mesh import MeshPart  # noqa: E402
from yam_parts import Part, box, cyl  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
DIAG = False  # GPU diagnostic copies print SC_DIAG lines
DEMO_WAY = "scratch"  # "scratch": the demo; wrong ways "scrub" (a few places scrubbed over and over), "nopin" (no left hand)
PROBE_WAYS = None  # GPU probe copies only: a list of ways, one per episode in turn (counted by demo() calls)
DEMO_PACE = 1.0
TEST_GRIPPERS = None  # offline layouts: {"left": (tcp, finger), "right": (tcp, finger)}
TEST_REC = None  # offline layouts: a scratch record (see blank_record)

TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S, SCORE_OVER = 1.0, "end"
RUN_LIMIT_STEPS = 3600  # 180 s at 20 Hz


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


# ---------------------------------------------------------------- the hanging card (one 30 mil PVC piece)
# Plate frame ("card" body): origin at the piece's centre, x along the card's length, y along the hanger (+y toward
# the hang slot, away from the card's free edge), z up (the back, with the code, faces up). PlasCards "Hanging Tear-Off
# Card": trim 3.375 x 6.625 in, tear-off card 3.375 x 2.125 in, hang slot 1.129 in long (maker's template).
PLATE_X, PLATE_Y, CARD_T = 0.085725, 0.168275, 0.000762
CARD_L, CARD_W = 0.085725, 0.053975  # the tear-off CR80 card at the plate's -y end
CARD_MASS = 0.015  # estimate: 85.7 x 168.3 x 0.762 mm of PVC at 1.38 g/cm3, less the slot (a CR80 card is ~5 g)
SLOT_L, SLOT_H, SLOT_TOP = 0.0287, 0.0055, 0.008  # hang slot: length from the template; height, top margin estimates
TEAR_Y = -PLATE_Y / 2 + CARD_W  # the perforated tear line between card and hanger
CARD_WHITE = (0.95, 0.95, 0.93)
CARD_FRICTION = (0.15, 0.10)  # PhysX averages pairs: PVC on the oak table (default 0.50) 0.325 / 0.30 sliding (PVC on
# wood measured ~0.30 sliding), on the finger pads the same; with the quarter 0.45 / 0.425 (estimates)
STK_W, STK_L = 0.0095, 0.0635  # 3/8 x 2-1/2 in scratch-off panel
STK_IN = 0.008  # panel's edge from the card's free long edge
STK_Y = -PLATE_Y / 2 + STK_IN + STK_W / 2  # panel (and code line) centre: -71.39 mm in the plate frame
PIN_V = 0.130  # the left hand presses the hanger this far beyond the code line (25.5 mm below its top end)
CODE = "7KQ2M-X94TR-C3H8P"  # what the simulated card shows (a fresh random code on every real card)
CODE_LEN = 0.046
PITCH = CODE_LEN / len(CODE)  # 2.706 mm per character position
CHAR_W, CHAR_H = 0.0020, 0.0025
CHAR_X = [(k - (len(CODE) - 1) / 2) * PITCH for k, ch in enumerate(CODE) if ch != "-"]  # 15 character centres
N_CHARS = len(CHAR_X)
CELL = 0.0005  # the sticker's coating as a grid of 0.5 mm cells
CELL_X = (np.arange(int(round(STK_L / CELL))) + 0.5) * CELL - STK_L / 2
CELL_Y = (np.arange(int(round(STK_W / CELL))) + 0.5) * CELL - STK_W / 2 + STK_Y
CELLS = np.array([(x, y) for y in CELL_Y for x in CELL_X])  # card frame, 127 x 19 = 2413
CELL_CHAR = np.full(len(CELLS), -1, int)  # which character's box each cell lies over (-1: none)
for _k, _cx in enumerate(CHAR_X):
    _in = (np.abs(CELLS[:, 0] - _cx) < CHAR_W / 2) & (np.abs(CELLS[:, 1] - STK_Y) < CHAR_H / 2)
    CELL_CHAR[_in] = _k
CHAR_CELLS = [np.flatnonzero(CELL_CHAR == k) for k in range(N_CHARS)]

# Layout: the code line's centre L and the direction V across it, away from the near long edge (the strokes pull the
# coin back along -V, toward the right arm).
L_XY = np.array([0.200, -0.090])
V_DEG = 90.0
V_DIR = np.array([math.cos(math.radians(V_DEG)), math.sin(math.radians(V_DEG)), 0.0])
U_DIR = np.array([V_DIR[1], -V_DIR[0], 0.0])  # U x V = up, so the card's +y is V_DIR
CARD_YAW = math.atan2(U_DIR[1], U_DIR[0])  # the card's x (length) runs along U_DIR
CARD_XY = L_XY - STK_Y * V_DIR[:2]


def _plate_boxes():
    """The piece as four boxes round the hang slot (box colliders, not one thin convex hull)."""
    y_lo = PLATE_Y / 2 - SLOT_TOP - SLOT_H  # slot's lower edge
    y_hi = PLATE_Y / 2 - SLOT_TOP
    side = (PLATE_X - SLOT_L) / 2
    return [box((0.0, (-PLATE_Y / 2 + y_lo) / 2, 0.0), (PLATE_X, y_lo + PLATE_Y / 2, CARD_T), CARD_WHITE),
            box((-(SLOT_L + side) / 2, (y_lo + PLATE_Y / 2) / 2, 0.0), (side, PLATE_Y / 2 - y_lo, CARD_T), CARD_WHITE),
            box(((SLOT_L + side) / 2, (y_lo + PLATE_Y / 2) / 2, 0.0), (side, PLATE_Y / 2 - y_lo, CARD_T), CARD_WHITE),
            box((0.0, (y_hi + PLATE_Y / 2) / 2, 0.0), (SLOT_L, SLOT_TOP, CARD_T), CARD_WHITE)]


CARD = Part("card", (float(CARD_XY[0]), float(CARD_XY[1]), TOP + CARD_T / 2), _plate_boxes(), mass=CARD_MASS,
            color=CARD_WHITE, yaw=CARD_YAW, friction=CARD_FRICTION, contact_offset=0.001)
PRESSED_PARTS = ("card",)  # the left fingertips press the hanger flat (a deliberate press below the usual tip floor)

# ---------------------------------------------------------------- the coin: a US quarter (US Mint figures)
COIN_R, COIN_T, COIN_MASS = 0.02426 / 2, 0.00175, 0.00567
COIN_SILVER = (0.66, 0.67, 0.70)
COIN_FRICTION = (0.75, 0.75)  # cupronickel: 0.45 / 0.425 on the card's coating, 0.625 on the pads and the table (estimates)
COIN_SIDES = 32  # collider: a 32-sided prism (64 corners, the most a PhysX convex hull keeps), square-edged like a coin
COIN_RC = COIN_R * math.sqrt(2 * math.pi / (COIN_SIDES * math.sin(2 * math.pi / COIN_SIDES)))  # same area as the disc
COIN_XY = np.array([0.130, -0.175])
# The prism is turned so that two opposite flats face along U_DIR, where the fingers will close on it.
COIN_YAW = CARD_YAW - math.pi / COIN_SIDES


def _coin():
    a = np.arange(COIN_SIDES) * 2 * math.pi / COIN_SIDES
    disc = yam_mesh.prism(np.column_stack((COIN_RC * np.cos(a), COIN_RC * np.sin(a))), COIN_T, name="disc",
                          collision="convex", color=COIN_SILVER, proxy=(cyl((0.0, 0.0, 0.0), COIN_R, COIN_T, "z"),))
    return MeshPart("quarter", (float(COIN_XY[0]), float(COIN_XY[1]), TOP + COIN_T / 2), (), mass=COIN_MASS,
                    color=COIN_SILVER, yaw=COIN_YAW, friction=COIN_FRICTION, contact_offset=0.001, meshes=(disc,))


COIN = _coin()
OBJECTS = [CARD, COIN]

# ---------------------------------------------------------------- the scratch rule (estimates, see the README)
F_MIN = 0.5  # N: the rim lifts the coating where it passes pressing at least this hard
F_MAX = 6.0  # N: pressing harder than this over a character scrapes its print away
MAX_PASSES = 3  # a fourth pass of the pressing rim over a character wears it
FLECK = 1  # a character still reads with at most this many 0.5 mm cells of silver left on its 2.0 x 2.5 mm box (of 20)
F_WINDOW = 6  # physics steps (1/40 s) over which the contact force is averaged
RIM_BAND = 0.001  # a contact counts as the rim's when it is within 1 mm of the coin's edge ...
ON_EDGE_DEG = 30.0  # ... and the coin's face stands at least 30 deg off the card
FOOT_ALONG, FOOT_ACROSS = 0.0015, 0.0005  # half-sizes of the rim's footprint: along its edge / across the coin
PASS_GAP_S = 0.1  # the rim off a character for longer than this ends a pass


def blank_record():
    return {"cells": np.ones(len(CELLS), bool), "passes": np.zeros(N_CHARS, int), "hard": np.zeros(N_CHARS, bool),
            "in_pass": np.zeros(N_CHARS, bool), "last_hit": np.full(N_CHARS, -9.0), "t": 0.0,
            "forces": [], "peak": 0.0, "rim_steps": 0, "card0": None, "dirty": True}


_REC = blank_record()


def rec_with(readable=(), passes=None, hard=(), left=()):
    """Offline layouts: a record with the given characters fully uncovered (indices or a count from the left); each
    index in `left` puts one cell of silver back on that character's box."""
    rec = blank_record()
    idx = range(readable) if isinstance(readable, int) else readable
    for k in idx:
        rec["cells"][CHAR_CELLS[k]] = False
    for k in left:
        rec["cells"][CHAR_CELLS[k][~rec["cells"][CHAR_CELLS[k]]][0]] = True
    if passes is not None:
        rec["passes"][:] = passes
    for k in hard:
        rec["hard"][k] = True
    return rec


def char_state(rec):
    """(readable, worn) flags per character: readable when at most FLECK cells of silver are left on its box."""
    readable = np.array([int(rec["cells"][c].sum()) <= FLECK for c in CHAR_CELLS])
    worn = (rec["passes"] > MAX_PASSES) | rec["hard"]
    return readable, worn


# ---------------------------------------------------------------- scoring
MOVE_MAX, TURN_MAX_DEG = 0.05, 30.0  # the card must end within 5 cm and 30 deg of where it started, face up
HANDS_CLEAR = 0.030  # every fingertip and grip point at least 3 cm from the card and its hanger at the end


def pose(st):
    return np.asarray(st.origin, float), np.asarray(st.rotation(), float)


def card_box_distance(card, p):
    """Distance from point p (world) to the card-and-hanger slab."""
    o, rot = pose(card)
    q = rot.T @ (np.asarray(p, float) - o)
    d = np.maximum(np.abs(q) - np.array([PLATE_X / 2, PLATE_Y / 2, CARD_T / 2]), 0.0)
    return float(np.linalg.norm(d))


def hand_points():
    """{arm: [grip point, fingertip point]} (world), or None offline without TEST_GRIPPERS."""
    if TEST_GRIPPERS is not None:
        return {a: [np.asarray(v[0], float)] for a, v in TEST_GRIPPERS.items()}
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS
    out = {}
    for arm in ("left", "right"):
        art = env.unwrapped.scene[arm + "_robot"]
        ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        q = _np(art.data.joint_pos)[0][ids]
        tcp, rot = ik.tcp_pose(np.asarray(q, float), station.BASES[arm])
        out[arm] = [tcp, tcp + 0.013 * rot[:, 2]]
    return out


def score_terms(objects):
    card = objects["card"]
    rec = TEST_REC if TEST_REC is not None else _REC
    o, rot = pose(card)
    if rec.get("card0") is None:
        rec["card0"] = (o.copy(), rot.copy()) if _RUN["t"] is not None else (
            np.array([CARD_XY[0], CARD_XY[1], TOP + CARD_T / 2]), _rz(CARD_YAW))
    o0, r0 = rec["card0"]
    moved = float(np.linalg.norm((o - o0)[:2]))
    turn = math.degrees(abs(math.atan2(*(lambda m: (m[1, 0], m[0, 0]))(r0.T @ rot))))
    flipped = rot[2, 2] < 0.0
    readable, worn = char_state(rec)
    hands = hand_points()
    clear = True if hands is None else all(card_box_distance(card, p) >= HANDS_CLEAR for v in hands.values() for p in v)
    if _RUN["t"] is not None:
        _show_coating()
        if DIAG:
            _diag(objects, rec, readable, worn, moved, turn)
    return {"readable": int(readable.sum()), "worn": int(worn.sum()), "moved": moved, "turn": turn,
            "flipped": bool(flipped), "clear": bool(clear)}


def score(objects):
    t = score_terms(objects)
    if t["worn"] or t["flipped"] or t["moved"] > MOVE_MAX or t["turn"] > TURN_MAX_DEG:
        return 0.0
    n = t["readable"]
    if n == N_CHARS:
        return 1.0 if t["clear"] else 0.75
    if n >= 13:
        return 0.75
    if n >= 9:
        return 0.5
    if n >= 4:
        return 0.25
    return 0.0


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- simulator side: contact readout and the sticker look
_TR = {"armed": False, "mgr": None, "view": None, "env": None, "paths": {}, "err": None, "buf": [], "polls": 0,
       "inst": None, "shown": None}


def _np(x):
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if not isinstance(x, np.ndarray) and hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    try:
        import warp as wp
        if isinstance(x, wp.array):
            return wp.to_torch(x).detach().cpu().numpy()
    except Exception:  # noqa: BLE001
        pass
    return np.asarray(x)


def _quat_rot(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _pose(name):
    """(position, rotation) of a free object straight from PhysX, world frame."""
    obj = _TR["env"].unwrapped.scene[name]
    t = _np(obj.root_physx_view.get_transforms()).reshape(-1)[:7].astype(np.float64)
    return t[:3], _quat_rot(t[3:7])


def _enable_contact_processing():
    try:
        import carb
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    except Exception:  # noqa: BLE001
        pass


def _make_view(mgr):
    sv = mgr.get_physics_sim_view()
    if sv is None:
        return
    coin, card = _TR["paths"].get("quarter"), _TR["paths"].get("card")
    if coin is None or card is None:
        from isaaclab.sim.utils.stage import get_current_stage
        for prim in get_current_stage().Traverse():
            p = prim.GetPath().pathString
            if "/env_0/" in p and p.endswith("/env_0/quarter"):
                coin = p
            elif "/env_0/" in p and p.endswith("/env_0/card"):
                card = p
    _TR["view"] = sv.create_rigid_contact_view([coin], filter_patterns=[[card]], max_contact_data_count=64)
    print(f"SC contact monitor: {coin} against {card}", flush=True)
    if DIAG:  # GPU copies: the left fingertips' press on the hanger
        try:
            from isaaclab.sim.utils.stage import get_current_stage
            from pxr import UsdPhysics
            tips = [str(prim.GetPath()) for prim in get_current_stage().Traverse()
                    if "/env_0/" in str(prim.GetPath()) and ("LeftArm" in str(prim.GetPath()) or "left_robot" in str(prim.GetPath()))
                    and prim.GetName() in ("tip_left", "tip_right") and prim.HasAPI(UsdPhysics.RigidBodyAPI)]
            if tips:
                _TR["pin_view"] = sv.create_rigid_contact_view([card], filter_patterns=[tips], max_contact_data_count=64)
            print(f"SC pin monitor: {card} against {tips}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"SC pin monitor error: {e!r}", flush=True)


def _track(dt):
    """One physics step: read the quarter-card contacts just simulated and apply the scratch rule."""
    if not _TR["armed"] or _TR["env"] is None:
        return
    try:
        if _TR["view"] is None:
            _make_view(_TR["mgr"])
        forces, pts, nrm, sep, counts, starts = _TR["view"].get_contact_data(dt=dt)
        n, s0 = int(_np(counts).reshape(-1)[0]), int(_np(starts).reshape(-1)[0])
        f = _np(forces).reshape(-1)[s0:s0 + n].astype(np.float64)
        p = _np(pts).reshape(-1, 3)[s0:s0 + n].astype(np.float64)
        c, rc = _pose("quarter")
        k, rk = _pose("card")
    except Exception as e:  # noqa: BLE001
        if _TR["err"] is None:
            _TR["err"] = repr(e)
            print(f"SC tracker error: {e!r}", flush=True)
        return
    _TR["polls"] += 1
    scratch_step(dt, f, p, c, rc, k, rk)
    if DIAG and _TR.get("pin_view") is not None:
        try:
            pf, _, _, _, pc, ps = _TR["pin_view"].get_contact_data(dt=dt)
            pn, p0 = int(_np(pc).reshape(-1)[0]), int(_np(ps).reshape(-1)[0])
            _TR["pin_f"] = abs(float(_np(pf).reshape(-1)[p0:p0 + pn].astype(np.float64).sum()))
            acc = _TR.setdefault("pin_acc", [0.0, 0, 0.0, 0])  # sum, steps, max, steps pressing (> 0.05 N)
            acc[0] += _TR["pin_f"]
            acc[1] += 1
            acc[2] = max(acc[2], _TR["pin_f"])
            acc[3] += _TR["pin_f"] > 0.05
        except Exception as e:  # noqa: BLE001
            if _TR.get("pin_err") is None:
                _TR["pin_err"] = repr(e)
                print(f"SC pin read error: {e!r}", flush=True)
    if DIAG and len(f) and f.sum() > 0 and _TR.get("fine", 0) < 400:  # per-step contact detail, the first presses only
        _TR["fine"] = _TR.get("fine", 0) + 1
        if _TR["fine"] % 2 == 0:
            na = rc[:, 2] if rc[2, 2] <= 0 else -rc[:, 2]  # the coin's axis, pointing down
            d = -(np.array([0.0, 0.0, 1.0]) - na[2] * na)
            d /= np.linalg.norm(d) + 1e-12
            e = np.cross(na, d)
            P = (f[:, None] * p).sum(0) / f.sum() - c
            nn = _np(nrm).reshape(-1, 3)[s0:s0 + n].astype(np.float64)
            nz = float((f * nn[:, 2]).sum() / f.sum())
            print(f"SC_FINE t={_REC['t']:.4f} f={f.sum():.2f} n_pts={len(f)} P_mm(down,side,axis)=({P @ d * 1000:.2f},"
                  f"{P @ e * 1000:.2f},{P @ na * 1000:.2f}) normal_z={nz:.2f} tilt={math.degrees(math.asin(min(1.0, abs(float(na[2]))))):.1f}",
                  flush=True)


def scratch_step(dt, f, p, c, rc, k, rk, rec=None):
    """The scratch rule for one physics step. f: contact normal forces (N) between the quarter and the card, p: their
    points; (c, rc), (k, rk): the quarter's and the card's poses. Only contacts on the coin's rim count, and only while
    the coin stands on edge; the force is averaged over the last F_WINDOW steps."""
    rec = _REC if rec is None else rec
    rec["t"] += dt
    t = rec["t"]
    n_axis, m = rc[:, 2], rk[:, 2]
    f_rim, foot = 0.0, None
    if len(f) and f.sum() > 0:
        tot = float(f.sum())
        P = (f[:, None] * p).sum(0) / tot
        r = P - c
        radial = np.linalg.norm(r - np.dot(r, n_axis) * n_axis)
        on_edge = abs(float(np.dot(n_axis, m))) <= math.cos(math.radians(ON_EDGE_DEG))
        if radial >= COIN_R - RIM_BAND and on_edge:
            f_rim, foot = tot, P
    buf = rec.setdefault("buf", [])
    buf.append(f_rim)
    del buf[:-F_WINDOW]
    f_avg = float(np.mean(buf)) if len(buf) == F_WINDOW else 0.0
    hit = np.zeros(N_CHARS, bool)
    if foot is not None and f_avg >= F_MIN:
        rec["rim_steps"] += 1
        rec["peak"] = max(rec["peak"], f_avg)
        rec["forces"].append(f_avg)
        local = rk.T @ (foot - k)
        along = rk.T @ np.cross(n_axis, m)  # the rim's edge direction, in the card's plane
        along = along[:2] / (np.linalg.norm(along[:2]) + 1e-12)
        across = np.array([-along[1], along[0]])
        d = CELLS - local[:2]
        under = (np.abs(d @ along) <= FOOT_ALONG) & (np.abs(d @ across) <= FOOT_ACROSS)
        if under.any():
            if rec["cells"][under].any():
                rec["dirty"] = True
            rec["cells"][under] = False
            ks = np.unique(CELL_CHAR[under])
            hit[ks[ks >= 0]] = True
    new = hit & ~rec["in_pass"]
    rec["passes"][new] += 1
    rec["in_pass"] |= hit
    rec["last_hit"][hit] = t
    if f_avg > F_MAX:
        rec["hard"] |= hit
    rec["in_pass"] &= (t - rec["last_hit"]) <= PASS_GAP_S
    return f_avg


def _reset_record():
    _REC.clear()
    _REC.update(blank_record())
    _TR["fine"] = 0
    _TR["pin_f"] = 0.0
    _TR["pin_acc"] = [0.0, 0, 0.0, 0]


# -- the sticker and the code, drawn on the card (visual only; the scorer reads the record, never the graphics)
LIFT_CODE, LIFT_STK = 0.00002, 0.00006
FONT = {  # 5 x 7 dots per character, top row first
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"], "2": ["01110", "10001", "00001", "00010",
    "00100", "01000", "11111"], "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"], "7": ["11111", "00001", "00010", "00100",
    "01000", "01000", "01000"], "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"], "C": ["01110", "10001", "10000", "10000",
    "10000", "10001", "01110"], "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"], "M": ["10001", "11011", "10101", "10101",
    "10001", "10001", "10001"], "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"], "R": ["11110", "10001", "10001", "11110",
    "10100", "10010", "10001"], "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"], "-": ["00000", "00000", "00000", "11111",
    "00000", "00000", "00000"]}


def code_dots():
    """Card-frame centres of the printed dots (5 x 7 per character, 0.4 x 0.357 mm each)."""
    out = []
    dx, dy = CHAR_W / 5, CHAR_H / 7
    for k, ch in enumerate(CODE):
        cx = (k - (len(CODE) - 1) / 2) * PITCH
        for row, bits in enumerate(FONT[ch]):
            for col, b in enumerate(bits):
                if b == "1":
                    out.append((cx + (col - 2) * dx, STK_Y + (3 - row) * dy))
    return np.array(out), (dx, dy)


def _look(stage, path, color, roughness, metallic):
    from pxr import Gf, Sdf, UsdShade
    from yam_arena.station import linear_rgb
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr().Set("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*linear_rgb(color)))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _instancer(stage, path, positions, scale, mat):
    from pxr import Gf, UsdGeom, UsdShade, Vt
    inst = UsdGeom.PointInstancer.Define(stage, path)
    cube = UsdGeom.Cube.Define(stage, path + "/Prototype")
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    inst.CreatePrototypesRel().SetTargets([cube.GetPath()])
    inst.CreateProtoIndicesAttr().Set(Vt.IntArray([0] * len(positions)))
    inst.CreateIdsAttr().Set(Vt.Int64Array(list(range(len(positions)))))
    inst.CreatePositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, q)) for q in positions]))
    return inst


def author_card(stage, root):
    """The printed code (black dots) and the sticker's coating (silver cells) on the card's top face."""
    from pxr import UsdGeom
    from pxr import Vt
    UsdGeom.Xform.Define(stage, root + "/Print")
    dots, (dx, dy) = code_dots()
    z0 = CARD_T / 2
    _instancer(stage, root + "/Print/Code", [(x, y, z0 + LIFT_CODE / 2) for x, y in dots], (dx * 0.9, dy * 0.9,
               LIFT_CODE), _look(stage, root + "/Print/Ink", (0.03, 0.03, 0.04), 0.6, 0.0))
    inst = _instancer(stage, root + "/Print/Coating", [(x, y, z0 + LIFT_STK / 2) for x, y in CELLS],
                      (CELL * 1.02, CELL * 1.02, LIFT_STK), _look(stage, root + "/Print/Silver", (0.74, 0.75, 0.77),
                                                                 0.45, 0.55))
    inst.CreateInvisibleIdsAttr().Set(Vt.Int64Array([]))
    dash = 0.0025  # the perforated tear line: 2.5 mm cuts every 4 mm (drawn, not cut)
    xs = np.arange(-PLATE_X / 2 + 0.002, PLATE_X / 2 - 0.002, 0.004) + dash / 2
    _instancer(stage, root + "/Print/Tear", [(x, TEAR_Y, z0 + LIFT_CODE / 2) for x in xs], (dash, 0.0004, LIFT_CODE),
               _look(stage, root + "/Print/Cut", (0.55, 0.55, 0.55), 0.7, 0.0))
    if DIAG:
        from pxr import PhysxSchema
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(root)).CreateThresholdAttr().Set(0.0)
    _TR["inst"], _TR["shown"] = inst, np.ones(len(CELLS), bool)
    _TR["paths"]["card"] = root
    print(f"SC card print: {len(dots)} dots, {len(CELLS)} coating cells at {root}", flush=True)


def author_coin(stage, root):
    from pxr import PhysxSchema
    PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(root)).CreateThresholdAttr().Set(0.0)
    _TR["paths"]["quarter"] = root


def _show_coating():
    inst = _TR["inst"]
    if inst is None or not _REC.get("dirty"):
        return
    from pxr import Vt
    _REC["dirty"] = False
    if np.array_equal(_TR["shown"], _REC["cells"]):
        return
    _TR["shown"] = _REC["cells"].copy()
    inst.GetInvisibleIdsAttr().Set(Vt.Int64Array(np.flatnonzero(~_REC["cells"]).tolist()))


def _wrap_spawn(obj, after):
    from isaaclab.sim.utils import clone
    original = getattr(obj.spawner_cfg.func, "__wrapped__", obj.spawner_cfg.func)

    def spawn(path, cfg, *args, **kwargs):
        prim = original(path, cfg, *args, **kwargs)
        after(prim.GetStage(), str(prim.GetPath()))
        return prim
    obj.spawner_cfg.func = clone(spawn)
    obj.object_cfg.spawn.func = obj.spawner_cfg.func


_TOKEN = object()


def _patch_scene(scene):
    yam_mesh._patch_scene(scene)
    if getattr(scene, "_scratch_token", None) is _TOKEN:
        return
    inner = scene.task_object

    @functools.wraps(inner)
    def task_object(spec):
        obj = inner(spec)
        if spec is CARD:
            _wrap_spawn(obj, author_card)
        elif spec is COIN:
            _wrap_spawn(obj, author_coin)
        return obj

    scene.task_object, scene._scratch_token = task_object, _TOKEN


class _SceneFinder(importlib.abc.MetaPathFinder):
    _scratch_finder = True

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "yam_arena.scene":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = yam_parts._SceneLoader(spec.loader, _patch_scene)
        return spec


_PHYSX_MGR = "isaaclab_physx.physics.physx_manager"


def _patch_manager(mod):
    mgr = mod.PhysxManager
    if getattr(mgr.step, "_scratch_hooked", False):
        return
    base_cfg, base_step = mgr._configure_physics.__func__, mgr.step.__func__

    def _configure_physics(cls):
        base_cfg(cls)
        _enable_contact_processing()

    def step(cls):
        base_step(cls)
        _TR["mgr"] = cls
        try:
            dt = float(cls.get_physics_dt())
        except Exception:  # noqa: BLE001
            dt = 1.0 / 240.0
        _track(dt)

    step._scratch_hooked = True
    mgr._configure_physics = classmethod(_configure_physics)
    mgr.step = classmethod(step)


class _MgrFinder(importlib.abc.MetaPathFinder):
    """Patches Isaac Lab's PhysxManager when it is first imported (it needs Kit, so not at task import)."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _PHYSX_MGR:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _patch_manager(module)

        spec.loader.exec_module = exec_module
        return spec


def _install():
    if not any(getattr(f, "_scratch_finder", False) for f in sys.meta_path):
        sys.meta_path.insert(0, _SceneFinder())
    if "yam_arena.scene" in sys.modules:
        _patch_scene(sys.modules["yam_arena.scene"])
    if _PHYSX_MGR in sys.modules:
        _patch_manager(sys.modules[_PHYSX_MGR])
    elif not any(isinstance(f, _MgrFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MgrFinder())
    try:
        from yam_arena import inspect_embodiment as ie
    except ImportError:  # offline
        return
    cls = ie.YamArenaEmbodiment
    if getattr(cls.reset, "_scratch_hooked", False):
        return
    base_reset = cls.reset

    def reset(self, *a, **k):
        _TR["armed"] = False
        out = base_reset(self, *a, **k)
        _TR["env"] = self._env
        _reset_record()
        _TR["armed"] = True
        return out

    reset._scratch_hooked = True
    cls.reset = reset


_install()

# ---------------------------------------------------------------- demo
# Right arm alone: pinch the quarter off the table by its rim (pointing straight down, fingers closing along the code
# line, the coin's centre 4.5 mm off the pads' middle), lift it, turn the hand down and out so the coin stands on edge in
# front of the fingertips, its face 26 deg off vertical. Then 24 strokes across the code, 2.3 mm apart along it (the
# first two left of the code, where the hand's corrections settle): the rim comes down onto the card already moving back
# toward the arm (so the card's friction on it always points away from the arm and the coin's lean holds), presses in
# 4.5 mm beyond the line, is drawn back across the line to 9 mm short of it, and lifts while still moving back. Under
# the drag the arm gives a little sideways, so each rim track runs ~3 mm along the line during the pull; every stroke
# is aimed PULL_DRIFT short of its place to cross the code where planned. Meanwhile the left hand, fingers closed and
# tilted 25 deg about their closing axis, presses the hanger's far end flat 130 mm beyond the code line (the pads' edge
# commanded 5.2 mm into the 0.76 mm plate, so the arm's compliance presses it with a few newtons) from before the first
# stroke until the coin is lifted clear. Every stroke reads where the card and the coin really are, and trims the
# hand's pitch (in the air) to keep the coin's face at 26 deg.
PAD_TIP = 0.013
TIP_Z = 0.0005
TIP_LOW = 0.008
GRASP_DY, GRASP_DZ = 0.0045, PAD_TIP + TIP_Z - COIN_T / 2  # the coin's centre in the hand frame at the pick
GRIP_OPEN = -0.020
GRIP_PINCH = -0.0105  # pads commanded 21 mm apart on the 24.26 mm coin: ~8 N
GRIP_HOLD = -0.007  # then, still on the table, squeeze harder: pads commanded 14 mm apart, ~25 N
PITCH_DEG = 26.0  # the coin's face off vertical while scratching (the hand pitched this far below level)
PITCH_TRIM = (-6.0, 15.0)  # deg: how far the hand's pitch may be trimmed to keep the coin at PITCH_DEG
LEAD = 0.005  # the coin's lowest point commanded this far into the card
HOVER = 0.006  # the rim this far above the card between strokes
APPROACH = 0.025  # stroke 1 comes down to its hover from this high, to settle before the first read
SET_H, SET_BACK = 0.0025, 0.003  # the rim waits 2.5 mm up, 3 mm beyond the stroke's start, then lands moving back
LIFT_BACK = 0.002  # and lifts at the end while still moving back this far
S_START, S_END = 0.0045, -0.009  # stroke start / end, across the line (+ beyond it, - toward the near edge)
PULL_DRIFT = 0.0028  # how far the rim's track runs along +u during a pull (measured on the GPU: 2.4-3.2 mm)
STROKE_U = np.arange(-28.75, 24.16, 2.3) * 0.001 - PULL_DRIFT  # 24 strokes along the line, each aimed short by the drift
CODE_FROM = 3  # the stroke place that crosses the code's first character (the wrong way starts its strokes here)
SCRUB_REPEAT = 4  # wrong way "scrub": each of 4 stroke places drawn 4 times over (a 4th pass wears a character)
CORR_MAX = 0.008
PARK_R = np.array([0.20, -0.30, TOP + 0.20])
RISE_Z = TOP + 0.120
_READS = {"episode": None, "values": {}}
_DEMO = {"calls": 0, "way": DEMO_WAY}


def _down():
    from yam_arena.demo import grip_rotation
    return np.asarray(getattr(grip_rotation, "__wrapped__", grip_rotation)(0.0), float)


def _values():
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    return _READS["values"]


def _objects_now(names):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        return arena_scene.object_states(env, list(names))
    except Exception as e:  # noqa: BLE001
        print(f"SC_READ failed: {e!r}", flush=True)
        return None


def read_card(key):
    """(origin, yaw) of the card, read once per key per episode."""
    v = _values()
    if key not in v:
        st = _objects_now(["card"])
        if st is None:
            v[key] = (np.array([CARD_XY[0], CARD_XY[1], TOP + CARD_T / 2]), CARD_YAW)
        else:
            o, r = pose(st["card"])
            v[key] = (o, math.atan2(r[1, 0], r[0, 0]))
            if DIAG:
                print(f"SC_READ {key} t={_RUN['t']:.2f} card_off_mm={tuple(round(float(x) * 1000, 1) for x in o[:2] - CARD_XY)}"
                      f" yaw_err_deg={math.degrees((v[key][1] - CARD_YAW + math.pi) % (2 * math.pi) - math.pi):.2f}", flush=True)
    return v[key]


def card_point(key, x, y, z=CARD_T / 2):
    o, yaw = read_card(key)
    return o + _rz(yaw) @ np.array([x, y, z])


def read_coin(key):
    v = _values()
    if key not in v:
        st = _objects_now(["quarter"])
        if st is None:
            v[key] = np.array([COIN_XY[0], COIN_XY[1], TOP + COIN_T / 2])
        else:
            v[key] = np.asarray(st["quarter"].origin, float)
    return v[key]


def coin_low_point(c, n):
    """Lowest point of a coin (centre c, axis n): its rim's lowest point on the lower face's edge."""
    z = np.array([0.0, 0.0, 1.0])
    d = -(z - np.dot(z, n) * n)
    nd = np.linalg.norm(d)
    d = d / nd if nd > 1e-9 else np.array([1.0, 0.0, 0.0])
    face = -np.sign(n[2]) * n if abs(n[2]) > 1e-9 else np.zeros(3)
    return c + COIN_R * d + COIN_T / 2 * face


def rotations(card_yaw, pitch_deg=PITCH_DEG):
    """(pick, pitched, half) hand rotations: fingers closing along the code line, the hand pointing across it, along
    the card's +y (V_DIR, away from the near long edge); pitched: pointing pitch_deg below level."""
    head = card_yaw + math.pi / 2
    pick = _rz(head) @ _down()
    pitched = _rz(head) @ _ry(-(math.pi / 2 - math.radians(pitch_deg))) @ _down()
    half = _rz(head) @ _ry(-(math.pi / 2 - math.radians(PITCH_DEG)) / 2) @ _down()
    return pick, pitched, half


def planned_low(tcp, rot):
    c = tcp + rot @ np.array([0.0, GRASP_DY, GRASP_DZ])
    return coin_low_point(c, rot[:, 2])


def stroke_correction(k):
    """(position, pitch) corrections for stroke k, read once as the stroke starts down: the previous stroke's, plus
    where the coin's lowest point really is (the hand settled over this stroke's start) minus where it should be, and
    the coin's tilt below level minus PITCH_DEG (the hand's pitch is trimmed to keep the coin's face at PITCH_DEG off
    vertical). Stroke -1: none."""
    if k < 0:
        return np.zeros(3), 0.0
    v = _values()
    key = f"corr{k}"
    if key not in v:
        prev, pprev = stroke_correction(k - 1)
        low_want = stroke_low(k, "over")
        st = _objects_now(["quarter"])
        if st is None:
            v[key] = (prev, pprev)
        else:
            q = st["quarter"]
            c, n = np.asarray(q.origin, float), np.asarray(q.rotation(), float)[:, 2]
            err = coin_low_point(c, n) - low_want
            tilt = math.degrees(math.asin(min(1.0, abs(float(n[2])))))
            dp = PITCH_DEG - tilt
            if np.linalg.norm(err) > 0.030:
                print(f"SC_READ stroke {k + 1}: coin not in hand (off {np.linalg.norm(err) * 1000:.0f} mm), no correction",
                      flush=True)
                err, dp = np.zeros(3), 0.0
            v[key] = (np.clip(prev + err, -CORR_MAX, CORR_MAX), float(np.clip(pprev + dp, *PITCH_TRIM)))
            if DIAG:
                fs = np.asarray(_REC["forces"][_TR.get("f_mark", 0):], float)
                _TR["f_mark"] = len(_REC["forces"])
                if len(fs):
                    print(f"SC_FORCE before stroke {k + 1}: {len(fs)} pressing steps, mean {fs.mean():.2f} N, "
                          f"p10 {np.percentile(fs, 10):.2f}, p90 {np.percentile(fs, 90):.2f}, max {fs.max():.2f}", flush=True)
                acc = _TR.get("pin_acc") or [0.0, 0, 0.0, 0]
                print(f"SC_PIN before stroke {k + 1}: mean {acc[0] / max(1, acc[1]):.2f} N over {acc[1]} steps, "
                      f"pressing {acc[3]}, max {acc[2]:.2f} N", flush=True)
                _TR["pin_acc"] = [0.0, 0, 0.0, 0]
                print(f"SC_READ stroke {k + 1} t={_RUN['t']:.2f} coin_low_err_mm={tuple(round(float(x) * 1000, 2) for x in err)}"
                      f" corr_mm={tuple(round(float(x) * 1000, 2) for x in v[key][0])} coin_axis_below_level_deg={tilt:.1f}"
                      f" pitch_trim_deg={v[key][1]:.1f}", flush=True)
    return v[key]


def stroke_u(k):
    """Where along the code line stroke k runs (the wrong way "scrub" draws each place SCRUB_REPEAT times)."""
    if _DEMO["way"] == "scrub":
        return STROKE_U[CODE_FROM + k // SCRUB_REPEAT]
    return STROKE_U[k]


def n_strokes():
    return 4 * SCRUB_REPEAT if _DEMO["way"] == "scrub" else len(STROKE_U)


def stroke_low(k, kind):
    """Where the coin's lowest point should be for stroke k at one of its stages (world)."""
    key = f"card{k}"
    u = stroke_u(k)
    vstart, vend = STK_Y + S_START, STK_Y + S_END
    vset, vlift = vstart + SET_BACK, vend - LIFT_BACK
    return {"approach": card_point(key, u, vset, CARD_T / 2 + APPROACH),
            "over": card_point(key, u, vset, CARD_T / 2 + HOVER), "set": card_point(key, u, vset, CARD_T / 2 + SET_H),
            "down": card_point(key, u, vstart, CARD_T / 2 - LEAD),
            "pull": card_point(key, u, vend, CARD_T / 2 - LEAD), "up": card_point(key, u, vlift, CARD_T / 2 + HOVER),
            "clear": card_point(key, u, vlift, CARD_T / 2 + 0.040)}[kind]


def stroke_target(k, kind):
    """(tcp, rotation) of the right hand for stroke k at one of its stages. "approach", "over" and "hold" use the
    previous stroke's corrections; computing "set" reads the coin (after "hold") and corrects the rest of the stroke, so
    the hand's pitch is trimmed in the air."""
    _, yaw = read_card(f"card{k}")
    corr, trim = stroke_correction(k - 1 if kind in ("approach", "over", "hold") else k)
    _, pitched, _ = rotations(yaw, PITCH_DEG + trim)
    low = stroke_low(k, "over" if kind == "hold" else kind)
    return low - planned_low(np.zeros(3), pitched) - corr, pitched


def pick_target(kind):
    _, yaw = read_card("card0")
    pick, pitched, half = rotations(yaw)
    local = np.array([0.0, GRASP_DY, GRASP_DZ])
    c = read_coin("coin_start")
    tcp = np.array([c[0], c[1], TOP + COIN_T / 2]) - pick @ local
    if kind == "above":
        return np.array([tcp[0], tcp[1], RISE_Z]), pick
    if kind == "over":
        return np.array([tcp[0], tcp[1], TOP + 0.060]), pick
    if kind == "low":
        return np.array([tcp[0], tcp[1], TOP + PAD_TIP + TIP_LOW]), pick
    if kind == "pick":  # sideways only: correct by where the hand really was at "low"; the height is sent as is
        low = np.array([tcp[0], tcp[1], TOP + PAD_TIP + TIP_LOW])
        err = read_hand("hand_low", low) * np.array([1.0, 1.0, 0.0])
        return np.array([tcp[0], tcp[1], TOP + PAD_TIP + TIP_Z]) - err, pick
    if kind == "lift":
        return np.array([tcp[0], tcp[1], TOP + 0.060]), pick
    if kind == "turn":  # high over the card's near side, half turned
        p = card_point("card0", 0.0, STK_Y - 0.060, 0.0)
        return np.array([p[0], p[1], RISE_Z]), half
    raise ValueError(kind)


def read_hand(key, commanded, limit=0.004):
    """Where the right hand's centre point really is, minus where it was sent (clamped), read once."""
    v = _values()
    if key not in v:
        hands = hand_points() if _RUN["env"] is not None else None
        if not hands:
            v[key] = np.zeros(3)
        else:
            err = np.asarray(hands["right"][0], float) - np.asarray(commanded, float)
            v[key] = np.clip(err, -limit, limit)
            if DIAG:
                print(f"SC_READ {key} t={_RUN['t']:.2f} hand_err_mm={tuple(round(float(x) * 1000, 2) for x in err)}",
                      flush=True)
    return v[key]


class RightWaypoint(Waypoint):
    def __init__(self, duration, kind, finger, label, k=None, keep=False):
        self.duration = duration * (1.0 if keep else DEMO_PACE)
        self.kind, self.finger, self.label, self.k = kind, finger, label, k

    def _target(self):
        return stroke_target(self.k, self.kind) if self.k is not None else pick_target(self.kind)

    @property
    def pos(self):
        return tuple(float(x) for x in self._target()[0])

    @property
    def yaw(self):
        return Grip(0.0, self._target()[1])


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

STROKE = [("over", 0.45), ("hold", 0.25), ("set", 0.35), ("down", 0.60), ("pull", 0.60), ("up", 0.30)]


def right_plan():
    O, P, H = GRIP_OPEN, GRIP_PINCH, GRIP_HOLD
    w = [RightWaypoint(1.5, "above", O, "high over the quarter"),
         RightWaypoint(1.0, "over", O, "down over the quarter"),
         RightWaypoint(1.0, "low", O, "pad tips 8 mm over the table"),
         RightWaypoint(0.25, "low", O, "hold low"),
         RightWaypoint(0.6, "pick", O, "pad tips to the table round it"),
         RightWaypoint(0.8, "pick", P, "pinch the rim", keep=True),
         RightWaypoint(0.25, "pick", P, "settle"),
         RightWaypoint(0.5, "pick", H, "squeeze harder, the coin still flat on the table", keep=True),
         RightWaypoint(0.25, "pick", H, "settle"),
         RightWaypoint(0.8, "lift", H, "lift"),
         RightWaypoint(1.5, "turn", H, "over the card, turning the coin on edge"),
         RightWaypoint(1.5, "approach", H, "stroke 1: rim 25 mm over the start", k=0),
         RightWaypoint(0.8, "over", H, "stroke 1: rim over the start", k=0)]
    n = n_strokes()
    for k in range(n):
        for kind, dur in STROKE:
            if k == 0 and kind == "over":
                continue
            w.append(RightWaypoint(dur, kind, H, f"stroke {k + 1}: {kind}", k=k))
    w.append(RightWaypoint(1.0, "clear", H, "lift clear", k=n - 1))
    w.append(Waypoint(1.5 * DEMO_PACE, tuple(float(x) for x in PARK_R), GRIP_HOLD, "park, still holding the quarter", 0.0))
    w.append(Waypoint(1.5, tuple(float(x) for x in PARK_R), GRIP_HOLD, "hold clear", 0.0))
    return w


# The left hand: fingers closed (pads touching), the straight-down grip closing along the hanger (grip yaw = card yaw;
# the wrist roll keeps 60 deg of margin), tilted PIN_TILT about the closing axis so the pads' edge on the robot's side
# is lowest and presses the plate (the wrist leans back). The TCP is sent PIN_TCP_Z over the table, so that edge is
# commanded PIN_LEAD (5.2 mm) into the 0.76 mm plate: a deliberate fingertip press. The arm stops on the plate and
# presses it through its own compliance (measured on the GPU: the hand stops ~5 mm above its target and presses the
# hanger with about 1.9 N). The offline check allows a press this deep only over a body
# listed in PRESSED_PARTS (warning press_below_floor).
PIN_TILT = 25.0
PIN_TCP_Z = 0.00955
PIN_ABOVE = 0.025
PAD_HALF_W, PAD_TIP_Z = 0.0065, 0.0124  # closed pads' rounded tip corner (URDF meshes): 6.5 mm off centre, 12.4 mm out
PARK_L = np.array([0.24, 0.24, TOP + 0.20])


def pin_rotation(card_yaw):
    r0 = _rz(card_yaw) @ _down()
    best = None
    for s in (1.0, -1.0):
        r = _aa(r0[:, 0], s * math.radians(PIN_TILT)) @ r0
        edge = min((r @ np.array([0.0, e, PAD_TIP_Z]) for e in (-PAD_HALF_W, PAD_HALF_W)), key=lambda p: p[2])
        if best is None or edge[0] < best[1][0]:  # the low edge toward -x (the robot's side), the wrist leaning back
            best = (r, edge)
    return best


PIN_LEAD = CARD_T - (PIN_TCP_Z + pin_rotation(0.0)[1][2])  # how far the low edge is commanded into the plate


def pin_target(kind):
    """(tcp, rotation) of the left hand: the pads' low edge centred on the pin point PIN_V beyond the code line."""
    _, yaw = read_card("card0")
    r, edge = pin_rotation(yaw)
    p = card_point("card0", 0.0, STK_Y + PIN_V, 0.0)
    tcp = np.array([p[0] - edge[0], p[1] - edge[1], TOP + PIN_TCP_Z])
    if kind == "above":
        return tcp + np.array([0.0, 0.0, PIN_ABOVE]), r
    if kind == "press":
        return tcp, r
    if kind == "off":
        return tcp + np.array([0.0, 0.0, 0.050]), r
    raise ValueError(kind)


class LeftWaypoint(RightWaypoint):
    def _target(self):
        return pin_target(self.kind)


def left_plan(right):
    """Pinned from 6.5 s (before the coin reaches the card at ~10 s) until the coin has been lifted clear."""
    t_clear = sum(w.duration for w in right[:-2])  # the right hand's "lift clear" ends here
    c = 0.0  # fingers closed, pads touching
    w = [LeftWaypoint(3.0, "above", c, "close the fingers, over the hanger's far end", keep=True),
         LeftWaypoint(1.5, "above", c, "steady over the hanger", keep=True),
         LeftWaypoint(1.2, "press", c, "press the hanger flat", keep=True),
         LeftWaypoint(0.8, "press", c, "settle", keep=True)]
    w.append(LeftWaypoint(max(1.0, t_clear - sum(x.duration for x in w)), "press", c, "hold the hanger down", keep=True))
    w.append(LeftWaypoint(0.8, "off", c, "lift off the hanger", keep=True))
    w.append(Waypoint(1.5, tuple(float(x) for x in PARK_L), c, "park", 0.0))
    w.append(Waypoint(max(0.5, sum(x.duration for x in right) - sum(x.duration for x in w)), tuple(float(x) for x in PARK_L),
                      c, "hold clear", 0.0))
    return w


def demo():
    _DEMO["calls"] += 1
    _DEMO["way"] = PROBE_WAYS[(_DEMO["calls"] - 1) % len(PROBE_WAYS)] if PROBE_WAYS else DEMO_WAY
    right = right_plan()
    if _DEMO["way"] == "nopin":
        return {"right": right}
    return {"right": right, "left": left_plan(right)}


# ---------------------------------------------------------------- offline test layouts (states.yaml)
def card_quat(turn_deg=0.0, flip=False):
    from yam_arena.ik import matrix_to_quat_xyzw
    rot = _rz(CARD_YAW + math.radians(turn_deg)) @ (np.diag([1.0, -1.0, -1.0]) if flip else np.eye(3))
    return [float(x) for x in matrix_to_quat_xyzw(rot)]


def card_at(dx=0.0, dy=0.0):
    return [float(CARD_XY[0] + dx), float(CARD_XY[1] + dy), float(TOP + CARD_T / 2)]


def over_card(h):
    """A point h above the card's centre (a hand still there)."""
    return np.array([CARD_XY[0], CARD_XY[1], TOP + CARD_T + h])


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


def _diag(objects, rec, readable, worn, moved, turn):
    step = _RUN["step"]
    if step == _DIAG["step"] or step % 4:
        return
    _DIAG["step"] = step
    if step % 100 == 0:  # every 5 s: which characters still show silver, and where (card frame, mm along the line)
        left = []
        for k in range(N_CHARS):
            c = CHAR_CELLS[k][rec["cells"][CHAR_CELLS[k]]]
            if len(c):
                left.append(f"{k}:{len(c)}@{CELLS[c, 0].min() * 1000:.1f}..{CELLS[c, 0].max() * 1000:.1f}")
        print(f"SC_CHARS t={_RUN['t'] or 0.0:.2f} covered={left} passes={rec['passes'].tolist()}", flush=True)
    q = objects["quarter"]
    c, n = np.asarray(q.origin, float), np.asarray(q.rotation(), float)[:, 2]
    low = coin_low_point(c, n)
    buf = rec.get("buf", [])
    hands = hand_points() or {}
    h = " ".join(f"{a}=({','.join(f'{x * 1000:.1f}' for x in hands[a][0])})" for a in hands)
    ang = ""
    env = _RUN["env"]
    if env is not None:  # the coin's axis against the right hand's pointing axis, turned about the closing axis
        from yam_arena import ik, station
        from yam_arena.constants import ARM_JOINTS
        art = env.unwrapped.scene["right_robot"]
        qr = _np(art.data.joint_pos)[0][art.find_joints(ARM_JOINTS, preserve_order=True)[0]]
        _, hr = ik.tcp_pose(np.asarray(qr, float), station.BASES["right"])
        nn = n if np.dot(n, hr[:, 2]) >= 0 else -n
        ang = (f" coin_in_hand_deg(about closing, about width)=({math.degrees(math.atan2(np.dot(nn, hr[:, 1]), np.dot(nn, hr[:, 2]))):.1f},"
               f"{math.degrees(math.atan2(np.dot(nn, hr[:, 0]), np.dot(nn, hr[:, 2]))):.1f})")
    ko, kr = pose(objects["card"])
    loc = kr.T @ (low - ko)
    ang += f" low_card_mm=({loc[0] * 1000:.1f},{(loc[1] - STK_Y) * 1000:.1f},{(loc[2] - CARD_T / 2) * 1000:.2f})"
    print(f"SC_DIAG t={_RUN['t'] or 0.0:.2f} f_now={buf[-1] if buf else 0.0:.2f} peak={rec['peak']:.2f} "
          f"rim_steps={rec['rim_steps']} coated={int(rec['cells'].sum())} readable={int(readable.sum())} "
          f"worn={int(worn.sum())} passes={rec['passes'].tolist()} coin_low_mm=({(low[0]) * 1000:.1f},"
          f"{low[1] * 1000:.1f},{(low[2] - TOP) * 1000:.2f}) coin_axis_below_level={math.degrees(math.asin(min(1.0, abs(float(n[2]))))):.1f} "
          f"card_moved_mm={moved * 1000:.1f} card_turn={turn:.2f} pin_N={_TR.get('pin_f', 0.0):.2f} polls={_TR['polls']} err={_TR['err']} {h}{ang}", flush=True)
