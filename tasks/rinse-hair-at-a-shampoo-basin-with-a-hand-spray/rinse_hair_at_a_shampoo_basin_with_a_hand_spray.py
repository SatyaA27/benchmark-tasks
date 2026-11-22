"""Rinse a practice mannequin head's hair at an inflatable shampoo basin with a hand spray, scalp wetted at the crown
and the front hairline, face and ears kept dry.

A hairdressing training head lies face up in an inflatable bed shampoo basin on the table, its crown toward the robot,
its top hair combed back over the crown into the basin in one tail. Water runs from a hand spray (a camping-shower head fed by a pump
in a bucket), which lies face down in the basin on the robot's left, so its water goes into the basin. Square water-
contact indicator patches are stuck on the head: two must be wetted (the scalp at the crown, under the hair that runs
over it; the scalp at the middle of the front hairline) and three must stay dry (the forehead just in front of the
hairline, the outside of each ear). Pick up the spray, wet both scalp patches, keep the face and ears dry, and lay the
spray back face down in the basin.

The water is a scored model (spray_kit.py), not a fluid: straight jets from the nozzle face, stopped by the hair, a
hand, the head or the basin floor. Hair: one capsule-chain tail (hair_kit.py). Metres; x forward from the arm bases, y to
the robot's left, z up.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
for _helper in ("hair_kit", "spray_kit", "yam_parts"):
    sys.modules.pop(_helper, None)
import hair_kit  # noqa: E402
import spray_kit as sk  # noqa: E402
import yam_parts  # noqa: E402
from hair_kit import Piece  # noqa: E402
from yam_parts import Part, box, cyl  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
RUN_LIMIT_STEPS = 2400  # 120 s at 20 Hz
DEMO_WAY = "rinse"  # "rinse"; wrong ways: "noshield" (no hand at the hairline), "toponly" (hair never moved aside)
DEMO_SEQ = None  # GPU probes only: one way per episode, e.g. ("rinse", "rinse", "rinse", "careless")
TEST_GRIPPERS = None  # offline checks: {"left"/"right": (TCP position, TCP rotation)}
DIAG = False  # True: print patch changes and a status line every second (probes)
TEST_WET = None  # offline checks: {patch name: wet?} in place of the spray's history
TEST_HAIR = None  # offline checks: [(A, B, radius)] hair capsules instead of the objects' states

# ---------------------------------------------------------------- the inflatable shampoo basin (fixed)
# 660 x 610 mm outside, 150 mm deep (published); tubes 75 mm and the inflated floor 30 mm are estimates.
BX0, BX1, BW = 0.070, 0.730, 0.305  # outside x from .. to, half width
WALL_H, FLOOR = 0.150, 0.030
TUBE = WALL_H / 2  # each wall: two stacked tubes, 75 mm across
IN_X0, IN_X1, IN_Y = BX0 + TUBE, BX1 - TUBE, BW - TUBE
PVC, PVC_FLOOR = (0.42, 0.66, 0.86), (0.78, 0.88, 0.95)


def basin_shapes():
    """Fixed Part shapes (local frame on the table under the basin): the floor and two stacked tubes per wall."""
    r = WALL_H / 4
    sh = [box(((BX0 + BX1) / 2, 0.0, FLOOR / 2), (BX1 - BX0 - 0.02, 2 * BW - 0.02, FLOOR), color=PVC_FLOOR)]
    for z in (r, 3 * r):
        sh += [cyl(((BX0 + BX1) / 2, s * (BW - r), z), r, BX1 - BX0, axis="x",
                   color=PVC) for s in (-1, 1)]
        sh += [cyl((x, 0.0, z), r, 2 * BW - 2 * r, axis="y", color=PVC) for x in (BX0 + r, BX1 - r)]
    return sh


BASIN = Part("shampoo_basin", (0.0, 0.0, TOP), basin_shapes(), mass=None, color=PVC, friction=(0.8, 0.7))


# ---------------------------------------------------------------- the training head, face up in the basin
# Head frame: origin on the basin floor below the crown (vertex), x from the crown toward the chin, z up. Adult
# training head (estimates: crown to chin 225 mm, 145 mm ear to ear, 185 mm back of the head to the brow).
CROWN_X = 0.350
HEAD_O = np.array((CROWN_X, 0.0, TOP + FLOOR))
SKIN, BROW, LIPS = (0.93, 0.79, 0.68), (0.30, 0.20, 0.14), (0.78, 0.42, 0.42)
HAIR = (0.20, 0.13, 0.08)  # dark brown synthetic hair
TAPE = (0.97, 0.97, 0.95)  # white indicator tape
CZ = 0.0925
CRANIUM = ((0.095, 0.0, CZ), (0.095, 0.0725, 0.0925))
FACE = ((0.160, 0.0, CZ + 0.032), (0.068, 0.056, 0.060))
NOSE = ((0.135, 0.0, CZ + 0.092), (0.024, 0.012, 0.014))
EARS = [((0.135, s * 0.074, CZ - 0.012), (0.031, 0.009, 0.018)) for s in (-1, 1)]


def cranium_point(hx, y=0.0):
    """(point, outward normal) on the top of the cranium at head-frame x = hx, y (head frame)."""
    (cx, cy, cz), (a, b, c) = CRANIUM
    k = 1 - ((hx - cx) / a) ** 2 - ((y - cy) / b) ** 2
    z = cz + c * math.sqrt(max(k, 0.0))
    n = np.array(((hx - cx) / a ** 2, (y - cy) / b ** 2, (z - cz) / c ** 2))
    return np.array((hx, y, z)), n / np.linalg.norm(n)


PATCH_SIDE = 0.015
W1_X, W2_X, D1_X = 0.003, 0.056, 0.090  # head-frame x of the crown, hairline and forehead patches (centre line)


def patches_head():
    """[(name, centre, normal, in-plane axis, must_wet)] in the head frame."""
    out = []
    for name, hx, wet in (("crown", W1_X, True), ("hairline", W2_X, True), ("forehead", D1_X, False)):
        p, n = cranium_point(hx)
        out.append((name, p, n, (0.0, 1.0, 0.0), wet))
    for (c, semi), side in zip(EARS, ("right_ear", "left_ear")):
        s = 1.0 if c[1] > 0 else -1.0
        out.append((side, np.array((c[0], c[1] + s * semi[1], c[2])), np.array((0.0, s, 0.0)), (1.0, 0.0, 0.0), False))
    return out


def _disc(p, n, color):
    """A thin visual square-ish patch on the surface: an ellipsoid flattened along the patch's dominant normal axis."""
    k = int(np.argmax(np.abs(n)))
    semi = [PATCH_SIDE / 2] * 3
    semi[k] = 0.0012
    return Piece("ellipsoid", tuple(float(v) for v in p + n * 0.0006), tuple(semi), color)


def head_pieces():
    ps = [
        Piece("ellipsoid", *CRANIUM, SKIN),
        Piece("ellipsoid", *FACE, SKIN),
        Piece("ellipsoid", *NOSE, SKIN),
        *[Piece("ellipsoid", c, s, SKIN) for c, s in EARS],
        Piece("ellipsoid", (0.118, -0.031, CZ + 0.0796), (0.006, 0.012, 0.003), BROW),  # painted eyes
        Piece("ellipsoid", (0.118, 0.031, CZ + 0.0796), (0.006, 0.012, 0.003), BROW),
        Piece("ellipsoid", (0.103, -0.031, CZ + 0.0823), (0.004, 0.017, 0.003), BROW),  # brows
        Piece("ellipsoid", (0.103, 0.031, CZ + 0.0823), (0.004, 0.017, 0.003), BROW),
        Piece("ellipsoid", (0.180, 0.0, CZ + 0.0883), (0.006, 0.018, 0.003), LIPS),
        Piece("cylinder", (0.225, 0.0, 0.052), (0.052, 0.120), SKIN),  # neck, to its flat base
        # hair over the sides (seen from above beside the tail); not a collider and not on any patch
        Piece("ellipsoid", (0.060, -0.050, CZ - 0.005), (0.065, 0.030, 0.080), HAIR),
        Piece("ellipsoid", (0.060, 0.050, CZ - 0.005), (0.065, 0.030, 0.080), HAIR),
        # colliders
        Piece("ellipsoid", *CRANIUM, SKIN, collide=True),
        Piece("ellipsoid", (0.165, 0.0, CZ + 0.020), (0.054, 0.054, 0.054), SKIN, collide=True),
        Piece("cylinder", (0.225, 0.0, 0.052), (0.050, 0.120), SKIN, collide=True),
    ]
    ps += [_disc(p, n, TAPE) for _, p, n, _, _ in patches_head()]
    return ps


HEAD = hair_kit.figure("training_head", tuple(HEAD_O), 0.0, head_pieces(), SKIN)
PATCHES = [sk.Patch(n, HEAD_O + p, nn, ax, PATCH_SIDE, w) for n, p, nn, ax, w in patches_head()]
MUST_WET = [p.name for p in PATCHES if p.must_wet]
MUST_DRY = [p.name for p in PATCHES if not p.must_wet]

# ---------------------------------------------------------------- the hair: one tail combed back over the crown
# The top hair, combed straight back into one loose tail held by a hair tie near its ends, rooted along the middle of
# the top just behind the hairline: back over the crown, down into the basin, along its floor toward the robot.
# Estimates: a 24 mm thick tail, 252 mm long, 250 g/m, EI 1.5e-4 N m^2, friction 0.8.
LOCK_R, PITCH, LINKS = 0.012, 0.028, 9
KG_PER_M, EI = 0.250, 1.5e-4
LOCK_SPREAD = 0.020  # the 24 mm tail lying on the scalp spreads to about 40 mm wide: the width it keeps water off
ROOT_X = 0.028


def lock_path(y, off):
    """Joint points (LINKS + 1, head frame) of the hair tail rooted at (ROOT_X, y), `off` above the scalp."""
    (cx, _, cz), (a, b, c) = CRANIUM
    f = math.sqrt(1 - (y / b) ** 2)
    a2, c2 = a * f, c * f
    phi0 = math.acos((cx - ROOT_X) / a2)
    dense = []
    for ph in np.linspace(phi0, 0.0, 200):
        x, z = cx - a2 * math.cos(ph), cz + c2 * math.sin(ph)
        n = np.array(((x - cx) / a2 ** 2, (z - cz) / c2 ** 2))
        n /= np.linalg.norm(n)
        dense.append((x + n[0] * off, z + n[1] * off))
    xv = dense[-1][0]
    zf = (LOCK_R + 0.001) if off < 2 * LOCK_R else (3 * LOCK_R + 0.002)
    rc = 0.025
    for z in np.linspace(cz, zf + rc, 60)[1:]:
        dense.append((xv, z))
    for t in np.linspace(0, math.pi / 2, 30)[1:]:
        dense.append((xv - rc + rc * math.cos(t), zf + rc - rc * math.sin(t)))
    for x in np.linspace(xv - rc, xv - rc - 0.4, 200)[1:]:
        dense.append((x, zf))
    D = np.array(dense)
    pts = [D[0]]
    i = 0
    while len(pts) < LINKS + 1:
        p = pts[-1]
        while np.linalg.norm(D[i + 1] - p) < PITCH:
            i += 1
        a_, b_ = D[i], D[i + 1]  # exact chord: |a_ + s (b_ - a_) - p| = PITCH
        d = b_ - a_
        w = a_ - p
        A, B, C = d @ d, 2 * d @ w, w @ w - PITCH ** 2
        s = (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)
        pts.append(a_ + s * d)
    return np.array([(px, y, pz) for px, pz in pts])


def _lock():
    return hair_kit.Rope(links=LINKS, pitch=PITCH, radius=LOCK_R, link_mass=KG_PER_M * PITCH, bend_k=EI / PITCH,
                         bend_d=0.1 * EI / PITCH, friction=(0.8, 0.8), color=HAIR, roughness=0.55)


_LOCKS = [("hair_tail", 0.0, LOCK_R + 0.0008)]
LOCK_NAMES = tuple(n for n, _, _ in _LOCKS)
LOCKS = [hair_kit.strand_path(n, HEAD_O + lock_path(y, off), _lock()) for n, y, off in _LOCKS]

# ---------------------------------------------------------------- the hand spray (free)
# A camping-shower hand spray: handle 25 mm across, 150 mm long, head 70 mm across, 25 mm thick, nozzle face
# underneath (part -z) when it lies flat; 120 g (estimates). Part frame: origin at the handle's centre, +x toward
# the head.
HANDLE_R, HANDLE_L, HEAD_R, HEAD_T, HEAD_CX = 0.0125, 0.130, 0.0325, 0.025, 0.095
SPRAY_MASS = 0.120
FACE_LOCAL = np.array((HEAD_CX, 0.0, -HEAD_T / 2))
GRIP_LOCAL = np.array((0.035, 0.0, 0.0))  # where the demo pinches the handle (by the spray's centre of mass)
SPRAY_START = (0.220, 0.090)  # handle centre, lying face down on the basin floor, head toward +y (the left)
SPRAY_YAW = math.pi / 2
SPRAY_WHITE, SPRAY_GREY = (0.95, 0.55, 0.10), (0.30, 0.32, 0.36)  # orange head, dark grey handle
SPRAY = Part("hand_spray", (SPRAY_START[0], SPRAY_START[1], TOP + FLOOR + HANDLE_R),
             [cyl((-0.025, 0.0, 0.0), HANDLE_R, 0.080, axis="x", color=SPRAY_GREY),
              cyl((0.035, 0.0, 0.0), HANDLE_R, 0.040, axis="x", color=SPRAY_GREY),
              cyl((HEAD_CX, 0.0, 0.0), HEAD_R, HEAD_T, axis="z", color=SPRAY_WHITE),
              cyl((-HANDLE_L / 2 - 0.005, 0.0, 0.0), 0.009, 0.010, axis="x", color=SPRAY_GREY)],
             mass=SPRAY_MASS, color=SPRAY_WHITE, yaw=SPRAY_YAW, friction=(1.0, 0.9), contact_offset=0.001, grip=1)

OBJECTS = [BASIN, HEAD, *LOCKS, SPRAY]


def _install_hooks():
    """Both kits hook yam_arena.scene; a meta-path finder serves only one, so one finder applies both patches."""
    import importlib.abc
    import importlib.machinery

    def patch(scene):
        yam_parts._patch_scene(scene)
        hair_kit._patch(scene)

    class Loader(importlib.abc.Loader):
        def __init__(self, inner):
            self.inner = inner

        def create_module(self, spec):
            fn = getattr(self.inner, "create_module", None)
            return fn(spec) if fn else None

        def exec_module(self, module):
            self.inner.exec_module(module)
            patch(module)

    class Finder(importlib.abc.MetaPathFinder):
        _hairrinse_finder = True

        def find_spec(self, fullname, path=None, target=None):
            if fullname != "yam_arena.scene":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = Loader(spec.loader)
            return spec

    sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, "_hairrinse_finder", False)]
    sys.meta_path.insert(0, Finder())
    if "yam_arena.scene" in sys.modules:
        patch(sys.modules["yam_arena.scene"])


_install_hooks()


# ---------------------------------------------------------------- run context
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


def _np(x):
    x = getattr(x, "torch", x)
    return x.detach().cpu().numpy().astype(float) if hasattr(x, "detach") else np.asarray(x, float)


def grippers():
    """{"left"/"right": (TCP position, TCP rotation)} from the arms' joint angles; None offline."""
    if TEST_GRIPPERS is not None:
        return {k: (np.asarray(p, float), np.asarray(r, float)) for k, (p, r) in TEST_GRIPPERS.items()}
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS
    out = {}
    for arm in ("left", "right"):
        art = env.unwrapped.scene[arm + "_robot"]
        ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        out[arm] = ik.tcp_pose(_np(art.data.joint_pos)[0][ids], station.BASES[arm])
    return out


# ---------------------------------------------------------------- scoring
def spray_pose(objects):
    """(origin, rotation) of the hand spray."""
    s = objects["hand_spray"]
    origin = np.asarray(getattr(s, "origin", s.pos), float)
    return origin, hair_kit.rotation(s.quat_xyzw)


def hair_capsules(objects):
    """[(A, B, r)] of every hair link."""
    if TEST_HAIR is not None:
        return TEST_HAIR
    out = []
    for n in LOCK_NAMES:
        o = objects[n]
        if not hasattr(o, "centres"):  # a plain object state (offline): the start shape
            c = hair_kit.world_centres(n)
            ax = hair_kit.registry()[n][2][:, :, 0]
        else:
            c, ax = np.asarray(o.centres, float), np.asarray(o.axes, float)
        h = PITCH / 2
        for ci, ai in zip(c, ax):
            out.append((ci - ai * h, ci + ai * h, LOCK_SPREAD))
    return out


def spray_landings(objects):
    """Points where the spray's jets land on the head this instant."""
    origin, R = spray_pose(objects)
    face = origin + R @ FACE_LOCAL
    O, D = sk.jets(face, -R[:, 2], R[:, 0])
    caps = hair_capsules(objects)
    A = np.array([c[0] for c in caps])
    B = np.array([c[1] for c in caps])
    rr = np.array([c[2] for c in caps])
    block = sk.ray_capsules(O, D, A, B, rr)
    grips = grippers()
    if grips:
        hands = sk.hand_capsules(grips)
        block = np.minimum(block, sk.ray_capsules(O, D, np.array([h[0] for h in hands]),
                                                  np.array([h[1] for h in hands]), sk.HAND_R))
    block = np.minimum(block, sk.ray_plane_z(O, D, TOP + FLOOR))
    head = np.full(len(O), np.inf)
    for c, s in (CRANIUM, FACE, NOSE, *EARS):
        head = np.minimum(head, sk.ray_ellipsoid(O, D, HEAD_O + np.asarray(c), s))
    return sk.landings(O, D, block, head)


_WET = {"key": None, "episode": None, "grid": {}}


def update(objects):
    """Wet the patches' grid points the jets land on now (once per control step)."""
    key = (_RUN["episode"], _RUN["step"])
    if _RUN["env"] is not None and _WET["key"] == key:
        return
    alone = _RUN["env"] is None and TEST_GRIPPERS is None  # an offline check of one state: judge it alone
    if _WET["episode"] != _RUN["episode"] or alone:
        _WET.update(episode=_RUN["episode"], grid={p.name: np.zeros(sk.GRID ** 2, bool) for p in PATCHES})
    _WET["key"] = key
    hits = spray_landings(objects)
    before = patch_states() if DIAG and _RUN["env"] is not None else None
    for p in PATCHES:
        near = hits[np.linalg.norm(hits - p.c, axis=1) < PATCH_SIDE + sk.SPLASH_R] if len(hits) else hits
        _WET["grid"][p.name] |= p.wet_by(near)
    if before is not None:
        after = patch_states()
        t = _RUN["step"]
        if t in (1, 5, 20, 50) or t % 100 == 0:
            for n in LOCK_NAMES:
                o = objects[n]
                c = np.asarray(getattr(o, "centres", np.full((1, 3), -1.0)))
                print(f"HL ep {_RUN['episode']} step {t} {n} {type(o).__name__} nan {int(np.isnan(c).any())} "
                      f"root {np.round(c[0] - (0, 0, TOP), 3).tolist()} end {np.round(c[-1] - (0, 0, TOP), 3).tolist()}",
                      flush=True)
            g = grippers()
            if g:
                for a, (tcp, rot) in g.items():
                    print(f"HG ep {_RUN['episode']} step {t} {a} tcp {np.round(tcp - (0, 0, TOP), 3).tolist()} "
                          f"approach {np.round(rot[:, 2], 2).tolist()}", flush=True)
        if after != before or t % 20 == 0:
            o, R = spray_pose(objects)
            w1 = PATCHES[0].c
            caps = hair_capsules(objects)
            hd = min(np.linalg.norm(0.5 * (a + b) - w1) for a, b, _ in caps)
            fr = {p.name: round(float(_WET["grid"][p.name].mean()), 2) for p in PATCHES}
            print(f"HR ep {_RUN['episode']} step {t} way {current_way()} spray {np.round(o - (0, 0, TOP), 3).tolist()} "
                  f"face_z {R[2, 2]:+.2f} hits {len(hits)} hair_to_crown {hd:.3f} wet {fr}", flush=True)


def patch_states():
    """{patch: wet?}: must-wet patches need WET_FRACTION of their area, must-dry ones fail on any wetting."""
    if TEST_WET is not None:
        return {p.name: bool(TEST_WET.get(p.name, False)) for p in PATCHES}
    out = {}
    for p in PATCHES:
        g = _WET["grid"][p.name]
        out[p.name] = bool(g.mean() >= sk.WET_FRACTION) if p.must_wet else bool(g.any())
    return out


def spray_put_back(objects):
    """The spray lies in the basin, face down, let go of (both hands' fingertips 5 cm or more from it), still."""
    origin, R = spray_pose(objects)
    s = objects["hand_spray"]
    inside = IN_X0 < origin[0] < IN_X1 and abs(origin[1]) < IN_Y and origin[2] < TOP + FLOOR + 0.03
    face_down = R[2, 2] > 0.9
    still = float(np.linalg.norm(s.lin_vel)) < 0.02
    grips = grippers()
    free = True
    if grips:
        for tcp, rot in grips.values():
            tips = np.asarray(tcp) + np.asarray(rot)[:, 2] * sk.HAND_TIP
            if np.linalg.norm(tips - (origin + R @ GRIP_LOCAL)) < 0.05:
                free = False
    return bool(inside and face_down and still and free)


def score(objects):
    """1.0: both scalp patches wet, face and ears dry, spray laid back face down in the basin. 0.75: the same with
    the spray not laid back. 0.5: one scalp patch wet, face and ears dry. 0.25: both scalp patches wet but the
    forehead or an ear wet. 0 otherwise (nothing wet; one scalp patch with the face or an ear wet)."""
    update(objects)
    st = patch_states()
    n_wet = sum(st[n] for n in MUST_WET)
    dry_ok = not any(st[n] for n in MUST_DRY)
    if n_wet == 2 and dry_ok:
        return 1.0 if spray_put_back(objects) else 0.75
    if n_wet == 1 and dry_ok:
        return 0.5
    if n_wet == 2:
        return 0.25
    return 0.0


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- scripted demonstration
G = grip_rotation(0.0)
G0 = grip_rotation(-math.pi / 2)  # the spray hand: fingers close along x, across the handle lying along y


def Ry(t):
    c, s = math.cos(t), math.sin(t)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def Rx(t):
    c, s = math.cos(t), math.sin(t)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


class Grip(float):
    """A waypoint yaw carrying a full TCP rotation (the demo's grips are pitched)."""

    def __new__(cls, rot):
        rot = np.asarray(rot, float)
        M = rot @ grip_rotation(0.0).T  # its value is the turn about the vertical, as a plain yaw would give
        obj = super().__new__(cls, math.atan2(M[1, 0], M[0, 0]))
        obj.rot = rot
        return obj


def _install_grip():
    import yam_arena.demo as dm
    old = getattr(dm.grip_rotation, "__wrapped__", dm.grip_rotation)

    def rotation(yaw=0.0):
        return yaw.rot.copy() if isinstance(yaw, Grip) else old(float(yaw))

    rotation.__wrapped__ = old
    dm.grip_rotation = rotation


_install_grip()


def pitched(deg):
    """Straight-down grip leaned forward by `deg` (fingertips toward +x)."""
    return Ry(-math.radians(deg)) @ G


def leaned(deg):
    """Straight-down grip leaned sideways by `deg` (fingertips toward +y, the hand's body toward -y)."""
    return Rx(math.radians(deg)) @ G


def Rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def spray_tcp_for(target, tilt_deg, dist, yaw_deg=0.0):
    """(TCP position, rotation) holding the spray (pinched at GRIP_LOCAL, part x along world x as it lay) so that
    its nozzle face centre is `dist` from `target` along the jets' axis: leaned `tilt_deg` forward, then turned
    `yaw_deg` about the vertical (positive: the jets head toward +y)."""
    grip = Rz(math.radians(yaw_deg)) @ Ry(-math.radians(tilt_deg)) @ G0  # leaned about the handle's own axis
    R = grip @ G0.T @ Rz(SPRAY_YAW)  # part rotation: the part is held as it lay
    approach = R @ np.array((0.0, 0.0, -1.0))
    face = np.asarray(target, float) - approach * dist
    return face - R @ (FACE_LOCAL - GRIP_LOCAL), grip


PINCH = -0.0105  # 21 mm between the pads on the 25 mm handle
SPRAY_OPEN = -0.022  # 44 mm: the fingers open only this far beside the basin's near tube
SPRAY_LET_GO = -0.028  # 56 mm when letting go, so the rising fingers clear the handle
PARK = {"left": (np.array((0.17, 0.25, TOP + 0.21)), G), "right": (np.array((0.17, -0.25, TOP + 0.21)), G)}
W1_TILT, W2_TILT, AIM_DIST, W1_DIST = 64.0, 60.0, 0.075, 0.090
W2_YAW, W1_YAW = -25.0, -38.0  # the spray comes from behind and to the robot's left
SHIELD_LEAN, SHIELD_PITCH = 45.0, 20.0
SWEEP_Z = TOP + FLOOR + CZ + 0.010  # fingertips just above the crown's widest line


class LazyW(Waypoint):
    """A waypoint whose target (position, rotation) may be a function, evaluated when the arm gets to it."""

    def __init__(self, dur, pos, rot, finger, label):
        self.duration, self._pos, self._rot, self.finger, self.label = dur, pos, rot, float(finger), label

    @property
    def pos(self):
        p = self._pos() if callable(self._pos) else self._pos
        return tuple(float(v) for v in p)

    @property
    def yaw(self):
        return Grip(self._rot() if callable(self._rot) else self._rot)


def _w(dur, pos, rot, finger, label):
    return LazyW(dur, pos, rot, finger, label)


_SEEN = {}


def _seen(key, fn):
    """fn() once per episode (a read of the scene when an arm first needs it)."""
    k = (_RUN["episode"], key)
    if k not in _SEEN:
        _SEEN[k] = fn()
    return _SEEN[k]


def _states(names):
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import scene as arena_scene
    return arena_scene.object_states(env, list(names))


def spray_grasp():
    """(TCP position, rotation) pinching the spray's handle where it lies now (the start pose offline)."""
    def read():
        st = _states(["hand_spray"])
        if st is None:
            o, R = np.array((SPRAY_START[0], SPRAY_START[1], TOP + FLOOR + HANDLE_R)), Rz(SPRAY_YAW)
        else:
            o, R = spray_pose(st)
        yaw = math.atan2(R[1, 0], R[0, 0])
        g = o + R @ GRIP_LOCAL
        return np.array((g[0], g[1], TOP + FLOOR + HANDLE_R + 0.002)), Rz(yaw - SPRAY_YAW) @ G0
    return _seen("spray", read)


def hair_pick():
    """Where the right hand pinches the hair tail lying on the basin floor behind the crown: between the two floor
    links nearest its ends, read from the tail when the hand first goes to it (the start shape offline)."""
    def read():
        st = _states(LOCK_NAMES)
        pts = []
        for n in LOCK_NAMES:
            c = hair_kit.world_centres(n) if st is None else np.asarray(st[n].centres, float)
            pts += [q for q in c if q[2] < TOP + FLOOR + 0.03 and q[0] < CROWN_X - 0.02]
        P = np.array(sorted(pts, key=lambda q: q[0])[:2])  # between the two links nearest the tail's ends
        return np.array((float(P[:, 0].mean()), float(P[:, 1].mean()), TOP + FLOOR + 0.002 + sk.HAND_TIP))
    return _seen("hair", read)


HAIR_OPEN, HAIR_PINCH = FINGER_OPEN, -0.009  # 94 mm open across the hair, closed to 18 mm on the 24 mm tail
HAIR_HOLD = np.array((0.180, -0.050, TOP + 0.275))  # the hair held up and back, off the crown
HAIR_HOLD_PITCH = 30.0
BEHIND_HEAD = np.array((0.250, -0.080, TOP + 0.200))  # the right hand passes here, clear of the back of the head


def plan(way):
    """{arm: [Waypoint]} for one way of doing it."""
    grip0 = lambda: spray_grasp()[0]  # noqa: E731
    rot0 = lambda: spray_grasp()[1]  # noqa: E731
    above = lambda: spray_grasp()[0] + (0.0, 0.0, 0.10)  # noqa: E731
    w1 = [p for p in PATCHES if p.name == "crown"][0]
    w2 = [p for p in PATCHES if p.name == "hairline"][0]
    aimA, rotA = spray_tcp_for(w2.c, W2_TILT, AIM_DIST, W2_YAW)
    aimB, rotB = spray_tcp_for(w1.c, W1_TILT, W1_DIST, W1_YAW)
    wait = (np.array((0.225, 0.140, TOP + 0.200)), Rz(math.pi / 2) @ G0)  # head toward the robot, jets down
    # shield: the right hand leaned sideways, its closed fingertips 10 mm above the forehead between the hairline
    # patch and the forehead patch, its body up to the robot's right: a wall across the jets' path to the forehead
    sh_tip = HEAD_O + cranium_point(0.5 * (W2_X + D1_X) + 0.004)[0] + np.array((0.0, 0.012, 0.010))
    sh_rot = Rx(math.radians(SHIELD_LEAN)) @ pitched(SHIELD_PITCH)
    sh_tcp = sh_tip - sh_rot[:, 2] * sk.HAND_TIP
    hold_rot = pitched(HAIR_HOLD_PITCH)
    C, O = 0.0, FINGER_OPEN
    L, Rt = [], []
    L.append(_w(1.5, PARK["left"][0], G, O, "park"))
    Rt.append(_w(1.5, PARK["right"][0], G, C, "park"))
    # right hand to the shield (not in the no-shield ways)
    if way in ("noshield", "careless"):
        Rt.append(_w(2.0, PARK["right"][0], G, C, "to shield"))
        Rt.append(_w(1.5, PARK["right"][0], G, C, "shield"))
    else:
        Rt.append(_w(2.0, sh_tcp + (0.0, -0.06, 0.08), sh_rot, C, "to shield"))
        Rt.append(_w(1.5, sh_tcp, sh_rot, C, "shield"))
    # left hand: pick up the spray
    L += [_w(1.5, above, rot0, SPRAY_OPEN, "over spray"), _w(1.2, grip0, rot0, SPRAY_OPEN, "descend"),
          _w(0.6, grip0, rot0, PINCH, "pinch"), _w(1.2, above, rot0, PINCH, "lift"),
          _w(0.8, above, rot0, PINCH, "settle")]
    # stage A: the hairline, with a slow side-to-side sweep
    L.append(_w(2.0, aimA + (0.0, 0.0, 0.03), rotA, PINCH, "to hairline"))
    for k, dy in enumerate((0.0, 0.008, -0.008, 0.008, -0.008, 0.0)):
        L.append(_w(0.8, aimA + (0.0, dy, 0.0), rotA, PINCH, f"rinse hairline {k}"))
    L.append(_w(1.5, np.array((CROWN_X - 0.060, 0.030, TOP + 0.215)), Ry(-math.radians(25.0)) @ G0, PINCH,
                "spray down onto the hair"))
    L.append(_w(1.5, *wait, PINCH, "spray down to the basin"))
    tA = sum(w.duration for w in L)
    t_r = sum(w.duration for w in Rt)
    if t_r < tA:
        Rt.append(_w(tA - t_r, Rt[-1]._pos, Rt[-1]._rot, C, "hold"))
    # right hand: leave the shield, gather the hair lying behind the crown and lift it up and back off the crown
    pick = hair_pick
    hover = lambda: hair_pick() + (0.0, 0.0, 0.08)  # noqa: E731
    up = lambda: hair_pick() + (0.0, 0.0, 0.09)  # noqa: E731
    hold = HAIR_HOLD
    if way in ("toponly", "careless"):  # the hair is never lifted: the hand rests
        pick = hover = up = hold = PARK["right"][0]
    off = (sh_tcp + (0.0, -0.04, 0.10), sh_rot) if way not in ("noshield", "careless") else (PARK["right"][0], G)
    back = BEHIND_HEAD if way not in ("toponly", "careless") else PARK["right"][0]
    Rt += [_w(1.5, *off, C, "off the forehead"),
           _w(1.2, back, G, C, "up and behind the head"),
           _w(1.5, hover, G, HAIR_OPEN, "over the hair"),
           _w(1.5, pick, G, HAIR_OPEN, "down to the hair"),
           _w(0.5, pick, G, HAIR_OPEN, "settle at the hair"),
           _w(1.0, pick, G, HAIR_PINCH, "gather the hair"),
           _w(0.6, pick, G, HAIR_PINCH, "hold the pinch"),
           _w(1.5, up, G, HAIR_PINCH, "lift the hair"),
           _w(1.8, hold, hold_rot, HAIR_PINCH, "hair up off the crown")]
    tR = sum(w.duration for w in Rt)
    L.append(_w(max(0.5, tR - sum(w.duration for w in L)), *wait, PINCH, "wait"))
    # stage B: the crown, from behind and to the left, under the lifted hair
    L.append(_w(1.5, *wait, PINCH, "behind the head"))
    L.append(_w(2.0, aimB + (0.0, 0.0, -0.02), rotB, PINCH, "to crown"))
    for k, dy in enumerate((0.0, 0.008, -0.008, 0.008, -0.008, 0.0)):
        L.append(_w(0.8, aimB + (0.0, dy, 0.0), rotB, PINCH, f"rinse crown {k}"))
    L += [_w(2.0, above, rot0, PINCH, "spray back down"),
          _w(1.2, lambda: spray_grasp()[0] + (0.0, 0.0, 0.001), rot0, PINCH, "lay down"),
          _w(0.6, lambda: spray_grasp()[0] + (0.0, 0.0, 0.001), rot0, SPRAY_LET_GO, "let go"),
          _w(1.0, above, rot0, SPRAY_LET_GO, "up"), _w(1.5, PARK["left"][0], G, O, "park")]
    tL = sum(w.duration for w in L)
    tR = sum(w.duration for w in Rt)
    Rt.append(_w(max(0.5, tL - tR), Rt[-1]._pos, Rt[-1]._rot, HAIR_PINCH, "hold"))
    down = (lambda: hair_pick() + (0.0, 0.0, 0.06)) if way not in ("toponly", "careless") else PARK["right"][0]
    Rt += [_w(1.5, down, G, HAIR_PINCH, "hair back down"), _w(0.6, down, G, HAIR_OPEN, "let the hair go"),
           _w(1.5, PARK["right"][0], G, C, "park")]
    tR = sum(w.duration for w in Rt)
    L.append(_w(max(0.5, tR - tL + 2.0), PARK["left"][0], G, O, "rest"))
    Rt.append(_w(2.0, PARK["right"][0], G, C, "rest"))
    return {"left": L, "right": Rt}


def current_way():
    if DEMO_SEQ is None:
        return DEMO_WAY
    return DEMO_SEQ[min(max(_RUN["episode"], 1), len(DEMO_SEQ)) - 1]


_PLANS = {}


def _plan_of(way):
    if way not in _PLANS:
        _PLANS[way] = plan(way)
    return _PLANS[way]


class WayWaypoint(Waypoint):
    """A waypoint read from the plan of the way being run (every way has the same steps and timing; only the
    targets differ), so a probe can run a different way in each episode of one launch."""

    def __init__(self, arm, i, base):
        self.arm, self.i, self.duration, self.label = arm, i, base.duration, base.label

    def _get(self):
        return _plan_of(current_way())[self.arm][self.i]

    @property
    def pos(self):
        return self._get().pos

    @property
    def yaw(self):
        return self._get().yaw

    @property
    def finger(self):
        return self._get().finger


def demo():
    base = _plan_of("rinse")
    for way in ("noshield", "toponly", "careless"):
        assert [w.duration for a in base for w in base[a]] == [w.duration for a in base for w in _plan_of(way)[a]]
    return {arm: [WayWaypoint(arm, i, w) for i, w in enumerate(ws)] for arm, ws in base.items()}
