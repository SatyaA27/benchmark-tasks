"""Stand a folded birthday card up on a shelf, open like a tent, picture facing the robot.

The card is a Nitelemn white 5 x 7 in folded tent card (110 lb / 300 g/m2 cover: a 10 x 7 in sheet scored and folded
to 5 x 7 in), its front leaf printed with a birthday picture: five coloured balloons above a red HAPPY BIRTHDAY
banner. It is two leaves of stiff card, each 127 x 178 mm, joined along a 178 mm crease, 13.5 g in all. It lies closed
on a short oak-effect melamine shelf (220 mm long, top 120 mm above the desk), picture up, crease away from the robot
and its middle over a 12 mm blue dot; the crease's spring holds the front leaf open about 12 degrees, so its free edge
stands about 26 mm above the back leaf. Stood on the shelf on the two free edges with the crease on top, the card
stays up if it is opened between about 36 and 83 degrees: the crease pulls a wider card closed and pushes a narrower
one open until the friction under its edges holds it. Metres; x forward from the arm bases, y left, z up.
"""
import math
import pathlib
import sys
from collections import deque

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import yam_joints as _yam_joints  # noqa: E402
from yam_parts import Part, box  # noqa: E402
from yam_joints import Hinge, Link, Mechanism  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TEST_GRIPPERS = None
DIAG = False  # GPU diagnostic copies print CARD_DIAG lines
DEMO_WRONG = ()  # GPU-only wrong-way copies: per episode, "" (the right way), "one", "drop" or "straight"


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
        opening = float(np.clip(np.mean(q[finger_ids]) / FINGER_OPEN, 0.0, 1.0))
        out[arm] = (tcp, opening)
    return out


def held_for(key, cond):
    """Elapsed time over consecutive observed control steps; repeat calls in one step are harmless."""
    global _MEMORY_EPISODE
    if _MEMORY_EPISODE != _RUN["episode"]:
        _TIMERS.clear()
        _MEMORY_EPISODE = _RUN["episode"]
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
RUN_LIMIT_STEPS = 1800  # 90 s at 20 Hz
MM = 0.001
TAU = 2 * math.pi

# ---------------------------------------------------------------- the card
# Nitelemn 30-pack white folded tent cards, 10 x 7 in folding to 5 x 7 in, 110 lb / 300 g/m2 cover, scored down the
# middle (Amazon B0GQXMQW8J). A 254 x 177.8 mm sheet at 300 g/m2 weighs 13.5 g. Its caliper is about 0.36 mm (14 pt,
# the figure given for comparable 300 g/m2 tent-card stock; an estimate for this card). Card frame: the middle of the
# back leaf's underside when lying closed; x toward the crease, y along it, z up.
LEAF_L = 127.0 * MM  # crease to free edge (5 in)
LEAF_W = 177.8 * MM  # along the crease (7 in)
LEAF_T = 1.0 * MM  # each leaf is about 0.36 mm thick; its collider is 1.0 mm (see the README)
LEAF_REAL_T = 0.36 * MM
CARD_MASS = 13.5e-3
LEAF_MASS = CARD_MASS / 2
# The crease (estimates: no maker or test figure exists for a greeting card's crease). A hinge with a spring toward
# 45 degrees open: 0.0071 N m per radian, so the front leaf's own weight (6.75 g at 63.5 mm) holds it at 12 degrees
# when the card lies closed (its free edge 26 mm up), and a card stood on 300 g/m2 card-on-melamine friction (0.30)
# stays put between about 36 and 83 degrees open. Damping 3e-4 N m s/rad (a quarter of critical).
CREASE_REST_DEG = 45.0
CREASE_K = 0.0071
CREASE_DAMP = 3.0e-4
GAPE_DEG = 11.8  # the lying card's opening: the crease spring against the front leaf's weight (equilibrium)
CARD_FRICTION = (0.30, 0.25)  # coated card stock (estimate)
CARD_WHITE = (0.96, 0.96, 0.94)
PICTURE = (  # (centre x, centre y, size x, size y, colour) on the front leaf's outside, card frame (crease at +x)
    (0.025, -0.048, 0.016, 0.014, (0.86, 0.12, 0.14)),  # balloons
    (0.031, -0.024, 0.016, 0.014, (0.98, 0.78, 0.10)),
    (0.035, 0.000, 0.016, 0.014, (0.15, 0.42, 0.88)),
    (0.031, 0.024, 0.016, 0.014, (0.20, 0.70, 0.30)),
    (0.025, 0.048, 0.016, 0.014, (0.58, 0.25, 0.70)),
    (0.006, -0.048, 0.022, 0.0012, (0.30, 0.30, 0.30)),  # strings
    (0.012, -0.024, 0.022, 0.0012, (0.30, 0.30, 0.30)),
    (0.016, 0.000, 0.022, 0.0012, (0.30, 0.30, 0.30)),
    (0.012, 0.024, 0.022, 0.0012, (0.30, 0.30, 0.30)),
    (0.006, 0.048, 0.022, 0.0012, (0.30, 0.30, 0.30)),
    (-0.030, 0.000, 0.016, 0.104, (0.82, 0.10, 0.12)),  # the HAPPY BIRTHDAY banner
)
PICTURE_T = 0.1 * MM

# ---------------------------------------------------------------- the shelf
# A short oak-effect melamine desk shelf: an 18 mm board 220 x 220 mm whose top is 120 mm above the desk, on an 18 mm
# back panel under its back edge. It is short (21 mm past the card at each side) so the grippers' bodies, which reach
# 51 mm below the TCP when a hand points along the crease, pass beyond its ends while the fingers pinch the card's
# side edges just above it.
SHELF_H = 0.120  # its top above the desk
SHELF_D, SHELF_W, SHELF_T = 0.220, 0.220, 0.018
DOT_XY = (0.285, 0.000)  # the blue dot: where the card should stand (the middle of its crease above it)
SHELF_X0 = DOT_XY[0] - 0.110  # the shelf's front edge
DOT_SIZE, DOT_T = 12.0 * MM, 0.05 * MM
SHELF_FRICTION = (0.30, 0.25)  # card on melamine 0.30 (estimate)
MELAMINE = (0.60, 0.43, 0.26)  # oak-effect melamine: the white card shows against it
DOT_BLUE = (0.10, 0.35, 0.85)
SHELF_TOP = TOP + SHELF_H
CARD_XY = DOT_XY  # the closed card lies with its middle over the dot


def make_card(pos):
    t = LEAF_T
    back = [box((0.0, 0.0, t / 2), (LEAF_L, LEAF_W, t), CARD_WHITE)]
    front = [box((0.0, 0.0, 1.5 * t), (LEAF_L, LEAF_W, t), CARD_WHITE)]
    front += [box((x, y, 2 * t + PICTURE_T / 2), (sx, sy, PICTURE_T), c) for x, y, sx, sy, c in PICTURE]
    crease = Hinge("y", (LEAF_L / 2, 0.0, t), 0.0, 178.0, damping=CREASE_DAMP, spring=CREASE_K,
                   rest=CREASE_REST_DEG, initial=GAPE_DEG)
    return Mechanism("card", pos, back, [Link("front", front, crease, LEAF_MASS, color=CARD_WHITE)],
                     mass=LEAF_MASS, color=CARD_WHITE, friction=CARD_FRICTION, contact_offset=0.001)


def make_shelf():
    cx = SHELF_X0 + SHELF_D / 2
    s = [box((0.0, 0.0, SHELF_H - SHELF_T / 2), (SHELF_D, SHELF_W, SHELF_T), MELAMINE),
         box((SHELF_D / 2 - SHELF_T / 2, 0.0, (SHELF_H - SHELF_T) / 2), (SHELF_T, SHELF_W, SHELF_H - SHELF_T),
             MELAMINE),
         box((DOT_XY[0] - cx, DOT_XY[1], SHELF_H + DOT_T / 2), (DOT_SIZE, DOT_SIZE, DOT_T), DOT_BLUE)]
    return Part("shelf", (cx, 0.0, TOP), s, color=MELAMINE, friction=SHELF_FRICTION)


CARD = make_card((CARD_XY[0], CARD_XY[1], SHELF_TOP + DOT_T))
SHELF = make_shelf()
OBJECTS = [SHELF, CARD]


def _root_on_base():
    """Put the card's articulation root on its back leaf (the base). Spawned with the root on the card's Xform, PhysX
    may pick either leaf as the floating root, and the harness would then place and report the card by that leaf's
    frame. Rooted on the back leaf, the card is placed and read by it, as the kit's state assumes."""
    original = _yam_joints._spawn_mechanism
    if getattr(original, "_card_root_on_base", False):
        return

    def spawn(prim_path, cfg, *args, **kwargs):
        prim = original(prim_path, cfg, *args, **kwargs)
        from pxr import PhysxSchema, UsdPhysics
        mech = _yam_joints._SPECS[cfg.mech_name]
        if mech.mass is None:
            return prim
        base = prim.GetStage().GetPrimAtPath(f"{prim_path}/base")
        prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        UsdPhysics.ArticulationRootAPI.Apply(base)
        art = PhysxSchema.PhysxArticulationAPI.Apply(base)
        art.CreateEnabledSelfCollisionsAttr().Set(mech.self_collide)
        art.CreateSolverPositionIterationCountAttr().Set(32)
        art.CreateSolverVelocityIterationCountAttr().Set(4)
        return prim

    spawn._card_root_on_base = True
    _yam_joints._spawn_mechanism = spawn


_root_on_base()


def _close_card(env):
    """At a reset: put the crease back at its lying opening, still (the scene's reset puts the back leaf back on its
    mark but leaves an articulated object's joint where the last episode left it)."""
    try:
        import torch
        art = env.unwrapped.scene["card"]
        q = torch.full((1, 1), math.radians(GAPE_DEG), device=art.device)
        art.write_joint_position_to_sim_index(position=q, joint_ids=[0])
        art.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=[0])
    except Exception as e:  # noqa: BLE001
        print(f"CARD close failed: {e!r}", flush=True)


def _install_reset_hook():
    try:
        from yam_arena import inspect_embodiment as _ie
    except ImportError:
        return
    cls = _ie.YamArenaEmbodiment
    if getattr(cls, "_card_reset_hook", False):
        return
    original_reset = cls.reset

    def reset(self, *args, **kwargs):
        if getattr(self, "_env", None) is not None:
            _close_card(self._env)
        result = original_reset(self, *args, **kwargs)
        _close_card(self._env)
        return result

    cls.reset = reset
    cls._card_reset_hook = True


_install_reset_hook()


# ---------------------------------------------------------------- scoring
# What a person checks with a ruler, a protractor and a photo from above, 10 s after both hands have let go.
ON_SHELF = 4.0 * MM  # a free edge rests on the shelf: both its ends within this of the shelf's top
FOLD_ABOVE = 40.0 * MM  # the crease at least this far above both free edges (crease on top)
RELEASE_CLEAR = 0.100  # both grippers' centre points at least this far from the card (its bounding box)
CHECK_AFTER_S = 10.0  # standing, let go and still for this long before it counts
STILL_S, STILL_MOVE = 1.0, 1.0 * MM
SPEED = 0.01
TIERS = (  # (score, opening range in degrees, facing within degrees, crease middle within metres of the dot)
    (1.0, (30.0, 90.0), 20.0, 0.040),
    (0.6, (20.0, 120.0), 180.0, 0.080),
)


def _T(R, p):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = R, p
    return m


def _inv(m):
    R, p = m[:3, :3], m[:3, 3]
    return _T(R.T, -R.T @ p)


def _ry(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def leaf_frames(card):
    """World 4x4 poses of the back leaf (card frame) and the front leaf (frame at the crease's middle)."""
    back = _T(card.rotation(), np.asarray(card.origin, float))
    p, _q = card.links["front"]
    front = _T(card.link_rotation("front"), np.asarray(p, float))
    return back, front


def card_points(card):
    """The crease's middle and both free edges' ends (world), in the frames of the two leaves."""
    back, front = leaf_frames(card)
    h, w, t = LEAF_L / 2, LEAF_W / 2, LEAF_T
    crease = front[:3, 3]
    back_edge = [back[:3, :3] @ np.array([-h, s * w, z]) + back[:3, 3] for s in (-1, 1) for z in (0.0, t)]
    front_edge = [front[:3, :3] @ np.array([-LEAF_L, s * w, z]) + front[:3, 3] for s in (-1, 1) for z in (0.0, t)]
    return crease, np.array(back_edge), np.array(front_edge)


def _edge_low(edge):
    """Each end's lowest corner height (the two corners of the edge's thickness at y -w and +w)."""
    return float(min(edge[0][2], edge[1][2])), float(min(edge[2][2], edge[3][2]))


def _on_shelf_xy(p):
    return SHELF_X0 <= p[0] <= SHELF_X0 + SHELF_D and abs(p[1]) <= SHELF_W / 2


def card_terms(card):
    crease, back_edge, front_edge = card_points(card)
    _, front = leaf_frames(card)
    lows = _edge_low(back_edge) + _edge_low(front_edge)
    edges_on = all(abs(z - SHELF_TOP) <= ON_SHELF for z in lows)
    over_shelf = all(_on_shelf_xy(p) for p in np.vstack([back_edge, front_edge]))
    crease_up = bool(crease[2] >= max(np.max(back_edge[:, 2]), np.max(front_edge[:, 2])) + FOLD_ABOVE)
    standing = bool(edges_on and over_shelf and crease_up)
    opening = math.degrees(float(card.joints["front"]))
    n = front[:3, :3] @ np.array([0.0, 0.0, 1.0])  # the picture side's outward direction
    facing = math.degrees(math.atan2(abs(n[1]), -n[0])) if math.hypot(n[0], n[1]) > 1e-6 else 180.0
    off = float(math.hypot(crease[0] - DOT_XY[0], crease[1] - DOT_XY[1]))
    return {"standing": standing, "edges_on": edges_on, "over_shelf": over_shelf, "crease_up": crease_up,
            "opening": opening, "facing": facing, "off": off, "crease": crease, "lows": lows}


def _box_clear(state, clear):
    hands = grippers()
    if hands is None:
        if _RUN["t"] is not None:
            return False  # in a run the grippers must be read; none read is not "let go"
        return True
    low, high = np.asarray(state.low, float), np.asarray(state.high, float)
    return all(float(np.linalg.norm(np.maximum(np.maximum(low - tcp, tcp - high), 0.0))) >= clear
               for tcp, _ in hands.values())


def released(card):
    return _box_clear(card, RELEASE_CLEAR)


_WINDOW = {"episode": None, "hist": deque()}


def still(card):
    """Over the last second the crease's middle and the four corners of the two free edges each moved under 1 mm (the
    solver's velocity of a light card standing on two thin edges is not used in a run). A single frame (the offline
    tests) uses the reported velocity."""
    t = _RUN["t"]
    if t is None:
        return bool(np.linalg.norm(card.lin_vel) < SPEED)
    hist = _WINDOW["hist"]
    if _WINDOW["episode"] != _RUN["episode"] or (hist and t < hist[-1][0]):
        hist.clear()
        _WINDOW["episode"] = _RUN["episode"]
    crease, back_edge, front_edge = card_points(card)
    now = np.vstack([crease[None], back_edge[[0, 2]], front_edge[[0, 2]]])
    if not hist or t > hist[-1][0] + 1e-9:
        hist.append((t, now))
    while len(hist) > 1 and hist[1][0] <= t - STILL_S + 1e-9:
        hist.popleft()
    if t - hist[0][0] < STILL_S - 0.051:
        return False
    return all(float(np.max(np.linalg.norm(now - old, axis=1))) <= STILL_MOVE for _, old in hist)


def tier(terms):
    if not terms["standing"]:
        return 0.0
    for value, (lo, hi), facing, off in TIERS:
        if lo <= terms["opening"] <= hi and terms["facing"] <= facing and terms["off"] <= off:
            return value
    return 0.3


def score_terms(objects):
    card = objects["card"]
    terms = card_terms(card)
    free = released(card)
    rest = still(card)
    done_s = held_for("standing, let go and still", terms["standing"] and free and rest)
    if DIAG:
        _diag(card, terms, free, rest, done_s)
    return {**terms, "released": free, "still": rest, "done_s": done_s}


def score(objects):
    t = score_terms(objects)
    if t["done_s"] + 1e-9 < CHECK_AFTER_S:
        return 0.0
    return tier(t)


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


def _diag(card, terms, free, rest, done_s):
    step = _RUN["step"]
    if _RUN["t"] is None or step == _DIAG["step"] or step % 5:
        return
    _DIAG["step"] = step
    back, front = leaf_frames(card)
    u_back = back[:3, :3] @ np.array([-1.0, 0.0, 0.0])
    u_front = front[:3, :3] @ np.array([-1.0, 0.0, 0.0])
    hands = grippers() or {}
    tcp = " ".join(f"{a[0]}=({','.join(f'{v * 1000:.1f}' for v in hands[a][0])};{hands[a][1]:.2f})" for a in hands)
    c = terms["crease"]
    print(f"CARD_DIAG t={_RUN['t']:.2f} open={terms['opening']:.1f} "
          f"crease=({(c[0] - DOT_XY[0]) * 1000:.1f},{(c[1] - DOT_XY[1]) * 1000:.1f},{(c[2] - SHELF_TOP) * 1000:.1f}) "
          f"ub=({u_back[0]:.2f},{u_back[1]:.2f},{u_back[2]:.2f}) uf=({u_front[0]:.2f},{u_front[1]:.2f},{u_front[2]:.2f}) "
          f"lows=({','.join(f'{(z - SHELF_TOP) * 1000:.1f}' for z in terms['lows'])}) face={terms['facing']:.0f} "
          f"st={int(terms['standing'])} free={int(free)} rest={int(rest)} held={min(done_s, 99):.1f} {tcp} "
          f"F={_hand_forces()}", flush=True)


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
                d = r1 @ r0.T
                ang = np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]]) / 2
                jac[:3, j] = (p1 - p0) / 1e-5
                jac[3:, j] = ang / 1e-5
            w = np.linalg.lstsq(jac.T, tau, rcond=None)[0]
            out.append(f"{arm[0]}({w[0]:.1f},{w[1]:.1f},{w[2]:.1f})")
        return " ".join(out)
    except Exception as e:  # noqa: BLE001
        return f"err:{type(e).__name__}"


# ---------------------------------------------------------------- demo
# Two hands, each pinching the front leaf at one side edge (the gripper pointing along the crease, its pads on the
# leaf's two faces), so the card's weight is shared and neither thin pinch has to stop the card turning in it. The
# right hand slips its lower finger into the gap under the raised front leaf at the -y side edge, pinches the leaf
# and opens the card to 22 degrees about its crease (the back leaf stays flat); the left hand then pinches the +y side
# edge, and the right hand lets go and pinches again a little nearer the crease. Together they lift the card and turn
# the front leaf 82 degrees about the line between them, so the crease rises over it and the back leaf swings down
# under the crease and hangs. They lower it until the back leaf's free edge touches the shelf behind the dot, then
# carry the crease toward the robot round that edge until the front leaf's free edge lands too (the card is then open
# 60 degrees, crease over the dot), open their fingers and slide out sideways. The demo reads the card's pose
# (scripted demos may read simulator state); it never moves an object.
PAD_TIP = 0.0128  # the pad tips are 12.8 mm past the TCP
OVERLAP = 10.0 * MM  # the pads reach this far in from the leaf's side edge
Y_TCP = LEAF_W / 2 - OVERLAP + PAD_TIP  # TCP from the crease's middle, along the crease
PINCH_FIRST = 0.105  # the right hand's first pinch: this far from the crease (in the 12-degree gap, 21 mm high)
PINCH = 0.075  # both hands' working pinch: this far from the crease
RAISE_DEG = 22.0  # the right hand opens the card this far before the left hand pinches
TENT_PSI = -60.0  # the front leaf turned so its free edge points 60 degrees below level: open 60 degrees standing
OUT = 0.025  # before a pinch the TCP waits this far out along its approach axis
FINGER_ENTER = -0.0075  # pads 15 mm apart: the lower finger slips into the gap
FINGER_WIDE = -0.010  # pads 20 mm apart
CLOSE = FINGER_CLOSED
SWING_DEG = 30.0  # the back leaf's final angle from vertical (60 degrees open)
FINAL_HOLD_S = 14.0
PARK = {"left": np.array([0.323, 0.305, TOP + 0.261]), "right": np.array([0.323, -0.305, TOP + 0.261])}  # READY

# Hand frames in the front leaf's frame (origin at the crease's middle; x from the free edge toward the crease, y
# along the crease, z the picture side's outward normal). Columns: closing axis, pad width, approach.
R_LEFT = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])  # points -y; closes along the normal
R_RIGHT = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])  # points +y


def hand_in_leaf(arm, s, out=0.0):
    """TCP pose in the front leaf's frame for a pinch s from the crease, `out` back along the approach axis."""
    R = R_LEFT if arm == "left" else R_RIGHT
    p = np.array([-s, Y_TCP if arm == "left" else -Y_TCP, LEAF_T / 2]) - out * R[:, 2]
    return _T(R, p)


def front_nominal(opening_deg=GAPE_DEG, xy=CARD_XY):
    """The front leaf's frame for the closed card on its mark, the front leaf opened `opening_deg` about the crease."""
    return _T(_ry(opening_deg), np.array([xy[0] + LEAF_L / 2, xy[1], SHELF_TOP + DOT_T + LEAF_T]))


def leaf_at(psi_deg, pinch_point, s=PINCH, yaw_deg=0.0):
    """The front leaf's frame turned to psi (its free edge psi degrees above level) with its pinch line (s from the
    crease, mid-thickness) through `pinch_point`."""
    R = _rz(yaw_deg) @ _ry(psi_deg)
    return _T(R, np.asarray(pinch_point, float) - R @ np.array([-s, 0.0, LEAF_T / 2]))


def _card_now():
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        return arena_scene.object_states(env, ["card"])["card"]
    except Exception as e:  # noqa: BLE001
        print(f"CARD_READ failed: {e!r}", flush=True)
        return None


class _Ctx:
    """Per-episode reads and the targets worked out from them (each worked out once, when first needed)."""

    def __init__(self):
        self.memo = {}

    def get(self, key, fn):
        if key not in self.memo:
            self.memo[key] = fn()
        return self.memo[key]


_CTX = {"episode": None, "ctx": None}


def ctx():
    if _CTX["episode"] != _RUN["episode"] or _CTX["ctx"] is None:
        _CTX["episode"], _CTX["ctx"] = _RUN["episode"], _Ctx()
    return _CTX["ctx"]


MAX_FOLLOW = 0.025  # a read moves the plan at most this far


def read_front(key, planned):
    """The front leaf's frame as read now (clamped to MAX_FOLLOW of the planned frame); the planned frame offline."""
    def fn():
        card = _card_now()
        if card is None:
            return planned
        _, front = leaf_frames(card)
        d = front[:3, 3] - planned[:3, 3]
        n = float(np.linalg.norm(d))
        if n > MAX_FOLLOW:
            front = front.copy()
            front[:3, 3] = planned[:3, 3] + d * MAX_FOLLOW / n
        print(f"CARD_READ {key} t={_RUN['t']:.2f} crease_err_mm={tuple(round(float(v) * 1000, 2) for v in d)} "
              f"open={math.degrees(float(card.joints['front'])):.1f}", flush=True)
        return front
    return ctx().get(("front", key), fn)


def read_back_angle(key, nominal_deg):
    """The back leaf's angle from vertical in the x-z plane (positive: its free edge on the +x side of the crease)."""
    def fn():
        card = _card_now()
        if card is None:
            return nominal_deg
        back, _ = leaf_frames(card)
        u = back[:3, :3] @ np.array([-1.0, 0.0, 0.0])
        a = math.degrees(math.atan2(u[0], -u[2]))
        print(f"CARD_BACK {key} t={_RUN['t']:.2f} back_from_vertical={a:.1f}", flush=True)
        return float(np.clip(a, -5.0, 25.0))
    return ctx().get(("back", key), fn)


def hanging_back_deg():
    """The back leaf's angle from vertical when the card hangs from the front leaf at TENT_PSI: the crease spring
    against the back leaf's weight (no friction)."""
    psi_from_vertical = 90.0 + TENT_PSI  # the front leaf's angle from vertical
    w = LEAF_MASS * 9.81 * LEAF_L / 2
    lo, hi = -30.0, 60.0  # bisect: weight w sin a toward vertical = spring k (rest - opening) opening it
    for _ in range(60):
        a = (lo + hi) / 2
        if w * math.sin(math.radians(a)) > CREASE_K * math.radians(CREASE_REST_DEG - psi_from_vertical - a):
            hi = a
        else:
            lo = a
    return (lo + hi) / 2


BACK_HANG_DEG = hanging_back_deg()
BACK_FOOT = np.array([DOT_XY[0] + LEAF_L * math.sin(math.radians(SWING_DEG)), DOT_XY[1], SHELF_TOP])


def crease_on_arc(phi_deg, foot=BACK_FOOT, dz=0.0):
    """Where the crease's middle is when the back leaf stands on `foot` at phi degrees from vertical."""
    a = math.radians(phi_deg)
    return np.array([foot[0] - LEAF_L * math.sin(a), foot[1], foot[2] + LEAF_L * math.cos(a) + dz])


def leaf_with_crease(psi_deg, crease, yaw_deg=0.0):
    return _T(_rz(yaw_deg) @ _ry(psi_deg), np.asarray(crease, float))


# The lift and turn: (psi, pinch line point) keyframes. The pinch line turns about itself and moves; both hands sit
# on it, so their straight-line interpolation keeps them rigidly together.
def roll_keys(p0):
    """From the pinch line's start p0 (front leaf at RAISE_DEG): up and over to TENT_PSI with the crease above the
    point from which the back leaf, hanging, will set down its edge on BACK_FOOT."""
    crease3 = crease_on_arc(BACK_HANG_DEG, dz=0.012)
    p3 = crease3 + _ry(TENT_PSI) @ np.array([-PINCH, 0.0, LEAF_T / 2])
    k1 = np.array([p0[0] + 0.002, p0[1], SHELF_TOP + 0.055])
    k2 = np.array([(p0[0] + p3[0]) / 2, p0[1], SHELF_TOP + 0.066])
    return [(0.0, k1), (-30.0, k2), (TENT_PSI, p3)]


class CardWaypoint(Waypoint):
    """A waypoint whose TCP pose is worked out once per episode, when the plan first reaches it."""

    def __init__(self, duration, pose_fn, finger, label, key):
        self.duration, self.finger, self.label = duration, finger, label
        self._fn, self._key = pose_fn, key

    def pose(self):
        return ctx().get(("pose",) + self._key, self._fn)

    @property
    def pos(self):
        return tuple(float(v) for v in self.pose()[:3, 3])

    @property
    def yaw(self):
        return Grip(0.0, self.pose()[:3, :3])


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


def _ready_pose(arm):
    from yam_arena import ik, station
    from yam_arena.constants import READY
    p, R = ik.tcp_pose(np.array(READY, float), station.BASES[arm])
    return _T(R, p)


def _wrong(episode=None):
    if not DEMO_WRONG:
        return ""
    e = _RUN["episode"] if episode is None else episode
    return DEMO_WRONG[(max(e, 1) - 1) % len(DEMO_WRONG)]


def demo_phases():
    """[(label, seconds, left (pose_fn or None to hold, finger or None to hold), right (...))]. Pose functions are
    worked out once, when the plan reaches them; reads happen there."""
    nom0 = front_nominal()
    nom_raised = front_nominal(RAISE_DEG)
    A = lambda: read_front("A", nom0)  # noqa: E731  the lying card, read before the right hand comes in
    # the leaf raised about its (read) crease to RAISE_DEG
    raised = lambda: A() @ _T(_ry(RAISE_DEG - math.degrees(math.atan2(A()[0, 2], A()[2, 2]))), np.zeros(3))  # noqa: E731
    C = lambda: read_front("C", nom_raised)  # noqa: E731  read before the left hand comes in
    D = lambda: read_front("D", nom_raised)  # noqa: E731  read before the right hand pinches again
    E = lambda: read_front("E", nom_raised)  # noqa: E731  read when both hold it, before the lift

    def cmd(arm):  # the hands' commanded poses at the end of the regrip (both pinching at PINCH)
        return (C() if arm == "left" else D()) @ hand_in_leaf(arm, PINCH)

    def grip_E(arm):  # each hand's pose in the leaf's frame as read at E
        return ctx().get(("gripE", arm), lambda: _inv(E()) @ cmd(arm))

    def p0():
        return (E() @ np.array([-PINCH, 0.0, LEAF_T / 2, 1.0]))[:3]

    def key_leaf(i):
        def fn():
            psi, p = roll_keys(p0())[i]
            return leaf_at(psi, p)
        return lambda: ctx().get(("key", i), fn)

    K3 = key_leaf(2)
    F = lambda: read_front("F", K3())  # noqa: E731  after the turn, the back leaf hanging
    phi0 = lambda: read_back_angle("F", BACK_HANG_DEG)  # noqa: E731

    def grip_F(arm):  # each hand's pose in the leaf's frame as read at F
        return ctx().get(("gripF", arm), lambda: _inv(F()) @ (K3() @ grip_E(arm)))

    def touch_leaf(dz):
        return lambda: leaf_with_crease(TENT_PSI, crease_on_arc(phi0(), dz=dz))

    def arc_leaf(k, n):
        def fn():
            a = phi0() + (SWING_DEG - phi0()) * k / n
            return leaf_with_crease(TENT_PSI, crease_on_arc(a, dz=-0.0005 - 0.0005 * k / n))
        return fn

    def on_leaf(arm, leaf_fn, grip):
        return lambda: leaf_fn() @ grip(arm)

    def out_of(arm, base_fn, dist, dz=0.0):
        def fn():
            m = base_fn().copy()
            m[:3, 3] += -dist * m[:3, 2] + np.array([0.0, 0.0, dz])
            return m
        return fn

    def at(arm, leaf_fn, s, out=0.0):
        return lambda: leaf_fn() @ hand_in_leaf(arm, s, out)

    def above(fn, dz):
        def g():
            m = fn().copy()
            m[:3, 3] += np.array([0.0, 0.0, dz])
            return m
        return g

    def park(arm):
        return lambda: _T(_ready_pose(arm)[:3, :3], PARK[arm])

    right_pre = at("right", A, PINCH_FIRST, OUT)
    right_in = at("right", A, PINCH_FIRST)
    right_raised = at("right", raised, PINCH_FIRST)
    left_hover = above(at("left", lambda: nom_raised, PINCH, OUT), 0.050)
    left_pre = at("left", C, PINCH, OUT)
    left_in = at("left", C, PINCH)
    right_back = out_of("right", right_raised, OUT)
    right_pre2 = at("right", D, PINCH, OUT)
    right_in2 = at("right", D, PINCH)
    phases = [
        ("right hand over the card's -y side", 2.5, (left_hover, FINGER_WIDE), (above(right_pre, 0.040), FINGER_ENTER)),
        ("right fingers down beside the gap", 1.2, (None, None), (right_pre, None)),
        ("right lower finger into the gap", 1.2, (None, None), (right_in, None)),
        ("right pinches the front leaf", 0.8, (None, None), (None, CLOSE)),
        ("right opens the card about its crease", 1.5, (None, None), (right_raised, None)),
        ("left fingers beside the +y side edge", 1.2, (left_pre, None), (None, None)),
        ("left slides in", 1.2, (left_in, None), (None, None)),
        ("left pinches the front leaf", 0.8, (None, CLOSE), (None, None)),
        ("right lets go", 0.6, (None, None), (None, FINGER_WIDE)),
        ("right slides out", 1.0, (None, None), (right_back, None)),
        ("right moves nearer the crease", 1.0, (None, None), (right_pre2, None)),
        ("right slides in again", 1.2, (None, None), (right_in2, None)),
        ("right pinches again", 0.8, (None, None), (None, CLOSE)),
    ]
    wrong = _wrong()
    one = wrong == "one"  # wrong way: the left hand lets go and the right hand alone lifts and turns the card

    def left(fn):
        return None if one else fn

    if one:
        left_out = out_of("left", lambda: cmd("left"), OUT, dz=0.040)
        phases += [("left lets go (one hand only)", 0.6, (None, FINGER_WIDE), (None, None)),
                   ("left moves away", 1.0, (left_out, None), (None, None))]
    for i, label in enumerate(("lift and level the front leaf", "turn the crease up", "crease over the back edge")):
        lf = (above(left_out, 0.070), park("left"), None)[i] if one else on_leaf("left", key_leaf(i), grip_E)
        phases.append((label, 1.6, (lf, None), (on_leaf("right", key_leaf(i), grip_E), None)))
    phases.append(("let the back leaf hang still", 1.5, (None, None), (None, None)))
    if wrong == "drop":  # wrong way: let go of the card in the air, the back leaf hanging 12 mm above the shelf
        phases.append(("let go in the air", 0.8, (None, FINGER_WIDE), (None, FINGER_WIDE)))
    else:
        if wrong == "straight":  # wrong way: lower it straight down to where the right way ends, no swing
            final = lambda: leaf_with_crease(TENT_PSI, crease_on_arc(SWING_DEG, dz=-0.001))  # noqa: E731
            crease_now = lambda: crease_on_arc(phi0(), dz=0.012)  # noqa: E731
            over = lambda: leaf_with_crease(TENT_PSI, np.array([*final()[:2, 3], crease_now()[2]]))  # noqa: E731
            phases += [("over the final place", 1.5, (on_leaf("left", over, grip_F), None),
                        (on_leaf("right", over, grip_F), None)),
                       ("straight down", 2.0, (on_leaf("left", final, grip_F), None),
                        (on_leaf("right", final, grip_F), None))]
        else:
            phases += [("lower the back leaf's edge to the shelf", 1.5,
                        (left(on_leaf("left", touch_leaf(0.004), grip_F)), None),
                        (on_leaf("right", touch_leaf(0.004), grip_F), None)),
                       ("touch", 1.0, (left(on_leaf("left", touch_leaf(-0.0005), grip_F)), None),
                        (on_leaf("right", touch_leaf(-0.0005), grip_F), None))]
            n = 4
            for k in range(1, n + 1):
                phases.append((f"swing the crease over the back edge ({k})", 0.7,
                               (left(on_leaf("left", arc_leaf(k, n), grip_F)), None),
                               (on_leaf("right", arc_leaf(k, n), grip_F), None)))
        phases += [("hold it down", 0.5, (None, None), (None, None)),
                   ("let go", 0.8, (None, FINGER_WIDE), (None, FINGER_WIDE))]
    phases += [("slide out sideways", 1.2, (left("out"), None), ("out", None)),
               ("rise", 1.2, (left("away"), None), ("away", None)),
               ("park", 2.0, (park("left"), None), (park("right"), None)),
               ("hold clear", FINAL_HOLD_S, (None, None), (None, None))]
    return phases


def demo():
    phases = demo_phases()
    total = sum(p[1] for p in phases)
    out = {}
    for ai, arm in enumerate(("left", "right")):
        wps, prev_fn, prev_f = [], (lambda a=arm: _ready_pose(a)), FINGER_OPEN
        for i, phase in enumerate(phases):
            label, secs = phase[0], phase[1]
            fn, finger = phase[2 + ai]
            if fn == "out":
                fn = (lambda base=prev_fn: (lambda: _shift(base(), along=-0.020)))()
            elif fn == "away":
                fn = (lambda base=prev_fn: (lambda: _shift(base(), dz=0.070)))()
            elif fn is None:
                fn = prev_fn
            finger = prev_f if finger is None else finger
            wps.append(CardWaypoint(secs, fn, finger, label, (arm, i)))
            prev_fn, prev_f = (lambda w=wps[-1]: w.pose()), finger
        out[arm] = wps
    assert all(abs(sum(w.duration for w in out[a]) - total) < 1e-9 for a in out)
    return out


def _shift(m, along=0.0, dx=0.0, dz=0.0):
    m = m.copy()
    m[:3, 3] += along * m[:3, 2] + np.array([dx, 0.0, dz])
    return m


# ---------------------------------------------------------------- offline test layouts (states.yaml)
def _quat(rot):
    from yam_arena.ik import matrix_to_quat_xyzw
    return [float(v) for v in matrix_to_quat_xyzw(np.asarray(rot, float))]


def tent(opening=60.0, crease_xy=DOT_XY, facing=0.0, lift=0.0, tilt=0.0):
    """[origin, quat, opening] of the card standing as a tent: open `opening` degrees, its crease's middle over
    `crease_xy`, the picture turned `facing` degrees from facing -x, both free edges on the shelf (raised `lift`), and
    the whole card tipped `tilt` degrees about its crease's direction (positive: toward +x)."""
    half = opening / 2
    R_front = _rz(facing) @ _ry(tilt) @ _ry(-(90.0 - half))
    R_back = R_front @ _ry(-opening)
    crease_rel = np.zeros(3)
    feet = [R_front @ np.array([-LEAF_L, 0, 0]), R_front @ np.array([-LEAF_L, 0, LEAF_T]),
            R_back @ np.array([-LEAF_L, 0, 0]), R_back @ np.array([-LEAF_L, 0, -LEAF_T])]
    low = min(f[2] for f in feet)
    crease = np.array([crease_xy[0], crease_xy[1], SHELF_TOP - low + lift])
    origin = crease - R_back @ np.array([LEAF_L / 2, 0.0, LEAF_T]) + crease_rel
    return [float(v) for v in origin], _quat(R_back), float(opening)


def lying(opening=GAPE_DEG, xy=CARD_XY, upside_down=False):
    """[origin, quat, opening] of the card lying on the shelf, the back leaf flat (or face down: front leaf flat)."""
    if not upside_down:
        return [xy[0], xy[1], SHELF_TOP + DOT_T], _quat(np.eye(3)), float(opening)
    R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # turned over about x: the front leaf on the shelf
    return [xy[0], xy[1], SHELF_TOP + DOT_T + 2 * LEAF_T], _quat(R), 0.0
