"""Clip a sheet of card onto a clipboard.

A letter-size hardboard clipboard (Saunders 05612: 229 x 318 x 3.2 mm brown hardboard, 249.5 g with its clip) lies
flat on the bench in front of the left arm, its long edges running left to right and its clip at the left end. The
clip is a 4 in (99 mm wide) nickel-plated steel spring clip riveted to the board: a lever on a hinge pin, its jaw
reaching 30 mm down the board to a turned-down lip that a torsion spring presses onto the board, and its thumb tab
rising 32 degrees behind the hinge with a rolled end just past the board's top edge. Pressing the tab down lifts the
jaw; letting go snaps it shut. A sheet of 110 lb (199 g/m2) canary-yellow index card stock, 8.5 x 11 in (216 x 279 mm,
0.24 mm, 12 g), lies loose on the board, square to it, its top edge 45 mm short of the clip's stop (the clip base
plate's front edge) and its bottom end hanging 33 mm past the board's bottom edge. Metres; x forward from the arm
bases, y left, z up; 20 Hz.
"""
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import Part, box, cyl  # noqa: E402
from yam_joints import Hinge, Link  # noqa: E402
from yam_mesh import Mesh, MeshMechanism  # noqa: E402
from yam_arena import demo as _motion  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
DIAG = False  # GPU diagnostic copies print CLIPBOARD_DIAG lines
DEMO_WAYS = ("right",)  # per-episode demo variants (GPU-only copies): "right", "early" (clip let down before the
#                         sheet is under it), "closed" (the sheet pushed at the clip without pressing the tab)

# Release-and-stillness block (same as our other task files).
RELEASE_CLEAR = 0.030
STILL_S = 1.0
TEST_GRIPPERS = None


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


def grippers():
    if TEST_GRIPPERS is not None:
        return TEST_GRIPPERS
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, LEFT_BASE_POS, RIGHT_BASE_POS

    bases = station.BASES if SCENE == "station" else {"left": LEFT_BASE_POS, "right": RIGHT_BASE_POS}
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
        tcp = ik.tcp_pose(q[arm_ids], bases[arm])[0]
        opening = float(np.clip(np.mean(q[finger_ids]) / FINGER_OPEN, 0.0, 1.0))
        out[arm] = (tcp, opening)
    return out


def released(state, clear=RELEASE_CLEAR):
    hands = grippers()
    if hands is None:
        return True  # Offline arm poses are unknown, so release is assumed.
    low, high = np.asarray(state.low), np.asarray(state.high)
    return all(np.linalg.norm(np.maximum(np.maximum(low - tcp, tcp - high), 0.0)) >= clear
               for tcp, _ in hands.values())


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


# ------------------------------------------------------------------ geometry helpers
def _prism_mesh(name, profile_yz, x0, x1, color=None, proxy=None):
    """Closed convex prism: a convex (y, z) polygon (part frame) extruded along x from x0 to x1."""
    prof = [tuple(map(float, p)) for p in profile_yz]
    n = len(prof)
    pts = np.array([(x, y, z) for x in (x0, x1) for (y, z) in prof], float)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j), (i, n + j, n + i)]
    for k in range(1, n - 1):
        faces += [(0, k + 1, k), (n, n + k, n + k + 1)]
    faces = np.array(faces, int)
    tri = pts[faces]
    if np.einsum('ij,ij->i', tri[:, 0] - pts.mean(0), np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])).sum() < 0:
        faces = faces[:, ::-1]
    if proxy is None:
        lo, hi = pts.min(0), pts.max(0)
        proxy = (box((lo + hi) / 2, hi - lo),)
    return Mesh(name, pts, faces, collision="convex", color=color, proxy=proxy)


def _plate(p0, p1, t):
    """(y, z) quadrilateral of a plate of thickness t whose centre line runs from p0 to p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-d[1], d[0]]) * t / 2
    return (p0 - n, p1 - n, p1 + n, p0 + n)


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array(((1, 0, 0), (0, c, -s), (0, s, c)))


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))


# ------------------------------------------------------------------ the clipboard (Saunders 05612, letter)
TOP = TABLE_TOP[SCENE]
MM = 0.001
BOARD_W = 0.2286          # 9 in, across (world x)
BOARD_L = 0.3175          # 12.5 in, along (world y; +y is the clip end, the board's "top")
BOARD_T = 0.003175        # 1/8 in hardboard
BOARD_MASS = 0.2495       # 8.8 oz with the clip (maker/retailer listings)
LEVER_MASS = 0.0335       # estimate: 0.8 mm steel, 92 x ~58 mm developed
BASE_MASS = BOARD_MASS - LEVER_MASS
BX, BY = 0.200, 0.090     # board centre (world): the clip end sits in line with the left arm's base
BOARD_POS = (BX, BY, TOP)  # part origin: the board's underside centre
Y_TOP = BOARD_L / 2       # the board's top edge (board frame y)

# Clip profile (board frame: s = y - Y_TOP from the top edge, z above the board's face). All estimates of the common
# 4 in clipboard clip; see README.
CLIP_HW = 0.046           # lever half-width (92 mm lever between the hinge ears; 99 mm over the ears)
EAR_X = (0.0465, 0.0495)
PLATE_T = 0.0008          # 0.8 mm steel
HINGE_S, HINGE_H = -0.014, 0.007     # hinge pin: 14 mm in from the top edge, 7 mm above the face
STOP_S = -0.026           # base plate's front edge: the stop the sheet's top edge goes against
BASE_S = (STOP_S, -0.005)
STOP_HW = 0.045           # base plate (and stop) half-width
LIP_S = -0.044            # front face of the jaw's turned-down lip (30 mm in front of the hinge)
LIP_H = 0.0040            # lip height (flange from the face up to the jaw plate)
TAB_LEN, TAB_DEG = 0.024, 32.0       # thumb tab: 24 mm from the pin to its rolled end, 32 deg above the face
CURL_R = 0.0020
RIVET_X, RIVET_S = 0.025, -0.0155

HINGE = np.array([0.0, Y_TOP + HINGE_S, BOARD_T + HINGE_H])
_TAB_DIR = np.array([math.cos(math.radians(TAB_DEG)), math.sin(math.radians(TAB_DEG))])
CURL_C = HINGE[1:] + TAB_LEN * _TAB_DIR            # rolled tab end centre (y, z), board frame
LIP_FRONT_Y = Y_TOP + LIP_S

# Spring: torsion spring on the pin (estimate). Closed on the board the lip presses 8.0 N (0.240 N m about the pin);
# the spring is wound 120 deg at closed, so its rate is 0.1146 N m/rad. Pressing the tab's rolled end straight down
# then takes 11.7 N to start the jaw lifting and 12.4 N at 28 deg open; fully open (45 deg) the lip stands 23 mm
# off the face (the maker's 1 in capacity, roughly) and clamps 12.6 N.
LIP_FORCE = 8.0
SPRING_K = LIP_FORCE * (-LIP_S + HINGE_S) / math.radians(120.0)   # N m / rad
SPRING_REST = 120.0       # degrees past closed (preload); closing is +q, opening -q
OPEN_LIMIT = -45.0
HINGE_DAMPING = 1.0e-3
HINGE_ARMATURE = 1.0e-5

HARDBOARD = (0.45, 0.30, 0.18)
NICKEL = (0.60, 0.62, 0.66)
CARD_CANARY = (0.98, 0.86, 0.30)
RIVET = (0.50, 0.51, 0.54)

# Friction (static, dynamic) per material; PhysX averages the two materials of a contact. Pairs (estimates):
# card on hardboard 0.35/0.30, hardboard on the bench 0.35/0.33, card on the clip's steel 0.40/0.35, the robot's
# pads (0.5) on card 0.50/0.48.
BOARD_FRICTION = (0.20, 0.15)
CARD_FRICTION = (0.50, 0.45)
STEEL_FRICTION = (0.30, 0.25)


def _clip_base_shapes():
    out = [box((0.0, 0.0, BOARD_T / 2), (BOARD_W, BOARD_L, BOARD_T), HARDBOARD)]
    s0, s1 = BASE_S
    out.append(box((0.0, Y_TOP + (s0 + s1) / 2, BOARD_T + PLATE_T / 2), (2 * STOP_HW, s1 - s0, PLATE_T), NICKEL))
    for sx in (-1, 1):
        x = sx * (EAR_X[0] + EAR_X[1]) / 2
        out.append(box((x, Y_TOP + HINGE_S, BOARD_T + (HINGE_H + 0.0025) / 2),
                       (EAR_X[1] - EAR_X[0], 0.008, HINGE_H + 0.0025), NICKEL))
        out.append(cyl((sx * RIVET_X, Y_TOP + RIVET_S, BOARD_T + PLATE_T + 0.0006), 0.003, 0.0012, "z", RIVET))
    out.append(cyl(tuple(HINGE), 0.0018, 2 * EAR_X[1], "x", NICKEL))       # pin and spring coil
    return out


def _lever():
    hy, hz = HINGE[1], HINGE[2]
    lip_top = (LIP_FRONT_Y + PLATE_T / 2, BOARD_T + LIP_H)
    jaw_from = (hy - 0.0025, hz - 0.0003)
    jaw = _prism_mesh("jaw_plate", _plate(lip_top, jaw_from, PLATE_T), -CLIP_HW, CLIP_HW, NICKEL)
    tab_from = np.array([hy, hz]) + 0.0025 * _TAB_DIR
    tab = _prism_mesh("tab_plate", _plate(tab_from, CURL_C - CURL_R * 0.6 * _TAB_DIR, PLATE_T), -CLIP_HW, CLIP_HW,
                      NICKEL)
    lip = box((0.0, LIP_FRONT_Y + PLATE_T / 2, BOARD_T + LIP_H / 2), (2 * CLIP_HW, PLATE_T, LIP_H), NICKEL)
    curl = cyl((0.0, float(CURL_C[0]), float(CURL_C[1])), CURL_R, 2 * CLIP_HW, "x", NICKEL)
    knuckle = cyl(tuple(HINGE), 0.0026, 2 * CLIP_HW, "x", NICKEL)
    link = Link("lever", [lip, curl, knuckle],
                Hinge("x", tuple(HINGE), OPEN_LIMIT, 0.0, damping=HINGE_DAMPING, spring=SPRING_K, rest=SPRING_REST,
                      initial=0.0, armature=HINGE_ARMATURE),
                mass=LEVER_MASS, color=NICKEL, friction=STEEL_FRICTION)
    return link, (jaw, tab)


_LEVER, _LEVER_MESHES = _lever()
CLIPBOARD = MeshMechanism("clipboard", BOARD_POS, _clip_base_shapes(), [_LEVER], mass=BASE_MASS, color=HARDBOARD,
                          friction=BOARD_FRICTION, contact_offset=0.001, meshes={"lever": _LEVER_MESHES})

# The sheet: Neenah Exact Index 110 lb (199 g/m2) card, 8.5 x 11 in.
SHEET_W, SHEET_L, SHEET_T = 0.2159, 0.2794, 0.00024
SHEET_MASS = 0.0120
GOAL_TOP_Y = Y_TOP + STOP_S                  # sheet's top edge on the stop (board frame)
GOAL_SHEET_B = np.array([0.0, GOAL_TOP_Y - SHEET_L / 2, BOARD_T])   # sheet origin (its underside centre), board frame
START_GAP = 0.045                            # the sheet starts 45 mm short of the stop
START_SHEET_B = GOAL_SHEET_B - np.array([0.0, START_GAP, 0.0])
SHEET_POS = tuple(float(v) for v in np.array(BOARD_POS) + START_SHEET_B)
SHEET = Part("sheet", SHEET_POS, [box((0.0, 0.0, SHEET_T / 2), (SHEET_W, SHEET_L, SHEET_T), CARD_CANARY)],
             mass=SHEET_MASS, color=CARD_CANARY, friction=CARD_FRICTION, contact_offset=0.001)

def lever_deg_for_lip(h):
    """Lever angle (deg, opening negative) at which the lip's bottom edge stands h above the board's face."""
    dy, dz = LIP_FRONT_Y - HINGE[1], BOARD_T - HINGE[2]
    lo, hi = -45.0, 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        q = math.radians(mid)
        if dy * math.sin(q) + dz * math.cos(q) - dz > h:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


LEVER_ON_SHEET = lever_deg_for_lip(SHEET_T)     # -0.46 deg: the lip resting on the card
GOAL_SHEET_POS = tuple(float(v) for v in np.array(BOARD_POS) + GOAL_SHEET_B)

PRESSED_PARTS = ("clipboard", "sheet")
OBJECTS = [CLIPBOARD, SHEET]

# ------------------------------------------------------------------ scoring
BITE_MIN = 0.005          # clipped: the sheet's top edge at least 5 mm past the lip, across the lip's width ...
LIP_ON = (0.00005, 0.0015)  # ... and the lip resting on it (lip 0.05-1.5 mm off the board's face)
SEAT_GAP = 0.002          # against the stop: top edge within 2 mm of the stop at both ends of the stop
SEAT_PEN = 0.0005         # (contact tolerance past the stop face)
SEAT_X = 0.040            # where the two gaps are read (40 mm either side of the clip's centre line)
SQUARE_DEG = 2.0          # square: sheet edges within 2 deg of the board's
SIDE_TOL = 0.0005         # within the board's sides: no corner past a long edge (0.5 mm contact tolerance)
FLAT_DEG, FLAT_DZ = 3.0, 0.003   # the board lies flat on the bench before the check
TILT_DEG, TILT_RAMP_S, TILT_HOLD_S = 60.0, 2.5, 2.0   # the check: clip end raised until the board is at 60 deg
SLIP_MAX = 0.003          # the sheet may move at most 3 mm on the board during the check
STEP_M, STEP_DEG = 0.0005, 0.25
TEST_STICKY = ()
TEST_TILT = None          # offline tests: force the tilt check's result (None: passes exactly when clipped)
TILT_ANYWAY_AFTER = None  # GPU-only wrong-way copies: after this many seconds run the check even if not clipped


def sticky(key, cond):
    return ever(key, bool(cond) or key in TEST_STICKY)


def board_frame(cb):
    return np.asarray(cb.origin, float), np.asarray(cb.rotation(), float)


def to_board(cb, p_world):
    o, R = board_frame(cb)
    return R.T @ (np.asarray(p_world, float) - o)


def sheet_in_board(cb, sh):
    """Sheet corners (board frame, x-y) and its yaw relative to the board (deg)."""
    corners = [to_board(cb, sh.to_world((sx * SHEET_W / 2, sy * SHEET_L / 2, 0.0)))
               for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1))]   # top-left, top-right, bottom-right, bottom-left
    ax = to_board(cb, sh.to_world((0, 1.0, 0))) - to_board(cb, sh.to_world((0, 0, 0)))
    yaw = math.degrees(math.atan2(-ax[0], ax[1]))
    return np.array(corners), yaw


def top_edge_y(corners, x):
    """Board-frame y of the sheet's top edge at board-frame x (the edge's line, extended)."""
    a, b = corners[0], corners[1]
    if abs(b[0] - a[0]) < 1e-9:
        return float(max(a[1], b[1]))
    return float(a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))


def lip_height(cb):
    """Lip's bottom edge above the board's face (m), the lower of its two ends."""
    hs = []
    for sx in (-1, 1):
        p = cb.link_to_world("lever", (sx * CLIP_HW, LIP_FRONT_Y + PLATE_T / 2, BOARD_T))
        hs.append(to_board(cb, p)[2] - BOARD_T)
    return float(min(hs))


def terms(objects):
    cb, sh = objects["clipboard"], objects["sheet"]
    corners, yaw = sheet_in_board(cb, sh)
    tops = [top_edge_y(corners, sx * CLIP_HW) for sx in (-1, 1)]
    seats = [top_edge_y(corners, sx * SEAT_X) for sx in (-1, 1)]
    bite = min(tops) - LIP_FRONT_Y
    lip = lip_height(cb)
    gaps = [GOAL_TOP_Y - y for y in seats]
    z_sheet = to_board(cb, sh.origin)[2] - BOARD_T
    on_board = abs(z_sheet) <= 0.002 and abs(sh.rotation()[2, 2] - cb.rotation()[2, 2]) < 0.01
    clipped = bool(on_board and bite >= BITE_MIN and LIP_ON[0] <= lip <= LIP_ON[1])
    seated = bool(all(-SEAT_PEN <= g <= SEAT_GAP for g in gaps))
    square = bool(abs(yaw) <= SQUARE_DEG and float(np.max(np.abs(corners[:, 0]))) <= BOARD_W / 2 + SIDE_TOL)
    Rb = cb.rotation()
    flat = bool(Rb[2, 2] >= math.cos(math.radians(FLAT_DEG)) and abs(cb.origin[2] - TOP) <= FLAT_DZ)
    return dict(clipped=clipped, seated=seated, square=square, flat=flat, bite=bite, lip=lip, gaps=gaps, yaw=yaw,
                x_max=float(np.max(np.abs(corners[:, 0]))), z_sheet=z_sheet)


_POSE = {"episode": None, "token": None, "prev": None, "quiet": False}


def at_rest(objects):
    """Pose differences between consecutive control steps (contact chatter makes speeds unreliable)."""
    cb, sh = objects["clipboard"], objects["sheet"]
    if _RUN["t"] is None:
        return cb.resting() and sh.resting() and abs(cb.joint_vel["lever"]) < 0.05
    if _POSE["episode"] != _RUN["episode"]:
        _POSE.update(episode=_RUN["episode"], token=None, prev=None, quiet=False)
    token = (_RUN["step"], float(_RUN["t"]))
    if token != _POSE["token"]:
        pts = np.r_[cb.to_world((0, Y_TOP, 0)), cb.to_world((0, -Y_TOP, 0)),
                    sh.to_world((0, SHEET_L / 2, 0)), sh.to_world((0, -SHEET_L / 2, 0))]
        pose = (pts, cb.angle_deg("lever"))
        quiet = False
        if _POSE["prev"] is not None:
            step, old = _POSE["prev"]
            quiet = bool(token[0] == step + 1 and np.all(np.abs(pose[0] - old[0]) < STEP_M)
                         and abs(pose[1] - old[1]) < STEP_DEG)
        _POSE.update(prev=(token[0], pose), token=token, quiet=quiet)
    return _POSE["quiet"]


# ---- the tilt check: once both hands are clear and everything has been still for 1 s, the scorer raises the clip end
# of the board (an upward force at the middle of its top edge, as a person lifts a clipboard by its clip end) until
# the board stands at 60 deg on its bottom edge, over 2.5 s, and holds it there 2 s. Passed: the sheet moved no more
# than 3 mm on the board. The board is held at 60 deg until the episode ends.
_TILT = {"episode": None, "phase": 0, "t0": None, "ref": None, "passed": None, "api": None, "failed": False,
         "slip": 0.0, "angle": 0.0, "frozen": None}


def _tilt_deg(cb):
    """How far the board's top edge is raised: the angle of its +y axis above the bench."""
    a = cb.rotation()[:, 1]
    return math.degrees(math.asin(float(np.clip(a[2], -1.0, 1.0))))


def _sheet_mark(cb, sh):
    return to_board(cb, sh.to_world((0, SHEET_L / 2, 0)))[:2]


def _apply_lift(fz, cb):
    """Vertical force fz (N) at the middle of the board's top edge, as a force at the centre of mass plus its moment."""
    env = _RUN["env"]
    if env is None or _TILT["failed"]:
        return
    import torch
    art = env.unwrapped.scene["clipboard"]
    dev = art.device
    ids = art.find_bodies("base")[0]
    point = cb.to_world((0.0, Y_TOP, BOARD_T / 2))
    com = None
    for name in ("body_com_pos_w", "body_link_com_pos_w"):
        try:
            v = getattr(art.data, name)
            v = getattr(v, "torch", v)
            com = np.asarray(v[0][ids[0]].detach().cpu().numpy(), float) - np.asarray(
                env.unwrapped.scene.env_origins[0].detach().cpu().numpy(), float)
            break
        except Exception:  # noqa: BLE001
            continue
    if com is None:
        com = cb.to_world((0.0, 0.0, BOARD_T / 2))
    f = np.array([0.0, 0.0, float(fz)])
    tq = np.cross(point - com, f)
    F = torch.tensor(f, dtype=torch.float32, device=dev).view(1, 1, 3)
    T = torch.tensor(tq, dtype=torch.float32, device=dev).view(1, 1, 3)
    comp = art.permanent_wrench_composer
    calls = [("set_forces_and_torques(body_ids, global)",
              lambda: comp.set_forces_and_torques(forces=F, torques=T, body_ids=ids, is_global=True)),
             ("set_forces_and_torques_index(body_ids, global)",
              lambda: comp.set_forces_and_torques_index(forces=F, torques=T, body_ids=ids, is_global=True)),
             ("reset + add_forces_and_torques_index(body_ids, env_ids, global)",
              lambda: (comp.reset(), comp.add_forces_and_torques_index(
                  forces=F, torques=T, body_ids=ids, env_ids=torch.tensor([0], device=dev), is_global=True))),
             ("set_external_force_and_torque(body_ids, global)",
              lambda: art.set_external_force_and_torque(F, T, body_ids=ids, is_global=True))]
    order = [c for c in calls if c[0] == _TILT["api"]] or calls
    errors = []
    for name, call in order:
        try:
            call()
            if _TILT["api"] != name:
                _TILT["api"] = name
                print(f"CLIPBOARD_TILT api={name}", flush=True)
            return
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e!r}")
    _TILT["failed"] = True
    print(f"CLIPBOARD_TILT_FAILED {errors}", flush=True)


def _target_deg(dt):
    s = min(1.0, max(0.0, dt / TILT_RAMP_S))
    return TILT_DEG * (s * s * (3 - 2 * s))


def tilt_check(objects, ready):
    """Runs once `ready` holds; None while pending or running, then whether the sheet stayed (bool)."""
    cb, sh = objects["clipboard"], objects["sheet"]
    if _RUN["t"] is None:
        if not ready:
            return None
        return bool(TEST_TILT) if TEST_TILT is not None else bool(terms(objects)["clipped"])
    if _TILT["episode"] != _RUN["episode"]:
        _TILT.update(episode=_RUN["episode"], phase=0, t0=None, ref=None, passed=None, slip=0.0, angle=0.0,
                     frozen=None, integral=0.0)
    t = float(_RUN["t"])
    if _TILT["phase"] == 0:
        if not ready:
            return None
        _TILT.update(phase=1, t0=t, ref=_sheet_mark(cb, sh))
        if DIAG:
            print(f"CLIPBOARD_TILT start t={t:.2f}", flush=True)
    dt = t - _TILT["t0"]
    ang = _tilt_deg(cb)
    w = float(np.dot(cb.ang_vel, cb.rotation()[:, 0]))       # about the board's x axis (raising the top end: +)
    target = _target_deg(dt)
    m = BOARD_MASS + SHEET_MASS
    reach = BOARD_L * max(0.2, math.cos(math.radians(ang)))
    inertia = m * BOARD_L ** 2 / 3
    err = math.radians(target - ang)
    _TILT["integral"] = float(np.clip(_TILT.get("integral", 0.0) + err * 0.05, -0.05, 0.05))
    acc = 64.0 * err + 40.0 * _TILT["integral"] - 11.2 * w
    # the weight carried at the top edge: half the board's and the card's, nearly all of the lever's (at the hinge)
    lever_share = (BOARD_L / 2 + HINGE[1]) / BOARD_L
    hold = 9.81 * ((BASE_MASS + SHEET_MASS) / 2 + LEVER_MASS * lever_share)
    fz = float(np.clip(hold + inertia * acc / reach, 0.0, 6.0))
    _apply_lift(fz, cb)
    slip = float(np.linalg.norm(_sheet_mark(cb, sh) - _TILT["ref"]))
    _TILT.update(slip=max(_TILT["slip"], slip), angle=ang)
    if DIAG and _RUN["step"] % 5 == 0:
        print(f"CLIPBOARD_TILT t={t:.2f} angle={ang:.1f} target={target:.1f} fz={fz:.2f} slip_mm={slip * 1000:.2f}",
              flush=True)
    if _TILT["phase"] == 1 and dt >= TILT_RAMP_S + TILT_HOLD_S:
        ok = bool(_TILT["slip"] <= SLIP_MAX and ang >= TILT_DEG - 5.0)
        _TILT.update(phase=2, passed=ok)
        print(f"CLIPBOARD_TILT end t={t:.2f} angle={ang:.1f} slip_max_mm={_TILT['slip'] * 1000:.2f} passed={ok}",
              flush=True)
    return None if _TILT["phase"] == 1 else _TILT["passed"]


_SCORE = {"token": None, "value": 0.0}
_FROZEN = {"episode": None, "terms": None}


def score(objects):
    token = (_RUN["episode"], _RUN["step"]) if _RUN["t"] is not None else None
    if token is not None and token == _SCORE["token"]:
        return _SCORE["value"]
    value = _score(objects)
    if token is not None:
        _SCORE.update(token=token, value=value)
    if DIAG and _RUN["t"] is not None and _RUN["step"] % 10 == 0:
        _diag(objects, value)
    return value


def _score(objects):
    cb, sh = objects["clipboard"], objects["sheet"]
    if _FROZEN["episode"] != _RUN["episode"]:
        _FROZEN.update(episode=_RUN["episode"], terms=None)
    off_bench = sticky("board off the bench", cb.origin[2] < TOP - 0.02)
    testing = _TILT["episode"] == _RUN["episode"] and _TILT["phase"] >= 1 and _RUN["t"] is not None
    if testing:
        ready = True
    else:
        tm = terms(objects)
        quiet = at_rest(objects)
        clear = released(cb) and released(sh)
        anyway = TILT_ANYWAY_AFTER is not None and _RUN["t"] is not None and _RUN["t"] >= TILT_ANYWAY_AFTER
        ready = (tm["clipped"] or anyway) and tm["flat"] and quiet and clear
        seconds = held_for("clipped, released, still", ready)
        ready = ready and seconds + 1e-9 >= STILL_S
        if ready and _FROZEN["terms"] is None:
            _FROZEN["terms"] = (tm["clipped"], tm["seated"], tm["square"])
    if off_bench:
        return 0.0
    passed = tilt_check(objects, ready)
    if not passed:
        return 0.0
    clipped, seated, square = _FROZEN["terms"] if _FROZEN["terms"] is not None else (False, False, False)
    if _RUN["t"] is None:
        tm = terms(objects)
        clipped, seated, square = tm["clipped"], tm["seated"], tm["square"]
    if not clipped:
        return 0.0
    return 0.5 + 0.25 * float(seated) + 0.25 * float(square)


def success(objects):
    return score(objects) >= 1.0


def _diag(objects, value):
    cb, sh = objects["clipboard"], objects["sheet"]
    tm = terms(objects)
    o = np.asarray(cb.origin) - np.asarray(BOARD_POS)
    hands = grippers() or {}
    hs = " ".join(f"{k[0]}=({','.join(f'{v * 1000:.0f}' for v in tcp)})" for k, (tcp, _) in hands.items())
    print(f"CLIPBOARD_DIAG t={_RUN['t']:.2f} score={value:.2f} board_off_mm=({o[0] * 1000:.1f},{o[1] * 1000:.1f},"
          f"{o[2] * 1000:.1f}) board_yaw={math.degrees(math.atan2(cb.rotation()[1, 0], cb.rotation()[0, 0])):.2f} "
          f"lever={cb.angle_deg('lever'):.2f} lip_mm={tm['lip'] * 1000:.2f} bite_mm={tm['bite'] * 1000:.1f} "
          f"gaps_mm=({tm['gaps'][0] * 1000:.1f},{tm['gaps'][1] * 1000:.1f}) yaw={tm['yaw']:.2f} "
          f"xmax_mm={tm['x_max'] * 1000:.1f} zsheet_mm={tm['z_sheet'] * 1000:.2f} clipped={int(tm['clipped'])} "
          f"seated={int(tm['seated'])} square={int(tm['square'])} flat={int(tm['flat'])} {hs} F={_hand_forces()}",
          flush=True)


def _hand_forces():
    """Each hand's force on what it touches (N, world x, y, z) from the joints' PD error: J^T F = K (q_target - q)."""
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
            p0, _ = ik.tcp_pose(q, station.BASES[arm])
            jac = np.zeros((3, 6))
            for j in range(6):
                dq = q.copy()
                dq[j] += 1e-5
                jac[:, j] = (ik.tcp_pose(dq, station.BASES[arm])[0] - p0) / 1e-5
            w = np.linalg.lstsq(jac.T, tau, rcond=None)[0]
            out.append(f"{arm[0]}({w[0]:.1f},{w[1]:.1f},{w[2]:.1f})")
        return " ".join(out)
    except Exception as e:  # noqa: BLE001
        return f"err:{type(e).__name__}"


# ------------------------------------------------------------------ scripted demo
# Left hand: straight down, fingers closed along the clip's width, presses the tab's rolled end 35 mm from the clip's
# centre and holds it (the press also pins the board). Right hand: pitched 60 deg, pointing up the board, fingers
# closed, rests its fingertips on the sheet's centre line 75 mm behind its top edge and drags it up the board under the
# lifted jaw and on 15 mm past the stop: the sheet's top edge meets the stop, and the push from its middle turns it
# square against the stop while the fingertips slip (one push point on the centre line, as a person squares a sheet
# with one finger; two fingertips 94 mm apart turned it 4-7 deg on the GPU because one pressed harder). The left hand then
# rises slowly so the jaw comes down on the sheet; the right hand lifts off; both go back to their ready poses.
# The demo reads the board's and the sheet's poses at the start and again once the fingertips are on the sheet.
class GripOrientation(float):
    def __new__(cls, matrix):
        obj = super().__new__(cls, math.atan2(-matrix[0, 0], matrix[1, 0]))
        obj.matrix = np.asarray(matrix, float).copy()
        return obj


_original_grip_rotation = getattr(_motion.grip_rotation, "_cb_original", _motion.grip_rotation)


def _grip_rotation(yaw=0.0):
    matrix = getattr(yaw, "matrix", None)
    if isinstance(matrix, np.ndarray) and matrix.shape == (3, 3):
        return matrix
    return _original_grip_rotation(yaw)


_grip_rotation._cb_original = _original_grip_rotation
_motion.grip_rotation = _grip_rotation


def _frame(approach, closing):
    a = np.asarray(approach, float) / np.linalg.norm(approach)
    c = np.asarray(closing, float)
    c = c - (c @ a) * a
    c /= np.linalg.norm(c)
    return np.column_stack([c, np.cross(a, c), a])


PAD = 0.013               # pad tips past the TCP along the approach
PRESS_X = -0.035          # press 35 mm from the clip's centre line (board frame x)
PRESS_Y_PAST = 0.0015     # fingertips aimed 1.5 mm past the curl's top, toward the board's edge
PRESS_TCP_Z = TOP + 0.0145  # pressed: tips commanded 1.5 mm over the bench (about 18 mm past first touch)
DRAG_PITCH = 60.0
DRAG_BACK = 0.075         # fingertips 75 mm behind the sheet's top edge
DRAG_TCP_Z = TOP + 0.0140  # dragging: the fingers' lowest corners about 3.5 mm into the card's top face
DRAG_LOW = 0.0028         # the fingers' lowest corners sit this far below the pad-tip point at 60 deg (hull model)
OVER = 0.015              # drag 15 mm past the stop: the fingertips slip 4-8 mm on the card on the way, then the
#                          card stops on the stop and they slip on
DEMO_PACE = 1.0
_READ = {"episode": None, "plan": None, "drag": None}


def _way(playing=True):
    """This episode's demo variant. demo() is built at the policy's reset, before the embodiment's reset counts the
    episode, so it looks one episode ahead; while playing, the episode is already counted."""
    k = int(_RUN["episode"] or 0) - (1 if playing else 0)
    return DEMO_WAYS[max(0, k) % len(DEMO_WAYS)]


def _now(name):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        st = arena_scene.object_states(env, [name])[name]
        return np.asarray(st.origin, float), np.asarray(st.rotation(), float)
    except Exception as e:  # noqa: BLE001
        print(f"CLIPBOARD_READ failed: {e!r}", flush=True)
        return None


def _episode_reads():
    if _READ["episode"] != _RUN["episode"]:
        _READ.update(episode=_RUN["episode"], plan=None, drag=None)


def _plan():
    """Board and sheet poses read once at the start (nominal offline)."""
    _episode_reads()
    if _READ["plan"] is None:
        b = _now("clipboard") or (np.array(BOARD_POS, float), np.eye(3))
        s = _now("sheet") or (np.array(SHEET_POS, float), np.eye(3))
        _READ["plan"] = (b, s)
        if _RUN["env"] is not None:
            print(f"CLIPBOARD_READ start t={_RUN['t']} board_off_mm={tuple(np.round((b[0] - BOARD_POS) * 1000, 1))} "
                  f"sheet_off_mm={tuple(np.round((s[0] - SHEET_POS) * 1000, 1))} way={_way()}", flush=True)
    return _READ["plan"]


def _drag_end():
    """Re-read once the fingertips rest on the sheet: where the contact must go so the sheet ends on its goal."""
    _episode_reads()
    if _READ["drag"] is None:
        (ob, Rb), (os_, Rs) = _plan()
        contact = _drag_contact()
        now_b, now_s = _now("clipboard"), _now("sheet")
        if now_b is not None and now_s is not None:
            ob, Rb = now_b
            os_, Rs = now_s
        goal_s = ob + Rb @ GOAL_SHEET_B
        move = goal_s - os_
        move[2] = 0.0
        end = contact + move + Rb @ np.array([0.0, OVER, 0.0])
        _READ["drag"] = end
        if _RUN["env"] is not None:
            print(f"CLIPBOARD_READ drag t={_RUN['t']:.2f} move_mm={tuple(np.round(move * 1000, 1))}", flush=True)
    return _READ["drag"]


def _board_point(p_b):
    (ob, Rb), _ = _plan()
    return ob + Rb @ np.asarray(p_b, float)


def _press_contact():
    return _board_point((PRESS_X, float(CURL_C[0]) + PRESS_Y_PAST, float(CURL_C[1]) + CURL_R))


def _drag_contact():
    (ob, Rb), (os_, Rs) = _plan()
    return os_ + Rs @ np.array([0.0, SHEET_L / 2 - DRAG_BACK, SHEET_T])


def _left_rot():
    (_, Rb), _ = _plan()
    return _frame((0.0, 0.0, -1.0), Rb[:, 0])


def _right_rot():
    (_, Rb), _ = _plan()
    p = math.radians(DRAG_PITCH)
    a = Rb @ np.array([0.0, math.cos(p), 0.0]) + np.array([0.0, 0.0, -math.sin(p)])
    return _frame(a, Rb[:, 0])


def _right_tcp(contact, z=None):
    """TCP for the dragging hand whose fingertips' lowest corners sit at `contact` (x, y); z absolute if given."""
    R = _right_rot()
    a = R[:, 2]
    tcp = np.asarray(contact, float) - PAD * a
    tcp[2] = (contact[2] - PAD * a[2] + DRAG_LOW) if z is None else z
    return tcp


def _ready_pose(arm):
    from yam_arena import ik, station
    p, r = ik.tcp_pose(np.array(READY, float), station.BASES[arm])
    return p, r


class LazyWaypoint(Waypoint):
    """A waypoint whose target is worked out when it is played (after the reads it depends on)."""

    def __init__(self, duration, label, finger, where, rot, keep=False):
        self.duration = duration * (1.0 if keep else DEMO_PACE)
        self.label, self.finger = label, finger
        self._where, self._rot = where, rot
        self._cache = None

    def _get(self):
        _episode_reads()
        if self._cache is None or self._cache[0] != _RUN["episode"]:
            self._cache = (_RUN["episode"], np.asarray(self._where(), float), GripOrientation(self._rot()))
        return self._cache

    @property
    def pos(self):
        return tuple(float(v) for v in self._get()[1])

    @property
    def yaw(self):
        return self._get()[2]


def demo_steps(way=None):
    """[(seconds, label, left (where, finger), right (where, finger))]; where = callable -> TCP position."""
    way = way or _way(playing=False)
    up = np.array([0.0, 0.0, 1.0])

    def press_tcp(lift):
        return lambda: _press_contact() + up * (PAD + lift)

    def press_down():
        p = _press_contact()
        return np.array([p[0], p[1], PRESS_TCP_Z])

    def drag_tcp(lift, end=False):
        def f():
            c = _drag_end() if end else _drag_contact()
            return _right_tcp(c) + up * lift if lift else _right_tcp(c, DRAG_TCP_Z)
        return f

    def ready(arm):
        return lambda: _ready_pose(arm)[0]

    C = FINGER_CLOSED          # both hands keep their fingers closed throughout
    hover_l, touch_l = press_tcp(0.040), press_tcp(0.002)
    steps = [
        (3.0, "to the clip; to the sheet", (hover_l, C), (drag_tcp(0.040), C)),
        (1.5, "onto the tab; over the sheet", (touch_l, C), (drag_tcp(0.010), C)),
    ]
    pressed = way != "closed"
    if pressed:
        steps += [(2.0, "press the tab down (the jaw lifts)", (press_down, C), (drag_tcp(0.010), C)),
                  (0.5, "hold the clip open", (press_down, C), (drag_tcp(0.010), C))]
    else:
        steps += [(2.5, "rest on the tab without pressing", (touch_l, C), (drag_tcp(0.010), C))]
    hold_l = press_down if pressed else touch_l
    if way == "early":
        steps += [(1.5, "let the clip down too soon", (hover_l, C), (drag_tcp(0.010), C))]
        hold_l = hover_l
    steps += [
        (1.0, "fingertips onto the sheet", (hold_l, C), (drag_tcp(0.0), C)),
        (0.5, "settle", (hold_l, C), (drag_tcp(0.0), C)),
        (4.0, "drag the sheet under the jaw to the stop", (hold_l, C), (drag_tcp(0.0, end=True), C)),
        (0.5, "hold it on the stop", (hold_l, C), (drag_tcp(0.0, end=True), C)),
        (1.5, "let the clip down onto the sheet", (hover_l, C), (drag_tcp(0.0, end=True), C)),
        (0.5, "settle", (hover_l, C), (drag_tcp(0.0, end=True), C)),
        (1.0, "lift off the sheet", (hover_l, C), (drag_tcp(0.040, end=True), C)),
        (2.5, "back to ready", (ready("left"), C), (ready("right"), C)),
        (8.0, "hold clear (the scripted tilt check runs)", (ready("left"), C), (ready("right"), C)),
    ]
    return steps


def demo():
    plan = {"left": [], "right": []}
    left_rot = lambda: _left_rot()  # noqa: E731
    right_rot = lambda: _right_rot()  # noqa: E731
    for secs, label, (lw, lf), (rw, rf) in demo_steps():
        last = label.startswith("back to ready") or label.startswith("hold clear")
        plan["left"].append(LazyWaypoint(secs, label, lf, lw, (lambda: _ready_pose("left")[1]) if last else left_rot,
                                         keep=label.startswith("hold clear")))
        plan["right"].append(LazyWaypoint(secs, label, rf, rw, (lambda: _ready_pose("right")[1]) if last else right_rot,
                                          keep=label.startswith("hold clear")))
    return plan


def demo_duration():
    return sum(w.duration for w in demo()["right"])
