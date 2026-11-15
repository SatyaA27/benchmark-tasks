"""Fit a mop frame to its quick-connect handle: keep the frame's swivelling socket square to the handle until the
handle's spring button clicks out through the socket's window.

A Rubbermaid HYGEN 17-inch Quick-Connect frame (FGQ56000YL00: aluminium plate 406 x 76 mm, nylon universal joint,
black nylon socket) lies flat on the bench on the robot's right, its long side along x. Its socket is folded down
toward the robot's left on the yellow swivel base and turned 12 degrees aside on its arm pivot. On the left lies a
HYGEN 20-40 inch Quick-Connect Short Extension Handle (FGQ74500YL00, collapsed, 640 mm), along y, its aluminium tip
pointing at the socket, the tip's spring button and locating lug facing up. Metres; x forward from the arm bases,
y left, z up.
"""
import math
import pathlib
import sys
from collections import deque

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import yam_parts as parts  # noqa: E402
from yam_parts import box, cyl  # noqa: E402
from yam_joints import Hinge, Link, Slide  # noqa: E402
from yam_mesh import Mesh, MeshMechanism  # noqa: E402
from yam_mesh import _extrude  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TEST_GRIPPERS = None
DIAG = False  # GPU diagnostic copies print MOP_DIAG lines
DEMO_PACE = 1.0
DEMO_SKEW = False  # the wrong way, for a check run: leave the socket turned aside and push the handle straight
DEMO_SHORT = False  # the wrong way, for a check run: stop the push 6 mm short of the click
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
_MEMORY_EPISODE = None
_TIMERS = {}
_EVER = {}


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


TOP = TABLE_TOP[SCENE]
MM = 0.001
ALU, BLACK, YELLOW, GRIP_BLACK = (0.66, 0.67, 0.70), (0.06, 0.06, 0.07), (0.95, 0.76, 0.05), (0.10, 0.10, 0.11)
HANDLE_YELLOW, TIP_ALU = (0.96, 0.80, 0.08), (0.62, 0.63, 0.66)

# ---------------------------------------------------------------- the frame. Socket coordinates (s): origin at the
# arm's hub, +z along the socket's axis toward the mouth, +y toward the window. Frame coordinates: plate underside at
# z 0, long axis x; the mechanism's joint angles 0 are the start: the socket folded flat toward frame -y (window up)
# and turned TURN_DEG about the arm's pin (vertical when folded). All joints start at 0.
PLATE_L, PLATE_W, PLATE_T = 0.374, 0.076, 0.006  # aluminium plate between the end caps (406 mm over the caps)
CAP_L, CAP_H = 0.016, 0.012
PIVOT_Z = 0.020  # the yellow swivel base's pin (along x), above the plate's underside
HUB_R = 0.029  # the socket arm's hub (its pin) from the base's pin
MOUTH_S = 0.124  # the socket's mouth, 124 mm from the hub
BORE_R = 0.0114  # socket bore radius (22.8 mm), 0.4 mm over the 22.0 mm tube
SOCKET_R = 0.017  # socket outside radius (34 mm)
CHAMFER = 0.003  # 45-degree lead-in at the mouth
BORE_FLOOR_S = 0.044  # bore bottom (80 mm below the mouth)
NOTCH_S = MOUTH_S - 0.008  # notch floor: the lug bottoms here (8 mm below the mouth's top)
NOTCH_DEG = (62.0, 118.0)  # notch in the mouth's top, either side of the window line (90 deg = +y)
WINDOW_DEG = (72.0, 108.0)
WINDOW_S = (MOUTH_S - 0.0605, MOUTH_S - 0.0435)  # 17 mm long, 7.0 mm wide at the bore
BAR_S = (0.005, 0.040)  # the arm between the hub and the socket: 24 mm across (x), 14 mm thick (y)
BAR_GRIP_S = 0.026  # the right hand pinches the arm 4 mm clear of the hub disc
TURN_DEG = 12.0  # the socket starts turned 12 degrees aside (toward the robot) on its arm pin
FRAME_MASS = {"base": 0.30, "cradle": 0.02, "arm": 0.04}  # 0.36 kg in all (estimate, <= 0.37 from the case weight)
NYLON = (0.30, 0.25)
FRAME_FRICTION = (0.30, 0.25)
HUB = np.array([0.0, -HUB_R, PIVOT_Z])  # the arm's pin in frame coordinates (folded)
_A_SQ = np.column_stack(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)))  # socket axes in frame axes, square


def _rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


A_START = _rz(TURN_DEG) @ _A_SQ  # socket axes in frame axes at the start (joint angles 0)


def s2f(p_s):
    """A socket-coordinate point in frame coordinates at joint angles 0."""
    return HUB + A_START @ np.asarray(p_s, float)


# ---------------------------------------------------------------- the handle (handle coordinates: the tip's end at
# the origin, the axis along +x toward the grip, the button and lug on +z)
TUBE_R = 0.011  # aluminium inner tube, 22 mm (the straight handle's tube: 0.87 in)
SEG = {"tube": (0.0, 0.300, 0.011), "ring": (0.300, 0.349, 0.020), "outer": (0.349, 0.546, 0.0125),
       "grip": (0.546, 0.640, 0.020)}  # x0, x1, radius (collapsed length 640 mm; ring and grip 40 mm)
LUG = (0.065, 0.085, 0.005, 0.0085, 0.0145)  # x0, x1, half-width, z0, z1: oval lug's box, 3.5 mm proud
BUTTON_X = (0.013, 0.029)  # 16 mm long, 5 mm wide, 2.5 mm proud when out; 30-degree lead ramp toward the tip
BUTTON_HALF_W = 0.0025
BUTTON_TRAVEL = 0.0027
BUTTON_K, BUTTON_PRELOAD = 1300.0, 2.0  # N/m and N (estimate: Valco-type single-leg snap button leaf)
HANDLE_MASS, BUTTON_MASS = 0.327, 0.003
HANDLE_FRICTION = (0.45, 0.40)
BUTTON_FRICTION = (0.20, 0.15)  # acetal on nylon
GRIP_X = 0.270  # the left hand pinches the tube 270 mm behind the tip
TUBE_SPLIT = 2 * GRIP_X - SEG["tube"][1]  # 240 mm: the tube piece from here to the ring is centred on the grip
RING_SIDES = 12  # the locking ring and the grip are ribbed: 12 flats, 40 mm across the flats

SEATED_DEPTH = MOUTH_S - (NOTCH_S - LUG[0])  # tip below the mouth's top when the lug bottoms: 0.073


def _closed_mesh(name, pts, f, color):
    pts = np.asarray(pts, float)
    tri = (pts - pts.mean(axis=0))[f]
    if np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        f = f[:, ::-1]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return Mesh(name, pts, np.ascontiguousarray(f), collision="convex", color=color,
                proxy=(box(tuple((lo + hi) / 2), tuple(np.maximum(hi - lo, 1e-4))),))


def _hull_mesh(name, profile_rz, th0, th1, color, vertex_r_scale):
    """A convex wall piece of the socket: an (r, s) profile swept between two angles (deg) about the socket's axis,
    placed in frame coordinates. Inner vertices sit at r / cos(half-angle) so each flat inner face is tangent to the
    round bore."""
    p2, f = _extrude(np.asarray(profile_rz, float), 1.0)
    pts = []
    for (r, z, w) in p2:
        th = math.radians(th0 if w < 0 else th1)
        rr = r * vertex_r_scale
        pts.append(s2f((rr * math.cos(th), rr * math.sin(th), z)))
    return _closed_mesh(name, pts, f, color)


def _wall(name, th0, th1, z0, z1, chamfer):
    k = 1.0 / math.cos(math.radians(th1 - th0) / 2)
    ri, ro = BORE_R, SOCKET_R / k
    if chamfer:
        prof = [(ri, z0), (ro, z0), (ro, z1), (ri + CHAMFER, z1), (ri, z1 - CHAMFER)]
    else:
        prof = [(ri, z0), (ro, z0), (ro, z1), (ri, z1)]
    return _hull_mesh(name, prof, th0, th1, BLACK, k)


def _socket_box(name, lo, hi, color=BLACK):
    """A box given in socket coordinates, as a convex mesh in frame coordinates."""
    outline = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
    p2, f = _extrude(np.asarray(outline), hi[2] - lo[2])
    pts = [s2f((x, y, z + (lo[2] + hi[2]) / 2)) for x, y, z in p2]
    return _closed_mesh(name, pts, f, color)


def _socket_disc(name, r, s0, s1, sides=16, color=BLACK):
    a = np.arange(sides) * 2 * math.pi / sides
    outline = np.column_stack((r * np.cos(a), r * np.sin(a))) / math.cos(math.pi / sides)
    p2, f = _extrude(outline, s1 - s0)
    pts = [s2f((x, y, z + (s0 + s1) / 2)) for x, y, z in p2]
    return _closed_mesh(name, pts, f, color)


def _socket_meshes():
    out = [_socket_box("bar", (-0.012, -0.007, BAR_S[0]), (0.012, 0.007, BAR_S[1])),
           _socket_disc("floor", SOCKET_R, BAR_S[1], BORE_FLOOR_S)]
    plain = [(118, 150), (150, 180), (180, 210), (210, 240), (240, 270), (270, 300), (300, 330), (330, 360), (0, 30),
             (30, 62)]
    for i, (a, b) in enumerate(plain):
        out.append(_wall(f"wall{i:02d}", a, b, BORE_FLOOR_S, MOUTH_S, True))
    out.append(_wall("notch_side_a", NOTCH_DEG[0], WINDOW_DEG[0], BORE_FLOOR_S, NOTCH_S, False))
    out.append(_wall("notch_side_b", WINDOW_DEG[1], NOTCH_DEG[1], BORE_FLOOR_S, NOTCH_S, False))
    out.append(_wall("below_window", WINDOW_DEG[0], WINDOW_DEG[1], BORE_FLOOR_S, WINDOW_S[0], False))
    out.append(_wall("above_window", WINDOW_DEG[0], WINDOW_DEG[1], WINDOW_S[1], NOTCH_S, False))
    return out


def _frame():
    hx = PLATE_L / 2
    base = [
        box((0, 0, PLATE_T / 2), (PLATE_L, PLATE_W, PLATE_T), ALU),
        box((hx + CAP_L / 2, 0, CAP_H / 2), (CAP_L, PLATE_W + 0.004, CAP_H), BLACK),
        box((-hx - CAP_L / 2, 0, CAP_H / 2), (CAP_L, PLATE_W + 0.004, CAP_H), BLACK),
        box((0.028, 0, PLATE_T + 0.010), (0.010, 0.016, 0.020), BLACK),  # the swivel base's two bearing blocks
        box((-0.028, 0, PLATE_T + 0.010), (0.010, 0.016, 0.020), BLACK),
    ]
    cradle = [box((0, -0.012, PIVOT_Z), (0.044, 0.030, 0.010), YELLOW),  # the yellow base, folded flat
              cyl((0, 0, PIVOT_Z), 0.007, 0.044, axis="x", color=YELLOW)]
    arm = [cyl(tuple(HUB), 0.014, 0.016, axis="z", color=BLACK)]  # the hub disc on its (vertical) pin
    links = [
        Link("cradle", cradle, Hinge("x", (0, 0, PIVOT_Z), -180.0, 0.0, damping=2e-4, armature=2e-5,
                                     static_friction=0.004, dynamic_friction=0.003), FRAME_MASS["cradle"],
             color=YELLOW, friction=NYLON),
        Link("arm", arm, Hinge("z", tuple(HUB), -80.0 - TURN_DEG, 80.0 - TURN_DEG, damping=2e-4, armature=2e-5,
                               static_friction=0.004, dynamic_friction=0.003), FRAME_MASS["arm"], parent="cradle",
             color=BLACK, friction=NYLON),
    ]
    return MeshMechanism("frame", FRAME_POS, base, links, mass=FRAME_MASS["base"], color=ALU, yaw=FRAME_YAW,
                         friction=FRAME_FRICTION, contact_offset=0.002, meshes={"arm": _socket_meshes()})


def _button_mesh():
    x0, x1 = BUTTON_X
    top = TUBE_R + 0.0025
    ramp_x = x0 + 0.0025 / math.tan(math.radians(30.0))
    prof = [(x0, 0.0090), (x1, 0.0090), (x1, top), (ramp_x, top), (x0, TUBE_R)]  # (x, z)
    p2, f = _extrude(np.asarray(prof), 2 * BUTTON_HALF_W)
    pts = np.column_stack((p2[:, 0], p2[:, 2], p2[:, 1]))  # (x, y, z): extrusion along y
    return _closed_mesh("button", pts, f, BLACK)


def _ribbed(name, x0, x1, r_flat, color):
    n = RING_SIDES
    a = (np.arange(n) + 0.5) * 2 * math.pi / n  # a flat at the bottom
    outline = np.column_stack((np.cos(a), np.sin(a))) * r_flat / math.cos(math.pi / n)  # (y, z)
    p2, f = _extrude(outline, x1 - x0)
    pts = np.column_stack((p2[:, 2] + (x0 + x1) / 2, p2[:, 0], p2[:, 1]))
    return _closed_mesh(name, pts, f, color)


def _handle():
    shapes = []
    x0, x1, r = SEG["tube"]
    for a, b in ((x0, TUBE_SPLIT), (TUBE_SPLIT, x1)):  # one tube as two collinear pieces: the second is where the
        shapes.append(cyl(((a + b) / 2, 0, 0), r, b - a, axis="x", color=TIP_ALU))  # left hand pinches, centred on it
    x0, x1, r = SEG["outer"]
    shapes.append(cyl(((x0 + x1) / 2, 0, 0), r, x1 - x0, axis="x", color=HANDLE_YELLOW))
    x0, x1, hw, z0, z1 = LUG
    shapes.append(box(((x0 + x1) / 2, 0, (z0 + z1) / 2), (x1 - x0, 2 * hw, z1 - z0), BLACK))
    button = Link("button", [], Slide("z", ((BUTTON_X[0] + BUTTON_X[1]) / 2, 0, TUBE_R), -BUTTON_TRAVEL, 0.0,
                                      damping=0.3, spring=BUTTON_K, rest=BUTTON_PRELOAD / BUTTON_K, armature=0.002),
                  BUTTON_MASS, color=BLACK, friction=BUTTON_FRICTION)
    meshes = {"base": [_ribbed("ring", SEG["ring"][0], SEG["ring"][1], SEG["ring"][2], GRIP_BLACK),
                       _ribbed("grip", SEG["grip"][0], SEG["grip"][1], SEG["grip"][2], GRIP_BLACK)],
              "button": [_button_mesh()]}
    return MeshMechanism("handle", HANDLE_POS, shapes, [button], mass=HANDLE_MASS, color=TIP_ALU, yaw=HANDLE_YAW,
                         friction=HANDLE_FRICTION, contact_offset=0.002, meshes=meshes)


# ---------------------------------------------------------------- layout
FRAME_YAW = math.pi  # frame x -> world -x, frame -y (the folded socket's direction) -> world +y
FRAME_XY = (0.280, -0.240)
FRAME_POS = (FRAME_XY[0], FRAME_XY[1], TOP)
FRAME = _frame()
Q_SQUARE = {"cradle": 0.0, "arm": -math.radians(TURN_DEG)}
_SQ = FRAME.offline_state(FRAME_POS, FRAME_YAW, Q_SQUARE)
MOUTH_SQ = _SQ.link_to_world("arm", s2f((0, 0, MOUTH_S)))  # the square socket's mouth (world)
HANDLE_YAW = math.pi / 2  # handle +x -> world +y
TIP_START_GAP = 0.045
HANDLE_POS = (float(MOUTH_SQ[0]), float(MOUTH_SQ[1] + TIP_START_GAP), TOP + SEG["ring"][2])
HANDLE = _handle()
OBJECTS = [FRAME, HANDLE]

# ---------------------------------------------------------------- scoring
CLICK_OUT = 0.0008  # the button within 0.8 mm of fully out
SEAT_TOL = 0.0015  # the lug within 1.5 mm of the notch floor
COAX_MM, COAX_DEG = 0.002, 3.0
PARTIAL_DEPTH = 0.030  # the tip at least 30 mm inside the bore
RELEASE_CLEAR = 0.030
STILL_WINDOW, STILL_MOVE = 0.5, 0.0010
TABLE_X, TABLE_Y = (-0.19, 0.91), 0.65


def frame_base(fr):
    """(origin, rotation) of the frame's plate, from its swivel base's link and joint angle. PhysX may make any body
    of a floating articulation its root, so the reported root pose is not used."""
    Rc = fr.link_rotation("cradle")
    R = Rc @ _rx(fr.joints["cradle"]).T
    return np.asarray(fr.links["cradle"][0], float) - R @ np.array([0.0, 0.0, PIVOT_Z]), R


def handle_base(h):
    """(origin, rotation) of the handle's body, from its button's link and slide position (root-independent)."""
    R = h.link_rotation("button")
    anchor = np.array([(BUTTON_X[0] + BUTTON_X[1]) / 2, 0.0, TUBE_R + h.joints["button"]])
    return np.asarray(h.links["button"][0], float) - R @ anchor, R


def _rx(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])


def socket_frame(fr):
    """(mouth top centre, axis out of the mouth, window direction) in world."""
    mouth = fr.link_to_world("arm", s2f((0, 0, MOUTH_S)))
    R = fr.link_rotation("arm") @ A_START
    return np.asarray(mouth, float), R @ np.array([0.0, 0.0, 1.0]), R @ np.array([0.0, 1.0, 0.0])


def handle_frame(h):
    o, R = handle_base(h)
    return o, R @ np.array([1.0, 0.0, 0.0]), R @ np.array([0.0, 0.0, 1.0])


def joint_terms(fr, h):
    mouth, a, win = socket_frame(fr)
    tip, hx, hz = handle_frame(h)
    depth = float((mouth - tip) @ a)
    lateral = float(np.linalg.norm((tip - mouth) - ((tip - mouth) @ a) * a))
    angle = math.degrees(math.acos(float(np.clip(hx @ a, -1.0, 1.0))))
    button = float(h.joints["button"])
    seated = depth >= SEATED_DEPTH - SEAT_TOL
    coax = lateral <= COAX_MM and angle <= COAX_DEG
    clicked = bool(seated and coax and button >= -CLICK_OUT)
    inside = bool(coax and depth >= PARTIAL_DEPTH)
    return {"depth": depth, "lateral": lateral, "angle": angle, "button": button, "seated": bool(seated),
            "coax": bool(coax), "clicked": clicked, "inside": inside}


def _seg_dist(p, a, b):
    ab = b - a
    s = float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + s * ab)))


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


def released(fr, h, clear=RELEASE_CLEAR):
    """Both grippers' centre points at least `clear` from the socket and from the handle's tube and grip."""
    hands = grippers()
    if hands is None:
        return True
    mouth, a, _ = socket_frame(fr)
    hub = fr.link_to_world("arm", HUB)
    tip, hx, _ = handle_frame(h)
    back = tip + hx * SEG["grip"][1]
    for tcp, _ in hands.values():
        p = np.asarray(tcp, float)
        if _seg_dist(p, hub, mouth) - SOCKET_R < clear or _seg_dist(p, tip, back) - SEG["ring"][2] < clear:
            return False
    return True


_WINDOW = {"episode": None, "hist": deque()}


def _ref_points(fr, h):
    tip, hx, _ = handle_frame(h)
    o, R = frame_base(fr)
    ends = [o + R @ np.array([s * PLATE_L / 2, 0.0, 0.0]) for s in (1, -1)]
    mouth = socket_frame(fr)[0]
    return np.array([tip, tip + hx * SEG["grip"][1], ends[0], ends[1], mouth])


def at_rest(fr, h):
    t = _RUN["t"]
    if t is None:
        return bool(np.linalg.norm(h.lin_vel) < 0.01 and np.linalg.norm(fr.lin_vel) < 0.01)
    hist = _WINDOW["hist"]
    if _WINDOW["episode"] != _RUN["episode"] or (hist and t < hist[-1][0]):
        hist.clear()
        _WINDOW["episode"] = _RUN["episode"]
    ref = _ref_points(fr, h)
    if not hist or t > hist[-1][0] + 1e-9:
        hist.append((t, ref))
    while len(hist) > 1 and hist[1][0] <= t - STILL_WINDOW + 1e-9:
        hist.popleft()
    if t - hist[0][0] < STILL_WINDOW - 0.051:
        return False
    return all(float(np.max(np.linalg.norm(r - ref, axis=1))) <= STILL_MOVE for _, r in hist)


def off_bench(fr, h):
    for o in (frame_base(fr)[0], handle_base(h)[0]):
        if o[2] < TOP - 0.01 or not TABLE_X[0] <= o[0] <= TABLE_X[1] or abs(o[1]) > TABLE_Y:
            return True
    return False


def score_terms(objects):
    fr, h = objects["frame"], objects["handle"]
    j = joint_terms(fr, h)
    fell = ever("frame or handle off the bench", off_bench(fr, h))
    free = released(fr, h)
    rest = at_rest(fr, h)
    full_s = held_for("clicked, released, at rest", j["clicked"] and free and rest)
    part_s = held_for("in the socket, released, at rest", j["inside"] and free and rest)
    if DIAG:
        _diag(fr, h, j, free, rest)
    return {**j, "fell": fell, "released": free, "at_rest": rest, "full_s": full_s, "part_s": part_s}


def score(objects):
    t = score_terms(objects)
    if t["fell"]:
        return 0.0
    if t["clicked"] and t["full_s"] + 1e-10 >= SUCCESS_HOLD_S:
        return 1.0
    if t["clicked"]:
        return 0.5
    if t["inside"] and t["part_s"] + 1e-10 >= SUCCESS_HOLD_S:
        return 0.25
    return 0.0


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


def _diag(fr, h, j, free, rest):
    step = _RUN["step"]
    if step == _DIAG["step"] or step % 4:
        return
    _DIAG["step"] = step
    mouth, a, win = socket_frame(fr)
    tip, hx, hz = handle_frame(h)
    hands = grippers() or {}
    tcp = " ".join(f"{k[0]}=({','.join(f'{v * 1000:.1f}' for v in hands[k][0])};{hands[k][1]:.2f})" for k in hands)
    print(f"MOP_DIAG t={_RUN['t']:.2f} depth={j['depth'] * 1000:.1f} lat={j['lateral'] * 1000:.2f} "
          f"ang={j['angle']:.2f} btn={j['button'] * 1000:.2f} clk={int(j['clicked'])} "
          f"cradle={math.degrees(fr.joints['cradle']):.1f} arm={math.degrees(fr.joints['arm']):.1f} "
          f"mouth=({','.join(f'{v * 1000:.1f}' for v in mouth)}) axis=({','.join(f'{v:.3f}' for v in a)}) "
          f"tip=({','.join(f'{v * 1000:.1f}' for v in tip)}) hx=({','.join(f'{v:.3f}' for v in hx)}) "
          f"hz=({','.join(f'{v:.2f}' for v in hz)}) frame=({','.join(f'{v * 1000:.1f}' for v in frame_base(fr)[0])}) "
          f"free={int(free)} rest={int(rest)} {tcp} F={_hand_forces()}", flush=True)


def _hand_forces():
    """Each hand's force on what it holds (N, world x, y, z), from the joints' PD error. Diagnostic only."""
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
            for jj in range(6):
                dq = q.copy()
                dq[jj] += 1e-5
                p1, r1 = ik.tcp_pose(dq, station.BASES[arm])
                ax, ang = _axis_angle(r1 @ r0.T)
                jac[:3, jj] = (p1 - p0) / 1e-5
                jac[3:, jj] = ax * ang / 1e-5
            w = np.linalg.lstsq(jac.T, tau, rcond=None)[0]
            out.append(f"{arm[0]}({w[0]:.1f},{w[1]:.1f},{w[2]:.1f})")
        return " ".join(out)
    except Exception as e:  # noqa: BLE001
        return f"err:{type(e).__name__}"


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


# ---------------------------------------------------------------- demo
# Right hand: pinch the socket's arm (between the hub and the socket, fingers closing across its 24 mm), turn the
# socket square (its axis along +y), and hold it there. Left hand: pinch the handle's tube 270 mm behind the tip,
# lay the tip on the socket's axis 15 mm short of the mouth, then push it home along that axis until the button
# clicks out through the window and the lug bottoms in the notch. Both let go and rise.
YAWS = {"left": -math.pi / 2, "right": math.pi / 2}
GRIP_OPEN = {"left": -0.020, "right": -0.022}  # 40 / 44 mm between the pads
HOVER = 0.055
AIM_GAP = 0.015
NEAR_GAP = 0.006  # the tip 6 mm short of the mouth for the last read before it goes in
ENTRY_DEPTH = 0.012
PUSH_OVER = 0.030  # commanded past the seated depth: the soft arms' error makes the push force (~220 N/m along y)
RIGHT_PUSH = 0.012  # the right hand presses the socket onto the tip meanwhile (the hands squeeze together)
PARK = {"left": np.array([0.150, 0.300, TOP + 0.200]), "right": np.array([0.150, -0.300, TOP + 0.200])}
FINAL_HOLD_S = 3.0
_READS = {"episode": None, "values": {}}


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _grip_rot(arm):
    from yam_arena.demo import grip_rotation
    return np.asarray(grip_rotation.__wrapped__(YAWS[arm]) if hasattr(grip_rotation, "__wrapped__")
                      else grip_rotation(YAWS[arm]), float)


# Grips, each in its part's frame: (TCP position, TCP rotation) relative to the socket arm's link / the handle.
_ARM_SQ_R = _SQ.link_rotation("arm")
_H_NOM_R = _rot_z(HANDLE_YAW)


def _grip_in_arm():
    p = A_START @ np.array([0.0, 0.0, BAR_GRIP_S])  # link frame: anchor-centred, frame axes at joint angles 0
    return p, _ARM_SQ_R.T @ _grip_rot("right")


def _grip_in_handle():
    return np.array([GRIP_X, 0.0, 0.0]), _H_NOM_R.T @ _grip_rot("left")


def _states():
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        return arena_scene.object_states(env, ["frame", "handle"])
    except Exception as e:  # noqa: BLE001
        print(f"MOP_READ failed: {e!r}", flush=True)
        return None


def _tcp_actual(arm):
    env = _RUN["env"]
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS
    art = env.unwrapped.scene[arm + "_robot"]
    ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
    joints = art.data.joint_pos
    q = getattr(joints, "torch", joints)[0]
    q = np.asarray(q.detach().cpu().numpy() if hasattr(q, "detach") else q, float)[ids]
    p, r = ik.tcp_pose(q, station.BASES[arm])
    return np.asarray(p, float), np.asarray(r, float)


def _T(p, R):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def _inv(T):
    R, p = T[:3, :3], T[:3, 3]
    return _T(-R.T @ p, R.T)


def _nominal_states():
    fr = FRAME.offline_state(FRAME_POS, FRAME_YAW, {"cradle": 0.0, "arm": 0.0})
    h = HANDLE.offline_state(HANDLE_POS, HANDLE_YAW, {"button": 0.0})
    return {"frame": fr, "handle": h}


def _read(key, cmd=None):
    """Cached per episode: object states (and, for a hand holding a part, the hand's tracking offset)."""
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    values = _READS["values"]
    if key not in values:
        st = _states()
        if st is None:  # offline (no simulator): the plan's own picture, the socket square once it has been turned
            st = _nominal_states()
            if key not in ("start", "pinched"):
                st["frame"] = FRAME.offline_state(FRAME_POS, FRAME_YAW, Q_SQUARE)
            values[key] = {"states": st}
        else:
            values[key] = {"states": st}
            fr, h = st["frame"], st["handle"]
            j = joint_terms(fr, h)
            print(f"MOP_READ {key} t={_RUN['t']:.2f} cradle={math.degrees(fr.joints['cradle']):.2f} "
                  f"arm={math.degrees(fr.joints['arm']):.2f} frame_off_mm="
                  f"{tuple(round(float(v) * 1000, 1) for v in frame_base(fr)[0] - FRAME_POS)} "
                  f"tip_mm={tuple(round(float(v) * 1000, 1) for v in handle_base(h)[0])} depth={j['depth'] * 1000:.1f} "
                  f"lat={j['lateral'] * 1000:.2f} ang={j['angle']:.2f} btn={j['button'] * 1000:.2f}", flush=True)
    return values[key]


def _arm_link_T(fr):
    p = fr.link_to_world("arm", HUB)
    return _T(np.asarray(p, float), fr.link_rotation("arm"))


def _handle_T(h):
    return _T(*handle_base(h))


def _square_arm_T(fr):
    """The arm link's pose with the socket turned square (perpendicular to the frame's long side) on the frame as it
    lies, its swivel base as read."""
    arm = fr.joints["arm"] if DEMO_SKEW else Q_SQUARE["arm"]  # the wrong way leaves the socket turned aside
    o, R = frame_base(fr)
    sq = FRAME.offline_state(o, 0.0, {"cradle": fr.joints["cradle"], "arm": arm}, R=R)
    return _arm_link_T(sq)


def _aim_handle_T(fr, depth):
    """The handle pose with its tip on the socket's axis `depth` inside the mouth (negative: outside), its axis on the
    socket's axis and its button side toward the window."""
    mouth, a, win = socket_frame(fr)
    z = win - (win @ a) * a
    z /= np.linalg.norm(z)
    y = np.cross(z, a)
    R = np.column_stack((a, y, z))
    return _T(mouth - a * depth, R)


class LazyWaypoint(Waypoint):
    """A hand waypoint worked out when played, from the episode's reads."""

    def __init__(self, duration, arm, fn, finger, label, keep=False):
        self.duration = duration * (1.0 if keep else DEMO_PACE)
        self.arm, self.fn, self.finger, self.label = arm, fn, finger, label

    def _pose(self):
        return self.fn()

    @property
    def pos(self):
        T = self._pose()
        return tuple(float(v) for v in T[:3, 3])

    @property
    def yaw(self):
        T = self._pose()
        return Grip(YAWS[self.arm], T[:3, :3])


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


def _lift(T, dz):
    T = T.copy()
    T[2, 3] += dz
    return T


def _right_on_arm(read_key, square=False, dz=0.0):
    """TCP pose pinching the socket's arm: at the read arm pose, or turned square (corrected by the read)."""
    r = _read(read_key)
    fr = r["states"]["frame"]
    gp, gR = _grip_in_arm()
    G = _T(gp, gR)
    if not square:
        return _lift(_arm_link_T(fr) @ G, dz)
    # held: the arm link in the commanded hand frame (grasp offset and the arm's sag together; the commanded rotation is
    # used, not the measured one, which IK may report turned 180 degrees about the symmetric grip's axis).
    S_act = _arm_link_T(fr)
    cmd = r.get("cmd_right")
    G_act = _inv(cmd if cmd is not None else S_act @ G) @ S_act
    return _lift(_square_arm_T(fr) @ _inv(G_act), dz)


def _left_on_handle(read_key, dz=0.0):
    r = _read(read_key)
    h = r["states"]["handle"]
    gp, gR = _grip_in_handle()
    return _lift(_handle_T(h) @ _T(gp, gR), dz)


def _left_aim(read_key, socket_key, depth, cmd_prev):
    """TCP command placing the held handle at the aim pose P: command = P * G^-1, G the handle in the previous
    command's frame at the read (grasp offset and the arm's sag together); the socket's axis from `socket_key`'s
    read."""
    r = _read(read_key)
    st = r["states"]
    fr = _read(socket_key)["states"]["frame"]
    if DEMO_SKEW:  # the wrong way: aim along the frame's square line, whatever the socket's turn
        o, R = frame_base(fr)
        fr = FRAME.offline_state(o, 0.0, {"cradle": fr.joints["cradle"], "arm": Q_SQUARE["arm"]}, R=R)
    H_act = _handle_T(st["handle"])
    key = "G_left_" + str(id(cmd_prev))
    if key not in r:  # the handle in the commanded hand frame at the read (grasp offset and sag together)
        r[key] = _inv(cmd_prev()) @ H_act if _RUN["env"] is not None else _inv(_T(*_grip_in_handle()))
    return _aim_handle_T(fr, depth) @ _inv(r[key])


def demo_plan():
    """Both arms' waypoints, in phase: [(arm, seconds, pose function, finger, label, keep)]."""
    OR, OL, C = GRIP_OPEN["right"], GRIP_OPEN["left"], FINGER_CLOSED
    right = [
        (2.5, lambda: _right_on_arm("start", dz=HOVER), OR, "right: over the socket's arm"),
        (1.5, lambda: _right_on_arm("start"), OR, "right: pads down beside the arm"),
        (1.0, lambda: _right_on_arm("start"), C, "right: pinch the arm", True),
        (0.5, lambda: _right_on_arm("start"), C, "right: settle"),
    ]
    left = [
        (2.5, lambda: _left_on_handle("start", dz=HOVER), OL, "left: over the handle's tube"),
        (1.5, lambda: _left_on_handle("start"), OL, "left: pads down beside the tube"),
        (1.0, lambda: _left_on_handle("start"), C, "left: pinch the tube", True),
        (0.5, lambda: _left_on_handle("start"), C, "left: settle"),
    ]
    # phase 2 (5.5 s on): the right hand turns the socket square; the left hand holds still
    right_cmd_pinch = lambda: _right_on_arm("start")  # noqa: E731

    def turn():
        r = _read("pinched")
        if r.get("cmd_right") is None:
            r["cmd_right"] = right_cmd_pinch()
        return _right_on_arm("pinched", square=True)

    right += [(2.0, turn, C, "right: turn the socket square"), (0.6, turn, C, "right: hold the socket square")]
    left += [(2.0, lambda: _left_on_handle("start"), C, "left: hold"),
             (0.6, lambda: _left_on_handle("start"), C, "left: hold")]
    # phase 3 (8.1 s on): the left hand lays the tip on the socket's axis 15 mm short of the mouth, then closes to
    # 6 mm short in two corrected steps (each from a fresh read: the soft wrist lets the handle turn a little in the
    # pinch while its far end drags on the bench)
    left_hold = lambda: _left_on_handle("start")  # noqa: E731
    aim1 = lambda: _left_aim("squared", "squared", -AIM_GAP, left_hold)  # noqa: E731
    aim2 = lambda: _left_aim("aimed", "aimed", -NEAR_GAP, aim1)  # noqa: E731
    aim3 = lambda: _left_aim("near", "near", -NEAR_GAP, aim2)  # noqa: E731
    left += [(2.0, aim1, C, "left: tip onto the socket's axis"), (0.5, aim1, C, "left: hold"),
             (0.8, aim2, C, "left: closer, corrected to the axis"), (0.5, aim2, C, "left: hold"),
             (0.5, aim3, C, "left: corrected again"), (0.4, aim3, C, "left: hold")]
    right += [(4.7, turn, C, "right: hold the socket square")]
    # phase 4 (12.8 s on): push into the mouth, read, push home while the right hand presses the socket onto it
    in1 = lambda: _left_aim("near2", "near2", ENTRY_DEPTH, aim3)  # noqa: E731
    final = SEATED_DEPTH + PUSH_OVER - (0.006 + PUSH_OVER if DEMO_SHORT else 0.0)
    in2 = lambda: _left_aim("entered", "entered", final, in1)  # noqa: E731
    left += [(1.2, in1, C, "left: tip into the mouth"), (0.3, in1, C, "left: hold"),
             (2.5, in2, C, "left: push home until it clicks"), (1.2, in2, C, "left: hold home")]

    def squeeze():
        T = turn()
        a = socket_frame(_read("entered")["states"]["frame"])[1]
        T = T.copy()
        T[:3, 3] += a * (0.0 if DEMO_SHORT else RIGHT_PUSH)
        return T

    right += [(1.5, turn, C, "right: hold the socket square"), (2.5, squeeze, C, "right: press the socket onto the tip"),
              (1.2, squeeze, C, "right: hold")]
    # phase 5 (18.0 s on): ease both pushes off (targets back to the seated place), let go and rise
    seated = lambda: _left_aim("entered", "entered", SEATED_DEPTH - (0.006 if DEMO_SHORT else 0.0), in1)  # noqa: E731
    left += [(0.6, seated, C, "left: ease the push off"), (1.0, seated, OL, "left: let go", True),
             (1.5, lambda: _lift(seated(), 0.06), OL, "left: rise clear")]
    right += [(0.6, turn, C, "right: ease the push off"), (1.0, turn, OR, "right: let go", True),
              (1.5, lambda: _lift(turn(), 0.06), OR, "right: rise clear")]
    return {"left": left, "right": right}


def demo():
    plan = demo_plan()
    out = {}
    for arm in ("left", "right"):
        wps = []
        for item in plan[arm]:
            secs, fn, finger, label = item[:4]
            keep = item[4] if len(item) > 4 else False
            wps.append(LazyWaypoint(secs, arm, fn, finger, label, keep))
        wps.append(Waypoint(2.0 * DEMO_PACE, tuple(float(v) for v in PARK[arm]), GRIP_OPEN[arm], "park", YAWS[arm]))
        wps.append(Waypoint(FINAL_HOLD_S, tuple(float(v) for v in PARK[arm]), GRIP_OPEN[arm], "hold clear", YAWS[arm]))
        out[arm] = wps
    return out
