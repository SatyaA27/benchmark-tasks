"""Mount a shoe-mount flash on a camera: hold the camera by its grip, slide the flash's foot into the hot shoe to the
stop and slide the foot's lock lever across to LOCK.

A Canon EOS Rebel T7 (129.0 x 101.3 x 77.6 mm, 475 g with battery and card) stands on its base on the table with a
body cap on, its back to the robot. A Canon Speedlite 430EX III-RT (70.5 x 113.8 x 98.2 mm, 295 g, with four AA cells
about 387 g) lies on its right side to the camera's left, its lock lever at UNLOCK. The robot holds the camera by its
grip with one hand, picks the flash up with the other, stands it upright, slides its mounting foot into the camera's
ISO 518 hot shoe from the back until it stops, lets go and pushes the lock lever's tab across to LOCK.

Metres; x forward from the arm bases, y left, z up. Published sizes are the makers' (see the README); every other
size, mass, force and friction is an estimate marked as such.
"""
import math
import pathlib
import sys
from collections import deque

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import box  # noqa: E402
from yam_joints import Hinge, Link, Slide  # noqa: E402,F401
from yam_mesh import Mesh, MeshMechanism  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
RELEASE_CLEAR = 0.030
STILL_S = 1.0
TEST_GRIPPERS = None
DIAG = False  # GPU diagnostic copies print FLASH_DIAG lines
DEMO_PACE = 1.0  # scales every demo duration except the fingers' close and open
DEMO_SHORT = 0.0  # wrong-way copies: stop the foot this far (m) short of the stop, then lock the lever anyway
DEMO_NO_HOLD = False  # wrong-way copy: the right hand never holds the camera


# ---------------------------------------------------------------- run context (episode time, grippers)
# The run-context helpers below follow the pattern of the task garnish-glass-rim-with-lime-wedge.
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


def grippers():
    if TEST_GRIPPERS is not None:
        return TEST_GRIPPERS
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN as OPEN

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
        opening = float(np.clip(np.mean(q[finger_ids]) / OPEN, 0.0, 1.0))
        out[arm] = (tcp, opening)
    return out


def _episode_memory():
    global _MEMORY_EPISODE
    if _MEMORY_EPISODE != _RUN["episode"]:
        _EVER.clear()
        _MEMORY_EPISODE = _RUN["episode"]


def ever(key, cond):
    """Remember a satisfied condition until the next episode; offline each call stands alone."""
    if _RUN["t"] is None:
        return bool(cond)
    _episode_memory()
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]


TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S, SCORE_OVER = 1.0, "end"
RUN_LIMIT_STEPS = 2400  # 120 s at 20 Hz
MM = 0.001
DEG = math.pi / 180

CAMERA_BLACK = (0.07, 0.07, 0.075)
SHOE_METAL = (0.55, 0.56, 0.58)
FLASH_BLACK = (0.10, 0.10, 0.105)
WINDOW_WHITE = (0.86, 0.86, 0.84)
LEVER_GREY = (0.30, 0.30, 0.32)
CAP_BLACK = (0.13, 0.13, 0.14)


# ---------------------------------------------------------------- convex pieces
def _oriented(name, points, faces, color, proxy):
    points, faces = np.asarray(points, float), np.asarray(faces, int)
    tri = (points - points.mean(axis=0))[faces]
    if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        faces = faces[:, ::-1]
    return Mesh(name, points, faces, collision="convex", color=color, proxy=tuple(proxy))


def loft(name, rings, color, rot=None, offset=(0.0, 0.0, 0.0), proxy="box"):
    """A convex body through planar convex rings (same number of points, same order), joined ring to ring, with flat
    end caps. Points are in a local frame (metres), then turned by `rot` and moved by `offset` into the part frame.
    Offline stand-in: the box round its points (proxy="box") or none."""
    k = len(rings[0])
    points = np.array([p for ring in rings for p in ring], float)
    if rot is not None:
        points = points @ np.asarray(rot, float).T
    points = points + np.asarray(offset, float)
    faces = []
    for i in range(len(rings) - 1):
        for j in range(k):
            a, b = i * k + j, i * k + (j + 1) % k
            c, d = a + k, b + k
            faces += [(a, b, d), (a, d, c)]
    faces += [(0, j + 1, j) for j in range(1, k - 1)]
    last = (len(rings) - 1) * k
    faces += [(last, last + j, last + j + 1) for j in range(1, k - 1)]
    pr = ()
    if proxy == "box":
        lo, hi = points.min(axis=0), points.max(axis=0)
        pr = (box(tuple((lo + hi) / 2), tuple(np.maximum(hi - lo, 0.4 * MM))),)
    return _oriented(name, points, faces, color, pr)


def _outline(x0, x1, y0, y1, c=(0.0, 0.0, 0.0, 0.0)):
    """Rectangle x0..x1, y0..y1 with its corners cut at 45 degrees by c = (x0y0, x1y0, x1y1, x0y1); a corner with no
    cut is one point, a cut corner two (so outlines with the same cut pattern have the same number of points)."""
    a, b, d, e = c
    pts = []
    for cut, p_cut, p_corner in ((a, [(x0, y0 + a), (x0 + a, y0)], (x0, y0)),
                                 (b, [(x1 - b, y0), (x1, y0 + b)], (x1, y0)),
                                 (d, [(x1, y1 - d), (x1 - d, y1)], (x1, y1)),
                                 (e, [(x0 + e, y1), (x0, y1 - e)], (x0, y1))):
        pts += p_cut if cut > 0 else [p_corner]
    return pts


def block(name, x0, x1, y0, y1, z0, z1, color, corners=(0.0, 0.0, 0.0, 0.0), bottom=None, top=None, rot=None,
          offset=(0.0, 0.0, 0.0), proxy="box"):
    """A box x0..x1, y0..y1, z0..z1 with vertical corner cuts `corners` (see _outline) and optional horizontal edge
    chamfers: bottom/top = (dx0, dx1, dy0, dy1, h): that face's outline is pulled in by dx0.. at its edges over the
    height h (a 45-degree chamfer when the pull-in equals h)."""
    def ring(z, pull):
        px0, px1, py0, py1 = pull
        return [(x, y, z) for x, y in _outline(x0 + px0, x1 - px1, y0 + py0, y1 - py1, corners)]
    rings = []
    if bottom is not None:
        rings.append(ring(z0, bottom[:4]))
        rings.append(ring(z0 + bottom[4], (0, 0, 0, 0)))
    else:
        rings.append(ring(z0, (0, 0, 0, 0)))
    if top is not None:
        rings.append(ring(z1 - top[4], (0, 0, 0, 0)))
        rings.append(ring(z1, top[:4]))
    else:
        rings.append(ring(z1, (0, 0, 0, 0)))
    return loft(name, rings, color, rot, offset, proxy)


def _circle(r, n, x, y0=0.0, z0=0.0):
    return [(x, y0 + r * math.cos(2 * math.pi * k / n), z0 + r * math.sin(2 * math.pi * k / n)) for k in range(n)]


# ---------------------------------------------------------------- the camera (Canon EOS Rebel T7)
# Published (Canon instruction manual): 129.0 W x 101.3 H x 77.6 D mm, 475 g with battery and card. Hot shoe: ISO 518
# (internal 18.6 wide, 16.0 +3/-2 deep, 2 +0.15/-0 high; lips 12.5 apart). Everything else here is an estimate from
# the manual's drawings: the lens axis 54 mm from the left end and 75 mm from the grip end; body 64 deep without the
# grip and 78 tall at the shoulders; grip 31 wide, to 77.6 deep; prism hump to 96.2 mm with the shoe on it, the shoe
# 20.7 x 18 x 5.1 mm outside (Wikipedia's example figures for a D3400 shoe), its rear 4 mm in front of the back face;
# eyecup 36 wide, 18 deep, 72-95 mm up; body cap 60 mm across, 16 mm proud.
# Frame: origin on the table under the lens axis, in the plane of the camera's back face; x forward (the lens), y to
# the camera's left (photographer's view), z up.
CAM_LEFT, CAM_RIGHT = 54.0 * MM, -75.0 * MM
CAM_BODY_D, CAM_GRIP_D, CAM_H = 64.0 * MM, 77.6 * MM, 101.3 * MM
CAM_SHOULDER = 78.0 * MM
GRIP_Y0, GRIP_Y1 = -75.0 * MM, -44.0 * MM
PRISM_TOP = 96.2 * MM
SHOE_X0, SHOE_LEN = 4.0 * MM, 18.0 * MM  # shoe rear edge (x) and outside length
SHOE_FLOOR = 98.2 * MM  # top of the shoe's floor plate
SLOT_H = 2.1 * MM  # ISO 518: 2 +0.15/-0
LIP_T = 1.0 * MM
SLOT_W = 18.6 * MM  # ISO 518 internal width
LIP_GAP = 12.5 * MM  # ISO 518 spacing of the teeth
SHOE_W = 20.7 * MM
SHOE_DEPTH = 17.0 * MM  # internal depth to the front stop (ISO 518: 16 +3/-2)
SHOE_LEAD = 1.0 * MM  # 45-degree lead-in at the rear of the rails, lips and floor (estimate)
LIP_UNDER = SHOE_FLOOR + SLOT_H  # 100.3
SHOE_STOP_X = SHOE_X0 + SHOE_DEPTH  # 21.0: the foot's front edge stops here
CAMERA_MASS = 0.475 + 0.010  # with battery, card and body cap (cap ~10 g, estimate)
CAMERA_FRICTION = (0.45, 0.40)  # body on the table (estimate)
SHOE_FRICTION = (0.20, 0.15)  # the foot plate and the shoe's rails and spring plate: smooth metal on metal (estimate)
# The kit gives each body one friction, so the shoe's rails (on the camera) and the foot (on the flash) are their own
# links on stiff, preloaded slides that cannot move more than LOCKED_TRAVEL: in effect fixed, with metal's friction.
LOCKED_K, LOCKED_PRELOAD, LOCKED_TRAVEL = 20000.0, 1.0 * MM, 0.05 * MM  # 20 N preload against the slide's stop
SHOE_RAILS_MASS, FOOT_LINK_MASS = 0.005, 0.002
# The shoe's spring: a plate in the floor sprung up, pressing the foot against the lips (estimate: ~0.9 N at rest,
# ~1.8 N with the foot over it), so the foot drags (~1 N) as in a real shoe. Plate 10 x 12 mm, its top 0.5 mm over the floor
# at rest, with a 3.5 mm ramp at its rear end that the foot rides up.
SPRING_X0, SPRING_X1 = 7.0 * MM, 17.0 * MM
SPRING_TOP = SHOE_FLOOR + 0.5 * MM
SPRING_K, SPRING_PRELOAD, SPRING_TRAVEL = 3000.0, 0.3 * MM, 0.8 * MM


def _camera_meshes():
    c = CAMERA_BLACK
    m = []
    m.append(block("body", 0.0, CAM_BODY_D, CAM_RIGHT, CAM_LEFT, 0.0, CAM_SHOULDER, c,
                   corners=(6 * MM, 6 * MM, 6 * MM, 6 * MM), top=(2 * MM, 2 * MM, 2 * MM, 2 * MM, 2 * MM)))
    m.append(block("grip", CAM_BODY_D - 6 * MM, CAM_GRIP_D, GRIP_Y0, GRIP_Y1, 0.0, CAM_SHOULDER - 2 * MM, c,
                   corners=(0.0, 9 * MM, 9 * MM, 0.0), top=(0, 3 * MM, 3 * MM, 3 * MM, 3 * MM)))
    m.append(loft("prism", [[(x, y, CAM_SHOULDER - 1 * MM) for x, y in _outline(2 * MM, 58 * MM, -28 * MM, 28 * MM,
                                                                                (4 * MM,) * 4)],
                            [(x, y, PRISM_TOP) for x, y in _outline(2 * MM, 30 * MM, -22 * MM, 22 * MM,
                                                                    (3 * MM,) * 4)]], c))
    m.append(block("eyecup", -18 * MM, 0.5 * MM, -18 * MM, 18 * MM, 72 * MM, 95 * MM, CAP_BLACK,
                   corners=(5 * MM, 0, 0, 5 * MM), top=(2 * MM, 0, 2 * MM, 2 * MM, 2 * MM)))
    m.append(loft("mount", [_circle(30 * MM, 20, CAM_BODY_D - 1 * MM, 0.0, 46 * MM),
                            _circle(30 * MM, 20, CAM_BODY_D + 6 * MM, 0.0, 46 * MM)], SHOE_METAL))
    m.append(loft("cap", [_circle(30 * MM, 20, CAM_BODY_D + 6 * MM, 0.0, 46 * MM),
                          _circle(28.5 * MM, 20, CAM_BODY_D + 16 * MM, 0.0, 46 * MM)], CAP_BLACK))
    return tuple(m)


def _shoe_meshes():
    """The hot shoe: floor plate, two side walls, two lips, front stop; lead-ins at the rear."""
    m = []
    s = SHOE_METAL
    xs1 = SHOE_X0 + SHOE_LEN
    m.append(block("shoe_floor", SHOE_X0, xs1, -SHOE_W / 2, SHOE_W / 2, PRISM_TOP - 0.5 * MM, SHOE_FLOOR, s,
                   top=(0.8 * MM, 0, 0, 0, 0.8 * MM)))
    wi, wo = SLOT_W / 2, SHOE_W / 2
    lead = SHOE_LEAD
    for side, sg in (("l", 1.0), ("r", -1.0)):
        # side wall: inner face at |y| = wi, its rear inner corner cut (lateral lead-in)
        corners = (0.0, 0.0, 0.0, lead) if sg > 0 else (lead, 0.0, 0.0, 0.0)
        y0, y1 = (wi, wo) if sg > 0 else (-wo, -wi)
        m.append(block(f"shoe_wall_{side}", SHOE_X0, xs1, y0, y1, SHOE_FLOOR, LIP_UNDER + LIP_T, s, corners=corners))
        # lip: from the wall in to |y| = LIP_GAP/2; its rear lower edge chamfered (vertical lead-in), rear inner
        # corner cut
        g = LIP_GAP / 2
        y0, y1 = (g, wi + 0.2 * MM) if sg > 0 else (-wi - 0.2 * MM, -g)
        corners = (0.0, 0.0, 0.0, lead) if sg > 0 else (lead, 0.0, 0.0, 0.0)
        m.append(block(f"shoe_lip_{side}", SHOE_X0, xs1, y0, y1, LIP_UNDER, LIP_UNDER + LIP_T, s, corners=corners,
                       bottom=(lead, 0.0, 0.0, 0.0, LIP_T - 0.05 * MM)))
    m.append(block("shoe_stop", SHOE_STOP_X, xs1, -wi, wi, SHOE_FLOOR, LIP_UNDER + LIP_T, s))
    return tuple(m)


def _spring_mesh():
    return block("spring_plate", SPRING_X0, SPRING_X1, -6 * MM, 6 * MM, SPRING_TOP - 1.5 * MM, SPRING_TOP,
                 SHOE_METAL, top=(3.5 * MM, 1.2 * MM, 0.0, 0.0, 0.7 * MM))


CAMERA_LIFT = 0.0003


def make_camera(pos):
    rails = Link("shoe", [], Slide("z", (SHOE_X0 + SHOE_LEN / 2, 0.0, SHOE_FLOOR), -LOCKED_TRAVEL, 0.0, damping=5.0,
                                   spring=LOCKED_K, rest=LOCKED_PRELOAD, armature=0.01),
                 SHOE_RAILS_MASS, color=SHOE_METAL, friction=SHOE_FRICTION)
    spring = Link("shoe_spring", [], Slide("z", (0.5 * (SPRING_X0 + SPRING_X1), 0.0, SPRING_TOP - 0.75 * MM),
                                           -SPRING_TRAVEL, 0.0, damping=2.0, spring=SPRING_K,
                                           rest=SPRING_PRELOAD, armature=0.01),
                  1.0e-3, color=SHOE_METAL, friction=SHOE_FRICTION)
    return MeshMechanism("camera", pos, (), [rails, spring], mass=CAMERA_MASS - 1.0e-3 - SHOE_RAILS_MASS,
                         color=CAMERA_BLACK, friction=CAMERA_FRICTION, contact_offset=0.001, grip=1,
                         meshes={"base": _camera_meshes(), "shoe": _shoe_meshes(), "shoe_spring": (_spring_mesh(),)})


# ---------------------------------------------------------------- the flash (Canon Speedlite 430EX III-RT)
# Published (Canon instruction manual): 70.5 W x 113.8 H x 98.2 D mm; 295 g without batteries; the mounting foot lock
# lever slides to the right and clicks at LOCK. Estimates: four AA alkaline cells ~92 g (total ~387 g); foot plate
# 18.2 x 19.0 x 1.9 mm, its front corners cut 2.0 mm and its front edges 0.6 mm; neck 11 mm wide through the shoe's
# lips; foot housing 36 x 28 mm, 4.5-14 mm up; body (battery box) 70.5 x 50 mm, 14-69 mm up, back face 3.5 mm behind
# the foot's rear edge; head 70.5 x 98.2 mm, 69-113.8 mm up, a hollow 2.5 mm shell (so the batteries' weight sits
# low), reaching 48 mm in front of the body; lever: a bar across the housing's back, 4 mm proud, with a tab 6 mm
# proud and 8.4 mm tall at its left end; travel 7 mm; friction ~2 N to start, ~1.5 N moving (estimates; no click).
# Frame (the flash's own): origin at the middle of the foot plate's underside; x forward (the window), y left, z up.
FOOT_W, FOOT_L, FOOT_T = 18.2 * MM, 19.0 * MM, 1.9 * MM
FOOT_CORNER, FOOT_EDGE = 2.0 * MM, 0.6 * MM
NECK_W, NECK_TOP = 11.0 * MM, 4.5 * MM
HOUSE_X0, HOUSE_X1, HOUSE_W, HOUSE_TOP = -13.0 * MM, 15.0 * MM, 36.0 * MM, 14.0 * MM
BODY_X0, BODY_X1, BODY_W, BODY_Z0, BODY_Z1 = -13.0 * MM, 37.0 * MM, 70.5 * MM, 14.0 * MM, 69.0 * MM
HEAD_X0, HEAD_X1, HEAD_Z1, SHELL = -13.0 * MM, 85.2 * MM, 113.8 * MM, 2.5 * MM
LEVER_PROUD, TAB_PROUD, TAB_W, TAB_Z0, TAB_Z1 = 4.0 * MM, 6.0 * MM, 4.0 * MM, 4.5 * MM, 12.9 * MM
LEVER_Y0, LEVER_Y1 = -16.0 * MM, 18.0 * MM  # the bar at UNLOCK (its left end carries the tab)
LEVER_TRAVEL, LOCKED_AT = 7.0 * MM, 6.0 * MM
FLASH_MASS = 0.387
LEVER_MASS = 0.004
FLASH_FRICTION = (0.55, 0.45)  # plastic body (estimate)
FLASH_LIFT = 0.0003
# The flash starts lying on its right side: its own frame is the part frame turned +90 degrees about x (the kit's
# parts start upright, so the turn is built into the part's points).
START_ROT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _foot_meshes(rot):
    b, m = FLASH_BLACK, []
    hw = FOOT_W / 2
    m.append(block("foot", -FOOT_L / 2, FOOT_L / 2, -hw, hw, 0.0, FOOT_T, SHOE_METAL,
                   corners=(0.3 * MM, FOOT_CORNER, FOOT_CORNER, 0.3 * MM),
                   bottom=(0.0, FOOT_EDGE, 0.0, 0.0, FOOT_EDGE), top=(0.0, FOOT_EDGE, 0.0, 0.0, FOOT_EDGE), rot=rot))
    m.append(block("neck", -FOOT_L / 2 + 1 * MM, FOOT_L / 2 - 2 * MM, -NECK_W / 2, NECK_W / 2, FOOT_T - 0.2 * MM,
                   NECK_TOP + 0.2 * MM, b, rot=rot))
    return tuple(m)


def _flash_base_meshes(rot):
    b, m = FLASH_BLACK, []
    m.append(block("house", HOUSE_X0, HOUSE_X1, -HOUSE_W / 2, HOUSE_W / 2, NECK_TOP, HOUSE_TOP + 0.5 * MM, b,
                   corners=(2 * MM,) * 4, rot=rot))
    m.append(block("body", BODY_X0, BODY_X1, -BODY_W / 2, BODY_W / 2, BODY_Z0, BODY_Z1 + 0.5 * MM, b,
                   corners=(4 * MM,) * 4, bottom=(2 * MM, 2 * MM, 2 * MM, 2 * MM, 2 * MM), rot=rot))
    hw, s = BODY_W / 2, SHELL
    z0 = BODY_Z1
    shell = [("head_top", HEAD_X0, HEAD_X1, -hw, hw, HEAD_Z1 - s, HEAD_Z1, b),
             ("head_bottom", BODY_X1 - 2 * MM, HEAD_X1, -hw, hw, z0, z0 + s, b),
             ("head_left", HEAD_X0, HEAD_X1, hw - s, hw, z0, HEAD_Z1, b),
             ("head_right", HEAD_X0, HEAD_X1, -hw, -hw + s, z0, HEAD_Z1, b),
             ("head_back", HEAD_X0, HEAD_X0 + s, -hw, hw, z0, HEAD_Z1, b),
             ("window", HEAD_X1 - s, HEAD_X1, -hw + s, hw - s, z0 + s, HEAD_Z1 - s, WINDOW_WHITE)]
    for name, x0, x1, y0, y1, za, zb, col in shell:
        m.append(block(name, x0, x1, y0, y1, za, zb, col, rot=rot))
    return tuple(m)


def _lever_meshes(rot):
    xb = HOUSE_X0
    bar = block("lever_bar", xb - LEVER_PROUD, xb + 0.5 * MM, LEVER_Y0, LEVER_Y1, NECK_TOP, NECK_TOP + 3.0 * MM,
                LEVER_GREY, corners=(1 * MM, 0, 0, 1 * MM), rot=rot)
    tab = block("lever_tab", xb - TAB_PROUD, xb + 0.5 * MM, LEVER_Y1 - TAB_W, LEVER_Y1, TAB_Z0, TAB_Z1, LEVER_GREY,
                corners=(1 * MM, 0, 0, 1 * MM), rot=rot)
    return (bar, tab)


def make_flash(pos, rot=START_ROT):
    """The flash as a free MeshMechanism whose part origin is `pos` and whose own frame is part frame @ `rot`. The
    lever is a slide along the flash's y (a part axis), from 0 (UNLOCK) toward the flash's right by LEVER_TRAVEL."""
    rot = np.asarray(rot, float)
    y_axis = rot[:, 1]
    axis = next(k for k, v in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1)))
                if abs(abs(float(np.dot(v, y_axis))) - 1.0) < 1e-9)
    sign = float(np.dot(dict(x=(1, 0, 0), y=(0, 1, 0), z=(0, 0, 1))[axis], y_axis))
    lower, upper = (-LEVER_TRAVEL, 0.0) if sign > 0 else (0.0, LEVER_TRAVEL)
    anchor = rot @ np.array([HOUSE_X0 - LEVER_PROUD / 2, 0.5 * (LEVER_Y0 + LEVER_Y1), NECK_TOP + 1.5 * MM])
    lever = Link("lever", [], Slide(axis, tuple(float(v) for v in anchor), lower, upper, damping=1.0,
                                    static_friction=2.0, dynamic_friction=1.5, armature=0.01),
                 LEVER_MASS, color=LEVER_GREY, friction=FLASH_FRICTION)
    # the foot: its own link (metal friction), on a slide along the flash's z that its 20 N preload holds at the top
    z_axis = rot[:, 2]
    zname = next(k for k, v in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1)))
                 if abs(abs(float(np.dot(v, z_axis))) - 1.0) < 1e-9)
    zsign = float(np.dot(dict(x=(1, 0, 0), y=(0, 1, 0), z=(0, 0, 1))[zname], z_axis))
    flo, fhi = (-LOCKED_TRAVEL, 0.0) if zsign > 0 else (0.0, LOCKED_TRAVEL)
    foot = Link("foot", [], Slide(zname, tuple(float(v) for v in rot @ np.array([0.0, 0.0, FOOT_T / 2])), flo, fhi,
                                  damping=5.0, spring=LOCKED_K, rest=zsign * LOCKED_PRELOAD, armature=0.01),
                FOOT_LINK_MASS, color=SHOE_METAL, friction=SHOE_FRICTION)
    return MeshMechanism("flash", pos, (), [lever, foot], mass=FLASH_MASS - LEVER_MASS - FOOT_LINK_MASS,
                         color=FLASH_BLACK, friction=FLASH_FRICTION, contact_offset=0.001, grip=1,
                         meshes={"base": _flash_base_meshes(rot), "lever": _lever_meshes(rot), "foot": _foot_meshes(rot)})


def lever_travel(flash):
    """How far (m) the lever has been slid from UNLOCK toward LOCK (0 .. LEVER_TRAVEL)."""
    return abs(float(flash.joints["lever"]))


def flash_frame(flash):
    """(origin, rotation) of the flash's own frame in the world."""
    return np.asarray(flash.origin, float), flash.rotation() @ START_ROT


# ---------------------------------------------------------------- layout
SHOE_Y = 0.020
CAMERA_XY = (0.125, SHOE_Y)  # the camera frame's origin: back plane, lens axis
FLASH_START = np.array([0.190, 0.232, TOP + BODY_W / 2 + FLASH_LIFT])  # lying flash: foot-plate origin

CAMERA = make_camera((*CAMERA_XY, TOP + CAMERA_LIFT))
FLASH = make_flash(tuple(FLASH_START))
OBJECTS = [CAMERA, FLASH]
PRESSED_PARTS = ("camera",)  # the flash's foot is slid into the camera's shoe; replay can't press the shoe's spring

# Home: the foot's front edge at the shoe's stop, its underside against the lips (pushed up by the spring).
HOME_IN_CAMERA = np.array([SHOE_STOP_X - FOOT_L / 2, 0.0, LIP_UNDER - FOOT_T])


def _T(R, p):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = R, p
    return m


def _inv(m):
    R, p = m[:3, :3], m[:3, 3]
    return _T(R.T, -R.T @ p)


def camera_pose_start():
    return _T(np.eye(3), np.array([*CAMERA_XY, TOP]))


def flash_pose_start():
    return _T(START_ROT, FLASH_START)


# ---------------------------------------------------------------- scoring
HOME_TOL = 1.5 * MM  # home: the foot's front edge within this of the stop
SHORT_TOL = 8.0 * MM  # part way: within this of the stop
IN_SHOE_MIN = 3.0 * MM  # in the shoe: the foot's front edge at least this far past the shoe's rear entry ...
RAIL_TOL = 2.0 * MM  # ... centred between the rails within this, its underside within this of the slot
LOCK_MOVED = 2.0 * MM
CAM_MOVE_OK, CAM_TURN_OK = 20.0 * MM, 10.0
FALLEN_DEG = 30.0
STILL_MOVE, STILL_DEG, SPEED = 1.0 * MM, 2.0, 0.01


def foot_terms(flash, camera):
    """Where the flash's foot is in the camera's shoe (metres, camera frame)."""
    oc, rc = np.asarray(camera.origin, float), camera.rotation()
    of, rf = flash_frame(flash)
    p = rc.T @ (of - oc)  # foot-plate origin in the camera frame
    front = p[0] + FOOT_L / 2
    short = SHOE_STOP_X - front
    up_in_cam = rc.T @ rf[:, 2]
    in_shoe = bool(front >= SHOE_X0 + IN_SHOE_MIN and abs(p[1]) <= RAIL_TOL and
                   abs(p[2] - (LIP_UNDER - FOOT_T)) <= RAIL_TOL and up_in_cam[2] > math.cos(20 * DEG))
    return {"foot": p, "short": float(short), "in_shoe": in_shoe,
            "flash_tilt": math.degrees(math.acos(float(np.clip(rf[2, 2], -1.0, 1.0))))}


def camera_terms(camera):
    o = np.asarray(camera.origin, float)
    r = camera.rotation()
    move = float(math.hypot(o[0] - CAMERA_XY[0], o[1] - CAMERA_XY[1]))
    turn = abs(math.degrees(math.atan2(r[1, 0], r[0, 0])))
    return {"move": move, "turn": turn, "tilt": camera.tilt_deg()}


def _box_clear(state, clear):
    hands = grippers()
    if hands is None:
        return True
    low, high = np.asarray(state.low, float), np.asarray(state.high, float)
    return all(float(np.linalg.norm(np.maximum(np.maximum(low - tcp, tcp - high), 0.0))) >= clear
               for tcp, _ in hands.values())


def _rot_deg(a, b):
    return math.degrees(math.acos(float(np.clip((np.trace(a.T @ b) - 1) / 2, -1.0, 1.0))))


_WINDOW = {"episode": None, "hist": deque()}


def still(flash):
    """Over the last second the flash moved under 1 mm and turned under 2 degrees (offline: its velocity)."""
    t = _RUN["t"]
    if t is None:
        return bool(np.linalg.norm(flash.lin_vel) < SPEED)
    hist = _WINDOW["hist"]
    if _WINDOW["episode"] != _RUN["episode"] or (hist and t < hist[-1][0]):
        hist.clear()
        _WINDOW["episode"] = _RUN["episode"]
    now = (np.asarray(flash.origin, float), flash.rotation())
    if not hist or t > hist[-1][0] + 1e-9:
        hist.append((t, now))
    while len(hist) > 1 and hist[1][0] <= t - STILL_S + 1e-9:
        hist.popleft()
    if t - hist[0][0] < STILL_S - 0.051:
        return False
    return all(np.linalg.norm(now[0] - old[0]) <= STILL_MOVE and _rot_deg(now[1], old[1]) <= STILL_DEG
               for _, old in hist)


def score_terms(objects):
    flash, camera = objects["flash"], objects["camera"]
    f = foot_terms(flash, camera)
    c = camera_terms(camera)
    lever = lever_travel(flash)
    fallen = ever("camera fell over", c["tilt"] > FALLEN_DEG)
    free = _box_clear(flash, RELEASE_CLEAR) and _box_clear(camera, RELEASE_CLEAR)
    rest = still(flash)
    if DIAG:
        _diag(flash, camera, f, c, lever, free, rest)
    return {**f, **{"cam_" + k: v for k, v in c.items()}, "lever": lever, "fallen": fallen, "released": free,
            "still": rest}


def score(objects):
    t = score_terms(objects)
    if t["fallen"] or not t["in_shoe"]:
        return 0.0
    foot = 1.0 if t["short"] <= HOME_TOL else 0.5 if t["short"] <= SHORT_TOL else 0.25
    lever = 1.0 if t["lever"] >= LOCKED_AT else 0.75 if t["lever"] >= LOCK_MOVED else 0.5
    cam = 1.0 if t["cam_move"] <= CAM_MOVE_OK and t["cam_turn"] <= CAM_TURN_OK else 0.5
    hands = 1.0 if t["released"] and t["still"] else 0.5
    return min(foot, lever, cam, hands)


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


def _diag(flash, camera, f, c, lever, free, rest):
    step = _RUN["step"]
    if _RUN["t"] is None or step == _DIAG["step"] or step % 5:
        return
    _DIAG["step"] = step
    hands = grippers() or {}
    tcp = " ".join(f"{a[0]}=({','.join(f'{v * 1000:.1f}' for v in hands[a][0])};{hands[a][1]:.2f})" for a in hands)
    p = f["foot"]
    print(f"FLASH_DIAG t={_RUN['t']:.2f} foot=({p[0] * 1000:.1f},{p[1] * 1000:.2f},{p[2] * 1000:.2f}) "
          f"short={f['short'] * 1000:.1f} in={int(f['in_shoe'])} tilt={f['flash_tilt']:.1f} "
          f"lever={lever * 1000:.2f} spring={float(camera.joints['shoe_spring']) * 1000:.2f} "
          f"cam=({c['move'] * 1000:.1f};{c['turn']:.1f};{c['tilt']:.1f}) free={int(free)} rest={int(rest)} {tcp}",
          flush=True)


# ---------------------------------------------------------------- demo
# The right hand holds the camera by its grip (a pinch across the grip, front to back, so it holds along the push),
# coming in from the camera's right, pitched 45 degrees. The left hand picks the flash up across its body (front and
# back faces), turns it upright about the pinch, carries it behind the camera, holds it still over the shoe (the arm
# sags under the flash, so a reading there corrects the hand's aim), lowers the foot to the shoe's rear entry, slides
# it in past the stop (the arm's give sets the push), rests it in the shoe, lets go, backs its fingers out to the left and,
# with its closed fingertips, pushes the lever's tab across to LOCK. Then both hands let go and park. The demo reads
# the camera's and the flash's poses (scripted demos may read simulator state); it never moves an object.
PAD_TIP = 0.013  # the pad tips are 13 mm past the TCP
PICK_Z = 46.0 * MM  # pinch height on the flash (its own z): as near its centre of mass (52.6 mm) as the head's
# forward overhang lets the front finger go (~3 mm clear), so gravity barely twists it about the pinch
PICK_X = 0.5 * (BODY_X0 + BODY_X1)  # pinch centre on the body (its own x)
FLASH_OPEN = -(0.5 * (BODY_X1 - BODY_X0) + 7.0 * MM)  # pads 64 mm apart round the 50 mm body
FLASH_PINCH = -(0.5 * (BODY_X1 - BODY_X0) - 6.0 * MM)  # aimed 38 mm apart round the 50 mm body: ~30 N a finger,
# a firm pinch, so the flash turns with the hand (a pinch holds a turn about its own line by friction only)
FLASH_RELEASE = FINGER_OPEN  # fully open (~94 mm) to back out
CAM_OPEN = FINGER_OPEN  # ~94 mm
CAM_PINCH = -(0.5 * CAM_GRIP_D - 1.5 * MM)  # aimed 74.6 mm apart across the 77.6 mm grip
GRIP_HOLD_Y, GRIP_HOLD_Z = -60.0 * MM, 45.0 * MM  # camera frame: the grip pinch's centre
HOLD_PITCH = 45.0
PUSH_PAST = 10.0 * MM  # the push aims the foot this far past the stop (the arm's give sets the push, ~3 N)
HOME_PUSH_MAX = 6.0 * MM
EASE_PAST = 2.0 * MM  # easing off, resting and letting go, the hand still aims this far past the stop (a light push),
# so it does not draw the foot back off the stop  # if the foot stalls short, the hand aims up to this much further once more
PUSH_STEPS, PUSH_STEP_S = 8, 0.55
LEVER_PITCH = 45.0
LEVER_PAST = 12.0 * MM  # the lever push aims this far past LOCK (the arm's give sets the push, ~2 N)
FINAL_HOLD_S = 3.0
PARK = {"left": np.array([0.150, 0.300, TOP + 0.200]), "right": np.array([0.150, -0.300, TOP + 0.200])}


def _grip(close, approach):
    c = np.asarray(close, float)
    a = np.asarray(approach, float)
    c, a = c / np.linalg.norm(c), a / np.linalg.norm(a)
    return np.column_stack((c, np.cross(a, c), a))


def _rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _hand_in_flash():
    """The left hand's pose in the flash's own frame: TCP at the body's middle (front to back) PICK_Z up, closing
    along the flash's -x, approaching from the flash's left (fingers pointing along the flash's -y)."""
    return _T(_grip((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)), np.array([PICK_X, 0.0, PICK_Z]))


_HAND_IN_FLASH = _hand_in_flash()


def flash_at_shoe(dx, camera=None):
    """The flash upright on the camera's shoe axis with its foot dx (m) forward of home (negative: behind)."""
    cam = camera_pose_start() if camera is None else camera
    return cam @ _T(np.eye(3), HOME_IN_CAMERA + np.array([dx, 0.0, 0.0]))


def _hold_hand(camera=None):
    """The right hand's pose for the grip pinch, in the world, for the camera at `camera` (4x4)."""
    cam = camera_pose_start() if camera is None else camera
    p = math.radians(HOLD_PITCH)
    hand = _T(_grip((1.0, 0.0, 0.0), (0.0, math.cos(p), -math.sin(p))),
              np.array([0.5 * CAM_GRIP_D, GRIP_HOLD_Y, GRIP_HOLD_Z]))
    return cam @ hand


TIP_HALF_W, TIP_HALF_T = 7.7 * MM, 4.6 * MM  # closed fingertips: half width and half thickness (first 10 mm)
LEVER_X_LEAD = 0.8 * MM  # the fingertips' front-upper edge aims this far forward of the housing's back face
LEVER_TIP_Z = 6.5 * MM  # height of the fingertips' end (flash frame): the tab spans 4.5-12.9 mm


def _lever_hand(y_face, flash_pose, dx=0.0):
    """The left hand (closed) pitched LEVER_PITCH forward-down, closing along y, its fingertips' -y face at y_face
    (flash frame y) beside the lever tab; dx moves it along the flash's x. flash_pose: the flash's frame (4x4)."""
    p = math.radians(LEVER_PITCH)
    a = np.array([math.cos(p), 0.0, -math.sin(p)])
    R = _grip((0.0, 1.0, 0.0), a)
    tip = np.array([HOUSE_X0 + LEVER_X_LEAD - TIP_HALF_W * math.sin(p) + dx, y_face + TIP_HALF_T + 0.2 * MM,
                    LEVER_TIP_Z])
    tcp = tip - a * PAD_TIP  # the fingertips' end is PAD_TIP past the TCP along the approach
    return flash_pose @ _T(R, tcp)


# ---------------------------------------------------------------- reads (scripted demos may read the simulator)
def _state_now(name):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        return arena_scene.object_states(env, [name])[name]
    except Exception as e:  # noqa: BLE001
        print(f"FLASH_READ {name} failed: {e!r}", flush=True)
        return None


_READS = {"episode": None, "values": {}}
CAM_READ_MAX = 0.025
HELD_READ_MAX = 0.040  # the left arm sags ~25 mm under the held flash (probe 2), so this read corrects that much


def _reads():
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    return _READS["values"]


def _camera_read():
    """The camera's pose (4x4), read once per episode before the right hand comes down (it is untouched then)."""
    v = _reads()
    if "camera" not in v:
        pose = camera_pose_start()
        st = _state_now("camera")
        if st is not None:
            o, r = np.asarray(st.origin, float), st.rotation()
            yaw = math.atan2(r[1, 0], r[0, 0])
            shift = np.clip(o[:2] - np.asarray(CAMERA_XY), -CAM_READ_MAX, CAM_READ_MAX)
            c, s = math.cos(yaw), math.sin(yaw)
            pose = _T(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]),
                      np.array([CAMERA_XY[0] + shift[0], CAMERA_XY[1] + shift[1], TOP]))
            print(f"FLASH_READ camera t={_RUN['t']:.2f} shift_mm=({shift[0] * 1000:.1f},{shift[1] * 1000:.1f}) "
                  f"yaw_deg={math.degrees(yaw):.2f}", flush=True)
        v["camera"] = pose
    return v["camera"]


def _flash_now():
    st = _state_now("flash")
    if st is None:
        return None
    o, r = flash_frame(st)
    return _T(r, o)


def _axis_angle(rot):
    c = (np.trace(rot) - 1.0) / 2.0
    a = math.acos(float(np.clip(c, -1.0, 1.0)))
    if a < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]) / (2 * math.sin(a))
    return axis, a


def _limit_rotation(rot, max_deg):
    axis, ang = _axis_angle(rot)
    if ang <= math.radians(max_deg):
        return rot
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = math.radians(max_deg)
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * (k @ k)


def _flash_read(key, planned, kind, ref, max_shift, max_deg):
    """A correction (4x4 world transform) from comparing the flash's actual pose with `planned` (4x4) about the
    reference point `ref` (flash frame). kind "follow": the hand goes to where the flash is (not yet held);
    "move": the held flash is brought to where it was planned."""
    v = _reads()
    if key in v:
        return v[key]
    actual = _flash_now()
    if actual is None:
        v[key] = np.eye(4)
        return v[key]
    pt_p = planned[:3, :3] @ ref + planned[:3, 3]
    pt_a = actual[:3, :3] @ ref + actual[:3, 3]
    if kind == "follow":
        rot = _limit_rotation(actual[:3, :3] @ planned[:3, :3].T, max_deg)
        src, dst = pt_p, pt_a
    else:
        rot = _limit_rotation(planned[:3, :3] @ actual[:3, :3].T, max_deg)
        src, dst = pt_a, pt_p
    move = dst - src
    n = float(np.linalg.norm(move))
    if n > max_shift:
        move *= max_shift / n
    d = _T(rot, src + move - rot @ src)
    err = math.degrees(_axis_angle(actual[:3, :3] @ planned[:3, :3].T)[1])
    print(f"FLASH_READ {key} t={_RUN['t']:.2f} ref_err_mm={tuple(round(float(x) * 1000, 2) for x in pt_p - pt_a)} "
          f"rot_err_deg={err:.2f}", flush=True)
    v[key] = d
    return d


class Grip(float):
    def __new__(cls, yaw, matrix=None):
        value = super().__new__(cls, yaw)
        value.grip_matrix = None if matrix is None else np.array(matrix, float)
        return value


def _g(R):
    """A Grip for the hand rotation R; its yaw value is the one whose straight-down grip closes along R's closing
    axis (projected on the table), so the offline checker's top-down finger model points the right way."""
    c = np.asarray(R, float)[:, 0]
    yaw = math.atan2(-c[0], c[1]) if math.hypot(c[0], c[1]) > 1e-6 else 0.0
    return Grip(yaw, R)


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


class PoseWaypoint(Waypoint):
    """A waypoint whose hand pose (4x4) is worked out when it is played, from a function of the reads."""

    def __init__(self, duration, pose_fn, finger, label, keep=False):
        self.duration = duration * (1.0 if keep else DEMO_PACE)
        self.pose_fn, self.finger, self.label = pose_fn, finger, label

    @property
    def pos(self):
        return tuple(float(v) for v in self.pose_fn()[:3, 3])

    @property
    def yaw(self):
        return _g(self.pose_fn()[:3, :3])


# Flash read reference points (flash frame): the pinch centre for the pick, the foot for the shoe.
REF_PICK = np.array([PICK_X, 0.0, PICK_Z])
REF_FOOT = np.array([0.0, 0.0, 0.0])


LIFT_DZ = 0.065  # straight up off the table before the turn
TURN_TCP = np.array([0.150, 0.100, TOP + 0.200])  # where the hand stands the flash upright (level, pointing -y)


def _lift_turn_poses():
    """Planned flash poses (4x4) after the pick: lifted straight up, then carried toward the camera while the hand
    turns 90 degrees about its closing axis, which stands the flash upright."""
    f0 = flash_pose_start()
    up = _T(f0[:3, :3], f0[:3, 3] + np.array([0.0, 0.0, LIFT_DZ]))
    hand_turned = _T((up @ _HAND_IN_FLASH)[:3, :3] @ np.eye(3), TURN_TCP)
    hand_turned[:3, :3] = _rot_x(-90.0) @ (up @ _HAND_IN_FLASH)[:3, :3]
    turn = hand_turned @ _inv(_HAND_IN_FLASH)
    return up, turn


ENTRY_BACK = 8.0 * MM  # the foot's front edge waits this far behind the shoe's rear entry (the housing clear of it)
ENTRY_HIGH = 0.3 * MM  # ... and its underside this far over its home height (the spring plate's top is 0.3 higher)
ENTRY_DX = (SHOE_X0 - ENTRY_BACK - FOOT_L / 2) - HOME_IN_CAMERA[0]  # -22 mm: dx of the waiting foot from home
READ_HIGH = 0.030  # the line-up read: the held flash waits this far over its entry height, in the air (the arm sags
# under the flash, and the eyecup's top is only 3.4 mm under the entry height)


def demo_plan():
    """Both arms' waypoint lists as (label, seconds, pose function, finger, keep)."""
    f0 = flash_pose_start()
    up, turn = _lift_turn_poses()
    H = _HAND_IN_FLASH

    def cam():
        return _camera_read()

    def pick_fix():
        return _flash_read("pick", f0, "follow", REF_PICK, 0.020, 15.0)

    def shoe_fix1():
        planned = flash_at_shoe(ENTRY_DX, cam())
        planned[2, 3] += READ_HIGH
        return _flash_read("shoe", planned, "move", REF_FOOT, HELD_READ_MAX, 8.0)

    def shoe_fix():
        d1 = shoe_fix1()
        if not _reads().get("shoe2_due"):
            return d1
        d2 = _flash_read("shoe2", flash_at_shoe(ENTRY_DX, cam()), "move", REF_FOOT, 0.004, 3.0)
        return d2 @ d1

    def at(dx, extra=(0.0, 0.0, 0.0), fix=True):
        def f():
            d = shoe_fix() if fix else np.eye(4)
            h = d @ flash_at_shoe(dx, cam()) @ H
            h[:3, 3] += np.asarray(extra, float)
            return h
        return f

    def at_checked(dx):
        def f():
            _reads()["shoe2_due"] = True
            return at(dx)()
        return f

    def settled(dx):
        """The hand where it is loaded (corrected x and y) but at its unloaded height and turn: the shoe takes the
        flash's weight before the fingers open, so the hand does not spring up into the head when it lets go."""
        def f():
            h = at(dx)()
            n = at(dx, fix=False)()
            n[:2, 3] = h[:2, 3]
            return n
        return f

    def pushed(dx):
        """Once at the stop: if the foot is short, aim further (twice the shortfall, at most HOME_PUSH_MAX)."""
        def f():
            v = _reads()
            if "stop" not in v:
                v["stop"] = 0.0
                fl, ca = _state_now("flash"), _state_now("camera")
                if fl is not None and ca is not None:
                    s = foot_terms(fl, ca)["short"]
                    v["stop"] = float(np.clip(2.0 * (s - 0.3 * MM), 0.0, HOME_PUSH_MAX))
                    print(f"FLASH_READ stop t={_RUN['t']:.2f} short_mm={s * 1000:.2f} more_mm={v['stop'] * 1000:.1f}",
                          flush=True)
            return at(dx + v["stop"])()
        return f

    def held(pose, fix=None, extra=(0.0, 0.0, 0.0)):
        def f():
            d = fix() if fix is not None else np.eye(4)
            h = d @ pose @ H
            h[:3, 3] += np.asarray(extra, float)
            return h
        return f

    short = DEMO_SHORT
    home_dx = -short
    left = [
        ("over the lying flash", 2.5, held(_T(f0[:3, :3], f0[:3, 3] + np.array([0, 0, 0.050])), pick_fix),
         FLASH_OPEN, False),
        ("fingers down either side of its body", 1.5, held(f0, pick_fix), FLASH_OPEN, False),
        ("pinch its body front and back", 1.0, held(f0, pick_fix), FLASH_PINCH, True),
        ("lift it", 1.2, held(up, pick_fix), FLASH_PINCH, False),
        ("stand it upright on the way", 2.5, held(turn), FLASH_PINCH, False),
        ("carry it behind the camera, high", 2.0, at(ENTRY_DX, (0.0, 0.0, 0.060), fix=False), FLASH_PINCH, False),
        ("lower it over the shoe", 1.0, at(ENTRY_DX, (0.0, 0.0, READ_HIGH), fix=False), FLASH_PINCH, False),
        ("hold it still over the shoe", 0.8, at(ENTRY_DX, (0.0, 0.0, READ_HIGH), fix=False), FLASH_PINCH, False),
        ("foot down behind the shoe", 1.2, at(ENTRY_DX, (0.0, 0.0, ENTRY_HIGH)), FLASH_PINCH, False),
        ("line the foot up with the shoe", 0.8, at(ENTRY_DX), FLASH_PINCH, False),
        ("hold at the shoe's entry", 0.6, at(ENTRY_DX), FLASH_PINCH, False),
        ("check the line-up", 0.6, at_checked(ENTRY_DX), FLASH_PINCH, False),
    ]
    target = home_dx + (0.0 if short else PUSH_PAST)
    for k in range(PUSH_STEPS):
        dx = ENTRY_DX + (target - ENTRY_DX) * (k + 1) / PUSH_STEPS
        left.append((f"slide the foot in ({k + 1})", PUSH_STEP_S, at(dx), FLASH_PINCH, False))
    left.append(("hold it at the stop", 0.8, at(target), FLASH_PINCH, False))
    if not short:
        left.append(("push again if short of the stop", 0.8, pushed(target), FLASH_PINCH, False))
    ease = home_dx + (0.0 if short else EASE_PAST)
    left += [
        ("ease off", 0.5, at(ease), FLASH_PINCH, False),
        ("rest it in the shoe", 0.6, settled(ease), FLASH_PINCH, False),
        ("let go", 0.8, settled(ease), FLASH_RELEASE, True),
        ("back the fingers out", 1.4, at(home_dx, (0.0, 0.070, 0.0), fix=False), FLASH_RELEASE, False),
    ]

    def lever_pose(y_face, extra=(0.0, 0.0, 0.0), dx=0.0):
        def f():
            v = _reads()
            if "lever" not in v:
                actual = _flash_now()
                v["lever"] = actual if actual is not None else flash_at_shoe(home_dx, cam())
                if actual is not None:
                    o = actual[:3, 3]
                    print(f"FLASH_READ lever t={_RUN['t']:.2f} flash=({o[0] * 1000:.1f},{o[1] * 1000:.1f},"
                          f"{o[2] * 1000:.1f})", flush=True)
            h = _lever_hand(y_face, v["lever"], dx)
            h[:3, 3] += np.asarray(extra, float)
            return h
        return f

    tab_left = LEVER_Y1
    left += [
        ("close the fingers, rise", 1.2, at(home_dx, (0.0, 0.090, 0.050), fix=False), 0.0, False),
        ("turn the hand, behind and left of the flash", 2.0,
         lever_pose(tab_left + 3.0 * MM, (0.0, 0.0, 0.050), dx=-0.040), 0.0, False),
        ("over the lever tab", 1.5, lever_pose(tab_left + 3.0 * MM, (0.0, 0.0, 0.030), dx=-0.004), 0.0, False),
        ("down beside the tab", 1.2, lever_pose(tab_left + 3.0 * MM, dx=-0.004), 0.0, False),
        ("in behind the tab", 0.6, lever_pose(tab_left + 3.0 * MM), 0.0, False),
        ("touch the tab", 0.6, lever_pose(tab_left), 0.0, False),
    ]
    end_face = tab_left - LEVER_TRAVEL - LEVER_PAST
    n = 6
    for k in range(n):
        y = tab_left + (end_face - tab_left) * (k + 1) / n
        left.append((f"push the lever to LOCK ({k + 1})", 0.45, lever_pose(y), 0.0, False))
    left += [
        ("hold at LOCK", 0.5, lever_pose(end_face), 0.0, False),
        ("back off the tab", 0.8, lever_pose(tab_left - LEVER_TRAVEL + 3.0 * MM, dx=-0.004), 0.0, False),
        ("lift the fingers away", 1.2, lever_pose(tab_left - LEVER_TRAVEL + 3.0 * MM, (-0.030, 0.040, 0.060), dx=-0.004),
         0.0,
         False),
    ]
    t_insert = sum(s * (1.0 if k else DEMO_PACE) for _, s, _, _, k in left[:10])
    return left, t_insert


def _right_plan(t_left_total, t_insert):
    """The right hand: read the camera, come in pitched from the camera's right, pinch the grip before the foot
    reaches the shoe, hold until the left hand is clear, let go, back out and park."""
    def hold(extra=(0.0, 0.0, 0.0)):
        def f():
            h = _hold_hand(_camera_read())
            h[:3, 3] += np.asarray(extra, float)
            return h
        return f
    p = math.radians(HOLD_PITCH)
    back = (0.0, -0.060 * math.cos(p), 0.060 * math.sin(p))
    if DEMO_NO_HOLD:
        return []
    right = [
        ("wait", 1.5, None, CAM_OPEN, False),
        ("pitched, out to the camera's right", 2.5, hold(back), CAM_OPEN, False),
        ("fingers down either side of the grip", 1.5, hold(), CAM_OPEN, False),
        ("pinch the grip front and back", 1.0, hold(), CAM_PINCH, True),
    ]
    held_s = t_left_total - sum(r[1] for r in right) - 0.5
    right += [
        ("hold the camera", max(held_s, 1.0), hold(), CAM_PINCH, False),
        ("let go of the grip", 0.8, hold(), CAM_OPEN, True),
        ("back out", 1.2, hold(back), CAM_OPEN, False),
    ]
    return right


def demo():
    left_plan, t_insert = demo_plan()
    waypoints = [PoseWaypoint(s, f, finger, label, keep) for label, s, f, finger, keep in left_plan]
    t_left = sum(w.duration for w in waypoints)
    park_l = [Waypoint(2.0 * DEMO_PACE, tuple(float(v) for v in PARK["left"]), 0.0, "park",
                       _g(_grip((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0))))]
    right_plan = _right_plan(t_left, t_insert)
    rw = []
    for label, s, f, finger, keep in right_plan:
        if f is None:
            rw.append(Waypoint(s, tuple(float(v) for v in PARK["right"]), finger, label,
                               _g(_grip((1.0, 0.0, 0.0), (0.0, 0.0, -1.0)))))
        else:
            rw.append(PoseWaypoint(s, f, finger, label, keep))
    t_right = sum(w.duration for w in rw)
    park_r = [Waypoint(2.0 * DEMO_PACE, tuple(float(v) for v in PARK["right"]), CAM_OPEN, "park",
                       _g(_grip((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))))]
    total = max(t_left + 2.0, t_right + 2.0) + FINAL_HOLD_S
    left = waypoints + park_l + [Waypoint(max(total - t_left - 2.0, 0.5), tuple(float(v) for v in PARK["left"]), 0.0,
                                          "hold clear", _g(_grip((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0))))]
    right = rw + park_r + [Waypoint(max(total - t_right - 2.0, 0.5), tuple(float(v) for v in PARK["right"]), CAM_OPEN,
                                    "hold clear", _g(_grip((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))))]
    return {"left": left, "right": right}


# ---------------------------------------------------------------- offline test layouts (states.yaml)
def _quat(rot):
    from yam_arena.ik import matrix_to_quat_xyzw
    return [float(v) for v in matrix_to_quat_xyzw(np.asarray(rot, float))]


def flash_layout(pose):
    """[origin, quat] of the flash's part for a pose (4x4) of the flash's own frame."""
    return [float(v) for v in pose[:3, 3]], _quat(pose[:3, :3] @ START_ROT.T)


def flash_in_shoe(short_mm=0.0, lateral_mm=0.0):
    """The flash upright in the shoe (camera at its mark), its foot short_mm short of the stop."""
    pose = flash_at_shoe(-short_mm * MM)
    pose[:3, 3] += np.array([0.0, lateral_mm * MM, 0.0])
    return flash_layout(pose)


def flash_upright_on_table(x, y):
    """The flash standing upright on its foot on the table at (x, y) (world)."""
    return flash_layout(_T(np.eye(3), np.array([x, y, TOP + FLASH_LIFT])))
