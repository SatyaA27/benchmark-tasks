"""Put a CD back in its jewel case: press it onto the hub until the prongs grip, then shut the lid.

A Denon DCD-600NE CD player (434 x 273 x 107 mm, tray shut) stands on the table in front of the arms, its front
panel toward them. On its top lies an open standard CD jewel case (142 x 125 x 10.4 mm shut, clear polystyrene,
70 g): the base flat, its tray's long side 2 mm over the player's front edge, spine on the right; the lid stands
open at about 100 degrees on the spine. To the left, a CD (120 mm, 1.2 mm thick, 15 mm hole, 16 g) lies label up
on the player, part of its rim over the front edge. Metres; x forward from the arm bases, y left, z up.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import yam_mesh  # noqa: E402
from yam_joints import Hinge, Link, Slide  # noqa: E402
from yam_mesh import Mesh, MeshMechanism, MeshPart  # noqa: E402
from yam_parts import Part, box, cyl  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
DIAG = False   # GPU diagnostic copies print JC_DIAG / JC_READ lines
PROBE = False  # probe copy: the lid is left alone and scoring stops at the seated disc
DEMO_WAY = "press"  # "press": seat the disc, then shut the lid; "lay": wrong way, disc laid on the hub, lid shut
DEMO_PACE = 1.0
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
_MEMORY = {"episode": None, "ever": {}, "timers": {}}


def _episode_memory():
    if _MEMORY["episode"] != _RUN["episode"]:
        _MEMORY["episode"], _MEMORY["ever"], _MEMORY["timers"] = _RUN["episode"], {}, {}


def ever(key, cond):
    """Remember a satisfied condition until the next episode; offline each call stands alone."""
    if _RUN["t"] is None:
        return bool(cond)
    _episode_memory()
    _MEMORY["ever"][key] = bool(_MEMORY["ever"].get(key, False) or cond)
    return _MEMORY["ever"][key]


def held_for(key, cond):
    """Seconds a condition has held over consecutive control steps (repeat calls in one step are harmless); offline
    a true condition counts as held."""
    t, step = _RUN["t"], _RUN["step"]
    if t is None:
        return math.inf if cond else 0.0
    _episode_memory()
    previous, start = _MEMORY["timers"].get(key, (None, None))
    if not cond:
        _MEMORY["timers"][key] = (step, None)
        return 0.0
    if start is None or previous is None or step not in (previous, previous + 1) or t < start:
        start = t
    _MEMORY["timers"][key] = (step, start)
    return float(t - start)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# ================================================================ the player (Denon DCD-600NE, Denon NA info sheet:
# 434 x 273 x 107 mm, 4.0 kg). It stands still under these loads, so it is fixed in place in the simulation.
PLAYER_W, PLAYER_D, PLAYER_H = 0.434, 0.273, 0.107
XF = 0.235                     # the player's front face (x)
YP = 0.000                     # the player's centre (y)
PTOP = TOP + PLAYER_H
BLACK, SILVER, DARK = (0.06, 0.06, 0.07), (0.62, 0.63, 0.65), (0.02, 0.02, 0.025)


def _player_shapes():
    d, w, h = PLAYER_D, PLAYER_W, PLAYER_H
    return [
        box((d / 2, 0.0, h / 2), (d, w, h), BLACK),                                   # cabinet (top cover)
        box((0.0015, 0.0, h / 2), (0.003, w, h - 0.004), SILVER),                     # aluminium front panel
        box((-0.0003, 0.040, 0.075), (0.002, 0.150, 0.016), DARK),                    # disc tray door
        box((-0.0003, -0.090, 0.060), (0.002, 0.110, 0.030), DARK),                   # display window
        box((-0.0003, 0.185, 0.035), (0.002, 0.022, 0.010), DARK),                    # power button
    ]


PLAYER = Part("cd_player", (XF, YP, TOP), _player_shapes(), mass=None, color=BLACK, friction=(0.45, 0.40))

# ================================================================ the disc (IEC 60908): 120 mm, 1.2 mm, 15.00 mm hole
DISC_R, DISC_T, HOLE_R = 60.0 * MM, 1.2 * MM, 7.5 * MM
DISC_MASS = 0.016               # 15-16 g polycarbonate
DISC_SEGS = 48                  # collider: 48 convex annular sectors; the hole's flats at exactly 7.50 mm
DISC_FRICTION = (0.35, 0.30)    # polycarbonate on painted steel / polystyrene (estimate)
LABEL = [(0.86, 0.26, 0.18), (0.95, 0.80, 0.30)]   # printed label, two colours in wedges (seen from above)
DISC_OVERHANG = 0.027           # its rim reaches this far past the player's front edge
YD = 0.145                      # its centre (y)
DISC_START = np.array([XF - DISC_OVERHANG + DISC_R, YD, PTOP + DISC_T / 2])


def _disc_meshes():
    n = DISC_SEGS
    ri = HOLE_R / math.cos(math.pi / n)
    out = []
    for k in range(n):
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        pts = [(ri * math.cos(a0), ri * math.sin(a0)), (DISC_R * math.cos(a0), DISC_R * math.sin(a0)),
               (DISC_R * math.cos(a1), DISC_R * math.sin(a1)), (ri * math.cos(a1), ri * math.sin(a1))]
        proxy = (cyl((0.0, 0.0, 0.0), DISC_R, DISC_T, "z"),) if k == 0 else ()
        out.append(yam_mesh.prism(pts, DISC_T, name=f"s{k:02d}", collision="convex", color=LABEL[(k // 3) % 2],
                                  proxy=proxy))
    return out


DISC = MeshPart("cd", tuple(float(v) for v in DISC_START), (), mass=DISC_MASS, color=LABEL[0],
                friction=DISC_FRICTION, contact_offset=0.001, meshes=_disc_meshes())

# ================================================================ the jewel case. Case frame: origin on the base's
# underside below the hub's axis; x from the base's opening end to its spine (hinge), y along the spine, z up.
# The lid is authored shut (hinge 0) and opens about +y; 90 deg stands it straight up over the spine.
CASE_L, CASE_W, CASE_H = 142.0 * MM, 125.0 * MM, 10.4 * MM   # standard jewel case, shut
CASE_MASS, LID_MASS = 0.070, 0.024        # whole case 68-100 g (cdrom2go), tray-less 44-48 g (Walvis): 70 g; lid E
BACK_T, WALL_T = 1.5 * MM, 1.2 * MM       # E
BASE_WALL_H = 3.4 * MM                    # base's long walls; the shut lid's walls cover 3.4-8.9 mm E
SPINE_T, SPINE_H = 1.5 * MM, 8.9 * MM     # the tray's end walls (the tray card's spines show through them) E
DECK_Z = 4.0 * MM                         # floor of the disc well, 1 mm under a seated disc E
RIM_Z, WELL_R = 6.6 * MM, 60.5 * MM       # the well's rim (0.4 mm over a seated disc) and radius E
PED_R, PED_Z = 16.5 * MM, 5.0 * MM        # support ring under the disc's 33 mm clamping area; disc rests here E
NOTCH_HW, NOTCH_X0 = 14.0 * MM, 55.5 * MM  # the two thumb cut-outs at the disc's 3 and 9 o'clock E
HINGE_X, HINGE_Z = CASE_L / 2, 5.2 * MM   # lid pins at the spine end, mid-height (the lid lies flat at 180) E
LID_T, LID_WALL_Z0 = 1.5 * MM, BASE_WALL_H
BOOKLET_T = 0.6 * MM                      # a folded booklet held in the lid E
LID_OPEN_DEG = 100.0                      # standing open, leaning 10 deg back past upright
LID_FRICTION_NM = (0.012, 0.008)          # hinge pins' static / sliding friction torque E (holds the 24 g lid open)
# The catch (E). On a real case two small tabs at the lid's free corners ride over bumps on the tray and snap past
# them when the lid is pressed shut. Here the two tabs are modelled as ONE detent at the lid's hinge: a torque between
# lid and base set by the lid's angle alone (an over-centre spring). It is free above 3 deg. From 3 deg down to 1.6
# deg it pushes the lid open, harder and harder, up to the force that must be beaten to click the lid shut: 3 N at
# the free edge (E: a jewel case needs roughly 2-5 N). A lid let go anywhere there springs back to about 3 deg. Past
# the crest at 0.8 deg it pulls the lid shut and holds it on the base, with 2 N at the free edge (E: ~2 N to open).
CATCH_ARM = HINGE_X + CASE_L / 2           # hinge axis to the lid's free edge, 142 mm
CATCH_FREE_DEG, CATCH_PEAK_DEG, CATCH_DEG, CATCH_HOLD_DEG = 3.0, 1.6, 0.8, 0.4
CATCH_CLOSE_N, CATCH_OPEN_N = 3.0, 2.0     # at the free edge: to click it shut, to pull it open again (E)
CATCH_DAMPING = 0.02                       # N m s/rad, inside the catch's last 3 deg only (E)
CLEAR, BLUE_CLEAR, TRAY_BLACK = (0.42, 0.52, 0.62), (0.36, 0.46, 0.58), (0.07, 0.07, 0.08)  # clear PS, tinted for the camera
BOOKLET = (0.20, 0.32, 0.62)
CASE_FRICTION = (0.45, 0.40)              # polystyrene (estimate)

# The hub: 12 prongs ("eight to twelve upwardly projecting prongs"; a conventional 12-prong post is 15.25-15.35 mm
# across; below a crown the patent's own hub is 14.80-14.95 mm: US 2002/0112976 A1). The prongs flex inward as the
# hole passes the crown, then spring back over the disc. The kit's slides run along x or y only, so the 12 prongs move
# as four groups of three (+x, +y, -x, -y), each group on a sprung slide.
N_PRONGS = 12
CROWN_R, STEM_R, PRONG_IN_R = 15.30 * MM / 2, 14.85 * MM / 2, 5.6 * MM
LIP_LOW = PED_Z + DISC_T                       # 6.2 mm: the crown starts just over a seated disc
LIP_Z0, LIP_Z1 = LIP_LOW + 0.35 * MM, LIP_LOW + 0.55 * MM
SHOULDER = (7.1 * MM, LIP_LOW + 1.25 * MM)     # the crown's rounded top: 52 deg near the lip, 27 deg higher up E
CROWN_TOP_Z = LIP_LOW + 2.0 * MM               # 8.2 mm; the shut lid's booklet is 0.1 mm above it
PRONG_SPAN = math.radians(20.0)                # 20 deg prongs, 10 deg slits E
GROUP_TRAVEL = 0.3 * MM                        # a group flexes in at most 0.3 mm (twice the 0.15 mm the hole needs) E
PRONG_K = 1100.0       # N/m per prong E: polystyrene (E 3.2 GPa) cantilever 6.5 mm long, 3.1 mm wide, 0.5 mm thick
GROUP_K = PRONG_K * (1.0 + 2.0 * math.cos(math.radians(30.0)) ** 2)  # three prongs' radial stiffness on the axis
GROUP_DAMPING = 3.0                            # N s/m
GROUP_MASS = 0.002                             # each group's body (solver: real teeth weigh ~0.05 g; disclosed)
TOOTH_FRICTION = (0.35, 0.30)                  # moulded polystyrene teeth on the polycarbonate hole (estimate)
GROUPS = (("px", "x", 0.0), ("py", "y", 90.0), ("nx", "x", 180.0), ("ny", "y", 270.0))
SEATED_Z = PED_Z + DISC_T / 2
RESTING_Z = (LIP_Z1 + (CROWN_R - HOLE_R) / (CROWN_R - SHOULDER[0]) * (SHOULDER[1] - LIP_Z1)
             + DISC_T / 2)   # disc centre when it sits centred on the crown's slope, not pressed on

CASE_YAW = -math.pi / 2         # case x -> world -y (spine on the right), case y -> world +x
CASE_FRONT_OVER = 0.002         # the tray's long side overhangs the player's front edge by 2 mm
XC = XF + CASE_W / 2 - CASE_FRONT_OVER   # hub axis (world x)
YC = 0.000                               # hub axis (world y)
CASE_START = np.array([XC, YC, PTOP])
CASE_R0 = _rz(CASE_YAW)


def _loft(profile, a0, a1, k=3):
    """Closed mesh sweeping a convex r-z section from angle a0 to a1 (chords outside, so it stays convex)."""
    prof = np.asarray(profile, float)
    m = len(prof)
    pts = np.asarray([(r * math.cos(a), r * math.sin(a), z) for a in np.linspace(a0, a1, k) for r, z in prof])
    faces = []
    for j in range(k - 1):
        for i in range(m):
            i2 = (i + 1) % m
            a, b, c, d = j * m + i, j * m + i2, (j + 1) * m + i2, (j + 1) * m + i
            faces += [(a, b, c), (a, c, d)]
    o = (k - 1) * m
    for i in range(1, m - 1):
        faces += [(0, i + 1, i), (o, o + i, o + i + 1)]
    f = np.asarray(faces, int)
    if yam_mesh.signed_volume((pts, f)) < 0:
        f = f[:, ::-1]
    return pts, f


STEM = [(PRONG_IN_R, PED_Z + 0.05 * MM), (STEM_R, PED_Z + 0.05 * MM), (STEM_R, LIP_LOW), (PRONG_IN_R, LIP_LOW)]
CROWN = [(PRONG_IN_R, LIP_LOW), (STEM_R, LIP_LOW), (CROWN_R, LIP_Z0), (CROWN_R, LIP_Z1), SHOULDER,
         (PRONG_IN_R, CROWN_TOP_Z)]


def _group_meshes(gname, centre_deg):
    out = []
    for j, d in enumerate((-30.0, 0.0, 30.0)):
        c = math.radians(centre_deg + d)
        for part, prof in (("stem", STEM), ("crown", CROWN)):
            p, f = _loft(prof, c - PRONG_SPAN / 2, c + PRONG_SPAN / 2)
            proxy = (box((math.cos(c) * 6.6 * MM, math.sin(c) * 6.6 * MM, (PED_Z + CROWN_TOP_Z) / 2),
                         (1.6 * MM, 1.6 * MM, CROWN_TOP_Z - PED_Z)),) if part == "crown" and j == 1 else ()
            out.append(Mesh(f"{gname}_{part}{j}", p, f, collision="convex", color=CLEAR, proxy=proxy))
    return tuple(out)


def _case_base():
    L, W = CASE_L, CASE_W
    inner_y = W / 2 - WALL_T
    inner_x = L / 2 - SPINE_T
    shapes = [
        box((0.0, 0.0, BACK_T / 2), (L, W, BACK_T), BLUE_CLEAR),                                   # back plate
        box((0.0, W / 2 - WALL_T / 2, (BACK_T + BASE_WALL_H) / 2), (L, WALL_T, BASE_WALL_H - BACK_T), CLEAR),
        box((0.0, -W / 2 + WALL_T / 2, (BACK_T + BASE_WALL_H) / 2), (L, WALL_T, BASE_WALL_H - BACK_T), CLEAR),
        box((inner_x + SPINE_T / 2, 0.0, (BACK_T + SPINE_H) / 2), (SPINE_T, 2 * inner_y, SPINE_H - BACK_T),
            TRAY_BLACK),                                                                           # spine-end wall
        box((-inner_x - SPINE_T / 2, 0.0, (BACK_T + SPINE_H) / 2), (SPINE_T, 2 * inner_y, SPINE_H - BACK_T),
            TRAY_BLACK),                                                                           # opening-end wall
        box((0.0, 0.0, (BACK_T + DECK_Z) / 2), (2 * NOTCH_X0, 2 * inner_y, DECK_Z - BACK_T), TRAY_BLACK),  # deck
        cyl((0.0, 0.0, (DECK_Z + PED_Z) / 2), PED_R, PED_Z - DECK_Z, "z", TRAY_BLACK),             # support ring
    ]
    for sx in (1, -1):        # deck beside each thumb cut-out, and the rim at the ends
        for sy in (1, -1):
            xm = sx * (NOTCH_X0 + inner_x) / 2
            ym = sy * (NOTCH_HW + inner_y) / 2
            shapes.append(box((xm, ym, (BACK_T + DECK_Z) / 2), (inner_x - NOTCH_X0, inner_y - NOTCH_HW,
                                                                  DECK_Z - BACK_T), TRAY_BLACK))
    meshes = []
    for k, pts in enumerate(_rim_sectors(inner_x, inner_y)):   # the well's rim, outside the 121 mm well
        p = yam_mesh.prism(pts, RIM_Z - DECK_Z, name=f"rim{k:02d}", collision="convex", color=TRAY_BLACK, proxy=())
        meshes.append(Mesh(p.name, p.points + np.array([0.0, 0.0, (RIM_Z + DECK_Z) / 2]), p.faces,
                           collision="convex", color=TRAY_BLACK))
    return shapes, tuple(meshes)


def _ray_box(a, hx, hy):
    """Where the ray from the origin at angle a leaves the rectangle |x| <= hx, |y| <= hy."""
    c, s = math.cos(a), math.sin(a)
    t = min(hx / abs(c) if abs(c) > 1e-12 else math.inf, hy / abs(s) if abs(s) > 1e-12 else math.inf)
    return (t * c, t * s)


def _rim_sectors(hx, hy, step_deg=7.5):
    """Convex rim pieces: each the part of a narrow wedge that lies outside the well's circle and inside the tray's
    walls. The inner edge is a chord whose middle touches the circle (its ends lie outside it), so no piece reaches
    inside WELL_R. The thumb cut-outs (|y| <= NOTCH_HW at the ends) are left open."""
    a_notch = math.asin(NOTCH_HW / WELL_R)
    corners = [(hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)]
    out = []
    for lo, hi in ((a_notch, math.pi - a_notch), (math.pi + a_notch, 2 * math.pi - a_notch)):
        n = max(1, math.ceil(math.degrees(hi - lo) / step_deg))
        for i in range(n):
            a0, a1 = lo + (hi - lo) * i / n, lo + (hi - lo) * (i + 1) / n
            r_in = WELL_R / math.cos((a1 - a0) / 2)
            pts = [(r_in * math.cos(a0), r_in * math.sin(a0)), _ray_box(a0, hx, hy)]
            for cx, cy in corners:
                ac = math.atan2(cy, cx) % (2 * math.pi)
                if a0 < ac < a1:
                    pts.append((cx, cy))
            pts += [_ray_box(a1, hx, hy), (r_in * math.cos(a1), r_in * math.sin(a1))]
            out.append(pts)
    return out


def _lid():
    L, W = CASE_L, CASE_W
    plate_z = CASE_H - LID_T / 2
    wall_h = CASE_H - LID_T - LID_WALL_Z0
    shapes = [box((0.0, 0.0, plate_z), (L, W, LID_T), CLEAR)]
    for sy in (1, -1):        # long walls
        shapes.append(box((0.0, sy * (W / 2 - WALL_T / 2), LID_WALL_Z0 + wall_h / 2), (L, WALL_T, wall_h), CLEAR))
    shapes += [
        box((-5.0 * MM, 0.0, CASE_H - LID_T - BOOKLET_T / 2), (120.0 * MM, 120.0 * MM, BOOKLET_T), BOOKLET),
        cyl((HINGE_X, W / 2 - WALL_T / 2, HINGE_Z), 1.3 * MM, WALL_T, "y", CLEAR),      # hinge pin bosses, on the
        cyl((HINGE_X, -W / 2 + WALL_T / 2, HINGE_Z), 1.3 * MM, WALL_T, "y", CLEAR),     # lid's walls, clear of the base
    ]
    return shapes


def catch_torque(lid_deg, lid_vel=0.0):
    """The catch's torque on the lid (N m, opening positive; the base takes the opposite) at a lid angle (deg) and
    speed (rad/s): the two tabs riding over their bumps and snapping past, as one detent at the hinge."""
    t_close, t_hold = CATCH_CLOSE_N * CATCH_ARM, CATCH_OPEN_N * CATCH_ARM
    if lid_deg >= CATCH_FREE_DEG:
        return 0.0                                                     # clear of the catch: the lid swings freely
    if lid_deg >= CATCH_PEAK_DEG:                                      # riding up the bump: pushed back open
        tau = t_close * (CATCH_FREE_DEG - lid_deg) / (CATCH_FREE_DEG - CATCH_PEAK_DEG)
    elif lid_deg >= CATCH_DEG:                                         # over the top of the bump
        tau = t_close * (lid_deg - CATCH_DEG) / (CATCH_PEAK_DEG - CATCH_DEG)
    elif lid_deg >= CATCH_HOLD_DEG:                                    # past the crest: snapped in, pulled shut
        tau = -t_hold * (CATCH_DEG - lid_deg) / (CATCH_DEG - CATCH_HOLD_DEG)
    else:
        tau = -t_hold                                                  # caught: held shut on the base
    return tau - CATCH_DAMPING * lid_vel


def _case():
    base_shapes, rim = _case_base()
    links = [Link("lid", _lid(), Hinge("y", (HINGE_X, 0.0, HINGE_Z), 0.0, 181.0, damping=0.0002,
                                       initial=LID_OPEN_DEG, static_friction=LID_FRICTION_NM[0],
                                       dynamic_friction=LID_FRICTION_NM[1], armature=2e-6),
                  mass=LID_MASS, color=CLEAR)]
    meshes = {"base": rim}
    for gname, axis, deg in GROUPS:
        outward = 1.0 if deg in (0.0, 90.0) else -1.0          # +x/+y groups move inward along -axis
        lo, hi = (-GROUP_TRAVEL, 0.0) if outward > 0 else (0.0, GROUP_TRAVEL)
        links.append(Link(gname, (), Slide(axis, (0.0, 0.0, PED_Z), lo, hi, damping=GROUP_DAMPING, spring=GROUP_K,
                                           ), mass=GROUP_MASS, color=CLEAR, friction=TOOTH_FRICTION))
        meshes[gname] = _group_meshes(gname, deg)
    base_mass = CASE_MASS - LID_MASS - 4 * GROUP_MASS
    # Self-collision on, so the shut lid rests on the base. Every pair of links collides (measured), so no link's
    # shapes overlap another's: the tooth stems stand 0.05 mm over the support ring, and the lid's stop at 0 deg is
    # where its walls rest on the base's walls. The catch acts at the hinge (catch_torque, applied by _hook_catch).
    return MeshMechanism("jewel_case", tuple(float(v) for v in CASE_START), base_shapes, links, mass=base_mass,
                         color=CLEAR, yaw=CASE_YAW, friction=CASE_FRICTION, contact_offset=0.001, meshes=meshes,
                         self_collide=True)


CASE = _case()
OBJECTS = [PLAYER, CASE, DISC]
PRESSED_PARTS = ("jewel_case", "cd", "cd_player")  # the fingertips press the disc (on the case, on the player) down


# ================================================================ scoring
SEAT_XY_MM, SEAT_Z_MM, SEAT_TILT_DEG = 0.6, 0.35, 1.5     # seated: centred, down on the ring, flat
PRONGS_BACK_MM = 0.12                                      # every prong group sprung back over the disc
LID_DOWN_DEG = 10.0                                        # lid this near shut counts as closed over the disc
CAUGHT_HOLD_S = 1.0                                        # caught: at or below the catch's crest (CATCH_DEG) and
HANDS_CLEAR_MM = 30.0                                      # still there 1 s after both hands are 30 mm clear
CASE_MOVE_MM, CASE_TIP_DEG = 50.0, 10.0
IN_WELL_MM = 12.0


def pose(st):
    return np.asarray(st.origin, float), np.asarray(st.rotation(), float)


def disc_in_case(case, disc):
    """Disc centre (case frame, metres) and the angle between its axis and the case's."""
    o, r = pose(case)
    d_o, d_r = pose(disc)
    c = r.T @ (d_o - o)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(np.dot(r[:, 2], d_r[:, 2])))))))
    return c, tilt


def prong_offsets_mm(case):
    return {g: abs(float(case.joints[g])) * 1000.0 for g, _, _ in GROUPS}


def seated(case, disc):
    c, tilt = disc_in_case(case, disc)
    z_ok = abs(c[2] - (PED_Z + DISC_T / 2)) * 1000.0 <= SEAT_Z_MM
    xy_ok = math.hypot(c[0], c[1]) * 1000.0 <= SEAT_XY_MM
    back = max(prong_offsets_mm(case).values()) <= PRONGS_BACK_MM
    return bool(z_ok and xy_ok and tilt <= SEAT_TILT_DEG and back)


def hands_clear(case):
    """Both grippers' centre points at least HANDS_CLEAR_MM outside the case's bounding box (offline: assumed)."""
    hands = grippers()
    if hands is None:
        return True
    low, high = np.asarray(case.low, float), np.asarray(case.high, float)
    return all(float(np.linalg.norm(np.maximum(np.maximum(low - h[0], h[0] - high), 0.0))) * 1000.0 >= HANDS_CLEAR_MM
               for h in hands.values())


def score_terms(objects):
    case, disc = objects["jewel_case"], objects["cd"]
    _hook_catch(_RUN["env"])            # GPU only: the catch's torque at the hinge (installed once)
    o, r = pose(case)
    moved = float(np.linalg.norm(o[:2] - CASE_START[:2])) * 1000.0
    case_tilt = case.tilt_deg()
    c, tilt = disc_in_case(case, disc)
    lid = case.angle_deg("lid")
    t = {
        "seated": seated(case, disc), "disc_xy_mm": math.hypot(c[0], c[1]) * 1000.0,
        "disc_z_mm": c[2] * 1000.0, "disc_tilt": tilt, "lid_deg": lid, "case_moved_mm": moved,
        "case_tilt": case_tilt, "prongs_mm": prong_offsets_mm(case),
        "case_off": ever("case off the player or tipped", case_tilt > CASE_TIP_DEG or o[2] < PTOP - 0.02),
        "disc_off": ever("disc fell off the player", float(disc.origin[2]) < PTOP - 0.02),
    }
    t["caught"] = lid <= CATCH_DEG          # past the catch's crest: the catch holds it shut
    t["hands_clear"] = hands_clear(case)
    t["caught_s"] = held_for("lid caught, both hands clear", t["caught"] and t["hands_clear"])
    t["latched"] = t["caught_s"] >= CAUGHT_HOLD_S
    t["lid_down"] = lid <= LID_DOWN_DEG
    t["in_well"] = math.hypot(c[0], c[1]) * 1000.0 <= IN_WELL_MM and c[2] < 0.012
    if DIAG:
        _diag(objects, t)
    return t


def score(objects):
    t = score_terms(objects)
    if t["case_off"] or t["disc_off"]:
        return 0.0                      # case knocked off the player or tipped, or the disc dropped
    if PROBE:
        return 1.0 if t["seated"] else (0.2 if t["in_well"] else 0.0)
    if not t["seated"]:
        if t["lid_down"]:
            return 0.0                  # lid shut (or held down) over a disc that is not on the hub: it rides loose
        return 0.2 if t["in_well"] else 0.0
    if t["latched"]:
        s = 1.0                         # seated; lid caught, and still caught 1 s after both hands are clear
    elif t["lid_down"]:
        s = 0.8                         # seated, lid down but not caught (or only held down by a hand)
    else:
        s = 0.6                         # seated, lid left open
    return min(s, 0.5) if t["case_moved_mm"] > CASE_MOVE_MM else s


def success(objects):
    return score(objects) >= 1.0


# ================================================================ demo
# Right hand: level, fingers one above the other, pads 8 mm onto the disc's rim where it passes the player's front
# edge; lift, carry level over the hub, read the disc and correct, lower until the hole is 6.9 mm over the crown, read
# again, open to 5 mm (the hole drops onto the crown's slope and centres; the rim rests on the lower pad), slide back
# out from under the rim. Then straight down over the hub with the fingers open 24 mm, so the two fingertips press the
# disc's clamping ring on both sides of the hub at once, slowly, with a hold, until the teeth snap over the hole; again
# with the hand turned 90 deg. Then the right hand's fingertips push the standing lid over from the angle it is read
# at; the left hand presses its free end down until the catch clicks.
PAD_TIP = 0.013               # pad tips past the hand's centre point
OVERLAP = 0.008               # the pads reach this far onto the disc
TIP_CLEAR = 0.004             # the lower pad's tip stays this far short of the player's front face
PICK_HEADING = math.radians(10.0)   # the hand points 10 deg left of straight ahead when it takes the rim
PINCH_OPEN = -0.006           # pads 12 mm apart around the 1.2 mm disc
PINCH = 0.0                   # pads commanded shut on it: ~3 N squeeze (2.5 N per mm of part)
RELEASE_UNDER = 14.5 * MM     # disc's mid-plane over the case's underside when let go (lower finger clears the rim)
RELEASE_GAP = -0.0025         # pads opened to 5 mm: the hole drops onto the crown, the rim rests on the lower pad
SLIP_BACK = 0.018             # then the hand slides straight back so the lower pad leaves from under the rim
HOVER_UP = 0.025
PRESS_LEAD = 0.032            # sent this far below the resting disc's top: ~350 N/m down here, ~11 N at full lead
CORR_MAX, CORR_MAX_DEG = 0.015, 6.0
FINAL_HOLD_S = 3.0           # both hands parked: the caught lid's 1 s, then the 1 s success hold
_READS = {"episode": None, "values": {}}


def _down():
    from yam_arena.demo import grip_rotation
    return np.asarray(getattr(grip_rotation, "__wrapped__", grip_rotation)(0.0), float)


def _level(heading=0.0):
    """Hand pointing along world heading (rad), level, fingers closing along +z (one pad above the other)."""
    d = _down()
    for base in (_ry(math.pi / 2) @ d, _ry(-math.pi / 2) @ d):
        for k in range(4):
            r = base @ _rz(k * math.pi / 2)
            if np.allclose(r[:, 2], [1, 0, 0], atol=1e-6) and np.allclose(r[:, 0], [0, 0, 1], atol=1e-6):
                return _rz(heading) @ r
    raise RuntimeError("no level rotation")


def _objects_now(names):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        return arena_scene.object_states(env, list(names))
    except Exception as e:  # noqa: BLE001
        print(f"JC_READ failed: {e!r}", flush=True)
        return None


def _values():
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    return _READS["values"]


def read_case():
    v = _values()
    if "case" not in v:
        st = _objects_now(["jewel_case"])
        if st is None:
            v["case"] = (CASE_START.copy(), CASE_R0.copy())
        else:
            o, r = pose(st["jewel_case"])
            v["case"] = (o, r)
            print(f"JC_READ case t={_RUN['t']:.2f} off_mm={tuple(round(float(x) * 1000, 1) for x in o - CASE_START)} "
                  f"lid={st['jewel_case'].angle_deg('lid'):.1f}", flush=True)
    return v["case"]


def read_disc_start():
    v = _values()
    if "disc0" not in v:
        st = _objects_now(["cd"])
        if st is None:
            v["disc0"] = (DISC_START.copy(), np.eye(3))
        else:
            o, r = pose(st["cd"])
            v["disc0"] = (o, r)
            print(f"JC_READ disc0 t={_RUN['t']:.2f} off_mm={tuple(round(float(x) * 1000, 1) for x in o - DISC_START)} "
                  f"tilt={st['cd'].tilt_deg():.2f}", flush=True)
    return v["disc0"]


def hub_point(z_case):
    o, r = read_case()
    return o + r @ np.array([0.0, 0.0, z_case])


def pick_geometry():
    """(hand rotation, disc centre in the hand frame) for the rim pinch; the overlap shrinks if the disc's overhang
    is short, so the lower pad never reaches the player's front face."""
    c, _ = read_disc_start()
    h = PICK_HEADING
    over = min(OVERLAP, DISC_R - (c[0] - XF + TIP_CLEAR) / math.cos(h))   # pad tips stay short of the player
    over = max(over, 0.008)
    rot = _level(h)
    local = np.array([0.0, 0.0, PAD_TIP + DISC_R - over])   # disc centre along the approach axis from the TCP
    return rot, local, over


def correction(stage):
    """(R, t) for the carried disc: rotates and shifts the hand's targets so the disc's centre lands on the hub's
    axis, level, chained over the stages ("a" high, "b" low)."""
    key = f"corr{stage}"
    v = _values()
    if key in v:
        return v[key]
    prior = correction("a") if stage == "b" else (np.eye(3), np.zeros(3))
    want = hub_point(HOVER_UP + RELEASE_UNDER if stage == "a" else RELEASE_UNDER)
    st = _objects_now(["cd", "jewel_case"])
    if st is None:
        v[key] = prior
        return prior
    d_o, d_r = pose(st["cd"])
    _, c_r = pose(st["jewel_case"])
    if float(np.linalg.norm(d_o - want)) > 0.040:
        print(f"JC_READ {key} t={_RUN['t']:.2f} disc not where planned ({np.round((d_o - want) * 1000, 1)} mm): "
              f"no correction", flush=True)
        v[key] = prior
        return prior
    n_act, n_plan = d_r[:, 2], c_r[:, 2]
    if float(np.dot(n_act, n_plan)) < 0:
        n_act = -n_act
    axis = np.cross(n_act, n_plan)
    s = float(np.linalg.norm(axis))
    a = math.atan2(s, float(np.dot(n_act, n_plan)))
    dR = _aa(axis, min(a, math.radians(CORR_MAX_DEG))) if s > 1e-9 else np.eye(3)
    move = want - d_o
    n = float(np.linalg.norm(move))
    if n > CORR_MAX:
        move *= CORR_MAX / n
    Rp, tp = prior
    v[key] = (dR @ Rp, dR @ (tp - d_o) + d_o + move)
    print(f"JC_READ {key} t={_RUN['t']:.2f} disc_err_mm={tuple(round(float(x) * 1000, 2) for x in d_o - want)} "
          f"tilt_err_deg={math.degrees(a):.2f}", flush=True)
    return v[key]


def _aa(axis, a):
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(a) * k + (1.0 - math.cos(a)) * (k @ k)


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


def disc_target(kind):
    """(hand centre point, hand rotation) for the right hand at one stage of the disc's plan."""
    rot, local, over = pick_geometry()
    if kind in ("pre", "in", "pick", "lift"):
        c, _ = read_disc_start()
        tcp = c - rot @ local
        if kind == "pre":
            return tcp - 0.020 * rot[:, 2], rot
        if kind == "lift":
            return tcp + np.array([0.0, 0.0, 0.040]), rot
        return tcp, rot
    if kind in ("carry", "high", "near", "near2", "drop", "slip", "up"):
        rot, local = _level(0.0), np.array([0.0, 0.0, PAD_TIP + DISC_R - over])
        stage = {"carry": None, "high": None, "near": "a", "near2": "b", "drop": "b", "slip": "b", "up": "b"}[kind]
        z = RELEASE_UNDER + (HOVER_UP if kind in ("carry", "high") else 0.0)
        want = hub_point(z)
        R, t = correction(stage) if stage else (np.eye(3), np.zeros(3))
        disc_c = R @ want + t
        hand_r = R @ rot
        tcp = disc_c - hand_r @ local
        if kind == "carry":
            tcp = tcp + np.array([0.0, 0.0, 0.020])
        if kind == "slip":
            tcp = tcp - SLIP_BACK * hand_r[:, 2]
        if kind == "up":
            tcp = tcp - SLIP_BACK * hand_r[:, 2] + np.array([0.0, 0.0, 0.050])
        return tcp, hand_r
    raise ValueError(kind)


PRESS_GAP = 0.024             # fingers opened 24 mm, straddling the hub: each 4 x 15.4 mm fingertip lands on the
PRESS_OPEN = -PRESS_GAP / 2   # disc's clamping ring 12-16 mm from the axis (crown 7.65 mm), one on each side
PRESS_TURNS = (0, 1)          # first with the fingers across world x, then turned 90 deg (across world y)


def press_target(kind):
    """Hand straight down over the hub's axis, fingers open 24 mm so the two fingertips press the disc's clamping ring
    on opposite sides of the hub at once (an even push, no tilt). kind: stage + turn, e.g. "over0", "press1";
    turn 0 has the fingers closing along world x, turn 1 along world y."""
    stage, turn = kind[:-1], int(kind[-1])
    rot = (_rz(math.pi / 2) @ _down()) if turn == 0 else _down()
    top_rest = RESTING_Z + DISC_T / 2          # a disc resting centred on the crown: its top this high
    tip = {"over": top_rest + 0.020, "touch": top_rest + 0.003, "press": top_rest - PRESS_LEAD,
           "hold": top_rest - PRESS_LEAD, "lift": top_rest + 0.020}[stage]
    return hub_point(tip) + np.array([0.0, 0.0, PAD_TIP]), rot


LID_PUSH_D = 0.090            # the right hand's fingertips push the lid's outer face this far from the hinge
LID_PUSH_LEAD = 0.003         # ... sent this far into the face, so they keep pushing
LID_HAND_MAX_DEG = 85.0       # past this the hand would have to point upward: it keeps the 85 deg pointing
LID_SWEEP_N = 6               # the push swings the lid from where it stands (read) to LID_SWEEP_END in 6 steps
LID_SWEEP_END = 12.0
LID_READ_RANGE = (90.0, 115.0)   # past ~115 deg the right arm cannot reach round behind the lid
LID_PRESS_X = -0.062          # the left hand presses the lid's free end here (case x), over the corner posts' line
LID_PRESS_LEAD = 0.020


def read_lid():
    """The lid's angle (deg) when the lid push starts: it can creep open a few degrees while the disc is pressed."""
    v = _values()
    if "lid" not in v:
        st = _objects_now(["jewel_case"])
        a = LID_OPEN_DEG if st is None else float(st["jewel_case"].angle_deg("lid"))
        v["lid"] = min(max(a, LID_READ_RANGE[0]), LID_READ_RANGE[1])
        if st is not None:
            print(f"JC_READ lid t={_RUN['t']:.2f} lid={a:.1f} sweep_from={v['lid']:.1f}", flush=True)
    return v["lid"]


def lid_sweep(i):
    th0 = read_lid()
    return th0 - (th0 - LID_SWEEP_END) * i / LID_SWEEP_N


def lid_frame(theta_deg, d):
    """(contact point on the lid's outer face, outward normal) in the world with the lid at theta (0 shut)."""
    o, r = read_case()
    th = math.radians(theta_deg)
    ry = _ry(th)
    hinge = np.array([HINGE_X, 0.0, HINGE_Z])
    pc = hinge + ry @ np.array([-d, 0.0, CASE_H - HINGE_Z])
    nc = ry @ np.array([0.0, 0.0, 1.0])
    return o + r @ pc, r @ nc


def lid_push_target(kind):
    """Right hand, closed, pointing into the lid's outer face; kind 'pre<i>', 'push<i>' (step i of the sweep from the
    lid's read angle down to 12 deg) or 'away'."""
    _, r = read_case()
    if kind == "away":
        p, n = lid_frame(LID_SWEEP_END, LID_PUSH_D)
        return p + np.array([0.0, -0.060, 0.060]), _push_rot(n, r)
    th = lid_sweep(int(kind[4:]) if kind.startswith("push") else int(kind[3:]))
    p, n = lid_frame(th, LID_PUSH_D)
    _, n_hand = lid_frame(min(th, LID_HAND_MAX_DEG), LID_PUSH_D)
    rot = _push_rot(n_hand, r)
    a = rot[:, 2]
    tips = p - LID_PUSH_LEAD * n if kind.startswith("push") else p + 0.015 * n
    return tips - PAD_TIP * a, rot


def _push_rot(n, r):
    a = -n / np.linalg.norm(n)
    x = r[:, 1] - np.dot(r[:, 1], a) * a
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(a, x), a])


def lid_press_target(kind):
    """Left hand straight down on the lid's free end (over the catch), closed."""
    o, r = read_case()
    top = o + r @ np.array([LID_PRESS_X, 0.0, CASE_H])
    z = {"over": 0.050, "touch": 0.004, "press": -LID_PRESS_LEAD, "lift": 0.050}[kind]
    return top + np.array([0.0, 0.0, z + PAD_TIP]), _down()


class HandWaypoint(Waypoint):
    """A waypoint worked out when it is played (reads are taken the first time a waypoint needs them)."""

    def __init__(self, duration, fn, kind, finger, label, keep=False):
        self.duration = duration * (1.0 if keep else DEMO_PACE)
        self.fn, self.kind, self.finger, self.label = fn, kind, finger, label

    @property
    def pos(self):
        return tuple(float(x) for x in self.fn(self.kind)[0])

    @property
    def yaw(self):
        return Grip(0.0, self.fn(self.kind)[1])


FRONT_X = 0.160               # the right hand changes sides in front of the case (its front edge is at x 0.233)


def route_target(kind):
    """Via points that keep the right arm clear of the lid standing open on the case's right (spine) side."""
    if kind == "in":                         # from the start to the disc: in front of the case, already level
        rot, _, _ = pick_geometry()
        return np.array([0.190, -0.150, TOP + 0.185]), rot
    if kind == "out1":                       # from over the hub, still pointing down, out in front of the case
        return np.array([FRONT_X, 0.000, TOP + 0.200]), press_target(f"lift{PRESS_TURNS[-1]}")[1]
    if kind == "out2":                       # in front of the lid's outer face, turned to push it; then it slides in
        p, r = lid_push_target("pre0")
        return np.array([FRONT_X + 0.010, p[1] - 0.006, p[2]]), r
    if kind == "out3":                       # probe copies: clear of the lid on the way to the park
        return np.array([FRONT_X, -0.250, TOP + 0.200]), press_target(f"lift{PRESS_TURNS[-1]}")[1]
    raise ValueError(kind)


RIGHT_PARK = np.array([0.20, -0.30, TOP + 0.20])
LEFT_PARK = np.array([0.20, 0.30, TOP + 0.20])


def right_plan():
    O, P = PINCH_OPEN, PINCH
    D, T = disc_target, press_target
    plan = [
        HandWaypoint(1.5, route_target, "in", O, "round the case's front, clear of the standing lid"),
        HandWaypoint(1.5, D, "pre", O, "disc: in front of its rim, level"),
        HandWaypoint(1.2, D, "in", O, "disc: pads over and under its rim"),
        HandWaypoint(0.8, D, "pick", P, "disc: pinch the rim", keep=True),
        HandWaypoint(0.3, D, "pick", P, "disc: settle"),
        HandWaypoint(1.0, D, "lift", P, "disc: lift"),
        HandWaypoint(2.0, D, "carry", P, "disc: carry level over the case"),
        HandWaypoint(1.0, D, "high", P, "disc: over the hub"),
        HandWaypoint(0.5, D, "high", P, "disc: hold"),
        HandWaypoint(1.0, D, "near", P, "disc: read it, lower to 6 mm over the crown"),
        HandWaypoint(0.5, D, "near", P, "disc: hold"),
        HandWaypoint(0.5, D, "near2", P, "disc: read again, correct"),
        HandWaypoint(0.4, D, "near2", P, "disc: hold"),
        HandWaypoint(0.3, D, "drop", RELEASE_GAP, "disc: let go, the hole drops onto the crown", keep=True),
        HandWaypoint(0.5, D, "drop", RELEASE_GAP, "disc: settle"),
        HandWaypoint(0.8, D, "slip", RELEASE_GAP, "slide the pads out from its rim"),
        HandWaypoint(0.5, D, "slip", RELEASE_GAP, "disc: settles flat on the crown"),
        HandWaypoint(1.0, D, "up", RELEASE_GAP, "up"),
    ]
    if DEMO_WAY == "press":
        O2 = PRESS_OPEN
        plan += [
            HandWaypoint(1.5, T, f"lift{PRESS_TURNS[0]}", RELEASE_GAP, "press: turn to point straight down over the hub"),
        ]
        for k, turn in enumerate(PRESS_TURNS):
            tag = "" if k == 0 else " (turned 90 deg)"
            plan += [
                HandWaypoint(1.0, T, f"over{turn}", O2, f"press{tag}: fingertips either side of the hub"),
                HandWaypoint(1.0, T, f"touch{turn}", O2, f"press{tag}: fingertips just over the disc"),
                HandWaypoint(2.5, T, f"press{turn}", O2, f"press{tag}: push both sides down until the teeth snap"),
                HandWaypoint(1.5, T, f"hold{turn}", O2, f"press{tag}: hold"),
                HandWaypoint(1.0, T, f"lift{turn}", O2, f"press{tag}: lift"),
            ]
    if PROBE:
        plan += [HandWaypoint(1.2, route_target, "out1", PINCH, "out in front of the case"),
                 HandWaypoint(1.2, route_target, "out3", PINCH, "out to the right, clear of the lid")]
    else:
        Lp = lid_push_target
        plan += [HandWaypoint(1.2, route_target, "out1", PINCH, "out in front of the case"),
                 HandWaypoint(1.5, route_target, "out2", PINCH, "round to the lid's outer side"),
                 HandWaypoint(1.5, Lp, "pre0", PINCH, "lid: fingertips by its outer face"),
                 HandWaypoint(0.8, Lp, "push0", PINCH, "lid: touch it")]
        plan += [HandWaypoint(0.9, Lp, f"push{i}", PINCH, f"lid: push it over (step {i} of {LID_SWEEP_N})")
                 for i in range(1, LID_SWEEP_N + 1)]
        plan += [HandWaypoint(1.2, Lp, "away", PINCH_OPEN, "lid: hand away, the lid settles on its catch")]
    plan.append(Waypoint(1.5 * DEMO_PACE, tuple(float(x) for x in RIGHT_PARK), PINCH_OPEN, "park", 0.0))
    return plan


def left_plan(wait_s):
    Lt = lid_press_target
    park = tuple(float(x) for x in LEFT_PARK)
    return [
        Waypoint(2.0, park, PINCH_OPEN, "left: out of the way", 0.0),
        Waypoint(max(wait_s - 2.0, 0.5), park, PINCH_OPEN, "left: wait", 0.0),
        HandWaypoint(1.5, Lt, "over", PINCH, "lid: over its free end"),
        HandWaypoint(1.0, Lt, "touch", PINCH, "lid: fingertips on it"),
        HandWaypoint(1.0, Lt, "press", PINCH, "lid: press until the catch clicks"),
        HandWaypoint(0.5, Lt, "press", PINCH, "lid: hold"),
        HandWaypoint(1.0, Lt, "lift", PINCH, "lid: lift"),
        Waypoint(1.5 * DEMO_PACE, park, PINCH_OPEN, "park", 0.0),
    ]


def demo():
    right = right_plan()
    if PROBE:
        right.append(Waypoint(FINAL_HOLD_S, tuple(float(x) for x in RIGHT_PARK), PINCH_OPEN, "hold clear", 0.0))
        return {"right": right}
    t_lid = sum(w.duration for w in right if not w.label.startswith("park"))
    plans = {"right": right, "left": left_plan(t_lid)}
    end = max(sum(w.duration for w in p) for p in plans.values()) + FINAL_HOLD_S   # both parked, then hold
    for side, park in (("right", RIGHT_PARK), ("left", LEFT_PARK)):
        plans[side].append(Waypoint(end - sum(w.duration for w in plans[side]), tuple(float(x) for x in park),
                                    PINCH_OPEN, "hold clear", 0.0))
    return plans


# ================================================================ offline test layouts (states.yaml)
def case_point(x, y, z, dx=0.0, dy=0.0):
    """World point from case coordinates (case moved by dx, dy in the world)."""
    return [float(v) for v in CASE_START + np.array([dx, dy, 0.0]) + CASE_R0 @ np.array([x, y, z])]




# ================================================================ on the GPU: the catch, the hands, diagnostics
_DIAG = {"step": -1}
_CATCH = {"err": None}


def _hook_catch(env):
    """The catch's torque (catch_torque) between lid and base about the hinge axis: before every physics substep
    (Isaac Lab calls write_data_to_sim then) +tau on the lid and -tau on the base, an internal couple like the tabs'
    spring force between them. The kit's articulations have no actuators, so it is applied as body torques through
    the articulation's permanent wrench composer. Installed once per articulation; nothing offline."""
    if env is None:
        return
    try:
        art = env.unwrapped.scene.articulations["jewel_case"]
    except Exception as e:  # noqa: BLE001
        if _CATCH["err"] is None:
            _CATCH["err"] = repr(e)
            print(f"JC_CATCH no articulation: {e!r}", flush=True)
        return
    if getattr(art, "_jc_catch_hooked", False):
        return
    import torch
    from yam_arena.urdf import quat_xyzw_to_matrix
    j_lid = list(art.joint_names).index("lid_joint")
    b_lid, b_base = list(art.body_names).index("lid"), list(art.body_names).index("base")
    poses = _arr(art.data.body_link_pose_w)
    r_rel = quat_xyzw_to_matrix(poses[b_base, 3:7]).T @ quat_xyzw_to_matrix(poses[b_lid, 3:7])
    q0 = float(_arr(art.data.joint_pos)[j_lid])
    sign = 1.0 if np.linalg.norm(r_rel - _ry(q0)) <= np.linalg.norm(r_rel - _ry(-q0)) else -1.0
    print(f"JC_CATCH hooked lid={math.degrees(q0):.2f} deg, hinge axis sign {sign:+.0f}", flush=True)
    original = art.write_data_to_sim

    def write_data(*args, **kwargs):
        try:
            q, w = float(_arr(art.data.joint_pos)[j_lid]), float(_arr(art.data.joint_vel)[j_lid])
            tau = catch_torque(math.degrees(q), w)
            axis = sign * quat_xyzw_to_matrix(_arr(art.data.body_link_pose_w)[b_base, 3:7])[:, 1]
            torques = torch.tensor([[list(tau * axis), list(-tau * axis)]], dtype=torch.float32, device=art.device)
            composer = getattr(art, "permanent_wrench_composer", None)
            if composer is not None:
                composer.set_forces_and_torques_index(forces=torch.zeros_like(torques), torques=torques,
                                                      body_ids=[b_lid, b_base], is_global=True)
            else:
                import warp as wp
                n = len(art.body_names)
                tq = torch.zeros((1, n, 3), dtype=torch.float32, device=art.device)
                tq[0, b_lid], tq[0, b_base] = torques[0, 0], torques[0, 1]
                art.root_physx_view.apply_forces_and_torques_at_position(
                    force_data=wp.from_torch(torch.zeros_like(tq)), torque_data=wp.from_torch(tq),
                    position_data=None, indices=getattr(art, "_ALL_INDICES", torch.arange(1, device=art.device)),
                    is_global=True)
            _CATCH["tau"] = tau
        except Exception as e:  # noqa: BLE001
            if _CATCH["err"] is None:
                _CATCH["err"] = repr(e)
                print(f"JC_CATCH error: {e!r}", flush=True)
        return original(*args, **kwargs)

    art.write_data_to_sim = write_data
    art._jc_catch_hooked = True


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
        push = _push(art, arm_ids, q, station.BASES[arm]) if DIAG else None
        out[arm] = (tcp, float(np.mean(q[finger_ids])), push)
    return out


def _arr(x):
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, float)
    return x[0] if x.ndim > 1 else x


def _push(art, arm_ids, q, base):
    """Force (N, world) the hand's fingertips push with: the arm's drive torques K (q_target - q) - D qdot (gravity is
    fed forward separately) mapped through the fingertip Jacobian; ~0 when the hand moves freely."""
    try:
        from yam_arena import ik
        d = art.data
        qa = q[arm_ids]
        tau = (_arr(d.joint_stiffness)[arm_ids] * (_arr(d.joint_pos_target)[arm_ids] - qa)
               - _arr(d.joint_damping)[arm_ids] * _arr(d.joint_vel)[arm_ids])

        def tip(qq):
            p, r = ik.tcp_pose(qq, base)
            return p + PAD_TIP * r[:, 2]
        t0 = tip(qa)
        jac = np.zeros((3, len(qa)))
        for i in range(len(qa)):
            dq = qa.copy()
            dq[i] += 1e-6
            jac[:, i] = (tip(dq) - t0) / 1e-6
        return np.linalg.lstsq(jac.T, tau, rcond=None)[0]
    except Exception as e:  # noqa: BLE001
        if not _DIAG.get("push_err"):
            _DIAG["push_err"] = True
            print(f"JC_DIAG push unavailable: {e!r}", flush=True)
        return None


def _diag(objects, t):
    step = _RUN["step"]
    if step == _DIAG["step"] or step % 4:
        return
    _DIAG["step"] = step
    disc = objects["cd"]
    pr = " ".join(f"{g}={v:.3f}" for g, v in t["prongs_mm"].items())
    hands = grippers() or {}
    h = " ".join(f"{a[0]}=({','.join(f'{x * 1000:.1f}' for x in hands[a][0])};{hands[a][1] * 1000:.1f})"
                 + ("" if len(hands[a]) < 3 or hands[a][2] is None
                    else f"push=({','.join(f'{x:.1f}' for x in hands[a][2])})")
                 for a in hands)
    print(f"JC_DIAG t={_RUN['t'] or 0.0:.2f} disc_case=({t['disc_xy_mm']:.2f}xy,{t['disc_z_mm']:.2f}z;"
          f"tilt {t['disc_tilt']:.2f}) disc_w=({','.join(f'{x * 1000:.1f}' for x in disc.origin)};"
          f"v {np.linalg.norm(disc.lin_vel) * 1000:.0f}) seated={int(t['seated'])} prongs[{pr}] "
          f"lid={t['lid_deg']:.2f} catch_tau={_CATCH.get('tau', float('nan')):+.3f} clear={int(t['hands_clear'])} "
          f"caught_s={min(t['caught_s'], 99.0):.2f} case_mv={t['case_moved_mm']:.1f} "
          f"case_tilt={t['case_tilt']:.2f} {h}", flush=True)
