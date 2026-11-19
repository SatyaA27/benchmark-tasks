"""A countertop coin gumball machine (VEVOR 16-inch "ak201-01" class: 25-cent rotary coin mechanism, one gumball
wheel, chrome door over the exit), built from the kit's parts. One self-contained module: copy it into a sibling task.

Machine frame M: origin at the middle of the footprint on the table, axes world aligned, the coin mechanism facing -x
(toward the robot). Millimetres in the comments, metres in the code.

Published (maker listing): 215 x 200 x 410 mm, 2.95 kg, 25-cent coin, "rotate the control lever clockwise 360 degrees",
"the candy exit is fitted with a metal door guard", 1 in (25.4 mm) round candies, one gumball wheel.
Estimated from the maker's photos scaled to the 410 mm height (check with a ruler): coin mouth on top of the mech plate
182 mm up, knob axis 141 mm up, T-bar 41 x 13 mm standing 15 mm proud, plate 90 x 82 mm, door 46 x 47 mm.

Parts (machine(), a MeshMechanism with self-collision, fixed to the table in the simulation):
  - dial (hinge about x through the knob axis; joint -180 deg is home (authored turned half a turn), the coin slot
    straight up; it turns clockwise as seen from the front, which is +x). It carries the T-bar, the coin dial (a 54 mm disc in three layers: the middle
    layer has a 26.2 mm wide channel open at the top, in which a quarter dropped edge-down stands on a floor so that
    6 mm of it sticks out above the rim; the front and back layers have a 10 mm wide, 4 mm deep notch over the
    channel, with 1.5 mm lead-ins), an 8-tooth ratchet and the gumball wheel (64 mm, one 27 mm pocket, between two
    flanges) on a 10 mm shaft. Dynamic friction 0.12 N m, static 0.15 N m (the turning torque estimate).
  - gate (the coin pin): a sprung plunger whose flat end presses on the dial's rim at the side (90 degrees round).
    With the slot empty the pin drops into the notch after about 75 degrees and the notch's wall bears on its side,
    held by its guide: the handle is blocked. A seated quarter fills the notch and pushes the pin out over it.
  - pawl (anti-reverse): a sprung arm on the 8-tooth ratchet; the handle cannot turn back more than ~2 degrees.
  - door: the chrome exit door, hinged along its top edge (y axis), closed by its own weight (20 g), opens outward.
The pocket starts 90 degrees before the top (holding one gumball), passes under the feed tube at 90 degrees (full,
so no second ball enters) and lets its ball fall into the chute at 245-270 degrees: one ball per full turn, in its
last third. The quarter leaves the coin channel past the bottom (about 180 degrees), lands on a slope that
tips it back into the coin box inside the base. A second gumball waits in the feed tube.

Simulation stand-ins (disclosed): the globe's gumballs are fixed scenery; the machine's horizontal gumball turntable
(geared to the knob in the real machine) is a one-pocket wheel on the knob's own shaft fed by a short tube; the coin
pin, ratchet and coin path follow the common rotary-mech arrangement, not a drawing of this model.
"""
import dataclasses
import math

import numpy as np

from yam_joints import Hinge, Link, Slide
import yam_mesh
from yam_mesh import Mesh, MeshMechanism, MeshPart
from yam_parts import Part, ball, box, cyl

MM = 0.001
TAU = 2 * math.pi

RED, CHROME, DARK, BRASS = (0.74, 0.07, 0.07), (0.80, 0.81, 0.84), (0.16, 0.16, 0.17), (0.72, 0.62, 0.30)
GLASS = (0.86, 0.90, 0.93)
GUM_COLORS = ((0.93, 0.25, 0.35), (0.98, 0.80, 0.15), (0.20, 0.60, 0.95), (0.25, 0.75, 0.30), (0.98, 0.55, 0.15),
              (0.95, 0.95, 0.95), (0.62, 0.30, 0.80))
MACHINE_MASS = None  # fixed to the table in the simulation (real: 2.95 kg published + the globe's gumballs)

# ---- dimensions (mm)
FOOT_H = 25.0
BODY_X, BODY_Y = 85.0, 90.0
BODY_TOP = 185.0
BAND_TOP = 205.0
AX_Z = 141.0  # knob axis height
PLATE_FRONT, PLATE_BACK = -97.0, -85.0
PLATE_Y, PLATE_Z0, PLATE_Z1 = 45.0, 100.0, 182.0
CAV_X0, CAV_X1 = -93.0, -87.0  # the cavity behind the plate's front layer (coin sweep, gate)
DIAL_R = 27.0
HOLE_R = 27.5
SWEEP_R = 36.0
F_X = (-97.0, -91.4)  # dial front layer
S_X = (-91.4, -88.6)  # coin channel (2.8 mm)
B_X = (-88.6, -85.0)  # back layer
SLIT_X = (S_X[0] + S_X[1]) / 2  # -90.0: the coin's mid-plane
CHAN_W = 13.1  # half width of the coin channel
CHAN_FLOOR = 8.74  # channel floor above the axis: the seated coin's top is 6 mm above the rim
NOTCH_W, NOTCH_FLOOR, LEAD = 5.0, 23.0, 1.5
MOUTH_W = 14.0  # half width of the mouth on top of the plate
BAR = (-112.0, -97.0, 20.5, 6.5)  # T-bar: x from, x to, half width, half height
SHAFT_R, SHAFT_X = 5.0, (-85.0, -38.0)
RATCHET_X, RATCHET_TEETH, RATCHET_ROOT, RATCHET_TIP = (-82.0, -76.0), 8, 19.0, 25.0
WHEEL_R, WHEEL_X, FLANGE_T = 32.0, (-72.0, -38.0), 3.0
POCKET_W, POCKET_FLOOR, POCKET_HOME = 13.5, 5.0, -90.0  # half width, floor radius, home angle (deg from the top)
RING_R0, RING_R1, RING_OPEN = 33.5, 37.0, 25.0
TUBE_X, TUBE_Y, TUBE_Z = (-69.0, -41.0), 13.5, (175.0, 204.0)
PIN_LEN, PIN_HALF_Z, PIN_OUT, PIN_IN = 8.2, 1.5, 8.0, 4.5  # the coin pin: length, half height, travel out / in (mm)
PAWL_ANGLE, PAWL_ARM = 90.0, 15.0
PAWL_LIFT = 5.7  # the pawl's arm runs outside the teeth's tips (r 25); its tip is a finger down to the root
DOOR_HINGE = (-87.25, 89.0)
DOOR_X, DOOR_Y, DOOR_Z0 = (-88.0, -86.5), 23.0, 42.0
EXIT_W = 15.0  # inner half width of the chute and exit
FLOOR_Z_FRONT, FLOOR_Z_BACK, FLOOR_X = 40.3, 52.0, (-86.0, -38.0)

COIN_R, COIN_T, COIN_MASS = 24.26 / 2, 1.75, 0.00567  # US Mint quarter (mm, mm, kg)
COIN_SIDES = 32
GUM_R, GUM_MASS = 25.4 / 2, 0.009  # 1 in gumball; 850 per 17 lb case

# ---- 2-D helpers in the (y, z) plane about the knob axis (z relative to the axis)


def ang_pt(alpha_deg, r):
    """A point at angle alpha (degrees from straight up, positive toward -y: the dial's forward turn), radius r."""
    a = math.radians(alpha_deg)
    return np.array([-r * math.sin(a), r * math.cos(a)])


def rot2(poly, alpha_deg):
    """Polygon (y, z) turned by alpha about +x (the dial's forward turn)."""
    a = math.radians(alpha_deg)
    c, s = math.cos(a), math.sin(a)
    p = np.asarray(poly, float)
    return np.column_stack((c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]))


def disc(r, n=48):
    a = np.arange(n) * TAU / n
    return np.column_stack((r * np.cos(a), r * np.sin(a)))


def clip(poly, a, b, c):
    """Keep the part of a convex polygon with a*y + b*z <= c."""
    out = []
    p = [np.asarray(v, float) for v in poly]
    for i in range(len(p)):
        cur, nxt = p[i], p[(i + 1) % len(p)]
        fc, fn = a * cur[0] + b * cur[1] - c, a * nxt[0] + b * nxt[1] - c
        if fc <= 0:
            out.append(cur)
        if fc * fn < 0:
            t = fc / (fc - fn)
            out.append(cur + t * (nxt - cur))
    return _clean(out)


def _clean(pts):
    out = []
    for v in pts:
        if not out or np.linalg.norm(v - out[-1]) > 1e-4:
            out.append(np.asarray(v, float))
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) < 1e-4:
        out.pop()
    keep = []
    n = len(out)
    for i in range(n):
        a, b, c = out[i - 1], out[i], out[(i + 1) % n]
        if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 1e-6:
            keep.append(b)
    return keep


def clip_all(poly, planes):
    for a, b, c in planes:
        poly = clip(poly, a, b, c)
        if len(poly) < 3:
            return []
    return poly


def _solid(name, bottom, top, color):
    """A closed convex mesh from two matching convex rings of 3-D points (metres), faces turned outward."""
    b, t = np.asarray(bottom, float), np.asarray(top, float)
    n = len(b)
    p = np.vstack((b, t))
    c = p.mean(axis=0)
    tris = []
    for i in range(1, n - 1):
        tris.append((0, i, i + 1))
        tris.append((n, n + i, n + i + 1))
    for i in range(n):
        j = (i + 1) % n
        tris.append((i, j, n + j))
        tris.append((i, n + j, n + i))
    out = []
    for tri in tris:
        nrm = np.cross(p[tri[1]] - p[tri[0]], p[tri[2]] - p[tri[0]])
        cen = p[list(tri)].mean(axis=0)
        out.append(tri if np.dot(nrm, cen - c) > 0 else (tri[0], tri[2], tri[1]))
    lo, hi = p.min(axis=0), p.max(axis=0)
    return Mesh(name, p, np.asarray(out, int), collision="convex", color=color,
                proxy=(box(tuple((lo + hi) / 2), tuple(hi - lo)),))


def prism_x(name, poly_yz, x0, x1, color, turn=0.0):
    """Convex polygon in (y, z) (mm, z from the knob axis) turned by `turn` degrees about the axis, extruded along x
    from x0 to x1 (mm)."""
    poly = rot2(poly_yz, turn) if turn else np.asarray(poly_yz, float)
    ring = lambda x: [(x * MM, y * MM, (z + AX_Z) * MM) for y, z in poly]  # noqa: E731
    return _solid(name, ring(x0), ring(x1), color)


def prism_y(name, poly_xz, y0, y1, color, turn=0.0):
    """Convex polygon in (x, z) (mm, machine frame) extruded along y from y0 to y1 (mm), then turned by `turn`
    degrees about the knob axis."""
    def ring(y):
        pts = []
        for x, z in poly_xz:
            yy, zz = rot2([(y, z - AX_Z)], turn)[0] if turn else (y, z - AX_Z)
            pts.append((x * MM, yy * MM, (zz + AX_Z) * MM))
        return pts
    return _solid(name, ring(y0), ring(y1), color)


def bx(x0, x1, y0, y1, z0, z1, color):
    lo, hi = np.array([x0, y0, z0]) * MM, np.array([x1, y1, z1]) * MM
    return box(tuple((lo + hi) / 2), tuple(hi - lo), color)


# ---- the dial: knob, coin dial, ratchet, gumball wheel


def _layer_pieces(r, cut_w, floor, lead=None):
    """A disc of radius r with a channel |y| < cut_w above z > floor removed (pieces: below, left, right)."""
    d = disc(r)
    below = clip(d, 0.0, 1.0, floor)
    left = clip_all(d, [(1.0, 0.0, -cut_w), (0.0, -1.0, -floor)])
    right = clip_all(d, [(-1.0, 0.0, -cut_w), (0.0, -1.0, -floor)])
    return [below, left, right]


DIAL_AUTHOR = 180.0  # the dial is authored turned half a turn and starts at joint -180 deg: its whole travel
DIAL_HOME = -180.0  # (home -> one full turn and the gate's catch, about 390 deg) stays inside +-225 deg


def dial_meshes():
    """The dial's meshes as authored (home geometry turned by DIAL_AUTHOR about the knob axis)."""
    T = DIAL_AUTHOR
    out = []
    for tag, (x0, x1), lead_side in (("f", F_X, +1), ("b", B_X, -1)):
        below, left, right = _layer_pieces(DIAL_R, NOTCH_W, NOTCH_FLOOR)
        lower = clip(below, 0.0, 1.0, NOTCH_FLOOR - LEAD)
        out.append(prism_x(f"dial_{tag}_low", lower, x0, x1, CHROME, T))
        out.append(prism_x(f"dial_{tag}_left", left, x0, x1, CHROME, T))
        out.append(prism_x(f"dial_{tag}_right", right, x0, x1, CHROME, T))
        # the strip under the notch floor with a 1.5 mm lead-in on the edge facing the coin channel
        yw = math.sqrt(DIAL_R ** 2 - NOTCH_FLOOR ** 2)
        z0, z1 = AX_Z + NOTCH_FLOOR - LEAD, AX_Z + NOTCH_FLOOR
        if lead_side > 0:  # front layer: chamfer on its back edge (x1)
            poly = [(x0, z0), (x1, z0), (x1 - LEAD, z1), (x0, z1)]
        else:  # back layer: chamfer on its front edge (x0)
            poly = [(x0, z0), (x1, z0), (x1, z1), (x0 + LEAD, z1)]
        out.append(prism_y(f"dial_{tag}_lead", poly, -yw, yw, CHROME, T))
    below, left, right = _layer_pieces(DIAL_R, CHAN_W, CHAN_FLOOR)
    for nm, poly in (("s_low", below), ("s_left", left), ("s_right", right)):
        out.append(prism_x(f"dial_{nm}", poly, S_X[0], S_X[1], CHROME, T))
    # ratchet: hub and 8 teeth (radial face at the tooth's back, slope ahead)
    out.append(prism_x("ratchet_hub", disc(RATCHET_ROOT + 0.5, 32), RATCHET_X[0], RATCHET_X[1], BRASS, T))
    for k in range(RATCHET_TEETH):
        a_k = PAWL_ANGLE + 6.0 + 45.0 * k
        tri = [ang_pt(a_k, RATCHET_ROOT), ang_pt(a_k, RATCHET_TIP), ang_pt(a_k + 45.0, RATCHET_ROOT)]
        out.append(prism_x(f"ratchet_t{k}", tri, RATCHET_X[0], RATCHET_X[1], BRASS, T))
    # gumball wheel: two flanges and the pocket layer (pocket at POCKET_HOME)
    f0, f1 = WHEEL_X[0], WHEEL_X[1]
    out.append(prism_x("wheel_flange_a", disc(WHEEL_R), f0, f0 + FLANGE_T, DARK))
    out.append(prism_x("wheel_flange_b", disc(WHEEL_R), f1 - FLANGE_T, f1, DARK))
    for nm, poly in zip(("wheel_low", "wheel_left", "wheel_right"), _layer_pieces(WHEEL_R, POCKET_W, POCKET_FLOOR)):
        out.append(prism_x(nm, rot2(poly, POCKET_HOME), f0 + FLANGE_T, f1 - FLANGE_T, DARK, T))
    return tuple(out)


def dial_shapes():
    x0, x1, hw, hh = BAR
    return (bx(x0, x1, -hw, hw, AX_Z - hh, AX_Z + hh, CHROME),
            cyl(((PLATE_FRONT - 3.0) * MM, 0.0, AX_Z * MM), 9.0 * MM, 6.0 * MM, "x", CHROME),
            cyl(((SHAFT_X[0] + SHAFT_X[1]) / 2 * MM, 0.0, AX_Z * MM), SHAFT_R * MM, (SHAFT_X[1] - SHAFT_X[0]) * MM,
                "x", CHROME))


AXIS = (SLIT_X * MM, 0.0, AX_Z * MM)


def dial_link():
    j = Hinge("x", AXIS, DIAL_HOME - 5.0, 225.0, damping=0.02, armature=2e-4, static_friction=0.15,
              dynamic_friction=0.12, initial=DIAL_HOME)
    return Link("dial", dial_shapes(), j, 0.16, color=CHROME, friction=(0.35, 0.30))


# ---- the gate (coin pin) and the anti-reverse pawl


def _arm(alpha_tip, r_tip, arm_len, backward, tip_out=3.0, half=1.5, lift=0.0):
    """(pivot, arm polygon, tip polygon) for a strut arm whose tip rests on a rim of radius r_tip at alpha_tip.
    The pivot lies `arm_len` along the rim's tangent from the tip, behind (-) or ahead (+) of it; `lift` raises the
    arm that far further out (the tip then reaches down to the rim as a radial finger)."""
    t = ang_pt(alpha_tip, r_tip)
    radial = t / np.linalg.norm(t)
    tang = np.array([-radial[1], radial[0]]) * -1.0  # direction of the forward turn at the tip
    tang = tang / np.linalg.norm(tang)
    if np.dot(ang_pt(alpha_tip + 1.0, r_tip) - t, tang) < 0:
        tang = -tang
    d = tang * (1.0 if not backward else -1.0)
    pivot = t + d * arm_len + radial * (half + 0.3 + lift)
    base = t + radial * (half + 0.3 + lift)
    tip_out = max(tip_out, lift + 2 * half + 0.3)
    arm = [base - d * 1.0 + radial * -half * 0.0 - radial * half, pivot + d * 2.0 - radial * half,
           pivot + d * 2.0 + radial * half, base - d * 1.0 + radial * half]
    tip = [t - tang * 1.5 + radial * 0.3, t + tang * 1.5 + radial * 0.3, t + tang * 1.5 + radial * tip_out,
           t - tang * 1.5 + radial * tip_out]
    return pivot, arm, tip


def gate_link():
    """The coin pin (the interlock): a sprung plunger sliding along y in the plate's cavity at the knob axis height,
    pressing its flat end on the dial's rim at the side. The notch reaches it after about 75 degrees: with the channel
    empty the pin drops 4 mm into the notch and the notch's wall bears on its side, which its guide holds: the handle
    stops. A seated quarter fills the notch and pushes the pin out over it (6 mm)."""
    x0, x1 = CAV_X0 + 0.5, CAV_X1 - 0.5
    y_in = -(DIAL_R + 0.3)
    shapes = (bx(x0, x1, y_in - PIN_LEN, y_in, AX_Z - PIN_HALF_Z, AX_Z + PIN_HALF_Z, BRASS),)
    anchor = (SLIT_X * MM, (y_in - PIN_LEN / 2) * MM, AX_Z * MM)
    # positive slide moves it in (+y, toward the axis); the spring presses it on the rim (~0.3 N, 0.6 N fully out)
    j = Slide("y", anchor, -PIN_OUT * MM, PIN_IN * MM, damping=0.05, spring=50.0, rest=0.006, armature=1e-3,
              static_friction=0.05, dynamic_friction=0.04)
    return Link("gate", shapes, j, 0.003, color=BRASS, friction=(0.25, 0.20)), ()


def pawl_link():
    pivot, arm, tip = _arm(PAWL_ANGLE, RATCHET_ROOT + 0.8, PAWL_ARM, backward=True, lift=PAWL_LIFT)
    x0, x1 = RATCHET_X[0] + 0.5, RATCHET_X[1] - 0.5
    meshes = (prism_x("pawl_arm", arm, x0, x1, BRASS), prism_x("pawl_tip", tip, x0, x1, BRASS))
    anchor = ((x0 + x1) / 2 * MM, pivot[0] * MM, (pivot[1] + AX_Z) * MM)
    # positive turn moves its tip inward; the spring presses it onto the teeth (~0.2 N at the tip)
    j = Hinge("x", anchor, -30.0, 12.0, damping=2e-4, spring=0.02, rest=10.0, armature=1e-5)
    return Link("pawl", (), j, 0.003, color=BRASS, friction=(0.25, 0.20)), meshes


def _pawl_sign():
    """+1 if a positive pawl turn moves its tip inward (toward the ratchet), else -1 (offline check helper)."""
    pivot, _, tip = _arm(PAWL_ANGLE, RATCHET_ROOT + 0.8, PAWL_ARM, backward=True, lift=PAWL_LIFT)
    t = np.mean(tip, axis=0) - pivot
    moved = rot2([t], 1.0)[0] + pivot
    return 1 if np.linalg.norm(moved) < np.linalg.norm(t + pivot) else -1


# ---- the door


DOOR_TAB = (-104.0, -88.0, 42.0, 44.0)  # the door's turned-out bottom lip (16 mm, estimate): x from, x to, z from, z to


def door_link():
    """The chrome exit door: a 46 x 45 mm plate hanging from its top edge, with a 16 mm turned-out hem at the bottom
    (the finger lip) whose flat underside a fingertip lifts."""
    hx, hz = DOOR_HINGE
    x0, x1, z0, z1 = DOOR_TAB
    shapes = (bx(DOOR_X[0], DOOR_X[1], -DOOR_Y, DOOR_Y, DOOR_Z0, hz - 1.0, CHROME),
              bx(x0, x1, -DOOR_Y, DOOR_Y, z0, z1, CHROME),
              cyl((hx * MM, 0.0, hz * MM), 2.0 * MM, 2 * DOOR_Y * MM, "y", CHROME))
    j = Hinge("y", (hx * MM, 0.0, hz * MM), 0.0, 110.0, damping=2e-4, armature=2e-5)
    return Link("door", shapes, j, 0.020, color=CHROME, friction=(0.30, 0.25)), ()


# ---- the fixed body


def _ring_sectors(name, r0, r1, x0, x1, skip, color, step=15.0):
    """Annular sectors about the knob axis, leaving out angle ranges in `skip` [(from, to)] (degrees from the top)."""
    out = []
    a = 0.0
    k = 0
    while a < 360.0 - 1e-6:
        b = a + step
        mid = (a + b) / 2
        if not any(lo <= ((mid + 180) % 360) - 180 <= hi or lo <= mid <= hi for lo, hi in skip):
            poly = [ang_pt(a, r0), ang_pt(b, r0), ang_pt(b, r1), ang_pt(a, r1)]
            out.append(prism_x(f"{name}{k}", poly, x0, x1, color))
            k += 1
        a = b
    return out


def _plate_layer(name, x0, x1, r, gaps, color):
    """The plate's rectangle minus a circle of radius r about the axis, leaving the gaps [(side, half width)] open
    where side 'top' or 'bottom' crosses the rectangle's edge."""
    y1, z0, z1 = PLATE_Y, PLATE_Z0 - AX_Z, PLATE_Z1 - AX_Z
    out = []
    top_gap = dict(gaps).get("top", 0.0)
    bot_gap = dict(gaps).get("bottom", 0.0)

    def add(nm, poly):
        poly = _clean([np.asarray(v, float) for v in poly])
        if len(poly) >= 3:
            out.append(prism_x(f"{name}_{nm}", poly, x0, x1, color))

    rt = r  # band above / below the circle
    for side, sgn, zz, gap in (("t", 1, z1, top_gap), ("b", -1, z0, bot_gap)):
        za = sgn * rt
        lo, hi = (za, zz) if sgn > 0 else (zz, za)
        if gap > 0:
            add(f"{side}l", [(gap, lo), (y1, lo), (y1, hi), (gap, hi)])
            add(f"{side}r", [(-y1, lo), (-gap, lo), (-gap, hi), (-y1, hi)])
        else:
            add(side, [(-y1, lo), (y1, lo), (y1, hi), (-y1, hi)])
    add("sl", [(r, -rt), (y1, -rt), (y1, rt), (r, rt)])
    add("sr", [(-y1, -rt), (-r, -rt), (-r, rt), (-y1, rt)])
    # the four corners between the circle and the square of side 2r: triangle fans from each square corner
    for qy, qz, nm in ((1, 1, "c1"), (-1, 1, "c2"), (-1, -1, "c3"), (1, -1, "c4")):
        corner = np.array([qy * r, qz * r])
        gap = top_gap if qz > 0 else bot_gap
        n = 8
        pts = []
        for i in range(n + 1):
            th = math.radians(90.0 * i / n)
            p = np.array([qy * r * math.cos(th), qz * r * math.sin(th)])
            if abs(p[0]) < gap:  # keep the gap open: stop the fan at the gap's edge
                p = np.array([qy * gap, qz * math.sqrt(max(r * r - gap * gap, 0.0))])
            pts.append(p)
        for i in range(n):
            tri = [corner, pts[i], pts[i + 1]]
            area = abs((tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1]) - (tri[1][1] - tri[0][1]) * (tri[2][0] - tri[0][0]))
            if area > 1e-3:
                add(f"{nm}{i}", tri)
    if top_gap > 0:  # close the gap above the circle's top only where the circle is below the band
        pass
    return out


def base_meshes():
    m = []
    m += _plate_layer("plate_front", PLATE_FRONT, CAV_X0, HOLE_R, [], CHROME)
    cav = _plate_layer("plate_cav", CAV_X0, CAV_X1, SWEEP_R, [("top", MOUTH_W), ("bottom", 15.0)], CHROME)
    # the coin pin's guide: no cavity piece within 2 mm of the axis height on the -y side (base boxes above and below)
    def _in_pin_slot(mesh):
        p = np.asarray(mesh.points) / MM
        return p[:, 1].min() < -SWEEP_R + 1.0 and p[:, 2].min() < AX_Z + 2.0 and p[:, 2].max() > AX_Z - 2.0
    m += [c for c in cav if not _in_pin_slot(c)]
    m += _plate_layer("plate_back", CAV_X1, PLATE_BACK, HOLE_R, [("bottom", 15.0)], CHROME)
    m += _ring_sectors("ring", RING_R0, RING_R1, WHEEL_X[0], WHEEL_X[1],
                       [(-RING_OPEN, RING_OPEN), (180 - RING_OPEN, 180 + RING_OPEN)], DARK)
    # the coin slot: a 2.8 mm slit right over the dial's channel, with chamfered lips 6 mm apart at the top
    m.append(prism_y("slot_lip_front", [(CAV_X0, 177.0), (S_X[0], 177.0), (CAV_X0, PLATE_Z1)], -MOUTH_W, MOUTH_W,
                     CHROME))
    m.append(prism_y("slot_lip_back", [(S_X[1], 177.0), (CAV_X1, 177.0), (CAV_X1, PLATE_Z1)], -MOUTH_W, MOUTH_W,
                     CHROME))
    # the coin slope under the dial: tips a falling quarter back into the coin box
    slope = prism_y("coin_slope", [(-93.0, 98.0), (-72.0, 81.0), (-72.0, 84.0), (-93.0, 101.0)], -17.0, 17.0, DARK)
    # its stand-in for the offline checks: six steps that each enclose the sloped slab over 3.5 mm of x (the slab's
    # own bounding box would claim the whole 20 mm high corner under it)
    steps = []
    for i in range(6):
        xa, xb = -93.0 + 3.5 * i, -93.0 + 3.5 * (i + 1)
        steps.append(bx(xa, xb, -17.0, 17.0, 98.0 - 17.0 * (xb + 93.0) / 21.0, 101.0 - 17.0 * (xa + 93.0) / 21.0, DARK))
    m.append(dataclasses.replace(slope, proxy=tuple(steps)))
    # chute floor, sloping down to the door
    m.append(prism_y("exit_floor", [(FLOOR_X[0], FLOOR_Z_FRONT - 4.0), (FLOOR_X[1], FLOOR_Z_BACK - 4.0),
                                    (FLOOR_X[1], FLOOR_Z_BACK), (FLOOR_X[0], FLOOR_Z_FRONT)], -EXIT_W - 2.0,
                     EXIT_W + 2.0, DARK))
    return tuple(m)


def base_shapes():
    s = [
        bx(-88, 88, -107.5, 107.5, 0, FOOT_H, CHROME),  # chrome skirt (215 mm wide)
        bx(-35, BODY_X, -BODY_Y, BODY_Y, FOOT_H, BODY_TOP, RED),  # body behind the mechanism
        bx(-BODY_X, -35, 45, BODY_Y, FOOT_H, BODY_TOP, RED),
        bx(-BODY_X, -35, -BODY_Y, -45, FOOT_H, BODY_TOP, RED),
        bx(-BODY_X, -82, -45, -17, FOOT_H, PLATE_Z0, RED),  # front wall beside and under the exit
        bx(-BODY_X, -82, 17, 45, FOOT_H, PLATE_Z0, RED),
        bx(-BODY_X, -82, -17, 17, FOOT_H, FLOOR_Z_FRONT - 4.0, RED),
        # roof of the mechanism bay, around the feed tube
        bx(-BODY_X, TUBE_X[0] - 2, -45, 45, PLATE_Z1, BODY_TOP, RED), bx(TUBE_X[1] + 2, -35, -45, 45, PLATE_Z1, BODY_TOP, RED),
        bx(TUBE_X[0] - 2, TUBE_X[1] + 2, TUBE_Y + 2, 45, PLATE_Z1, BODY_TOP, RED),
        bx(TUBE_X[0] - 2, TUBE_X[1] + 2, -45, -TUBE_Y - 2, PLATE_Z1, BODY_TOP, RED),
        bx(-97, -93, -17, 17, 92.0, PLATE_Z0, CHROME),  # hood over the door's hinge
        # top band, around the feed tube
        bx(-88, TUBE_X[0] - 2, -93, 93, BODY_TOP, BAND_TOP, CHROME), bx(TUBE_X[1] + 2, 88, -93, 93, BODY_TOP, BAND_TOP, CHROME),
        bx(TUBE_X[0] - 2, TUBE_X[1] + 2, TUBE_Y + 2, 93, BODY_TOP, BAND_TOP, CHROME),
        bx(TUBE_X[0] - 2, TUBE_X[1] + 2, -93, -TUBE_Y - 2, BODY_TOP, BAND_TOP, CHROME),
        # the coin pin's guide in the plate's side (the cavity's side piece, split around the pin)
        bx(CAV_X0, CAV_X1, -PLATE_Y, -SWEEP_R, AX_Z - SWEEP_R, AX_Z - 2.0, CHROME),
        bx(CAV_X0, CAV_X1, -PLATE_Y, -SWEEP_R, AX_Z + 2.0, AX_Z + SWEEP_R, CHROME),
        # the slit's walls below the lips, down to 2 mm over the dial (clear of the dog at y < -12)
        bx(CAV_X0, S_X[0], -10, 10, 170, 177, CHROME), bx(S_X[1], CAV_X1, -10, 10, 170, 177, CHROME),
        # coin box and chute walls
        # (in front of x -85 they start at 92, clear of the door's hinge barrel and its swing)
        bx(-85, -72, 15, 17, 81, PLATE_Z0, DARK), bx(-85, -72, -17, -15, 81, PLATE_Z0, DARK),
        bx(-93, -85, 15, 17, 92, PLATE_Z0, DARK), bx(-93, -85, -17, -15, 92, PLATE_Z0, DARK),
        bx(-74, -72, -17, 17, 81, 107, DARK),  # chute front wall = coin box back wall (the ball passes under)
        bx(-38, -36, -17, 17, FLOOR_Z_BACK - 4.0, 107, DARK),
        bx(-82, -38, EXIT_W, EXIT_W + 2, FLOOR_Z_FRONT - 4.0, 81, DARK),
        bx(-82, -38, -EXIT_W - 2, -EXIT_W, FLOOR_Z_FRONT - 4.0, 81, DARK),
        bx(-72, -38, EXIT_W, EXIT_W + 2, 81, 107, DARK), bx(-72, -38, -EXIT_W - 2, -EXIT_W, 81, 107, DARK),
        # feed tube above the wheel
        bx(TUBE_X[0] - 2, TUBE_X[0], -TUBE_Y - 2, TUBE_Y + 2, TUBE_Z[0], TUBE_Z[1], DARK),
        bx(TUBE_X[1], TUBE_X[1] + 2, -TUBE_Y - 2, TUBE_Y + 2, TUBE_Z[0], TUBE_Z[1], DARK),
        bx(TUBE_X[0], TUBE_X[1], TUBE_Y, TUBE_Y + 2, TUBE_Z[0], TUBE_Z[1], DARK),
        bx(TUBE_X[0], TUBE_X[1], -TUBE_Y - 2, -TUBE_Y, TUBE_Z[0], TUBE_Z[1], DARK),
        bx(TUBE_X[0], TUBE_X[1], -TUBE_Y, TUBE_Y, TUBE_Z[1] - 2, TUBE_Z[1], DARK),  # cap
    ]
    # the globe: chrome ring, a pale back and sides, the gumballs (fixed scenery), the lid
    s.append(cyl((0.0, 0.0, 0.2075), 0.082, 0.005, "z", CHROME))
    s.append(box((0.062, 0.0, 0.296), (0.004, 0.110, 0.176), GLASS))
    for sy in (-1, 1):
        s.append(box((0.020, sy * 0.080, 0.296), (0.080, 0.004, 0.176), GLASS))
    s.append(cyl((0.0, 0.0, 0.390), 0.082, 0.010, "z", CHROME))
    s.append(cyl((0.0, 0.0, 0.401), 0.060, 0.012, "z", CHROME))
    s.append(cyl((0.0, 0.0, 0.245), 0.028, 0.070, "z", GUM_COLORS[0]))  # the pile's core
    k = 0
    for layer, z in enumerate((0.223, 0.248, 0.273)):
        for r, n in ((0.064, 16), (0.040, 10)):
            for i in range(n):
                a = TAU * (i + 0.5 * (layer % 2)) / n
                s.append(ball((r * math.cos(a), r * math.sin(a), z), GUM_R * MM * 0.98, GUM_COLORS[k % len(GUM_COLORS)]))
                k += 1
    return s


def machine(name, pos):
    gate, _ = gate_link()
    pawl, pawl_m = pawl_link()
    door, _ = door_link()
    return MeshMechanism(name, pos, base_shapes(), (dial_link(), gate, pawl, door), mass=MACHINE_MASS, color=RED,
                         friction=(0.6, 0.5), contact_offset=0.001, self_collide=True,
                         meshes={"base": base_meshes(), "dial": dial_meshes(), "pawl": pawl_m})


# ---- loose parts


def quarter(name, pos):
    """A US quarter lying flat (faces along z), a 32-sided prism of the coin's area."""
    rc = COIN_R * math.sqrt(TAU / (COIN_SIDES * math.sin(TAU / COIN_SIDES)))
    a = np.arange(COIN_SIDES) * TAU / COIN_SIDES
    disc_m = yam_mesh.prism(np.column_stack((rc * np.cos(a), rc * np.sin(a))) * MM, COIN_T * MM, axis="z",
                            name="disc", collision="convex", color=(0.66, 0.67, 0.70),
                            proxy=(cyl((0.0, 0.0, 0.0), COIN_R * MM, COIN_T * MM, "z"),))
    return MeshPart(name, pos, (), mass=COIN_MASS, color=(0.66, 0.67, 0.70), friction=(0.40, 0.35),
                    contact_offset=0.001, meshes=(disc_m,))


def gumball(name, pos, color):
    return Part(name, pos, (ball((0.0, 0.0, 0.0), GUM_R * MM, color),), mass=GUM_MASS, color=color,
                friction=(0.5, 0.4), contact_offset=0.001)


def pocket_ball_local():
    """The first gumball's centre in the wheel's pocket at home (machine frame, m)."""
    yz = ang_pt(POCKET_HOME, POCKET_FLOOR + GUM_R + 0.4)
    return np.array([(WHEEL_X[0] + WHEEL_X[1]) / 2, yz[0], yz[1] + AX_Z]) * MM


def tube_ball_local():
    """The second gumball resting on the wheel's rim in the feed tube (machine frame, m)."""
    return np.array([(TUBE_X[0] + TUBE_X[1]) / 2, 0.0, AX_Z + WHEEL_R + GUM_R + 1.2]) * MM


def coin_seated_local():
    """The quarter's centre standing on the coin channel's floor at home (machine frame, m)."""
    return np.array([SLIT_X, 0.0, AX_Z + CHAN_FLOOR + COIN_R + 0.05]) * MM


# ---- scorer helpers (machine-frame tests; `m` is the machine's state)


def in_box(p_local, lo, hi):
    p = np.asarray(p_local, float)
    return bool(np.all(p >= np.asarray(lo, float) * MM) and np.all(p <= np.asarray(hi, float) * MM))


def coin_in_channel(m, coin_world):
    """The quarter sits in the dial's coin channel (wherever the dial is turned)."""
    p = m.to_local(np.asarray(coin_world, float))
    d = p - np.array(AXIS)
    r = math.hypot(d[1], d[2])
    return bool(abs(d[0]) < 2.0 * MM and r < (DIAL_R + 7.0) * MM)


def coin_inside(m, coin_world):
    """The quarter has gone in: seated in the coin channel, or fallen through to the coin box."""
    p = m.to_local(np.asarray(coin_world, float))
    return bool(coin_in_channel(m, coin_world) or in_box(p, (-96, -18, 76), (-70, 18, 112)))


def dial_turn_deg(m):
    """How far the handle has been turned from home (degrees, clockwise from the front)."""
    return math.degrees(m.joints["dial"]) - DIAL_HOME


def ball_place(m, ball_world):
    """'held': in the wheel or the feed tube; 'chute': in the chute or behind the door; 'free': out of the machine
    (centre outside its base's footprint, in front of the door or anywhere else on the table); 'inside': anywhere
    else (not reachable through the mechanism)."""
    p = m.to_local(np.asarray(ball_world, float))
    if in_box(p, (-73, -24, 100), (-36, 24, 210)):
        return "held"
    if in_box(p, (-89, -18, 38), (-36, 18, 100)):
        return "chute"
    if not in_box(p, (-88.0, -107.5, -50.0), (88.0, 107.5, 500.0)):
        return "free"
    return "inside"
