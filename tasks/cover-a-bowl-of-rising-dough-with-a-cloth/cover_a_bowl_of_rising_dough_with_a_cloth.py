"""Cover a bowl of rising bread dough with a cloth: draw a washcloth off a shelf insert over the bowl and let go.

A clear Duralex Lys 14 cm glass bowl (500 ml) stands on the counter with a dome of dough in it (one 4 oz can of
Play-Doh, about 8 cm across, its top 2.3 cm below the rim). Behind it, 3 cm beyond its far rim, stands an IKEA VARIERA
shelf insert (32 x 28 x 16 cm, white steel). A 10 x 10 in (25.4 cm) cotton terry washcloth lies flat on the shelf with
3 cm of its near edge over the shelf's front edge, where it droops a little. The robot takes that lip with both
hands 20 cm apart, draws the cloth toward itself off the shelf and over the bowl, brings the held edge down past the
near rim so the rest of the cloth slides off the shelf and settles over the bowl, and lets go. The cloth must end up
covering the whole rim with overhang all round, not touching the dough, not resting on the shelf, and the bowl must
stay where it was.

The cloth is a true PhysX cloth (cloth_kit.py, a surface deformable of 1,369 nodes), not rigid pieces. Pads hold it
by a contact-gated constraint (cloth_kit.GripRule): only while both pads are stopped on the cloth with a measured
squeeze; opening the hand releases it. The bowl and the dough are one rigid body (the dough is pressed into the
bowl). The shelf insert is fixed. Metres; x forward from the arm bases, y left, z up.
"""
import dataclasses
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import Part, box  # noqa: E402
from yam_mesh import Mesh, MeshPart, revolve  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
import cloth_kit as ck  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S, SCORE_OVER = 1.0, "end"
RUN_LIMIT_STEPS = 1200  # 60 s at 20 Hz
DIAG = False             # GPU diagnostic copies print DOUGH lines
DEMO_ONE_HAND = False    # wrong-way copy: only the left hand takes the cloth; the right hand stays parked
TEST_GRIPPERS = None     # offline tests: {"left": (tcp xyz, opening 0..1), "right": ...}
TEST_HELD = False        # offline tests: pretend a hand still holds the cloth
TEST_CLOTH = None        # offline tests: cloth node positions (n, 3) for a checker state that has none (test_drape)
MM = 0.001


# ================================================================================================ scoring (first)
# Everything below the scoring section only builds the scene and the demonstration. The scorer reads the cloth's
# nodes (objects["cloth"].nodes) and the bowl's pose. What a person checks in a real trial is given with each test.
RIM_ANGLES = 72          # the rim is checked every 5 degrees
STILL_WINDOW = 1.0       # s: the cloth is still when no node has moved more than STILL_MOVE over the last second
STILL_MOVE = 0.010       # m   (real: no point moves 5 mm in 5 s; a window, not one step's velocity, which carries solver jitter)
CLEAR = 0.020            # m: a hand counts as let go when its tool point is this far from every cloth node
TIP_DEG = 15.0           # the bowl tipped: rim plane tilted more than this at any time
SAG_OK = 0.015           # m: the cloth's lowest point inside the rim at most this far below the rim's top
RIM_REST = 0.020         # m: covering cloth lies on the rim: no more than this above the rim's top


def bowl_frame(objects):
    """Bowl state -> (rotation, origin): bowl frame = the centre of its foot, z up its axis."""
    b = objects["bowl"]
    return b.rotation(), np.asarray(b.origin, float)


def cloth_nodes(objects):
    """The cloth's node positions (n, 3). A state without nodes (a rigid-body checker that cannot move cloth) is
    taken as the cloth lying where it starts."""
    nodes = getattr(objects["cloth"], "nodes", None)
    if nodes is None:
        return _NODES0 if TEST_CLOTH is None else np.asarray(TEST_CLOTH, float)
    return np.asarray(nodes, float)


def cloth_speed(objects):
    vel = getattr(objects["cloth"], "node_vel", None)
    if vel is None:
        return float(np.linalg.norm(getattr(objects["cloth"], "lin_vel", np.zeros(3))))
    return float(np.linalg.norm(np.asarray(vel, float), axis=1).max())


def cloth_motion(objects):
    """The largest distance any cloth node has moved over the last STILL_WINDOW seconds of the run (m). The scorer
    keeps the nodes of the last second; until it has a full second (and for a state outside a run) it takes the
    fastest node's speed times the window."""
    nodes = cloth_nodes(objects)
    t = _RUN["t"]
    if t is None:
        return cloth_speed(objects) * STILL_WINDOW
    if not _HIST or t > _HIST[-1][0] + 1e-6:
        _HIST.append((float(t), nodes.copy()))
    while len(_HIST) > 1 and _HIST[1][0] <= t - STILL_WINDOW + 1e-6:
        _HIST.pop(0)
    t0, n0 = _HIST[0]
    if t - t0 < STILL_WINDOW - 1e-6:
        return cloth_speed(objects) * STILL_WINDOW
    return float(np.linalg.norm(nodes - n0, axis=1).max())


def in_bowl_frame(objects, points):
    rot, org = bowl_frame(objects)
    return (np.asarray(points, float) - org) @ rot


def rim_cover(objects):
    """For each of RIM_ANGLES directions round the rim: is the rim's top covered by cloth (some cloth triangle lies
    over that point of the rim, from 3 mm under to RIM_REST over the rim's top: resting on it), and how much cloth runs from the rim's outer edge to the nearest free
    edge of the cloth (along the cloth, from the cloth's own coordinates; 0 where the rim is bare).
    Real trial: the rim is hidden under the cloth along that arc in the top photo; the overhang is read with a
    flexible tape along the cloth from the rim's outer edge to the nearest cloth edge."""
    loc = in_bowl_frame(objects, cloth_nodes(objects))
    tri = loc[FACES]                                               # (m, 3, 3)
    ang = np.arange(RIM_ANGLES) * 2 * math.pi / RIM_ANGLES
    covered = np.zeros(RIM_ANGLES, bool)
    overhang = np.zeros(RIM_ANGLES)
    px, py = RIM_MID_R * np.cos(ang), RIM_MID_R * np.sin(ang)
    a, b, c = tri[:, 0, :2], tri[:, 1, :2], tri[:, 2, :2]
    v0, v1 = b - a, c - a
    den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    dx = px[:, None] - a[None, :, 0]
    dy = py[:, None] - a[None, :, 1]
    s = (dx * v1[None, :, 1] - dy * v1[None, :, 0]) / den[None]
    t = (v0[None, :, 0] * dy - v0[None, :, 1] * dx) / den[None]
    inside = (s >= -1e-9) & (t >= -1e-9) & (s + t <= 1 + 1e-9)        # (angles, triangles)
    z = tri[None, :, 0, 2] + s * (tri[None, :, 1, 2] - tri[None, :, 0, 2]) + t * (tri[None, :, 2, 2] - tri[None, :, 0, 2])
    above = inside & (z >= BOWL_H - 0.003) & (z <= BOWL_H + RIM_REST)
    covered = above.any(axis=1)
    uv = UV[FACES]                                                     # material coordinates (m, 3, 2)
    u = uv[None, :, 0] + s[..., None] * (uv[None, :, 1] - uv[None, :, 0]) + t[..., None] * (uv[None, :, 2] - uv[None, :, 0])
    edge = np.minimum(CLOTH_W / 2 - np.abs(u[..., 0]), CLOTH_W / 2 - np.abs(u[..., 1]))   # to the nearest cloth edge
    best = np.where(above, edge, -1.0).max(axis=1) - BOWL_WALL / 2                   # from the rim's outer edge
    overhang = np.where(covered, np.maximum(best, 0.0), 0.0)
    return covered, overhang


def dough_terms(objects):
    """(touching, sag): does the cloth touch the dough (a node within the cloth's half thickness + 1.5 mm of the
    dome), and how far below the rim's top the cloth's lowest point inside the rim lies (m, 0 if none inside).
    Real trial: a side photo through the clear bowl with a ruler behind it."""
    loc = in_bowl_frame(objects, cloth_nodes(objects))
    r = np.hypot(loc[:, 0], loc[:, 1])
    d = np.linalg.norm(loc - DOME_CENTRE, axis=1) - DOME_SPHERE_R
    touching = bool(np.any((d <= CLOTH_T / 2 + 0.0015) & (loc[:, 2] >= DOUGH_Z0) & (r <= DOME_BASE_R + 0.002)))
    inner = r <= RIM_R - BOWL_WALL - 0.003
    sag = float(max(0.0, BOWL_H - loc[inner, 2].min())) if inner.any() else 0.0
    return touching, sag


def on_shelf(objects):
    """Any of the cloth still lying on the shelf insert's top (real trial: seen in the photo)."""
    n = cloth_nodes(objects)
    return bool(np.any((n[:, 0] > SHELF_X + 0.002) & (np.abs(n[:, 1]) < SHELF_W / 2) & (n[:, 2] > SHELF_TOP - 0.004)))


def hands_clear(objects):
    """Both hands have let go: no gated hold is on and each tool point is CLEAR from every cloth node (real trial:
    both grippers open and off the cloth in the final photo)."""
    if TEST_HELD:
        return False
    rule = ck.grip_rule()
    if rule is not None and any(a.on for a in rule._att.values()):
        return False
    hands = grippers()
    if hands is None:
        return True
    n = cloth_nodes(objects)
    return all(float(np.min(np.linalg.norm(n - np.asarray(tcp, float), axis=1))) >= CLEAR for tcp, _ in hands.values())


def score_terms(objects):
    """All the scored quantities and the score. score 0 if the cloth is still held, not still, covers no part of
    the rim, the bowl tipped (> 15 deg at any time) or the bowl moved more than 2 cm; otherwise the lowest of
    the five bands (coverage, overhang, bowl travel, dough, shelf)."""
    _episode_memory()
    covered, overhang = rim_cover(objects)
    frac = float(covered.mean())
    min_over = float(overhang[covered].min()) if covered.any() else 0.0
    touching, sag = dough_terms(objects)
    shelf = on_shelf(objects)
    bowl = objects["bowl"]
    start = _START.setdefault("bowl", np.asarray(bowl.origin, float)[:2] if _RUN["t"] is not None else np.asarray(BOWL_XY))
    travel = ever_max("travel", float(np.linalg.norm(np.asarray(bowl.origin, float)[:2] - start)))
    tipped = ever("tipped", bowl.tilt_deg() > TIP_DEG)
    speed = cloth_speed(objects)
    move = cloth_motion(objects)
    still = move <= STILL_MOVE
    released = hands_clear(objects)
    terms = dict(frac=frac, min_over=min_over, touching=touching, sag=sag, shelf=shelf, travel=travel,
                 tipped=tipped, still=still, speed=speed, move=move, released=released)
    if not released or not still or frac == 0.0 or tipped or travel > 0.02:
        terms["score"] = 0.0
        return terms
    band_cover = 1.0 if frac >= 1.0 else 0.5 if frac >= 0.75 else 0.25
    band_over = 1.0 if min_over >= 0.02 else 0.75 if min_over >= 0.01 else 0.5
    band_travel = 1.0 if travel <= 0.01 else 0.75
    band_dough = 0.5 if touching else 1.0 if sag <= SAG_OK else 0.75
    band_shelf = 0.25 if shelf else 1.0
    terms.update(bands=(band_cover, band_over, band_travel, band_dough, band_shelf))
    terms["score"] = min(band_cover, band_over, band_travel, band_dough, band_shelf)
    return terms


def score(objects):
    t = score_terms(objects)
    if DIAG:
        _diag(objects, t)
    return t["score"]


def success(objects):
    return score_terms(objects)["score"] >= 1.0


# ================================================================================================ run context
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
        context["step"] = 0
        context["env"] = self._env
        return result

    task_step._rc_wrapped = reset._rc_wrapped = True
    cls._rc_run = context
    cls._task_step, cls.reset = task_step, reset
    return context


_RUN = _run_context()
_MEMORY = {"episode": None}
_EVER, _START, _HIST = {}, {}, []


def _episode_memory():
    if _MEMORY["episode"] != _RUN["episode"]:
        _EVER.clear()
        _START.clear()
        _HIST.clear()
        _MEMORY["episode"] = _RUN["episode"]


def ever(key, cond):
    """Remember a condition met earlier in this episode (offline each call stands alone)."""
    if _RUN["t"] is None:
        return bool(cond)
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]


def ever_max(key, value):
    if _RUN["t"] is None:
        return float(value)
    _EVER[key] = max(float(_EVER.get(key, 0.0)), float(value))
    return _EVER[key]


def grippers():
    """{side: (tcp xyz, opening 0..1)} from the simulator (or TEST_GRIPPERS offline)."""
    if TEST_GRIPPERS is not None:
        return TEST_GRIPPERS
    env = _RUN["env"]
    if env is None:
        return None
    hands = ck.read_hands(env)
    return {s: (h.tcp, float(np.clip(h.gap / (2 * ck.FINGER_TRAVEL), 0, 1))) for s, h in hands.items()}


# ================================================================================================ the scene
GLASS_CLEAR = (0.80, 0.86, 0.89)   # clear glass renders as a pale grey-blue (no transparency in this renderer)
DOUGH_COLOR = (0.96, 0.93, 0.84)
SHELF_WHITE = (0.93, 0.93, 0.92)
CLOTH_BLUE = (0.16, 0.30, 0.72)

# IKEA VARIERA shelf insert, white: 32 x 28 x 16 cm (W x D x H), powder-coated steel, perforated top on four legs,
# open at the front. Its frame: the middle of its front edge on the counter. The sheet thickness, the depth of the
# top's turned-down edges and the leg section are not published: 1.5 mm, 12 mm and 8 x 8 mm (estimates).
SHELF_X = 0.25                      # front edge (x)
SHELF_W, SHELF_D, SHELF_H = 0.32, 0.28, 0.16
SHEET, SKIRT, LEG = 0.0015, 0.012, 0.008
SHELF_TOP = TOP + SHELF_H
SHELF = Part("shelf", (SHELF_X, 0.0, TOP), [
    box((SHELF_D / 2, 0.0, SHELF_H - SHEET / 2), (SHELF_D, SHELF_W, SHEET)),
    box((SHEET / 2, 0.0, SHELF_H - SKIRT / 2), (SHEET, SHELF_W, SKIRT)),
    box((SHELF_D - SHEET / 2, 0.0, SHELF_H - SKIRT / 2), (SHEET, SHELF_W, SKIRT)),
    *[box((x, y, (SHELF_H - SHEET) / 2), (LEG, LEG, SHELF_H - SHEET))
      for x in (LEG / 2, SHELF_D - LEG / 2) for y in (SHELF_W / 2 - LEG / 2, -SHELF_W / 2 + LEG / 2)]],
    color=SHELF_WHITE, friction=(0.35, 0.30))

# Duralex Lys 14 cm bowl, tempered glass: 14 cm across the rim, 5.5 cm high, 500 ml. Foot 7 cm across, wall 4.5 mm,
# bottom 6 mm (estimates that give the published 500 ml and a mass of ~0.27 kg). Outer profile r(z) = foot +
# (rim - foot)(1 - (1 - z/H)^2): flaring from the foot, upright at the rim. Frame: the centre of its foot.
BOWL_XY = (0.150, 0.000)
BOWL_H = 0.055
RIM_R = 0.070
FOOT_R = 0.035
BOWL_WALL = 0.0045
BOWL_BOTTOM = 0.006
RIM_MID_R = RIM_R - BOWL_WALL / 2
GLASS_DENSITY = 2500.0
WALL_BANDS = (0.0, 0.012, 0.026, 0.040, BOWL_H)
SEGMENTS = 24
# Dough: one 4 oz (112 g) can of Play-Doh pressed into the bottom as a dome 8 cm across and 2.6 cm high, its top
# 2.3 cm below the rim (shaped by ruler each trial). A rigid dome, part of the bowl's body.
DOUGH_MASS = 0.112
DOUGH_Z0 = BOWL_BOTTOM
DOME_BASE_R, DOME_H = 0.040, 0.026
DOME_SPHERE_R = (DOME_BASE_R ** 2 + DOME_H ** 2) / (2 * DOME_H)
DOME_CENTRE = np.array([0.0, 0.0, DOUGH_Z0 + DOME_H - DOME_SPHERE_R])


def r_out(z):
    z = min(max(z, 0.0), BOWL_H)
    return FOOT_R + (RIM_R - FOOT_R) * (1 - (1 - z / BOWL_H) ** 2)


def r_in(z):
    return r_out(z) - BOWL_WALL


def _volumes():
    zs = np.linspace(BOWL_BOTTOM, BOWL_H, 400)
    inner = float(np.trapezoid(math.pi * np.array([r_in(z) for z in zs]) ** 2, zs))
    zo = np.linspace(0.0, BOWL_H, 400)
    outer = float(np.trapezoid(math.pi * np.array([r_out(z) for z in zo]) ** 2, zo))
    return inner, outer - inner


BOWL_ML, _GLASS_M3 = _volumes()
BOWL_ML *= 1e6                       # ~497 ml (500 published)
GLASS_MASS = _GLASS_M3 * GLASS_DENSITY  # ~0.27 kg, estimate


def _arc_mesh(name, section, a0, a1, segments, color, proxy=()):
    """A closed (r, z) section swept about z from angle a0 to a1 with flat end caps (one convex piece)."""
    profile = np.asarray(section, float)
    n = len(profile)
    angles = np.linspace(a0, a1, segments + 1)
    points = np.array([(r * math.cos(a), r * math.sin(a), z) for a in angles for r, z in profile])
    faces = []
    for i in range(segments):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            faces += [(a, b, b + n), (a, b + n, a + n)]
    faces += [(0, j + 1, j) for j in range(1, n - 1)]
    last = segments * n
    faces += [(last, last + j, last + j + 1) for j in range(1, n - 1)]
    points, faces = np.asarray(points, float), np.asarray(faces, int)
    tri = (points - points.mean(axis=0))[faces]
    if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        faces = faces[:, ::-1]
    return Mesh(name, points, faces, collision="convex", color=color, proxy=tuple(proxy))


def make_bowl(pos):
    from yam_parts import cyl
    base = revolve([(0, 0), (r_out(0), 0), (r_out(BOWL_BOTTOM), BOWL_BOTTOM), (0, BOWL_BOTTOM)], samples=32,
                   name="foot", collision="convex", color=GLASS_CLEAR,
                   proxy=(cyl((0, 0, BOWL_BOTTOM / 2), FOOT_R, BOWL_BOTTOM),))
    meshes = [base]
    half = math.pi / SEGMENTS
    bands = list(WALL_BANDS)
    bands[0] = BOWL_BOTTOM
    for k in range(SEGMENTS):
        a = k * 2 * math.pi / SEGMENTS
        for i, (z0, z1) in enumerate(zip(bands[:-1], bands[1:])):
            section = [(r_in(z0), z0), (r_out(z0), z0), (r_out(z1), z1), (r_in(z1), z1)]
            zm = (z0 + z1) / 2
            pin = box((r_out(zm) * math.cos(a) - BOWL_WALL / 2 * math.cos(a), r_out(zm) * math.sin(a) - BOWL_WALL / 2 * math.sin(a), zm),
                      (0.001, 0.001, max(0.001, z1 - z0 - 0.002)))
            meshes.append(_arc_mesh(f"wall_{k}_{i}", section, a - half, a + half, 2, GLASS_CLEAR, (pin,)))
    # the dough dome: a spherical cap on the inside of the bottom
    phis = np.linspace(0.0, math.asin(DOME_BASE_R / DOME_SPHERE_R), 8)
    prof = [(0.0, DOUGH_Z0), (DOME_BASE_R, DOUGH_Z0)] + [(DOME_SPHERE_R * math.sin(p), DOME_CENTRE[2] + DOME_SPHERE_R * math.cos(p))
                                                        for p in phis[::-1][1:-1]] + [(0.0, DOUGH_Z0 + DOME_H)]
    dough = revolve(prof, samples=32, name="dough", collision="convex", color=DOUGH_COLOR,
                    proxy=(cyl((0, 0, DOUGH_Z0 + DOME_H / 2), DOME_BASE_R * 0.6, DOME_H * 0.9),))
    meshes.append(dough)
    return MeshPart("bowl", pos, (), mass=GLASS_MASS + DOUGH_MASS, color=GLASS_CLEAR, friction=(0.45, 0.40),
                    contact_offset=0.002, meshes=tuple(meshes))


BOWL = make_bowl((BOWL_XY[0], BOWL_XY[1], TOP))

# Burt's Bees Baby woven terry washcloth, 10 x 10 in (25.4 cm), 100 % cotton. Mass ~29 g (450 g/m2) and 2.5 mm
# thickness are estimates. Bend stiffness set so that its 30 mm lip droops about 8 mm at rest, as estimated for the
# real cloth (no published figure; check: lay it with 3 cm over the shelf edge, read the tip's height below the top).
CLOTH_W = 0.254
CLOTH_T = 0.0025
CLOTH_MASS = 0.029
LIP = 0.030                        # cloth over the shelf's front edge at the start
class _Cloth(ck.Cloth):
    """cloth_kit's Cloth plus the two fields the offline rigid-body layout checker reads for a type it does not know
    (it places such an object as a small fixed block at ``xy``); the simulator side is cloth_kit's, unchanged."""

    @property
    def xy(self):
        return (float(self.pos[0]), float(self.pos[1]))

    @property
    def extent(self):
        return 0.02


_SHEET = ck.Cloth.sheet("cloth", pos=(SHELF_X - LIP + CLOTH_W / 2, 0.0, SHELF_TOP + CLOTH_T / 2 + 0.001),
                       width=CLOTH_W, length=CLOTH_W, thickness=CLOTH_T, mass_kg=CLOTH_MASS, youngs=1.0e6,
                       poisson=0.25, bend=4.5e3, damping=1.0, bend_damping=5.0, friction=1.5,
                       rest_offset=CLOTH_T / 2, contact_offset=CLOTH_T / 2 + 0.004, iterations=24,
                       self_collide=True, edge=0.007, color=CLOTH_BLUE)
CLOTH = _Cloth(**{f.name: getattr(_SHEET, f.name) for f in dataclasses.fields(_SHEET)})
ck.use_grip_rule("cloth")
_NODES0, FACES = CLOTH.mesh(world=True)
UV = _NODES0[:, :2] - np.asarray(CLOTH.pos[:2])   # each node's place in the flat cloth (material coordinates)
EDGE_NEAR = CLOTH.edge_nodes("-x")                 # the near edge's nodes (the lip's tip)

OBJECTS = [SHELF, BOWL, CLOTH]


def test_drape(dx=0.0, dy=0.0, sag=0.0, lift=0.0, corner_on_shelf=False):
    """Offline test shape (states.yaml): the cloth laid over the bowl at its start place, its middle dx, dy from the
    bowl's axis, the top sagging ``sag`` at the middle, raised ``lift``; the part beyond the rim hangs straight down
    and then lies on the counter (lengths along the cloth kept). Not used by the simulator."""
    uv = UV + np.array([dx, dy])
    rho = np.hypot(uv[:, 0], uv[:, 1])
    out = np.zeros((len(uv), 3))
    top = BOWL_H + CLOTH_T / 2 + 0.0005 + lift
    inside = rho <= RIM_R
    out[inside, :2] = uv[inside]
    out[inside, 2] = top - sag * (1 - (rho[inside] / RIM_R) ** 2)
    o = ~inside
    drop = rho[o] - RIM_R
    out[o, :2] = uv[o] / rho[o, None] * (RIM_R + 0.002 + np.maximum(drop - top, 0.0))[:, None]
    out[o, 2] = top - np.minimum(drop, top)
    out += np.array([BOWL_XY[0], BOWL_XY[1], TOP])
    if corner_on_shelf:
        far = np.argsort(-UV[:, 0] - UV[:, 1])[:12]
        out[far] = np.array([SHELF_X + 0.03, 0.10, SHELF_TOP + CLOTH_T / 2]) + 0.002 * np.arange(12)[:, None]
    return out

# ================================================================================================ the demonstration
# Both hands level, pointing +x, fingers one above the other (stacked), take the lip 20 cm apart: the lower pad goes
# in 6 mm under the lip's tip, the fingers close while the hand comes down so the lower pad rises only ~8 mm and the
# upper pad presses the lip onto it. The contact gate then attaches the cloth. The hands lift 15 mm, draw the cloth
# toward the robot, pitch down and bring the held edge to just above the counter 6 cm short of the near rim, which
# drags the rest of the cloth off the shelf; they wait, open and rise. Reach (offline): the pitched carry needs the
# finger order opposite to cloth_kit.edge_pinch_ways (closing axis -z on the left arm, +z on the right).
G = ck.FINGER_TRAVEL
OPEN, CLOSED = -G, 0.0
Y_HAND = 0.100
X_IN = SHELF_X - 0.020             # tool point at the close: fingertips 7 mm short of the shelf edge
NOMINAL_DROOP = 0.008
LEAN = 20.0                         # deg, see _rot


def episode():
    return _RUN["episode"]


def _rot(side, pitch_deg, lean_deg=0.0):
    """Stacked grip pitched down by pitch_deg, then leaned outward by lean_deg (wrist out, fingertips in) so the two
    wrists (link4) stay apart when the hands are 20 cm apart and pitched."""
    s = 1.0 if side == "left" else -1.0
    p = math.radians(pitch_deg)
    base = ck.grip_matrix((math.cos(p), 0.0, -math.sin(p)), (-s * math.sin(p), 0.0, -s * math.cos(p)))
    q = -s * math.radians(lean_deg)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(q), -math.sin(q)], [0.0, math.sin(q), math.cos(q)]])
    return rx @ base


def lip_tip(side):
    """(x, z) of the lip's tip in front of this hand: the near-edge nodes within 15 mm of the hand's y, as settled
    in the simulator (offline: the nominal droop)."""
    y = (1.0 if side == "left" else -1.0) * Y_HAND
    env = _RUN["env"]
    if env is None:
        return SHELF_X - LIP + 0.001, SHELF_TOP - NOMINAL_DROOP
    nodes = ck._np(env.unwrapped.arena_world.get_nodal_positions_w("cloth"))[0] - ck._np(env.unwrapped.scene.env_origins)[0]
    sel = EDGE_NEAR[np.abs(nodes[EDGE_NEAR, 1] - y) < 0.015]
    tip = nodes[sel].mean(axis=0)
    if DIAG:
        print(f"DOUGH LIP {side} ep={_RUN['episode']} t={_RUN['t']}: tip x {tip[0]:.4f} droop {(SHELF_TOP - tip[2]) * 1e3:.1f} mm", flush=True)
    return float(tip[0]), float(tip[2])


def _pick_z(side, stage):
    _, tz = lip_tip(side)
    droop = max(0.0, SHELF_TOP - tz)
    if stage == "in":                       # lower face 6 mm under the tip, wide open
        return tz - 0.006 + G
    lower = tz + max(0.002, 0.4 * droop)    # closed: lower face lifts the tip a little; tool point mid-cloth
    return lower + CLOTH_T / 2 + 0.0005


PHASES = (("approach", 2.5), ("in", 1.5), ("close", 1.4), ("settle", 0.6), ("lift", 1.0), ("draw", 2.0),
          ("over", 2.0), ("lower", 2.0), ("pull", 2.0), ("wait", 1.5), ("open", 0.6), ("away", 1.5), ("rest", 2.5))


def arm_ways(side):
    s = 1.0 if side == "left" else -1.0
    y = s * Y_HAND
    dur = dict(PHASES)
    level = _rot(side, 0)
    nominal_in = SHELF_TOP - NOMINAL_DROOP - 0.006 + G
    cache = {}

    def z_of(stage, offset=0.0):
        def f():
            key = (stage, _RUN["episode"])
            if key not in cache:
                cache[key] = _pick_z(side, stage)
            return (X_IN, y, cache[key] + offset)
        return f

    ways = [
        ck.Way(dur["approach"], (X_IN - 0.04, y, nominal_in), OPEN, "approach", rot=level),
        ck.Way(dur["in"], z_of("in"), OPEN, "in", rot=level, episode=episode),
        ck.Way(dur["close"], z_of("close"), CLOSED, "close", rot=level, episode=episode),
        ck.Way(dur["settle"], z_of("close"), CLOSED, "settle", rot=level, episode=episode),
        ck.Way(dur["lift"], z_of("close", 0.015), CLOSED, "lift", rot=level, episode=episode),
        ck.Way(dur["draw"], (0.19, y, SHELF_TOP + 0.010), CLOSED, "draw", rot=_rot(side, 25, LEAN * 0.7)),
        ck.Way(dur["over"], (0.13, y, TOP + 0.120), CLOSED, "over", rot=_rot(side, 55, LEAN)),
        ck.Way(dur["lower"], (0.07, y, TOP + 0.070), CLOSED, "lower", rot=_rot(side, 80, LEAN)),
        ck.Way(dur["pull"], (0.02, y, TOP + 0.025), CLOSED, "pull", rot=_rot(side, 90, LEAN)),
        ck.Way(dur["wait"], (0.02, y, TOP + 0.025), CLOSED, "wait", rot=_rot(side, 90, LEAN)),
        ck.Way(dur["open"], (0.02, y, TOP + 0.025), -0.020, "open", rot=_rot(side, 90, LEAN)),
        ck.Way(dur["away"], (0.02, s * 0.14, TOP + 0.110), -0.020, "away", rot=_rot(side, 90, LEAN)),
        ck.Way(dur["rest"], (0.06, s * 0.20, TOP + 0.170), -0.020, "rest", rot=_rot(side, 90, LEAN)),
    ]
    return ways


def parked_ways(side):
    """The wrong-way copy's idle hand: stays clear above its own side of the counter."""
    s = 1.0 if side == "left" else -1.0
    total = sum(d for _, d in PHASES)
    return [ck.Way(total, (0.12, s * 0.30, TOP + 0.22), OPEN, "parked", rot=_rot(side, 60))]


def demo():
    return {"left": arm_ways("left"), "right": parked_ways("right") if DEMO_ONE_HAND else arm_ways("right")}


# ================================================================================================ diagnostics
_LOG = {"step": -1}


def _diag(objects, t):
    step = _RUN["step"]
    if step == _LOG["step"] or _RUN["t"] is None:
        return
    _LOG["step"] = step
    if not (step in (2, 20, 40) or step % 10 == 0):
        return
    n = cloth_nodes(objects)
    rule = ck.grip_rule()
    att = "" if rule is None else ",".join(f"{k[1][0]}" for k, a in rule._att.items() if a.on)
    st = "" if rule is None else " | ".join(f"{k[1][0]}:{'HELD' if v.get('attached') else 'ok' if v.get('ok') else 'no'}"
                                           f" p={v.get('pinch', 0):.2f} g={v.get('gap', 0) * 1e3:.2f} b={v.get('between', '-')}"
                                           for k, v in sorted(rule.status.items()))
    lip = n[EDGE_NEAR]
    print(f"DOUGH ep={_RUN['episode']} step={step} t={_RUN['t']:.2f} score={t['score']} cover={t['frac']:.2f} "
          f"over={t['min_over'] * 1e3:.0f}mm touch={t['touching']} sag={t['sag'] * 1e3:.0f}mm shelf={t['shelf']} "
          f"travel={t['travel'] * 1e3:.1f}mm tipped={t['tipped']} v={t['speed']:.3f} move={t['move'] * 1e3:.1f}mm released={t['released']} "
          f"att=[{att}] {st} | cloth x {n[:, 0].min():.3f}..{n[:, 0].max():.3f} z {n[:, 2].min() - TOP:.3f}..{n[:, 2].max() - TOP:.3f} "
          f"near edge x {lip[:, 0].mean():.3f} z {lip[:, 2].mean() - TOP:.3f}", flush=True)
    vel = getattr(objects["cloth"], "node_vel", None)
    if vel is not None and _RUN["t"] >= 15.0:
        sp = np.linalg.norm(np.asarray(vel, float), axis=1)
        k = int(sp.argmax())
        hands = grippers() or {}
        gap = {s: float(np.min(np.linalg.norm(n - np.asarray(tcp, float), axis=1))) for s, (tcp, _) in hands.items()}
        print(f"DOUGH FAST step={step} nodes>0.02: {int((sp > 0.02).sum())} fastest {sp[k]:.3f} at "
              f"({n[k, 0]:.3f}, {n[k, 1]:.3f}, {n[k, 2] - TOP:.3f}) | hand-to-cloth "
              + " ".join(f"{s} {g * 1e3:.0f}mm" for s, g in gap.items()), flush=True)
    if step in (2, 20, 40):
        for yy in (-Y_HAND, 0.0, Y_HAND):
            sel = EDGE_NEAR[np.argsort(np.abs(_NODES0[EDGE_NEAR, 1] - yy))[:2]]
            print(f"DOUGH DROOP step={step} y={yy:+.2f}: tip x {n[sel, 0].mean():.4f} droop {(SHELF_TOP - n[sel, 2].mean()) * 1e3:.1f} mm "
                  f"below the shelf top", flush=True)
