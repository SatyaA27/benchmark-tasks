"""Printed loose parts for YAM Arena tasks: wooden letter tiles, a printed game board, a tile rack.

What this module adds to the unchanged compound-parts and mesh kits beside it:

* ``PrintedPart``: a mesh part that also carries *prints*, flat coloured rectangles drawn on a face. A print is
  ink: it is drawn 0.05 mm proud of the face and has no collider and no mass, so nothing can catch on it and the
  scorer and the checkers only ever see the real body.
* ``loose=True``: the part is an ordinary dynamic rigid body (gravity, friction, contacts), but the harness's seeded
  reset does not shift it on its own (``free`` is False). Parts that only make sense in a fixed relation to others
  (tiles standing on a rack, tiles in the squares of a board) use this; the task shifts such groups itself.
* ``matte=True``: mesh pieces are drawn as plain wood or card instead of the mesh kit's metal look.
* a stroke font (``GLYPHS``) and ``glyph_prints`` to write a letter and its point value on a tile face.

Numbers used here: print step 0.05 mm; 32 position iterations for loose bodies (they are light and thin).
Everything from Isaac Lab and USD is imported lazily, inside the spawn hooks, so the module loads without a simulator.
"""
import math

import numpy as np

import yam_parts as parts
import yam_mesh as meshkit
from yam_mesh import Mesh, MeshPart

PRINT_STEP = 0.00005
LOOSE_ITERATIONS = 32

_BOX_FACES = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7),
              (2, 7, 6), (3, 0, 4), (3, 4, 7)]


def hull8(name, pts, color=None, proxy=()):
    """A convex six-sided solid from 8 corners ordered as a box (0-3 one face ring, 4-7 the opposite ring)."""
    pts = np.asarray(pts, float)
    vol = sum(float(np.dot(pts[a], np.cross(pts[b], pts[c]))) for a, b, c in _BOX_FACES) / 6.0
    faces = _BOX_FACES if vol > 0 else [f[::-1] for f in _BOX_FACES]
    return Mesh(name, pts, faces, collision="convex", color=color, proxy=tuple(proxy))


def box_hull(name, size, rot=None, centre=(0.0, 0.0, 0.0), color=None, proxy=()):
    """A convex box of `size`, turned by `rot` (3x3) about its centre, as a collision hull."""
    h = np.asarray(size, float) / 2
    ring = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    pts = np.array([(sx * h[0], sy * h[1], -h[2]) for sx, sy in ring] + [(sx * h[0], sy * h[1], h[2]) for sx, sy in ring])
    if rot is not None:
        pts = pts @ np.asarray(rot, float).T
    return hull8(name, pts + np.asarray(centre, float), color, proxy)


def bar_hull(name, profile_xz, length, color=None, proxy=()):
    """A convex bar: a four-cornered profile in (x, z), extruded `length` along y, centred on y = 0."""
    prof = [(float(x), float(z)) for x, z in profile_xz]
    pts = [(x, -length / 2, z) for x, z in prof] + [(x, length / 2, z) for x, z in prof]
    return hull8(name, pts, color, proxy)


class Print:
    """One flat rectangle of ink: centre (u, v), size (du, dv) and turn (degrees) in a frame's x-y plane, on layer
    `layer` (1 = lowest); `frame` indexes the part's print frames."""

    __slots__ = ("u", "v", "du", "dv", "deg", "color", "layer", "frame")

    def __init__(self, u, v, du, dv, color, deg=0.0, layer=1, frame=0):
        self.u, self.v, self.du, self.dv = float(u), float(v), float(du), float(dv)
        self.deg, self.color, self.layer, self.frame = float(deg), tuple(color), int(layer), int(frame)


class PrintedPart(MeshPart):
    """A mesh part with prints. `frames` = ((origin, 3x3 rotation), ...) in the part frame: prints lie in a frame's
    x-y plane and stand off along its z."""

    def __init__(self, name, pos, shapes=(), mass=None, color=parts.GREY, yaw=0.0, friction=None,
                 contact_offset=0.002, grip=None, *, meshes=(), prints=(), frames=(((0, 0, 0), np.eye(3)),),
                 loose=False, matte=True):
        super().__init__(name, pos, shapes, mass, color, yaw, friction, contact_offset, grip, meshes=meshes)
        object.__setattr__(self, "prints", tuple(prints))
        object.__setattr__(self, "frames", tuple((np.asarray(o, float), np.asarray(r, float)) for o, r in frames))
        object.__setattr__(self, "loose", bool(loose))
        object.__setattr__(self, "matte", bool(matte))

    @property
    def free(self):
        return self.mass is not None and not self.loose

    def twin(self):
        """The same body as a plain mesh part (what the kits spawn)."""
        return MeshPart(self.name, self.pos, self.shapes, self.mass, self.color, self.yaw, self.friction,
                        self.contact_offset, self.grip, meshes=self.meshes)


# ---------------------------------------------------------------- stroke font
# Stroke centre lines as polylines on a 4 wide by 6 tall grid (u to the right, v up).
GLYPHS = {
    "A": (((0, 0), (2, 6), (4, 0)), ((0.8, 2.2), (3.2, 2.2))),
    "E": (((4, 0), (0, 0), (0, 6), (4, 6)), ((0, 3), (3, 3))),
    "H": (((0, 0), (0, 6)), ((4, 0), (4, 6)), ((0, 3), (4, 3))),
    "I": (((2, 0), (2, 6)), ((0.8, 0), (3.2, 0)), ((0.8, 6), (3.2, 6))),
    "L": (((0, 6), (0, 0), (4, 0)),),
    "N": (((0, 0), (0, 6), (4, 0), (4, 6)),),
    "O": (((1, 0), (3, 0), (4, 1), (4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0)),),
    "P": (((0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)),),
    "R": (((0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)), ((2, 3), (4, 0))),
    "S": (((4, 5), (3, 6), (1, 6), (0, 5), (0, 4), (1, 3), (3, 3), (4, 2), (4, 1), (3, 0), (1, 0), (0, 1)),),
    "T": (((0, 6), (4, 6)), ((2, 6), (2, 0))),
    "U": (((0, 6), (0, 1), (1, 0), (3, 0), (4, 1), (4, 6)),),
    "X": (((0, 0), (4, 6)), ((0, 6), (4, 0))),
    "1": (((1, 4.8), (2.2, 6), (2.2, 0)),),
    "3": (((0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (3, 3), (1.5, 3)), ((3, 3), (4, 2), (4, 1), (3, 0), (1, 0), (0, 1))),
    "4": (((3, 0), (3, 6), (0, 2), (4, 2)),),
    "8": (((1, 3), (0, 4), (0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (3, 3), (1, 3), (0, 2), (0, 1), (1, 0), (3, 0),
           (4, 1), (4, 2), (3, 3)),),
}


def glyph_prints(char, centre, width, height, stroke, color, layer=1, frame=0):
    """Prints for one character, read with the frame's +x up and +y to the reader's left (so u runs along -y)."""
    out = []
    su, sv = width / 4.0, height / 6.0
    for line in GLYPHS[char]:
        for (u0, v0), (u1, v1) in zip(line, line[1:]):
            a = np.array([centre[0] + (v0 - 3) * sv, centre[1] - (u0 - 2) * su])
            b = np.array([centre[0] + (v1 - 3) * sv, centre[1] - (u1 - 2) * su])
            d = b - a
            out.append(Print((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, float(np.linalg.norm(d)) + stroke, stroke, color,
                             deg=math.degrees(math.atan2(d[1], d[0])), layer=layer, frame=frame))
    return out


# ---------------------------------------------------------------- spawn hooks (simulator host only)
def _look(stage, prefix, color, cache):
    import isaaclab.sim as sim_utils
    from yam_arena.station import linear_rgb
    key = tuple(round(float(c), 4) for c in color)
    if key not in cache:
        look_path = f"{prefix}_{len(cache)}"
        look = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(key), roughness=0.6, metallic=0.0)
        look.func(look_path, look)
        cache[key] = look_path
    return cache[key]


def _author_prints(stage, prim_path, spec):
    import isaaclab.sim as sim_utils
    from pxr import Gf, UsdGeom
    cache = {}
    for index, (origin, rot) in enumerate(spec.frames):
        if not any(p.frame == index for p in spec.prints):
            continue
        frame_path = f"{prim_path}/prints_{index}"
        frame = UsdGeom.Xform.Define(stage, frame_path)
        r = np.asarray(rot, float)
        frame.AddTransformOp().Set(Gf.Matrix4d(
            float(r[0, 0]), float(r[1, 0]), float(r[2, 0]), 0.0,
            float(r[0, 1]), float(r[1, 1]), float(r[2, 1]), 0.0,
            float(r[0, 2]), float(r[1, 2]), float(r[2, 2]), 0.0,
            float(origin[0]), float(origin[1]), float(origin[2]), 1.0))
        for k, p in enumerate(q for q in spec.prints if q.frame == index):
            path = f"{frame_path}/ink_{k:04d}"
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(p.u, p.v, (p.layer - 0.5) * PRINT_STEP))
            cube.AddRotateZOp().Set(p.deg)
            cube.AddScaleOp().Set(Gf.Vec3f(p.du, p.dv, PRINT_STEP))
            sim_utils.bind_visual_material(path, _look(stage, f"{prim_path}/ink_look", p.color, cache),
                                           stage=stage)


def _matte_meshes(stage, prim_path, spec):
    import isaaclab.sim as sim_utils
    cache = {}
    for mesh in spec.meshes:
        color = spec.color if mesh.color is None else mesh.color
        sim_utils.bind_visual_material(f"{prim_path}/mesh_{mesh.name}", _look(stage, f"{prim_path}/matte_look", color, cache),
                                       stage=stage)


_ORIGINAL = {}


def _printed(spec):
    """A PrintedPart, whichever import of this module made it."""
    return hasattr(spec, "twin") and hasattr(spec, "prints")


def _spawn_part(prim_path, cfg, *args, **kwargs):
    spec = parts._SPECS[cfg.part_name]
    if not _printed(spec):
        return _ORIGINAL["spawn_part"](prim_path, cfg, *args, **kwargs)
    parts._SPECS[cfg.part_name] = spec.twin()
    try:
        prim = _ORIGINAL["spawn_part"](prim_path, cfg, *args, **kwargs)
    finally:
        parts._SPECS[cfg.part_name] = spec
    if spec.loose and spec.mass is not None:
        from pxr import PhysxSchema
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSolverPositionIterationCountAttr().Set(LOOSE_ITERATIONS)
    return prim


def _spawn_mesh_part(path, cfg, *args, **kwargs):
    prim = _ORIGINAL["spawn_mesh_part"](path, cfg, *args, **kwargs)
    spec = parts._SPECS.get(cfg.part_name)
    if _printed(spec):
        import isaaclab.sim as sim_utils
        stage = kwargs.get("stage") or sim_utils.get_current_stage()
        try:
            if spec.matte:
                _matte_meshes(stage, str(path), spec)
            _author_prints(stage, str(path), spec)
        except Exception as e:  # noqa: BLE001  (looks only: never stop a run for them)
            print(f"TILE_KIT prints failed on {spec.name}: {e!r}", flush=True)
    return prim


def _part_object(spec):
    if _printed(spec):
        obj = _ORIGINAL["part_object"](spec.twin())
        parts._SPECS[spec.name] = spec
        return obj
    return _ORIGINAL["part_object"](spec)


def install():
    """Wrap the kits' spawn hooks (again on every import: the checkers reload the kits, and the newest wins)."""
    for module, names in ((parts, ("_spawn_part", "_part_object")), (meshkit, ("_spawn_mesh_part",))):
        kept = getattr(module, "_tile_kit_originals", None)
        if kept is None:
            kept = {n: getattr(module, n) for n in names}
            module._tile_kit_originals = kept
        _ORIGINAL.update({n.lstrip("_"): f for n, f in kept.items()})
    parts._spawn_part, parts._part_object = _spawn_part, _part_object
    meshkit._spawn_mesh_part = _spawn_mesh_part


install()
