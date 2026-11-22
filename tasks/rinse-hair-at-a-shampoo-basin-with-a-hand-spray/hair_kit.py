"""Soft strands for YAM Arena tasks (a rope, or a section of hair held as one bundle), anchored at one end and
simulated as a chain of capsules, plus fixed figures built from simple shapes (a mannequin head).

A strand is declared in the task file as an ordinary fixed :class:`Box` (its straight starting footprint, so layout
checks see where it lies) and registered here by name with its parameters. When the simulator builds the scene,
the hook below replaces that box with an articulation rooted at the held end: ``links`` capsule links of length
``pitch`` laid along the box's long axis from the held end, each joined to the next by a joint whose stretch and twist are locked and whose two bending axes are free
(+-60 degrees) on passive spring/damper drives toward the rest shape: straight for ``strand``, the declared
start shape for ``strand_path`` (a strand laid along a curved path, such as hair combed over a head). The first link is the held end (under
a clamp, or inside a hair elastic): it is fixed to the world where it was spawned. Every bending axis has a small
joint armature (5e-6 kg m^2); without it, closing a gripper on a light capsule chain destabilises the solver.
Self-collision is on, so strands lie over and under each other. A strand may carry a coloured band (a visual ring
on one link, such as an elastic binding a hair section's tip). ``object_states`` reports each strand as a
:class:`RopeState` with every link's centre, axis and speed.

A figure is declared as a fixed :class:`Box` (its bounding box) and registered with a list of pieces in its own
frame: visible ellipsoids, cylinders and boxes, each with a colour, and invisible static colliders (pieces marked
``collide``: spheres, cylinders and boxes; a non-uniform ellipsoid collides through its convex hull). It is a fixed body: nothing in it moves. Only primitive specs and
NumPy are imported before the simulator starts.
"""
import functools
import importlib.abc
import importlib.machinery
import math
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena.task_file import Box, ObjectState


@dataclass(frozen=True)
class Rope:
    """Rope parameters of one strand (SI units)."""
    links: int
    pitch: float  # link length (m)
    radius: float
    link_mass: float
    bend_k: float  # N m/rad per joint (bending stiffness EI / pitch)
    bend_d: float  # N m s/rad per joint
    friction: tuple = (0.8, 0.8)
    bend_limit: float = 60.0  # degrees
    armature: float = 5.0e-6
    contact_offset: float = 0.001
    lin_damp: float = 0.5
    ang_damp: float = 2.0
    color: tuple = (0.8, 0.8, 0.8)
    roughness: float = 0.85
    band_link: int = -1  # the link carrying a coloured band (visual only); -1: none
    band_color: tuple = (1.0, 1.0, 1.0)
    band_width: float = 0.008

    @property
    def length(self):
        return self.links * self.pitch


def registry():
    """{strand name: (Rope, link centres (N, 3), link rotations (N, 3, 3))}, kept on the yam_arena package so every
    import of this helper shares it."""
    import yam_arena
    if not hasattr(yam_arena, "_hair_strands"):
        yam_arena._hair_strands = {}
    return yam_arena._hair_strands


def _frames_from_path(points):
    """Per-link (centre, rotation) of a strand laid along the polyline `points` (links + 1 joint points, world): link
    i runs from points[i] to points[i + 1], its X along that segment; Y and Z are carried along by parallel transport,
    so the start shape has no twist."""
    P = np.asarray(points, float)
    xs = P[1:] - P[:-1]
    xs /= np.linalg.norm(xs, axis=1)[:, None]
    ref = np.array((0.0, 0.0, 1.0)) if abs(xs[0][2]) < 0.9 else np.array((0.0, 1.0, 0.0))
    y = np.cross(ref, xs[0])
    y /= np.linalg.norm(y)
    rots = []
    for x in xs:
        y = y - x * (y @ x)
        y /= np.linalg.norm(y)
        rots.append(np.column_stack((x, y, np.cross(x, y))))
    return (P[1:] + P[:-1]) / 2, np.array(rots)


def strand(name, held_end, heading, rope: Rope, table_top):
    """A strand lying straight on the table from its held end (x, y) along `heading` (radians, world yaw).
    Returns the fixed Box that stands for it in the task's OBJECTS."""
    heading = math.atan2(math.sin(heading), math.cos(heading))
    d = np.array((math.cos(heading), math.sin(heading), 0.0))
    z = table_top + rope.radius + 0.0005
    start = np.array((held_end[0], held_end[1], z))
    return strand_path(name, [start + d * rope.pitch * i for i in range(rope.links + 1)], rope)


def strand_path(name, points, rope: Rope):
    """A strand whose start (and rest) shape is the polyline `points` (rope.links + 1 joint points, world, each
    rope.pitch apart); points[0] is the held end. The joints' zero is this shape, so the bending springs pull toward it
    (with hair-like stiffness gravity and contact dominate). Returns the fixed Box (the path's bounding box) that
    stands for it in the task's OBJECTS."""
    P = np.asarray(points, float)
    if rope.pitch <= 2 * rope.radius:
        raise ValueError(f"{name}: pitch must exceed the capsule's diameter (the capsule's cylinder has length pitch - 2 r)")
    if P.shape != (rope.links + 1, 3):
        raise ValueError(f"{name}: need {rope.links + 1} points")
    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
    if np.max(np.abs(seg - rope.pitch)) > 1e-4:
        raise ValueError(f"{name}: path points must be one pitch apart")
    centres, rots = _frames_from_path(P)
    registry()[name] = (rope, centres, rots)
    lo, hi = P.min(axis=0) - rope.radius, P.max(axis=0) + rope.radius
    return Box(name=name, pos=tuple(float(v) for v in (lo + hi) / 2), size=tuple(float(v) for v in hi - lo),
               color=rope.color)


def quat_xyzw(R):
    """Unit quaternion (x, y, z, w) of a rotation matrix (Shepperd's method: stable for every angle, 180 deg too)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        r = math.sqrt(1.0 + t) * 2
        q = ((R[2, 1] - R[1, 2]) / r, (R[0, 2] - R[2, 0]) / r, (R[1, 0] - R[0, 1]) / r, r / 4)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        r = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = (r / 4, (R[0, 1] + R[1, 0]) / r, (R[0, 2] + R[2, 0]) / r, (R[2, 1] - R[1, 2]) / r)
    elif R[1, 1] > R[2, 2]:
        r = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = ((R[0, 1] + R[1, 0]) / r, r / 4, (R[1, 2] + R[2, 1]) / r, (R[0, 2] - R[2, 0]) / r)
    else:
        r = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = ((R[0, 2] + R[2, 0]) / r, (R[1, 2] + R[2, 1]) / r, r / 4, (R[1, 0] - R[0, 1]) / r)
    q = np.array(q)
    q = q / np.linalg.norm(q)
    return q if q[3] >= 0 else -q


@dataclass(frozen=True)
class Piece:
    """One shape of a figure, in the figure's frame (m). kind "ellipsoid": dims = semi-axes (x, y, z); "cylinder":
    dims = (radius, length) along ``axis``; "box": dims = edge lengths. ``collide``: an invisible static collider
    instead of a visible shape (only uniform spheres, cylinders and boxes)."""
    kind: str
    centre: tuple
    dims: tuple
    color: tuple
    axis: str = 'X'
    collide: bool = False


def figures():
    """{figure name: (pieces, origin (x, y, z), yaw)}, shared like registry()."""
    import yam_arena
    if not hasattr(yam_arena, '_hair_figures'):
        yam_arena._hair_figures = {}
    return yam_arena._hair_figures


def piece_bounds(p: Piece):
    c = np.asarray(p.centre, float)
    if p.kind == 'ellipsoid' or p.kind == 'box':
        half = np.asarray(p.dims, float) * (1.0 if p.kind == 'ellipsoid' else 0.5)
    else:
        r, h = p.dims
        half = np.full(3, float(r))
        half['XYZ'.index(p.axis)] = h / 2
    return c - half, c + half


def figure(name, origin, yaw, pieces, color):
    """A fixed figure at `origin` (world, m; its frame's z = 0 on the surface it stands on) turned by `yaw`; returns
    the fixed Box (its bounding box above that surface) that stands for it in the task's OBJECTS."""
    pieces = tuple(pieces)
    for p in pieces:
        if p.collide and p.kind == 'ellipsoid' and len(set(p.dims)) > 1 and p.dims != tuple(p.dims):
            raise ValueError(f'{name}: bad collider dims')
    figures()[name] = (pieces, tuple(float(v) for v in origin), float(yaw))
    lo = np.min([piece_bounds(p)[0] for p in pieces], axis=0)
    hi = np.max([piece_bounds(p)[1] for p in pieces], axis=0)
    lo[2] = max(lo[2], 0.0)  # the frame's z = 0 is the surface it stands on; anything below is hidden in it
    c, s = math.cos(yaw), math.sin(yaw)
    centre = np.asarray(origin, float) + np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))) @ ((lo + hi) / 2)
    return Box(name=name, pos=tuple(float(v) for v in centre), size=tuple(float(v) for v in hi - lo), color=color,
               yaw=float(yaw))


def world_centres(name):
    """Link centres of the starting strand (world frame)."""
    return np.asarray(registry()[name][1])


def rotation(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                     (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


@dataclass(frozen=True)
class RopeState(ObjectState):
    """A strand's state: per-link centres (N, 3), unit axes toward the free end (N, 3) and speeds (N,)."""
    centres: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    axes: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    speeds: np.ndarray = field(default_factory=lambda: np.empty(0))
    pitch: float = 0.015

    def polyline(self):
        """The centre-line through the joints: start of the held link, each shared joint point, the free end."""
        h = self.pitch / 2
        c, a = self.centres, self.axes
        mids = ((c[:-1] + a[:-1] * h) + (c[1:] - a[1:] * h)) / 2
        return np.vstack((c[0] - a[0] * h, mids, c[-1] + a[-1] * h))


# ---------------------------------------------------------------- simulator side (imported lazily)
_CFG = None
_SPAWN = None


def _object(box):
    global _CFG, _SPAWN
    _, centres, rots = registry()[box.name]
    held, q0 = tuple(float(v) for v in centres[0]), tuple(float(v) for v in quat_xyzw(rots[0]))
    import isaaclab.sim as sim
    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.terms.events import reset_articulation_pose_and_joints
    from isaaclab_arena.utils.pose import Pose

    if _CFG is None:
        @configclass
        class StrandSpawnCfg(sim.SpawnerCfg):
            strand_name: str = ''
        _CFG = StrandSpawnCfg
        _SPAWN = clone(_spawn)

    class StrandObject(Object):
        def _build_reset_event(self):
            return EventTermCfg(func=reset_articulation_pose_and_joints, mode='reset',
                                params={'asset_cfg': SceneEntityCfg(self.name), 'pose': self.get_initial_pose(),
                                        'velocity': None})

    return StrandObject(name=box.name, object_type=ObjectType.ARTICULATION,
                        spawner_cfg=_CFG(func=_SPAWN, strand_name=box.name),
                        initial_pose=Pose(position_xyz=held, rotation_xyzw=q0))  # the root is the held link


def _spawn(prim_path, cfg, *args, **kwargs):
    import isaaclab.sim as sim
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    from yam_arena.station import linear_rgb

    s, centres, rots = registry()[cfg.strand_name]
    R0 = rots[0]
    stage = kwargs.get('stage') or sim.get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    translation = kwargs.get('translation', args[0] if args else None)
    orientation = kwargs.get('orientation', args[1] if len(args) > 1 else None)
    if translation is not None:
        root.AddTranslateOp().Set(Gf.Vec3d(*map(float, translation)))
    if orientation is not None:
        x, y, z, w = map(float, orientation)
        root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    matpath, vispath = f'{prim_path}/rope_material', f'{prim_path}/rope_look'
    mat = sim.RigidBodyMaterialCfg(static_friction=s.friction[0], dynamic_friction=s.friction[1], restitution=0.)
    mat.func(matpath, mat)
    look = sim.PreviewSurfaceCfg(diffuse_color=linear_rgb(s.color), roughness=s.roughness)
    look.func(vispath, look)
    if s.band_link >= 0:
        bandpath = f'{prim_path}/band_look'
        band = sim.PreviewSurfaceCfg(diffuse_color=linear_rgb(s.band_color), roughness=.6)
        band.func(bandpath, band)
    r, h, m = s.radius, s.pitch - 2 * s.radius, s.link_mass  # capsule inertia (cylinder + two half-spheres)
    vc, vs = math.pi * r * r * h, 4 * math.pi * r ** 3 / 3
    mc, ms = m * vc / (vc + vs), m * vs / (vc + vs)
    axial = mc * r * r / 2 + 2 * ms * r * r / 5
    trans = mc * (3 * r * r + h * h) / 12 + ms * (2 * r * r / 5 + h * h / 4 + 3 * h * r / 8)
    for i in range(s.links):
        path = f'{prim_path}/seg_{i:02d}'
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTranslateOp().Set(Gf.Vec3d(*map(float, R0.T @ (centres[i] - centres[0]))))
        qx, qy, qz, qw = quat_xyzw(R0.T @ rots[i])
        body.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(qw), Gf.Vec3d(float(qx), float(qy), float(qz))))
        prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr().Set(s.link_mass)
        mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.))
        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(axial, trans, trans))
        rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rigid.CreateMaxDepenetrationVelocityAttr().Set(1.)
        rigid.CreateLinearDampingAttr().Set(s.lin_damp)
        rigid.CreateAngularDampingAttr().Set(s.ang_damp)
        rigid.CreateSolverPositionIterationCountAttr().Set(32)
        rigid.CreateSolverVelocityIterationCountAttr().Set(4)
        geom = UsdGeom.Capsule.Define(stage, path + '/capsule')
        geom.CreateAxisAttr().Set('X')
        geom.CreateRadiusAttr().Set(s.radius)
        geom.CreateHeightAttr().Set(s.pitch - 2 * s.radius)
        geom.CreateVisibilityAttr().Set('invisible')
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        col = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
        col.CreateContactOffsetAttr().Set(s.contact_offset)
        col.CreateRestOffsetAttr().Set(0.)
        sim.bind_physics_material(path + '/capsule', matpath, stage=stage)
        vis = UsdGeom.Capsule.Define(stage, path + '/visual')  # overlapping visual capsules read as one rope
        vis.CreateAxisAttr().Set('X')
        vis.CreateRadiusAttr().Set(s.radius)
        vis.CreateHeightAttr().Set(s.pitch - 0.2 * s.radius)
        sim.bind_visual_material(path + '/visual', vispath, stage=stage)
        if i == s.band_link:  # a ring round the strand, visual only
            ring = UsdGeom.Cylinder.Define(stage, path + '/band')
            ring.CreateAxisAttr().Set('X')
            ring.CreateRadiusAttr().Set(1.15 * s.radius)
            ring.CreateHeightAttr().Set(s.band_width)
            sim.bind_visual_material(path + '/band', bandpath, stage=stage)
    first = stage.GetPrimAtPath(f'{prim_path}/seg_00')
    # Articulation root on the container Xform + a world FixedJoint on the held link = a fixed-base articulation.
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    art = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    art.CreateEnabledSelfCollisionsAttr().Set(True)
    art.CreateSolverPositionIterationCountAttr().Set(32)
    art.CreateSolverVelocityIterationCountAttr().Set(4)
    UsdGeom.Xform.Define(stage, f'{prim_path}/joints')
    half = s.pitch / 2
    for i in range(1, s.links):
        j = UsdPhysics.Joint.Define(stage, f'{prim_path}/joints/bend_{i:02d}')
        PhysxSchema.PhysxJointAPI.Apply(j.GetPrim()).CreateArmatureAttr().Set(s.armature)
        j.CreateBody0Rel().SetTargets([f'{prim_path}/seg_{i - 1:02d}'])
        j.CreateBody1Rel().SetTargets([f'{prim_path}/seg_{i:02d}'])
        j.CreateLocalPos0Attr().Set(Gf.Vec3f(half, 0., 0.))
        j.CreateLocalPos1Attr().Set(Gf.Vec3f(-half, 0., 0.))
        qx, qy, qz, qw = quat_xyzw(rots[i - 1].T @ rots[i])  # joint zero = the start (rest) shape
        j.CreateLocalRot0Attr().Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
        j.CreateLocalRot1Attr().Set(Gf.Quatf(1.))
        j.CreateCollisionEnabledAttr().Set(False)
        for axis in ('transX', 'transY', 'transZ', 'rotX', 'rotY', 'rotZ'):
            lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
            free = axis in ('rotY', 'rotZ')
            lim.CreateLowAttr().Set(-s.bend_limit if free else 1.)
            lim.CreateHighAttr().Set(s.bend_limit if free else -1.)
        for axis in ('rotY', 'rotZ'):
            d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), axis)
            d.CreateTypeAttr().Set('force')
            d.CreateStiffnessAttr().Set(s.bend_k * math.pi / 180)  # USD angular drive gains are per degree
            d.CreateDampingAttr().Set(s.bend_d * math.pi / 180)
            d.CreateTargetPositionAttr().Set(0.)
            d.CreateTargetVelocityAttr().Set(0.)
            d.CreateMaxForceAttr().Set(1.e6)
    # The clamp: the held link fixed to the world where it was spawned.
    world = UsdGeom.XformCache().GetLocalToWorldTransform(first).RemoveScaleShear()
    fj = UsdPhysics.FixedJoint.Define(stage, f'{prim_path}/joints/clamp')
    fj.CreateBody1Rel().SetTargets([first.GetPath()])
    fj.CreateLocalPos0Attr().Set(Gf.Vec3f(world.ExtractTranslation()))
    fj.CreateLocalRot0Attr().Set(Gf.Quatf(world.ExtractRotationQuat()))
    fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.))
    fj.CreateLocalRot1Attr().Set(Gf.Quatf(1.))
    return root.GetPrim()


_FCFG = None
_FSPAWN = None


def _figure_object(box):
    """A registered figure as a fixed Arena object (static colliders, no rigid body), like the kit's fixed parts."""
    global _FCFG, _FSPAWN
    _, origin, yaw = figures()[box.name]
    import isaaclab.sim as sim
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    if _FCFG is None:
        @configclass
        class FigureSpawnCfg(sim.SpawnerCfg):
            figure_name: str = ''
        _FCFG = FigureSpawnCfg
        _FSPAWN = clone(_spawn_figure)
    return Object(name=box.name, object_type=ObjectType.BASE,
                  spawner_cfg=_FCFG(func=_FSPAWN, figure_name=box.name),
                  initial_pose=Pose(position_xyz=origin, rotation_xyzw=(0., 0., math.sin(yaw / 2), math.cos(yaw / 2))))


def _spawn_figure(prim_path, cfg, *args, **kwargs):
    import isaaclab.sim as sim
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    from yam_arena.station import linear_rgb

    pieces = figures()[cfg.figure_name][0]
    stage = kwargs.get('stage') or sim.get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    translation = kwargs.get('translation', args[0] if args else None)
    orientation = kwargs.get('orientation', args[1] if len(args) > 1 else None)
    if translation is not None:
        root.AddTranslateOp().Set(Gf.Vec3d(*map(float, translation)))
    if orientation is not None:
        x, y, z, w = map(float, orientation)
        root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    matpath = f'{prim_path}/figure_material'
    mat = sim.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5, restitution=0.)
    mat.func(matpath, mat)
    looks = {}
    for i, p in enumerate(pieces):
        if p.color not in looks and not p.collide:
            looks[p.color] = f'{prim_path}/look_{len(looks):02d}'
            look = sim.PreviewSurfaceCfg(diffuse_color=linear_rgb(p.color), roughness=.6)
            look.func(looks[p.color], look)
        path = f'{prim_path}/piece_{i:02d}'
        if p.kind == 'ellipsoid' and p.collide and len(set(p.dims)) > 1:  # a convex-hull ellipsoid collider
            geom = UsdGeom.Mesh.Define(stage, path)
            pts, faces = ellipsoid_mesh(p.dims)
            geom.CreatePointsAttr().Set([Gf.Vec3f(*map(float, v)) for v in pts])
            geom.CreateFaceVertexCountsAttr().Set([3] * len(faces))
            geom.CreateFaceVertexIndicesAttr().Set([int(i) for f in faces for i in f])
            geom.AddTranslateOp().Set(Gf.Vec3d(*map(float, p.centre)))
            UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr().Set('convexHull')
        elif p.kind == 'ellipsoid':
            geom = UsdGeom.Sphere.Define(stage, path)
            uniform = len(set(p.dims)) == 1
            geom.CreateRadiusAttr().Set(float(p.dims[0]) if uniform else 1.)
            geom.AddTranslateOp().Set(Gf.Vec3d(*map(float, p.centre)))
            if not uniform:
                geom.AddScaleOp().Set(Gf.Vec3f(*map(float, p.dims)))
        elif p.kind == 'cylinder':
            geom = UsdGeom.Cylinder.Define(stage, path)
            geom.CreateAxisAttr().Set(p.axis)
            geom.CreateRadiusAttr().Set(float(p.dims[0]))
            geom.CreateHeightAttr().Set(float(p.dims[1]))
            geom.AddTranslateOp().Set(Gf.Vec3d(*map(float, p.centre)))
        else:
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr().Set(1.)
            geom.AddTranslateOp().Set(Gf.Vec3d(*map(float, p.centre)))
            geom.AddScaleOp().Set(Gf.Vec3f(*map(float, p.dims)))
        if not p.collide:
            sim.bind_visual_material(path, looks[p.color], stage=stage)
        else:  # an invisible collider; the visible shape is a separate piece
            geom.CreateVisibilityAttr().Set('invisible')
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            col = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
            col.CreateContactOffsetAttr().Set(0.002)
            col.CreateRestOffsetAttr().Set(0.)
            sim.bind_physics_material(path, matpath, stage=stage)
    return root.GetPrim()


def ellipsoid_mesh(semi, n_lat=10, n_lon=20):
    """Points and triangles of an ellipsoid with semi-axes `semi` (a convex hull collider)."""
    a, b, c = semi
    pts = [(0.0, 0.0, c), (0.0, 0.0, -c)]
    for i in range(1, n_lat):
        th = math.pi * i / n_lat
        for k in range(n_lon):
            ph = 2 * math.pi * k / n_lon
            pts.append((a * math.sin(th) * math.cos(ph), b * math.sin(th) * math.sin(ph), c * math.cos(th)))
    faces = []
    ring = lambda i, k: 2 + (i - 1) * n_lon + (k % n_lon)  # noqa: E731
    for k in range(n_lon):
        faces.append((0, ring(1, k), ring(1, k + 1)))
        faces.append((1, ring(n_lat - 1, k + 1), ring(n_lat - 1, k)))
        for i in range(1, n_lat - 1):
            faces.append((ring(i, k), ring(i + 1, k), ring(i + 1, k + 1)))
            faces.append((ring(i, k), ring(i + 1, k + 1), ring(i, k + 1)))
    return np.array(pts), faces


def _array(v):
    v = getattr(v, 'torch', v)
    return v.detach().cpu().numpy().astype(float) if hasattr(v, 'detach') else np.asarray(v, float)


def read(env, name):
    s = registry()[name][0]
    art = env.unwrapped.scene.articulations[name]
    d = art.data
    poses = _array(d.body_link_pose_w)[0]
    vel = _array(d.body_link_vel_w)[0]
    origin = _array(env.unwrapped.scene.env_origins)[0]
    index = {n: i for i, n in enumerate(d.body_names)}
    idx = [index[f'seg_{i:02d}'] for i in range(s.links)]
    p = poses[idx, :3] - origin
    q = poses[idx, 3:]
    axes = np.array([rotation(qq)[:, 0] if np.all(np.isfinite(qq)) and np.linalg.norm(qq) > 1e-8
                     else np.full(3, np.nan) for qq in q])
    lo, hi = np.nanmin(p, axis=0) - s.radius, np.nanmax(p, axis=0) + s.radius
    return RopeState(pos=(lo + hi) / 2, quat_xyzw=q[0], lin_vel=vel[idx[0], :3], ang_vel=vel[idx[0], 3:],
                     low=lo, high=hi, centres=p, axes=axes, speeds=np.linalg.norm(vel[idx, :3], axis=1),
                     pitch=s.pitch)


def _patch(scene):
    if getattr(scene, '_hair_patched', False):
        return
    old_object, old_states = scene.task_object, scene.object_states

    @functools.wraps(old_object)
    def task_object(spec):
        if getattr(spec, 'name', None) in registry():
            return _object(spec)
        if getattr(spec, 'name', None) in figures():
            return _figure_object(spec)
        return old_object(spec)

    @functools.wraps(old_states)
    def object_states(env, names):
        names = tuple(names)
        reg = registry()
        own = [n for n in names if n in reg]
        result = old_states(env, tuple(n for n in names if n not in reg))
        for n in own:
            result[n] = read(env, n)
        return result

    scene.task_object, scene.object_states = task_object, object_states
    scene._hair_patched = True


class _Loader(importlib.abc.Loader):
    def __init__(self, delegate):
        self.delegate = delegate

    def create_module(self, spec):
        fn = getattr(self.delegate, 'create_module', None)
        return fn(spec) if fn else None

    def exec_module(self, module):
        self.delegate.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    _hair_finder = True

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'yam_arena.scene':
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader)
            return spec


def install():
    """Hook yam_arena.scene (now, or when the simulator imports it) so registered strands become rope articulations."""
    if not any(getattr(f, '_hair_finder', False) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    if 'yam_arena.scene' in sys.modules:
        _patch(sys.modules['yam_arena.scene'])
