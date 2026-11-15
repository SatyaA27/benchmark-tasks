"""Passive loop closures over the unchanged tree joints kit.

Constructor drivers use degrees/metres; solve and complete_q use radians/metres,
like initial_q and every offline/runtime state. No simulator imports until spawn.
"""
from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import math
import sys
from dataclasses import dataclass, replace

import numpy as np
import yam_parts as parts
import yam_joints as joints
from yam_parts import PartState  # factory's PartState discovery convention

GAP_TOL = 1e-4
AXIS_TOL = math.radians(.1)
_AXIS_SCALE = .05  # convert the angular residual to a comparable length


@dataclass(frozen=True)
class Closure:
    """Anchors and axis in each body's part frame at q=0, never body-local."""
    a: str
    b: str
    anchor_a: tuple
    anchor_b: tuple
    axis: str = 'z'
    kind: str = 'hinge'

    def __post_init__(self):
        if self.kind not in ('hinge', 'ball'):
            raise ValueError('closure kind must be hinge or ball')
        if self.axis not in ('x', 'y', 'z'):
            raise ValueError('closure axis must be x, y or z')
        for name in ('anchor_a', 'anchor_b'):
            a = np.asarray(getattr(self, name), float)
            if a.shape != (3,) or not np.isfinite(a).all():
                raise ValueError('closure anchors must contain three finite numbers')
            object.__setattr__(self, name, tuple(a))
        if self.a == self.b:
            raise ValueError('closure must connect two different bodies')


class LinkageMechanism(joints.Mechanism):
    """Tree articulation plus excluded, undriven closing joints.

    The initial configuration selects the assembly branch. solve follows that
    branch from initial in small driver increments, using damped Gauss-Newton.
    At a branch singularity the initial configuration/extra driver selects the
    branch; a loop alone cannot distinguish physically coincident assemblies.
    """
    def __init__(self, name, pos, shapes, links, mass=None, color=parts.GREY, yaw=0,
                 friction=None, contact_offset=.002, grip=None, self_collide=False,
                 lamps=(), lamp_logic=None, *, closures=(), drivers=None, **options):
        object.__setattr__(self, 'closures', tuple(closures))
        super().__init__(name, pos, shapes, links, mass, color, yaw, friction,
                         contact_offset, grip, self_collide, lamps, lamp_logic, **options)
        self._check_closure_specs()
        if drivers is not None:
            q = self.solve({n: float(v) * (math.pi/180 if isinstance(self._link_named(n).joint, joints.Hinge)
                                          else 1) for n, v in drivers.items()})
            object.__setattr__(self, 'links', tuple(replace(l, joint=replace(l.joint, initial=
                float(drivers[l.name]) if l.name in drivers else
                q[l.name] * (180/math.pi if isinstance(l.joint, joints.Hinge) else 1))) for l in self.links))

    def _link_named(self, name):
        if name not in {l.name for l in self.links}:
            raise ValueError(f'unknown joint link {name!r}')
        return self._link(name)

    def _check_closure_specs(self):
        names = {'base', *(l.name for l in self.links)}
        for i, c in enumerate(self.closures):
            if not isinstance(c, Closure):
                raise ValueError('closures must contain Closure objects')
            if c.a not in names or c.b not in names:
                raise ValueError(f'closure {i}: bodies must name base or a defined link')

    def _check(self):
        super()._check()
        self._check_closure_specs()
        self._validate_q(self.initial_q())

    def _body_pose(self, name, transforms):
        return (np.eye(3), np.zeros(3), np.zeros(3)) if name == 'base' else transforms[name]

    def _closure_frames(self, q):
        transforms = joints.Mechanism._transforms(self, q)
        frames = []
        for c in self.closures:
            ra, ta, aa = self._body_pose(c.a, transforms)
            rb, tb, ab = self._body_pose(c.b, transforms)
            axis = 'xyz'.index(c.axis)
            frames.append((ta + ra @ (np.asarray(c.anchor_a)-aa),
                           tb + rb @ (np.asarray(c.anchor_b)-ab), ra[:, axis], rb[:, axis]))
        return frames

    def _residual(self, q):
        out = []
        for c, (a, b, u, v) in zip(self.closures, self._closure_frames(q)):
            out.extend(a-b)
            if c.kind == 'hinge':
                out.extend(_AXIS_SCALE*np.cross(u, v))
        return np.asarray(out)

    def _validate_q(self, q):
        for l in self.links:
            scale = math.pi/180 if isinstance(l.joint, joints.Hinge) else 1
            v = q[l.name]
            if not math.isfinite(v) or not l.joint.lower*scale <= v <= l.joint.upper*scale:
                raise ValueError(f'{l.name}: solved joint value {v:g} outside limits '
                                 f'[{l.joint.lower*scale:g}, {l.joint.upper*scale:g}] (rad/m)')
        for i, (c, (a, b, u, v)) in enumerate(zip(self.closures, self._closure_frames(q))):
            gap = float(np.linalg.norm(a-b))
            angle = math.atan2(float(np.linalg.norm(np.cross(u, v))), abs(float(u @ v)))
            if gap > GAP_TOL or (c.kind == 'hinge' and angle > AXIS_TOL):
                raise ValueError(f'closure {i}: residual gap {gap*1000:.6f} mm, '
                                 f'axis {math.degrees(angle):.6f} deg (limits 0.1 mm / 0.1 deg)')

    def _gauss_newton(self, q, free):
        # Dimensionless unknowns keep slide columns comparable to hinge columns.
        scales = np.array([1 if isinstance(self._link(n).joint, joints.Hinge) else .05 for n in free])
        low = np.array([self._link(n).joint.lower * (math.pi/180 if isinstance(
            self._link(n).joint, joints.Hinge) else 1) for n in free])
        high = np.array([self._link(n).joint.upper * (math.pi/180 if isinstance(
            self._link(n).joint, joints.Hinge) else 1) for n in free])
        damping = 1e-7
        for _ in range(100):
            r = self._residual(q)
            if np.linalg.norm(r) < 1e-11 or not free:
                break
            columns = []
            for n, scale in zip(free, scales):
                delta = 1e-6*scale
                plus, minus = dict(q), dict(q)
                plus[n] += delta
                minus[n] -= delta
                columns.append((self._residual(plus)-self._residual(minus))/(2e-6))
            jac = np.column_stack(columns)
            # Augmented least squares tolerates redundant planar constraints and
            # rank loss at dead center; never invert J.T @ J.
            step = np.linalg.lstsq(np.vstack((jac, math.sqrt(damping)*np.eye(len(free)))),
                                   np.r_[-r, np.zeros(len(free))], rcond=None)[0]
            step /= max(1., float(np.max(np.abs(step)))/.35)
            values = np.clip(np.array([q[n] for n in free])+scales*step, low, high)
            trial = {**q, **dict(zip(free, values))}
            if np.linalg.norm(self._residual(trial)) < np.linalg.norm(r):
                q = trial
                damping = max(1e-14, damping/4)
            else:
                damping *= 10
                if damping > 1e6:
                    break
        return q

    def solve(self, drivers):
        """Complete fixed joint values in radians/metres, from the initial branch."""
        drivers = dict(drivers)
        initial = self.initial_q()
        for n, value in drivers.items():
            l = self._link_named(n)
            scale = math.pi/180 if isinstance(l.joint, joints.Hinge) else 1
            if not math.isfinite(value) or not l.joint.lower*scale <= value <= l.joint.upper*scale:
                raise ValueError(f'{n}: driver {value} outside limits or non-finite')
        free = [l.name for l in self.links if l.name not in drivers]
        steps = max([1, *(math.ceil(abs(v-initial[n])/(math.radians(5) if isinstance(
            self._link(n).joint, joints.Hinge) else .002)) for n, v in drivers.items())])
        q = dict(initial)
        for k in range(1, steps+1):
            q.update({n: initial[n]+(v-initial[n])*k/steps for n, v in drivers.items()})
            q = self._gauss_newton(q, free)
        q.update(drivers)  # preserve exact supplied values
        self._validate_q(q)
        return q

    def complete_q(self, partial):
        """Fill missing joints; a full consistent mapping retains its exact values."""
        partial = dict(partial)
        if partial.keys() == self.initial_q().keys():
            self._validate_q(partial)
            return partial
        return self.solve(partial)

    def _transforms(self, q):
        return joints.Mechanism._transforms(self, self.complete_q(q))

    def offline_state(self, origin, yaw, q, qd=None, *, R=None):
        return super().offline_state(origin, yaw, self.complete_q(q), qd, R=R)

    def world_shapes(self, origin, R):
        return self.world_shapes_at(origin, R, self.initial_q())

    def local_aabb(self):
        return self.aabb_at((0, 0, 0), np.eye(3), self.initial_q())

    def aabb(self, origin, R):
        return self.aabb_at(origin, R, self.initial_q())

    def footprints(self, origin, R):
        out = []
        for kind, center, rot, dims in self.world_shapes(origin, R):
            shape = (parts.box((0, 0, 0), dims) if kind == 'box' else
                     parts.cyl((0, 0, 0), *dims) if kind == 'cyl' else parts.ball((0, 0, 0), *dims))
            out.extend(parts.Part('footprint', (0, 0, 0), (shape,)).footprints(center, rot))
        return tuple(out)

    def _state(self, *args, **kwargs):
        state = super()._state(*args, **kwargs)
        return self._state_type(**state.__dict__)

    @property
    def _state_type(self):
        return LinkageMechanismState


@dataclass(frozen=True)
class LinkageMechanismState(joints.MechanismState):
    def fk_error_mm(self):
        # Runtime measurements may violate closure tolerance during a solver
        # step. Compare them with raw tree FK; offline completion/validation
        # would either hide the error or raise while collecting telemetry.
        transforms = joints.Mechanism._transforms(self._mechanism, self.joints)
        return 1000 * max((np.linalg.norm(self.links[n][0] -
                          (self.origin + self.rotation() @ t))
                          for n, (_, t, _) in transforms.items() if n is not None), default=0.)

    def fk_error_deg(self):
        transforms = joints.Mechanism._transforms(self._mechanism, self.joints)
        errors = []
        for n, (r, _, _) in transforms.items():
            if n is None:
                continue
            delta = (self.rotation() @ r).T @ self.link_rotation(n)
            sine = np.linalg.norm((delta-delta.T)[[2, 0, 1], [1, 2, 0]])/2
            errors.append(math.degrees(math.atan2(sine, np.clip((np.trace(delta)-1)/2, -1, 1))))
        return max(errors, default=0.)

    def _closure_frame(self, i):
        c = self._mechanism.closures[i]
        def frame(body, anchor):
            if body == 'base':
                return self.origin+self.rotation() @ anchor, self.rotation()
            return self.link_to_world(body, anchor), self.link_rotation(body)
        a, ra = frame(c.a, c.anchor_a)
        b, rb = frame(c.b, c.anchor_b)
        return a, b, ra[:, 'xyz'.index(c.axis)], rb[:, 'xyz'.index(c.axis)]

    def closure_gap_mm(self, i):
        a, b, _, _ = self._closure_frame(i)
        return float(np.linalg.norm(a-b))*1000

    def closure_axis_deg(self, i):
        """Unsigned parallel-axis error, including for diagnostic ball frames."""
        _, _, u, v = self._closure_frame(i)
        return math.degrees(math.atan2(float(np.linalg.norm(np.cross(u, v))), abs(float(u @ v))))


def _spawn_linkage_mechanism(path, cfg, *args, **kwargs):
    import isaaclab.sim as sim_utils
    from pxr import Gf, UsdPhysics
    mech = joints._SPECS[cfg.mech_name]
    if hasattr(mech, 'meshes'):
        import yam_mesh
        prim = yam_mesh._spawn_mesh_mechanism(path, cfg, *args, **kwargs)
    else:
        prim = joints._spawn_mechanism(path, cfg, *args, **kwargs)
    stage = kwargs.get('stage') or sim_utils.get_current_stage()
    transforms = joints.Mechanism._transforms(mech, mech.initial_q())
    for i, c in enumerate(mech.closures):
        cls = UsdPhysics.RevoluteJoint if c.kind == 'hinge' else UsdPhysics.SphericalJoint
        joint = cls.Define(stage, f'{path}/joints/closure_{i}')
        joint.CreateBody0Rel().SetTargets([f'{path}/{c.a}'])
        joint.CreateBody1Rel().SetTargets([f'{path}/{c.b}'])
        ra, ta, aa = mech._body_pose(c.a, transforms)
        rb, tb, ab = mech._body_pose(c.b, transforms)
        # Use the same world joint frame on both authored FK(initial) bodies.
        # In particular localRot1 must undo their relative initial rotation.
        a = ta + ra @ (np.asarray(c.anchor_a)-aa)
        b = tb + rb @ (np.asarray(c.anchor_b)-ab)
        pivot = (a+b)/2
        for index, (r, t) in enumerate(((ra, ta), (rb, tb))):
            getattr(joint, f'CreateLocalPos{index}Attr')().Set(Gf.Vec3f(*map(float, r.T @ (pivot-t))))
            x, y, z, w = map(float, joints._quat(r.T @ ra))
            getattr(joint, f'CreateLocalRot{index}Attr')().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        if c.kind == 'hinge':
            joint.CreateAxisAttr().Set(c.axis.upper())
        joint.CreateExcludeFromArticulationAttr().Set(True)
    return prim


_CFG = _SPAWN = None
_PATCH_TOKEN = object()


def spawner_config_type(base, configclass):
    # Isaac inspects sys.modules[cls.__module__]. Define here, never in a task
    # module (Arena deliberately does not register its imported task modules).
    @configclass
    class LinkageSpawnerCfg(base):
        mech_name: str = ''
    return LinkageSpawnerCfg


def _linkage_mechanism_object(spec):
    global _CFG, _SPAWN
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    obj = joints._mechanism_object(spec)
    if _CFG is None:
        _CFG = spawner_config_type(sim_utils.SpawnerCfg, configclass)
        _SPAWN = clone(_spawn_linkage_mechanism)
    cfg = _CFG(func=_SPAWN, mech_name=spec.name)
    obj.spawner_cfg = obj.object_cfg.spawn = cfg
    parts._PART_NAMES.discard(spec.name)
    return obj


def _patch_scene(scene):
    joints._patch_scene(scene)
    if getattr(scene.task_object, '_yam_linkage_patch_token', None) is _PATCH_TOKEN:
        return
    inner = scene.task_object
    @functools.wraps(inner)
    def task_object(spec):
        if isinstance(spec, LinkageMechanism):
            return _linkage_mechanism_object(spec)
        return inner(spec)
    task_object._yam_linkage_patch_token = _PATCH_TOKEN
    scene.task_object = task_object


class _SceneFinder(importlib.abc.MetaPathFinder):
    _yam_linkage_scene_finder = True
    def __init__(self, patch):
        self.patch = patch
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'yam_arena.scene':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = parts._SceneLoader(spec.loader, self.patch)
        return spec


def install():
    finder = next((f for f in sys.meta_path if getattr(f, '_yam_linkage_scene_finder', False)), None)
    if finder is None:
        finder = _SceneFinder(_patch_scene)
    else:
        sys.meta_path.remove(finder)
        finder.patch = _patch_scene
    sys.meta_path.insert(0, finder)
    if 'yam_arena.scene' in sys.modules:
        _patch_scene(sys.modules['yam_arena.scene'])


def __getattr__(name):
    if name != 'MeshLinkageMechanism':
        raise AttributeError(name)
    import yam_mesh
    class MeshLinkageState(LinkageMechanismState, yam_mesh.MeshMechanismState):
        pass
    class MeshLinkageMechanism(LinkageMechanism, yam_mesh.MeshMechanism):
        @property
        def _state_type(self):
            return MeshLinkageState
    globals()[name] = MeshLinkageMechanism
    install()  # ensure the linkage wrapper is outside the newly imported mesh kit
    return MeshLinkageMechanism


install()
__all__ = ['Closure', 'LinkageMechanism', 'LinkageMechanismState', 'MeshLinkageMechanism', 'install']
