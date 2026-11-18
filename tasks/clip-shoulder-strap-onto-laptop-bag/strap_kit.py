"""strap_kit: a flat webbing strap with swivel trigger snap hooks, and a D-ring on a short tab (one module, reusable).

What it models
- Strap: flat woven webbing as a chain of short flat box links (one floating articulation). Each pair of links is
  joined by a D6 joint with translation locked and three sprung rotations: flat bend (about the width axis, soft,
  +-85 deg), edge bend (about the thickness axis, stiff, +-6 deg) and twist (about the length axis, +-40 deg), so the
  strap folds easily one way, resists bending edge-wise and carries twist: a twisted strap stays twisted and can be
  measured (twist_deg sums the twist between neighbouring links from their poses). Links can be marked doubled (an
  adjuster's two-layer run: twice the thickness and mass).
- Swivel trigger snap hook (SnapHook) on either end, in the same articulation: webbing end -> eyelet (hinge about
  the eyelet's bar: the webbing loop folds over it) -> hook body (swivel: revolute about the hook's long axis with
  stiction) -> gate (revolute about the hook's normal, sprung shut against its stop with a preload; it opens INWARD
  when the trigger tab next to its pivot is squeezed toward the hook's back, or when its outer face is pushed).
  Hook frame: u (x) from the eyelet bar toward the nose, v (y) from the back toward the gate side, w (z) normal.
- D-ring on a tab (DRing): a fixed-base articulation: the tab (fixed to the world, i.e. to a bag modelled rigid and
  fixed) and the ring, a D of round wire (capsules), hinged on its straight bar (sprung toward square, damped).
  Ring frame: z along the bar, x the direction the ring points when square; the D lies in the x-z plane.
- Scoring geometry (NumPy, offline): hook_loop() is the hook's closed centre-line (gate shut); ring_wire() the ring's
  closed centre-line; linked() counts how often the ring's wire passes through the hook's loop (odd = the hook is
  round the wire). StrapState.gate_deg(end) (0 = shut, negative = open), twist_deg(), hook_frame(end).

Tuned numbers (first use: laptop-bag shoulder strap, Nov 18 2026; all estimates unless a task cites a source)
- light links need joint armature: webbing D6 5e-6 kg m^2 (cord recipe), eyelet 2e-6, swivel 2e-6, gate 2e-6;
  body damping 0.5/2.0; strap solver 32/4 iterations; contact offset 1 mm, rest 0.
- webbing 38 x 1.6 mm, 30 mm links, 30 g/m; flat EI 3e-4 N m^2 (k = EI/pitch), twist k ~ flat k, edge k 0.2 N m/rad
  with a +-6 deg limit (edge stiffness is ~600x flat; a limit stands in for it).
- hook: BIKICOCO 1-1/2 in swivel trigger snap geometry: 58 mm overall, 38 mm eyelet, 8 mm opening (gate 33 deg open),
  15 g; gate spring 0.0226 N m/rad toward +26.6 deg past shut (about 3 N at the trigger to start, 5 N fully open).
- ring: 38 mm inside width, 22 mm inside depth, 4 mm wire, 11 g; hinge spring 0.003 N m/rad, damping 5e-4.

Use: import strap_kit; strap_kit.install(); put Strap / DRing specs in OBJECTS; read states from the objects dict
(StrapState / RingState). Per-episode jitter: strap_kit.set_jitter(name, fn) with fn(rng) -> (dx, dy, dyaw_rad,
{joint prim name: angle rad}) applied by the object's own reset event (seeded by its reset count).
Only NumPy and the harness's task_file are imported before the simulator starts.
"""
import functools
import importlib.abc
import importlib.machinery
import math
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena.task_file import Box, ObjectState


# ------------------------------------------------------------------------------------------------ small maths
def quat_to_matrix(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                     (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def matrix_to_quat(R):
    m = np.asarray(R, float)
    K = np.array(((m[0, 0] - m[1, 1] - m[2, 2], m[0, 1] + m[1, 0], m[0, 2] + m[2, 0], m[2, 1] - m[1, 2]),
                  (m[0, 1] + m[1, 0], m[1, 1] - m[0, 0] - m[2, 2], m[1, 2] + m[2, 1], m[0, 2] - m[2, 0]),
                  (m[0, 2] + m[2, 0], m[1, 2] + m[2, 1], m[2, 2] - m[0, 0] - m[1, 1], m[1, 0] - m[0, 1]),
                  (m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], np.trace(m)))) / 3
    q = np.linalg.eigh(K)[1][:, -1]
    return q if q[3] >= 0 else -q


def rot_axis(axis, angle):
    v = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array(((0, -v[2], v[1]), (v[2], 0, -v[0]), (-v[1], v[0], 0)))
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def rot_z(a):
    return rot_axis((0, 0, 1), a)


def twist_about_x(R_rel):
    """Twist angle (rad) of a relative rotation about its local x axis (swing-twist decomposition)."""
    x, _y, _z, w = matrix_to_quat(R_rel)
    return 2 * math.atan2(x, w) if (abs(x) + abs(w)) > 1e-12 else 0.0


# ------------------------------------------------------------------------------------------------ the hook
@dataclass(frozen=True)
class SnapHook:
    """Swivel trigger snap hook geometry in the hook frame (u = x toward the nose, v = y toward the gate, w = z)."""
    length: float = .058  # overall, eyelet bar to nose (listing)
    eyelet_inner: float = .038  # strap width the eyelet takes (listing)
    bar: float = .0035  # wire/bar section of eyelet and body (estimate)
    eyelet_depth: float = .007  # inner, along u (estimate)
    body_width: float = .024  # outer, across v (estimate)
    thick: float = .006  # body thickness along w (estimate)
    gate_thick: float = .005
    tip_u: float = .047  # where the nose's tip ends on the gate side; the gate spans base -> tip
    swivel_u: float = .017  # swivel joint (eyelet boss | hook barrel)
    pivot: tuple = (.0265, .01025)  # gate pivot (u, v)
    trigger: tuple = (.0300, .0135)  # trigger tab centre (u, v), a round thumb rest proud of the gate
    trigger_r: float = .0025
    open_deg: float = 33.  # gate stop: about 8 mm opening (listing: opens 8 mm)
    spring: float = .0226  # N m/rad
    preload_deg: float = 26.6  # drive target past shut
    gate_damping: float = 2.e-5
    swivel_friction: float = .002  # N m stiction (estimate; cheap swivels turn stiffly)
    masses: tuple = (.0045, .0085, .002)  # eyelet, body, gate = 15 g (listing: 150 g per 10)
    friction: tuple = (.6, .5)
    color: tuple = (.16, .16, .17)
    armature: float = 2.e-6

    # each shape: (kind, centre (u,v,w), dims) ; box dims (du, dv, dw); cyl dims (radius, height, axis)
    def eyelet_shapes(self):
        b, wi, d = self.bar, self.eyelet_inner, self.eyelet_depth
        span = wi + 2 * b
        front = b + d  # front bar centre u (webbing bar centre at u = 0)
        side_v = wi / 2 + b / 2
        boss0, boss1 = front + b / 2, self.swivel_u
        return (("box", (0., 0., 0.), (b, span, b)),
                ("box", (front / 2, side_v, 0.), (front, b, b)),
                ("box", (front / 2, -side_v, 0.), (front, b, b)),
                ("box", (front, 0., 0.), (b, span, b)),
                ("cyl", ((boss0 + boss1) / 2, 0., 0.), (b, boss1 - boss0, "x")))

    def body_shapes(self):
        b, W, t, L = self.bar, self.body_width, self.thick, self.length
        u0 = self.swivel_u + .004  # barrel end
        base_end = u0 + b
        nose0 = L - b
        return (("cyl", ((self.swivel_u + u0) / 2, 0., 0.), (b, u0 - self.swivel_u, "x")),
                ("box", ((u0 + base_end) / 2, 0., 0.), (b, W, t)),  # base bar
                ("box", ((u0 + nose0) / 2, -(W / 2 - b / 2), 0.), (nose0 - u0, b, t)),  # back
                ("box", ((nose0 + L) / 2, 0., 0.), (b, W, t)),  # nose
                ("box", ((self.tip_u + nose0) / 2, W / 2 - b / 2, 0.), (nose0 - self.tip_u, b, t)))  # tip

    def gate_shapes(self):
        """In the hook frame at the shut position."""
        b, W = self.bar, self.body_width
        u0 = self.swivel_u + .004 + b + .0005
        u1 = self.tip_u - .0005
        return (("box", ((u0 + u1) / 2, W / 2 - b / 2, 0.), (u1 - u0, b, self.gate_thick)),
                ("cyl", (self.trigger[0], self.trigger[1], 0.), (self.trigger_r, self.gate_thick, "z")))

    def loop_uv(self, n_gate=4):
        """Closed centre-line of the shut hook (u, v) in the hook frame, for the linking test."""
        b, W, L = self.bar, self.body_width, self.length
        u0 = self.swivel_u + .004 + b / 2
        vb, vg, un = -(W / 2 - b / 2), W / 2 - b / 2, L - b / 2
        pts = [(u0, vb), (un, vb), (un, vg), (self.tip_u, vg)]
        pts += [(self.tip_u + (u0 - self.tip_u) * k / n_gate, vg) for k in range(1, n_gate + 1)]
        return np.array(pts)

    def interior_uv(self):
        """(u0, u1, v0, v1) of the hook's inside (where a captured wire sits)."""
        b, W, L = self.bar, self.body_width, self.length
        return (self.swivel_u + .004 + b, L - b, -(W / 2 - b), W / 2 - b)


def hook_loop(hook, R, p):
    """World polygon of a hook's shut loop: hook frame (R, p)."""
    uv = hook.loop_uv()
    return p + np.c_[uv, np.zeros(len(uv))] @ R.T


# ------------------------------------------------------------------------------------------------ the strap
@dataclass(frozen=True)
class Strap(Box):
    """pos = centre of the webbing (world), yaw = heading of the webbing's +x (end 'b'); laid straight and flat.
    End 'a' is at -x (link 0), end 'b' at +x (last link). Hooks: SnapHook at the ends named in `hook_ends`."""
    _is_strap_kit = True
    links: int = 20
    pitch: float = .030
    width: float = .038
    thick: float = .0016
    kg_per_m: float = .030
    doubled: tuple = ()  # link indices of a two-layer run (twice thickness and mass)
    flat_k: float = .010  # N m/rad per joint (EI 3e-4 / 0.03)
    twist_k: float = .010
    edge_k: float = .2
    flat_limit: float = 85.
    twist_limit: float = 40.
    edge_limit: float = 6.
    damp_ratio: float = .1  # joint damping = ratio * stiffness (s)
    armature: float = 5.e-6
    friction: tuple = (.7, .6)
    contact_offset: float = .001
    lin_damp: float = .5
    ang_damp: float = 2.
    hook: SnapHook = SnapHook()
    hook_ends: tuple = ("a", "b")
    flip_ends: tuple = ()  # ends whose hook lies on its other face at swivel 0 (turned half a turn about u)
    eye_limit: float = 100.  # webbing loop fold about the eyelet bar, deg
    eye_damping: float = 2.e-5

    @property
    def free(self):
        return False  # articulation: opt out of the harness's rigid-root jitter (jitter via set_jitter)

    # offline task checkers treat kit-less specs as approximate assets placed at xy with a cube of `extent`
    @property
    def xy(self):
        return (float(self.pos[0]), float(self.pos[1]))

    @property
    def extent(self):
        return float(self.hook.thick)

    # ---- layout (container frame: x along the webbing, z up, origin at the webbing's centre)
    def web_names(self):
        return [f"web_{i:02d}" for i in range(self.links)]

    def body_names(self):
        out = self.web_names()
        for e in self.hook_ends:
            out += [f"eye_{e}", f"hook_{e}", f"gate_{e}"]
        return out

    def web_length(self):
        return self.links * self.pitch

    def link_thick(self, i):
        return self.thick * (2 if i in self.doubled else 1)

    def link_mass(self, i):
        return self.kg_per_m * self.pitch * (2 if i in self.doubled else 1)

    def hook_frame_local(self, end):
        """(R, p) of the hook frame (eyelet bar centre) in the container frame, straight layout."""
        half = self.web_length() / 2
        if end == "b":
            return np.eye(3), np.array((half, 0., 0.))
        return rot_z(math.pi), np.array((-half, 0., 0.))

    def container_pose(self):
        """World (R, p) of the container (webbing centre, at the eyelet's mid-plane height)."""
        return rot_z(self.yaw), np.asarray(self.pos, float)


@dataclass(frozen=True)
class StrapState(ObjectState):
    poses: dict = field(default_factory=dict)  # body name -> (p (3,), R (3,3)) world, env origin removed
    speeds: dict = field(default_factory=dict)
    spec: object = None

    def R(self, body):
        return self.poses[body][1]

    def p(self, body):
        return self.poses[body][0]

    def hook_frame(self, end):
        return self.R(f"hook_{end}"), self.p(f"hook_{end}") - self.R(f"hook_{end}") @ np.array(
            (self.spec.hook.swivel_u, 0., 0.))

    def gate_deg(self, end):
        """Gate angle about the hook's normal: 0 shut, negative open (inward)."""
        Rh, Rg = self.R(f"hook_{end}"), self.R(f"gate_{end}")
        rel = Rh.T @ Rg
        return math.degrees(math.atan2(rel[1, 0], rel[0, 0]))

    def swivel_deg(self, end):
        rel = self.R(f"eye_{end}").T @ self.R(f"hook_{end}")
        return math.degrees(twist_about_x(rel))

    def twist_deg(self):
        """Total webbing twist (deg) from end a to end b: sum of the twist between neighbouring links."""
        names = self.spec.web_names()
        return math.degrees(sum(twist_about_x(self.R(a).T @ self.R(b)) for a, b in zip(names, names[1:])))

    def centreline(self):
        return np.array([self.p(n) for n in self.spec.web_names()])

    def finite(self):
        return all(np.all(np.isfinite(p)) and np.all(np.isfinite(R)) for p, R in self.poses.values())

    def max_speed(self):
        return max(self.speeds.values()) if self.speeds else 0.0


# ------------------------------------------------------------------------------------------------ the D-ring
@dataclass(frozen=True)
class DRing(Box):
    """pos = centre of the ring's straight bar (world); yaw = direction the ring points when square (world yaw of
    ring x); the bar is vertical (ring z = world z). The tab is fixed to the world (the bag is rigid and fixed)."""
    _is_strap_kit = True
    inner_width: float = .038
    inner_depth: float = .022
    wire: float = .004
    ring_mass: float = .011
    segments: int = 10
    spring: float = .003  # N m/rad toward square
    damping: float = 5.e-4
    limit: float = 75.
    armature: float = 2.e-6
    friction: tuple = (.5, .4)
    contact_offset: float = .001
    tab: tuple = (.011, .004, .026)  # tab box (along ring x behind the bar, thickness, height)
    tab_color: tuple = (.10, .09, .09)
    ring_color: tuple = (.55, .56, .58)

    @property
    def free(self):
        return False

    @property
    def xy(self):
        return (float(self.pos[0]), float(self.pos[1]))

    @property
    def extent(self):
        return float(self.inner_width + 2 * self.wire)

    def body_names(self):
        return ["tab", "ring"]

    def wire_local(self):
        """Closed centre-line of the D in the ring frame (bar at x = 0 along z; D in the x-z plane)."""
        h = (self.inner_width + self.wire) / 2
        d = self.inner_depth + self.wire / 2 + self.wire / 2
        phis = np.linspace(0., math.pi, self.segments + 1)
        curve = [(d * math.sin(f), 0., h * math.cos(f)) for f in phis]
        return np.array(curve + [(0., 0., h)])  # last segment runs back up the straight bar

    def apex_local(self):
        return np.array((self.inner_depth + self.wire, 0., 0.))


@dataclass(frozen=True)
class RingState(ObjectState):
    ring_p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ring_R: np.ndarray = field(default_factory=lambda: np.eye(3))
    tab_R: np.ndarray = field(default_factory=lambda: np.eye(3))
    spec: object = None

    def wire(self):
        return self.ring_p + self.spec.wire_local() @ self.ring_R.T

    def apex(self):
        return self.ring_p + self.ring_R @ self.spec.apex_local()

    def angle_deg(self):
        rel = self.tab_R.T @ self.ring_R
        return math.degrees(math.atan2(rel[1, 0], rel[0, 0]))


def ring_state_at(spec, angle_deg=0.0, origin=(0., 0., 0.)):
    """Offline RingState of a DRing at a hinge angle (deg) about its bar."""
    Rt = rot_z(spec.yaw)
    Rr = Rt @ rot_z(math.radians(angle_deg))
    p = np.asarray(spec.pos, float) + np.asarray(origin, float)
    pts = p + spec.wire_local() @ Rr.T
    return RingState(pos=pts.mean(0), quat_xyzw=matrix_to_quat(Rr), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                     low=pts.min(0), high=pts.max(0), ring_p=p, ring_R=Rr, tab_R=Rt, spec=spec)


# ------------------------------------------------------------------------------------------------ linking
def _point_in_poly(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def crossings(hook, R, p, wire_world):
    """How many times the closed wire polyline passes through the hook's shut loop (hook frame R, p)."""
    local = (np.asarray(wire_world, float) - p) @ R  # into the hook frame
    poly = hook.loop_uv()
    n, count = len(local), 0
    for i in range(n):
        a, b = local[i], local[(i + 1) % n]
        if (a[2] > 0) == (b[2] > 0) or a[2] == b[2]:
            continue
        s = a[2] / (a[2] - b[2])
        q = a + s * (b - a)
        if _point_in_poly(q[:2], poly):
            count += 1
    return count


def linked(hook, R, p, wire_world):
    return crossings(hook, R, p, wire_world) % 2 == 1


# ------------------------------------------------------------------------------------------------ registry
def specs():
    scene = sys.modules.get("yam_arena.scene")
    if scene is None:
        return {}
    if not hasattr(scene, "_strap_kit_specs"):
        scene._strap_kit_specs = {}
    return scene._strap_kit_specs


def _jitters():
    scene = sys.modules.get("yam_arena.scene")
    if scene is None:
        return {}
    if not hasattr(scene, "_strap_kit_jitter"):
        scene._strap_kit_jitter = {"fn": {}, "count": {}}
    return scene._strap_kit_jitter


def set_jitter(name, fn):
    """fn(rng) -> (dx, dy, dyaw, {joint prim name: rad}); applied after each reset of object `name`."""
    _PENDING[name] = fn
    if "yam_arena.scene" in sys.modules:
        _jitters()["fn"][name] = fn


_PENDING = {}
_CFG = None
_SPAWN = None


def _object(spec):
    global _CFG, _SPAWN
    import isaaclab.sim as sim
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    if _CFG is None:
        @configclass
        class StrapKitSpawnCfg(sim.SpawnerCfg):
            kit_name: str = ""
        _CFG = StrapKitSpawnCfg
        _SPAWN = clone(_spawn)

    class StrapKitObject(Object):
        def _build_reset_event(self):
            return EventTermCfg(func=_reset, mode="reset",
                                params={"asset_cfg": SceneEntityCfg(self.name),
                                        "pose": self.get_initial_pose(), "velocity": None})

    specs()[spec.name] = spec
    for k, v in list(_PENDING.items()):
        _jitters()["fn"].setdefault(k, v)
    return StrapKitObject(name=spec.name, object_type=ObjectType.ARTICULATION,
                          spawner_cfg=_CFG(func=_SPAWN, kit_name=spec.name),
                          initial_pose=Pose(position_xyz=tuple(spec.pos),
                                            rotation_xyzw=(0., 0., math.sin(spec.yaw / 2), math.cos(spec.yaw / 2))))


def _reset(env, env_ids, asset_cfg, pose, velocity=None):
    """Reset to the authored layout, then this object's seeded jitter. A strap's root pose is written for its
    actual root link (PhysX picks the root of a floating chain): container pose (+ jitter) composed with that
    link's authored pose; joints to their defaults (+ jitter), velocities zero. A ring uses Arena's reset (its tab,
    the root, sits at the container origin)."""
    import torch
    name = asset_cfg.name
    spec = specs()[name]
    art = env.scene[name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    jit = _jitters()
    k = jit["count"].get(name, 0)
    jit["count"][name] = k + 1
    fn = jit["fn"].get(name)
    try:
        dx, dy, dyaw, joints = fn(np.random.default_rng(7919 + 104729 * k + sum(map(ord, name)))) if fn else \
            (0., 0., 0., {})
    except Exception as e:  # noqa: BLE001
        print(f"STRAPKIT_JITTER {name} fn failed: {e!r}", flush=True)
        dx, dy, dyaw, joints = 0., 0., 0., {}
    if not isinstance(spec, Strap):
        from isaaclab_arena.terms.events import reset_articulation_pose_and_joints
        reset_articulation_pose_and_joints(env, env_ids, asset_cfg, pose, velocity)
    else:
        root = list(art.data.body_names)[0]
        pl, Rl = straight_poses(spec, np.eye(3), np.zeros(3))[root]
        Rc = rot_z(spec.yaw + dyaw)
        pc = np.asarray(spec.pos, float) + np.array((dx, dy, 0.))
        R, p = Rc @ Rl, pc + Rc @ pl
        q = matrix_to_quat(R)
        origin = env.scene.env_origins[env_ids]
        rp = torch.zeros((len(env_ids), 7), dtype=origin.dtype, device=origin.device)
        rp[:, :3] = origin + torch.tensor(p, dtype=origin.dtype, device=origin.device)
        rp[:, 3:7] = torch.tensor(q, dtype=origin.dtype, device=origin.device)
        art.write_root_pose_to_sim_index(root_pose=rp, env_ids=env_ids)
        art.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(env_ids), 6), dtype=origin.dtype,
                                                                        device=origin.device), env_ids=env_ids)
    try:
        jn = list(art.joint_names)
        qj = art.data.default_joint_pos.torch[env_ids].clone()
        for dof, v in (joints or {}).items():
            if dof in jn:
                qj[:, jn.index(dof)] = float(v)
        art.write_joint_position_to_sim_index(position=qj, env_ids=env_ids)
        art.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(qj), env_ids=env_ids)
        print(f"STRAPKIT_RESET {name} reset={k} root={list(art.data.body_names)[0]} dx={dx * 1e3:.1f}mm "
              f"dy={dy * 1e3:.1f}mm dyaw={math.degrees(dyaw):.2f}deg "
              f"joints={ {j: round(math.degrees(v), 2) for j, v in (joints or {}).items()} }", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"STRAPKIT_RESET {name} joints failed: {e!r}", flush=True)


# ------------------------------------------------------------------------------------------------ USD authoring
def _xf(stage, path, p, R):
    from pxr import Gf, UsdGeom
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, p)))
    qx, qy, qz, qw = matrix_to_quat(R)
    x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(qw), Gf.Vec3d(float(qx), float(qy), float(qz))))
    return x


def _body(stage, path, p, R, mass, ctx, damp=(.5, 2.), iters=(32, 4)):
    from pxr import Gf, PhysxSchema, UsdPhysics
    prim = _xf(stage, path, p, R).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr().Set(float(mass))
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateMaxDepenetrationVelocityAttr().Set(1.)
    rb.CreateLinearDampingAttr().Set(damp[0])
    rb.CreateAngularDampingAttr().Set(damp[1])
    rb.CreateSolverPositionIterationCountAttr().Set(iters[0])
    rb.CreateSolverVelocityIterationCountAttr().Set(iters[1])
    return prim


def _shape(stage, path, shape, mat, look, offset, R_local=None):
    """Author one collider (+ its visual) from (kind, centre, dims) in the body frame."""
    import isaaclab.sim as sim
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    kind, c, dims = shape
    if kind == "box":
        g = UsdGeom.Cube.Define(stage, path)
        g.CreateSizeAttr(1.)
        g.AddTranslateOp().Set(Gf.Vec3d(*map(float, c)))
        if R_local is not None:
            qx, qy, qz, qw = matrix_to_quat(R_local)
            g.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(qw), Gf.Vec3d(float(qx), float(qy), float(qz))))
        g.AddScaleOp().Set(Gf.Vec3f(*map(float, dims)))
    elif kind == "cyl":
        r, h, axis = dims
        g = UsdGeom.Cylinder.Define(stage, path)
        g.CreateRadiusAttr(float(r))
        g.CreateHeightAttr(float(h))
        g.CreateAxisAttr(axis.upper())
        g.AddTranslateOp().Set(Gf.Vec3d(*map(float, c)))
    else:  # capsule from a to b: dims = (radius, a (3,), b (3,))
        r, a, b = dims
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        L = float(np.linalg.norm(d))
        g = UsdGeom.Capsule.Define(stage, path)
        g.CreateAxisAttr("X")
        g.CreateRadiusAttr(float(r))
        g.CreateHeightAttr(max(L, 1e-5))
        g.AddTranslateOp().Set(Gf.Vec3d(*map(float, (a + b) / 2)))
        x = d / max(L, 1e-9)
        ref = np.array((0., 0., 1.)) if abs(x[2]) < .9 else np.array((0., 1., 0.))
        y = np.cross(ref, x)
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
        qx, qy, qz, qw = matrix_to_quat(np.c_[x, y, z])
        g.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(qw), Gf.Vec3d(float(qx), float(qy), float(qz))))
    prim = g.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    col.CreateContactOffsetAttr().Set(float(offset))
    col.CreateRestOffsetAttr().Set(0.)
    sim.bind_physics_material(path, mat, stage=stage)
    sim.bind_visual_material(path, look, stage=stage)


def _materials(stage, root, friction, colors, tag="material"):
    import isaaclab.sim as sim
    from yam_arena.station import linear_rgb
    mat = f"{root}/{tag}"
    m = sim.RigidBodyMaterialCfg(static_friction=friction[0], dynamic_friction=friction[1], restitution=0.)
    m.func(mat, m)
    looks = {}
    for k, c in colors.items():
        p = f"{root}/look_{k}"
        lk = sim.PreviewSurfaceCfg(diffuse_color=linear_rgb(c), roughness=.6)
        lk.func(p, lk)
        looks[k] = p
    return mat, looks


def _limit(prim, axis, lo, hi):
    from pxr import UsdPhysics
    lim = UsdPhysics.LimitAPI.Apply(prim, axis)
    lim.CreateLowAttr().Set(float(lo))
    lim.CreateHighAttr().Set(float(hi))


def _drive(prim, axis, k_per_rad, d_per_rad, target_deg=0.):
    from pxr import UsdPhysics
    d = UsdPhysics.DriveAPI.Apply(prim, axis)
    d.CreateTypeAttr().Set("force")
    d.CreateStiffnessAttr().Set(float(k_per_rad) * math.pi / 180)  # USD angular gains are per degree
    d.CreateDampingAttr().Set(float(d_per_rad) * math.pi / 180)
    d.CreateTargetPositionAttr().Set(float(target_deg))
    d.CreateTargetVelocityAttr().Set(0.)
    d.CreateMaxForceAttr().Set(1.e6)


def _joint_frames(j, stage, b0, b1, p0, q0, p1, q1):
    from pxr import Gf
    j.CreateBody0Rel().SetTargets([b0])
    j.CreateBody1Rel().SetTargets([b1])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, p0)))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(*map(float, p1)))
    for attr, R in ((j.CreateLocalRot0Attr(), q0), (j.CreateLocalRot1Attr(), q1)):
        qx, qy, qz, qw = matrix_to_quat(R)
        attr.Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
    j.CreateCollisionEnabledAttr().Set(False)


def _revolute(stage, path, b0, b1, p0, R0, p1, R1, axis, lo, hi, k=0., d=0., target=0., armature=0., stiction=0.):
    from pxr import PhysxSchema, Sdf, UsdPhysics
    j = UsdPhysics.RevoluteJoint.Define(stage, path)
    _joint_frames(j, stage, b0, b1, p0, R0, p1, R1)
    j.CreateAxisAttr().Set(axis)
    if lo is not None:
        j.CreateLowerLimitAttr().Set(float(lo))
        j.CreateUpperLimitAttr().Set(float(hi))
    if k > 0 or d > 0:
        _drive(j.GetPrim(), "angular", k, d, target)
    if armature > 0:
        PhysxSchema.PhysxJointAPI.Apply(j.GetPrim()).CreateArmatureAttr().Set(float(armature))
    if stiction > 0:
        prim = j.GetPrim()
        prim.ApplyAPI("PhysxJointAxisAPI", "angular")
        for eff, v in (("staticFrictionEffort", stiction), ("dynamicFrictionEffort", .8 * stiction)):
            name = f"physxJointAxis:angular:{eff}"
            a = prim.GetAttribute(name) or prim.CreateAttribute(name, Sdf.ValueTypeNames.Float)
            a.Set(float(v))
    return j


def _spawn(prim_path, cfg, *args, **kwargs):
    import isaaclab.sim as sim
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    s = specs()[cfg.kit_name]
    stage = kwargs.get("stage") or sim.get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    translation = kwargs.get("translation", args[0] if args else None)
    orientation = kwargs.get("orientation", args[1] if len(args) > 1 else None)
    if translation is not None:
        root.AddTranslateOp().Set(Gf.Vec3d(*map(float, translation)))
    if orientation is not None:
        x, y, z, w = map(float, orientation)
        root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    art = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    art.CreateEnabledSelfCollisionsAttr().Set(isinstance(s, Strap))
    art.CreateSolverPositionIterationCountAttr().Set(32)
    art.CreateSolverVelocityIterationCountAttr().Set(4)
    UsdGeom.Xform.Define(stage, f"{prim_path}/joints")
    if isinstance(s, Strap):
        _spawn_strap(stage, prim_path, s)
    else:
        _spawn_ring(stage, prim_path, s)
    return root.GetPrim()


def _hook_shapes_in(hook, which, R, p):
    """Hook-frame shapes moved into a body frame given the body's pose (R, p) in the hook frame."""
    out = []
    src = {"eye": hook.eyelet_shapes(), "hook": hook.body_shapes(), "gate": hook.gate_shapes()}[which]
    for kind, c, dims in src:
        out.append((kind, tuple(R.T @ (np.asarray(c) - p)), dims))
    return out


def _spawn_strap(stage, root, s):
    I3 = np.eye(3)
    mat, looks = _materials(stage, root, s.friction, {"web": s.color, "metal": s.hook.color})
    hmat, _ = _materials(stage, root, s.hook.friction, {}, tag="hook_material")
    for i in range(s.links):
        path = f"{root}/web_{i:02d}"
        c = np.array(((i + .5) * s.pitch - s.web_length() / 2, 0., 0.))
        _body(stage, path, c, I3, s.link_mass(i), None, (s.lin_damp, s.ang_damp))
        t = s.link_thick(i)
        _shape(stage, f"{path}/box", ("box", (0., 0., 0.), (s.pitch, s.width, t)), mat, looks["web"], s.contact_offset)
        _set_mass_props(stage, path, s.link_mass(i), [("box", (0., 0., 0.), (s.pitch, s.width, t))])
    h = s.pitch / 2
    for i in range(1, s.links):
        from pxr import PhysxSchema, UsdPhysics
        j = UsdPhysics.Joint.Define(stage, f"{root}/joints/wj_{i:02d}")
        _joint_frames(j, stage, f"{root}/web_{i - 1:02d}", f"{root}/web_{i:02d}", (h, 0, 0), I3, (-h, 0, 0), I3)
        PhysxSchema.PhysxJointAPI.Apply(j.GetPrim()).CreateArmatureAttr().Set(s.armature)
        for ax in ("transX", "transY", "transZ"):
            _limit(j.GetPrim(), ax, 1., -1.)
        for ax, lim, k in (("rotX", s.twist_limit, s.twist_k), ("rotY", s.flat_limit, s.flat_k),
                           ("rotZ", s.edge_limit, s.edge_k)):
            _limit(j.GetPrim(), ax, -lim, lim)
            _drive(j.GetPrim(), ax, k, s.damp_ratio * k)
    hk = s.hook
    for e in s.hook_ends:
        Rh, ph = s.hook_frame_local(e)  # hook frame in the container
        web = f"{root}/web_{0 if e == 'a' else s.links - 1:02d}"
        # bodies: eyelet origin at the bar centre; hook body origin at the swivel; gate origin at its pivot
        Fx = rot_axis((1, 0, 0), math.pi) if e in s.flip_ends else I3
        Rf = Rh @ Fx  # hook body and gate (turned about u when flipped)
        eye_p, hook_p = ph, ph + Rh @ np.array((hk.swivel_u, 0., 0.))
        gate_p = ph + Rf @ np.array((hk.pivot[0], hk.pivot[1], 0.))
        for name, P, which, local, mass, Rb in (("eye", eye_p, "eye", np.zeros(3), hk.masses[0], Rh),
                                                ("hook", hook_p, "hook", np.array((hk.swivel_u, 0., 0.)), hk.masses[1],
                                                 Rf),
                                                ("gate", gate_p, "gate", np.array((hk.pivot[0], hk.pivot[1], 0.)),
                                                 hk.masses[2], Rf)):
            path = f"{root}/{name}_{e}"
            _body(stage, path, P, Rb, mass, None, (s.lin_damp, s.ang_damp))
            shp_list = _hook_shapes_in(hk, which, I3, local)
            for k, shp in enumerate(shp_list):
                _shape(stage, f"{path}/s{k}", shp, hmat, looks["metal"], s.contact_offset)
            _set_mass_props(stage, path, mass, shp_list)
        # webbing end -> eyelet: hinge about the eyelet's bar (hook v axis), at the bar centre
        sign = -1. if e == "a" else 1.
        Rj = Rh  # joint frame = hook frame (axis Y = bar)
        _revolute(stage, f"{root}/joints/eye_{e}", web, f"{root}/eye_{e}", (sign * h, 0., 0.), Rj, (0., 0., 0.), I3,
                  "Y", -s.eye_limit, s.eye_limit, d=s.eye_damping, armature=hk.armature)
        _revolute(stage, f"{root}/joints/swivel_{e}", f"{root}/eye_{e}", f"{root}/hook_{e}",
                  (hk.swivel_u, 0., 0.), I3, (0., 0., 0.), Fx, "X", None, None, d=2.e-6, armature=hk.armature,
                  stiction=hk.swivel_friction)
        lo, hi = -hk.open_deg, 0.
        _revolute(stage, f"{root}/joints/gate_{e}", f"{root}/hook_{e}", f"{root}/gate_{e}",
                  (hk.pivot[0] - hk.swivel_u, hk.pivot[1], 0.), I3, (0., 0., 0.), I3, "Z", lo, hi,
                  k=hk.spring, d=hk.gate_damping, target=hk.preload_deg, armature=hk.armature)


def mass_props(shapes):
    """(centre of mass, bounding-box dims) of (kind, centre, dims) shapes at uniform density."""
    vols, cents, lo, hi = [], [], [], []
    for kind, c, dims in shapes:
        if kind == "box":
            v, c = float(np.prod(dims)), np.asarray(c, float)
            half = np.asarray(dims, float) / 2
        elif kind == "cyl":
            r, h, ax = dims
            v, c = math.pi * r * r * h, np.asarray(c, float)
            half = np.full(3, r)
            half["xyz".index(ax.lower())] = h / 2
        else:
            r, a, b = dims
            a, b = np.asarray(a, float), np.asarray(b, float)
            v, c = math.pi * r * r * np.linalg.norm(b - a) + 4 / 3 * math.pi * r ** 3, (a + b) / 2
            half = np.abs(b - a) / 2 + r
        vols.append(v)
        cents.append(c)
        lo.append(c - half)
        hi.append(c + half)
    com = np.average(np.array(cents), axis=0, weights=vols)
    return com, np.max(hi, 0) - np.min(lo, 0)


def _set_mass_props(stage, path, m, shapes):
    """Centre of mass at the shapes' centroid, box inertia of their bounds (tiny links get sane values)."""
    from pxr import Gf, UsdPhysics
    com, (a, b, c) = mass_props(shapes)
    I = (m * (b * b + c * c) / 12, m * (a * a + c * c) / 12, m * (a * a + b * b) / 12)
    mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(path))
    mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(*map(float, com)))
    mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, I)))


def _spawn_ring(stage, root, s):
    from pxr import Gf, UsdGeom, UsdPhysics
    I3 = np.eye(3)
    mat, looks = _materials(stage, root, s.friction, {"tab": s.tab_color, "ring": s.ring_color})
    tx, tt, tz = s.tab
    _body(stage, f"{root}/tab", np.zeros(3), I3, .005, None, iters=(16, 4))
    _shape(stage, f"{root}/tab/box", ("box", (-(s.wire / 2 + .001 + tx / 2), 0., 0.), (tx, tt, tz)), mat,
           looks["tab"], s.contact_offset)
    _set_mass_props(stage, f"{root}/tab", .005, [("box", (-(s.wire / 2 + .001 + tx / 2), 0., 0.), (tx, tt, tz))])
    _body(stage, f"{root}/ring", np.zeros(3), I3, s.ring_mass, None, iters=(16, 4))
    pts = s.wire_local()
    for k in range(len(pts) - 1):
        _shape(stage, f"{root}/ring/w{k:02d}", ("cap", None, (s.wire / 2, pts[k], pts[k + 1])), mat, looks["ring"],
               s.contact_offset)
    _set_mass_props(stage, f"{root}/ring", s.ring_mass,
                    [("cap", None, (s.wire / 2, pts[k], pts[k + 1])) for k in range(len(pts) - 1)])
    _revolute(stage, f"{root}/joints/hinge", f"{root}/tab", f"{root}/ring", (0., 0., 0.), I3, (0., 0., 0.), I3,
              "Z", -s.limit, s.limit, k=s.spring, d=s.damping, armature=s.armature)
    first = stage.GetPrimAtPath(f"{root}/tab")
    world = UsdGeom.XformCache().GetLocalToWorldTransform(first).RemoveScaleShear()
    fj = UsdPhysics.FixedJoint.Define(stage, f"{root}/joints/fixed")
    fj.CreateBody1Rel().SetTargets([first.GetPath()])
    fj.CreateLocalPos0Attr().Set(Gf.Vec3f(world.ExtractTranslation()))
    fj.CreateLocalRot0Attr().Set(Gf.Quatf(world.ExtractRotationQuat()))
    fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.))
    fj.CreateLocalRot1Attr().Set(Gf.Quatf(1.))


# ------------------------------------------------------------------------------------------------ reading
def _array(v):
    v = getattr(v, "torch", v)
    return v.detach().cpu().numpy().astype(float) if hasattr(v, "detach") else np.asarray(v, float)


def read(env, name):
    s = specs()[name]
    art = env.unwrapped.scene.articulations[name]
    d = art.data
    poses = _array(d.body_link_pose_w)[0]
    vel = _array(d.body_link_vel_w)[0]
    origin = _array(env.unwrapped.scene.env_origins)[0]
    bi = {n: i for i, n in enumerate(d.body_names)}
    out, speeds = {}, {}
    for n in s.body_names():
        i = bi[n]
        q = poses[i, 3:]
        R = quat_to_matrix(q) if np.all(np.isfinite(q)) and np.linalg.norm(q) > 1e-8 else np.full((3, 3), np.nan)
        out[n] = (poses[i, :3] - origin, R)
        speeds[n] = float(np.linalg.norm(vel[i, :3]))
    pts = np.array([p for p, _ in out.values()])
    lo, hi = np.nanmin(pts, 0), np.nanmax(pts, 0)
    first = s.body_names()[0]
    if isinstance(s, Strap):
        return StrapState(pos=(lo + hi) / 2, quat_xyzw=matrix_to_quat(out[first][1]) if np.all(np.isfinite(out[first][1]))
                          else np.array((0., 0., 0., 1.)), lin_vel=vel[bi[first], :3], ang_vel=vel[bi[first], 3:],
                          low=lo, high=hi, poses=out, speeds=speeds, spec=s)
    rp, Rr = out["ring"]
    st = RingState(pos=rp, quat_xyzw=matrix_to_quat(Rr) if np.all(np.isfinite(Rr)) else np.array((0., 0., 0., 1.)),
                   lin_vel=vel[bi["ring"], :3], ang_vel=vel[bi["ring"], 3:], low=lo, high=hi,
                   ring_p=rp, ring_R=Rr, tab_R=out["tab"][1], spec=s)
    w = st.wire()
    return RingState(pos=w.mean(0), quat_xyzw=st.quat_xyzw, lin_vel=st.lin_vel, ang_vel=st.ang_vel,
                     low=w.min(0), high=w.max(0), ring_p=rp, ring_R=Rr, tab_R=st.tab_R, spec=s)


# ------------------------------------------------------------------------------------------------ offline strap state
def strap_state_from(spec, poses):
    """StrapState from {body: (p, R)} (offline tests and layouts)."""
    pts = np.array([np.asarray(p, float) for p, _ in poses.values()])
    first = spec.body_names()[0]
    return StrapState(pos=(pts.min(0) + pts.max(0)) / 2, quat_xyzw=matrix_to_quat(poses[first][1]),
                      lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=pts.min(0), high=pts.max(0),
                      poses={k: (np.asarray(p, float), np.asarray(R, float)) for k, (p, R) in poses.items()},
                      speeds={k: 0.0 for k in poses}, spec=spec)


def straight_poses(spec, R0=None, p0=None, gate_deg=(0., 0.), web_twist_deg=0.0):
    """Body poses of the straight, flat layout (container pose (R0, p0), default the spec's), with optional gate
    angles (ends a, b) and a uniform total webbing twist spread over the joints (for tests)."""
    R0 = rot_z(spec.yaw) if R0 is None else np.asarray(R0, float)
    p0 = np.asarray(spec.pos, float) if p0 is None else np.asarray(p0, float)
    out = {}
    n = spec.links
    for i in range(n):
        c = np.array(((i + .5) * spec.pitch - spec.web_length() / 2, 0., 0.))
        tw = math.radians(web_twist_deg) * i / max(n - 1, 1)
        out[f"web_{i:02d}"] = (p0 + R0 @ c, R0 @ rot_axis((1, 0, 0), tw))
    hk = spec.hook
    for k, e in enumerate(spec.hook_ends):
        Rh, ph = spec.hook_frame_local(e)
        if e == "b" and web_twist_deg:
            Rh = rot_axis((1, 0, 0), math.radians(web_twist_deg)) @ Rh
        g = math.radians(gate_deg[0 if e == "a" else 1])
        Rw, pw = R0 @ Rh, p0 + R0 @ ph
        Rf = Rw @ (rot_axis((1, 0, 0), math.pi) if e in spec.flip_ends else np.eye(3))
        out[f"eye_{e}"] = (pw, Rw)
        out[f"hook_{e}"] = (pw + Rw @ np.array((hk.swivel_u, 0., 0.)), Rf)
        out[f"gate_{e}"] = (pw + Rf @ np.array((hk.pivot[0], hk.pivot[1], 0.)), Rf @ rot_z(g))
    return out


# ------------------------------------------------------------------------------------------------ scene hook
def _patch(scene):
    if getattr(scene, "_strap_kit_patched", False):
        return
    old_object, old_states = scene.task_object, scene.object_states

    @functools.wraps(old_object)
    def task_object(spec):
        if getattr(spec, "_is_strap_kit", False):
            return _object(spec)
        return old_object(spec)

    @functools.wraps(old_states)
    def object_states(env, names):
        names = tuple(names)
        reg = specs()
        own = [n for n in names if n in reg]
        result = old_states(env, tuple(n for n in names if n not in reg))
        for n in own:
            result[n] = read(env, n)
        return result

    scene.task_object, scene.object_states = task_object, object_states
    scene._strap_kit_patched = True


class _Loader(importlib.abc.Loader):
    def __init__(self, delegate):
        self.delegate = delegate

    def create_module(self, spec):
        fn = getattr(self.delegate, "create_module", None)
        return fn(spec) if fn else None

    def exec_module(self, module):
        self.delegate.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    _strap_kit_finder = True

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "yam_arena.scene":
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader)
            return spec


def install():
    if not any(getattr(f, "_strap_kit_finder", False) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    if "yam_arena.scene" in sys.modules:
        _patch(sys.modules["yam_arena.scene"])
