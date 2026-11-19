"""Wooden toy railway kit: grooved wooden track pieces and magnet-coupled vehicles (BRIO-style wooden trains).

What it models
- Track: each wooden track piece is a static body made of exact triangle meshes: a 9 mm bed and three 3 mm rails on
  top, which leave two grooves 6 mm wide and 3 mm deep. curve_piece() builds one curved piece (any radius and sweep;
  offline its rails are tangent boxes, about 0.3 mm inside the true arcs).
- Vehicles: one rigid body each (vehicle()): four wheels fixed to the body that sit down in the grooves, a chassis,
  an upper body (engine, box wagon or load wagon with its load), and a round coupling at each end. The wheels do not
  turn: they slide in the grooves on a low friction (0.05 static, 0.04 sliding where vehicle meets track) that stands
  for the free-rolling wheels' resistance. Sideways, only the groove walls hold them, as on the real track.
- Couplings: small magnets as forces between named points, applied every physics step (no joints): any two points on
  DIFFERENT parts attract along the line between them, equal and opposite,

      F(d) = F_MATE * ((1 + D_MATE / A) / (1 + d / A)) ** 4 * fade(d)          (newtons, d in metres)

  where d is the distance between the points, D_MATE the distance when the parts are joined, A a softening length
  that stands for the magnets' finite size and fade() a smooth cut from full at D_FADE to zero at D_CUT. The pull
  never depends on which way round the parts meet (the real couplings attract one way round only; a train built
  the right way round never meets the other case). Each part gets its pairs' forces at its centre of mass plus the
  matching torque. The law and the simulator hook are copied unchanged from the magnetic-tile kit
  (magnet_kit.py); only the numbers and the railway builders are new.

Tuned numbers (ESTIMATES unless a source is named in the task README; a person checks each with a ruler or a
kitchen scale): track 40 mm wide, 12 mm thick, grooves 6 mm wide and 3 mm deep, groove centres 26 mm apart;
wheels 13 mm discs 4 mm wide (an 8 x 4 mm runner under each sits in its groove), axles 44 mm apart; vehicle bodies 70 mm long and 30-34 mm wide; coupling magnets
modelled as 9 mm balls whose centres sit 37 mm from the vehicle's middle, so coupled vehicles are 83 mm apart
(four vehicles 332 mm long in all, against the maker's 333 mm for the train). TRAIN_COUPLING: F_MATE 1.0 N (pull-off
of one coupling), D_MATE 9 mm (ball centres touching), A 6 mm, D_FADE 14 mm, D_CUT 18 mm: 0.48 N at a 3 mm gap,
0.22 N at 6 mm. Friction: vehicles (0.05, 0.04), track (0.05, 0.04), so 0.05/0.04 where they meet.

How to use it (pure Python offline; the simulator hook installs itself only on a simulator host):

    import train_kit as kit
    track = [kit.curve_piece(f"curve_{i}", centre, top, kit.RADIUS, i * math.pi / 4) for i in range(8)]
    engine = kit.vehicle("engine", pos, yaw, "engine")
    MAGNETS = kit.Magnets({"engine": [kit.REAR], "wagon": [kit.FRONT, kit.REAR]}, coms=..., law=kit.TRAIN_COUPLING)
    kit.install(MAGNETS)
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

from yam_parts import Part, ball, box, cyl
from yam_mesh import MeshPart, prism


@dataclass(frozen=True)
class Law:
    f_mate: float  # N, one point pair at the joined distance
    d_mate: float  # m, distance between the two points of a joined pair
    soft: float  # m, softening length A
    d_fade: float  # m, full law below this distance
    d_cut: float  # m, zero at and beyond this distance
    d_min: float = 0.0005  # m, distances below this are treated as this (keeps the direction defined)

    def magnitude(self, d):
        d = np.maximum(np.asarray(d, float), self.d_min)
        f = self.f_mate * ((1.0 + self.d_mate / self.soft) / (1.0 + d / self.soft)) ** 4
        u = np.clip((d - self.d_fade) / (self.d_cut - self.d_fade), 0.0, 1.0)
        return f * (1.0 - u * u * (3.0 - 2.0 * u))

    def pull_off(self, pairs, d=None):
        """Force (N) to pull a joint of `pairs` point pairs straight apart from distance d (default: joined)."""
        return float(pairs * self.magnitude(self.d_mate if d is None else d))

    def gap_for(self, force, pairs, d0=None):
        """Extra separation (m) beyond d0 (default d_mate) at which `pairs` pairs pull with `force` (N)."""
        d0 = self.d_mate if d0 is None else d0
        lo, hi = d0, self.d_cut
        if self.pull_off(pairs, lo) < force:
            return 0.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.pull_off(pairs, mid) > force else (lo, mid)
        return 0.5 * (lo + hi) - d0


# ---------------------------------------------------------------- railway geometry (ESTIMATES, see the docstring)
RADIUS = 0.216  # m, centre line of a large curve: the maker's 170 mm track length over a 45-degree sweep
TRACK_W, BED_H, RAIL_H = 0.040, 0.009, 0.003  # track width, bed under the grooves, groove depth
GROOVE_W, GAUGE = 0.006, 0.026  # groove width, distance between the groove centres
WHEEL_R, WHEEL_W, AXLE_X = 0.0065, 0.004, 0.022  # visible disc radius and width, axles at +-AXLE_X
RUNNER = (0.008, WHEEL_W, 0.004)  # the part of each wheel inside its groove: 8 mm long, 4 mm wide, 4 mm high
BODY_L = 0.070
MAG_X, MAG_R, MAG_Z = 0.037, 0.0045, 0.013  # coupling ball centres at +-MAG_X, radius, height above the wheel bottoms
FRONT, REAR = (MAG_X, 0.0, MAG_Z), (-MAG_X, 0.0, MAG_Z)
PITCH = 2 * MAG_X + 2 * MAG_R  # 83 mm between the middles of coupled vehicles
TRAIN_COUPLING = Law(f_mate=1.0, d_mate=2 * MAG_R, soft=0.006, d_fade=0.014, d_cut=0.018)
VEHICLE_FRICTION, TRACK_FRICTION = (0.05, 0.04), (0.05, 0.04)
WOOD = (0.62, 0.42, 0.22)
WHEELS = tuple((x, y) for x in (-AXLE_X, AXLE_X) for y in (-GAUGE / 2, GAUGE / 2))  # wheel middles, vehicle frame
MASS = {"engine": 0.060, "load": 0.045, "box": 0.055}  # kg, ESTIMATES (the maker's shipping weight is 0.29 kg)

# rails over the bed: (name, inner offset, outer offset) from the centre line, across the track
_RAILS = (("inner", -TRACK_W / 2, -GAUGE / 2 - GROOVE_W / 2), ("middle", -GAUGE / 2 + GROOVE_W / 2, GAUGE / 2 - GROOVE_W / 2),
          ("outer", GAUGE / 2 + GROOVE_W / 2, TRACK_W / 2))


def _ring(name, r0, r1, a0, sweep, z0, h, n_arc, n_proxy, color):
    """Exact mesh of a flat ring sector r0..r1, angles a0..a0+sweep, z0..z0+h; offline: n_proxy tangent boxes."""
    angles = np.linspace(a0, a0 + sweep, n_arc + 1)
    outline = [(r1 * math.cos(t), r1 * math.sin(t)) for t in angles] + \
              [(r0 * math.cos(t), r0 * math.sin(t)) for t in angles[::-1]]
    proxy = []
    step = sweep / n_proxy
    for k in range(n_proxy):
        t = a0 + (k + 0.5) * step
        r = 0.5 * (r0 + r1)
        # a box tangent at the middle of each step, as long as the step's chord at the outer edge
        length = 2 * r1 * math.sin(step / 2)
        proxy.append((r * math.cos(t), r * math.sin(t), z0 + h / 2, t, (r1 - r0, length, h)))
    mesh = prism(outline, h, name=name, collision="exact", color=color)
    from dataclasses import replace
    lift = np.array([0.0, 0.0, z0 + h / 2])
    # the mesh's own proxy stays its bounding box (bounds only); TrackPiece.world_shapes gives the arcs offline
    mesh = replace(mesh, points=mesh.points + lift,
                   proxy=tuple(box(tuple(np.asarray(b.center) + lift), b.size) for b in mesh.proxy))
    return mesh, proxy


class TrackPiece(MeshPart):
    """A fixed wooden track piece: exact meshes on the simulator; offline, turned boxes along the arcs."""

    def world_shapes(self, origin, R):
        out = []
        for (x, y, z, t, size) in self.track_boxes:
            rot = np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])
            out.append(("box", np.asarray(origin) + R @ np.array([x, y, z]), R @ rot, np.array(size)))
        return tuple(out)


def curve_piece(name, centre, top, radius=RADIUS, a0=0.0, sweep=math.pi / 4, color=WOOD):
    """One curved piece about `centre` (x, y) on a table at height `top`, from angle a0 through `sweep` (radians)."""
    meshes, boxes = [], []
    n_arc = max(8, int(round(math.degrees(sweep) / 1.5)))
    n_proxy = max(4, int(round(math.degrees(sweep) / 5.6)))
    parts = [("bed", -TRACK_W / 2, TRACK_W / 2, 0.0, BED_H)] + [(n, lo, hi, BED_H, RAIL_H) for n, lo, hi in _RAILS]
    for label, lo, hi, z0, h in parts:
        mesh, proxy = _ring(label, radius + lo, radius + hi, a0, sweep, z0, h, n_arc, n_proxy, color)
        meshes.append(mesh)
        boxes += proxy
    # offline proxies as box shapes, so every offline check (overlaps, footprints, bounds) sees the grooves
    proxy_shapes = []
    for (x, y, z, t, size) in boxes:
        proxy_shapes.append((x, y, z, t, size))
    piece = TrackPiece(name, (float(centre[0]), float(centre[1]), float(top)), meshes=meshes, color=color,
                       friction=TRACK_FRICTION, contact_offset=0.001)
    object.__setattr__(piece, "track_boxes", tuple(proxy_shapes))
    return piece


def vehicle(name, pos, yaw, kind, color, load_color=None):
    """One vehicle as a rigid body; origin at the middle of its wheelbase, level with the wheel bottoms; +x forward.

    kind: "engine" (boiler, cab and chimney), "box" (closed wagon with a sliding-door body) or "load" (flat wagon
    with a container on its pins, one body). Couplings: a 9 mm ball at each end (the engine has one at the front too,
    as the real one does, though nothing couples there in this task)."""
    dark, metal = (0.10, 0.10, 0.11), (0.28, 0.29, 0.30)
    # each wheel: a disc whose rim clears the rail tops by 0.5 mm, and under it a short runner that sits in the groove
    shapes = [cyl((x, y, RAIL_H + 0.0005 + WHEEL_R), WHEEL_R, WHEEL_W, axis="y", color=dark) for x, y in WHEELS]
    shapes += [box((x, y, RUNNER[2] / 2), RUNNER, color=dark) for x, y in WHEELS]
    shapes += [box((0.0, 0.0, 0.015), (BODY_L, 0.030, 0.010), color=color),
               ball(FRONT, MAG_R, color=metal), ball(REAR, MAG_R, color=metal)]
    if kind == "engine":
        shapes += [cyl((0.008, 0.0, 0.030), 0.012, 0.052, axis="x", color=color),
                   box((-0.022, 0.0, 0.034), (0.026, 0.034, 0.030), color=color),
                   box((-0.022, 0.0, 0.0505), (0.030, 0.034, 0.003), color=dark),
                   cyl((0.024, 0.0, 0.045), 0.0055, 0.012, axis="z", color=dark)]
    elif kind == "box":
        shapes += [box((0.0, 0.0, 0.031), (BODY_L, 0.034, 0.022), color=color),
                   box((0.0, 0.0, 0.0435), (BODY_L, 0.030, 0.003), color=dark)]
    elif kind == "load":
        shapes += [box((0.0, 0.0, 0.0215), (BODY_L, 0.030, 0.003), color=color),
                   box((0.0, 0.0, 0.0335), (0.060, 0.030, 0.021), color=load_color or color)]
    else:
        raise ValueError(f"unknown vehicle kind {kind!r}")
    return Part(name, tuple(pos), shapes, MASS[kind], color, yaw=yaw, friction=VEHICLE_FRICTION,
                contact_offset=0.001)


def wheel_points(part_origin, R):
    """World points at the bottoms of the four wheels."""
    return np.array([np.asarray(part_origin) + np.asarray(R) @ np.array([x, y, 0.0]) for x, y in WHEELS])


def quat_matrix(q):
    """Rotation matrix from an (x, y, z, w) quaternion."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class Magnets:
    """Magnet points on named parts and the law between them."""

    def __init__(self, points, coms=None, law=TRAIN_COUPLING):
        self.names = list(points)
        self.law = law
        self.local = {n: np.asarray(points[n], float).reshape(-1, 3) for n in self.names}
        self.coms = {n: np.asarray((coms or {}).get(n, np.zeros(3)), float) for n in self.names}
        self.owner = np.concatenate([np.full(len(self.local[n]), i) for i, n in enumerate(self.names)])
        self.last = {}

    def world_points(self, poses):
        """{name: (origin, R)} -> (N, 3) world magnet points in self.names order."""
        return np.concatenate([poses[n][0] + self.local[n] @ np.asarray(poses[n][1]).T for n in self.names])

    def pair_table(self, poses):
        """(i, j, d, f) for every attracting pair of points on different parts (i < j, f > 0)."""
        p = self.world_points(poses)
        diff = p[None, :, :] - p[:, None, :]
        d = np.linalg.norm(diff, axis=2)
        mask = (self.owner[:, None] != self.owner[None, :]) & (d < self.law.d_cut)
        mask = np.triu(mask, 1)
        i, j = np.nonzero(mask)
        return i, j, d[i, j], self.law.magnitude(d[i, j]), p, diff

    def wrenches(self, poses):
        """{name: (force (3,), torque (3,))} in the world frame, torque about each part's centre of mass."""
        i, j, d, f, p, diff = self.pair_table(poses)
        force = np.zeros((len(p), 3))
        if len(i):
            u = diff[i, j] / np.maximum(d, self.law.d_min)[:, None]  # from point i toward point j
            np.add.at(force, i, f[:, None] * u)
            np.add.at(force, j, -f[:, None] * u)
        out, k = {}, 0
        for n in self.names:
            m = len(self.local[n])
            origin, rot = poses[n]
            com = origin + np.asarray(rot) @ self.coms[n]
            fs = force[k:k + m]
            out[n] = (fs.sum(axis=0), np.cross(p[k:k + m] - com, fs).sum(axis=0))
            k += m
        self.last = {"pairs": int(len(i)), "max_f": float(f.max()) if len(f) else 0.0,
                     "joined": int(np.sum(d < self.law.d_mate + 1.0e-3)) if len(d) else 0}
        return out

    def gap(self, poses, a, b):
        """Smallest distance (m) between a magnet point of part a and one of part b, less the joined distance."""
        pa = poses[a][0] + self.local[a] @ np.asarray(poses[a][1]).T
        pb = poses[b][0] + self.local[b] @ np.asarray(poses[b][1]).T
        d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
        return float(d.min() - self.law.d_mate)

    def diag_line(self, wrench=None):
        s = f"pairs={self.last.get('pairs', 0)} joined={self.last.get('joined', 0)} maxF={self.last.get('max_f', 0):.2f}"
        if wrench:
            s += " " + " ".join(f"{n}:{np.linalg.norm(w[0]):.2f}" for n, w in wrench.items())
        return s


# ---------------------------------------------------------------- simulator hook (simulator hosts only)
_STATE = {"magnets": None, "tick": 0, "cache": None, "hooked": set(), "printed": False, "wrench": None}


def _arr(x):
    x = getattr(x, "torch", x)
    return x.detach().cpu().numpy().astype(np.float64) if hasattr(x, "detach") else np.asarray(x, np.float64)


def _hook(env):
    base = env.unwrapped
    mags = _STATE["magnets"]
    objs = {}
    for n in mags.names:
        try:
            objs[n] = base.scene.rigid_objects[n]
        except (AttributeError, KeyError):
            objs[n] = base.scene[n]
    origins = _arr(base.scene.env_origins)
    import warp as wp

    def compute():
        poses = {}
        for n, o in objs.items():
            pose = _arr(o.data.root_link_pose_w)[0]
            poses[n] = (pose[:3] - origins[0], quat_matrix(pose[3:7]))
        _STATE["wrench"] = mags.wrenches(poses)
        _STATE["poses"] = poses

    first = mags.names[0]
    for n, obj in objs.items():
        if getattr(obj, "_magnet_kit_hooked", False):
            continue
        orig = obj.write_data_to_sim

        def write(*args, _n=n, _obj=obj, _orig=orig, **kwargs):
            result = _orig(*args, **kwargs)
            if _n == first or _STATE["wrench"] is None:
                compute()
            f, t = _STATE["wrench"][_n]
            if np.any(f) or np.any(t):
                view = getattr(_obj, "root_view", None) or _obj.root_physx_view
                view.apply_forces_and_torques_at_position(
                    force_data=wp.array(np.asarray(f, np.float32).ravel(), dtype=wp.float32, device=base.device),
                    torque_data=wp.array(np.asarray(t, np.float32).ravel(), dtype=wp.float32, device=base.device),
                    position_data=None,
                    indices=_obj._ALL_INDICES,
                    is_global=True,
                )
            return result

        obj.write_data_to_sim = write
        obj._magnet_kit_hooked = True
    if not _STATE["printed"]:
        print(f"MAGNET_KIT hook on {list(objs)}; law={mags.law}", flush=True)
        _STATE["printed"] = True


def install(magnets):
    """Remember the magnets and, on a simulator host, hook every reset so the parts get their pull."""
    _STATE["magnets"] = magnets
    _STATE["wrench"] = None
    try:
        from yam_arena.inspect_embodiment import YamArenaEmbodiment as cls
    except ImportError:
        return
    if getattr(cls, "_magnet_kit_wrapped", False):
        return
    inner_reset = cls.reset

    @functools.wraps(inner_reset)
    def reset(self, *args, **kwargs):
        result = inner_reset(self, *args, **kwargs)
        _STATE["wrench"] = None
        mags = _STATE["magnets"]
        if mags is not None and self.task is not None and all(n in self.task.names for n in mags.names):
            _hook(self._env)
        return result

    cls.reset = reset
    cls._magnet_kit_wrapped = True


def last_wrench():
    """The wrench of the latest physics step on the simulator ({} offline)."""
    return _STATE["wrench"] or {}
