"""Small permanent magnets between free parts, as forces applied every physics step (no joints, pins or grooves).

What it models: parts that hold together only by small magnets near their edges or ends (magnetic building tiles,
toy-train couplings, magnetic catches). Each part carries magnet points in its own frame. Any two points on
DIFFERENT parts attract each other along the line between them, equal and opposite, with

    F(d) = F_MATE * ((1 + D_MATE / A) / (1 + d / A)) ** 4 * fade(d)          (newtons, d in metres)

where d is the distance between the two points, D_MATE the distance at which the two parts sit when joined, A a
softening length that stands for the magnets' finite size (a point dipole would give A = 0) and fade() a smooth
cut that takes the pull from full at D_FADE to zero at D_CUT. The pull never depends on which way round the parts
meet (real tiles reach that with alternating or free-turning magnets); a point pair always attracts.
F_MATE is the pull of ONE point pair when the parts are joined; a joint made of k pairs takes k * F_MATE to pull
straight apart (its pull-off force). Each part receives the sum of its pairs' forces at its centre of mass plus the
matching torque (lever = magnet point - centre of mass), so a magnet inside a part also gives a righting torque.

Tuned numbers for magnetic building tiles (MAGNET_TILES below; all ESTIMATES, see the task README): two magnet
points per edge, 1.5 mm in from the edge's ridge along the mid-plane and 19 mm either side of the edge's middle;
D_MATE 2.12 mm (two tiles joined at a right angle), A 5 mm, F_MATE 0.70 N (a right-angle edge takes 1.4 N to pull
off; two tiles joined flat sit 3.0 mm apart and take 0.88 N), D_FADE 12 mm, D_CUT 15 mm. Catch: one edge pulls with
the weight of a 32 g tile (0.31 N) from a 3.3 mm gap; 0.1 N from about 6 mm.

How to use it (pure Python offline; the simulator hook installs itself only on a simulator host):

    import magnet_kit
    MAGNETS = magnet_kit.Magnets({"tile_a": points_a, "tile_b": points_b}, coms={"tile_a": com_a, ...},
                                 law=magnet_kit.MAGNET_TILES)
    magnet_kit.install(MAGNETS)                 # once, at import of the task module
    MAGNETS.wrenches({"tile_a": (origin, R), ...})   # offline: forces and torques for given poses

On the simulator it wraps the arena's reset; after each reset it wraps write_data_to_sim of every magnet part
(rigid objects named as the parts) so that, before every physics step, the first such call reads all the parts'
poses, works out every part's wrench once, and each part then applies its own wrench (world frame) through
apply_forces_and_torques_at_position. Nothing else is written to the simulation. diag_line() returns a one-line
summary for GPU diagnostics.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np


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


MAGNET_TILES = Law(f_mate=0.70, d_mate=1.5e-3 * math.sqrt(2.0), soft=5.0e-3, d_fade=12.0e-3, d_cut=15.0e-3)


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

    def __init__(self, points, coms=None, law=MAGNET_TILES):
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

    def joined_pairs(self, poses, a, b, within):
        """Number of point pairs between parts a and b closer than `within` (m)."""
        pa = poses[a][0] + self.local[a] @ np.asarray(poses[a][1]).T
        pb = poses[b][0] + self.local[b] @ np.asarray(poses[b][1]).T
        d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
        return int(np.sum(d.min(axis=1) < within))

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
