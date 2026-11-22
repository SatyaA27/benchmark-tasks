"""A magnetic levitator of the hanging kind (a levitating desk globe under a C-shaped frame), as forces applied every
physics step. No joints, snapping or proximity rule: the globe is carried by the force law from wherever it is.

What it models. A permanent magnet sits at the floating part's top (the globe's north pole). The frame's head holds an
iron-cored electromagnet with a position sensor; a controller drives the coil so the magnet hangs a fixed gap below
the head. The coil can only AID the magnet (one-way drive), so the part floats below the point where the magnet alone
would hold its weight. Forces on the magnet point (the part's top), directed toward an attraction point A_EFF above
the head's face on its axis, with g the gap from the head's face down to the magnet's top:

    magnet alone     F_pm = W * ((G_U + A) / (g + A)) ** 3        equals the weight at g = G_U: closer, it snaps up
    coil             F_c  = u * C_MAX * W * ((G0 + A) / (g + A)) ** 3
    controller       u    = clamp(U0 + KP * (g - G0) + KD * dg/dt, 0, 1)       (one-way drive, saturates)

U0 makes F_pm + F_c = W at the float gap G0; C_MAX is sized so full current just holds the part at G_LOW. Released at
rest with G_U < g < G_LOW the part settles at G0 (a stiff, damped hold); above G_U the magnet alone wins and it
sticks to the head; below G_LOW even full current is too weak and it falls. Off the axis the pull points toward the
attraction point, which centres the part (field centring); beyond R_FADE0 the sensor and field lose it, faded to zero
at R_FADE1. Small damping (C_LAT on sideways motion, C_ROT on the part's wobble and spin) stands for eddy-current
and air losses. The frame receives the equal and
opposite force at the head (it carries the floating part's weight).

Tuned numbers (all ESTIMATES; see the task README): floating part 70 g (a 3-inch globe), G0 15 mm, G_U 10 mm,
G_LOW 20 mm, A 10 mm, so C_MAX 1.216 and U0 0.401; KP 123 /m, KD 1.49 s/m (net vertical stiffness about 61 N/m,
damping ratio about 0.3, 4.7 Hz); R_FADE0 12 mm, R_FADE1 20 mm; C_LAT 0.5 N s/m, C_ROT 0.0016 N m s/rad (damping ratio about 0.2 sideways, 0.5 on
the wobble). Electronics are not simulated; the
controller is this force law, applied every physics step (240 Hz).

How to use it (pure Python offline; the simulator hook installs itself only on a simulator host):

    import levitator_kit as lev
    LEV = lev.Levitator("globe", "frame", magnet_local=(0, 0, R), head_local=(hx, 0, hz), mass=0.070)
    lev.install(LEV)
    LEV.force(magnet_world, head_world, magnet_vel, head_vel)   # offline: the force on the magnet point (N)

The hook follows the magnetic-tiles kit (another task folder): it wraps the arena's reset, then wraps
write_data_to_sim of the two rigid objects so that before every physics step their poses and velocities are read and
each receives its force and torque (world frame) through apply_forces_and_torques_at_position.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass(frozen=True)
class Law:
    g0: float = 0.015      # float gap (m)
    g_u: float = 0.010     # magnet alone holds the weight here
    g_low: float = 0.020   # full coil current just holds the weight here
    a: float = 0.010       # softening length (magnet and pole size)
    kp: float = 123.0      # controller: current fraction per metre of gap error
    kd: float = 1.49       # controller: current fraction per m/s
    r_fade0: float = 0.012
    r_fade1: float = 0.020
    c_lat: float = 0.5     # N s/m
    c_rot: float = 0.0016  # N m s/rad, on the floating part's spin and wobble

    @property
    def c_max(self):
        return ((self.g_low + self.a) ** 3 - (self.g_u + self.a) ** 3) / (self.g0 + self.a) ** 3

    @property
    def u0(self):
        return (1.0 - ((self.g_u + self.a) / (self.g0 + self.a)) ** 3) / self.c_max

    def lift(self, g, gdot=0.0, w=1.0):
        """(vertical pull in N for weight w, coil fraction u) at gap g (m) closing at -gdot."""
        g = max(float(g), 0.0)
        u = float(np.clip(self.u0 + self.kp * (g - self.g0) + self.kd * gdot, 0.0, 1.0))
        s = 1.0 / (g + self.a) ** 3
        return w * s * ((self.g_u + self.a) ** 3 + u * self.c_max * (self.g0 + self.a) ** 3), u

    def fade(self, r):
        x = float(np.clip((r - self.r_fade0) / (self.r_fade1 - self.r_fade0), 0.0, 1.0))
        return 1.0 - x * x * (3.0 - 2.0 * x)


LAW = Law()


class Levitator:
    def __init__(self, part, frame, magnet_local, head_local, mass, law=LAW):
        self.part, self.frame, self.law, self.mass = part, frame, law, float(mass)
        self.magnet_local = np.asarray(magnet_local, float)
        self.head_local = np.asarray(head_local, float)
        self.last = {}

    def force(self, magnet, head, v_magnet=np.zeros(3), v_head=np.zeros(3), head_axis=(0.0, 0.0, 1.0)):
        """World force (N) on the magnet point; the head gets minus this."""
        L = self.law
        axis = np.asarray(head_axis, float)
        d = np.asarray(head, float) - np.asarray(magnet, float)
        g = float(d @ axis)
        lat = d - g * axis
        r = float(np.linalg.norm(lat))
        gdot = float((np.asarray(v_head, float) - np.asarray(v_magnet, float)) @ axis)
        w = self.mass * G
        f, u = L.lift(g, gdot, w)
        if g < -0.01:              # magnet above the head's face: no pull (not a real configuration)
            f = 0.0
        k = L.fade(r)
        to_pole = d + L.a * axis
        n = to_pole / max(float(np.linalg.norm(to_pole)), 1e-6)
        vlat = np.asarray(v_magnet, float) - np.asarray(v_head, float)
        vlat = vlat - (vlat @ axis) * axis
        F = k * (f * n - L.c_lat * vlat)
        self.last = {"g_mm": g * 1000, "r_mm": r * 1000, "u": u, "F": float(np.linalg.norm(F)), "fade": k}
        return F


def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------- simulator hook (simulator hosts only)
_STATE = {"lev": None, "wrench": None, "printed": False}


def _arr(x):
    x = getattr(x, "torch", x)
    return x.detach().cpu().numpy().astype(np.float64) if hasattr(x, "detach") else np.asarray(x, np.float64)


def _vel(o):
    for name in ("root_com_lin_vel_w", "root_lin_vel_w", "root_link_lin_vel_w"):
        v = getattr(o.data, name, None)
        if v is not None:
            return _arr(v)[0][:3]
    return np.zeros(3)


def _hook(env):
    base = env.unwrapped
    lev = _STATE["lev"]
    objs = {}
    for n in (lev.part, lev.frame):
        try:
            objs[n] = base.scene.rigid_objects[n]
        except (AttributeError, KeyError):
            objs[n] = base.scene[n]
    import warp as wp

    def compute():
        po = _arr(objs[lev.part].data.root_link_pose_w)[0]
        fo = _arr(objs[lev.frame].data.root_link_pose_w)[0]
        pR, fR = quat_matrix(po[3:7]), quat_matrix(fo[3:7])
        magnet = po[:3] + pR @ lev.magnet_local
        head = fo[:3] + fR @ lev.head_local
        F = lev.force(magnet, head, _vel(objs[lev.part]), _vel(objs[lev.frame]), fR[:, 2])
        w = np.zeros(3)
        for name in ("root_com_ang_vel_w", "root_ang_vel_w", "root_link_ang_vel_w"):
            v = getattr(objs[lev.part].data, name, None)
            if v is not None:
                w = _arr(v)[0][:3]
                break
        tp = np.cross(magnet - po[:3], F) - lev.law.c_rot * w
        tf = np.cross(head - fo[:3], -F)
        _STATE["wrench"] = {lev.part: (F, tp), lev.frame: (-F, tf)}

    first = lev.part
    for n, obj in objs.items():
        if getattr(obj, "_levitator_hooked", False):
            continue
        orig = obj.write_data_to_sim

        def write(*args, _n=n, _obj=obj, _orig=orig, **kwargs):
            result = _orig(*args, **kwargs)
            if _n == first or _STATE["wrench"] is None:
                compute()
            f, t = _STATE["wrench"][_n]
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
        obj._levitator_hooked = True
    if not _STATE["printed"]:
        L = lev.law
        print(f"LEVITATOR hook on {list(objs)}; law={L} c_max={L.c_max:.3f} u0={L.u0:.3f}", flush=True)
        _STATE["printed"] = True


def install(lev):
    """Remember the levitator and, on a simulator host, hook every reset so the parts get their forces."""
    _STATE["lev"] = lev
    _STATE["wrench"] = None
    try:
        from yam_arena.inspect_embodiment import YamArenaEmbodiment as cls
    except ImportError:
        return
    if getattr(cls, "_levitator_wrapped", False):
        return
    inner_reset = cls.reset

    @functools.wraps(inner_reset)
    def reset(self, *args, **kwargs):
        result = inner_reset(self, *args, **kwargs)
        _STATE["wrench"] = None
        lev = _STATE["lev"]
        if lev is not None and self.task is not None and all(n in self.task.names for n in (lev.part, lev.frame)):
            _hook(self._env)
        return result

    cls.reset = reset
    cls._levitator_wrapped = True


def last():
    """The latest force summary ({} offline before any call)."""
    lev = _STATE["lev"]
    return dict(lev.last) if lev is not None else {}
