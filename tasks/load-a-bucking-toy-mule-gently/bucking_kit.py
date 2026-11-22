"""A hair-trigger bucking toy mule (the spring-loaded saddle-stacking game), as forces applied every physics step.

What it models. The mule is three rigid bodies: the front body on its base (head, neck, chest, front legs), the
hindquarters with the blanket on top (the sprung part), and a loose saddle with hangers that rests on the blanket.
In the toy a spring-biased catch carries the hindquarters' weight and the saddle's load; extra load or a jolt pushes
the hindquarters down until the catch slips, and then the spring kicks the hindquarters up and the kicker strikes the
saddle, throwing it and the gear.

Here the catch is a spring-damper hold of the hindquarters at their set pose relative to the front body: soft
vertically (the catch's give), stiff sideways and in rotation, preloaded with the hindquarters' and saddle's own
weight so that an empty saddle sits at zero give. All load reaches it only through contact (gear on the saddle, the
saddle on the blanket, a hand on either). When the give passes D_TRIP the catch lets go for the rest of the episode:
the hindquarters' hold target swings rear-up by KICK_DEG about the front-top edge of the barrel within KICK_S, and the
saddle receives an upward push for KICK_STEPS physics steps (the kicker strike). Equal and opposite forces act on the
front body. No proximity, speed or timer rule: a careful set-down simply does not deflect the catch far enough.

Tuned numbers (all ESTIMATES, see the task README): vertical stiffness 200 N/m, damping ratio about 0.3 with the
hindquarters and saddle (75 g); trip at 2.9 mm of give = 60 g of static load (sensitivity level 3); sideways
1500 N/m, 8 N s/m; rotation 0.5 N m/rad, 0.006 N m s/rad; hold forces capped at 4 N and 0.2 N m (a hand dragging the bucked hindquarters pulled the light mule over
and blew the scene up when the hold was uncapped); kick 35 deg in 0.10 s at 2x stiffness (a stronger kick about the barrel's lower edge drove the
hindquarters into the chest and blew the scene up in a probe), saddle push to about
1.5 m/s. Offline (1-D, the same law): an 8 g piece set down still gives 0.6 mm, dropped 5 cm onto an empty saddle
1.6 mm; with two pieces on, a third dropped 10 cm reaches 3.0 mm and bucks; pressing a piece onto the saddle with
1.5 N or more bucks at once.

How to use it (pure Python offline; the simulator hook installs itself only on a simulator host):

    import bucking_kit as bk
    CATCH = bk.Catch("mule", "rump", "saddle", rump_com=(...), mule_com=(...), seat=(...), m_rump=0.05,
                     m_saddle=0.025)
    bk.install(CATCH)
    bk.tripped()      # True once the mule has bucked this episode

The hook follows the levitator kit (another task folder): it wraps the arena's reset (re-seating the hindquarters
and saddle on the mule after seeded shifts, then re-arming the catch), then wraps write_data_to_sim of the three
rigid objects so that before every physics step their poses and velocities are read and each receives its force and
torque through apply_forces_and_torques_at_position.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

G = 9.81
DT = 1.0 / 240.0


@dataclass(frozen=True)
class Law:
    k_v: float = 200.0        # N/m, the catch's vertical give
    c_v: float = 2.3          # N s/m (damping ratio ~0.3 with 75 g)
    k_h: float = 1500.0       # N/m sideways / fore-aft
    c_h: float = 8.0
    k_r: float = 0.5          # N m/rad
    c_r: float = 0.006        # N m s/rad
    d_trip: float = 0.0029    # m of give at which the catch slips (60 g static)
    kick_deg: float = 35.0
    kick_s: float = 0.10
    kick_mult: float = 2.0
    kick_v: float = 1.5       # m/s given to the saddle (with ~30 g of gear) by the kicker strike
    kick_steps: int = 4
    f_max: float = 4.0        # N, the hold's pull back toward its pose is capped (a plastic catch, not a vice)
    t_max: float = 0.2        # N m
    pivot: tuple = (0.012, 0.0, 0.140)   # hindquarters swing about the barrel's front-top edge (mule frame)


LAW = Law()


def _cap(v, m):
    n = float(np.linalg.norm(v))
    return v * (m / n) if n > m else v


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotvec(R):
    """Axis-angle vector of a rotation matrix."""
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    a = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if a < 1e-6:
        return 0.5 * v
    s = math.sin(a)
    if abs(s) < 1e-6:
        return np.zeros(3)
    return a / (2.0 * s) * v


class Catch:
    def __init__(self, mule, rump, saddle, rump_com, mule_com, seat, m_rump, m_saddle, law=LAW):
        self.mule, self.rump, self.saddle, self.law = mule, rump, saddle, law
        self.rump_com = np.asarray(rump_com, float)    # rump COM in the rump's (= mule's set) frame
        self.mule_com = np.asarray(mule_com, float)
        self.seat = np.asarray(seat, float)             # where the saddle's weight bears (mule frame)
        self.m_rump, self.m_saddle = float(m_rump), float(m_saddle)
        self.reset()

    def reset(self):
        self.armed = False
        self.tripped = False
        self.t = 0.0
        self.t_trip = None
        self.kick_left = 0
        self.max_give = 0.0
        self.last = {}

    def arm(self):
        self.reset()
        self.armed = True

    def target(self):
        """(rotation, COM position) the hold aims the hindquarters at, in the mule frame."""
        L = self.law
        if not self.tripped:
            return np.eye(3), self.rump_com.copy()
        f = min(1.0, (self.t - self.t_trip) / L.kick_s)
        R = rot_y(L.kick_deg * f)
        P = np.asarray(L.pivot, float)
        return R, P + R @ (self.rump_com - P)

    def wrench(self, pm, Rm, vm, wm, cr, Rr, vr, wr):
        """World (F, T) on the hindquarters' COM; also the saddle kick force. Advances time by one physics step."""
        L = self.law
        self.t += DT
        Rt, ct = self.target()
        mult = L.kick_mult if self.tripped else 1.0
        goal = pm + Rm @ ct
        ep = Rm.T @ (cr - goal)                                  # mule frame
        ev = Rm.T @ (vr - (vm + np.cross(wm, cr - pm)))
        give = -float(ep[2])
        if self.armed:
            self.max_give = max(self.max_give, give)
            self.win_max = max(getattr(self, "win_max", 0.0), give)
        K = np.array([L.k_h, L.k_h, L.k_v]) * mult
        C = np.array([L.c_h, L.c_h, L.c_v]) * math.sqrt(mult)
        F = _cap(Rm @ (-K * ep - C * ev), L.f_max)
        F = F + np.array([0.0, 0.0, (self.m_rump + self.m_saddle) * G])
        seat_w = pm + Rm @ self.seat
        T_pre = -np.cross(seat_w - cr, np.array([0.0, 0.0, -self.m_saddle * G]))
        er = (Rm @ Rt) @ rotvec((Rm @ Rt).T @ Rr)
        T = _cap(-L.k_r * mult * er - L.c_r * math.sqrt(mult) * (wr - wm), L.t_max) + T_pre
        if not (np.all(np.isfinite(F)) and np.all(np.isfinite(T))):
            F, T = np.zeros(3), np.zeros(3)
        Fs = np.zeros(3)
        if self.armed and not self.tripped and give > L.d_trip:
            self.tripped, self.t_trip, self.kick_left = True, self.t, L.kick_steps
            print(f"BUCK t={self.t:.3f} give_mm={give * 1000:.2f}", flush=True)
        if self.kick_left > 0:
            Fs = Rm @ np.array([0.0, 0.0, (self.m_saddle + 0.03) * L.kick_v / (L.kick_steps * DT)])
            self.kick_left -= 1
        self.last = {"give_mm": give * 1000, "max_give_mm": self.max_give * 1000, "tripped": self.tripped,
                     "armed": self.armed}
        return F, T, Fs


def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------- simulator hook (simulator hosts only)
_STATE = {"catch": None, "wrench": None, "printed": False, "objs": None}


def _arr(x):
    x = getattr(x, "torch", x)
    return x.detach().cpu().numpy().astype(np.float64) if hasattr(x, "detach") else np.asarray(x, np.float64)


def _get(o, names):
    for name in names:
        v = getattr(o.data, name, None)
        if v is not None:
            return _arr(v)[0][:3]
    return np.zeros(3)


def _objs(base, catch):
    objs = {}
    for n in (catch.mule, catch.rump, catch.saddle):
        try:
            objs[n] = base.scene.rigid_objects[n]
        except (AttributeError, KeyError):
            objs[n] = base.scene[n]
    return objs


def _hook(env):
    base = env.unwrapped
    catch = _STATE["catch"]
    objs = _objs(base, catch)
    _STATE["objs"] = objs
    import warp as wp

    def compute():
        mo = _arr(objs[catch.mule].data.root_link_pose_w)[0]
        ro = _arr(objs[catch.rump].data.root_link_pose_w)[0]
        Rm, Rr = quat_matrix(mo[3:7]), quat_matrix(ro[3:7])
        lin = ("root_com_lin_vel_w", "root_lin_vel_w", "root_link_lin_vel_w")
        ang = ("root_com_ang_vel_w", "root_ang_vel_w", "root_link_ang_vel_w")
        cr = ro[:3] + Rr @ catch.rump_com
        F, T, Fs = catch.wrench(mo[:3], Rm, _get(objs[catch.mule], lin), _get(objs[catch.mule], ang), cr, Rr,
                                _get(objs[catch.rump], lin), _get(objs[catch.rump], ang))
        cm = mo[:3] + Rm @ catch.mule_com
        Tm = -T - np.cross(cr - cm, F)
        _STATE["wrench"] = {catch.rump: (F, T), catch.saddle: (Fs, np.zeros(3)), catch.mule: (-F - Fs, Tm)}

    first = catch.rump
    for n, obj in objs.items():
        if getattr(obj, "_bucking_hooked", False):
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
        obj._bucking_hooked = True
    if not _STATE["printed"]:
        print(f"BUCKING hook on {list(objs)}; law={catch.law}", flush=True)
        _STATE["printed"] = True


def _reseat(env, saddle_local):
    """Put the hindquarters and saddle back on the mule exactly (seeded shifts move each free body on its own)."""
    base = env.unwrapped
    catch = _STATE["catch"]
    objs = _objs(base, catch)
    import torch

    mo = objs[catch.mule].data.root_link_pose_w
    mo = getattr(mo, "torch", mo).clone()
    Rm = quat_matrix(_arr(mo)[0][3:7])
    for name, off in ((catch.rump, np.zeros(3)), (catch.saddle, np.asarray(saddle_local, float))):
        pose = mo.clone()
        pose[:, :3] += torch.tensor(Rm @ off, dtype=pose.dtype, device=pose.device)
        objs[name].write_root_pose_to_sim_index(root_pose=pose)
        vel = objs[name].data.default_root_vel
        objs[name].write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(getattr(vel, "torch", vel)))


def install(catch, saddle_local):
    """Remember the catch and, on a simulator host, hook every reset: re-seat, then arm the catch."""
    _STATE["catch"] = catch
    _STATE["wrench"] = None
    _STATE["saddle_local"] = saddle_local
    try:
        from yam_arena.inspect_embodiment import YamArenaEmbodiment as cls
    except ImportError:
        return
    if getattr(cls, "_bucking_wrapped", False):
        return
    inner_reset = cls.reset

    @functools.wraps(inner_reset)
    def reset(self, *args, **kwargs):
        c = _STATE["catch"]
        if c is not None:
            c.reset()                       # disarmed through the arena's reset and settle
        result = inner_reset(self, *args, **kwargs)
        _STATE["wrench"] = None
        if c is not None and self.task is not None and all(n in self.task.names for n in (c.mule, c.rump, c.saddle)):
            _hook(self._env)
            try:
                _reseat(self._env, _STATE["saddle_local"])
                base = self._env.unwrapped
                for _ in range(24):         # 0.1 s for the saddle to bed in, catch still disarmed
                    base.scene.write_data_to_sim()
                    base.sim.step(render=False)
                    base.scene.update(dt=base.physics_dt)
            except Exception as e:  # noqa: BLE001
                print(f"BUCKING reseat failed: {e!r}", flush=True)
            c.arm()
            print(f"BUCKING armed after reset", flush=True)
        return result

    cls.reset = reset
    cls._bucking_wrapped = True


def tripped():
    c = _STATE["catch"]
    return bool(c is not None and c.tripped)


def last():
    c = _STATE["catch"]
    return dict(c.last) if c is not None else {}


def take_window():
    """Largest give (mm) since the previous call."""
    c = _STATE["catch"]
    if c is None:
        return 0.0
    v = getattr(c, "win_max", 0.0)
    c.win_max = 0.0
    return v * 1000
