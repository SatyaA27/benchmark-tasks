"""A die-cast pull-back toy car and its pull-back motor (reusable module).

The car: KiNSMART 2017 Ford GT, 1:38 scale, maker part KT5391. Retail listing: 5 x 2.25 x 1.25 in (127 x 57 x 32 mm)
and 3.84 oz (109 g). Built here from boxes and cylinders: a lower body 127 x 54 x 15 mm (5 to 20 mm above the floor),
a cabin 60 x 40 x 12 mm on top (to 32 mm), and two solid axles, each with two rubber tyres 18 mm across (rear 9 mm
wide, front 7 mm), wheelbase 71 mm, the tyres' outer faces flush with the body sides. Body 95 g, rear axle 8 g, front
axle 6 g. The body width, tyre size, wheelbase, ground clearance and the mass split are estimates from the listing and
the real car's size at 1:38 (a ruler and a kitchen scale check them in a minute).

The motor (behaviour after the public descriptions of pull-back motors; every number is an estimate): rolling the car
backwards on its rear tyres winds a spiral spring; it only winds while the rear wheels really turn backwards, so the car
must be pressed down hard enough that the tyres roll instead of skidding. The wind is kept as metres of rearward tyre
roll, W, up to W_MAX = 0.10 m; past that a clutch slips and the wheels turn on without winding more. Turning the rear
wheels backwards takes F_HOLD = 2.4 N at the tyres (the clutch and the fast winding gear train). Let go, the spring
drives the rear wheels forward through a gear train 15 times slower (RATIO), pushing at the tyres with
ETA * (S0 + K * W) / RATIO, while W falls by the forward roll / RATIO; when W reaches 0 the motor disengages and the
car rolls on freely. Tuned so that 7.6 cm (3 in) of pull runs the car about 2.5 m on a hard floor, inside the
8 to 12 ft a pull-back car is described to run from "a few inches" of pull. Rolling resistance (ROLL, as a fraction of
the car's weight) is axle friction.

How it is simulated: the direction-dependent clutch is the rear axle joint's own friction (F_HOLD * R / 2 plus the
rolling share), plus a forward torque HOLD + drive on the rear axle with the equal and opposite torque on the body,
set before every physics substep from W. Forward, the bias cancels the friction and leaves only the drive; backwards,
friction and bias add up to F_HOLD. W is updated from the rear axle's measured angle; no speed is ever written to the
car. install(env) hooks the car's write_data_to_sim (as an accepted task does for its lock model); restore(env)
clears W at a reset. spawn_settings() keeps the car awake and lets its wheels spin past the default joint speed cap.

Tuned numbers: R 0.009 m, W_MAX 0.10 m, F_HOLD 2.4 N, S0 1.2 N, K 12 N/m, ETA 0.75, RATIO 15, ROLL 0.035, rear
armature 1e-6 kg m^2. car(name, pos, heading_deg): pos is the middle of the wheelbase at axle height. Metres, kg.
"""
from __future__ import annotations

import math

import numpy as np
from yam_joints import Hinge, Link, Mechanism
from yam_parts import box, cyl

G = 9.81
R = 0.009  # tyre radius
WHEELBASE = 0.071
LENGTH, WIDTH, HEIGHT = 0.127, 0.054, 0.032
CLEARANCE = 0.005
BODY_TOP = 0.020  # top of the lower body above the floor
CABIN = (0.060, 0.040)  # cabin length and width; its roof is at HEIGHT
CABIN_X = -0.004
MASS_BODY, MASS_REAR, MASS_FRONT = 0.095, 0.008, 0.006
MASS = MASS_BODY + MASS_REAR + MASS_FRONT
REAR_W, FRONT_W = 0.009, 0.007

W_MAX = 0.10  # m of rearward roll before the clutch slips
F_HOLD = 2.4  # N at the tyres to turn the rear wheels backwards
S0, K = 1.2, 12.0  # spring force referred to the tyres: S0 + K * W (N)
ETA = 0.75  # gear train efficiency while driving
RATIO = 15.0  # winding is this many times faster than unwinding
ROLL = 0.035  # rolling resistance, fraction of the weight
ROLL_TORQUE = ROLL * MASS * G * R  # N m, over both axles
REAR_ROLL, FRONT_ROLL = 0.55 * ROLL_TORQUE, 0.45 * ROLL_TORQUE
HOLD = F_HOLD * R / 2  # N m: half the clutch's resistance, cancelled by the bias when driving forward
REAR_FRICTION = HOLD + REAR_ROLL
REAR_ARMATURE = 1e-6  # kg m^2, the gear train's inertia seen at the axle (estimate)
MAX_SPIN_DEG = 30000.0  # deg/s allowed to the wheels (about 520 rad/s; 0.9 m/s needs 100 rad/s)
RED, BLACK, GLASS, GREY = (0.80, 0.07, 0.05), (0.06, 0.06, 0.06), (0.10, 0.12, 0.16), (0.45, 0.45, 0.47)


def car(name="car", pos=(0.2, 0.0, 0.76), heading_deg=90.0):
    """The Mechanism: base = body (origin at the middle of the wheelbase, axle height), links rear and front axles."""
    zb = (CLEARANCE + BODY_TOP) / 2 - R
    zc = (BODY_TOP + HEIGHT) / 2 - R
    xr, xf = -WHEELBASE / 2, WHEELBASE / 2
    body = [box((0, 0, zb), (LENGTH, WIDTH, BODY_TOP - CLEARANCE), color=RED),
            box((CABIN_X, 0, zc), (CABIN[0], CABIN[1], HEIGHT - BODY_TOP), color=GLASS)]

    def axle(x, width):
        y = WIDTH / 2 - width / 2
        return [cyl((x, y, 0), R, width, axis="y", color=BLACK), cyl((x, -y, 0), R, width, axis="y", color=BLACK),
                cyl((x, 0, 0), 0.0015, 2 * y - width, axis="y", color=GREY)]

    rear = Link("rear", axle(xr, REAR_W), Hinge("y", (xr, 0, 0), -100000, 100000, armature=REAR_ARMATURE,
                                                 static_friction=REAR_FRICTION, dynamic_friction=REAR_FRICTION),
                MASS_REAR, friction=(0.9, 0.8))
    front = Link("front", axle(xf, FRONT_W), Hinge("y", (xf, 0, 0), -100000, 100000,
                                                    static_friction=FRONT_ROLL, dynamic_friction=FRONT_ROLL),
                 MASS_FRONT, friction=(0.9, 0.8))
    return Mechanism(name, pos, body, [rear, front], mass=MASS_BODY, color=RED, yaw=math.radians(heading_deg),
                     friction=(0.6, 0.5), contact_offset=0.001)


class Motor:
    """The spring's state: W (m of rearward roll stored) from the rear axle's angle, and the torque it gives."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.wind, self.q, self.peak, self.rolled_back = 0.0, None, 0.0, 0.0

    def step(self, q):
        if self.q is None:
            self.q = q
            return
        d = (q - self.q + math.pi) % (2 * math.pi) - math.pi  # this substep's turn, forward positive
        self.q = q
        if d < 0:
            self.rolled_back -= d * R
            self.wind = min(self.wind - d * R, W_MAX)
        elif self.wind > 0:
            self.wind = max(self.wind - d * R / RATIO, 0.0)
        self.peak = max(self.peak, self.wind)

    def drive(self):
        """Forward torque of the spring at the rear axle (N m), 0 when unwound."""
        return ETA * (S0 + K * self.wind) * R / RATIO if self.wind > 0 else 0.0

    def bias(self):
        return HOLD + self.drive()


MOTOR = Motor()
CAR_NAME = "car"


def _np(x):
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def spawn_settings(name=CAR_NAME):
    """Wrap the kit's mechanism spawner (before the scene is built) so the car's articulation never sleeps and its
    wheels may spin as fast as a real car's (USD authored at spawn time only)."""
    import yam_joints as J
    if getattr(J._spawn_mechanism, "_pullback", False):
        return
    inner = J._spawn_mechanism

    def spawn(prim_path, cfg, *args, **kwargs):
        prim = inner(prim_path, cfg, *args, **kwargs)
        if getattr(cfg, "mech_name", "") == name:
            try:
                from pxr import PhysxSchema
                stage = prim.GetStage()
                art = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                art.CreateSleepThresholdAttr().Set(0.0)
                art.CreateStabilizationThresholdAttr().Set(0.0)
                for link in ("rear", "front"):
                    body = stage.GetPrimAtPath(f"{prim_path}/{link}")
                    if body.IsValid():
                        PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateMaxAngularVelocityAttr().Set(MAX_SPIN_DEG)
                    joint = stage.GetPrimAtPath(f"{prim_path}/joints/{link}_joint")
                    if joint.IsValid():
                        PhysxSchema.PhysxJointAPI.Apply(joint).CreateMaxJointVelocityAttr().Set(MAX_SPIN_DEG)
                print(f"PULLBACK_SPAWN ok {prim_path}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"PULLBACK_SPAWN failed {e!r}", flush=True)
        return prim

    spawn._pullback = True
    J._spawn_mechanism = spawn


def restore(environment):
    """At a reset: the spring starts unwound."""
    MOTOR.clear()


def install(environment, name=CAR_NAME):
    """Before every physics substep: update W from the rear axle and set the motor's torque pair. Idempotent."""
    asset = environment.unwrapped.scene[name]
    if getattr(asset, "_pullback_hooked", False):
        return
    import torch
    jid = list(asset.joint_names).index("rear_joint")
    bodies = list(asset.body_names)
    ids = [bodies.index("rear"), bodies.index("base")]
    zeros = torch.zeros((1, 2, 3), dtype=torch.float32, device=asset.device)
    orig = asset.write_data_to_sim

    def write():
        try:
            MOTOR.step(float(_np(asset.data.joint_pos)[0, jid]))
            x, y, z, w = _np(asset.data.root_pose_w)[0, 3:7]
            axis = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])
            tau = MOTOR.bias() * axis
            torques = torch.tensor([[tau, -tau]], dtype=torch.float32, device=asset.device)
            asset.set_external_force_and_torque(zeros, torques, body_ids=ids, is_global=True)
        except Exception as e:  # noqa: BLE001
            if not getattr(asset, "_pullback_err", False):
                print(f"PULLBACK_HOOK error {e!r}", flush=True)
                asset._pullback_err = True
        orig()

    asset.write_data_to_sim = write
    asset._pullback_hooked = True
    print("PULLBACK_HOOK installed", flush=True)
