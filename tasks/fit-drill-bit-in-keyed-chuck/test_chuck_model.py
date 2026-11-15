"""CPU tests of the chuck model (chuck_runtime.py) with synthetic poses: no simulator."""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import importlib  # noqa: E402

MOD = next(p for p in pathlib.Path(__file__).resolve().parent.glob("*.py")
           if p.name not in ("chuck_runtime.py", "yam_parts.py") and not p.name.startswith("test_"))
T = importlib.import_module(MOD.stem)
CR = T.CR
I3, Z3 = np.eye(3), np.zeros(3)
DT = 1 / 240


def rz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


class Rig:
    """Drill at its start; the key seated in the top hole (or lifted by `lift`), turned at `w` rad/s clockwise."""

    def __init__(self, bit_depth=0.028, bit_drop=0.00243):
        CR.MODEL.clear()
        self.pd = T.DRILL_START.copy()
        self.hole = self.pd + np.array((0.0, T.HOLE_Y, T.HOLE_TOP))
        self.pb = self.pd + np.array((0.0, bit_depth, -bit_drop))
        self.ang = 0.0

    def step(self, w=1.0, lift=0.0, held=True, n=1):
        acc = {}
        for _ in range(n):
            self.ang -= w * DT
            pk = self.hole - np.array((0, 0, T.PINION_Z0 - lift))
            bar = pk + np.array((0, 0, T.BAR_Z))
            fr = {"drill": (self.pd, I3, Z3, Z3), "bit": (self.pb, I3, Z3, Z3),
                  "key": (pk, rz(self.ang), Z3, np.array((0, 0, -w)))}
            acc = {}
            cmd = CR.MODEL.update(fr, acc, [(bar, 0.08 if held else 1.0)])
            if cmd is not None:
                self.pb = cmd[0]
        return acc

    def turns(self, k, **kw):
        return self.step(n=int(round(2 * math.pi * k / DT)), **kw)


def test_lifted_key_does_not_mesh():
    r = Rig()
    r.turns(0.5, lift=0.002)
    assert not CR.MODEL.engaged and CR.MODEL.theta == 0.0


def test_jaws_close_half_a_millimetre_a_turn_and_centre_the_shank():
    r = Rig()
    r.turns(1.0)
    assert abs(CR.MODEL.d() - (T.OPEN_0 - 0.0005)) < 2e-5
    r.turns(1.9)
    assert not CR.MODEL.gripped and abs(r.pb[2] - r.pd[2]) < 0.0004  # pushed most of the way to the axis
    r.turns(0.3)
    assert CR.MODEL.gripped and abs(r.pb[2] - r.pd[2]) < 1e-6 and CR.MODEL.d() == 2 * CR.BIT_R


def test_once_the_jaws_reach_the_shank_it_stays_gripped_while_the_key_turns_on():
    r = Rig()
    r.turns(2.9)
    seen = []
    for _ in range(int(round(2 * math.pi * 0.3 / DT))):
        r.step()
        seen.append(CR.MODEL.gripped)
    first = seen.index(True)
    assert all(seen[first:]) and CR.MODEL.tightness() > 0.1


def test_tightness_rises_past_contact_and_the_teeth_push_the_key_out():
    r = Rig()
    r.turns(3.0)
    acc = r.turns(0.1)
    t1 = CR.MODEL.tightness()
    assert 0.2 < t1 < 0.5
    assert acc["key"][0][2] > 0 and abs(acc["key"][0][2] - CR.C_LIFT * CR.MODEL.torque) < 1e-6
    assert abs(acc["drill"][0][2] + acc["key"][0][2]) < 1e-9  # equal and opposite on the drill


def test_let_go_the_chuck_stays_tight():
    r = Rig()
    r.turns(3.12)
    th, t1 = CR.MODEL.theta, CR.MODEL.tightness()
    r.step(w=0.0, held=False, n=240)
    assert CR.MODEL.theta == th and CR.MODEL.tightness() == t1 and CR.MODEL.gripped


def test_skipping_teeth_lose_no_tightness():
    r = Rig()
    r.turns(3.1)
    t1 = CR.MODEL.tightness()
    r.step(lift=0.0015, n=10)
    assert not CR.MODEL.engaged and CR.MODEL.tightness() == t1


def test_turned_back_the_jaws_let_go():
    r = Rig()
    r.turns(3.1)
    assert CR.MODEL.gripped
    r.turns(0.5, w=-1.0)
    assert not CR.MODEL.gripped and CR.MODEL.d() > 2 * CR.BIT_R


def test_without_a_bit_the_jaws_close_on_nothing():
    r = Rig(bit_depth=-0.050)
    r.turns(3.3)
    assert not CR.MODEL.gripped and CR.MODEL.d() < 2 * CR.BIT_R and CR.MODEL.tightness() == 0.0


def test_a_shallow_shank_is_gripped_but_not_captured():
    r = Rig(bit_depth=0.010)
    r.turns(3.2)
    assert CR.MODEL.gripped and CR.MODEL.depth < T.CAPTURE_MIN


def test_closing_jaws_hold_the_shank_along_the_axis_unless_a_hand_pushes_it():
    r = Rig()
    r.turns(1.0)  # the jaws have moved: the model holds the shank
    fr = {"drill": (r.pd, I3, Z3, Z3), "bit": (r.pb, I3, np.array((0, -0.02, 0)), Z3),
          "key": (r.hole - np.array((0, 0, T.PINION_Z0 - 0.003)), I3, Z3, Z3)}
    cmd = CR.MODEL.update(fr, {}, [])
    assert cmd is not None and abs(cmd[2][1]) < 1e-12  # with no hand at the bit, even a fast slide is stopped
    hand = (r.pd + np.array((0.0, -0.050, 0.010)), 0.0)  # pads closed on the flutes, 50 mm out from the nose
    fr["bit"] = (r.pb, I3, np.array((0, 0.02, 0)), Z3)
    cmd = CR.MODEL.update(fr, {}, [hand])
    assert cmd is None or abs(cmd[2][1] - 0.02) < 1e-9  # a hand at the bit pushes it on in


def test_closing_jaws_keep_the_shank_at_its_depth_against_slow_creep():
    r = Rig(bit_depth=0.023)
    r.turns(1.0)  # the jaws have moved: the model holds the shank
    d0 = r.pb[1] - r.pd[1]
    pk = r.hole - np.array((0, 0, T.PINION_Z0 - 0.003))
    for _ in range(240):  # contact corrections nudge it 0.02 mm out every substep, with no velocity
        r.pb = r.pb - np.array((0.0, 0.00002, 0.0))
        fr = {"drill": (r.pd, I3, Z3, Z3), "bit": (r.pb, I3, Z3, Z3), "key": (pk, I3, Z3, Z3)}
        cmd = CR.MODEL.update(fr, {}, [])
        assert cmd is not None
        r.pb = cmd[0]
    assert abs((r.pb[1] - r.pd[1]) - d0) < 1e-6


def test_a_shallow_tipped_shank_is_clamped_crooked():
    r = Rig(bit_depth=0.010)
    a = math.radians(5.0)
    tipped = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    CR.MODEL.clear()
    for _ in range(int(round(2 * math.pi * 3.2 / DT))):
        r.ang -= DT
        pk = r.hole - np.array((0, 0, T.PINION_Z0))
        fr = {"drill": (r.pd, I3, Z3, Z3), "bit": (r.pb, tipped, Z3, Z3), "key": (pk, rz(r.ang), Z3, np.array((0, 0, -1.0)))}
        cmd = CR.MODEL.update(fr, {}, [(pk + np.array((0, 0, T.BAR_Z)), 0.08)])
        if cmd is not None:
            r.pb, tipped = cmd[0], cmd[1]
    assert CR.MODEL.gripped and CR.MODEL.runout(fr, T.RUNOUT_AT) > T.RUNOUT_MAX
