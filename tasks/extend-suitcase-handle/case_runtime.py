"""The release button's coupling to the two rails' lock pins, and the pins' holes (pure NumPy; the simulator hook is
installed lazily on the GPU host only).

The handle is a three-section telescope on each side (outer tube fixed in the shell, middle tube, inner tube with the
grip on top). The two middle tubes move as one body and so do the two inner tubes with the grip (joints "mid" and
"inner", metres, negative = drawn out of the case). The release button in the grip (joint "button", metres pressed in)
works two rods, one down each inner tube, each ending in a spring pin at the inner tube's foot. A pin that is out
passes through a hole in its middle tube into a hole in its outer tube and so locks the handle; the holes line up at
the closed position and at the two working heights. The rods are not modelled as bodies: each pin is in or out from
the button's travel (the left rod takes up its play first, PIN_IN), and a pin that is out drops into its holes when
they line up (within CAPTURE), which this module turns into the joints' limits (PLAY either way). Pressed far enough
for both pins to clear, nothing holds the handle but the tubes' friction (the joints' own friction).
"""
from __future__ import annotations

import math

import numpy as np

STAGE = 0.260  # each section's travel; the handle's full extension is 2 x STAGE (estimate from the 52 cm listing)
# Locked configurations (inner, mid) joint values: closed, first working height, full extension
CONFIGS = {"closed": (0.0, 0.0), "first": (-STAGE, 0.0), "full": (-STAGE, -STAGE)}
CAPTURE = 0.0015  # a pin that is out drops into its holes within this of lining up
PLAY = 0.0005  # pin in its hole: the handle's play either way
# Button travel at which each pin is clear of its holes (pressed in, metres); the left rod's play is taken up first.
PIN_CLEAR = {"left": 0.0030, "right": 0.0034}
PIN_HYST = 0.0003  # a clear pin springs out again once the button is back by this much
BUTTON_TRAVEL = 0.004
NAMES = ("mid", "inner", "button")


class Model:
    def __init__(self):
        self.clear()

    def clear(self):
        self.pin_out = {"left": True, "right": True}
        self.locked = None  # name of the configuration the pins hold, or None
        self.limits_dirty = True

    def full_limits(self):
        return {"mid": (-STAGE, 0.0), "inner": (-STAGE, 0.0)}

    def limits(self):
        if self.locked is None:
            return self.full_limits()
        si, sm = CONFIGS[self.locked]
        return {"inner": (max(-STAGE, si - PLAY), min(0.0, si + PLAY)), "mid": (max(-STAGE, sm - PLAY), min(0.0, sm + PLAY))}

    def update(self, sm, si, b):
        """Joint values in metres; returns the joints' limits when they change, else None."""
        for side, clear in PIN_CLEAR.items():
            if self.pin_out[side] and b >= clear:
                self.pin_out[side] = False
            elif not self.pin_out[side] and b <= clear - PIN_HYST:
                self.pin_out[side] = True
        before = self.locked
        if not any(self.pin_out.values()):
            self.locked = None
        elif self.locked is None:
            for name, (ci, cm) in CONFIGS.items():
                if abs(si - ci) <= CAPTURE and abs(sm - cm) <= CAPTURE:
                    self.locked = name
                    break
        if self.locked != before or self.limits_dirty:
            self.limits_dirty = False
            return self.limits()
        return None

    def readout(self):
        return {"locked": self.locked, "pin_left": self.pin_out["left"], "pin_right": self.pin_out["right"]}


MODEL = Model()
CASE_NAME = "suitcase"


def _np(x):
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _ids(asset):
    names = list(asset.joint_names)
    return {n: names.index(n + "_joint") for n in NAMES}


def _write_limits(asset, ids, lim):
    import torch
    order = ["mid", "inner"]
    t = torch.tensor([[list(lim[n]) for n in order]], dtype=torch.float32, device=asset.device)
    asset.write_joint_position_limit_to_sim_index(limits=t, joint_ids=[ids[n] for n in order], warn_limit_violation=False)


def restore(environment):
    """At a reset: free the joints and put the handle back closed and still (the scene's reset leaves an articulated
    object's joints where they were)."""
    import torch
    try:
        asset = environment.unwrapped.scene[CASE_NAME]
    except Exception:  # noqa: BLE001
        return
    ids = _ids(asset)
    _write_limits(asset, ids, MODEL.full_limits())
    order = [ids[n] for n in NAMES]
    asset.write_joint_position_to_sim_index(position=torch.zeros((1, len(order)), device=asset.device), joint_ids=order)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros((1, len(order)), device=asset.device), joint_ids=order)
    MODEL.clear()


def install(environment):
    """Run the model before every physics substep (Isaac Lab calls ``write_data_to_sim`` then). Idempotent."""
    asset = environment.unwrapped.scene[CASE_NAME]
    if getattr(asset, "_case_hooked", False):
        return
    ids = _ids(asset)
    orig = asset.write_data_to_sim

    def write():
        q = _np(asset.data.joint_pos)[0]
        lim = MODEL.update(float(q[ids["mid"]]), float(q[ids["inner"]]), float(q[ids["button"]]))
        if lim is not None:
            _write_limits(asset, ids, lim)
        orig()

    asset.write_data_to_sim = write
    asset._case_hooked = True
