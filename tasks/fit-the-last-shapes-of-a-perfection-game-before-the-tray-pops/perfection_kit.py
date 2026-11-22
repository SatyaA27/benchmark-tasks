"""The clock of a pop-up shape-matching game (Perfection-style), as a pure-Python state machine, plus the shape
geometry shared by the pieces and the holes.

What it models. The real game's clock is a wind-up mechanical timer under a dial, started and stopped by a slide
switch beside the dial. While the switch is at START the clock runs down; moving the switch to STOP halts it; if it
runs out while running, it releases the latch under the spring-loaded tray, which pops up and throws the pieces out.

Here the switch is a slide joint of the game unit (`update()` is given its position, metres from START toward STOP);
the clock stops when the switch has been moved STOP_AT along its travel. At time-out the game reports `popped` and the
task applies the tray's spring force to the tray body (a force between the tray and the unit, as the real spring
pushes): nothing else here moves a body. The game also remembers which pieces sat in their holes at the moment the
switch reached STOP and at the moment of the pop (the task passes a function that reads that from the bodies).

Tuned numbers (all ESTIMATES; a person checks them on the real game with a ruler, a stopwatch and a kitchen scale):
  T_LEFT = 40 s on the clock when the trial starts (the dial set to the 20 mark of its 60); STOP_AT = 8 mm of the
  switch's 12 mm travel stops the clock; the pop spring pushes POP_PRELOAD = 6 N at the top of its 25 mm stroke and
  POP_PRELOAD + POP_K x 25 mm = 21 N when the tray is down.

Shapes (piece sizes, mm; holes are each piece's outline grown by CLEAR a side):
  square 26 x 26; circle 32 across; cross 34 x 34 with 12 mm arms; hot dog 36 long, 16 wide (round ends);
  every piece 6 mm thick with a round knob 8 mm across and 11 mm tall in its middle; no piece fits another's hole.
"""
from __future__ import annotations

import math

T_LEFT = 40.0
STOP_AT = 0.008
POP_PRELOAD, POP_K, POP_STROKE = 6.0, 600.0, 0.025
CLEAR = 0.0005
THICK = 0.006
KNOB_R, KNOB_H = 0.004, 0.011

# Each shape: (components, symmetry in degrees). Component ("box", cx, cy, sx, sy) or ("disc", cx, cy, r), piece frame
# (origin at the middle of its underside), metres.
SHAPES = {
    "square": ([("box", 0.0, 0.0, 0.026, 0.026)], 90.0),
    "circle": ([("disc", 0.0, 0.0, 0.016)], 0.0),
    "cross": ([("box", 0.0, 0.0, 0.034, 0.012), ("box", 0.0, 0.0, 0.012, 0.034)], 90.0),
    "hotdog": ([("box", 0.0, 0.0, 0.020, 0.016), ("disc", -0.010, 0.0, 0.008), ("disc", 0.010, 0.0, 0.008)], 180.0),
}


def yaw_error_deg(piece_yaw_deg, hole_yaw_deg, shape):
    """Smallest turn (deg) that lines the piece up with the hole, given the shape's symmetry."""
    sym = SHAPES[shape][1]
    d = piece_yaw_deg - hole_yaw_deg
    if sym <= 0:
        return 0.0
    return (d + sym / 2) % sym - sym / 2


def hole_strips(shape, cell, step=0.0015):
    """Wall boxes for one hole cut through a square cell of side `cell` (hole centred, hole frame = cell frame):
    a list of (cx, cy, sx, sy). The hole is the shape grown by CLEAR; curved edges become steps no narrower than the
    curve (a step never cuts into the grown outline)."""
    comps = SHAPES[shape][0]
    h = cell / 2
    cuts = {-h, h}
    for c in comps:
        if c[0] == "box":
            _, cx, cy, sx, sy = c
            cuts.update((cx - sx / 2 - CLEAR, cx + sx / 2 + CLEAR))
        else:
            _, cx, cy, r = c
            r += CLEAR
            n = max(2, int(math.ceil(2 * r / step)))
            cuts.update(cx - r + 2 * r * k / n for k in range(n + 1))
    xs = sorted(x for x in cuts if -h <= x <= h)
    walls = []
    prev = None
    for xa, xb in zip(xs[:-1], xs[1:]):
        if xb - xa < 1e-6:
            continue
        ivs = []
        for c in comps:
            if c[0] == "box":
                _, cx, cy, sx, sy = c
                if xa < cx + sx / 2 + CLEAR - 1e-9 and xb > cx - sx / 2 - CLEAR + 1e-9:
                    ivs.append((cy - sy / 2 - CLEAR, cy + sy / 2 + CLEAR))
            else:
                _, cx, cy, r = c
                r += CLEAR
                if xa < cx + r - 1e-9 and xb > cx - r + 1e-9:
                    near = min(max(cx, xa), xb)
                    w = math.sqrt(max(r * r - (near - cx) ** 2, 0.0))
                    ivs.append((cy - w, cy + w))
        ivs.sort()
        merged = []
        for a, b in ivs:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        solid, y = [], -h
        for a, b in merged:
            if a > y:
                solid.append((y, a))
            y = max(y, b)
        if y < h:
            solid.append((y, h))
        key = tuple((round(a, 6), round(b, 6)) for a, b in solid)
        if prev is not None and prev[0] == key:
            prev[2] = xb
            continue
        prev = [key, xa, xb]
        walls.append(prev)
    out = []
    for key, xa, xb in walls:
        for a, b in key:
            if b - a > 1e-6:
                out.append(((xa + xb) / 2, (a + b) / 2, xb - xa, b - a))
    return out


def in_outline(shape, x, y, grow=0.0):
    """Whether the point (piece frame) lies inside the piece's outline grown by `grow`."""
    for c in SHAPES[shape][0]:
        if c[0] == "box":
            _, cx, cy, sx, sy = c
            if abs(x - cx) <= sx / 2 + grow and abs(y - cy) <= sy / 2 + grow:
                return True
        else:
            _, cx, cy, r = c
            if math.hypot(x - cx, y - cy) <= r + grow:
                return True
    return False


class Clock:
    """The game's timer, switch and pop latch."""

    def __init__(self, t_left=T_LEFT):
        self.t_left0 = t_left
        self.reset()

    def reset(self):
        self.t_left = self.t_left0
        self.t = None
        self.stopped = False       # the switch is at STOP now
        self.popped = False
        self.t_pop = None
        self.t_stop = None         # the last time the switch reached STOP
        self.seated_at_stop = frozenset()
        self.seated_at_pop = frozenset()
        self.events = []

    def update(self, t, switch_q, seated=None):
        """One control step: t (s since the trial began), switch_q (m from START toward STOP), seated() -> the set of
        pieces sitting in their own holes now (called only at a stop or at the pop)."""
        stop_now = switch_q >= STOP_AT
        if self.t is not None and not self.popped:
            if not self.stopped:
                self.t_left -= max(0.0, t - self.t)
        self.t = t
        if self.popped:
            return
        if stop_now and not self.stopped:
            self.stopped = True
            self.t_stop = t
            self.seated_at_stop = frozenset(seated() if seated else ())
            self.events.append((round(t, 2), "stop", round(self.t_left, 2), sorted(self.seated_at_stop)))
        elif not stop_now and self.stopped:
            self.stopped = False
            self.events.append((round(t, 2), "start", round(self.t_left, 2)))
        if not self.stopped and self.t_left <= 0.0:
            self.t_left = 0.0
            self.popped = True
            self.t_pop = t
            self.seated_at_pop = frozenset(seated() if seated else ())
            self.events.append((round(t, 2), "pop", sorted(self.seated_at_pop)))

    def dial_used_s(self):
        """Seconds the dial's arrow shows as used (60-second dial)."""
        return 60.0 - self.t_left


def pop_force(q):
    """Upward spring force (N) on the tray at slide position q (m above its locked-down position), once released."""
    return max(0.0, POP_PRELOAD + POP_K * (POP_STROKE - q))
