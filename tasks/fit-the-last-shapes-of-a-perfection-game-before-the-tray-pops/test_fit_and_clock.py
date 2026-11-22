"""Offline checks of the shapes, their holes and the clock (pure Python)."""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import perfection_kit as pk  # noqa: E402

CELL = 0.042


def _outline_points(shape, n=90):
    """Points on the piece's outline (piece frame)."""
    pts = []
    for c in pk.SHAPES[shape][0]:
        if c[0] == "box":
            _, cx, cy, sx, sy = c
            for t in np.linspace(-0.5, 0.5, 15):
                pts += [(cx + t * sx, cy - sy / 2), (cx + t * sx, cy + sy / 2),
                        (cx - sx / 2, cy + t * sy), (cx + sx / 2, cy + t * sy)]
        else:
            _, cx, cy, r = c
            pts += [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, n)]
    return [p for p in pts if not pk.in_outline(shape, p[0], p[1], -1e-6) or True]


def _hits_wall(shape, hole, yaw_deg, dx=0.0, dy=0.0):
    walls = pk.hole_strips(hole, CELL)
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    for x, y in _outline_points(shape):
        wx, wy = c * x - s * y + dx, s * x + c * y + dy
        if abs(wx) > CELL / 2 or abs(wy) > CELL / 2:
            return True
        for cx, cy, sx, sy in walls:
            if abs(wx - cx) < sx / 2 - 1e-7 and abs(wy - cy) < sy / 2 - 1e-7:
                return True
    return False


def test_each_shape_fits_its_own_hole_square():
    for shape in pk.SHAPES:
        assert not _hits_wall(shape, shape, 0.0)


def test_no_shape_fits_another_hole_at_any_turn_or_offset():
    for shape in pk.SHAPES:
        for hole in pk.SHAPES:
            if hole == shape:
                continue
            for yaw in range(0, 180, 3):
                for dx in (-0.002, 0.0, 0.002):
                    for dy in (-0.002, 0.0, 0.002):
                        assert _hits_wall(shape, hole, yaw, dx, dy), (shape, hole, yaw, dx, dy)


def test_own_hole_rejects_a_wrong_turn():
    for shape, turn in (("square", 10.0), ("cross", 10.0), ("hotdog", 30.0)):
        assert _hits_wall(shape, shape, turn)


def test_own_hole_rejects_an_offset_beyond_the_clearance():
    for shape in pk.SHAPES:
        assert _hits_wall(shape, shape, 0.0, dx=0.0012)


def test_symmetry_yaw_error():
    assert abs(pk.yaw_error_deg(92.0, 0.0, "square") - 2.0) < 1e-9
    assert abs(pk.yaw_error_deg(178.0, 0.0, "hotdog") + 2.0) < 1e-9
    assert pk.yaw_error_deg(37.0, 0.0, "circle") == 0.0


def test_clock_counts_down_stops_and_pops():
    c = pk.Clock(t_left=2.0)
    c.update(0.0, 0.0, seated=lambda: {"square"})
    c.update(1.0, 0.0)
    assert abs(c.t_left - 1.0) < 1e-9 and not c.stopped
    c.update(1.5, pk.STOP_AT + 0.001, seated=lambda: {"square", "circle"})
    assert c.stopped and c.seated_at_stop == {"square", "circle"}
    c.update(5.0, pk.STOP_AT + 0.001)
    assert not c.popped and abs(c.t_left - 0.5) < 1e-9
    c.update(5.1, 0.0)
    c.update(6.0, 0.0, seated=lambda: set())
    assert c.popped and c.seated_at_pop == set()


def test_switch_short_of_stop_does_not_stop():
    c = pk.Clock(t_left=1.0)
    c.update(0.0, pk.STOP_AT - 0.001)
    c.update(1.2, pk.STOP_AT - 0.001, seated=lambda: set())
    assert c.popped and not c.stopped


def test_pop_force_falls_with_stroke():
    assert abs(pk.pop_force(0.0) - 21.0) < 1e-9
    assert abs(pk.pop_force(pk.POP_STROKE) - 6.0) < 1e-9
