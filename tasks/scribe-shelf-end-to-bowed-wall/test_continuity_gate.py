"""The 2 mm continuity gate on the primary score (CPU only: yam_arena installed, no simulator):

    python -m pytest -v -p no:cacheprovider test_continuity_gate.py

Every case calls the task module's own scorer on all four layouts: line_credit() (the gate), line_gap() (the line's
longest gap) and score() (the primary score), with the compass hung back on its stand and the hands away. Each set of
marks has every station on target and the bend met, so line_quality() alone would give full line credit; only the
gate decides. Isolated station marks and a gap over 2 mm get line credit 0 and score 0; a continuous line within the
profile tolerance and a gap of 2 mm or less keep line credit 1.0 and score 1.0.
"""
from pathlib import Path

import pytest
from yam_arena.task_file import load

T = load(Path(__file__).with_name("scribe_shelf_end_to_bowed_wall.py")).module
LAYOUTS = [pytest.param(k, id=f"layout{k}") for k in range(len(T.LAYOUT_FRONT_X))]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Offline scoring: no live trial, so score() reads TEST_STROKES; hands away."""
    monkeypatch.setitem(T._RUN, "t", None)
    monkeypatch.setattr(T, "TEST_GRIPPERS", None)
    monkeypatch.setattr(T, "TEST_STROKES", None)
    monkeypatch.setattr(T, "TEST_LAYOUT", None)


def gate(k, strokes, monkeypatch):
    """(line credit without the gate, line credit, longest gap in metres, primary score) for these marks."""
    monkeypatch.setattr(T, "TEST_LAYOUT", k)
    monkeypatch.setattr(T, "TEST_STROKES", strokes)
    cap, gap = T.line_credit(strokes, k)
    return T.line_quality(strokes, k)[0], cap, gap, T.score(T.goal_objects(k))


def lifted_to_measure(k, target, at=0.061):
    """Layout k's line with the pencil lifted at `at` from the front edge for the longest stretch whose gap, measured
    straight across the board by line_gap(), is at most `target` metres."""
    lo, hi = 0.0, target
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if T.line_gap(T.broken_strokes(k, mid, at)) <= target else (lo, mid)
    return T.broken_strokes(k, lo, at)


def test_gap_max_is_2_mm():
    assert T.GAP_MAX == 0.0020


@pytest.mark.parametrize("k", LAYOUTS)
def test_isolated_station_marks_get_zero_line_credit(k, monkeypatch):
    """A single touch on target at each of the 14 stations, nothing between them."""
    dots = T.station_dots(k)
    assert len(dots) == 14 and all(len(d) == 1 for d in dots)
    alone, cap, gap, score = gate(k, dots, monkeypatch)
    assert alone == 1.0                                   # the stations alone read on target
    assert gap > T.GAP_MAX and cap == 0.0 and score == 0.0


GAPS_OVER = {
    "lifted 2.1 mm mid-line": lambda k: T.broken_strokes(k, 0.0021),
    "starts 2.5 mm in from the front edge": lambda k: [T.ideal_line(k, 0.0, 0.0025, T.SHELF_D)],
    "stops 2.5 mm short of the back edge": lambda k: [T.ideal_line(k, 0.0, 0.0, T.SHELF_D - 0.0025)],
}


@pytest.mark.parametrize("k", LAYOUTS)
@pytest.mark.parametrize("marks", GAPS_OVER)
def test_a_gap_over_2_mm_gets_zero_line_credit(marks, k, monkeypatch):
    alone, cap, gap, score = gate(k, GAPS_OVER[marks](k), monkeypatch)
    assert alone == 1.0
    assert 0.0020 < gap < 0.0026 and cap == 0.0 and score == 0.0


@pytest.mark.parametrize("k", LAYOUTS)
@pytest.mark.parametrize("bias", [0.0, 0.0004, -0.0004], ids=["on target", "0.4 mm off", "-0.4 mm off"])
def test_a_continuous_line_within_tolerance_gets_full_line_credit(bias, k, monkeypatch):
    alone, cap, gap, score = gate(k, T.nominal_strokes(k, bias), monkeypatch)
    assert gap < 1e-9 and cap == alone == 1.0 and score == 1.0


@pytest.mark.parametrize("k", LAYOUTS)
@pytest.mark.parametrize("marks", ["lifted 1.9 mm", "lifted to measure 2.0 mm"])
def test_a_gap_of_2_mm_or_less_keeps_full_line_credit(marks, k, monkeypatch):
    strokes = T.broken_strokes(k, 0.0019) if marks == "lifted 1.9 mm" else lifted_to_measure(k, T.GAP_MAX)
    alone, cap, gap, score = gate(k, strokes, monkeypatch)
    assert len(strokes) == 2 and 0.0018 < gap <= T.GAP_MAX
    assert cap == alone == 1.0 and score == 1.0
    if marks == "lifted to measure 2.0 mm":
        assert gap > T.GAP_MAX - 1e-9                    # at the limit itself


@pytest.mark.parametrize("k", LAYOUTS)
def test_the_task_modules_selftest(k):
    """continuity_gate_selftest() sits next to the scorer in the task module; it scores four sets of marks."""
    results = T.continuity_gate_selftest(k)
    assert [(cap, score) for _, _, cap, score in results] == [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0), (1.0, 1.0)]
    assert T.TEST_STROKES is None and T.TEST_LAYOUT is None   # it puts the module back as it found it
