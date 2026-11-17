# Scribe a shelf end to an out-of-flat wall with a pencil compass

Scribing copies an out-of-flat wall's line onto a board with a locked pencil compass, so that, cut to that line, the board fits the wall with no gap.

Here a pine shelf lies on the table with its end against a fixed wall section whose face has two humps and a hollow (up to 5.1 mm of gap along the shelf's end). One hand holds the shelf; the other takes the compass off its stand, scribes one unbroken line across the shelf's end from its front edge to its back edge, and puts the compass down off the shelf.

Category: Impact on jobs. Done by carpenters (O*NET 47-2031.00: "Measure and mark cutting lines on materials, using a ruler, pencil, chalk, and marking gauge."), including finish carpenters and cabinet and countertop installers.

**Layouts by seed.** The wall never changes; layout = trial seed mod 4 sets where along it the shelf lies, so each trial copies a different stretch of wall. A line copied from one layout scores 0 on the other three, and no straight line scores on any.

## Revision: the pencil must touch and press

Review TK-C9HWSV: "Construct validity (not met): The simulated pencil records a line up to 0.2 mm above the board with no force threshold, allowing marking without the physical work of writing. ... Require contact and a realistic minimum writing force to record a mark, retaining the existing continuity and profile-accuracy checks."

Changed: the pencil marks only while its lead touches the board (lowest point at most 0.05 above the top face, the contact tolerance; it was 0.2) AND presses on it with at least **0.5 N**, the simulator's contact force between the lead and the board, read every physics step (`marking()` line 79, `lead_force()` 89, the reader `_track()` 342). 0.5 N is a light writing force: pen-tip force on paper averages 0.95 N (SD 0.29) in 8-year-olds, 0.77 N (SD 0.36) with handwriting difficulties (Lin et al. 2022). Stations, targets, bend, continuity gate and tiers are unchanged. In a real trial the check is the pencil line visible on the board.

## Scoring

Millimetres. The scorer is the first code in `scribe_shelf_end_to_bowed_wall.py`, right after the imports (lines 30-281).

**Stations.** Fourteen lines square to the shelf's end, at 7, 17, 27, ... 137 from its front edge. At each, the reading is the pencil line's distance from the shelf's end to the centre of the line. Crossings of a station within 1 of each other read as one band, at its centre; more than 1 apart, the station is unmarked. A single touch counts at a station within 0.25 of it. **Target:** 10.0 (the compass setting) minus the gap between the wall and the shelf's end there (tables below).

**Bend.** Each layout has three gate stations (table below). No line credit unless all three are marked and the middle reading lies at least 1.5 off the straight line through the outer two, on the target's side. A ruler line fails this on every layout.

**One unbroken line (the continuity gate).** The line must run from the front edge to the back edge (0 to 140 across the shelf's depth) with no gap longer than 2.0. Follow it through the marks read at the stations; wherever it breaks, measure the unmarked stretch straight across the board from the end of one mark to the nearest point of the next. The stretches from the front edge to the line's start and from its end to the back edge count too. The longest stretch is the gap. The route may step through any mark on the board but must take in every mark read at a station. Why 2.0: a scribed line is drawn in one pass and the cut follows it by eye; a good line breaks only where the lead skips, for a millimetre or so. 2.0 is four HB line widths and a fifth of the station spacing.

In the code: `GAP_MAX = 0.0020` (line 47); `score()` (line 55), the primary score, takes its only line credit from `line_credit()` (line 64), which gives 0 when the gap is over GAP_MAX (line 108); `line_gap()` (112) and `Marks.gap()` (182) measure the gap; `success()` (75) is a score of 1.0.

**Continuity gate test.** `continuity_gate_selftest()` (line 117, beside the scorer) scores four sets of marks through `score()`. `test_continuity_gate.py` runs it and the cases below on all four layouts: 41 passed (`python -m pytest -v -p no:cacheprovider test_continuity_gate.py`). Every set has all 14 stations on target and the bend met, so without the gate each would get line credit 1.0; the compass hangs on its stand, hands away.

| Test | Marks | Gap | Line credit = score |
|---|---|---|---|
| `test_isolated_station_marks_get_zero_line_credit` | a touch at each station, nothing between | over 5 | 0 |
| `test_a_gap_over_2_mm_gets_zero_line_credit` | line lifted 2.1 mid-line; starting 2.5 in; stopping 2.5 short | 2.10-2.11; 2.5; 2.5 | 0 |
| `test_a_continuous_line_within_tolerance_gets_full_line_credit` | one line on target; 0.4 off either way | 0 | 1.0 |
| `test_a_gap_of_2_mm_or_less_keeps_full_line_credit` | line lifted 1.9; lifted to measure exactly 2.0 | 1.90-1.91; 2.0 | 1.0 |
| `test_the_task_modules_selftest` | touches; 2.1 lift; 1.9 lift; one line | over 5; 2.1; 1.9; 0 | 0; 0; 1.0; 1.0 |

With the gate taken out of `line_credit()`, or with `score()` using `line_quality()`'s credit directly, 20 of the 41 fail.

**Pencil force test** (`test_red_team.py`, step by step through `score()`): the whole line on target, lead 0.1 above the board, no force: nothing recorded, 0, on all four layouts (at 1.5 N: 1.0); on the face at 0.3 or 0.49 N: 0; at 0.5 or 3.0 N: 1.0; 0.2 N over 3.5 of the line breaks it (0); contacts at the steel point or a leg add no force. Without the force condition 4 of these fail.

| Line credit | One unbroken line (gap at most 2.0), the bend met, and stations within tolerance of their targets |
|---|---|
| 1.0 | all 14 within 0.5 |
| 0.5 | at least 12 within 1.0 |
| 0.25 | at least 10 within 2.0 |
| 0 | otherwise: a gap over 2.0 (separate touches, dashes, or a line stopping short of an edge), the bend not met, or fewer stations within tolerance |

**Finished:** the compass put down clear of the shelf and the wall (its bounding box at least 2 from theirs; hanging it on its stand does this); every gripper TCP at least 30 from the compass's and the shelf's bounding boxes; the compass slower than 10 mm/s and 0.5 rad/s, the shelf slower than 2 mm/s and 0.05 rad/s.

| Score | End of trial, with no sticky failure |
|---|---|
| 1.0 | line credit 1.0, and finished for the last 1 s |
| 0.75 | line credit 1.0, but not finished (compass still held, on the shelf or moving, or a hand on the shelf) |
| 0.5, 0.25 | that line credit |
| 0 | otherwise |

**Sticky failures** (0 for the rest of the trial): the shelf moved more than 30 across the table from its start; the shelf or the compass more than 20 below the table top. Success is a score of 1.0.

**In simulation.** The scorer records the lead's lowest point every control step, in the shelf's frame, while the pencil marks: the lead on the top face (0.05 above to 2 below it, over the board) pressing with at least 0.5 N (PhysX contact force at the lead, read every physics step, averaged per control step). Consecutive marking steps make one stroke; a lift, a step under 0.5 N or an unseen step starts another. Readings are where strokes cross the stations; for the gap, a stroke is the straight pieces between its points. The drawn graphite dots follow the same rule and are only a picture.

**In a real trial.** The line is the graphite visible on the board under the loupe; where the pencil rode above it or skimmed too lightly to leave a line, there is a gap. Before the trial, take the layout's targets from the table (or measure the wall's gap at each station with feeler gauges). After it, tick the 14 stations with a combination square. Find the gap under a 10x loupe with a steel rule graduated in 0.5 (front-edge stretch, every break, back-edge stretch); over 2.0, the line credit is 0. Otherwise read each station: rule's zero on a square block held to the shelf's end, centre of the line to 0.1.

## Layouts and targets

The shelf spans 140 forward from its front edge and 300 right from its end, which starts 0.05 from the wall's most proud point along its stretch.

| Layout | Front edge, forward | Shelf end, left of midline | Touches the wall at, from the front edge | Gate stations | Target bend |
|---|---|---|---|---|---|
| 0 | 120 | 207.97 | 27.8 | 27, 87, 137 | middle 2.64 nearer the end |
| 1 | 147 | 207.97 | 0.8 | 7, 77, 137 | middle 4.32 nearer the end |
| 2 | 174 | 208.38 | 128.7 | 7, 57, 127 | middle 3.12 nearer the end |
| 3 | 201 | 208.38 | 101.7 | 37, 97, 137 | middle 2.70 farther from the end |

Targets (the line's distance from the shelf's end) by station:

| Station | 7 | 17 | 27 | 37 | 47 | 57 | 67 | 77 | 87 | 97 | 107 | 117 | 127 | 137 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Layout 0 | 8.44 | 9.50 | 10.00 | 9.62 | 8.56 | 7.33 | 6.32 | 5.63 | 5.19 | 4.94 | 4.86 | 5.00 | 5.39 | 6.03 |
| Layout 1 | 9.82 | 8.92 | 7.69 | 6.59 | 5.81 | 5.30 | 5.00 | 4.87 | 4.94 | 5.24 | 5.81 | 6.63 | 7.63 | 8.64 |
| Layout 2 | 7.30 | 6.42 | 5.85 | 5.48 | 5.30 | 5.31 | 5.54 | 6.03 | 6.77 | 7.73 | 8.76 | 9.61 | 9.99 | 9.81 |
| Layout 3 | 5.57 | 5.34 | 5.28 | 5.44 | 5.85 | 6.52 | 7.43 | 8.46 | 9.39 | 9.94 | 9.92 | 9.42 | 8.80 | 8.44 |

## Run

With the official simulator installed, from this folder (zero object jitter, 20 Hz):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=scribe_shelf_end_to_bowed_wall.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --seed 0 \
  --instruction "Holding the shelf against the wall, scribe its end to the wall: run the compass from the shelf's front edge to its back edge in one unbroken line, point on the wall, pencil on the shelf, then put the compass down off the shelf and wall." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ./demo-run
```

The model run uses `--policy dream`. The trial seed is the CRC-32 of "eval_seed:scene_seed:epoch"; with one epoch, `--seed 1` gives layout 0, `--seed 3` layout 1, `--seed 0` layout 2, `--seed 2` layout 3. CPU checks: `python -m pytest -q -p no:cacheprovider` here, with yam_arena installed. Runtime: at most 2,400 steps (120 s); the demonstration's plan is 39.0 s, plus 1 s to success.

## Setup, demonstration, fidelity

Objects, sizes, materials and positions: task.json's setup, which gives the wall's profile. The offset every 5 mm from the wall's front end, as cut: 0.29 0.49 0.78 1.16 1.64 2.19 2.77 3.30 3.71 3.94 3.96 3.74 3.33 2.79 2.17 1.55 0.98 0.48 0.06 -0.27 -0.54 -0.75 -0.92 -1.04 -1.12 -1.16 -1.14 -1.06 -0.93 -0.73 -0.47 -0.14 0.25 0.70 1.20 1.72 2.23 2.71 3.10 3.39 3.55 3.55 3.42 3.17 2.86 2.54 2.27 2.08 1.96 1.88 1.76 1.55 1.26. In simulation the face is an exact triangle mesh through points every 2 mm; every collider is defined in the module.

The scripted demonstration (`demo()`) holds the shelf down with the right hand's fingertips and scribes with the left at 10 mm/s, the lead commanded 1 into the board and the point 4 into the wall (preloads: the board and the wall stop them). The pressed lead trails the hand by about 6 and springs forward only when lifted, too lightly to mark, so the hand runs 7 past the back edge (3 on layout 3, the arm's reach). GPU, seed 0 (layout 2): one line from 1.5 to 139.8 plus a touch at 0.5, 0.50-1.67 N, longest gap 1.19, score 1.0. Wrong way, lead commanded 0.5 above the board: 62 touches under 0.5 N recorded nothing, two at 0.52-0.54 N single points; no line, score 0.

Fidelity (beyond task.json's note): the grip is square because a round rigid grip crept in the rigid simulated pads.

Sources: General Tools 843/1 pencil compass and scriber (0.75 x 3.00 x 6.75 in, 0.05 lb), https://generaltools.com/products/pencil-compass-and-scriber-adjustable-up-to-8-inches; 1x6 board 19 x 140 mm (PS 20); eastern white pine, USDA Wood Handbook (FPL-GTR-190) table 5-3; writing force: Lin et al. 2022, "Pen-grip kinetics in children with and without handwriting difficulties", PLoS ONE 17: e0270466, table 2, https://doi.org/10.1371/journal.pone.0270466.

## Files and licence

All the code here is the task authors' own, with no third-party code or assets; geometry is primitives and a procedural mesh. `scribe_runtime.py`: per-seed layout and graphite dots. `yam_*.py`: kit modules copied unchanged.
