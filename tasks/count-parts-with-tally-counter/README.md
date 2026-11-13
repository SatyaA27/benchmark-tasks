# Count a heap of overlapping hex nuts one at a time on a desk tally counter

Loose fasteners sit in parts bins as heaps, and a stock count is made by lifting them out one at a time and clicking a tally counter once for each, so the count survives an interruption and is read off the counter at the end. Here eleven M16 hex nuts lie in an overlapping heap three layers deep in a parts bin; they go one at a time into an empty tray while a bolted desk tally counter keeps the count. The counter still shows the last count, 0038, and is zeroed first by its reset knob.

Category: Impact on jobs. Counting stock items is a core task of stockers and order fillers (O*NET 53-7065.00: "Receive and count stock items, and record data manually or using computer"). The hard part is physical: taking one nut cleanly off a heap where nuts lie across each other, with the open fingers fitted between its neighbours, without lifting a second one or knocking one out of the bin. The scorer makes that part count: two nuts out of the heap at once, or a count that runs two away from the nuts moved, is a sticky 0. One arm moves the nuts while the other works the counter's sprung key and reset knob; the demonstration uses the left arm for the nuts and the right arm for the counter, but the scorer does not care which arm does what. One fixed layout; no seed changes anything.

## Run and score

With the official simulator installed, run from this folder with zero object jitter and 20 Hz:

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=count_parts_with_tally_counter.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --seed 0 \
  --instruction "Count the hex nuts from the heap in the bin into the empty tray on the tally counter: zero it first, then lift the nuts out one at a time and click once for each." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ./demo-run
```

The model run uses the same command with `--policy dream`. The harness reports task_score at the end. Encode recorded cameras with `inspect-robots video <run.json>`. Scores are in task.json. CPU scorer checks: `python -m pytest -q -p no:cacheprovider` in this folder with yam_arena installed (`test_red_team.py` drives the scorer step by step through the latch, the sticky rules (dropped, two lifted, out of step), the stillness windows, the release rule and the demonstration's pick planner).

Runtime per trial: at most 3,600 control steps (180 s simulated). The scripted demonstration's plan lasts 70.6 s (1,412 steps) and success needs 1 s more, 39.8% of the limit. Wall-clock time depends on the GPU.

## Scoring

Containers. A nut is **in the bin** while its centre is inside the bin's inner wall rectangle (157 x 165 mm) and at most 50 mm above the table, and **over the tray** under the same rule for the tray. The 50 mm cap covers the heap's top nut (centre 35.5 mm up) and nuts stacked or standing on edge in the tray; a nut carried at the demonstration's travel height (centre about 68 mm up) is in neither.

**N** is the number of nuts in the tray: nuts over the tray that rest there, either on the floor (the nut's lowest point within 2 mm below to 3 mm above the floor top) or on a nut already in N (the nut's centre above that nut's outline, and its lowest point between that nut's mid-height and 3 mm above that nut's top), built up layer by layer. A nut still held over the tray with nothing under it is not in the tray, however low it hangs; a held nut that already rests on the floor or on a counted nut is in N, but full credit needs it released.

**M** is the number of nuts not in the bin: in the gripper, in the tray or anywhere else. A nut is **lifted** while it is neither in the bin nor over the tray.

**R**, the latched reading: the counter shows a valid reading only while the units and tens wheels are both within 6 degrees of a figure; R is the last valid reading the wheels showed for 0.25 s (5 control steps), starting at 38.

Sticky faults (once one happens the score is 0 for the rest of the trial):

- **Dropped.** A nut centre outside both trays' outer footprints and less than 20 mm above the table, or more than 20 mm below the table top anywhere. A nut lost on the bench or the floor is missing from the count.
- **Two lifted.** Two nuts lifted at the same moment: more than one taken off the heap at once (or a second lifted while one is still on its way).
- **Out of step.** After R first shows 0000, R and M differ by 2 or more at any moment. One click goes with each nut, made just before the nut is lifted or just after it is put down; a batch of nuts and then their clicks, clicks run ahead, a re-zero in the middle of the count, or a count kept on 0038 and zeroed with two or more nuts already out all break it. The first nut may be out of the heap when 0000 shows.

No credit unless R has been 0000 at some moment of the trial: a count that never shows a steady 0000 (one carried on from 0038, or a reset twist too short to bring both wheels to 0) scores 0.

| Score | End state, with no sticky fault |
|---|---|
| 1.0 | All 11 nuts in the tray (N = 11); R = 11 and the wheels showing 0011 now; released; at rest over the last 1 s |
| 0.8 | N = 11 and R = 11, but not released, not at rest, or the wheels between figures now |
| 0.25 | N = 11 and R = 10 or 12: one missed or one extra click that never left R 2 away from M (for example the last click missed) |
| 0.4 x N / 11, rounded to 3 decimals | 1 <= N <= 10 and R = N: an accurate but unfinished count (0.036 for one nut, 0.364 for ten) |
| 0 | Otherwise: the untouched start, a sticky fault, no steady 0000 at any moment, N = 11 with R off by 2 or more, or a partial count with R other than N |

Success is 1.0 held for 1 s.

- **Reading.** Halfway between figures (18 degrees) the window shows two half figures; within 6 degrees one figure fills it. A person reads a steady display, not a wheel in motion: during the 9 to 10 carry the units wheel can reach 0 a moment before the tens wheel reaches 1, which shows 00 for a few steps. The 0.25 s rule ignores that and other transients, while a finished click holds its figure for much longer.
- **Released.** Every gripper TCP at least 30 mm from each nut's bounding box and from the bounding boxes of the key and the reset knob. A finger resting on the key or holding the knob is not a finished count.
- **At rest, over the last 1 s.** Each nut within 1 mm and 2 degrees of its current pose and slower than 1 cm/s; both wheels within 2 degrees of a figure and each varying by at most 0.3 degrees; the key within 0.5 mm of its top stop and varying by at most 0.2 mm; the knob within 2 degrees of its rest stop. These are pose and angle windows: reported joint speeds of parts in sliding contact chatter even when nothing moves.
- **Miscounts.** A slip that stays within one of M ends on 10 or 12 and scores 0.25 (which slips stay within one depends on the click order: clicking after each placement, an extra click stays within one and a missed click is caught when the next nut is lifted; clicking before each lift, the reverse). A nut pushed across the bin floor while another is picked stays in the bin and does nothing to the count. A nut on the tray's wall top is not in the tray (its centre is over the wall, outside the inner wall rectangle). How nuts lie in the tray does not matter: flat, standing on edge, touching a wall or stacked, they count as long as they rest in the tray as defined for N.

A ruler, the counter's window and a video of the trial (for the moments two nuts are up together, and for R against M) are enough to score a physical trial.

## The counter

A desk tally counter of the common chrome type: a round case on a base plate, a push key on top and a reset knob on the side. Only the units and tens wheels turn.

- **Wheels.** Four drums, 20 mm across and 4 mm wide, 7 mm apart on one axis 31 mm above the table, behind a 28 x 7 mm window facing the robot. Each carries the figures 0 to 9 every 36 degrees, white on black, 4 x 2.3 mm. A spring-loaded roller (jumper, 60 N/m, 0.47 N preload) presses on each moving drum and holds it on a figure.
- **Count stroke.** The key cap (10 mm across, top 66.5 mm above the table) travels 7 mm against a 165 N/m spring with 1.2 N preload (2.35 N at the bottom). A hinged pawl hangs from the key, sprung toward the drums (0.001 N m per radian); on the way down its units prong enters one of ten notches on the units drum and turns it. The count registers at 4.25 mm of key travel: the drum has passed the crest halfway between figures and the jumper snaps it on to the next figure; the rest of the stroke is overtravel. The key force peaks at about 2.5 N. Measured on the simulator, plain presses count with the pads moving down at up to 150 mm/s; the carry is the press most likely to slip (see Key speed under Fidelity).
- **Partial strokes.** A stroke released before about 4.2 mm leaves the drum short of the crest, and the jumper pulls it back to the figure it left: no count. In between, a wheel more than 6 degrees from a figure gives no reading.
- **Carry.** The units drum's notch at 9 is 3.65 mm deep instead of 1.5 mm. Only on the stroke that starts at 9 can the pawl swing far enough for its stepped tens prong to reach the tens drum, so that stroke turns both drums (9 to 10); it registers a little later, at 5.10 mm. The tens prong alone pushes the tens drum 27 degrees, 9 past the crest, before the jumper takes over.
- **Reset.** The knob (13 mm across, standing 20 mm out of the case's right side) turns on the wheels' axis. An eccentric pin on it lifts a follower and a reset bar, 6.5 mm by 55 degrees of knob turn. The bar's two flat pads press on a heart-shaped cam on each moving drum; a heart cam's lowest point is its 0, so the pressed cam turns its drum the short way to 0 from any figure. From 55 degrees to the 70 degree stop a preloaded follower spring (9.3 N, 800 N/m) takes up the extra travel and the bar stays fully up. Turning the knob needs up to 0.088 N m. Let go, the bar spring (35 N/m) and the knob's return spring (0.01 N m per radian, damped at 0.01 N m s per radian so a knob released at its stop swings back in about a second) return the knob to its rest stop at -1 degree, and the bar drops clear of the cams while the jumpers hold 0.
- **Numbers.** The start shows 0038. After the reset it shows 0000 and after eleven counts 0011; eleven counts without the reset end at 0049.

Everything is passive contact and springs; the scorer only reads the units and tens wheel angles, the key and knob positions, the nut poses and the gripper positions.

## Physical setup

Positions are on the table top, measured forward from the line through the two arm bases and left (+) or right (-) of the midline between them; the arm bases are 610 mm apart. Nothing else is on the table.

1. **Tally counter:** matches the Redington Model 18 desk tally (model 13-1804) in size and controls: push key on top, knurled reset knob on the right side, 4 mm figures. Chrome-plated steel case, key and knob on a black ABS base plate (57 mm across, 5 mm thick), bolted to the table with the base centre 240 mm forward and 120 mm right; 66.5 mm tall overall. Round case 46 mm across and 29.6 mm deep, window facing the robot, showing 0038. Internal parts are acetal (POM) with steel springs.
2. **Bin:** blue polypropylene parts tray, 163 mm front to back, 171 mm wide and 33 mm tall, 3 mm walls and floor, fixed with its centre 262 mm forward and 225 mm left.
3. **Tray:** white polypropylene tray of the same build with 25 mm walls, fixed and empty, centre 98 mm forward and 115 mm left.
4. **Nuts:** eleven DIN 934 M16 hex nuts, zinc-plated steel, 24 mm across flats, 27.7 mm across corners, 13 mm thick, 16 mm bore, 33.3 g each, in one overlapping heap in the middle of the bin, as tipped out of a supplier's bag and set from the table below. Every nut lies flat. Six lie on the bin floor in two rows of three (about 38 mm apart along a row, the rows about 52 mm apart); four lie across pairs of neighbouring floor nuts, two on each row; one lies on top, across the two on the row nearer the midline, its top 39 mm above the bin floor. Seen from above, nut_02 is almost entirely hidden (88% of its outline) and seven others are partly covered (13 to 30%). Turn is the angle from a nut with two corners pointing forward, counter-clockwise seen from above (turns 60 degrees apart look the same). In the simulation each nut starts 0.3 mm above what it rests on and settles in the first moments.

| Nut | Forward (mm) | Left (mm) | Centre height above the table (mm) | Layer | Turn (degrees) | Lies on |
|---|---|---|---|---|---|---|
| nut_01 | 218 | 197 | 9.5 | 0 | +4 | bin floor |
| nut_02 | 256 | 195 | 9.5 | 0 | +52 | bin floor |
| nut_03 | 294 | 199 | 9.5 | 0 | +10 | bin floor |
| nut_04 | 220 | 249 | 9.5 | 0 | -66 | bin floor |
| nut_05 | 258 | 250 | 9.5 | 0 | +7 | bin floor |
| nut_06 | 295 | 247 | 9.5 | 0 | +57 | bin floor |
| nut_07 | 237 | 196 | 22.5 | 1 | +57 | nut_01, nut_02 |
| nut_08 | 275 | 198 | 22.5 | 1 | -52 | nut_02, nut_03 |
| nut_09 | 239 | 249 | 22.5 | 1 | +6 | nut_04, nut_05 |
| nut_10 | 276 | 248 | 22.5 | 1 | -9 | nut_05, nut_06 |
| nut_11 | 256 | 198 | 35.5 | 2 | +52 | nut_07, nut_08 |

To set the heap by hand: print the six floor-layer outlines at full size on a sheet cut to the bin's inner floor (157 x 165 mm), lay it in the bin and set the floor nuts on their outlines; then lay the four layer-1 nuts and the top nut by the table, measuring from the bin's inner walls (the near wall is 183.5 mm forward of the table origin, the left wall 307.5 mm left of it).

Nuts, bin and tray have 1.2 static and 1.0 dynamic friction. The counter's moving parts have 0.1 / 0.08 (lubricated acetal) and the knob 0.7 / 0.6 (knurled).

## Counter geometry, joints and robot clearances

Generated from the counter model itself. Part frame in millimetres: origin on the table under the key, x away from the robot (the window faces -x), y to the robot's left, z up; joints at their start values. The counter's origin is at x 240, y -120 mm on the table, yaw 0.

### Colliders

| Body | Collider | Shape | x range | y range | z range | Detail |
|---|---|---|---|---|---|---|
| base | shape_0 | cylinder | -28.50..28.50 | -28.50..28.50 | 0.00..5.00 | cylinder axis z, radius 28.5, length 5 |
| base | shape_1 | cylinder | -6.00..6.00 | -6.00..6.00 | 5.00..8.00 | cylinder axis z, radius 6, length 3 |
| base | shape_2 | box | -14.80..-14.00 | -13.00..13.00 | 14.00..27.00 | box 0.8 x 26 x 13 |
| base | shape_3 | box | -14.80..-14.00 | -13.00..13.00 | 37.00..49.00 | box 0.8 x 26 x 12 |
| base | shape_4 | box | 14.00..14.80 | -13.00..13.00 | 14.00..27.00 | box 0.8 x 26 x 13 |
| base | shape_5 | box | 14.00..14.80 | -4.00..4.00 | 50.00..53.00 | box 0.8 x 8 x 3 |
| base | shape_6 | box | -10.00..10.00 | -21.50..-19.50 | 23.00..25.00 | box 20 x 2 x 2 |
| base | shape_7 | box | -10.00..10.00 | -21.50..-19.50 | 37.00..39.00 | box 20 x 2 x 2 |
| base | shape_8 | cylinder | -10.00..10.00 | 1.50..5.50 | 21.00..41.00 | cylinder axis y, radius 10, length 4 |
| base | shape_9 | cylinder | -10.00..10.00 | 8.50..12.50 | 21.00..41.00 | cylinder axis y, radius 10, length 4 |
| base | shape_10 | box | 8.00..10.60 | -7.80..-6.20 | 50.42..51.22 | box 2.6 x 1.6 x 0.8 |
| base | shape_11 | box | 14.80..15.60 | -13.00..13.00 | 27.00..50.00 | box 0.8 x 26 x 23 |
| key | shape_0 | cylinder | -2.00..2.00 | -2.00..2.00 | 50.00..63.50 | cylinder axis z, radius 2, length 13.5 |
| key | shape_1 | box | -0.50..12.50 | -1.00..1.00 | 48.00..50.00 | box 13 x 2 x 2 |
| pawl | shape_0 | cylinder | 10.39..11.07 | -11.00..-10.00 | 33.99..34.67 | cylinder axis y, radius 0.3, length 1 |
| pawl | shape_1 | cylinder | 12.81..13.50 | -4.00..-3.00 | 34.67..35.36 | cylinder axis y, radius 0.3, length 1 |
| pawl | shape_2 | cylinder | 9.10..9.79 | -7.60..-6.40 | 49.78..50.47 | cylinder axis y, radius 0.3, length 1.2 |
| pawl | shape_3 | box | 11.93..13.07 | -11.00..-3.00 | 49.43..50.57 | box 1 x 8 x 1 |
| units_jumper | shape_0 | box | -0.50..0.50 | -11.25..-9.75 | 15.08..17.08 | box 1 x 1.5 x 2 |
| units_roller | shape_0 | cylinder | -2.00..2.00 | -11.25..-9.75 | 17.68..21.68 | cylinder axis y, radius 2, length 1.5 |
| tens_jumper | shape_0 | box | -0.50..0.50 | -4.25..-2.75 | 15.08..17.08 | box 1 x 1.5 x 2 |
| tens_roller | shape_0 | cylinder | -2.00..2.00 | -4.25..-2.75 | 17.68..21.68 | cylinder axis y, radius 2, length 1.5 |
| knob | shape_0 | cylinder | -6.50..6.50 | -43.00..-23.00 | 24.50..37.50 | cylinder axis y, radius 6.5, length 20 |
| knob | shape_1 | cylinder | -1.00..1.00 | -23.00..-18.00 | 30.00..32.00 | cylinder axis y, radius 1, length 5 |
| knob | shape_2 | cylinder | -7.24..-5.24 | -18.75..-17.75 | 26.75..28.75 | cylinder axis y, radius 1, length 1 |
| bar | shape_0 | box | -12.50..-11.50 | -17.00..-6.40 | 20.00..26.00 | box 1 x 10.6 x 6 |
| bar | shape_1 | box | -12.50..-11.50 | -17.00..-16.00 | 20.00..31.50 | box 1 x 1 x 11.5 |
| bar | shape_2 | box | -12.00..9.00 | -15.50..-14.00 | 20.35..21.00 | box 21 x 1.5 x 0.65 |
| bar | shape_3 | box | -12.00..9.00 | -7.60..-6.40 | 20.35..21.00 | box 21 x 1.2 x 0.65 |
| bar | shape_4 | box | -12.50..-9.00 | -17.00..-16.50 | 30.50..31.50 | box 3.5 x 0.5 x 1 |
| follower | shape_0 | box | -8.40..-3.00 | -19.00..-16.50 | 28.75..30.75 | box 5.4 x 2.5 x 2 |
| base | collar | SDF mesh | -4.00..4.00 | -4.00..4.00 | 54.00..56.00 | 768 triangles |
| key | cap | SDF mesh | -5.00..5.00 | -5.00..5.00 | 61.50..66.50 | 768 triangles |
| units | notched_drum | SDF mesh | -10.00..10.00 | -12.50..-8.50 | 21.32..40.68 | 808 triangles; R10 x W4; 10 mouths 2.017; units depth 1.5, deep 3.65 undercut; tens depth 2.0 hooked 1deg; lands>=4 mm |
| units | heart | SDF mesh | -9.00..4.01 | -16.00..-13.50 | 24.95..37.58 | 1988 triangles; support 3.5..9; crown at digit 5.5 |
| tens | notched_drum | SDF mesh | -10.00..10.00 | -5.50..-1.50 | 21.32..40.68 | 756 triangles; R10 x W4; 10 mouths 2.017; depth 2.0, drive face hooked 1deg; lands>=4 mm |
| tens | heart | SDF mesh | -4.01..9.00 | -8.00..-6.00 | 24.42..37.05 | 1988 triangles; support 3.5..9; crown at digit 5.5 |
| pawl | units_stem | SDF mesh | 10.68..14.41 | -11.00..-10.00 | 34.03..50.57 | 24 triangles; stepped POM prong; 1 mm thick |
| pawl | tens_stem | SDF mesh | 11.93..14.41 | -4.00..-3.00 | 34.72..50.57 | 24 triangles; stepped POM prong; 1 mm thick |
| pawl | pawl_tail | CONVEX mesh | 9.44..12.69 | -7.60..-6.40 | 48.81..50.12 | 12 triangles; lift-stop tail arm, 20 deg; convex |

Visual only (no collider): the chrome case shell, radius 23 mm about the window axis and 29.6 mm deep; the 28 x 7 mm window glass; three screw heads; the printed figures; the knob's knurl ribs. Cylinder bounds are exact; mesh bounds are the mesh vertices.

### Joints

| Link | Parent | Joint | Anchor (mm) | Range | Start | Spring (preload at start) | Damping | Mass (g) |
|---|---|---|---|---|---|---|---|---|
| units | base | hinge y | 0, -10.5, 31 | -180..360 deg | -144 deg | none | 3e-05 N m s/rad | 2.6 |
| tens | base | hinge y | 0, -3.5, 31 | -180..360 deg | 0 deg | none | 3e-05 N m s/rad | 2.6 |
| key | base | slide z | 0, 0, 50 | -7..0 mm | 0 mm | 165 N/m (1.2 N) | 0.4 N s/m | 7 |
| pawl | key | hinge y | 12.5, -7, 50 | -10..14 deg | -9 deg | 0.001 N m/rad (0.00068 N m) | 2e-05 N m s/rad | 1.2 |
| units_jumper | base | slide z | 0, -10.5, 19.68 | -1.2..0 mm | 0 mm | 60 N/m (0.47 N) | 0.3 N s/m | 0.7 |
| units_roller | units_jumper | hinge y | 0, -10.5, 19.68 | -5000..5000 deg | 0 deg | none | 1e-07 N m s/rad | 0.3 |
| tens_jumper | base | slide z | 0, -3.5, 19.68 | -1.2..0 mm | 0 mm | 60 N/m (0.47 N) | 0.3 N s/m | 0.7 |
| tens_roller | tens_jumper | hinge y | 0, -3.5, 19.68 | -5000..5000 deg | 0 deg | none | 1e-07 N m s/rad | 0.3 |
| knob | base | hinge y | 0, 0, 31 | -1..70 deg | 0 deg | 0.01 N m/rad (-0.00087 N m) | 0.01 N m s/rad | 11 |
| bar | base | slide z | -12, -18.25, 21 | 0..6.5 mm | 0 mm | 35 N/m (-0.12 N) | 0.4 N s/m | 2 |
| follower | bar | slide z | -8, -18.25, 28.75 | 0..1.7 mm | 0 mm | 800 N/m (-9.3 N) | 1.2 N s/m | 1 |

In brackets: the spring's force or torque at the start position, positive along the joint axis (up for the slides; for the hinges, right-handed about +y, which turns the top of a wheel or the knob away from the robot).

### Robot clearances

Key press: the closed pad block at yaw 90 degrees covers x -10.0..10.0 mm and y -20.0..20.0 mm. Counter colliders under it (key excluded): base/collar (top 56.00 mm), base/shape_10 (top 51.22 mm), pawl/shape_2 (top 50.42 mm), pawl/pawl_tail (top 50.12 mm), base/shape_9 (top 41.00 mm), base/shape_8 (top 41.00 mm), units/notched_drum (top 40.68 mm), tens/notched_drum (top 40.68 mm), base/shape_7 (top 39.00 mm), units/heart (top 37.58 mm), tens/heart (top 37.05 mm), knob/shape_1 (top 32.00 mm), bar/shape_4 (top 31.50 mm), follower/shape_0 (top 30.75 mm), knob/shape_2 (top 28.75 mm), base/shape_6 (top 25.00 mm), units_roller/shape_0 (top 21.68 mm), tens_roller/shape_0 (top 21.68 mm), bar/shape_3 (top 21.00 mm), bar/shape_2 (top 21.00 mm), units_jumper/shape_0 (top 17.08 mm), tens_jumper/shape_0 (top 17.08 mm), base/shape_1 (top 8.00 mm), base/shape_0 (top 5.00 mm). The commanded pad-tip height is 56.50 mm, 0.5 mm above the highest of them and 3.00 mm below the cap's bottom-of-stroke height (59.50 mm). Closest counter collider at that point: 0.50 mm (pad_b / base/collar); the visual case shell is 2.50 mm below the pads.

Knob twist (pitch -37.5 to +37.5 degrees, gripper modelled as two pads, two upper fingers and the palm, wheels, jumpers, reset bar and follower swept over their travel): closest counter collider other than the knob 4.34 mm, base plate 5.96 mm, table 10.64 mm. Over the whole knob phase (approach, twist, release and back-off, with the pads 20 mm apart while open): 4.34 mm, 4.30 mm and 8.51 mm. Visual case shell: 2.11 mm.

## Demonstration and runtime

**What the demonstration knows.** Every time is fixed before the trial. The right arm's targets (the knob and the key) are fixed too, from the counter's known position. The left arm plans each pick from the simulator's nut poses, as a person looks before reaching: when a pick's first move starts it takes the nut poses the scorer saw at the previous control step and chooses, among the nuts in the bin with no other nut above and over them, the highest, then the one with the most room round the open fingertips (more than 6 mm counts as enough, and then the smaller wrist turn wins), with a grip across a pair of its flats turned at most 60 degrees. That pick's targets are then fixed. It reads no wheel angle, key or knob position, contact or score, and it never checks that the reset reached 0 or that a click registered. It is an existence proof of the motion; a policy that watches the window would be more robust.

**Right arm: zero, then click.** The knob pinch has the pads 20 mm apart, closing front to back, with the TCP on the knob axis 44 mm right of the counter's centre so the pads cover the outer 11 mm of the knob. The arm approaches tilted 37.5 degrees (fingertips toward the counter's back) in 2 s, settles 0.4 s, slides 45 mm along its own axis onto the knob in 1.2 s, waits 0.3 s, closes in 0.5 s, waits 0.2 s and then rolls the wrist to 37.5 degrees the other way over 3 s: 75 degrees in all, the knob's 70 degree stop plus 5 degrees that slip in the pinch. It holds 0.3 s, turns the knob back by hand to about 5 degrees in 0.8 s, opens in 0.5 s, slides 40 mm off the knob's end in 0.8 s and rises to hover 15 mm above the key cap with the fingers closed. For each nut, the closed pads press down 0.8 s (at either pace) to the pad-tip target given under Robot clearances (0.5 mm above the highest counter collider under the pads, below the cap's bottom-of-stroke height), hold 0.3 s and rise 0.5 s. Each press bottoms 0.6 s after the left arm lets go of that nut (2.3 s for the first, while the right arm comes back from the knob). After the eleventh click it parks 120 mm above the table, 80 mm in front of and 120 mm to the right of the counter.

**Left arm: transfer.** The k-th nut lifted goes to tray slot k (twelve slots on a grid of four columns 36 mm apart and three rows 48 mm apart round the tray's centre, filled row by row from the robot's left; the twelfth stays empty). From the start layout the order is the top nut, the four nuts across the floor layer, then the six floor nuts (nut_11, nut_07, nut_09, nut_10, nut_08, nut_01, nut_04, nut_05, nut_02, nut_03, nut_06). Over the chosen nut, with the pads open to 40 mm and turned to the planned grip, the arm descends so the pad tips are 2 mm above the nut's lowest point for a floor nut and 3 mm for a nut lying on others (10 to 11 mm of pad on the 13 mm nut, and the pads clear of the nuts underneath), closes across the flats, lifts straight up until the TCP is 80 mm above the table (the nut's bottom about 60 mm up, over the heap and the bin's wall tops), carries it to its tray slot while turning the grip back to 0 degrees, lowers until the nut's bottom is 1.5 mm above the tray floor (allowing 3.5 mm for a gripped part riding low), opens, and rises. The first move goes over the middle of the heap in 1.6 s and then over the chosen nut in 0.6 s; nut 1 then waits above the tray until 0.5 s after the pads finish opening on the knob. A nut cycle takes 5.5 s. Finally the arm parks 120 mm above the bin. Both arms then hold still for 2.5 s.

**Timing.** The counter's 0000 is a steady reading by 9.45 s at the latest, while nut 1 waits in the gripper above the tray (M = 1 against R = 0). Each click bottoms 0.6 s after its nut is let go in the tray, and the next nut leaves the bin (its centre above the 50 mm cap) at least 0.33 s after that click's reading is latched, so R and M never differ by more than one, taking the earliest moment each nut could leave and each click could register against the latest; each nut is let go at least 3.2 s before the next one leaves the bin. A variant with every motion scaled to 0.8 of its duration ends at 60.0 s; the dwells whose length is physics keep it there: the 0.4 s release opening, the whole knob pinch (approach, descent onto the knob with 3.5 mm of pad clearance, grip, twist, hold, hand return and opening), the 0.8 s presses and 0.3 s press holds, the 0.5 s knob spring-back wait, the 0.6 s release-to-press delay and the 2.5 s tail.

**Reach and clearances.** Every TCP target stays 0.149 to 0.323 m (left) and 0.173 to 0.323 m (right) from its arm base in plan, with inverse-kinematics errors below 0.1 mm and 0.02 degrees; the two TCPs stay at least 0.224 m apart. Modelling the gripper as two pads (10 mm thick, 28 to 40 mm wide, 33 mm long), two upper fingers and the palm, along the planned picks from the start layout: the open pads and fingers clear the nuts beside the one picked (same layer or higher) by 6.0 mm and pass 3.3 mm above the nuts it lies on; they clear the walls by 4.5 mm; a carried nut clears the walls by 27.5 mm and the other nuts by 8.3 mm; a nut being lifted or lowered stays 10.6 mm from the walls. The counter clearances are listed above.

**Wrist camera.** The right wrist camera cannot show the final count legibly. The 4 mm figures need the camera within about 98 mm of them for 8 pixels, and the TCP is 86 mm ahead of the camera along its line of sight, so the fingers end up at the counter. A sweep of right-arm poses found none that shows all four figures at 8 pixels or more through the window, unobstructed by the fingers, with the gripper clear of the counter and the TCP 30 mm from the key and knob, and that the arm can reach. So the right arm parks clear.

## Fidelity and approximations

- **Internal mechanism:** our own design after the principle of US 2,572,784 (pawl drive, roller jumpers, stepped carry, heart-cam reset); manufacturers do not publish the internals.
- **Hundreds and thousands drums** are fixed at 0: the count never passes 99.
- **Figure spacing 7 mm** against about 5 mm on real counters: each moving drum is 4 mm wide with room beside it for the pawl prongs and jumpers. The drums are 20 mm across and their faces sit 4.4 mm behind the window, sized for the internal parts.
- **Key stroke and forces** (7 mm, 165 N/m with 1.2 N preload, jumper and pawl springs, knob torque) are engineering estimates; no manufacturer force data is published.
- **Key speed:** a real counter counts at any hand speed. Here a quick press can slip: the prong is thrown off the drive face, the drum turns part way and the jumper takes it back. Measured on the simulator: plain presses counted at up to 150 mm/s and one of two at 170 mm/s; the 9-to-10 carry, which drives two drums, slipped once in 8 trials pressed at 75 to 95 mm/s. The demonstration presses slowly, meeting the cap at about 45 mm/s (40 mm/s at the 4.25 mm registration point), at either pace.
- **Knob** stands 20 mm out of the case against about 8 mm: the gripper's pad contact face is about 24 mm wide where it meets the knob and the finger widens to 40 mm behind it, so a two-pad pinch from above covers only the outer part of a knob while staying clear of the case; 20 mm gives 11 mm of pad on the knob with the fingers 5 mm or more from the case.
- **Reset action:** one twist of the knob by 55 to 70 degrees zeroes the count through heart cams (the reset used on stroke counters). The Redington's knob is instead turned until every wheel reads 0, up to one full turn, picking the wheels up one by one. A full-turn reset would need about five regrasps of a 13 mm knob by a two-pad gripper; the twist keeps the same knob, place and purpose. Let go, our knob springs back to its rest stop (0.01 N m per radian, damped) so the reset bar drops clear of the cams; a knob that stayed turned would hold the drums at 0 and block counting. Neither the stroke nor the reset turn is published by the maker; the key's 7 mm travel, 4.25 mm count point and 2.5 N are this design's values, and the tables under Counter geometry give every part, spring and size needed to build it.
- **Case shell** is visual only. The colliders are the front face around the window, a closed back face from 14 to 53 mm above the table (base shape_4 from 14 to 27 mm, shape_11 from 27 to 50 mm and the 8 mm wide strip shape_5 from 50 to 53 mm), two strips on the knob side, the base plate, the key's collar and cap and the internal parts. The back panel shape_11 lies just behind the case back face (x 14.8 to 15.6 mm rather than 14.0 to 14.8 mm) because the pawl's back corner reaches x 14.41 mm at rest and 14.65 mm at its -10 degree outward stop; it closes the drums from behind, so they turn only through the key and the reset knob. During the twist an upper finger reaches the visual shell's outline (it may appear to graze the chrome on video) while keeping the collider clearances listed under Robot clearances.
- **Bolted down:** the real counter has screw holes on a 49.9 mm circle; a loose 99 g counter would slide under a press. The base is fixed, so its mass plays no part.
- **Nut colliders** are solid hexagons; the 16 mm bore is visual. Nothing ever enters the bore.
- **Friction** of nuts, bin and tray is 1.2 static and 1.0 dynamic, higher than zinc-plated steel on polypropylene, so a two-pad pinch holds the 33 g nut.
- **Bin walls** are 33 mm tall so the open upper fingers pass over them at the outer slots; the bin and tray are fixed to the table so that sliding trays do not decide a counting task.
- **Heap:** a real heap tipped from a bag is random and has nuts tilted against each other or standing on edge; this one is a fixed arrangement of flat-lying nuts in three layers, set from the pose table so every trial starts the same. The simulator's part kit starts every loose part upright with only a turn about the vertical, so no nut starts tilted. Each nut starts 0.3 mm above what it rests on and settles under gravity in the first moments of the trial.
- **Wheel travel:** the units and tens hinges allow -180 to +360 degrees, enough for 14 counts from 0038 or 12 from 0000 and for every reset; any count beyond that already scores 0.
- **Reading model:** the reading comes from the wheel angles (within 6 degrees of a figure, steady for 0.25 s), not from the rendered window.

## Dimension sources

- Redington Model 18 desk tally data sheet and the 13-1804 drawing (57 mm base, 49.9 mm screw circle, 46 x 29.6 mm case, 66.5 mm height, 99 g): https://www.relayspec.com/specs/180/model_18.pdf
- Line Seiki H-102 series hand tally counters (4.0 x 2.3 mm figures): https://www.lineseiki.com/product/2358/
- Traceable (Control Company) 3125 tally counter, same 4-figure chrome type (side reset knob returns the count to 0): https://www.globaltestsupply.com/product/traceable-3125-hand-tally-counter
- US 2,572,784, A. Van Veen, 1951 (counter with stepped carry): https://patents.google.com/patent/US2572784A/en
- DIN 934 M16 hex nut (s 24 mm, e 26.75 mm minimum, m 13 mm, 33.3 kg per 1000): https://fullerfasteners.com/tech/weights-for-din-934/

## Licence

All the code in this folder is the task authors' own and contains no third-party code or assets; all geometry is built from primitives and procedural meshes. `count_parts_with_tally_counter.py`, `tally_counter.py`, `tally_visuals.py` and `test_red_team.py` were written for this task. `yam_parts.py`, `yam_joints.py` and `yam_mesh.py` are copied kit modules (compound parts, jointed mechanisms and mesh colliders), copied unchanged; they are the task authors' own code too.
