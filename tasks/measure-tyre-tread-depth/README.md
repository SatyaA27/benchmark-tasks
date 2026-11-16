# Measure a tyre's tread depth with a pen gauge: probe on a groove's floor, foot flat across the ribs, then lift it off

Measure the tread depth of the tyre with the tread-depth gauge, then put the gauge back on its rest.

Automotive service technicians and mechanics (about 826,000 US workers) and tyre changers measure tread depth on every multipoint inspection, tyre rotation and seasonal tyre swap, with a pen gauge read in 32nds of an inch; here on a wheel with a new tyre, lying flat on the bench. The reading is right only if two contacts are made at once: the probe's tip on the floor of a main groove, not on a rib or on one of the molded wear bars that cross the groove (on a bar it reads 2/32 shallow), while the gauge's T-foot lies flat across the ribs on both sides of the groove, square to the tread (tipped 6 degrees, it reads about 2/32 deep). The gauge's stem is held only by friction, so the gauge must then come straight off without the probe catching the groove's wall. The robot doesn't read the scale: the stem's position is the result, checked by whoever writes up the inspection.

Sources: Milton Industries, "Measuring Tire Tread Depth with a Milton Tread Depth Gauge - P/N 448" (form 9X059):
"Place the flat part of the gauge on the tread of the tire making sure not to place the flat part of the gauge on any
of the raised surfaces of the tire tread design. Push the scale probe down into the bottom of the tire tread groove
being careful not to place the probe on any of the molded horizontal bars at the base of the grooves (Original Tire
Manufacturer's wear indicators). Read the scale from the top point of the cylinder". Distributor listings of the
S-448: length 3-1/2 in, aluminium, "Round aluminum indicator bar calibrated from 0 to 1" in 1/32 Inch Markings",
"large base, flat blade and pocket clip". Commercial Vehicle Safety Alliance, Inspection Bulletin 2019-03: "Depress the
plunger or probe fully. Place the probe into the center of a circumferential tire groove and push down on the gauge's
base. Do not measure at tread wear indicators or other raised surfaces that are part of the tread design. Carefully
remove the gauge by holding its barrel or outer case without touching the probe. Verify the tread depth reading".
FMVSS 139 (49 CFR 571.139): passenger tyres carry at least six tread wear indicators showing 2/32 in. Michelin
Defender2 205/55R16 91H (Michelin 07259) specification tables: overall diameter 24.8 in, section width 8.4 in on a
6.5 in rim, tread width 6.7 in, new tread depth 10.5/32 in, 21-22 lb; sold new at $179.99 (Tire Rack,
https://www.tirerack.com/tires/tires.jsp?tireMake=Michelin&tireModel=Defender2&partnum=055HR6DEF2) and $196.59 (Tire
Agent, https://www.tireagent.com/products/michelin-defender2-07259). NEIKO 01407A 6 in digital caliper (0.01 mm, with
depth rod), $27.99 (AutoZone, https://www.autozone.com/p/neiko-multi-purpose-measuring-caliper-01407a/1604222). O*NET
49-3023.00, Automotive Service Technicians and Mechanics (825,800): "Follow checklists to ensure all important parts are
examined", "Disassemble units and inspect parts for wear, using micrometers, calipers, and gauges"; 49-3093.00, Tire
Repairers and Changers (112,200).

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=measure_tyre_tread_depth.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Measure the tread depth of the tyre with the tread-depth gauge, then put the gauge back on its rest." \
  --scorer task_score --max-steps 1800 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial, replace `--policy yam_scripted` with `--policy dream`. `--scorer task_score` reports the score at
the end of the run. Each trial allows 1,800 control steps (90 seconds at 20 Hz). There are no
seeds: the layout is fixed (in a check of the scripted demonstration with the gauge's start shifted by up to 10 mm, 3 of 3 runs scored 1.0, reading 8.29, 8.29 and 8.74 mm on the rest). The scripted demonstration takes 31.0 s, the last
3 s holding still. `success()` requires a score of 1.0.

`pytest` in this folder runs the generated offline checks (no GPU): the scorer on the start, the goal and the named
partial and shortcut layouts.

## The scene

A bench in front of two arms (bases 61 cm apart), one overhead camera and a camera on each wrist. Positions in metres from the midpoint between the arm bases: x forward, y to the left (+) or right (-); heights above the bench. (1) A wheel lying flat (fixed): a new, unused Michelin Defender2 205/55R16 91H tyre (630 mm across, 213 mm tall lying flat, black rubber) mounted on a 16 x 6.5 in steel wheel, about 19 kg, its axis at (0.555, -0.305), straight in front of the right arm; the tread nearest the robot at x 0.240. The tread band runs 21.5 to 191.5 mm up (the tyre's 6.7 in tread width): five ribs and four circumferential grooves, 8, 10, 10 and 8 mm wide, their centres 57.5, 90.5, 122.5 and 155.5 mm up, each 8.33 mm (10.5/32 in, the tyre's published new tread depth) deep. Wear bars 1.59 mm (2/32 in) tall and 8 mm long cross every groove at six places, 60 degrees apart, one of them on the line from the wheel's axis toward the robot. Groove depths, before the first trial: with a digital caliper's depth rod (the end of its beam laid across both ribs), measure each groove at three places within 30 mm of the wear bar facing the robot (not on the bar) and record each groove's mean; use the tyre only if every groove is within 1/32 in of 10.5/32 in (7.5 to 9.1 mm). (2) Milton S-448 tread-depth gauge (aluminium and steel, 27 g, 89 mm long): a barrel 10 mm across and 49 mm long, a T-foot 32 x 8 x 1.5 mm at its front end, and a stem sliding through both: a probe with a 2.8 mm flat tip out of the foot's face and a 6 mm indicator bar out of the barrel's back, graduated 0 to 1 in in 1/32 in. The stem stays where it is put (about 2 N of friction). It starts pushed fully out (probe 25.4 mm past the foot). (3) Its rest (fixed, wood): a board 120 x 60 x 20 mm with a block under the barrel's back end; the gauge lies on it level, axis 36 mm up, pointing forward (+x), foot upright, the foot's face at (0.165, -0.215).

### The wheel

One fixed body in its own frame: origin on the bench at the wheel's axis, at (555, -305) mm;
the angle round the wheel is measured from +x, so the tread nearest the robot is at 180 degrees. The tyre's outside is
315 mm from the axis. Its tread band runs 21.5 to 191.5 mm up; ribs at 21.5 to 53.5, 61.5 to 85.5, 95.5 to 117.5, 127.5 to 151.5, 159.5 to 191.5 mm up;
grooves at 53.5 to 61.5, 85.5 to 95.5, 117.5 to 127.5, 151.5 to 159.5 mm up, their floors 8.33 mm (10.5/32 in) below the ribs. Wear bars stand 1.59 mm
(2/32 in) on the groove floors, 8 mm long, at 180, 240, 300, 0, 60, 120 degrees.
The tread is collided rib by rib as flat boxes, each 1.5 degrees of arc over 156 to
204 degrees (the arc facing the robot) and 6 degrees elsewhere (666
boxes, turned about the wheel's axis), with sidewall rings above and below; the steel wheel inside is visual only.
Friction 0.9 (static), 0.8 (sliding).

### The gauge

One articulated body: the barrel and foot (22 g) and the stem (5 g: probe,
blade tip and indicator bar) on a slide along the barrel's axis, 0 to 25.4 mm, with 2.0 N of
static and 1.6 N of sliding friction, plus 20 N s/m of damping and 0.02 kg
of armature (solver stabilisers for the light stem: 0.2 N at 10 mm/s). The slide's position is the reading: how far the probe's tip stands
past the foot's face. The gauge's own frame: origin at the middle of the foot's face, x out of the foot along the
probe, z along the foot's length. On its rest the frame's origin is at (165, -215,
36) mm above the bench, x pointing forward (+x), z up.

| Part | Collider | Centre (gauge frame, mm) | Size (mm) |
| --- | --- | --- | --- |
| gauge body | box | -0.75, 0, 0 | 1.5 x 8 x 32 |
| gauge body | cylinder | -26, 0, 0 | radius 5, length 49 along x |
| stem (at reading 0) | box | -1.5, 0, 0 | 3 x 1 x 2.8 |
| stem (at reading 0) | box | -5.5, 0, 0 | 5 x 1.1 x 1.1 |
| stem (at reading 0) | cylinder | -79.9, 0, 0 | radius 3, length 10 along x |
| stem (at reading 0) | cylinder | -86.4, 0, 0 | radius 3.2, length 3 along x |

### The rest

Fixed, wood, frame at (165, -215) mm on the bench (the foot's face's place).

| Part | Collider | Centre (rest frame, mm) | Size (mm) |
| --- | --- | --- | --- |
| rest | box | -35, 0, 10 | 120 x 60 x 20 |
| rest | box | -57.5, 0, 25.5 | 35 x 50 x 11 |

## Scoring

The reading is how far the probe's tip stands past the foot's face, read on the gauge's scale where the indicator bar leaves the barrel. Seated means the foot flat on the tread (both its ends within 1 mm of the tread surface) with the probe's tip on a groove's floor (within 0.8 mm). 1.0: the gauge was seated at some time, its reading is within 1/32 in (0.8 mm) of the 10.5/32 in (8.33 mm) depth, and it lies back on its rest (foot within 12 mm of its place, level within 5 degrees, pointing forward within 10 degrees, foot upright within 15 degrees), let go (grippers' centre points at least 3 cm from it) and still for 1 s. 0.5: seated, reading within 1/32 in, the gauge off the tyre but not back on its rest (held or not). 0 otherwise: never seated (for example the probe on a rib or a wear bar), or the reading off by more than 1/32 in (foot tipped, or the stem moved while taking it off). Real trial: watch the video for the seated foot, then read the scale and compare it with that groove's depth measured at setup.

How each condition is read in `score()`:

- **Reading:** the stem's slide position (the probe's tip past the foot's face), within 0.79 mm (1/32 in) of
  8.33 mm.
- **Seated (at some time in the run):** both ends of the foot (16 mm either side of the axis) within 1 mm
  of the tread surface (315 mm from the wheel's axis), and the probe's tip inside a groove's width, not on a
  wear bar, 8.33 +- 0.8 mm below the tread surface. Remembered for the rest of the run.
- **On its rest:** the foot's face within 12 mm of its place on the rest and its axis within 4 mm of 36 mm up; the
  axis level within 5 degrees and pointing within 10 degrees of +x; the foot upright within 15 degrees.
- **Let go:** both grippers' centre points at least 3 cm from the gauge's axis (bar end to probe tip).
- **Still:** the gauge under 1 cm/s and its stem under 0.05 m/s, for 1 s for the full score.
- **Off the bench:** the gauge below the bench top at any time: 0.

Why the shortcuts fail: measured on the GPU, the same scripted motion aimed at the wear bar that faces the robot: the stem slid in until the foot lay flat on both ribs (both ends within 0.25 mm of the tread), but the probe's tip stood on the bar, 6.73 mm below the tread; the gauge went back on its rest reading 6.66 mm (8.5/32 in, 1.67 mm short) and scored 0. On a rib the probe is pushed fully home (reads 0); with the foot tipped across the groove the reading is deep; a stem pushed in against anything but a groove floor is never seated.

## Scripted demonstration

The right hand pinches the barrel 20 mm behind the foot (each pad aimed 2.2 mm from the
axis, 2.8 mm into the barrel: about 14 N a pad, through the
pick, the carry and the push; once the foot is seated it eases to 3.5 mm, about
7.5 N a pad, to take the gauge off and carry it back), the hand pitched
30 degrees back from
straight down toward the probe (so the hand stays clear of the tread and of the indicator bar coming out of the
barrel's back). It reads the gauge's pose before the pick, its pose in the hand after the lift, and where the gauge
actually is when held square to the tread 8 mm out and again with the probe's tip 1.5 mm from the groove's mouth
(the arm sags and drifts a few millimetres; each read moves the later targets by the error), and aims the probe at the middle of the lower centre groove, 18 mm along the groove from the wear bar
facing the robot, square to the tread with the foot upright. It pushes straight in: the probe meets the groove's
floor, the stem slides back into the barrel and the foot lands on both ribs; the hand's target is 6 mm
past the tread surface, so the arm's own give sets the press. It then draws straight back out, lays the gauge on its
rest and lets go. The left arm only parks. Timeline (right hand):

| Time (s) | Step |
| --- | --- |
| 0.0 to 2.0 | over the gauge's barrel |
| 2.0 to 3.5 | pads down round the barrel |
| 3.5 to 4.3 | pinch the barrel |
| 4.3 to 4.7 | settle |
| 4.7 to 6.2 | lift the gauge |
| 6.2 to 6.8 | hold lifted |
| 6.8 to 9.8 | carry to the tread, probe 8 mm out |
| 9.8 to 10.6 | hold square to the tread |
| 10.6 to 12.1 | probe's tip to the groove's mouth |
| 12.1 to 12.7 | hold at the mouth |
| 12.7 to 16.7 | push in: probe to the floor, foot onto the ribs |
| 16.7 to 17.7 | hold the foot seated, easing the pinch |
| 17.7 to 19.7 | draw straight back out |
| 19.7 to 22.7 | carry back over the rest |
| 22.7 to 24.2 | lay it on its rest |
| 24.2 to 24.8 | let go |
| 24.8 to 26.0 | rise |
| 26.0 to 28.0 | to its parking point |

Measured (GPU, object_jitter 0, version cdd3c9e663c5): score 1.0, success at 26.5 s (549 control steps); the reading on the rest at the end, 8.28 mm (true depth 8.33 mm).

## Simulation and the real part

Rigid bodies (PhysX). The tyre is rigid and fixed (no rubber give under the few-newton press); its tread is flat 1.5 degree segments, flat across its width, straight-walled grooves, no sipes. Estimates: the stem's 2 N friction, the foot, barrel and blade sizes (scaled from the maker's photo against its 3.5 in length), the groove and rib widths and the bars' 8 mm length. The stem also has light damping (solver stabiliser). The pocket clip and the printed scale are not modelled.

- Sizes from published data: the tyre's 24.8 in diameter, 8.4 in section width, 6.7 in tread width and 10.5/32 in
  new tread depth, wear bars at 2/32 in (FMVSS 139), the gauge's 3.5 in length and 0-1 in stroke in 1/32 in. Estimates, to measure on
  the real tyre and gauge when the trial is set up: the groove and rib widths and positions, the bars' length, the
  gauge's foot (32 x 8 x 1.5 mm), barrel (10 mm), blade tip (2.8 x 1 mm), the split of its 27 g and the stem's friction
  (2.0 N static, 1.6 N sliding).
- Only the stem's blade, the front 5 mm of its pin and the last 10 mm of its indicator
  bar (with the end cap) have colliders; the rest of the pin and bar, which slide inside the barrel and the foot
  where nothing outside can reach them, are drawn only (the gripper's pads register contact a few millimetres out,
  so colliders inside the barrel could otherwise be touched through its wall).
- The real tread rubber gives a little under the foot and the probe (well under 0.1 mm at these loads); here it is
  rigid. The real tread is crowned across its width and has sipes; here it is flat and plain.
- The robot's own limits: two YAM arms, parallel grippers; the wheel is too heavy for it to move, so one hand works.

## Reset

Push the gauge's stem fully out (probe 1 in past the foot) and lay the gauge on its rest, foot upright, pointing at the wheel. The wheel stays where it is. Nothing is used up.

## Revision

Review (not accepted): "Please specify a repeatable tyre for real trials: the named tyre is sold new at 10.5/32 in,
whereas the task requires every groove to be worn to 5/32 in. Use a readily available tyre, describe how to establish
its groove depths at setup, and update the simulation and scoring targets to match the real tyre."

Changes: the same tyre, bought new ($180 to $197), so its grooves are at its published new depth, 10.5/32 in: the
simulated grooves, the scorer's target and its seated test now use 8.33 mm, and the tread band the published 6.7 in
tread width (it was 165 mm, an estimate). The setup says how to measure the grooves before the first trial and when to
reject the tyre; a real trial's reading is compared with that measurement. Demonstration, wrong-way check and Dream-1
run redone on the new version; tests renamed for the new readings.
