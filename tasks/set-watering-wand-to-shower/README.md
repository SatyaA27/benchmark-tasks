# Set a garden hose wand from Jet to Shower: hold the wand by its handle while the other hand clicks the dial round

Set the watering wand's spray dial from Jet to Shower, then leave the wand lying on the bench.

Garden hose sprayers, pistol nozzles and watering wands alike, pick their spray with a dial on the head: you turn it until the pattern you want clicks in at the index mark. Going from a hard Jet, for rinsing a path, to a soft Shower, for plants, is an everyday chore in the 80% of US households that do lawn and garden work (National Gardening Survey 2023). The dial is held in each click by a spring, so it takes a firm twist, and the light round wand has nothing to stop it turning: twisted with one hand, the whole wand just rolls along with the dial. One hand has to hold the handle still while the other turns the dial, two clicks, and stops in a click: a dial left between two patterns sprays from neither properly.

Sources: Melnor RelaxGrip 8-Pattern 16 in. Pivot Wand, model R8633 (retailer copies of the maker's specification:
15.9 x 4.29 x 2.48 in, 2/3 lb, plastic with a rubber grip, thumb control, pivoting head, 8 patterns). Melnor's
8-pattern set (Jet, Shower, Mist, Soaker, Cone, Flat, Center, Full) as listed for its 8-pattern wands and nozzles,
whose instructions select a pattern by turning the nozzle head until it lines up.
National Gardening Association, National Gardening Survey 2023 edition (80% of US households took part in lawn and
garden activities in 2022).

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=set_watering_wand_to_shower.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Set the watering wand's spray dial from Jet to Shower, then leave the wand lying on the bench." \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial, replace `--policy yam_scripted` with `--policy dream`. `--scorer task_score` reports the score at
the end of the run. Each trial allows 2,400 control steps (120 seconds at 20 Hz). There are no
seeds: the layout is fixed (in our runs the scripted demonstration also scored 1.0 in three episodes with
`-E object_jitter=0.01`, the wand's start moved up to 10 mm). The scripted demonstration takes 32.8 s,
the last 3 s holding still. `success()` requires a score of 1.0.

`pytest` in this folder runs the generated offline checks (no GPU): the scorer on the start, the goal and the named
partial and shortcut layouts.

## The scene

A bench in front of two arms (bases 61 cm apart), seen by one overhead camera and a camera on each wrist. Distances are measured forward from the line between the arm bases and sideways from the midpoint between them. One object: a Melnor RelaxGrip 8-Pattern 16 in. pivot watering wand (model R8633; 404 mm long, 0.30 kg, black and yellow plastic with a black rubber grip), dry, no hose. It lies on the bench across the robot, 17 cm forward, centred between the arms, its head toward the right arm, the yellow thumb lever on top and the head pivoted straight, in line with the shaft. From the left: the hose coupling (33 mm across), the handle (132 mm long, 33 mm round over a flat underside), a 16 mm shaft, the head's pivot, the yellow head and its dial: a black knurled ring 63 mm across and 22 mm long at the right end, which turns on the head and clicks into eight positions, 45 degrees apart. Each of the eight spray patterns has a coloured mark on the dial's front face; the pattern selected is the one whose mark is at the top, at the head's white index mark. Round the dial: JET (red), CENTER (orange), SHOWER (blue), MIST (cyan), SOAKER (green), CONE (purple), FLAT (yellow), FULL (brown). It starts on JET; turning the dial's top toward the robot, SHOWER is two clicks round (six the other way). The dial needs about 0.2 N m to turn; the wand itself rolls freely about its axis.

### The wand

One articulated body: the wand's body (0.255 kg) with two moving parts, the dial
(0.035 kg) on a hinge about the wand's axis, and the detent ball (0.01 kg, carrying its spring's moving
mass) on a slide in the head, sprung onto the dial's pins (spring 2000 N/m, 5 N preload).
The dial's hinge has 0.08 N m of static and 0.06 N m of sliding friction (the seal's drag). Masses
spread at uniform density over each part's colliders. The wand's own frame: x along the wand from the hose end to the
head, z up through the thumb lever, origin on the axis at mid-length; it lies with its origin at
(170, 0, 31.5) mm above the bench, its x pointing at
the right arm (world -y).

| Part | Collider | Centre (wand frame, mm) | Size (mm) |
| --- | --- | --- | --- |
| body | cylinder | -111, 0, 0 | radius 16.5, length 132 along x |
| body | box | -111, 0, -18.8 | 132 x 26 x 25.5 |
| body | cylinder | -189.5, 0, 0 | radius 16.5, length 25 along x |
| body | box | -189.5, 0, -20.8 | 25 x 20 x 21.5 |
| body | box | -65, 0, 19.5 | 34 x 16 x 8 |
| body | cylinder | 40.5, 0, 0 | radius 8, length 171 along x |
| body | box | 139, 0, 0 | 26 x 26 x 30 |
| body | cylinder | 166, 0, 0 | radius 22, length 28 along x |
| body | box | 174, 0, 23 | 10 x 5 x 2 |
| dial | cylinder | 191, 0, 0 | radius 31.5, length 22 along x |
| dial | 8 boxes | the pattern marks on its front face, radius 25.5, 45 degrees apart | 1.6 x 8 x 8 |
| dial | 8 cylinders | detent pins on its back, radius 12, between the marks | radius 1.5, length 12.5 along x |
| plunger | ball | 174.5, 0, 14.1 | radius 4 |

## Scoring

Judged at the end. A pattern is selected when its mark is within 5 degrees of the index mark, i.e. the dial sits in that pattern's click. 1.0: SHOWER selected, the wand lying on the bench (both ends of its axis 15.5 to 37.5 mm above the bench), still (under 1 cm/s) and both grippers' centre points at least 5 cm from the wand's axis, all held for 1 s. 0.5: SHOWER selected but the wand still held, lifted or moving. 0.25: one click only (CENTER selected). 0: anything else: still on JET, the dial left between two clicks (more than 5 degrees from every mark), any other pattern, or the wand off the bench. Check with a photo of the dial's face.

How each condition is read in `score()`:

- **Selected pattern:** the dial's angle on the head, rounded to the nearest click (every 45 degrees); it is in
  that click when within 5 degrees of it. The ball sits in a click within about 2 degrees; turned more
  than 5 degrees off, it is on its way over a pin, and a dial left there sprays from two patterns at once.
- **Lying on the bench:** both ends of the wand's axis between 15.5 and 37.5 mm above the
  bench (31.5 mm on the dial's rim or the handle's underside, 16.5 mm on the round coupling's side: any way round).
- **Still:** the wand under 1 cm/s, its dial turning under 0.2 rad/s and its ball under 0.2 m/s.
- **Let go:** both grippers' centre points at least 5 cm from the wand's axis.
- **Off the bench:** either end of the axis below the bench top: 0.

Why the shortcuts fail: turned with one hand while the other doesn't hold the handle, the dial doesn't turn on the head:
the whole wand rolls with it. In our GPU check of exactly that (the same demonstration with the left hand parked), each
50 degree turn of the wrist rolled the wand 47 to 49 degrees about its axis while the dial stayed on JET, and let go the
wand rolled back: score 0. Stopped between two clicks, no pattern is selected: in our GPU check (the same
demonstration, the second wrist turn cut from 50 to 17 degrees) the dial was left 12.9 degrees past CENTER with its ball
2.8 mm up the next pin, held there by the seal's drag: score 0 (0.25 while it sat in CENTER's click after the first
turn). One click too many (MIST) also scores 0.

## Scripted demonstration

The left hand pinches the handle from above, 301 mm from the dial, and keeps it from turning. The right hand pinches
the dial's rim from above at its axis height (the gripper's centre on the dial's axis), and for each click lifts the
head 40 mm (the wand pivots on the hose end's underside, the left hand following), rolls its wrist
50 degrees about the dial's axis (the dial goes 45 degrees into the next click; the 5 degrees over spring
back), lowers the head to 18 mm above the bench, eases its pads 3 mm off the rim (the head settles onto the
bench), draws back along its fingers and rolls back upright. Two clicks: JET, CENTER, SHOWER. The demonstration reads
where the wand lies twice, before each click, and centres the hands on its axis (at most 30 mm and
15 degrees from the plan). The right hand's timeline:

| Time (s) | Step |
| --- | --- |
| 0.0 to 2.0 | click 1: over the dial |
| 2.0 to 3.5 | click 1: pads down round the rim |
| 3.5 to 4.3 | click 1: pinch the rim |
| 4.3 to 4.6 | click 1: settle |
| 4.6 to 6.1 | click 1: lift the head |
| 6.1 to 8.6 | click 1: roll the wrist (turn the dial) |
| 8.6 to 9.0 | click 1: hold |
| 9.0 to 10.0 | click 1: lower the head |
| 10.0 to 10.6 | click 1: let go |
| 10.6 to 11.6 | click 1: draw back off the rim |
| 11.6 to 13.1 | click 1: roll back upright |
| 13.1 to 15.1 | click 2: over the dial |
| 15.1 to 16.6 | click 2: pads down round the rim |
| 16.6 to 17.4 | click 2: pinch the rim |
| 17.4 to 17.7 | click 2: settle |
| 17.7 to 19.2 | click 2: lift the head |
| 19.2 to 21.7 | click 2: roll the wrist (turn the dial) |
| 21.7 to 22.1 | click 2: hold |
| 22.1 to 23.1 | click 2: lower the head |
| 23.1 to 23.7 | click 2: let go |
| 23.7 to 24.7 | click 2: draw back off the rim |
| 24.7 to 26.2 | click 2: roll back upright |
| 26.2 to 28.2 | to its parking point |

The left hand pinches the handle by 4.3 s, follows each lift and lowering, lets go at 26.2 to 26.8 s,
rises clear and parks by 29.8 s.

## Simulation and the real part

Rigid bodies (PhysX). No hose or water. The dial's click is a sprung steel ball between eight pins (2000 N/m, 5 N preload, about 0.1 N m) plus 0.08 N m of seal drag: both estimates, as are the dial's 63 mm size, the patterns' order, the handle's shape and the 0.30 kg spread at uniform density. The head's pivot is fixed straight and the thumb lever doesn't move. Pattern names are coloured marks; the real dial prints names.

- Sizes from the maker's specification: the wand's length (15.9 in, 404 mm), its 2.48 in depth (which bounds the head's
  dial) and mass (2/3 lb). Estimates, to measure on the real wand when the trial is set up: the dial's diameter (63 mm)
  and length (22 mm), the order of the patterns round the dial and its 45 degree clicks, the torque to turn the dial
  (about 0.2 N m here: 0.1 N m of click plus 0.08 N m of drag), the handle's cross-section (33 mm round over a flat
  underside), the coupling's size (33 mm), the shaft (16 mm), and the friction of the wand on the bench (0.55) and of
  the knurled dial (0.9).
- The head's pivot (the real one ratchets through about 180 degrees) is set straight and modelled fixed; the thumb
  lever is part of the body. No hose is attached and the trial is dry.
- The robot's own limits: two YAM arms, parallel grippers; no fixture, the wand is held by its own handle.

## Reset

Turn the dial back to JET (two clicks, its top away from the robot) and lay the wand back at its start, thumb lever up, head straight. Nothing is used up.
