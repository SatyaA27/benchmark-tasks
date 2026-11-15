# Take a burger patty's temperature through its side: hold it with a turner, probe level to the centre

Take the burger patty's temperature: hold it still with the turner, push the thermometer's probe in level through the side of the patty until its tip is at the centre, hold it still for the reading, then draw it out and lay the thermometer and the turner back down.

Checking that meat is cooked through is core work for the 1.4 million US restaurant cooks (O*NET 35-2014: "Observe and test foods to determine if they have been cooked sufficiently"; ground beef must reach 160 F). For a burger the USDA's food-safety agency says: "When cooking hamburger patties ... insert the food thermometer through the side. Make sure the food thermometer reaches the center." Its fact sheet warns that "the reading will only be correct ... if the thermometer is placed in the proper location in the food". The patty is 23 mm thick: the tip must end 50 mm in, inside a band under 8 mm tall, so a probe tilted by more than about 4 degrees misses it and by more than about 13 comes out of the top or bottom face, and the push slides a patty nobody holds. So one hand holds it down with the turner while the other drives the probe in level and holds it still for the 1-second reading, hands about 34 cm apart.

Sources: O*NET OnLine, 35-2014.00 Cooks, Restaurant (tasks and 2025 employment),
https://www.onetonline.org/link/summary/35-2014.00 ; USDA Food Safety and Inspection Service, "Food Thermometers"
(placement for hamburger patties), https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/food-thermometers ,
and its "Kitchen Thermometers" fact sheet (Revised June 2011); ThermoWorks Thermapen ONE specifications
(156 x 48 x 19 mm, 115 g, 4.3 in probe 0.12 in reducing to 0.06 in, type K thermocouple at the tip, full reading in
1 second), https://www.thermoworks.com/products/thermapen-one ; Dexter-Russell Sani-Safe Cool Blue 19713H hamburger
turner (4 x 3 in solid bevelled blade, 5 in handle, 9 in overall); San Jamar CB121812WH Cut-N-Carry board
(12 x 18 x 1/2 in).

Note on method: ThermoWorks' own burger guide also allows a top-down insertion (push past the centre, read the lowest
value). This task follows the USDA FSIS method for patties, through the side, which is also the one that keeps the
thermometer's tip in the middle of a thin food without touching the hot pan side.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=take_burger_patty_temperature_through_the_side.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Take the burger patty's temperature: hold it still with the turner, push the thermometer's probe in level through the side of the patty until its tip is at the centre, hold it still for the reading, then draw it out and lay the thermometer and the turner back down." \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial, replace `--policy yam_scripted` with `--policy dream`. Each trial allows 2,400 control
steps (120 s at 20 Hz). The scripted demonstration takes 49.5 s, the last 3 s
holding still. `success()` requires a score of 1.0. `pytest` in this folder runs the generated offline checks of the
scorer (no GPU).

## The scene

Millimetres from the midpoint between the arm bases: x forward, y to the robot's left, heights above the bench top. (1) A white co-polymer cutting board, San Jamar CB121812WH, 457 x 305 x 12.7, held still (on a damp cloth), long side along y from y -50 to 407, centred on x 300. (2) A cooked 8 oz (227 g raw) 80/20 beef burger patty, 100 across and 23 thick with a rounded edge, 170 g, browned top and bottom, lying flat on the board, centre at x 300, y 30 (its right edge 30 from the board's right edge). (3) A Dexter-Russell Sani-Safe Cool Blue 19713H hamburger turner, 140 g: solid stainless blade 102 x 76 x 1.2 with bevelled edges, handle raised 28 by the offset, blue handle 127 x 24 x 18; lying on its blade on the board along y, centred on x 300, the blade's front edge at y 95 (15 left of the patty), handle toward the left arm. (4) A ThermoWorks Thermapen ONE instant-read thermometer, 115 g, red body 156 x 48 x 19, its 110 mm steel probe (3.3 thick, narrowing to 1.5 over the last 12) folded out straight; lying display up on the bench right of the board, body centre at x 215, y -262, probe pointing left at the patty, along y, 17 off the body's centre line toward x+, its tip at y -74. Real trials: tape marks at these positions. Real patty: 8 oz 80/20 ground beef pressed to 114 (4.5 in) across, cooked to 160 F, rested; measure it before the trial.

| Collider | Size (mm) | Centre at the start (mm; x, y, height above the bench) |
| --- | --- | --- |
| board 0 | 305 x 457 x 12.7 box | (300, 178.5, 6.3) |
| patty 0 | cylinder r 46.9, length 1.5 along z | (300, 30, 13.5) |
| patty 1 | cylinder r 48.9, length 2 along z | (300, 30, 15.2) |
| patty 2 | cylinder r 49.9, length 2.5 along z | (300, 30, 17.5) |
| patty 3 | cylinder r 50, length 11 along z | (300, 30, 24.2) |
| patty 4 | cylinder r 49.9, length 2.5 along z | (300, 30, 31) |
| patty 5 | cylinder r 48.9, length 2 along z | (300, 30, 33.2) |
| patty 6 | cylinder r 46.9, length 1.5 along z | (300, 30, 35) |
| turner 0 | 76 x 102 x 1.2 box | (300, 146, 13.3) |
| turner 1 | 20 x 4 x 28 box | (300, 199, 26.7) |
| turner 2 | 24 x 127 x 18 box | (300, 264.5, 49.7) |
| thermometer 0 | 48 x 156 x 19 box | (215, -262, 9.5) |
| thermometer 1 | 30 x 40 x 0.6 box | (215, -240, 19.3) |
| thermometer 2 | 8 x 6 x 8 box | (232, -181, 9.5) |
| thermometer 3 | cylinder r 1.6, length 98 along y | (232, -135, 9.5) |
| thermometer 4 | cylinder r 0.8, length 12 along y | (232, -80, 9.5) |

The patty is 23 mm thick, 100 mm across, its edge rounded to 6 mm,
0.17 kg; the turner 0.14 kg with its centre of mass over the blade (steel blade, light handle);
the thermometer 0.115 kg. Friction (static/dynamic): patty and board (0.35, 0.3), turner and
thermometer (0.9, 0.8) (estimates).

## The probe in the patty (penetration model)

The patty is one rigid body with no hole in it. The thermometer's probe (its two cylinders) does not collide with the
patty; everything else does (the probe still hits the board, the turner and the robot; the thermometer's body and
the probe's hinge knuckle hit the patty). Every control step the task file looks at where the probe's tip is in the
patty's frame:

- **Entry.** When the tip first crosses the patty's surface, a PhysX D6 joint is switched on between probe and
  patty, along the probe's own axis as it is at that moment, from the point where it crossed the surface. Nothing is
  snapped: the joint starts exactly at the probe's pose, so the hand alone sets the entry point and angle.
- **Along the probe** a zero-speed drive capped at 1 N of cutting (only while the tip advances into meat
  it has not been through) plus 15 N per metre of shaft in the meat (both ways) acts like friction: below
  the cap it holds, above it the probe slides. At the centre that is about 1.75 N in,
  0.75 N out.
- **Across the probe** spring drives hold the tip to the line it entered on and the probe parallel to it, with a
  stiffness of 100000 N/m per metre in the meat (rotation 100000 x L^3 / 3 N m/rad), tearing (capped) at
  50 N per metre in the meat. So once in, the meat steers the probe; an entry that is tilted or off
  mid-height keeps going the way it started and runs out through the top face (a breakout: the tip is outside the
  meat, as an exit hole would show) or down to the bottom face, where the board stops it.
- **The reaction goes into the patty.** It sits on the board with friction about 0.3, about 0.5 N against its 0.17 kg,
  less than the push needed to cut in, so a patty nobody holds slides away and the tip never reaches its centre.
- **Out.** When the tip is drawn back out through its entry hole the joint is switched off.

All the forces are estimates (no published insertion force for a 3.3 mm probe in cooked ground beef was found); they
must be measured. Not modelled: the meat squashing under the turner or the probe, juices, crumbling, heat. The
thermometer's reading is not simulated; the scorer judges where the sensing tip is, as the USDA guidance does.

## Scoring

Judged over the whole trial from where the probe's tip goes inside the patty. Side entry: the probe goes in through the patty's edge, not its top or bottom face, within 20 degrees of level. Centre: the tip within 12 mm of the patty's centre line and within 3.8 mm of its mid-height (the middle third). Breakout: the tip comes out through the top or bottom face. 1.0: a reading taken (the tip at the centre in a side entry without breakout, the patty moved under 15 mm, the tip still, under 5 mm/s, for at least 1 s, the Thermapen ONE's response time), then the probe drawn out, the thermometer lying flat on the bench off the board, the turner lying flat on its blade, both let go (grippers 3 cm clear of thermometer, turner and patty), the patty moved under 15 mm from its start and not flipped, everything still for 1 s. 0.75: the reading taken, but not all put down. 0.5: the tip reached the centre in a side entry without breakout, the patty moved under 15 mm. 0.25: a side entry. 0: no side entry (a probe pushed in through the top face never counts), or the patty flipped or off the board. Real trial: the probe's depth is 110 mm minus the length still showing (ruler); its height and angle from a side photo; after withdrawal the hole in the edge (ruler) and no exit hole on the top or bottom; tape marks round the patty; the reading held steady on the display for 1 s.

How each condition is read in `score()`:

- **Side entry:** the joint's entry point is within 8.5 mm of the patty's mid-height (the
  edge, not the top or bottom face) and the probe within 20 degrees of level.
- **Centre:** the tip within 12 mm of the patty's vertical centre line and within 3.8 mm of
  its mid-height.
- **Breakout:** during an insertion, with the probe more than 3 mm in, the tip's surface
  0.2 mm or more past a face of the patty. The board stops a tip run down through the
  bottom face, so that tip rests on the board at the face and is not flagged; it is never in the middle third there.
- **Reading:** centre, side entry, no breakout in this insertion, the patty moved under 15 mm, the
  tip moving under 5 mm/s against the patty, for 1 s in a row.
- **Put down:** the thermometer's body flat (within 15 degrees) on the bench, off the board; the turner flat (within
  10 degrees) on its blade, its blade's underside within 4 mm of the board's top (on the board or the bench, not on the
  patty); both grippers' centre points 30 mm clear of the thermometer, turner and patty.
- **Spoiled (0 for the trial):** the patty tipped over 30 degrees or off the board.

### What our GPU runs showed

Official YAM Arena environment (Isaac Lab, PhysX), 20 Hz, this version of the task file. The per-step numbers come
from a copy of the same file that only adds printing (and, for the wrong ways, changes the scripted hand).

- **Scripted demonstration**, no jitter: **1.0**, `success()` at step 897 (44.85 s).
- **The same demonstration with jittered start poses** (object_jitter 0.01, seed 2), 3 trials: **3 of 3 at 1.0**.
  The probe went in level (0.00 degrees), 11.4 to 11.6 mm above the board (mid-height 11.5), 0.2 to 2.8 mm to the
  side of the centre line. During the reading the tip was 3.5 to 4.0, 9.7 to 10.9 and 3.6 to 4.5 mm from the centre
  line (the limit is 12) and within 0.1 mm of mid-height. It was held there for 2.7 to 2.8 s, against 0.6 to 0.7 N of
  resistance along the probe. Under the turner the patty moved 6.9 to 7.3, 6.4 to 6.6 and 1.4 to 1.6 mm by the
  reading, and was 3.4 to 4.4 mm from its start at the end. The turner's blade stayed flat (0.2 to 0.3 degrees). The
  hand forces estimated from the arms' joint errors stayed under 7 N after the start, and about 2.5 N on the turner
  during the reading.
- **Wrong way, no turner** (the left hand stays parked, the right does the same push): **0.25**. The probe went in
  level at mid-height, but the patty slid 59 mm ahead of it, so the tip got only 32 mm in, never nearer than 17.6 mm
  to the centre line.
- **Wrong way, probe tilted 18 degrees nose down** (the same aim, at mid-height): **0.25**. The meat kept the probe
  on its tilted line. The tip ran down to the bottom face 37 mm in and rode along the board under it, never in the
  middle third near the centre.
- **Wrong way, pushed in through the top face** at 25 degrees (the tip aimed at the centre): **0.0**. The tip passed
  within 1.1 mm of the centre line at mid-height, but through the top face, which never counts as a side entry.
- **Dream-1** (the task's full 2,400 steps): **0.0**. The left hand went to the turner's handle and the right hand
  over the thermometer, but neither tool was picked up.

## Scripted demonstration

Reads (simulator state, which a scripted demo may use) aim the probe at the patty where it actually lies and correct
each hand for how it holds its tool. Left hand (turner):

| Seconds | Step |
| --- | --- |
| 0.0 to 2.5 | over the turner's handle |
| 2.5 to 4.0 | pads down round the handle |
| 4.0 to 5.0 | pinch the handle |
| 5.0 to 6.5 | lift the turner |
| 6.5 to 9.0 | blade over the patty's far half |
| 9.0 to 10.5 | blade down on the patty |
| 10.5 to 11.5 | hold it down |
| 11.5 to 36.5 | keep holding |
| 36.5 to 38.0 | lift the turner off |
| 38.0 to 40.5 | back over its place |
| 40.5 to 42.0 | lay it down |
| 42.0 to 43.0 | let go |
| 43.0 to 44.5 | rise clear |
| 44.5 to 46.5 | park |
| 46.5 to 49.5 | hold clear |

Right hand (thermometer):

| Seconds | Step |
| --- | --- |
| 0.0 to 2.5 | over the thermometer |
| 2.5 to 4.0 | pads down round the body |
| 4.0 to 5.0 | pinch the body |
| 5.0 to 6.5 | lift it |
| 6.5 to 12.0 | wait for the turner |
| 12.0 to 15.0 | probe beside the edge, level |
| 15.0 to 16.5 | probe at mid-height, short of the edge |
| 16.5 to 18.0 | settle |
| 18.0 to 19.0 | line the probe up on the patty's centre |
| 19.0 to 20.5 | settle |
| 20.5 to 24.5 | push it in to the centre |
| 24.5 to 27.0 | hold for the reading |
| 27.0 to 30.0 | draw it straight out |
| 30.0 to 32.5 | back over its place |
| 32.5 to 34.0 | lay it down |
| 34.0 to 35.0 | let go |
| 35.0 to 36.5 | rise clear |
| 36.5 to 38.5 | park |
| 38.5 to 41.5 | hold clear |

## Real-world trial

Use a freshly cooked patty each trial. Lay tape marks round the patty, the turner and the thermometer. Score from a
side photo at the moment of the reading (probe height and angle, the probe length still showing: depth = 110 mm minus
that), the hole in the edge and both faces after withdrawal, and the tape marks. A probed patty can be cut through
along the hole to confirm where the tip was. Keep hands away from the probe tip.

## Files

- `take_burger_patty_temperature_through_the_side.py`: the scene, the penetration model, `score()` / `success()` and the scripted `demo()`.
- `yam_parts.py`: compound rigid parts (boxes and cylinders) for YAM Arena task files.
- `task.json`: the submission fields.
- `test_take_burger_patty_temperature_through_the_side.py`: generated offline checks of the scorer.
