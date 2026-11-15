# Open a robot vacuum's brush door to clean its rollers: squeeze both catch buttons in before swinging it up

The robot vacuum is upside down for brush cleaning. Open its brush door so the rollers can come out, and leave the door standing open.

Robot vacuums need their brush rollers cleaned of wound-in hair about once a month, and the first step is opening the brush door. iRobot's Roomba i7 owner's guide: "Pinch the brush frame release tab, lift the tab, and remove any obstructions" (clean the brushes "once a month (twice a month if you have a pet)"). The i7's brush door is a narrow arm at the rollers' end, hinged at its front end; its rear end is held by a spring catch whose two green buttons stick out of either side of it in small finger recesses. Both buttons must be held squeezed in while the door's end is lifted: pulling the end without squeezing gets nowhere (the catch hooks under the recesses' floors), squeezing one is not enough, and letting go before lifting lets it catch again. Then the door has to be turned up about its hinge, the hand turning with it, until it stands open past upright.

Sources: iRobot Roomba i7/i8 Owner's Guide, "Cleaning the Multi-Surface Brushes" and the care table,
https://prod-help-content.care.irobotapi.com/files/i_series/i7/ownersGuide/og-i7-en-US.pdf ; iRobot's i Series care
page: "Open the Cleaning Head Module door by pushing on the green tab on the right side of the Multi-Surface Brushes"
(https://homesupport.irobot.com/articles/en_US/Knowledge/2450) ; robot size and mass (33.9 cm across, 9.22 cm high,
3.4 kg): iRobot's Roomba i7 product listing.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=open_robot_vacuum_brush_door.py -E object_jitter=0 -E control_hz=20 \
  --instruction "The robot vacuum is upside down for brush cleaning. Open its brush door so the rollers can come out, and leave the door standing open." \
  --scorer task_score --max-steps 1200 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial, replace `--policy yam_scripted` with `--policy dream`. `--scorer task_score` reports the score at
the end of the run. Each trial allows 1,200 control steps (60 seconds at 20 Hz). There are no
seeds and nothing loose: the vacuum is fixed where it lies, so `object_jitter` has nothing to move and every trial
starts the same. The scripted demonstration takes 20.9 s, the last 3 s
holding still. `success()` requires a score of 1.0.

`pytest` in this folder runs the generated offline checks (no GPU): the scorer on the start, the goal and the named
partial and shortcut layouts.

## The scene

Millimetres from the midpoint between the arm bases: x forward, y to the robot's left, heights above the bench top. An iRobot Roomba i7 robot vacuum (339 across, 3.4 kg; switched off) lies upside down on the bench, as its owner's guide shows for brush cleaning, its footprint centred at x 233, y 50, its front (caster) toward the robot; its underside, now on top, is 80 above the bench. Across its middle lies the cleaning head: two green rubber brush rollers (25 across, 184 long) side by side across the robot (along y), the front one dark green, the rear one light green, their tops 2 below the underside. At the rollers' left end lies the brush door: a black arm 16 wide (y 144 to 160), 94 long (x 197 to 291) and 14 thick, flush with the underside, hinged at its front end on a pin across it (at x 205, 7 below the underside), covering the rollers' bearing ends. Its rear end is held by the spring catch: two green buttons, 14 long and 15 tall, standing 3.5 out of either side of the door's end (x 273 to 287), sprung outward (2.0 N, 3.4 N squeezed flush), each in a finger recess 18 wide, 30 long and 16 deep beside the door's end; below each button a tongue reaches 3 under the recess's floor. With both buttons squeezed flush both tongues clear the floors and the door lifts; it turns up to 100 degrees (just past upright) and stays where it is put. Also on the underside: the two drive wheels in their wells either side of the cleaning head (72 across, 15 proud), the front caster, the side brush's hub and the bin's bottom. Every collider is listed in README.md.

### Collider table

Millimetres in the vacuum's own frame: its footprint's centre on the bench is the origin (in the world it is at
x 233, y 50, turned 0 degrees), x toward its back (away from the robot), y to the robot's
left, z up from the bench. The vacuum's body is one rigid body fixed where it lies (the real one weighs
3.4 kg); the brush door turns on its hinge
pin at x -28, z 73 (axis y, 0 to 100 degrees open); each catch button
slides along y on the door, sprung outward. The body's moulded underside layer is cut into 8 mm strips that follow its
round outline, less the openings listed (roller trough, door channel, finger recesses, wheel and caster
wells); the floors, rollers, wheels and other parts inside them follow. Listed with the door shut:

| Body | Shape | x | y | z |
| --- | --- | --- | --- | --- |
| vacuum body | cylinder r 169.5, axis z | -169.5 to 169.5 | -169.5 to 169.5 | 0 to 50 |
| vacuum body | cylinder r 170, axis z | -170 to 170 | -170 to 170 | 42 to 50 |
| vacuum body: underside layer | 98 boxes | the 339 mm disc in 8 mm strips along x | (each strip as wide as the disc at its narrower edge) | 50 to 80 |
| opening cut through the underside layer | box | -32 to 31 | -94 to 94 | 50 to 80 |
| opening cut through the underside layer | box | -38 to 70 | 94 to 110 | 50 to 80 |
| opening cut through the underside layer | box | 31 to 70 | 76 to 94 | 50 to 80 |
| opening cut through the underside layer | box | 31 to 70 | 110 to 128 | 50 to 80 |
| opening cut through the underside layer | box | -40 to 40 | 130 to 156 | 50 to 80 |
| opening cut through the underside layer | box | -40 to 40 | -156 to -130 | 50 to 80 |
| opening cut through the underside layer | box | -152 to -128 | -12 to 12 | 50 to 80 |
| vacuum body | box | -38 to 31 | 94 to 110 | 50 to 66 |
| vacuum body | box | 31 to 70 | 76 to 128 | 50 to 56 |
| vacuum body | box | 31 to 70 | 76 to 85.5 | 56 to 62 |
| vacuum body | box | 31 to 70 | 118.5 to 128 | 56 to 62 |
| vacuum body | box | 31 to 36 | 85.5 to 118.5 | 56 to 62 |
| vacuum body | box | 58 to 70 | 85.5 to 118.5 | 56 to 62 |
| vacuum body | box | 31 to 70 | 76 to 90.5 | 62 to 64 |
| vacuum body | box | 31 to 70 | 113.5 to 128 | 62 to 64 |
| vacuum body | cylinder r 12.5, axis y | -22 to 3 | -88 to 88 | 53 to 78 |
| vacuum body | cylinder r 10.5, axis y | -20 to 1 | 88 to 92 | 55 to 76 |
| vacuum body | cylinder r 10.5, axis y | -20 to 1 | -92 to -88 | 55 to 76 |
| vacuum body | cylinder r 12.5, axis y | 5 to 30 | -88 to 88 | 53 to 78 |
| vacuum body | cylinder r 10.5, axis y | 7 to 28 | 88 to 92 | 55 to 76 |
| vacuum body | cylinder r 10.5, axis y | 7 to 28 | -92 to -88 | 55 to 76 |
| vacuum body | cylinder r 36, axis y | -36 to 36 | 132 to 154 | 23 to 95 |
| vacuum body | cylinder r 36, axis y | -36 to 36 | -154 to -132 | 23 to 95 |
| vacuum body | cylinder r 12, axis y | -152 to -128 | -8 to 8 | 62 to 86 |
| vacuum body | cylinder r 13, axis z | -115 to -89 | -124 to -98 | 80 to 86 |
| vacuum body | box | 80 to 150 | -100 to 100 | 80 to 81 |
| brush door | box | -36 to 58 | 94 to 110 | 66 to 80 |
| brush door | cylinder r 3, axis y | -31 to -25 | 93 to 111 | 70 to 76 |
| catch button (outer) | box | 40 to 54 | 106 to 113.5 | 68 to 83 |
| catch button (outer) | box | 40 to 54 | 110.5 to 113 | 58.5 to 68 |
| catch button (outer) | box | 40 to 54 | 113 to 116.5 | 58.5 to 61.5 |
| catch button (inner) | box | 40 to 54 | 90.5 to 98 | 68 to 83 |
| catch button (inner) | box | 40 to 54 | 91 to 93.5 | 58.5 to 68 |
| catch button (inner) | box | 40 to 54 | 87.5 to 91 | 58.5 to 61.5 |

## Scoring

Judged at the end of the run with a protractor at the door's hinge and a ruler. The door's angle is measured between the door and the vacuum's underside. 1.0 (success): the door open at least 80 degrees, both grippers at least 3 cm clear of the door and its buttons, the vacuum in place (its centre within 40 mm of its start, its underside within 10 degrees of level), and the door and the vacuum still (the door's angle within 0.5 degrees and the vacuum within 2 mm over 1 s), all held for 1 s. 0.5: the door open at least 30 degrees with the vacuum in place (including open 80 or more but still held or moving). 0.25: the door lifted at least 5 degrees (the catch released) with the vacuum in place. 0: anything else: the door shut (under 5 degrees; caught, it cannot lift more than half a degree), or the vacuum dragged more than 40 mm, tipped more than 10 degrees or knocked off the bench.

How each condition is read in `score()`:

- **Door angle:** the hinge's angle from shut, in degrees (the angle between the door and the vacuum's underside).
- **Vacuum in place:** its footprint's centre within 40 mm of the start and its underside within
  10 degrees of level.
- **Let go:** both grippers' centre points at least 30 mm outside the door's and both catch buttons'
  boxes.
- **Still:** the door's angle within 0.5 degrees and the vacuum within 2 mm over the last
  1 s, held for 1 s.

## Scripted demonstration

The left hand does the work; the right arm stays parked. The left hand's fingers close across the door's end (along
y), its approach tilted 15 degrees toward the robot, one pad in each finger recess; closing to a
12 mm gap pushes both buttons flush with the door and grips the door's end. The hand then turns with
the door about its hinge (a straight-down hand cannot follow the door past about 90 degrees; the tilt keeps the whole
turn within the arm's joints), lets go at 95 degrees and slides off the door's end.

| Time (s) | Left hand |
| --- | --- |
| 0.0 to 3.0 | over the door's end |
| 3.0 to 4.5 | pads down into the recesses |
| 4.5 to 5.5 | pads beside the catch buttons |
| 5.5 to 6.5 | squeeze the catch |
| 6.5 to 7.0 | hold the squeeze |
| 7.0 to 12.2 | swing the door up in 7.5 degree steps to 95 degrees |
| 12.2 to 12.7 | hold open |
| 12.7 to 13.7 | let go |
| 13.7 to 14.9 | slide off the door's end |
| 14.9 to 15.9 | rise clear |
| 15.9 to 17.9 | both hands to their parking points |

## Checked in simulation

In the YAM Arena simulation (PhysX, 240 Hz) the scripted demonstration scores 1.0: both buttons go in 3.5 mm
(flush), the door follows the hand up and stays at 95 degrees after the hand lets go, and the vacuum does not move.
The score reaches 0.25 at 7.3 s, 0.5 at 8.6 s and 1.0 at 16.3 s.

Two wrong ways, with the same hand path and only the finger gap changed, score 0 with the door shut:

- Pads closed only onto the buttons' faces (a 23.5 mm gap, no squeeze), then the lift: the buttons stay out, the
  pads slide up off them and the door never moves.
- A half squeeze (a 19 mm gap: both buttons pressed 1.4 mm of the 3.5 mm needed), then the lift: the door rises
  0.35 degrees and stops on the tongues, the pads slip off the buttons, the buttons spring back out and the door
  drops shut.

## Simulation and the real part

Rigid bodies, 240 Hz physics, 20 Hz control. The vacuum's body (iRobot's size) is fixed where it lies: at 3.4 kg the arms' few newtons cannot shift or tip it. iRobot publishes no drawings of the door, so its size and 12 g mass, the buttons and recesses, the tongues' 3 mm hook, the springs (2.0-3.4 N), the hinge friction (0.010 N m), the 100 degree stop and the rollers' size are estimates from photos, to be measured on a real i7. The rollers are fixed parts of the body.
