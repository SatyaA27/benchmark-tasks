# Extend a suitcase's handle to full height: keep its button pressed past the first stop, then let both rails lock

Pull the suitcase's handle out to its full height and leave it locked there.

Pulling out a wheeled suitcase's handle is something most people do every trip. The handle is two telescopic rails joined by the grip; a button in the grip works a spring pin in each rail, and the pins lock the handle closed, at a first height and at full height. Thumb on the button, fingers under the grip, you draw it out in one straight stroke and let the button up at the top, where both pins click in. Let it up early and the handle stops at the first height; let it up at the top before the rails reach their stop and it is not locked; pull it crooked and the paired rails drag. Here a 55 cm hard-shell cabin case lies flat on the bench, handle side up (stood up, its grip would be above the arms' reach): one hand holds the case down, the other pinches the grip with one finger pad on the button and draws the handle out 520 mm along its rails.

Product: American Tourister Airconic Spinner 55/20 TSA (item 128186), maker's figures 55 x 40 x 20 cm including
wheels, 2.0 kg, polypropylene shell, "Double tube" wheel handle:
https://www.americantourister.co.uk/airconic-spinner-4-wheels-55cm-cool-grey/128186-2447.html and the maker's
catalogue (55 x 40 x 20 cm, 33 L, 2 kg). Handle travel: a retailer's listing of the same range's 55 cm spinner gives
"telescopic trolley system adjustable, fixable: approx. 0 cm - approx. 52 cm (2 steps)". Release-button force: a
telescopic-handle parts maker's specification, "Button pressure: 5-8N".

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=extend_suitcase_handle.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Pull the suitcase's handle out to its full height and leave it locked there." \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial, replace `--policy yam_scripted` with `--policy dream`. `--scorer task_score` reports the score at
the end of the run. Each trial allows 2,400 control steps (120 seconds at 20 Hz). The scripted
demonstration takes 26.7 s, the last 3 s holding still. `success()` requires
a score of 1.0.

`pytest` in this folder runs the generated offline checks (no GPU): the scorer on the start, the goal and the named
partial and shortcut layouts.

## The scene

Millimetres from the midpoint between the arm bases: x forward, y to the robot's left, heights above the bench top. An American Tourister Airconic Spinner 55/20 hard-shell cabin case (Deep Ocean teal polypropylene, 550 long including its wheels, 400 wide, 200 deep, 2.0 kg, empty, zipped shut) lies flat on its front on the bench, its back panel (the handle side) up and its handle end toward the robot's right: the handle end's corners at (6, -32) and (394, -128), the wheel end's at (139, 502) and (527, 405). Its twin-tube telescopic handle is closed: the black grip (155 long, 21 thick, 32 deep) lies in a pocket in the handle end's upper edge (190 wide, 49 deep, 45 down from the back panel), its outer face flush with the end, a 40 x 20 grey release button in the middle of that face standing 2 proud, and a 26 wide finger gap between the grip and the pocket's floor. The grip's centre is at (203, -68), 178 up. Each rail is an aluminium telescope inside the back panel, the rails 130 apart: a middle tube (17 across) and an inner tube (13) that carries the grip. Drawn fully out, the grip is 520 further along the case's axis, at (77, -572); the first locking height is half way, 260 out. Nothing else is on the bench. Every collider is listed in README.md.

### Collider table

Millimetres, in the case's own frame: x across the case (positive toward the case's right-hand side as it lies), y
from the handle end toward the wheels, z up from the bench. The case's frame origin is at x 200, y -80
in the world (midpoint between the arm bases), on the bench, turned -14 degrees about the
vertical (so the handle draws out toward the robot's right and slightly toward the robot). Tube and grip rows are at
the closed handle; the middle tubes move with joint `mid`, the inner tubes and grip with `mid` + `inner`, the button
with those + `button`.

| Body | x | y | z |
| --- | --- | --- | --- |
| shell (fixed to the case) | -200 to 200 | 0 to 480 | 0 to 155 |
| shell (fixed to the case) | -200 to 200 | 49 to 480 | 155 to 200 |
| shell (fixed to the case) | 95 to 200 | 0 to 49 | 155 to 200 |
| shell (fixed to the case) | -200 to -95 | 0 to 49 | 155 to 200 |
| shell (fixed to the case) | -75 to 75 | -4 to 0 | 80 to 110 |
| shell (fixed to the case) | -204 to -200 | 180 to 300 | 85 to 115 |
| shell (fixed to the case) | -201 to 201 | 0 to 480 | 98 to 104 |
| shell (fixed to the case) | -195 to -140 | 480 to 498 | 6 to 62 |
| shell (fixed to the case) | cylinder r 26, length 40 along x, centre (-168, 524, 34) | | |
| shell (fixed to the case) | -195 to -140 | 480 to 498 | 138 to 194 |
| shell (fixed to the case) | cylinder r 26, length 40 along x, centre (-168, 524, 166) | | |
| shell (fixed to the case) | 140 to 195 | 480 to 498 | 6 to 62 |
| shell (fixed to the case) | cylinder r 26, length 40 along x, centre (168, 524, 34) | | |
| shell (fixed to the case) | 140 to 195 | 480 to 498 | 138 to 194 |
| shell (fixed to the case) | cylinder r 26, length 40 along x, centre (168, 524, 166) | | |
| middle tubes (joint mid) | -73.5 to -56.5 | 49 to 470 | 169.5 to 186.5 |
| middle tubes (joint mid) | 56.5 to 73.5 | 49 to 470 | 169.5 to 186.5 |
| inner tubes and grip (joint inner) | -71.5 to -58.5 | 47 to 450 | 171.5 to 184.5 |
| inner tubes and grip (joint inner) | 58.5 to 71.5 | 47 to 450 | 171.5 to 184.5 |
| inner tubes and grip (joint inner) | -77.5 to 77.5 | 2 to 23 | 162 to 194 |
| inner tubes and grip (joint inner) | -77.5 to -52.5 | 23 to 47 | 162 to 194 |
| inner tubes and grip (joint inner) | 52.5 to 77.5 | 23 to 47 | 162 to 194 |
| release button (joint button) | -20 to 20 | -2 to 10 | 168 to 188 |

## How the handle works in the simulation

The case is one articulated body with a free (floating) shell of 1.68 kg: it lies on the bench under its own
weight and slides if pushed hard enough (friction 0.4). Joint `mid` slides both middle tubes together,
joint `inner` slides both inner tubes and the grip, each 260 mm; their sum is the handle's extension (closed 0,
first height 260, full 520). Joint `button` is the release button: 4 mm of
travel on a spring that holds it out with 5 N and takes
8 N fully pressed.

The two lock pins and the rods from the button to them are inside the tubes and are modelled in `case_runtime.py`, run
before every physics step: the left pin clears its holes once the button is 3 mm in, the right
once it is 3.4 mm in (they spring out again 0.3 mm short of that). A pin that is out
drops into its holes when they line up within 1.5 mm (closed, first height or full height), and the handle
is then held there with 0.5 mm of play by the joints' limits. While either pin is out and in its holes, the
handle cannot be moved; with the button partly pressed one pin is still in and holds the handle. The tubes slide with
bushing friction (static 1 N inner, 2 N middle) plus PhysX
joint friction proportional to the load across the rails (coefficient 0.2), so a crooked pull that loads
the rails sideways drags harder.

## Scoring

Judged at the end of the run with a tape measure and by hand. Full height: the grip 520 mm (+-3) out from closed, both rails at their stops. Locked: both pins in their holes with the button back out (on the real case: a 10 N push on the grip toward the case moves it less than 3 mm). Clear: both grippers at least 5 cm from the grip. Kept in place: the case within 50 mm and 10 degrees of where it lay. 1.0 (success): locked at full height, clear, held 1 s, and the case kept in place. 0.5: the same, but the case was dragged or turned further. 0.25: locked at the first height (260 out) instead, clear, held 1 s. 0: anything else, including the handle out but not locked (button still held, or let go between the heights), still closed, or the case tipped off its front at any time.

Why the shortcuts fail, as measured in the simulator (GPU check runs of this scene, same scripted demonstration):

- Pinched beside the button (the pads 40 mm along the grip, clear of it), the button stays out and both pins stay in
  their holes: the handle did not move (0.5 mm); the pull dragged the whole case 389 mm and turned it 53 degrees
  instead. Score 0.
- Drawn crooked (the hand's path turned 12 degrees off the rails' line), the case was dragged 46 mm and turned 4.7
  degrees, and the middle tubes stopped 22 mm short of their stop; let go there, the pins found no holes, the handle
  was not locked and slid back to 487 mm. Score 0.
- Let up at the first height, the pins drop in at 260 mm: 0.25 (offline layout). Out at full height with the button
  still held, or between the heights: not locked, 0.

The scripted demonstration itself scored 1.0 in three episodes with `-E object_jitter=0.01` (the case's start moved
up to 10 mm) and in the run with `object_jitter=0`.

## Scripted demonstration

The right hand pinches the grip straight down from above, fingers closing along the case's axis: the far pad on the
button, the near pad in the finger well under the grip, so the pinch itself holds the button in. It draws the handle
out along the case's axis (read from the case's pose at the start) to the stop at full height, eases the pinch so the
button springs back and both pins drop into their holes, lets go and rises clear. The left hand presses its closed
pads 10 mm into the case's upturned back panel 270 mm down the case from the handle end, so the
case cannot slide or turn while the handle is pulled, then lifts off.

Right hand:

| Time (s) | Step |
| --- | --- |
| 0.0 to 2.5 | over the grip, pads open across it |
| 2.5 to 3.7 | pads beside the grip and button |
| 3.7 to 4.9 | pinch: button in |
| 4.9 to 5.5 | settle |
| 5.5 to 7.5 | draw out: start |
| 7.5 to 17.5 | draw out to the stop |
| 17.5 to 18.1 | hold at the stop |
| 18.1 to 18.9 | ease the pinch: button back |
| 18.9 to 19.7 | pins in |
| 19.7 to 20.5 | let go |
| 20.5 to 21.7 | rise clear |
| 21.7 to 23.7 | to its parking point |

Left hand:

| Time (s) | Step |
| --- | --- |
| 0.0 to 2.5 | over the case's back panel |
| 2.5 to 3.5 | press: brace the case |
| 3.5 to 20.2 | hold the case |
| 20.2 to 21.2 | lift off |
| 21.2 to 23.2 | to its parking point |

## Simulation and the real case

Rigid bodies, 240 Hz physics. The case is one articulated body: a free shell, the middle tubes and the inner tubes with grip on two slides, the button on a third. Pins and button rods are not bodies: case_runtime.py turns the button's travel into pins in or out and stops the slides where the holes line up. Estimates to measure on the real case: 260 mm per stage, pin clearances, tube friction, 4 mm button travel, grip, pocket and tube sizes. Laid flat, handle up: stood up it is out of reach.

Simulation shortcuts, all of them:

- The pins, their springs and the rods from the button are not simulated bodies: `case_runtime.py` decides from the
  button's travel which pin is out, and holds the slides with joint limits where an out pin's holes line up.
- The two middle tubes are one body and the two inner tubes with the grip are one body, each on one slide: the rails
  cannot rack against each other. A crooked pull only adds PhysX joint friction in proportion to the load across the
  slide (coefficient 0.2, an estimate).
- The tubes do not touch the shell (they pass through its solid colliders, guided by the slides); the shell is boxes
  without its ribs and rounded corners; the wheels and carry handles are fixed shapes.
- At every reset the handle's joints are put back closed (the simulator's reset leaves articulated joints where they
  were).
- The demonstration reads the case's pose once, at the start, to place its hands.
- The case lies flat, handle side up: stood on its wheels its grip would be about 55 cm up, beyond the arms' reach.

Estimates, to measure on the real case when the trial is set up (the maker gives only the outside size, mass and
"double tube" handle): travel per section 260 mm (from a retailer's "approx. 0 - 52 cm (2 steps)" for the
same range) and the first height at half way; grip 155 x 21 x
32 mm and rails 130 mm apart (scaled from the maker's photo); pocket
190 x 49 x 45 mm and the 26 mm finger gap;
tube sizes (17 and 13 mm, square colliders as wide as round tubes); button
40 x 20 mm, 4 mm travel, 2 mm
proud (its 5-8 N is the parts maker's range); the pins' clear points (3 and
3.4 mm of button), capture 1.5 mm and play 0.5 mm; tube friction
(static 1 and 2 N); handle masses
(0.14 kg middle tubes, 0.17 kg inner tubes and grip,
0.01 kg button, the rest of the 2.0 kg in the shell); shell-on-bench friction
0.4; the wheel housings and wheels.

## Reset

Press the button and push the handle fully closed until it clicks, then lay the case back on its front at the marked place, handle end toward the robot's right. Nothing is consumed or fastened.
