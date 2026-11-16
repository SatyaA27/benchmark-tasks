# Load a clipboard: one hand holds the clip's lever pressed down while the other slides a loose card under it to the stop

Clip the yellow card onto the clipboard: under the clip as far as it goes, straight on the board.

One hand presses the clip's lever down and holds it (the jaw lifts; the hand also pins the light board) while the
other slides a loose card, too thin to grip, flat under the jaw to the clip's stop and squares it; then the lever is
let go and only the clip's spring holds the card. A scripted tilt of the board checks that.

## Scoring

Judged once the robot has let go, with a ruler, a protractor (or a phone level) and a pencil.

- **Let go and still.** Both grippers at least 3 cm from the clipboard and the card (`released`, 30 mm from each
  body's bounding box), nothing moving more than 0.5 mm or 0.25 deg between control steps for 1 s (`at_rest`), and the
  board flat on the bench (under 3 deg, within 3 mm). Only then is anything judged.
- **Held by the clamp** (`clipped`): the card lies on the board's face; its top edge is at least 5 mm past the front
  face of the jaw's lip at both ends of the 92 mm lip (`BITE_MIN`); the lip rests on the card, 0.05 to 1.5 mm off the
  board's face (`LIP_ON`; 0 on the bare board, several mm held open); and in the check below the card's top edge
  moves no more than 3 mm on the board (`SLIP_MAX`).
- **The check** (`tilt_check`): the board is lifted by the middle of its clip end until it stands at 60 deg on its
  bottom edge (2.5 s), held there 2 s and left there. The scorer applies this as an upward force at that point; a
  person does it by hand. A pencil tick on the board at the card's top edge shows any slip.
- **On the stop** (`seated`): the card's top edge within 2 mm of the stop (the clip base plate's front edge) at 40 mm
  either side of the clip's centre line, measured along the board (0.5 mm allowed past it for contact).
- **Square** (`square`): the card's edges within 2 deg of the board's, and no corner past either long edge of the
  board (card 216 mm wide, board 229 mm).

| Score | Condition |
|---|---|
| 1.0 | let go and still, held by the clamp through the check, on the stop, and square |
| 0.75 | let go and still, held, and only one of: on the stop; square |
| 0.5 | let go and still, held, neither on the stop nor square |
| 0 | anything else: the jaw resting on the board, under 5 mm of card under the jaw, a slip over 3 mm in the check, a gripper within 3 cm of the clipboard or card, the board not flat, or the clipboard off the bench |

`score()` returns these values; `success()` is `score() >= 1.0`. On the stop and square are taken when the check
starts; the score is reported once the check is over and stays for the rest of the run.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=clip_a_sheet_onto_a_clipboard.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Clip the yellow card onto the clipboard: under the clip as far as it goes, straight on the board." \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial, use `--policy dream`. Each trial allows 2,400 control steps (120 s at 20 Hz). The official layout
is fixed (`object_jitter=0`); `-E object_jitter=0.01` shifts the clipboard and the card each up to 10 mm in x and y.
The scripted demonstration's plan is 26.5 s; it scores 1.0 at about 21.4 s,
after the check. `pytest` in this folder runs the offline checks of the scorer (no GPU).

## The scene

Objects, sizes and materials are listed in `task.json` (setup). Millimetres from the midpoint between the arm bases:
x forward, y to the robot's left, z above the bench top. Every collider (the lever drawn closed, its lip on the board):

| Body | Part | Shape | Size | Centre / extent |
|---|---|---|---|---|
| base | board | box | 228.6 x 317.5 x 3.2 | (200.0, 90.0, 1.6) |
| base | clip base plate (its front edge is the stop) | box | 90.0 x 21.0 x 0.80 | (200.0, 233.2, 3.6) |
| base | hinge ear | box | 3.0 x 8.0 x 9.5 | (152.0, 234.8, 7.9) |
| base | rivet head | cylinder | r 3.0, 1.2 long (z) | (175.0, 233.2, 4.6) |
| base | hinge ear | box | 3.0 x 8.0 x 9.5 | (248.0, 234.8, 7.9) |
| base | rivet head | cylinder | r 3.0, 1.2 long (z) | (225.0, 233.2, 4.6) |
| base | hinge pin and spring coil | cylinder | r 1.8, 99.0 long (x) | (200.0, 234.8, 10.2) |
| lever | lip (turned-down front edge of the jaw) | box | 92.0 x 0.80 x 4.0 | (200.0, 205.2, 5.2) |
| lever | tab roll | cylinder | r 2.0, 92.0 long (x) | (200.0, 255.1, 22.9) |
| lever | hinge knuckle | cylinder | r 2.6, 92.0 long (x) | (200.0, 234.8, 10.2) |
| lever | jaw plate, hinge to lip | convex prism, 0.8 thick | 92.0 wide | y 205.1 to 232.3, z 6.8 to 10.3 |
| lever | thumb tab, 32 deg | convex prism, 0.8 thick | 92.0 wide | y 236.7 to 254.3, z 11.2 to 22.6 |
| card | sheet | box | 215.9 x 279.4 x 0.24 | (200.0, 38.1, 3.3) |

The clipboard is one free articulation: the board with the riveted clip base (216.0 g; mass
properties from its colliders at uniform density) and the lever (33.5 g) on a revolute joint
along the pin. The lever does not collide with its own base; its closed position (lip on the board) is the joint's
limit, so anything under the lip holds it open by its thickness. The card (12.0 g) is a free body.

## The clip and its spring (estimates)

Saunders publish the board's size and the clip's capacity (1 in) but no clip drawing or force, and no measured force
for a clipboard clip was found. The profile is that of the common 4 in clipboard clip (hinge pin 14 mm in from the
board's top edge and 7 mm above its face, jaw 30 mm to a square turned-down lip, tab 24 mm at 32 deg ending in a 4 mm
roll). The spring is set from what the clip must do: with its rated 1 in of paper (1.1 kg) the board should still
hang by the clip, about 10-13 N on the lip fully open. A torsion spring wound 120 deg at closed with
0.1146 N m/rad gives 8.0 N on the lip closed and 12.6 N at 45 deg open; pressed straight down at the roll
it takes 11.7 N to start the jaw lifting. Frictions (static/dynamic, the mean of the two materials): card on
hardboard 0.35/0.30, hardboard on the bench 0.35/0.33, card on steel 0.40/0.35, the robot's pads on card 0.50/0.48.

## Scripted demonstration

It reads the board's and card's poses at the start and again when its fingertips are on the card. Left hand:
straight down, fingers closed, onto the tab's roll 35 mm from the clip's centre and down until the spring stops it
(13 N from the joints, the jaw 33-34 deg open, the lip 17-18 mm up); the press also
pins the board. Right hand: pitched 60 deg, fingers closed, fingertips on the card's centre line 75 mm behind its top
edge; it slides the card 45 mm up the board under the jaw to the stop, the fingertips going on to
15 mm past it: the card's top edge meets the stop and the push from its middle turns it square while the
fingertips slip on it, as one finger squares a sheet. The left hand
rises slowly so the jaw comes down on the card, the right hand lifts off, both go back to ready, and the check runs.
In the demonstration the card reaches the stop (within 0.1 mm) and scores 1.0; in six seeded diagnostic runs (objects shifted up to 10 mm) the left hand held 12.8-13.3 N on the lever, the lip came down 0.12-0.13 mm off the board onto the card, and the card did not move (0.00 mm) in the 60 deg check.

## Why the shortcuts fail

Measured on the GPU with a copy of this module that runs the check anyway at 21 s. Letting the lever go before the card is under the jaw: the card is pushed up the board to the closed jaw and stops against its lip (its top edge 0.2 mm short of the lip's face), the lip rests on the bare board (0.00 mm), and the score is 0 (not clipped). Pushing the card at the clip without pressing the lever: the same, stopped at the lip, 0. A card turned a few degrees and stopped at the lip slid 11-13 mm down the board in the check; the clipped card did not move.

## Simulation and the real objects

Rigid bodies, 240 Hz, 20 Hz control. The card is one rigid 0.24 mm plate; it always lies supported on the board or under the jaw, where real card stays flat too. The clip is a rigid lever on a sprung hinge; its profile, the spring (8 N on the lip, ~12 N at the tab) and every friction are estimates to be measured. The lip-on-board contact is the hinge's stop. The 60 deg check is an upward force the scorer applies at the board's clip end. The demo reads the board's and card's poses. The card is rigid (real 110 lb card bends, but here it is never lifted: it lies on the board or under
the jaw). Estimates to be measured on a real clipboard: the clip's profile, its spring and its 33.5 g
lever, and every friction coefficient; the board's mass is spread evenly.

Real parts: Saunders 05612 Recycled Hardboard Clipboard, letter (9 x 12-1/2 in, 1/8 in hardboard, 1 in clip, 8.8 oz);
Neenah Exact Index card stock, 110 lb (199 g/m2), 8.5 x 11 in, canary.

## Reset

Press the clip's tab, take the card out and lay it back on the board square and centred, its top edge 45 mm short of the stop; lay the clipboard back at its marks. Nothing is consumed or bent.

## Files

- `clip_a_sheet_onto_a_clipboard.py`: scene, clip mechanism, scorer (with the check) and scripted demonstration
- `yam_parts.py`, `yam_joints.py`, `yam_mesh.py`: kit modules, copied unchanged
- `test_clip_a_sheet_onto_a_clipboard.py`: offline checks of the scorer (generated)
- `task.json`: the task's fields
