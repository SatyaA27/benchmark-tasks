# Scratch the silver PIN coating off a gift card with a coin without wearing away the code, while the other hand pins it

Pick up the quarter and use its edge to scratch the silver coating off the code on the back of the gift card, which is still on its plastic hanger, while your other hand presses the hanger's far end flat on the table. Uncover every character without scratching any of them away.

Sources: PlasCards "Hanging Tear-Off Card" ("Hanging Gift Card - 6.625\""), product page and print template: trim
3.375 x 6.625 in, tear-off card 3.375 x 2.125 in, 1.129 in sombrero hang slot, 30 mil PVC, scratch-off bars an option;
ISO/IEC 7810 ID-1 (the CR80 card); US Mint, "Quarter" (24.26 mm, 1.75 mm, 5.670 g, cupronickel clad); the panel's
9.5 x 63.5 mm is the size of the common 3/8 x 2-1/2 in PIN scratch-off label. The code's character size and length,
the slot's height and the piece's 15 g are estimates.

## Scoring

Judged 2 s after the run ends, from the video and a top-down photo of the card at 10 pixels per mm or better. A character is readable when at most one fleck of silver no bigger than 0.5 x 0.5 mm is left on its 2.0 x 2.5 mm box and a person who does not know the code reads it right from the photo; it is worn when any of its printed strokes shows a gap of 0.3 mm or more. Only the quarter's rim may remove coating (checked in the video): coating peeled or lifted off, or rubbed off by a finger or by dragging the card on the table, scores 0. Score: 0 if any character is worn, or the card ends more than 5 cm or 30 degrees from where it started, or flipped. Otherwise 1.0 for all 15 readable with both grippers at least 3 cm clear of the card and hanger; 0.75 for 13-14 readable (or 15 with a gripper still within 3 cm); 0.5 for 9-12; 0.25 for 4-8; 0 for 0-3. How the card is held is not scored by itself: an unheld card slides under the strokes, which then miss the code. In simulation the coating is 2,413 cells of 0.5 x 0.5 mm (20 over each character's box; a character reads with at most 1 left). Each physics step, a cell under the rim's footprint (3 mm along its edge, 1 mm across) is cleared if the quarter's rim, standing at least 30 degrees off flat, presses on the card with at least 0.5 N (contact force from the physics engine, averaged over 1/40 s); a character is worn if the rim crosses it pressing above 6 N, or comes onto it pressing more than 3 separate times. These three numbers are estimates, to be calibrated on real cards on a kitchen scale (see README).

| End state | Score |
|---|---|
| All 15 characters readable, none worn, card in place, both grippers 3 cm clear | 1.0 |
| 13-14 readable (or all 15 with a gripper still within 3 cm of the card or hanger) | 0.75 |
| 9-12 / 4-8 / 0-3 readable | 0.5 / 0.25 / 0 |
| Any character worn (pressed over 6 N, or a 4th pass) | 0 |
| Card moved over 5 cm, turned over 30 degrees, or flipped | 0 |

The simulated record (`scratch_step()` in `scratch_gift_card_pin_off_with_a_coin.py`): a PhysX contact view reports, every physics step (240 Hz), the
normal force and points of every contact between the quarter and the card. A contact is the rim's when its
force-weighted point lies within 1 mm of the coin's edge and the coin's face stands at least 30 degrees off the card (a
coin dragged flat removes nothing). The force is averaged over the last 6 steps (1/40 s); while it is at least 0.5 N,
every coating cell centred within the footprint (1.5 mm either way along the rim's edge, 0.5 mm across, on the contact
point) is cleared. A character's pass count goes up when the pressing footprint comes onto its box after being off it
for over 0.1 s. A character is readable when at most one of the 20 cells over its 2.0 x 2.5 mm box is still coated (a
lone 0.5 mm fleck does not stop a person reading it). The scorer reads only this record, the card's pose and the
grippers' positions; the silver squares in the video are drawn from the same record.

Calibration (the three numbers are estimates until then): pass a quarter on edge across real cards lying on a kitchen
scale at set forces; the lowest force that clears a whole track sets the 0.5 N, the lowest that leaves a 0.3 mm gap in a
printed stroke sets the 6 N, and the passes at 2 N before such a gap, less one, set the 3.

Why the shortcuts fail (measured in simulation): with no left hand on the hanger (same start layout as the demo video) the strokes drag the card and hanger 315 mm across the table and turn them 20 degrees, and only 6 of the 2,413 cells come off: 0. Scrubbing the first character's place four times over wears it on the fourth pass: 0 (the quarter then slipped out of the fingers). A coin laid flat or slid on its face removes
nothing (the rim rule), and dragging the card off its place scores 0.

## Run and score

From this task folder, with the official YAM Arena environment installed (the demo video's run; seed and 10 mm start
jitter as given):

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=scratch_gift_card_pin_off_with_a_coin.py -E control_hz=20 --seed 2 -E object_jitter=0.01 \
  --instruction "Pick up the quarter and use its edge to scratch the silver coating off the code on the back of the gift card, which is still on its plastic hanger, while your other hand presses the hanger's far end flat on the table. Uncover every character without scratching any of them away." \
  --scorer task_score --max-steps 3600 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial, replace `--policy yam_scripted` with `--policy dream` (the Dream-1 run used `-E object_jitter=0`
and no seed). Each trial allows 3,600 control steps (180 seconds at 20 Hz). The scripted demo is not fully reliable: the fingers hold the quarter by only the top half-millimetre of its edge, and it sometimes twists out in the first strokes. Measured with this version, 10 mm start jitter: full score in 3 of 5 episodes (`--seed 1 --epochs 3`: 0, 1.0, 1.0; `--seed 2`: 1.0); 0 with `-E object_jitter=0` (the quarter fell at stroke 2). The video is the `--seed 2` run: full score at 73.0 s of the 75.5 s plan.

`pytest` in this folder runs the generated offline checks (no GPU): the scorer on the start, the goal and the named
partial and shortcut layouts, and the demo's reach.

## The scene

Positions and products: task.json's setup. Card and hanger are one rigid body of four box colliders round the hang slot (85.7 x 168.3 x 0.762 mm, 15 g); the code
(5 x 7 black dots per character), the coating (2,413 silver squares of 0.5 mm) and the tear line are drawn on its top
face without collision. The quarter is a square-edged 32-sided prism of the coin's area and mass. Friction (PhysX
averages the pair; static / sliding): card on the oak table 0.325 / 0.30 (PVC on wood is measured at about 0.30
sliding), the quarter's rim on the coating 0.45 / 0.425, the quarter on the finger pads 0.625 (estimates).

## Scripted demonstration

Both arms (75.5 s plan). The left hand closes its fingers, comes over the hanger's far end and, tilted 25 degrees
about the fingers' closing axis, presses the closed pads' edge onto the hanger 130 mm beyond the code line (12 mm below
the hang slot) until the coin has been lifted clear. This is a deliberate fingertip press, as a person holds a card down
with a fingertip: the pads' edge is commanded 5.2 mm into the 0.76 mm plate, so the arm stops on the hanger and presses
it through its own give (1.8-2.0 N, measured). The offline check flags any fingertip aimed below its usual floor over the table;
it reports this press as a warning (press_below_floor), allowed because the hanger is listed in `PRESSED_PARTS` and the
fingertips meet the hanger, not the table. The right hand comes down over the quarter, corrects sideways from where it
really is, lowers the pad tips to 0.5 mm above the table and pinches the coin's rim (about 8 N), then squeezes to about
25 N while the coin still lies flat (the table keeps it from turning in the pinch). It lifts and turns the hand to point
along the hanger, 26 degrees below level, so the coin stands on edge in front of the fingertips. Then 24 strokes across
the code, 2.3 mm apart, the first two just left of it: the rim comes down onto the card already moving back toward the
arm (so the card's friction keeps the coin's lean in the pinch), presses 5 mm in (0.9-1.2 N on average, measured), is drawn back
across the line to 9 mm short of it and lifts while still moving back. The arm gives sideways under the drag, so each
stroke is aimed 2.8 mm short to cross the code where planned. Before each stroke the hand reads the card's pose and the
coin's lowest point and lean, and corrects the stroke and, in the air, its pitch. The demo reads object poses from the
simulator; a model under test sees only the cameras.

Measured on the GPU (the video's run and diagnostic copies of the same motion): the fingertip presses the hanger with 1.8-2.0 N; the hanger moves 3.7-4.4 mm in all, mostly as the fingertip lands, under 1 mm during the strokes; the rim presses 0.9-1.2 N on average per stroke (peaks about 2 N); each character is crossed 2-3 times and all 15 read. Dream-1 (nominal layout, 3,600 steps): 0.0.

## Simulation and the real objects

Rigid bodies, 240 Hz physics, 20 Hz control. Published: the piece's outline and thickness, the card's size and the
slot's length (maker's template), the quarter's size, mass and alloy (US Mint). Estimates, to be measured: the code's
character size and length, the slot's height (5.5 mm) and place, the piece's 15 g, every friction value, and the
scratch rule's 0.5 N, 6 N, three passes, one-fleck allowance and 3 x 1 mm footprint (calibration above). The card is
joined to its hanger along the tear line by uncut ties; here the two are one rigid body (the strokes pull on it with
well under 1 N). The real coating is a thin latex film that flakes off and the print can smear; here both are the
per-cell record. A real card has a fresh random code; the simulated one shows 7KQ2M-X94TR-C3H8P.

## Reset

A scratched panel cannot be put back: use a fresh hanging card each trial (cards come from the maker numbered, each with its own random code). Lay it back up on its outline and the quarter flat on its mark; tolerance 1 cm and 5 degrees.
