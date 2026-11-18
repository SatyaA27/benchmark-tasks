# Spin a Twister spinner with one flick: hold the light card still and send the arrow round at least two full turns

Hold the spinner's card still with one hand and flick the arrow once with the other so it spins at least two full turns.

## Scoring

Judged from the overhead trial video (30 frames/s or more, so 0.2 s is 6 frames and 0.5 s is 15) and a ruler against the card's pencil outline. A spell of contact is a hand touching the arrow; touches under 0.2 s apart are one spell, and a spell that moves the arrow is a push (a strike is a short push). Turns are counted on the card from the first push until the arrow stops or is touched again. Apply the first rule that fits. 0: the card's centre ends more than 5 cm from its start or the card has turned more than 30 degrees; or a push lasted more than 0.5 s (the arrow walked or swept round in contact); or under a quarter turn was counted (doing nothing scores 0). 0.25: the arrow was pushed more than once, or under one full turn was counted. 1.0: at least two full turns, then it coasted to rest with nothing touching it, and the card's centre is within 2 cm of its start. 0.75: at least two full turns, but the card moved 2 to 5 cm, or something touched the arrow before it stopped, or it was still turning at the end. 0.5: one to two full turns.

How `score()` reads each condition, once per 50 ms control step (the rules above are applied in the same order):

- **Turns** are the arrow's hinge angle on the card, unwrapped, from the start of the first push until the arrow
  is at rest on the card (under 0.05 rad/s, 0.2 s after the last contact) or a new spell of contact begins.
- **Contact, from the arrow's own spin** (the card's spin about its axis plus the hinge speed, so the card's own
  wobble cancels out). Coasting, the free arrow always loses speed at the rivet's rate (4.5 to 4.7 rad/s²). A step
  in which it turns faster than 0.2 rad/s and loses speed at under 2 rad/s², or gains speed, was **pushed**; a step
  in which an arrow faster than 1 rad/s loses speed at over 12 rad/s², or reverses, was **braked**. Pushed or
  braked steps under 0.2 s apart form one spell; a spell with a pushed step is a **push**, and its length runs from
  its first contact step to its last. The finger's follow-through after a strike stays inside the strike's spell.
- **Card:** its centre's distance from where it lay at the first step, and its turn about the vertical.
- `success()` is a score of 1.0, which needs the arrow at rest; the trial ends 1 s later.

Why the shortcuts fail: the rivet's friction is honest (below), so a slow push only walks the arrow round in
contact, a push longer than 0.5 s (0); pushing it round and letting go at speed is the same (0); two strikes score
0.25; leaving the card unheld lets the strike shove it away (the card is a free 90 g body).
Our wrong-way run walks the arrow round with the closed fingers on its short arm instead of striking it:
the fingers, 60 out on the short arm, pushed it clockwise for 2.6 s; it turned a third of a
turn with the hand on it, one push spell 2.6 s long, and stopped when the hand lifted: 0 (pushed round in contact).

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=spin_twister_spinner_with_one_flick.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Hold the spinner's card still with one hand and flick the arrow once with the other so it spins at least two full turns." \
  --scorer task_score --max-steps 600 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial use `--policy dream`. Each trial allows 600 control steps (30 s at 20 Hz). `pytest` here checks
the scorer on named layouts.

## The scene

Sizes in mm, positions from the midpoint between the arm bases (x forward, y to the robot's left), heights above
the table top. Both arms start at their ready pose. `spinner_kit.py` builds the spinner (reusable).

- **Card (free body, 90 g):** a disc 250 across and 2 thick, lying flat with its centre at (160, 0); friction
  0.45 static / 0.35 sliding (estimate), contact offset 0.5 mm. A black rivet post 8 across stands 6 high at its
  centre.
- **Arrow (8 g, on a hinge at the rivet):** a bar 185 x 14 x 4, its middle 12.5 from the rivet toward the pointing
  end (so 80 to the short end, 105 to the pointing end), its underside 4 above the card. Inertia about the rivet
  2.42e-5 kg m². Hinge friction 1.09e-4 N m sliding (1.20e-4 N m to start it), no limit, no spring, no drive. At the
  start the pointing end points to +y (left) and the short end toward the right arm.
- Nothing else is on the table.

## The scripted demonstration

The left hand closes and presses the card down at (120, 117), 124 from the rivet: on the card, outside the
arrow's 105 sweep. The right hand, fingers closed, comes down 70 to the right of the rivet, with the
fingertips 5 above the table and 60 behind the arrow's short arm, then makes one stroke of 150 along +x in 0.17 s (the
fingers reach about 0.75 m/s and cross the short arm 70 out from the rivet) and lifts 47 out of the arrow's path at
once, before the pointing end comes round. The demo reaches full score: the arrow makes 2.56 turns from the single strike and
coasts to rest about 2.5 s later, the card's centre moved 2 mm; full score at 6.6 s, and the trial ends 1 s later.

## Sources and estimates

- Product: Hasbro Twister, item 98831 (the spinner is included, the maker's care page:
  https://consumercare.hasbro.com/en-au/product/twister/C71921FA-5056-900B-10D3-25880E1E5284); about 20 USD
  (https://www.fleetfarm.com/detail/hasbro-twister-game/0000000082979). Box 10.5 x 10.5 x 1.6 in
  (https://wpn.wizards.com/en/products/twister), which the 250 mm card fits.
- Round white card and black arrow: product photo https://m.media-amazon.com/images/I/81uRaMSGS4L._AC_SL1500_.jpg.
  The card's and arrow's sizes and masses, the rivet's place and the arrow's height are **estimates** from these
  photos; a ruler and a kitchen scale check them in a minute.
- The rivet's friction: the retail video linked from https://manuals.plus/asin/B008J87PVC shows a person flicking
  this black arrow. Tracking the arrow's axis frame by frame (29.97 frames/s) from release to rest gives 1.88 turns
  in 2.44 s from about 10.3 rad/s, a steady slowing of 4.46 rad/s² (fit residual 0.09 rad); we use 4.5 rad/s².
  This is an **estimate** from video (release and rest are uncertain by about two frames; about 3.5 to 5.5 rad/s²),
  not a bench measurement. To check on a real spinner: film a flick from above with a phone's slow-motion mode and
  count the turns and seconds from release to rest. In the simulator the free arrow slows at 4.7 rad/s² (its own
  small damping added), never less than the estimate.
- Card-on-table friction (0.45 / 0.35) is an estimate.

## Simulation shortcuts

The arrow's taper (about 20 to 9 wide) is drawn as a straight 14 bar of the same length and mass; the card's
printed colours are not drawn (it is plain white). All bodies are rigid. Nothing pushes or turns the arrow but the
robot's fingers.
