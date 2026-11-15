# Fit a mop frame to its quick-connect handle: hold the swivelling socket square and push until the button clicks

Click the mop frame onto the handle: hold the frame's socket in line with the handle's tip and push the tip in until its spring button clicks out through the socket's window. Then let go of both and leave the mop on the bench.

Clicking a mop's head onto its handle is something most people who mop floors at home have done: many flat mops, spray mops and floor sweepers have a head that pushes onto the end of the handle and clicks, put on when the mop is new and again whenever a head is swapped. Here the head is a Rubbermaid HYGEN 17-inch flat mop frame and the handle its quick-connect handle; the maker's manual says to connect them "by clicking into place". The frame's socket swivels freely on a two-axis nylon joint, so one hand must hold it in line with the handle's tip while the other pushes the tip in: the button's ramp must be pushed past and the tip's lug must drop into the socket's notch before the button can spring out. With the socket left turned aside the tip misses its mouth or stops on its rim; a push that stops short leaves the button pressed inside, and the frame falls off at the first lift.

Sources: Rubbermaid Commercial Products, HYGEN Quick-Connect Frames (FGQ56000YL00: 16.00 x 3.00 x 1.50 in, aluminium,
nylon swivel joint), https://www.rubbermaidcommercial.com/cleaning/hygen-microfiber/hygen-quick-connect-frames/ ;
HYGEN Quick Connect Extension Handle (FGQ74500YL00, 20-40 in, aluminium),
https://www.rubbermaidcommercial.com/cleaning/hygen-microfiber/hygen-quick-connect-extension-handle-20-40/ ; HYGEN
Product Use Manual ("CONNECT the Quick-Connect Adjustable Handle to the Quick-Connect Frame by clicking into place"),
https://rcp.structpim.com/media/22100/hygen-product-use-manual.pdf ; HYGEN Microfiber Selection Guide, Quick-Connect
Handles and Frames (the close-up of the handle's tip going into the frame's socket).

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=fit_mop_frame_to_quick_connect_handle.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Click the mop frame onto the handle: hold the frame's socket in line with the handle's tip and push the tip in until its spring button clicks out through the socket's window. Then let go of both and leave the mop on the bench." \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial, replace `--policy yam_scripted` with `--policy dream`. `--scorer task_score` reports the score at
the end of the run. Each trial allows 2,400 control steps (120 seconds at 20 Hz). There are no
seeds: the layout is fixed (in our runs the scripted demo also scored 1.0 in three episodes with
`-E object_jitter=0.01`, the frame and the handle each moved up to 10 mm). The scripted demonstration takes
26.1 s, the last 3 s holding still. `success()` requires a score of 1.0.

`pytest` in this folder runs the generated offline checks (no GPU): the scorer on the start, the goal and the named
partial and shortcut layouts.

## The scene

Millimetres from the midpoint between the arm bases: x forward, y to the robot's left, heights above the bench top. A Rubbermaid HYGEN 17-inch Quick-Connect frame (FGQ56000YL00, 0.36 kg) lies flat on the bench on the robot's right, its middle at x 280, y -240, its long side along x: an aluminium plate 374 x 76 x 6 with black end caps (406 over the caps) and, at its middle, the nylon universal joint: a yellow swivel base on a pin along the frame 20 above the bench, carrying the black socket arm on a second pin 29 from it. The socket (34 across; bore 22.8 across and 80 deep with a 3 lead-in; a window 17 long and 7 wide in its side, 43.5 to 60.5 below the mouth; a notch 8 deep in the mouth on the window's side; mouth 124 from the arm's pin) lies folded flat toward the robot's left, window up, its axis 20 above the bench, turned 12 degrees toward the robot on the arm's pin, so its mouth is at x 254, y -90. A HYGEN Quick-Connect Short Extension Handle (FGQ74500YL00, collapsed to 640, 0.33 kg) lies along y on its ribbed locking ring and grip (both 40 across), at x 280, axis 20 above the bench: aluminium inner tube 22 across from its tip's end at y -42 to the ring at y 258, yellow outer tube 25 across, grip to its end at y 598. Near the tip, facing up: the spring button (16 long, 5 wide, 2.5 proud, ramped toward the tip; 13 to 29 from the end) and the locating lug (20 long, 10 wide, 3.5 proud; 65 to 85 from the end). Every collider is listed in README.md.

### Bodies, joints and colliders

Two free articulated bodies, each on the bench:

- **frame** (0.36 kg): the plate with its end caps and bearing blocks
  (0.3 kg); the yellow swivel base (`cradle`, 0.02 kg) on a pin along the frame's
  length 20 mm above the plate's underside; the black socket arm (`arm`, 0.04 kg) on a
  second pin 29 mm out from the first, square to it (the universal joint). Joint angles 0 are the start: the
  socket folded flat on the swivel base's stop, window up, turned 12 degrees on the arm's pin. The swivel
  base turns 180 degrees (folded one way, upright, folded the other); the arm turns 80 degrees either side of square.
  Both pins have light friction (0.004 N m) and no spring.
- **handle** (0.330 kg): the tube, locking ring, outer tube, grip and the tip's lug as one
  body, and the spring button (3 g) on a slide across the tube: 2.7 mm of
  travel, spring 1.3 N/mm preloaded to 2 N
  (5.5 N pressed flush). The ring and the grip have
  12 flats (ribbed), so the handle lies still on the bench.

Coordinates in each body's own frame at the start, millimetres (frame: plate underside at z 0, long axis x, the
socket folded toward -y; handle: the tip's end at 0, the axis along +x, button and lug on +z). The socket's walls are
convex pieces round the bore (listed by their extent):

| Body | Collider | Centre or extent | Size |
| --- | --- | --- | --- |
| frame plate and caps | box | 0, 0, 3 | 374 x 76 x 6 |
| frame plate and caps | box | 195, 0, 6 | 16 x 80 x 12 |
| frame plate and caps | box | -195, 0, 6 | 16 x 80 x 12 |
| frame plate and caps | box | 28, 0, 16 | 10 x 16 x 20 |
| frame plate and caps | box | -28, 0, 16 | 10 x 16 x 20 |
| frame: cradle | box | 0, -12, 20 | 44 x 30 x 10 |
| frame: cradle | cylinder along x | 0, 0, 20 | 14 across, 44 long |
| frame: arm | cylinder along z | 0, -29, 20 | 28 across, 16 long |
| frame: arm, socket wall `bar` | convex piece | x -10.7 to 20.1, y -70.6 to -31.4 | z 13 to 27 |
| frame: arm, socket wall `floor` | convex piece | x -8.6 to 26.1, y -75.6 to -64.5 | z 2.7 to 37.3 |
| frame: arm, socket wall `wall00` | convex piece | x -5.3 to 19.7, y -153.4 to -73.2 | z 25.9 to 35 |
| frame: arm, socket wall `wall01` | convex piece | x -7.5 to 15.2, y -153.8 to -74.2 | z 20 to 28.5 |
| frame: arm, socket wall `wall02` | convex piece | x -7.5 to 15.2, y -153.8 to -74.2 | z 11.5 to 20 |
| frame: arm, socket wall `wall03` | convex piece | x -5.3 to 19.4, y -153.4 to -73.3 | z 5.3 to 14.1 |
| frame: arm, socket wall `wall04` | convex piece | x 0.8 to 25.8, y -152.1 to -72 | z 3 to 9.8 |
| frame: arm, socket wall `wall05` | convex piece | x 9.1 to 34.1, y -150.3 to -70.3 | z 3 to 9.8 |
| frame: arm, socket wall `wall06` | convex piece | x 14.9 to 40.2, y -148.7 to -69 | z 5.3 to 14.1 |
| frame: arm, socket wall `wall07` | convex piece | x 19.1 to 42.4, y -147.6 to -68.5 | z 11.5 to 20 |
| frame: arm, socket wall `wall08` | convex piece | x 19.1 to 42.4, y -147.6 to -68.5 | z 20 to 28.5 |
| frame: arm, socket wall `wall09` | convex piece | x 14.6 to 40.2, y -148.8 to -69 | z 25.9 to 35 |
| frame: arm, socket wall `notch_side_a` | convex piece | x 12.6 to 31.9, y -141.7 to -70.4 | z 30.1 to 36.2 |
| frame: arm, socket wall `notch_side_b` | convex piece | x 1.3 to 20.7, y -144.1 to -72.8 | z 30.1 to 36.2 |
| frame: arm, socket wall `below_window` | convex piece | x 4 to 18.3, y -92.2 to -70.9 | z 31.4 to 36.2 |
| frame: arm, socket wall `above_window` | convex piece | x 11.6 to 29.3, y -143.6 to -106.6 | z 31.4 to 36.2 |
| handle | cylinder along x | 120, 0, 0 | 22 across, 240 long |
| handle | cylinder along x | 270, 0, 0 | 22 across, 60 long |
| handle | cylinder along x | 447.5, 0, 0 | 25 across, 197 long |
| handle | box | 75, 0, 11.5 | 20 x 10 x 6 |
| handle: button | convex piece (30-degree lead ramp) | x 13 to 29, y -2.5 to 2.5 | z 9 to 13.5 |

The socket's bore is 22.8 mm across (the tube is 22), 80 mm deep,
with a 3 mm 45-degree lead-in at the mouth. The window (17 mm long) is
43.5 to 60.5 mm below the mouth's top, between 72
and 108 degrees round the bore; the notch (8 mm deep) spans 62
to 118 degrees. With the tip 73 mm in, the lug bottoms in the notch and the button
lies under the window.

## Scoring

Judged at the end of the run from photos and a 1.5 mm feeler gauge. Clicked: the handle's tip is in the frame's socket with its lug down in the socket's notch (the 1.5 mm gauge does not pass under the lug) and its spring button standing out through the socket's window. Still: no point of the frame or the handle moves more than 1 mm in 0.5 s. 1.0 (success): clicked, both grippers at least 3 cm clear of the frame's socket and of the handle, and still, all held for 1 s. 0.5: clicked, but not yet let go and still for 1 s. 0.25: the tip at least 30 mm inside the socket but not clicked (the button still pressed inside), let go and still for 1 s. 0: anything else, including the tip stopped on the socket's mouth or beside it, or the frame or the handle off the bench at any time.

How each condition is read in `score()`:

- **Clicked:** the tip is at least 71.5 mm inside the socket's mouth (the lug within
  1.5 mm of the notch's floor), the handle's axis within 2 mm and 3 degrees of
  the socket's, and the button within 0.8 mm of fully out. At that depth the button lies under the window,
  so a button that is out is out through the window.
- **Let go:** both grippers' centre points at least 30 mm from the socket (its axis from hub to
  mouth, less its 17 mm radius) and from the handle (its axis, less 20 mm).
- **Still:** none of five points (the handle's two ends, the plate's two ends, the socket's mouth) moves more than
  1 mm over 0.5 s, held for 1.0 s.
- **0.25:** the tip at least 30 mm in, on the socket's axis, but not clicked; let go and still.
- **Zero for the rest of the run:** the frame or the handle leaving the bench.

## Why the shortcuts fail

Measured on the GPU with copies of this task whose demonstration was changed (scene and scorer unchanged):

- **Pushing the handle in without turning the socket square** (the right hand holds the socket's arm where it lies,
  the left pushes the handle along the frame's square line): the turned socket's mouth lies about 30 mm to the side of
  the handle's line, more than the tube's and the socket's radii together, so the tip slid past beside the socket. The
  run ended with the tip 29 mm off the socket's axis and the button never pressed: **0.0**.
- **A push that stops short** (the left hand stops 6 mm short of the click and the right hand does not press the
  socket toward the tip): the tip went in on the axis and the button rode in and stayed pressed inside the bore, while
  the frame slid 11 mm along the bench under the push. Let go with the tip 38 mm in, not clicked: **0.25**.
- **Clicked but still holding** either part, or the parts still moving: 0.5 (the demonstration passes through this
  for about 4 s, from the click until 1 s after both hands have let go).
- **Either part off the bench** at any time: 0 for the rest of the run.

The generated tests also score the tip resting on the mouth of a socket left turned aside (0), the handle at the
seated place beside a socket left turned aside (0) and the tip only 20 mm in (0).

## Scripted demonstration

The right hand pinches the socket's arm between the hub and the socket (fingers closing across its 24 mm); the left
hand pinches the handle's tube 270 mm behind the tip. The hands stay about 0.3 m apart. The demonstration
reads the frame's and the handle's poses seven times: at the start (both hands go to where the parts actually lie);
once both hands have pinched (how the socket's arm sits in the right hand); after the socket is turned square (the
socket's axis as it now lies, and how the handle sits in the left hand); with the tip 15 mm short of the
mouth; twice with it 6 mm short; and with the tip 12 mm into the mouth. At each read it
works out how the part sits in the hand and how far the arm sags from its command, and moves so the part lands where the
plan has it (the soft wrist lets the handle turn a little in the pinch while its far end drags on the bench). For the
last push the left hand is commanded 30 mm past the seated depth and the right hand presses the socket
12 mm toward the tip: the arms' sag under load sets the push force (about 10 N at the left hand while
the button rides its ramp, by the joint-controller error; there is no force sensor). Both pushes are eased off to the
seated place before the hands open.

| Time (s) | Step |
| --- | --- |
| 0.0 to 2.5 | left: over the handle's tube |
| 0.0 to 2.5 | right: over the socket's arm |
| 2.5 to 4.0 | left: pads down beside the tube |
| 2.5 to 4.0 | right: pads down beside the arm |
| 4.0 to 5.0 | left: pinch the tube |
| 4.0 to 5.0 | right: pinch the arm |
| 5.0 to 5.5 | left: settle |
| 5.0 to 5.5 | right: settle |
| 5.5 to 7.5 | left: hold |
| 5.5 to 7.5 | right: turn the socket square |
| 7.5 to 8.1 | left: hold |
| 7.5 to 8.1 | right: hold the socket square |
| 8.1 to 10.1 | left: tip onto the socket's axis |
| 8.1 to 12.8 | right: hold the socket square |
| 10.1 to 10.6 | left: hold |
| 10.6 to 11.4 | left: closer, corrected to the axis |
| 11.4 to 11.9 | left: hold |
| 11.9 to 12.4 | left: corrected again |
| 12.4 to 12.8 | left: hold |
| 12.8 to 14.0 | left: tip into the mouth |
| 12.8 to 14.3 | right: hold the socket square |
| 14.0 to 14.3 | left: hold |
| 14.3 to 16.8 | left: push home until it clicks |
| 14.3 to 16.8 | right: press the socket onto the tip |
| 16.8 to 18.0 | left: hold home |
| 16.8 to 18.0 | right: hold |
| 18.0 to 18.6 | left: ease the push off |
| 18.0 to 18.6 | right: ease the push off |
| 18.6 to 19.6 | left: let go |
| 18.6 to 19.6 | right: let go |
| 19.6 to 21.1 | left: rise clear |
| 19.6 to 21.1 | right: rise clear |
| 21.1 to 23.1 | both hands to their parking points |
| 23.1 to 26.1 | both hands hold still at their parking points |

## Simulation and the real part

Rigid bodies, 240 Hz physics. Shortcuts: no mop pad; the handle is one body, its ring and grip with 12 flats so it lies still; the button is a sprung slide; the socket is convex pieces; the demo reads poses. Estimates (no maker figure): bore 22.8 x 80 mm, 3 mm lead-in, window 17 x 7, notch 8 deep, button 16 x 5 x 2.5 (2.7 travel, 1.3 N/mm, 2 N preload), lug 20 x 10 x 3.5, arm 124, pins 20 and 29, swivel ranges, masses (0.36, 0.33 kg), frictions. README lists all.

### What is the maker's and what is estimated

From the maker: the frame FGQ56000YL00 is 16.00 x 3.00 x 1.50 in (406 x 76 x 38 mm), an aluminium plate with a nylon
universal joint; a case of six weighs 4.80 lb, so a frame with its share of the packaging is at most 0.37 kg. The
handle FGQ74500YL00 is 25.20 x 1.57 x 1.57 in collapsed (640 x 40 x 40 mm) and 0.73 lb (0.33 kg) packed; the
Quick-Connect straight handle's tube is 0.87 in (22.1 mm) with the same tip. The maker's close-up photo shows the
handle's tip (spring button near the end, lug behind it) and the socket (notch in the mouth, window in the side).

Estimated (no maker figure; sizes scaled from the close-up by the 22 mm tube):

- the socket: bore 22.8 mm across (0.4 mm over the tube a side),
  80 mm deep, 3 mm 45-degree lead-in, 34 mm outside, its
  mouth 124 mm from the arm's pin; the window 17 mm long and about 7 mm
  wide; the notch 8 mm deep and 56 degrees wide;
- the universal joint: the swivel base's pin 20 mm above the plate's underside, the arm's pin
  29 mm from it; 180 degrees of swivel and 80 degrees either side of square; pin friction
  0.004 N m; the arm 24 x 14 mm where the right hand pinches it;
- the handle's tip: the button 16 x 5 mm, 2.5 mm proud with a
  30-degree ramp, 13 to 29 mm from the end, 2.7 mm of travel; its
  spring 1.3 N/mm preloaded to 2 N, from beam theory on a single-leg snap
  button's spring leaf (0.018 x 0.365 in, the size a snap-button maker lists for 0.45 to 0.87 in tubes; no force is
  published); the lug 20 x 10 mm, 3.5 mm proud, 65 to
  85 mm from the end;
- masses: the frame 0.36 kg (plate and caps 0.3, swivel base
  0.02, socket arm 0.04), the handle 0.33 kg (its
  packed weight), the button 3 g;
- friction (static, dynamic): the frame and its nylon parts 0.3, 0.25; the handle
  0.45, 0.4; the button 0.2, 0.15;
- the start: the socket folded toward the handle and turned 12 degrees aside.

Simplified in the simulation: no microfibre pad on the frame; the handle's tubes, ring, grip and lug are one rigid body
(its length lock is not modelled); the ring and the grip have 12 flats so the handle lies still instead of
rolling; the button is a block on a sprung slide, not a bent spring leaf; the socket's round walls are convex pieces
with flat inner faces touching the round bore; the demonstration reads the parts' poses from the simulator.

To measure on the real parts: the button's force over its travel and its travel, the bore, window, notch and lug sizes
and places, the socket's reach from its pins, the pins' friction and the masses.

## Reset

Press the button in through the window and pull the frame off. Lay the frame flat at x 280, y -240, long side along x; fold its socket flat toward the robot's left, window up, and turn it 12 degrees toward the robot. Lay the handle along y on its ring and grip, tip end at y -42, button and lug up. Nothing is consumed.
