# Swipe a magnetic-stripe card through a desktop reader in one pass

Swipe the card through the reader once, then put it back in its stand.

A desktop magnetic-stripe reader only decodes a card when four physical conditions hold together: the stripe faces
the read head, the card's edge rides the slot floor so its track crosses the head gap, the whole encoded stripe
passes the head in one continuous stroke, and the card moves inside the reader's rated speed. The card is a true
0.76 mm ID-1 card and the reader is a true-size desktop swipe reader with a sprung head. The card starts standing in a
desk stand with its stripe facing away from the robot, a quarter turn from the reader's head side, so the robot must
also turn it before the swipe. Swipes are daily work for cashiers (O*NET 41-2011.00) and hotel desk clerks (O*NET 43-4081.00): gift and
prepaid cards, payment cards when the chip fails, hotel key cards, badges at time clocks.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=swipe_card_through_reader.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Swipe the card through the reader once, then put it back in its stand." \
  --scorer task_score --max-steps 1200 --epochs 2 --store-frames --log-dir ./demo-run
```

For a model trial, replace `--policy yam_scripted` with `--policy dream`. The output of `--scorer task_score` is the
score, taken at the end of the run.

Each trial allows 1,200 control steps (60 seconds at 20 Hz). There are no seeds: the layout is fixed and the card is
never jittered, so every trial starts identically. The scripted demo lasts 32.55 s; with the 3 s stop grace it ends
at about 711 steps. `success()` requires a score of 1.0, and the run ends after a further second at full score.

`pytest` in this folder runs the generated offline checks (no GPU): the scorer on the start, the goal and the named
layouts built from recorded 20 Hz swipe samples, and the demo's reach and length.

## The demo is open loop

The scripted demo uses only the known starting layout, time and the robot's own joint state. It never reads the card,
the head or the lamp. The right arm does the work; the left arm parks. Times are seconds from the start.

| Time | Right arm |
| --- | --- |
| 0.00-6.40 | Park, move over the stand, lower the open pads to either side of the card (yaw -90 degrees, pads closing along x) 26 mm from its centre toward its +y end, pinch it 30 mm above its bottom edge (pad tips) over 1.9 s, settle |
| 6.40-10.00 | Lift it out of the stand, carry it toward the reader's near end, on the line straight out from the right arm's base, and turn it a quarter (wrist yaw -90 to 0 degrees) so the stripe faces the head side, edge still down, the pinched end trailing and the leading end 8 mm short of the reader |
| 10.00-13.80 | Lower the card until its edge is 0.8 mm above the track floor's height, wait 0.6 s, turn the wrist 1 degree past its mark and back onto it |
| 13.80-15.80 | Slide the leading end 3 mm into the mouth, set the edge 0.6 mm into the floor, settle |
| 15.80-16.55 | One smoothstep stroke of 147.4 mm in 0.75 s straight out from the arm's base (+x), the edge commanded down to 1.0 mm into the floor: the head crosses the encoded data at 0.22 to 0.30 m/s commanded |
| 16.55-18.15 | Stop with the trailing end 15 mm past the head, lift the card straight up out of the slot |
| 18.15-24.45 | Carry it back, turn it back, lower it to 6 mm above the stand's lips, wait 0.8 s, lower it slowly into the slot to 1 mm above the floor, settle |
| 24.45-28.55 | Open over 1.9 s, rise clear, back off |
| 28.55-32.55 | Hold still |

The reader is turned so that its slot lies on the line straight out from the right arm's base, and the stroke runs
outward along that line. The arm's joints are soft and lag their targets; along this line the base and wrist yaw
joints do not move (commanded change under 0.001 degrees during the stroke), so the lag cannot swing the card's yaw,
and a rigid card wedges in the 1.06 mm slot once its yaw passes about 0.25 degrees. The card goes down beside the
reader's near end and enters the mouth level with the floor, as a hand does, because the arm strays up to about 4 mm
while it moves vertically. The pads hold the card 26 mm behind its centre, as a hand holds the trailing end: the
head, the back wall and the floor drag the card below the pads, and the 1.9 N pinch has little grip against turning,
so the card would tip nose-down once its leading end leaves the reader's far end; held this way the floor under its
edge still reaches about 40 mm ahead of the pads when the last data bit passes the head (14 mm if held at its
centre). The drag still slides the card about 4 mm back along the pads during the stroke; the stand's slot is 7.2 mm
longer than the card at each end, so it goes back in anyway. One constant compensates the wrist's steady yaw error from joint friction (0.38 degrees,
measured in simulation): from the turn until the carry back the wrist is commanded 0.38 degrees the other way, and
before entering the slot it is turned 1 degree past that mark and back, so it always comes to rest from the same
side. These are fixed numbers in the script, not feedback.

All demo targets are within 0.36 m (planar) of the right arm's base (worst 0.348 m, at the end of the stroke). The
wrist stays 27.1 degrees inside joint 6's limit, every per-step joint change stays inside the action guard (worst
0.50 of joint 6's cap, 0.62 at 0.8x speed), and the pad tips stay 6.5 mm above the reader's top while the card is in
the slot. Both fingers are clear of the card by 27.10 s; the plan allows a full score from 29.10 s, and the scripted
policy stops at about 711 steps. A copy with every demo duration scaled by 0.8 passes the same checks.

## Rebuild the setup

Station tabletop (the counter) z=0.7475 m, spanning x -0.19 to 0.91 m, y -0.65 to 0.65 m; x forward, y left; arm bases
(0,+/-0.305,0.75) m. Metres unless stated; the stand has yaw 0 and the reader yaw -90 degrees.

- **Card reader:** MagTek Dynamag MagneSafe magnetic-stripe swipe reader, part 21073075 (USB HID), modelled in
  black ABS plastic and fixed to the counter with fastening tape, as its manual allows. Origin (0.316,-0.30723,0.7475),
  yaw -90 degrees: the slot runs along world x on the line y = -0.305, straight out from the right arm's base, and
  the head side faces +y. In the reader's own frame (x across the slot, y along it): body 31.6 (x) by 99.5 (y) by
  30.4 (z) mm. The slot runs along y between x -2.76 and -1.7 mm (1.06 mm wide, inside the 0.864 to 1.067 mm of the
  maker's design kit); the floor is 7.76 mm above the counter (slot 22.64 mm deep), with 0.8 mm rounding on the slot's top edges. The slot is straight for
  |y| <= 34.75 mm, then four fixed 2 mm black plastic mouth walls flare it to 5.06 mm at both ends (y = +/-49.75 mm).
- **Read head:** steel cylinder, radius 4.5 mm, 12.3 mm tall (12.36 to 24.66 mm above the counter), axis at reader
  (-6.2, 0) mm, in a 10.4 mm window in the front (head-side, world +y) wall. It slides along x on a spring of 1,700 N/m
  preloaded 0.1 mm against its stop, travel 0 to -1.5 mm, damping 2 N s/m, mass 3 g. At rest its face is level with
  the slot's back wall, so a card must push it back by its own thickness; the head gap is 10.4 mm above the slot
  floor, on track 2's centreline.
- **Markings:** a 0.2 mm silver strip on the top of the front half marks the head side. Status LED, radius 2.8 mm, on
  the top at reader (8.0, 31.2) mm: green when ready, red for 2 s after a failed swipe.
- **Card stand:** fixed grey ABS plastic desk stand, origin (0.235,-0.410,0.7475): base 24 by 108 by 4 mm; a 1.6 mm
  slot, 100 mm long along y (7.2 mm longer than the card at each end), with straight walls up to 15 mm, then 3 mm radius rounded lips that open it to 7.6 mm at
  the 18 mm top.
- **Card:** ISO/IEC 7810 ID-1 bank-card size, rigid PVC plastic, 85.60 by 53.98 by 0.76 mm, 4.9 g, square corners,
  dynamic. It stands in the stand's slot, centre (0.235,-0.410,0.77859), long side along y, faces normal to x, 0.1 mm
  above the slot floor. Its black magnetic stripe (a visual band 5.54 to 15.95 mm from one long edge, the ISO/IEC 7811
  three-track area) is on the face toward +x (away from the robot), with that edge down; the blue printed front faces
  the robot.
- **Contact:** friction static/dynamic (estimates for dry plastics and a polished head, not measured): card
  0.40/0.35, stand 0.40/0.35, reader body and mouths 0.30/0.25, head 0.20/0.15. Contact offsets: card and stand 0.5 mm, reader and mouths 1.5 mm; rest offsets zero, restitution zero.
  The card has 32 position and 4 velocity solver iterations, 0.5 m/s maximum depenetration velocity and 0.05 linear
  and angular damping. The simulation steps at 240 Hz.

### Colliders

Reader-frame and stand-frame coordinates are in millimetres, measured from each origin (z up from the counter). The
reader frame is the world turned -90 degrees about z (reader +x is world -y, reader +y is world +x). Boxes give
centre and size; cylinders give centre, radius, length and axis.

| Object | Collider | Centre (mm) | Size (mm) |
| --- | --- | --- | --- |
| reader | base and slot floor | (0, 0, 3.88) | 31.6 x 99.5 x 7.76 |
| reader | front wall, -y of the head window | (-9.68, -19.83, 19.08) | 12.24 x 29.25 x 22.64 |
| reader | front wall lip, -y of the window | (-3.16, -19.83, 18.68) | 0.8 x 29.25 x 21.84 |
| reader | front wall, +y of the head window | (-9.68, 19.83, 19.08) | 12.24 x 29.25 x 22.64 |
| reader | front wall lip, +y of the window | (-3.16, 19.83, 18.68) | 0.8 x 29.25 x 21.84 |
| reader | below the head | (-9.28, 0, 9.86) | 13.04 x 10.4 x 4.2 |
| reader | above the head | (-9.68, 0, 27.73) | 12.24 x 10.4 x 5.34 |
| reader | above the head, lip | (-3.16, 0, 27.33) | 0.8 x 10.4 x 4.54 |
| reader | behind the head | (-14.20, 0, 18.51) | 3.2 x 10.4 x 13.1 |
| reader | front top-edge rounding | (-3.56, 0, 29.60) | cylinder r 0.8, 68.9 along y |
| reader | back wall | (7.45, 0, 19.08) | 16.7 x 68.9 x 22.64 |
| reader | back wall lip | (-1.30, 0, 18.68) | 0.8 x 68.9 x 21.84 |
| reader | back top-edge rounding | (-0.90, 0, 29.60) | cylinder r 0.8, 68.9 along y |
| reader | front end block, -y | (-11.38, -42.25, 19.08) | 8.84 x 15.0 x 22.64 |
| reader | back end block, -y | (9.15, -42.25, 19.08) | 13.3 x 15.0 x 22.64 |
| reader | front end block, +y | (-11.38, 42.25, 19.08) | 8.84 x 15.0 x 22.64 |
| reader | back end block, +y | (9.15, 42.25, 19.08) | 13.3 x 15.0 x 22.64 |
| reader | marking strip | (-8.50, 0, 30.50) | 7.0 x 60.0 x 0.2 |
| reader, head link | read head (slides along x) | (-6.20, 0, 18.51) | cylinder r 4.5, 12.3 along z |
| reader_mouth_front_left | flare wall, world origin (0.35812, -0.30248, 0.7475), yaw -82.405 deg | (0, 0, 19.08) | 2.0 x 15.13 x 22.64 |
| reader_mouth_back_left | flare wall, world origin (0.35812, -0.30752, 0.7475), yaw -97.595 deg | (0, 0, 19.08) | 2.0 x 15.13 x 22.64 |
| reader_mouth_front_right | flare wall, world origin (0.27388, -0.30248, 0.7475), yaw 82.405 deg | (0, 0, 19.08) | 2.0 x 15.13 x 22.64 |
| reader_mouth_back_right | flare wall, world origin (0.27388, -0.30752, 0.7475), yaw 97.595 deg | (0, 0, 19.08) | 2.0 x 15.13 x 22.64 |
| card_stand | base | (0, 0, 2.0) | 24.0 x 108.0 x 4.0 |
| card_stand | lower wall, -x | (-6.40, 0, 9.5) | 11.2 x 108.0 x 11.0 |
| card_stand | upper wall, -x | (-7.90, 0, 16.5) | 8.2 x 108.0 x 3.0 |
| card_stand | rounded lip, -x | (-3.80, 0, 15.0) | cylinder r 3.0, 108 along y |
| card_stand | lower wall, +x | (6.40, 0, 9.5) | 11.2 x 108.0 x 11.0 |
| card_stand | upper wall, +x | (7.90, 0, 16.5) | 8.2 x 108.0 x 3.0 |
| card_stand | rounded lip, +x | (3.80, 0, 15.0) | cylinder r 3.0, 108 along y |
| card_stand | slot end, -y | (0, -52.0, 11.0) | 7.6 x 4.0 x 14.0 |
| card_stand | slot end, +y | (0, 52.0, 11.0) | 7.6 x 4.0 x 14.0 |
| card | card (dynamic, 4.9 g) | world (0.235, -0.410, 0.77859) | 0.76 x 85.60 x 53.98 |

The card's stripe, its blue print and the white panel on the print, and the lamp are visual only (no collider, no
mass). The head link's cylinder is the only moving collider besides the card.

## Scoring

The scorer reads the card's simulated pose and velocity and the head's slide position at every 20 Hz control step.
It never reads contact forces, the lamp, or any flag set by the demo. In a real trial the same quantities come from
a side and a top video with a scale, and the reader's own output decides the read; the rules below are the reader's
documented conditions.

**A swipe.** The head counts as touched when it is pushed back at least 0.38 mm (half a card's thickness). A swipe is
a run of touched samples plus the untouched sample before it and after it. Swipes that never bring the head between
6.94 mm and 78.67 mm from a card end are ignored: no encoded data reached the head (ISO/IEC 7811-4 puts the first
data bit 7.44 +/- 0.50 mm from one end and the last at least 6.93 mm from the other).

**A good read** needs every condition, each visible in the video. The stripe side, edge and height conditions count
while the encoded data passes the head (a reader decodes nothing before the first data bit or after the last): at
every sample on the 6.94 to 78.67 mm span, the untouched samples just before and after the touched run included, and,
where two consecutive samples cross the 6.94 mm or 78.67 mm end of the span, at that end, interpolated linearly
between the two samples. So the card reaches and leaves the head along the slot, not over it: a card lifted out, or
dropped in, over the head mid-stripe is not a read, while a card that leaves or reaches the floor only beyond the
span reads whatever the 20 Hz sampling phase.

- **One continuous pass:** the samples are consecutive control steps; the head crosses the whole 6.94 to 78.67 mm
  span in one direction, with no reversal over 0.1 mm while the head is on that span. A stop and restart, or a push
  in and pull back across the data, fails; a small pull-back after the data has passed the head does not.
- **Speed:** the card's speed past the head is 0.1524 to 1.524 m/s (the reader's rated 6 to 60 in/s), both at every
  sample on the span, as the average between every pair of consecutive samples that are both on the span, and as the
  average between consecutive samples on opposite sides of the whole span. Motion past the span's ends is not
  checked: a card that slows to a stop on its blank last 7 mm after the last data bit still reads.
- **Stripe side:** the stripe face points toward the head (its normal within 60 degrees of
  the reader's front), and
- **stripe edge down** (within 60 degrees of vertical);
- **Height:** the stripe edge at the head is between 2.0 mm below and 1.0 mm above the slot floor (the 2.0 mm allows
  for simulated contact penetration). The head gap is on track 2's centreline 10.4 mm above the floor; with the card
  1.0 mm off the floor, the 1.5 mm read core already overhangs the 2.79 mm wide track by about 0.3 mm and is 0.2 mm
  from track 1. A card tilted in its own plane (one end up) is judged by the same rule at the head: riding one corner
  on the floor, it fails above about 0.7 degrees of tilt. There is no separate tilt rule, because a real head pivots
  to follow the card.

Either swipe direction reads, as the reader's manual says.

**The re-swipe rule.** A clean read is exactly one swipe, and it is good. A late read is a good swipe among two or
more: a good read after a failed swipe, or a second swipe after a good one. Real readers allow retries
indefinitely; each failed swipe shows a red light and costs the operator a retry, and a second good read would
enter the card twice. A nudge that touches the head without reaching the encoded data is not a swipe.

**A swipe still on the head at the end.** A swipe whose head contact has not ended by the end of the run is judged on
its samples so far (no sample after it is needed) if the head has already crossed the whole 6.94 to 78.67 mm span:
a card stopped with the head on its blank last 7 mm after one clean stroke is a clean read, left in the reader
(0.5). Otherwise it is a failed swipe: alone it gives no read (0); after a good swipe it makes the read late.

**Back in the stand.** Both ends of the card's bottom edge are inside the 1.6 by 100 mm slot (0.5 mm tolerance) and
within 2 mm of the slot floor, the long axis is within 10 degrees of the slot and the card's plane within 15 degrees
of vertical. Either face may point toward the robot. **Done** means back in the stand, both gripper TCPs at least
30 mm from the card's bounding box, and the card below 0.01 m/s and 0.5 rad/s, held for 1 s.

**Dropped (sticky for the episode).** The card leaves the counter (more than 20 mm below its surface, or outside its
x/y edges), or, with both gripper TCPs at least 30 mm from its bounding box (let go), its centre is less than 20 mm
above the counter or it lies within 45 degrees of flat anywhere. A card still held low or flat is not dropped. A 0.76 mm card lying flat cannot be picked up again with these fingers. Putting
it back afterwards does not clear the drop.

| Score | Condition at the end of the run |
| --- | --- |
| 0 | No good read, or the card was dropped |
| 0.25 | Late read; the card not done |
| 0.5 | Clean read, the card not done (still held, moving, left in the reader, or anywhere but the stand); or late read and done |
| 1.0 | Clean read and done |

**Not scored, and why.** The card cannot bend in this simulation (a rigid body), so a bent card is not represented
and not scored. The reader is fixed to the counter with fastening tape, so it cannot be knocked or moved and has no
score term.

**The lamp.** The reader's LED is driven by its own tracker of the same physical state (the card's pose and velocity
and the head's position): green while ready, red for 2 s after a swipe that fails the conditions above. The scorer
never reads it. The real Dynamag also goes dark during a swipe and shows amber at power-up; the lamp here has two
colours only.

## Reset

Stand the card back in the stand, stripe edge down, stripe away from the robot, at rest; the head returns to its stop.
Nothing is consumed. For a real trial use a test card with track 2 encoded, fix the reader with its fastening tape, and clear the
host's last read before each trial; the reader's LED and decoded data confirm each read.

## Fidelity

- **Card:** rigid box at true ID-1 size, thickness and mass (PVC, 1.39 g/cm3). Real cards flex; bending is not
  modelled or scored. ID-1 corners are rounded (3.18 mm radius); the collider's are square.
- **Read:** magnetic decoding is not simulated. The scorer applies the reader's documented conditions to the
  simulated card at the head: stripe face, edge height, a single pass and the rated speed.
- **Thin plate in a narrow slot:** a 0.76 mm card in a 1.06 mm slot at 240 Hz, with 32 solver iterations on the card
  and contact offsets of 0.5 mm (card) and 1.5 mm (reader). The contact offset is a detection distance only; the rest
  offset is zero, so the 1.06 mm gap is geometric. The card's sideways play in the slot is 0.15 mm per side.
- **Head:** a rigid steel cylinder on a sprung slide. With a card in, the spring force is 1.46 N, inside the 1.39 to
  2.23 N (5 to 8 oz) that MagTek's design kit gives; the 0.01 kg slide armature is a numerical stabiliser.
- **Mouths:** straight flat flares (the real ones are curved), 2.0 mm per side over 15 mm.
- **Reader mount:** fixed (fastening tape, as the manual allows); the tape is not modelled.
- **Pads:** the arm's finger pads carry the simulator's default friction (0.5 static and dynamic) rather than a
  rubber value, so the 1.9 N pinch lets the card slide a few millimetres under the swipe's drag; the card's own
  friction is not raised to compensate.
- **Friction:** the coefficients are estimates, not measurements. The read rules do not use them; they set how hard
  the card drags in the slot and how well the pinch holds it.
- **Stand:** a generic desk card stand (no datasheet), fixed to the counter.
- **Sampling:** the scorer samples at 20 Hz, so a pause shorter than 50 ms between two samples is judged by the
  average speed between them.

## Sources

- MagTek, *Dynamag Magnetic Stripe Card Reader, Technical Reference Manual*, part D99875482 (reader parts 21073062
  keyboard emulation and 21073075 USB HID): 99.5 x 31.6 x 30.4 mm,
  50 g; card speed 6 to 60 in/s; swiped "stripe down, through the slot either forward or backward"; the stripe faces
  the front of the reader; LED green on a good read, red for about 2 s on an error, retries indefinitely; mounted
  with fastening tape or screws; slot floor 0.305 in (7.76 mm) above the base (Figure 1-2).
- MagTek, *Magnetic Card Reader Design Kit*, part D99821002: head gap midway along the slot and on track 2's
  centreline 0.408 in (10.4 mm) from the card's edge; slot width 0.034 to 0.042 in (0.864 to 1.067 mm); at least
  1.75 in from the head to each slot end; the head rests against the far wall under a light spring; 5 to 8 oz of
  spring force with a card in.
- ISO/IEC 7810 (ID-1: 85.60 x 53.98 x 0.76 mm) and ISO/IEC 7811-2/-4 (track positions from the top edge: track 1
  5.66 to 8.46 mm, track 2 8.97 to 11.76 mm; stripe area to 15.95 mm with track 3; first data bit 7.44 +/- 0.50 mm
  from one end, last at least 6.93 mm from the other).
- Three-track magnetic read heads list a 1.5 mm read-core width per track.
- O*NET 41-2011.00 Cashiers, https://www.onetonline.org/link/summary/41-2011.00; O*NET 43-4081.00 Hotel, Motel, and
  Resort Desk Clerks, https://www.onetonline.org/link/summary/43-4081.00.

## Licence

All the code in this folder is the task authors' own and is released under the MIT licence. That includes the parts
kit, `yam_parts.py` (compound rigid parts) and `yam_joints.py` (passive jointed parts), which is copied unchanged
into this folder. Neither kit file carries a separate licence notice or contains third-party code. All geometry is
original primitives built with that kit. There are no third-party meshes, textures, images or other assets. Dimensions come from the public
datasheets and standards above; no text or figures from them are reproduced.
