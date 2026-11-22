# Float a magnetic levitating desk globe: lift it under the frame's magnet to its balance point and let go

**Instruction:** "Pick the globe up off the frame's base, lift it under the magnet in the frame's top arm to the point where the magnet takes its weight, let go, and take your hand away so the globe hangs in mid-air touching nothing."

A C-shaped magnetic levitating globe frame stands loose on the desk, plugged in, its open side toward the robot. Its
3-inch world globe rests on the frame's base under the head (the electromagnet in the end of the top arm). The robot
picks the globe up, brings it up under the head to the balance point where the magnet takes its weight, opens its
fingers without disturbing it and takes its hand away, so the globe hangs in mid-air touching nothing.

## Scoring

Score 0 if the frame was knocked over (tilted over 30 degrees) or the globe left the desk. Otherwise the lower of two numbers. Globe: 1.0 it has hung free for 5 s in a row: below the head, its top 3-30 mm under the head's face and within 15 mm of the head's axis, clear of the base, every part of the gripper 10 mm or more off the globe, and the globe still (under 5 cm/s); 0.25 stuck up against the magnet (gap under 2 mm) with the hand away; 0 otherwise (on the base, dropped, or still in the hand; includes doing nothing). Frame: 1.0 within 15 mm of its start; 0.5 moved 15-40 mm; 0.25 further. Success: 1.0 held 1 s more. In the real trial: the video (gap and clearance, timed) and a ruler on the frame's tape outline.

| Outcome | Score |
|---|---|
| Globe hangs free 5 s, frame within 15 mm of its outline | 1.0 |
| Globe hangs free 5 s, frame shoved 15-40 mm | 0.5 |
| Globe hangs free 5 s, frame shoved over 40 mm | 0.25 |
| Globe stuck up against the magnet, hand away (let go too high) | 0.25 |
| Globe dropped onto the base or desk, still in the hand, or nothing done | 0 |
| Frame knocked over (over 30 degrees), or globe off the desk | 0 |

Why letting go in the wrong place scores low: let go with the globe's top under about 10 mm from the head, the magnet
alone outpulls the globe's weight and snaps it up against the head (0.25: it is on the magnet but not floating); let
go more than about 20 mm below, even full coil current cannot hold it and it drops onto the base (0, as at the start).

**Real trial, read from the video and a ruler.** The globe hanging free is plain on the video: a visible gap above it
(under the head) and below it (over the base), no finger on it; time 5 s from the moment the hand is 10 mm or more
away. Stuck: no gap at the head. Frame: a masking-tape outline of its base and a ruler. Practical points: the frame's
12 V adapter is plugged in and its light on; keep phones and steel tools 30 cm away (the makers' advice); the pads
pinch a smooth plastic sphere, nothing sharp or messy. Reset is seconds (set the globe back on the base).

## The product and its numbers

- Senders Floating Globe with LED Lights, C shape (Amazon ASIN B0191DLE9M): "1pc C shape frame, 1pc 3inches globe,
  1pc 12v 100ma adapter", about US$30
  (https://www.amazon.com/Senders-Floating-Magnetic-Levitation-Decoration/dp/B0191DLE9M). The same C frame sold by
  other sellers is listed at 18 x 4.8 x 17.3 cm with a floating distance of 10-20 mm
  (https://www.wayfair.com/decor-pillows/pdp/orren-ellis-3-magnetic-levitation-world-globe-c-shape-colorful-desk-decor-blue-in-gold-w006964017.html,
  https://www.amazon.com/Floating-Magnetic-Levitation-Spinning-Decoration/dp/B0BJDYLFFJ). Whole item 0.32-0.41 kg.
  Some sellers' "3-inch" globes are 85 mm (https://www.youbeli.com/inch-shape-magnetic-levitation-floating-globe-p-5702890.html):
  check the globe with a ruler; the 94 mm finger opening also clears an 85 mm globe by 4.5 mm a side.
- Placing instructions from the C-frame listings (for example
  https://www.amazon.com/Gresus-Magnetic-Levitation-Floating-Boyfriend/dp/B085HSXPJW): "Hold the globe with your
  hand, keep globe north pole upwards ... When the globe is in the proper position, you'll feel a strong force
  holding the globe. This is the floating distance. Then take out the hand below the globe gently ... After you
  release the globe, The globe will levitate." Without the supplied stick: "pull the globe vertically down about 1.8
  to 2 cm from the top. When you feel the pulling force becomes lighter and more balanced, that is the balance
  point." The globe "floats closer to the top". The stick is optional ("you can still use one finger"); this task
  uses none.
- How it works (a published teardown of a commercial levitating globe,
  https://www.embeddedrelated.com/showarticle/526.php and https://www.embeddedrelated.com/showarticle/527.php): a
  rare-earth magnet at the globe's north pole; a Hall sensor and an iron-cored electromagnet in the head; the coil
  only aids the magnet, so the globe "always hangs below the unpowered equilibrium point", about half an inch below
  the head; close enough to the head it "jumps up to the head and sticks there"; released near the attraction limit
  the control overshoots up to the head; with the power off it falls.

Estimates (no published figure; how to check one in a minute): globe 70 g (kitchen scale); float gap 15 mm and a hold
window of 10-20 mm, from the sellers' 1.0-2.0 cm instructions and floating distance and the teardown's half inch
(ruler beside the floating globe; let go slightly higher or lower); frame proportions (base and arms 18 mm thick,
upright 25 mm, head a 40 mm disc with its face 143 mm above the desk) from photos (ruler); base 18 x 8.4 cm from a
seller's "C shape base size: 7'' x 3.3''".

## The simulated levitation (levitator_kit.py)

No proximity rule and no snapping: every physics step (240 Hz) the globe's magnet point (its north pole) gets a
force toward a point 10 mm above the head's face, with g the gap from the head's face to the globe's top and W the
globe's weight:

- magnet on the head's iron core: W x (20 mm / (g + 10 mm))^3, equal to W at g = 10 mm;
- coil: u x 1.216 W x (25 mm / (g + 10 mm))^3, with the controller u = clamp(0.401 + 123/m x (g - 15 mm) + 1.49 s/m
  x dg/dt, 0, 1) (one-way drive, saturating); at full current it just holds the globe at 20 mm;
- off the axis the pull points toward the head (field centring), fading out between 12 and 20 mm off the axis;
  damping 0.5 N s/m on sideways motion and 0.0016 N m s/rad on the globe's wobble (eddy-current and air losses).
  The frame gets the equal and opposite force at its head.

Released at rest with g between 10 and 20 mm the globe settles at 15 mm (net stiffness about 61 N/m, damping ratio
about 0.3); offline, released still at 9.5 mm it snaps up, at 20.5 mm it drops, and at the float point it is thrown
out by 0.2 m/s up or down or 0.35 m/s sideways. The globe hangs upright because the pull acts at its pole.

## Running a trial

```
inspect-robots run --embodiment yam_arena -E task=float_a_levitating_globe_under_its_magnet.py \
    --instruction "Pick the globe up off the frame's base, lift it under the magnet in the frame's top arm to the point where the magnet takes its weight, let go, and take your hand away so the globe hangs in mid-air touching nothing." \
    --policy yam_scripted --scorer task_score --max-steps 1200
```

- Scene `station`, 20 Hz, 1,200 steps. `--scorer task_score`: 1.0 on success, otherwise the score at the end. The
  demo reads where the globe and the frame are, so seeded shifts are fine.
- Offline checks: `pytest` in this folder (scorer on the goal, partial and cheat layouts; demo reach and length).

## The scripted demo

The right hand comes in from the robot's right, pitched 45 degrees down, pads open either side of the globe (its palm
stays outside the frame's 48 mm band and below the top arm), pinches the globe across its equator (each pad 2.5 mm
in), lifts it straight up under the head to the balance point (globe centre 90 mm above the desk, its top 15 mm
under the head), reads where the globe really is twice and corrects for the arm's tracking and sag, as a person
feels for the balance point, then opens its fingers, waits 0.5 s, slides the open hand out sideways and moves away.
The left arm stays parked; the frame is not held. The plan takes 21.4 s. On the GPU (no seeded shift) the demo scored 1.0: the globe was let go 15 mm under the head and hung free from 11.9 s, success at step 335 (16.8 s); the frame moved 0.1 mm. In three runs with seeded shifts of up to 10 mm for the frame and the globe, each scored 1.0 (globe hanging at a 15.0 mm gap, under 1.1 mm off the axis; frame moved at most 0.2 mm).

Wrong-way runs: (1) let go 10 mm too low (a 26 mm gap): the globe dropped onto the base and stayed there: 0. (2) pushed up onto the magnet: as the globe neared the head the magnet's pull and the hand's push tipped the light frame over, the globe ended stuck to the toppled head: 0 (frame knocked over). In a probe, holding it 9 mm too high also dragged the frame away: 0.

Scores: Dream-1 run: 0.0; demo: 1.0.

## Simplifications

Electronics, Hall sensor and coil heating are not simulated: the controller is the force law above. The frame and
globe are each one rigid body (the frame's proportions are estimates); the continents are drawn only, not a map; the
frame's LEDs and the globe's slow spin are not modelled.
