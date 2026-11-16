# Push a straw through the X-slot in a cold cup's lid, forcing its four flaps open, down to the bottom of the drink

Put the straw into the drink through the slot in the cup's lid, push it down to the bottom of the cup, and let go of it.

## Scoring

Judged from the end of the run with a ruler and a photo through the clear cup. Through: the straw passes through the lid at the slot (its axis crosses the lid within 6 mm of the slot's centre) and its lower end is at least 10 mm below the lid's top. At depth: through, and its lower end within 20 mm of the cup's inside bottom (equivalently, its top stands no more than 95 mm above the lid). Let go: both grippers' centre points at least 3 cm from the straw and from the cup. Still: over the last 1 s each end of the straw and the cup moved less than 1 mm, and the straw's axis and the cup turned less than 2 degrees (a straw turning about its own axis counts as still). Scores: 1.0 at depth, let go and still, held for 1 s; 0.5 at depth but still held or moving, or through and let go and still but not at depth; 0.25 through but still held or moving; 0 for anything else: the straw not through the slot (on the counter, on the lid, in a hand), or the cup tipped more than 15 degrees at any moment, or more than 30 mm from its start mark at the end, or (real trials) its lid off the rim anywhere at the end or taken off at any time. Possible scores: 0, 0.25, 0.5, 1.0.

How `score()` reads each condition (cup frame: origin at the centre of the cup's base, z up its axis):

- **Through:** the straw's lower end is at least 10 mm below the deck's top (124.7 mm), inside the
  cup, and the straw's axis crosses the deck's top plane within 6 mm of the slot's centre. The deck is solid
  everywhere else, so a straw that far in has gone through the slot.
- **At depth:** the lower end is at most 20 mm above the inside bottom (3 mm above the counter).
- **Let go:** both grippers' centre points at least 30 mm outside the straw's and the cup's bounding boxes.
- **Still:** over the last 1 s the straw's two ends and the cup each moved under 1 mm, the straw's
  axis and the cup turned under 2 degrees, and the cup is under 1 cm/s now (the straw's solver velocity is not
  used: wedged in the flaps it jitters in place).
- **Zero:** the cup's axis tilted more than 15 degrees at any moment (remembered for the rest of the
  run), or the cup's base more than 30 mm from its start at the end.

`success()` is a score of 1.0, which itself needs the full-credit state held for 1 s.

Why the shortcuts fail: the lid has no opening but the slot, so a straw can only reach the drink through the flaps.
Our wrong-way run is the scripted demonstration aimed 12 mm off the slot's centre: the straw's end sat on the solid
deck, the flaps never moved, the push (about 11 N, estimated from the arm) slid the straw in the fingers, and at the
re-grip it fell onto the counter. Score 0; the cup did not move. A straw stood on the counter beside the cup, or laid
across the lid, is not through: 0. Shoving or tipping the cup instead of pushing straight scores 0.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=push_straw_through_cup_lid_slot.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Put the straw into the drink through the slot in the cup's lid, push it down to the bottom of the cup, and let go of it." \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial use `--policy dream`. The score is read at the end of the run; each trial allows 2,400
control steps (120 s at 20 Hz). No seeds: the layout is fixed. The scripted demonstration takes
36.3 s. `pytest` here runs the offline checks of the scorer on the start, the goal and named layouts.

## The scene

The products, their sizes and the slot's model are in task.json's `setup`. Positions (mm, from the midpoint between the
arm bases; x forward, y to the robot's left; heights above the counter): the cup's base centred at
(240, -60); the straw lying on the counter along x, its middle at (230, -190). Both arms
start at their ready pose with the grippers open; nothing else is on the counter.

### Bodies, hinges and masses

- **Cup and lid (one free body, 510 g with the drink):** convex pieces (mm): base disc
  (0 to 3 up); 20 wall pieces 1 thick, outer radius 30.9 at the base to 48.2 at
  118.2; rolled lip to radius 49.2, top 120.7; lid skirt radius 49.5-50.6 from 116.6; raised ring
  radius 44.5-50.6, 122.7-125.7; four deck pieces (123.9-124.7) round a 11.2 mm square that
  the flaps fill. The drink's mass is spread over these pieces (no liquid body): centre of mass about 67 mm up, a real
  full cup's about 60.
- **Flaps (four links on hinges):** triangular plates 0.8 mm thick, tips at the centre (the cuts drawn
  0.5 mm wide), 0.2 g each, armature 3e-06 kg m^2. Each hinge: limits
  +-85 degrees, spring 0.004 N m/rad toward flat, joint friction (the yield) 0.0024 N m static and
  0.0018 N m moving, damping 0.0001 N m s/rad.

| Flap | Hinge axis | Hinge line through (mm, cup frame) | Bends down at angles |
| --- | --- | --- | --- |
| flap_px | y | (5.6, 0, 124.3) | negative |
| flap_nx | y | (-5.6, 0, 124.3) | positive |
| flap_py | x | (0, 5.6, 124.3) | positive |
| flap_ny | x | (0, -5.6, 124.3) | negative |

- **Straw (free body):** one cylinder 197 mm long, radius 3 mm, 3 g (a real one
  0.45 g; see below), friction
  0.5/0.4. Cup friction 0.4/0.35.

The push, quasi-statically: the straw's edge meets each flap 2.6 mm from its hinge, so the flaps
need 4 x 0.0024 / 0.00262 = 3.7 N to start bending, about 5.2 N at the peak as their springs load, until they stand 62
degrees down and the straw slides past their tips.

## Scripted demonstration

One hand does the work, as a person does with a cup on a counter; the left arm parks. The demonstration reads (it
never moves an object) the cup's position once, before carrying the straw over (every pose over and in the cup
follows it), and the straw's pose: where it lies, after it is turned upright, over the lid, on the flaps, after each
push step, before the re-grip and at the end of the push. Each read moves the hand so the straw is where the plan has
it (in the push steps sideways only, at most 4 mm in all: the arm gives under the push and the straw can pivot in
the thin pinch, so the hand leads the straw down and re-aims it every 0.25 s).

| Time (s) | Step |
| --- | --- |
| 0.0 to 2.0 | over the straw |
| 2.0 to 3.5 | fingers down beside the straw |
| 3.5 to 4.7 | pinch the straw lightly |
| 4.7 to 5.7 | lift it |
| 5.7 to 8.2 | turn it upright |
| 8.2 to 8.8 | squeeze it |
| 8.8 to 11.4 | carry it over the lid, hold |
| 11.4 to 13.1 | lower toward the slot, hold |
| 13.1 to 13.9 | end on the flaps |
| 13.9 to 19.4 | push through the flaps (20 steps of 0.25 s, re-aimed sideways after each), hold |
| 19.4 to 20.0 | open the fingers |
| 20.0 to 22.0 | slide the fingers up the straw |
| 22.0 to 22.8 | pinch higher up |
| 22.8 to 26.8 | push it down to the bottom (7 steps of 0.5 s, the straw stood upright about the slot after each), hold |
| 26.8 to 28.3 | push the last of the way, hold |
| 28.3 to 29.1 | let go |
| 29.1 to 30.3 | back away |
| 30.3 to 31.3 | rise clear |
| 31.3 to 33.3 | the hand to its parking point; then still |

The first pinch is 90 mm from the end that goes in, the straw 10.5 mm past the TCP (the pad
tips 0.7 mm above the counter at the pick), light (pads aimed 4 mm apart, about 5 N) until the straw
is upright, then full; the second pinch is 80 mm higher up the straw and
5.8 mm deeper between the pads. With the fingers open the straw leans in the slot; in the second push each
read finds where the straw crosses the lid and moves the hand sideways (at most 25 mm in all) until the pinch
is straight above that point. The push ends with the straw's end 8 mm above the inside bottom; let go, the
straw settles onto the bottom, leaning in the flaps. In our recorded demonstration (no object jitter) full credit
came at 32.0 s; the straw stood 14 degrees from upright with its end on the bottom; the cup did not move.

## Simulation and the real objects

Rigid bodies. The lid is part of the cup (snapped on; nothing pulls it). Slot flaps: four hinged plates with a yield moment and a spring (estimates: about 5 N to push through), 0.2 g each with a numerical armature. Deck 0.8 mm, wall 1 mm thick (real about 0.25). The drink is a fixed mass: no liquid drag. The straw is rigid (a real one can kink) and weighs 3 g, not 0.45 g: lighter free parts are unstable in a pinch. Clear plastic renders opaque. Frictions are estimates.

- From the makers: the cup's 98.4 / 61.9 / 120.7 mm, the lid's 101.2 x 9.1 mm (Dart Solo TP16D and 626TS sheets as
  listed by WebstaurantStore and Lakeshore), the straw's 197 x 6 mm (WebstaurantStore 485UJ7BK).
- Estimates, to measure on the real parts: the slot's cut length (15.9 mm, "about 5/8 in", from the lid maker's patent
  US 5,147,065), the push needed (about 5 N) and the flaps' spring-back, the deck's and wall's thickness, the inside
  bottom's height (3 mm), the masses (cup 15 g and lid 3 g, under their case weights per piece; the real straw's 0.45 g), the
  lid's profile inside its 101.2 x 9.1 mm outline, and every friction.
- The flaps here are rigid plates on hinges; a real lid's flaps also bend and may tear at a cut's end.

## Reset

Use a new lid each trial (the bent flaps do not spring fully flat again) and a new straw if it is kinked; top the cup back up to 473 ml, snap the lid on all round and stand the cup on its mark; lay the straw on its mark along x. Zero object jitter.
