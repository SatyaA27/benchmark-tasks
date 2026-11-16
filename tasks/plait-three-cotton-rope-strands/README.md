# Plait three cotton rope strands from a clamp: six crossings, an outer strand over the middle each time, that stay put

Plait the three rope strands from the clamp: lay an outer strand over the middle one, then the other outer strand over the new middle one, and keep alternating sides until there are six crossings. Then let go of the rope.

A three-strand plait is the braid most people learn first: in their own or a child's hair, in friendship bracelets and
lanyards, in plaited bread, and in the rope tug toys people make for their dogs from three lengths of cotton rope. It
is done the same way every time: an outer strand over the middle one, then the other outer strand over the new middle,
and repeat. It is hard for a robot because the strands are soft: each must be pinched near its free end, carried over
the middle strand without dragging the others along, and laid down straight on the far side so the crossing forms close
to the last one. Laid slack, a strand bows near the plait, the crossings drift outward and the strands run out before
the sixth; a crossing left with a short or springy tail slides off when the hand lets go.

## Scoring

Judged from a top photo, which may be taken whenever both grippers are at least 3 cm from the rope and no part of the
rope moves faster than 1 cm/s (on a real trial, a top-video frame with both hands 3 cm clear where the frame 1 s later
shows no rope point moved 1 cm); the last such photo of the run counts (none: 0). Read the crossings from the clamp
outward, tracing each strand by its colour: at a crossing the upper strand lies over (hides) the lower one. A crossing
counts when an outer strand lies over the middle strand and the sides alternate (right over middle, left over middle,
and so on, starting on either side). Counting stops at the first crossing that breaks the pattern: the same side twice in a row, a strand passing
under the middle one, or a strand laid across both others (only its first crossing counts). A strand laid over and
straight back is no crossing. Score = counted crossings / 6, at most 1.0 (six or more). If a rope end has moved in the
clamp, the score is 0. The run ends early with success once photos have shown six crossings for 5 seconds in a row.

| Last photo with the hands clear | Score |
|---|---|
| Six or more alternating outer-over-middle crossings from the clamp | 1.0 |
| Five / four / three / two / one, then the pattern breaks or the strands end | 0.83 / 0.67 / 0.5 / 0.33 / 0.17 |
| First crossing already wrong (a strand under the middle), or no crossing | 0 |
| No photo all run (a hand never 3 cm clear, or the rope never still) | 0 |

How the simulation takes the photo (`score()` and `success()` in `plait_three_cotton_rope_strands.py`): each strand's
centre-line is read from its 30 links. A crossing is a point where two centre-lines cross seen from above with the
upper one at least 4 mm higher (a 10 mm strand lying over another sits about 10 mm higher; less is two strands pressed
side by side). Crossings are read in order of distance from the clamp; the strands' places (right, middle, left) start
as they leave the clamp and swap at each crossing, so "outer over middle" and "same side twice" follow from the places,
as when reading a photo. Two crossings within 15 mm of each other along the plait may be read in either order; a pair
of neighbours is read first. The hands are clear when no rope point is within 3 cm of either gripper's envelope (a
cylinder of 50 mm radius, the open fingers' reach, from the pad tips 120 mm up the hand). The score kept while a hand
is on the rope is the last photo's, so holding a crossing in place earns nothing until the hand lets go and it stays.

Why the shortcuts fail: one hand laying its outer strand over the middle again and again makes a twist, not a plait;
its second crossing is on the same side, so it scores 1/6 (the scripted twist, six crossings by the right hand, scored
0.17 in the simulation). Passing strands under the middle (an inside-out braid) scores 0 from the first
crossing. Holding the strands in a plait shape never yields a photo. The offline tests (`pytest`) check these layouts
and partial counts with synthetic centre-lines.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=plait_three_cotton_rope_strands.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Plait the three rope strands from the clamp: lay an outer strand over the middle one, then the other outer strand over the new middle one, and keep alternating sides until there are six crossings. Then let go of the rope." \
  --scorer task_score --max-steps 3600 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial, replace `--policy yam_scripted` with `--policy dream`. Each trial allows 3,600 control steps
(180 seconds at 20 Hz). The layout is fixed (the strands are anchored, so object jitter does not move them).
The scripted demonstration scores 1.0 on this layout: each crossing shows in the next photo
(1/6 at 7.4 s, then one more every 10.2 s), six from 58.4 s, and the run ends with success at step 1,267 (63.4 s)
of the 3,600 allowed.

`pytest` in this folder runs the offline checks (no GPU) on synthetic centre-lines: the start, full plaits (either
side first), partial plaits, the shortcuts above, a hand still on the rope and a moving rope.

## The scene

Millimetres from the midpoint between the arm bases: x forward, y to the robot's left. Three 450 mm lengths of 10 mm
(3/8 in) three-strand twisted cotton rope, 70 g/m (32 g each; red, natural and blue), lie flat on the bare wooden
table. Their ends lie side by side, centres 11 mm apart (red on the robot's right, natural, blue), the last 15 mm of
each under a steel clamp bar (34 x 60 x 9 mm, its centre at x 552, y 0) whose back is fixed on a hardwood block
(26 x 60 x 21 mm, centre x 582) screwed to the table; the bar is tightened so a 50 N pull does not move an end. From the
clamp's front edge (x 535) the strands run straight toward the robot in a fan: natural straight along -x to its end at
x 100, red 20 degrees to the robot's right to its end near (127, -165), blue 20 degrees to the left to its end near
(127, 165). Nothing else is on the table.

The clamp stands in for the knot or hook that holds the start of a plait (a rope toy is started with an overhand knot
of the three ends; hair is held at the scalp). It is 550 mm out, beyond where the hands work: the crossings must form
near it by how each strand is laid from its end, not by reaching in.

## Scripted demonstration

Two hands, one crossing per cycle (10.2 s): the right hand lays the right-hand strand over the middle, the left hand the
left-hand strand, alternately, six times; the other hand waits parked. In each cycle the hand pinches the outer strand
22.5 mm from its free end (pads about 9.4 mm apart on the 10 mm rope), lifts it so the chord from the strand's root
(its furthest crossing, or the clamp) to the hand is 5 mm short of the strand's length there, swings it on that arc
over the middle strand to 65% of the way from the middle strand's end toward the far strand's, lowers it along the arc
so it lands straight, pulls it 1.5 mm past its length (a light tension of at most a few newtons), and lets go. Then
it pinches the old middle strand, now outer, near its end and slides it out to 30 degrees about its previous root,
which draws the new crossing toward the clamp and opens room for the next one. After the sixth crossing both hands
park. The demo reads the strands' link poses (simulator state) to pick grasp points and roots; a model under test sees
only the cameras.

Measured over four probe runs of this plan (three with each strand's starting line bent by up to 2.5 degrees at two
points, moving its end up to 16 mm; one exact, with a firmer 3.5 mm pull): six crossings in all four, right-left-right-
left-right-left from the clamp, unchanged 5 s after both hands left (crossings moved at most 0.1 mm). Crossings lay
62-72, 120-135, 179-191, 237-246, 284-290 and 319-327 mm from the clamp; the strand laid for the sixth still had
160-172 mm free. Every pinch held (slip at most 3.7 mm), every crossing was a clean over (the upper strand 9.6-10.6 mm
higher), and each crossing dragged the other strands 1-23 mm. An earlier plan that lowered each strand with 8 mm of
slack bowed it near its root, pushed the crossings 60-80 mm outward and held six in only one run of three.

## Simulation and the real objects

Each strand is an articulation of 30 capsule links (15 mm long, 10 mm across, 1.05 g each) joined by joints with
stretch and twist locked and two bending axes free to +-60 degrees, each on a spring toward straight (bending
stiffness EI 1.0e-4 N m^2, an estimate: a 10 cm overhang of the rope droops about 35 degrees) with 10% damping and a
small joint armature (5e-6 kg m^2) for solver stability. Strands collide with each other, the table and the grippers
(cotton friction 0.8 static and dynamic, an estimate). The first link of each strand is fixed to the world under the
clamp bar, so an end cannot slip in the simulation; the real clamp must be tight enough (50 N) for that to hold. The
rope's own three-ply twist and its flattening under the pads are not modelled. Measured or published: the rope's
diameter, length and mass per metre (sellers' figures for 10 mm twisted cotton: about 70 g/m). Physics at 240 Hz,
control at 20 Hz.

Files: `plait_three_cotton_rope_strands.py` (scene, scorer, demo), `rope_strand.py` (builds each strand as the
capsule-chain articulation and reports its links' poses), `plait_word.py` (reads the crossings and counts the plait;
also builds the synthetic plaits the tests use), `test_plait_three_cotton_rope_strands.py` (offline checks).
