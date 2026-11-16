# Clip a car air freshener onto a dashboard vent vane: fork over one pivoting vane, pushed home along it, vent aim kept

Clip the air freshener onto one of the centre vent's horizontal vanes, pushed all the way on, without changing where the vent is aimed.

## Scoring

Judged at the end of the run (real trial: the end of the video, then a ruler, a protractor and a 100 g weight). Any of the four vanes counts. Engaged: the vane lies between the two prongs (real: seen from the side, the vane between both prongs) with at least 8 mm of the 20 mm prongs over it (the vane's front edge 12 mm or less from the fork's root). Home: the vane's front edge within 2 mm of the root. Let go: both grippers' tool points at least 3 cm from the clip and from the vent opening. Then, once it has been engaged, let go and still for 1 s (three points on the clip moved under 1 mm and no vane turned more than 1 degree over that second), a hold check: the clip is pulled straight down with 1 N for 2 s (real: a 100 g weight hung from a thread looped round the clip's body, lowered by hand without touching the clip); it must move less than 2 mm. Aim: the largest angle any vane is turned from level at the end (protractor; the vanes keep any angle they are pushed to). Square: the angle between the fork and the dashboard face's normal (real: a protractor against the clip's back face, from above and from the side). Score 0: not engaged (in a gap, on a vane's face, on a prong, dropped), still touched, hold check failed, or the aim off by more than 15 degrees (the vent knocked). Otherwise aim is ok up to 5 degrees and minor from 5 to 15; square is ok up to 15 degrees, minor from 15 to 30, major beyond. Home with aim and square ok: 1.0; home with one minor: 0.75; home with two minors: 0.5; engaged but not home (root gap 2-12 mm) with both ok: 0.5; not home with a minor: 0.25; square major (home or not): 0.25. Doing nothing scores 0. Possible scores: 0, 0.25, 0.5, 0.75, 1.0.

How `score()` reads each condition (clip frame: origin at the body's centre, x along the fork, z across the gap):

- **Engaged:** for any vane, the point of its front edge nearest the fork, taken into the clip's frame, lies between the
  prongs: the vane's centre plane where a 2 mm vane fits between their inner faces (1.6 mm apart plus
  however far the sprung prong has opened), within 0.8 mm; 12 mm or less in front of the fork's root (at least
  8 mm of the 20 mm prongs over the vane), and across the prongs' width inside
  the vane's length.
- **Home:** that edge at most 2 mm from the root.
- **Let go:** both grippers' tool points at least 30 mm outside the clip's bounding box and the vent
  opening's box.
- **Still:** over the last 1 s three points on the clip moved under 1 mm and no vane turned more
  than 1 degree (positions, not solver velocities), held for a further 1 s.
- **Hold check:** then a 1 N force straight down on the clip's body (ramped over 0.5 s, held
  2 s, applied by the scorer to the free body); the clip must move under 2 mm. A clip resting
  loosely on a vane slides off; one gripped by the sprung prong stays.
- **Aim:** the largest angle of any vane from level at the end. **Square:** the angle between
  the fork's axis and the dashboard face's normal; turning about the fork itself is not counted.

`success()` is a score of 1.0.

Why the shortcuts fail: the vanes keep any angle they are pushed to (joint friction, no spring back), so a fork landed
on a vane's face, or pushed on pitched to the vane, turns the vane: past 15 degrees the vent is knocked off
its aim and the score is 0. Our wrong-way run is the scripted demonstration with the fork pitched
30 degrees nose-down to the vane and pushed along its own axis: the fork bound on the vane 5 mm on; the push (about 9 N in, 7 N down) forced the vane's front edge down 9 degrees, and turned the clip in the fingers. Let go, the clip hung tilted on the vane's edge 15 mm from its root, not engaged: score 0, with the vent left aimed 9 degrees down. A clip
pushed into the gap between two vanes, laid on top of the vent, or left in the hand scores 0. Doing nothing scores 0.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=clip_car_air_freshener_onto_vent_vane.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Clip the air freshener onto one of the centre vent's horizontal vanes, pushed all the way on, without changing where the vent is aimed." \
  --scorer task_score --max-steps 1200 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial use `--policy dream`. The score is read at the end of the run; each trial allows 1,200
control steps (60 s at 20 Hz). The scripted demonstration takes 23.2 s. `pytest` here checks the scorer
on named layouts.

## The scene

Products, sizes and positions are in task.json's `setup` (mm from the midpoint between the arm bases; x forward, y to
the robot's left; heights above the table). The dashboard, its vent and the mat are fixed; the clip is the one free
body. Both arms start at their ready pose with the grippers open.

- **Dashboard and vent (fixed, the vanes on hinges):** dashboard 600 x 150 x
  190 mm, face at x = 240; opening 110 x 70 mm centred at
  (240, -120, 130), duct 45 mm deep. Vanes 108 x 17 x
  2 mm, 3 g, centre lines at heights 107.5, 122.5, 137.5, 152.5; each on a hinge about y
  8.5 mm behind its front edge, limits +-30 degrees, joint friction 0.03 N m
  static and 0.025 N m moving (the detent), damping 0.0002 N m s/rad, armature
  2e-06 kg m^2 (numerical). Seven vertical vanes 30 mm behind the face, fixed.
- **Clip (free, 7 g):** body a rounded-rectangle prism 41 x 38 x 16 mm;
  lower prong fixed to it; upper prong (0.3 g) on a slide across the gap, 0 to 0.8 mm
  open, spring 1250 N/m with 1.0 N preload (1.5 N
  spread over a 2 mm vane), damping 5.0 N s/m, armature 0.02 kg (numerical). Prongs 20 mm long, 7 mm wide, 1.4 mm thick at the root,
  1.6 mm apart, each tip turned out 1.45 mm over its last 3 mm (mouth
  4.5 mm); each prong is one convex piece, its outer face running straight from the root to the
  tip, so its inner face has no seam for a vane's corner to catch on. Friction: clip 0.5/0.45, ABS
  0.4/0.35, mat 0.8/0.7.

## Scripted demonstration

The right hand does the work; the left arm stays at its ready pose. The hand points at the dashboard pitched
30 degrees down and pinches the clip's two flat sides; the clip sits in the pinch turned the same
30 degrees, so its fork is level. The demonstration reads (it never moves an object) where the
standing clip is before the pick, and the held clip's pose in front of the vane (twice, each read moving the hand so
the clip is where the plan has it: at most 15 mm / 10 degrees, then
8 mm / 5 degrees) and after each push step (height and pitch only,
at most 0.8 mm / 1.5 degrees a step, 3 mm /
4 degrees in all): the 4.5 mm mouth must meet the 2 mm vane within
about a millimetre, and the arm sags a little under load. The pinch is firm (pads commanded 23 mm apart on the
41 mm body: about 45 N, within the arena's 50 N limit for the YAM gripper); held at about 10 N, the clip turned in the
fingers when the fork met the vane.

| Time (s) | Step |
| --- | --- |
| 0.0 to 1.5 | out in front of the dashboard (from the ready pose, so the hand clears its top) |
| 1.5 to 3.0 | over the clip |
| 3.0 to 4.2 | down behind the clip |
| 4.2 to 5.0 | pinch its sides |
| 5.0 to 5.3 | settle |
| 5.3 to 6.3 | lift it |
| 6.3 to 8.8 | carry it in front of the vane |
| 8.8 to 9.4 | hold in front of the vane |
| 9.4 to 10.0 | line up with the vane |
| 10.0 to 12.8 | push along the vane (8 steps of 0.35 s, height and pitch re-read after each) |
| 12.8 to 13.4 | hold it home |
| 13.4 to 14.2 | let go |
| 14.2 to 15.2 | back away |
| 15.2 to 16.2 | rise clear |
| 16.2 to 18.2 | the hand to its parking point |
| 18.2 to 23.2 | hold still (the run stops at full credit) |

The push starts with the fork's tips 8 mm short of the third vane's edge and goes 5 mm past
home along the vane's plane; the vane's edge stops at the fork's root and the arm's compliance takes the rest. The pads stay
4 mm behind the clip's back, out of the vent. In our recorded
demonstration (no object jitter): full credit at 18.5 s, the vane's edge at the fork's root (0.0 mm), every vane within 0.01 degrees of level, the sprung prong 0.38 mm open on the 2 mm vane. In three runs of the same scene with the clip's start jittered by up to 10 mm, all three scored 1.0; the hold check moved the clip 0.01 mm.

## Simulation and the real objects

True-size rigid parts. Each vane has its own pins and its own detent; the rear link that turns a car's vanes together is left out, in the simulation and in the built vent. The fork's sprung prong is a plate on a sprung slide (1.5 N on a vane), with a numerical armature; its small barbs are left out. The oil is mass only. The clip starts clicked open, standing on its flat edge.

- The vent: no car maker or parts seller we found publishes a centre vent's vane thickness, spacing and depth
  (universal in-dash louvres list only bezel and cut-out sizes), so the vent is built to the simulated sizes and
  detent (Building the vent). Vent-mount makers publish what real louvres must be for their hooks: blades under
  3.5-4.2 mm thick, 10-34 mm deep, at least 8 mm apart; our vanes are 2.0 thick, 17 deep and 13 apart.
- The clip, looked up: the Febreze CAR Vent Clip holds 2 mL of scented oil; the single pack with card weighs 13.6 g
  (7 g used). The maker publishes no clip sizes. Its patent for this kind of vent-clip dispenser gives prongs 10 to 40 mm long (narrowest stated range 16 to
  20), gaps 0.5 to 8 mm at their widest (3 to 5) and 0.5 to 3 mm at their narrowest (1 to 2). Our 20 mm prongs,
  4.5 mm mouth and 1.6 mm gap lie inside the narrowest ranges; they stay estimates for the sold clip (checked: The
  clip for real trials).
- Estimates: the clip's body size, spreading force (1.5 N) and mass; every friction.

## Building the vent

Sizes in mm, taken from the task's module. Register coordinates: from the opening's centre on the face, x back into
the dashboard, y along the vanes, z up. Parts (US prices looked up November 2026, about $110 in all):

- Black ABS filament, 1.75 mm, 1 kg (Bambu Lab, $18.99; about 150 g used): https://us.store.bambulab.com/products/abs-filament
- 3 x 16 mm stainless dowel pins, 80 (Walmart, $10.99; 8 used): https://www.walmart.com/ip/16651518561
- Nitrile O-rings 3 mm ID x 1.5 mm (6 mm OD), 8 (Totally Seals, $1.00 each): https://www.totallyseals.com/en-us/products/3mm-x-1-5mm-nitrile-o-rings
- Set-screw shaft collars, 3.05 mm bore, 8 mm OD, 4 mm wide, 10 (Walmart, $10.49; 8 used): https://www.walmart.com/ip/5275673485
- M3 x 12 mm flat-head machine screws, 4 (Everbilt 837751, Home Depot, $1.95): https://www.homedepot.com/p/321071710
- 1/2 in. (12 mm) MDF panel, 2 x 4 ft (ProWood 109097, Home Depot, $24.99 to $28.00): https://www.homedepot.com/p/202093821;
  dark grey spray paint (Rust-Oleum Painter's Touch 2X Satin Charcoal Gray, Home Depot, $6.98):
  https://www.homedepot.com/b/Paint-Spray-Paint/Rust-Oleum-Painters-Touch-2X/Gray/N-5yc1vZapz5Zbm2Z1z1a3o5
- 6 in. digital caliper (Pittsburgh 63711, Harbor Freight, $24.99): https://www.harborfreight.com/6-in-digital-caliper-63711.html
- For the checks: thread, a small bag, 80 US nickels (5.000 g each; or a kitchen scale) and a dime.

Prints (black ABS, solid round the pins):

1. Vanes, 4: blade 108 (y) x 17 (x) x 2.0 (z), front edge at x = 0; at each end a hub 6 across and 5 long (|y| 54
   to 59) on the pin axis (x = 8.5, mid-thickness) with a 3.0 hole, 5 deep. Print standing on the front edge,
   supports under the hubs. Caliper check: 2.0 +-0.1 thick, 17.0 +-0.2 deep, 108.0 +-0.3 long.
2. Rear half, x 8.5 to 47: a block 140 (y) x 90 (z) with the 110 x 70 duct through it to a back wall at x 45 to 47;
   seven fins 1.5 thick at y = -45, -30, -15, 0, 15, 30, 45, from x = 26 back to the back wall, joining the duct's
   top and bottom (the simulated fins end at x = 34; nothing reaches past 26).
3. Front plate, x 0 to 8.5: 140 x 90 with the 110 x 70 opening, square corners; four countersunk 3.4 holes at
   y = +-62, z = +-38, over 2.5 holes in the rear half.

Both halves carry half-bores on their joint face (x = 8.5) for each pin (z = -22.5, -7.5, 7.5, 22.5, both sides): 8
across for |y| 55 to 60 (the hub's pocket), 3.2 for 60 to 63, 10 for 63 to 70 (O-ring and collar); in the rear half
the 10 bore runs back to x = 47 as a slot, so the collar's set screw can be reached from behind.

Assembly and the detent:

1. Press a pin to the bottom of each hub's hole (a drop of super glue if loose); its outer end is then flush with
   the register's side (|y| = 70). Thread an O-ring, then a collar, onto each pin.
2. Lay the vanes in the rear half's half-bores and screw the front plate on.
3. With a vane level, push each collar inward to squeeze its O-ring against the wall and tighten its set screw;
   both ends alike, so the vane does not twist.
4. Check: hang the bag from two thread loops round the vane's front edge, 10 mm in from each end (8.5 mm in front of
   the pins). 330 g (66 nickels, 0.0275 N m) must leave the vane level for 10 s; 390 g (78 nickels, 0.0325 N m) must
   turn it. The simulated 0.030 N m is 360 g there. Adjust the collars until both hold; recheck before each
   session. The 100 g hold check puts about 0.017 N m on a vane, under the lower bound.
5. Box from the MDF: front 600 x 190, top 600 x 138, sides 138 x 178, glued and screwed, open at back and bottom,
   painted dark grey with the register's front. Cut a 140 x 90 hole in the front centred on the vent's centre
   (y = -120, 130 above the table), push the register in from behind, fronts flush, and hot-glue it. Clamp the box
   to the table, face at x = 240, ends at y = -300 and 300.

A built vane weighs about 6 g with its pins (3 g simulated); its weight sits on the pin axis, so it adds no turning
torque. The simulated +-30 degree stops are not built: a vane past 15 degrees already scores 0.

## The clip for real trials

Before the first trial, check the bought clip (clicked open). Gap between the prongs at the root: simulated 1.6,
accepted 1.35 to 1.95: a US dime (1.35 thick) slides in to the root without force, a nickel (1.95) goes in only by
spreading the sprung prong, as the 2.0 vane must. With the caliper, the mouth at the tips: 4.5, accepted 3.5 to 5.5;
prong length from the clip's back: 20, accepted 18 to 22 (the fixed fins stand 26 behind the face). If the clip
has four prongs (the patent's preferred form), check the pair that takes a horizontal vane. A clip outside these
bands is not the simulated fork: use another. Then push it onto a vane by hand: it must pass the 100 g hold check.

Scented oil (review note): clicking the clip open pierces the foil over the oil and the maker's FAQ covers oil
leaking onto surfaces, so real trials use a used-up clip, not a fresh one: clicked open, dial on high, left until the
scent is gone (the maker says up to 30 days on low), then squeezed hard by its flat sides in a paper towel: no oil
mark. It is about 2 g lighter (2 mL of oil); the simulated 7 g is an estimate either way.

Sources: maker's FAQ, https://circle.supersavvyme.co.uk/uk/projects/febreze/faq; patent US 12,708,692 B2,
https://patents.google.com/patent/US12708692B2/en; a universal louvre listing (sizes of bezel and cut-out only),
https://watsons-streetworks.com/product/trique-mfg-rectangular-a-c-vent/; coin mass and thickness,
https://en.wikipedia.org/wiki/Nickel_(United_States_coin), https://en.wikipedia.org/wiki/Dime_(United_States_coin).

## Reset

Pull the clip straight off its vane along the vane's plane, stand it on its flat bottom edge on its mark on the mat (within 1 cm), fork toward the dashboard, and set all four vanes level (within 1 degree). The same clip is reused; the vent is never removed. Zero object jitter.

## Revision

Review (not accepted): "Please specify a compatible, readily obtainable car vent, or provide build files and a parts
list for the four-vane fixture, including how to reproduce its pivot friction. Check the freshener's fork and the
selected vent's dimensions and forces against the real parts, and update the setup and simulation where needed so the
real trial exercises the same insertion." Note: protect the shared rig from scented oil.

Changes, text only: Building the vent (no obtainable car vent publishes its vane sizes); the fork checked against
the maker's patent ranges and measured on the bought clip before the first trial; a used-up clip for real trials;
setup and fidelity to match. The built vent matches the simulation: code, demonstration and runs are unchanged.
