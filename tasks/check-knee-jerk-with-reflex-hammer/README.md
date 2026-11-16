# Test the knee jerk: swing a reflex hammer so it taps the tendon just below the kneecap briskly and lifts straight off

Test the knee-jerk reflex on the leg trainer: take the reflex hammer from its cup, tap the tendon just below the kneecap so the leg kicks, then stand the hammer back in its cup.

Having a doctor tap just below your kneecap with a rubber hammer, and watching your lower leg kick, is one of the most familiar moments of a check-up; physicians, nurse practitioners, physical therapists and nurses doing neurological checks use it to test the L2-L4 nerve roots. Clinical guides: hold the Taylor hammer at the end of its handle and swing it in an arc onto the tendon (maker's directions), "in a quick striking motion by the wrist", with "the broad end" on the patellar tendon between the kneecap and the tibial tubercle, knee partly flexed and supported. The reflex needs a brisk tap: in a 100-person study the knee's kick grew with the hammer's speed and was near zero for slow taps. So the robot must swing the head at about 1 m/s or more, land it within a band about 30 mm tall below the kneecap, not on the kneecap or the bone below, and let it come straight off rather than push.

Sources: ADC (American Diagnostic Corporation) neurological hammers instructions for use, 93-Neuro-00 Rev 5 (the
Taylor hammer 3693 "is typically held at the end and swung in an arc-like motion onto the tendon"); Ooi L-W et al.,
"Deep Tendon Reflex: The Tools and Techniques", Malays J Med Sci 2021;28(2):48-62 (supine knee jerk with the knee and
hip partly flexed and supported; "tap the tendon with the broad end of the Taylor hammer"; the tendon lies between the
patella and the tibial tuberosity); Ernstmeyer and Christman (eds.), Nursing Skills (OpenRN, 2022), 6.9 Assessing
Reflexes ("use a reflex hammer in a quick striking motion by the wrist"); Tham LK et al., "Motion Analysis of Normal
Patellar Tendon Reflex", Can J Neurol Sci 2013;40:836-841 (100 adults; the knee's kick grows linearly with the
hammer's tapping speed: a mean of 0.2 deg of knee extension when the hammer was dropped from 15 deg, and about 2, 5
and 8 deg from 30, 45 and 60 deg; latency 0.10-0.13 s).

## Revision: review feedback and what changed

Review: "No source or build instructions are provided for the life-size hinged leg trainer with the specified knee
landmarks. ... Please identify an affordable, obtainable leg trainer that provides the specified hinged knee and
anatomical landmarks, or provide drawings and assembly instructions for making one. Include enough dimensions and
materials to reproduce the strike surface and leg motion for repeated real trials." Changes, text only: the new section
"Building the leg trainer" gives the trainer's drawing in words (materials with sources and prices, about $150 in all;
its outside sizes and lower-leg mass taken from the module; assembly; checks for the landmarks, the 2.0 kg lower leg and
its free swing), and the setup says the trainer is built to it. The fidelity notes now list the trainer's sizes and
lower-leg mass as built to the module rather than estimated. The scene, the reflex rule, the scorer, the demo, the run
limit and the recorded runs are unchanged.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=check_knee_jerk_with_reflex_hammer.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Test the knee-jerk reflex on the leg trainer: take the reflex hammer from its cup, tap the tendon just below the kneecap so the leg kicks, then stand the hammer back in its cup." \
  --scorer task_score --max-steps 1200 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial, replace `--policy yam_scripted` with `--policy dream`. `--scorer task_score` reports the score at
the end of the run. Each trial allows 1,200 control steps (60 seconds at 20 Hz). There are no
seeds: the layout is fixed. The scripted demonstration takes 22.4 s. `success()` requires a score of 1.0.

`pytest` in this folder runs the generated offline checks (no GPU): the scorer on the start, the goal and the named
partial and shortcut layouts (the reflex model's readouts are given as fixtures there; it runs only in the simulator).

## The scene

Left and right are the robot's. Positions are in metres from the midpoint between the two arm bases: x forward, y to the robot's left; heights are above the bench top. Leg trainer: a life-size adult right-leg trainer cut at mid-thigh, built from PVC tube, foam rings and wood to the drawing in the README (Building the leg trainer; about $150 of materials), lying on its back on the bench with its foot away to the robot's front right (the leg's line points 60 degrees right of straight ahead). Its thigh stub is clamped to the bench and rests on an 80 mm foam roll (250 mm long, fixed) 90 mm above the knee; the knee's hinge is at (0.338, -0.062), 150 mm up; the knee is bent about 18 degrees (the thigh rises 8 degrees toward the knee, the shin falls 10 degrees to the heel) and the heel rests on the bench. The lower leg and foot (2 kg) swing freely on the knee's hinge, lightly damped, and lie on the heel under their own weight. Kneecap: 50 mm across, its front 62 mm above the knee's axis. Patellar tendon: a strip 28 mm wide running 45 mm down the shin from the kneecap's lower pole (12 mm below the knee's axis) to the tibial tubercle, a bump 6 mm proud of it. The strike band is the middle of the tendon: 17 to 47 mm below the knee's axis along the shin and within 15 mm of the tendon's midline; its centre is at (0.358, -0.096), 189 mm up. Reflex hammer: an ADC Taylor percussion hammer, 190 mm long and about 70 g: a flat chrome handle 12 mm wide and 3 mm thick, and a triangular red rubber head 50 mm long, 28 mm tall and 13 mm thick, its broad striking end on one side of the handle and its apex on the other. It stands head-down in a rigid desk cup (80 mm inside, 60 mm tall, fixed to the bench) centred at (0.215, -0.400), the handle's end up at 194 mm, the head's broad end toward the robot. Both arms start in their ready pose.

Every collider is a box, a sphere or a cylinder (sizes in the module). The leg trainer is one articulation: its thigh,
kneecap and clamp are one fixed body; the lower leg and foot (one 2.0 kg body) turn on the knee's hinge with
0.3 N m s/rad of passive damping and rest on the heel pad under their own weight, the shin
10.2 deg below level and the knee bent 18.2 deg. The scored strike band's centre is at
(357.7, -96.2, 188.6) mm (x, y, height above the bench).

## Building the leg trainer

The trainer is built, not bought. Its outside (the thigh's and shin's profile, the kneecap, tendon and tubercle, the
foot and heel, the knee's height, the roll) and the lower leg's mass are the module's (`LEG`, `ROLL` and the constants
above them); the tubes, hinge and ballast inside are how to build it. Distances "along the thigh" or "along the shin"
are measured on the tube's centre line from the knee bolt's centre; heights are square to the bench. Tools: a saw, a
utility knife, a drill with a 5/16 in bit, sandpaper, a ruler, a square and a kitchen scale.

Materials (about $150; sources and prices below): 1 in (25.4 mm) XPS foam board, three 2 x 2 ft panels; 2 in Schedule
40 PVC pipe (60.3 mm outside, 52.5 mm inside) and 1-1/2 in Schedule 40 PVC pipe (48.3 mm outside), 2 ft of each, and a
2 in PVC cap; one 2 in (50.8 mm) and one 1 in (25.4 mm) birch craft ball; about 1 m of pine 2x4 (38 x 89 mm); one
bottle of steel BBs (6,000, 2.1 kg); polyurethane construction adhesive that bonds XPS, PVC and wood (PL Premium); a
5/16 x 3-1/2 in bolt, a nylon lock nut and two 1/16 in nylon washers; a 1-1/2 in two-hole pipe strap, wood screws; duct
tape; two C-clamps.

1. Knee hinge (a clevis). Shin tube: the 2 in pipe, 432 mm long. At its knee end saw away everything more than 20 mm
   above or below its centre line over the first 50 mm, leaving two side tongues, and drill a 5/16 in hole across both
   tongues on the centre line, 25 mm from the end. Thigh tube: the 1-1/2 in pipe, 400 mm long; drill the same hole
   across it 24 mm from its knee end and round that end, seen from the side, to a half circle of 24 mm radius about the
   hole. Put the thigh tube's end between the tongues with a nylon washer on each side and join them with the bolt and
   lock nut, snug but free: the shin tube must swing on it under its own weight. The rounded end turns inside the shin
   tube's clearance, so the knee turns freely from nearly straight to well past its rest angle.
2. Foam body. Cut 25.4 mm XPS rings, their holes the tubes' outside diameters (48 mm thigh, 60 mm shin) and their outer
   diameters the module's profile at each ring's middle. Thigh, 11 rings from 35 to 314 mm along it, starting at the
   knee: 103, 108, 112, 116, 120, 122, 125, 128, 130, 132, 132 mm. Shin, 12 rings from 25 to 330 mm along it: 82, 85,
   91, 97, 104, 104, 101, 95, 89, 82, 76, 70 mm; beyond 330 mm the bare 60 mm tube is the ankle (the module's 58 to 64
   mm). Glue the rings on, sand the steps into a smooth taper (within 3 mm of these diameters) and cover with one layer
   of duct tape. The knee between the thigh's and the shin's first rings stays bare so the hinge can turn.
3. Landmarks (the strike surface). Kneecap: the 2 in ball sawn flat 5 mm below its centre (a dome 50 mm across), glued
   on a 10 mm pine spacer on the thigh tube so that, with the leg in place, its centre is 12 mm toward the hip from the
   bolt's centre and its front 62 mm above it. Patellar tendon: a pine block 45 mm long, 28 mm wide and 10 mm thick, set
   into the shin's first two rings from 12 to 57 mm along the shin on its front midline (its first 13 mm overhang the
   bare knee), its front face 45 mm above the shin tube's centre line. Tibial tubercle: the 1 in ball half sunk into the
   foam, its centre 64 mm along the shin and its top 51 mm above the centre line (6 mm proud of the tendon). Tape over
   all three. With a fine marker, mark the strike band on the tendon: lines across at 17 and 47 mm along the shin, and
   15 mm either side of the midline.
4. Foot, heel and the lower leg's mass. Foot: a pine block 200 mm tall, 90 mm wide and 50 mm thick (two 2x4 offcuts
   glued face to face, sawn to 50 mm), toes up, its face toward the knee 407 mm along the shin, reaching from 25 mm
   below the shin's centre line to 175 mm above it. The shin tube, cut so that with its cap on it reaches the foot, is
   glued into the cap, and the cap is screwed to the foot. Heel: a pine block 60 mm along the shin, 64 mm across and 41
   mm tall, glued and screwed under the foot from 415 to 475 mm along the shin; its far bottom edge, 475 mm along the
   shin and 66 mm below its centre line, is what rests on the bench. Ballast: stand the lower leg on its foot, push a
   tight 52 mm XPS plug down the tube to 190 mm from the bolt hole, pour in BBs (about 0.8 kg, filling to about 110 mm;
   an estimate) and push a second plug down hard on them so they cannot shift. Adjust the BBs until the lower leg
   (tube, foam, tendon, tubercle, foot and heel, without the bolt) weighs 2.00 +/- 0.05 kg on the kitchen scale and
   balances across a pencil 240 +/- 10 mm from the bolt hole (the module's 2.0 kg; its centre of mass, from the
   colliders at uniform density, is 242 mm along the shin), then glue the second plug.
5. Thigh support and roll. Glue two 110 mm 2x4 offcuts face to face into a block 76 mm tall; clamp it to the bench
   under the thigh tube's bare end (the tube runs on 60 mm past its last ring) and strap the tube to it with the pipe
   strap. Roll: ten 80 mm XPS discs glued into a roll 250 mm long, taped to the bench across the leg, its centre 90 mm
   toward the hip from the bolt's centre (measured along the bench). Shim under the block until the bolt's centre is
   150 +/- 2 mm above the bench; the thigh then rises 8 deg toward the knee, as modelled.

Checks, before each session: the bolt's centre is 150 +/- 2 mm above the bench and the kneecap's front 62 mm above
that; with the heel resting, the shin lies 10 deg below level and the band's centre is 189 +/- 3 mm above the bench;
the heel lifted 40 mm and let go drops straight back onto the bench (in the simulation, with its 2.0 kg lower leg and
light knee damping, in about 0.1 s), and a hinge stiff enough to hold it up is loosened; the band's marks are
visible. Nothing in the trainer is used up by a trial; re-tape a scuffed shin.

Build sources (prices seen online in November 2026): XPS: Owens Corning FOAMULAR NGX 1 in x 2 ft x 2 ft project
panel, $9.97 to $10.98 each or $28.98 for three,
https://www.homedepot.com/p/Owens-Corning-FOAMULAR-NGX-1-in-x-2-ft-x-2-ft-R-5-Project-Panel-XPS-Rigid-Foam-Board-Insulation-9WE/315401957.
PVC: Charlotte Pipe 2 in x 2 ft Schedule 40, $8.36,
https://www.homedepot.com/b/Plumbing-Pipe-Fittings-Pipe-PVC-Pipe-PVC-Schedule-40-Pipe/2-ft/N-5yc1vZbueoZ1z10ytm, and
1-1/2 in x 2 ft, $5.38 to $5.93,
https://www.homedepot.com/p/Charlotte-Pipe-1-1-2-in-x-2-ft-PVC-DWV-Schedule-40-Pipe-PVC071120200HA/100581718;
dimensions (ASTM D1785: 2 in 2.375 in outside, 2.067 in inside; 1-1/2 in 1.900 in outside),
https://www.engineeringtoolbox.com/pvc-cpvc-pipes-dimensions-d_795.html. Balls: Woodpeckers birch, 2 in bag of 25,
$29.81,
https://www.walmart.com/ip/2-inch-Wooden-Round-Ball-Bag-25-Unfinished-Natural-Hardwood-Balls-Smooth-Birch-Crafts-DIY-Projects-2-Diameter-Woodpeckers/513620294,
and 1 in bag of 100, $17.00,
https://www.walmart.com/ip/1-inch-Wooden-Round-Ball-Bag-100-Unfinished-Natural-Hardwood-Balls-Smooth-Birch-Crafts-DIY-Projects-1-Diameter-Woodpeckers/196186236.
BBs: Daisy PrecisionMax 6,000-count bottle, 4.6 lb, $17.90,
https://www.daisy.com/product/PrecisionMax-6000-ct-BB-Bottle-Model-60?itemnumber=980060-001. Adhesive: Loctite PL
Premium 10 oz, $7.48, https://www.flooranddecor.com/tile-caulk-installation-materials/loctite-10oz.-pl-premium-construction-adhesive-101005379.html
(bonds foam insulation, PVC and wood:
https://www.acehardware.com/departments/paint-and-supplies/tape-glues-and-adhesives/construction-adhesives/1015395).
Estimates, not looked up: the 2x4, cap, bolt, nut, washers, strap and screws, duct tape and two C-clamps, about $30
together.

## The reflex rule (code, not a simulated muscle)

Before every physics step (240 Hz) `Reflex.update()` checks 31 points on the hammer (the broad
face, the apex and the handle) against the leg's surfaces. A touch starts when one comes within 3 mm;
its landing speed is that point's speed into the surface just before. The touch counts as a **tap on the tendon**
when the hammer's head touched first, the broad face's centre was over the band (17 to
47 mm below the knee's axis along the tendon, within 15 mm of its midline), a gripper
was holding the hammer (fingers under 28 mm apart round it), and the head lifted off (more than 5 mm
away) within 0.25 s. Such a tap fires the knee jerk 0.06 s after landing, or at lift-off if that is later:
the knee is given an extension speed sized so the lower leg kicks up by about
10 deg x (speed - 0.6) / (1.5 - 0.6), capped at 10 deg; landings under
0.6 m/s fire nothing. A slow push, a press held more than 0.25 s, a tap with the handle, or a tap on the
kneecap, the tibial tubercle, the shin or the thigh fires nothing. The leg then falls back onto its heel.

## Scoring

The knee jerk follows the trainer's reflex rule, as on a patient: a tap counts when the hammer is held in a gripper, its head (not the handle) touches the leg first with the broad face's centre over the strike band, it lands at 0.6 m/s or more into the skin, and it lifts off (5 mm clear) within 0.25 s. The lower leg then kicks (60 ms after landing, or at lift-off if later) by 10 degrees x (landing speed - 0.6) / 0.9, at most 10 degrees. A clear knee jerk is a kick of at least 5 degrees (the foot rises about 40 mm; a landing of about 1.05 m/s or more). Full score 1.0, ending the run: a clear knee jerk, and then the hammer stood back in its cup (its head inside the cup, below the rim) or laid on the bench (lowest point within 5 mm of the bench top), let go (no gripper within 30 mm of it) and still for 1 s. 0.75: a clear knee jerk, the hammer not yet put down and let go. 0.5: only a weak jerk (at least 1 degree; a landing of about 0.7 to 1.05 m/s). 0.25: the head landed on the band, held, but no jerk fired (too slow, or it stayed pressed on the tendon). 0 otherwise, including taps on the kneecap, the tubercle, the shin or the thigh, taps with the handle, and moving the leg by hand. 0 for the rest of the run if the hammer falls off the bench (lowest point 20 mm below the bench top).

## The wrong way, measured

A GPU-only copy of this module whose demonstration aims the same brisk tap 40 mm further up the leg: the head landed
on the kneecap at 1.70 m/s and lifted off after 0.14 s, no knee jerk fired, and the run scored 0 although the hammer
was then stood back in its cup and let go.

## Scripted demonstration

The right hand grips the flat handle 35 mm below its tip, fingers closing across the handle's
12 mm width (on its edges, so the pads themselves, not friction alone, stop the head turning in the swing's
plane), and lifts the hammer out of its cup; the hammer is then held like an extension of the hand. The demonstration
reads the hammer's pose four times (as it stands; once lifted; after the tap; hanging over the cup before it goes back
in) for the grip's offset, and plans the swing from the measured grip: the
pose that puts the broad face on the band's centre with the handle across the knee from its outer side, dipping
10 deg toward the head; then constant speeds of shoulder, elbow and wrist pitch
(under half of the runner's per-step joint limits) that bring the face down at 1.35 m/s along its own normal,
with each joint's command led by its measured tracking lag. On the GPU (three runs with the hammer's start jittered by
up to 10 mm) the head landed at 1.82 m/s, its face centre 32 mm down the tendon from the knee's axis and 3 mm off the
midline, and lifted off after 0.16 s; the lower leg kicked up 10.0 deg (the heel rose about 80 mm) and fell back.
Joint 1 does not move during the swing, so where along the leg the head lands is set before the swing starts. The
moment the head touches, the next command lifts it back up the swing's path. After letting go in the cup, the open
hand rises clear of the handle and moves 45 mm aside. The left hand rests its closed fingers
on the thigh 200 mm above the knee, steadying it, as in the clinical technique.

| Time (s) | Step |
| --- | --- |
| 0.00 to 2.50 | hand over the hammer's handle |
| 2.50 to 4.00 | fingers round the handle's end |
| 4.00 to 4.80 | pinch the handle |
| 4.80 to 6.30 | lift the hammer out of the cup |
| 6.30 to 6.80 | hold |
| 6.80 to 9.80 | raise the hammer over the knee |
| 9.80 to 10.80 | hold, cocked |
| 10.80 to 11.30 | swing: the head comes down on the tendon |
| 11.30 to 12.30 | hold up after the tap |
| 12.30 to 15.30 | carry the hammer back over its cup |
| 15.30 to 15.90 | hold it upright over the cup |
| 15.90 to 17.40 | lower it head first into the cup |
| 17.40 to 18.20 | let go |
| 18.20 to 19.40 | rise clear of the handle |
| 19.40 to 20.40 | move the open hand aside |
| 20.40 to 22.40 | hold clear |

## Simulation and the real trial

Rigid bodies, 240 Hz physics, 20 Hz control. No muscle or nerve is simulated: the knee jerk is a rule in the task's code (a timed knee-extension speed fired by a brisk tap on the tendon band), its speed thresholds set from Tham et al. 2013. The leg, skin and rubber head are rigid primitives; the thigh, roll and cup are fixed. Estimates: the hammer's mass and sizes, the knee's damping and the band's size; the trainer is built to the module's sizes and 2 kg lower leg (README).

- No muscle, spinal reflex or soft tissue is simulated: the knee jerk is the rule above, written in the task's code,
  with its thresholds set from the clinical sources (the speed for a visible jerk) and marked here as estimates.
- The simulated trainer is rigid (spheres along thigh and shin, boxes for the tendon, the heel pad and the foot); the
  built trainer's taped foam gives a little under the hammer away from its wooden kneecap, tendon and tubercle, which the
  simulation does not.
- Estimates: the hammer's mass (0.07 kg), head (50 x 28 x 13 mm) and handle (12 x 3 mm), to measure on the real
  hammer; the tendon band's size; the knee's damping. The trainer's sizes, its roll and its lower leg's mass (2.0 kg)
  are not estimates for the real trial: it is built to them (Building the leg trainer).
- Real trial: a life-size leg trainer has no reflex, so the tap is judged by the same rule from a 240 fps side video
  (the head's travel over the last 4 frames before touching gives its speed; the frame of first touch shows where the
  face landed against the 30 mm band marked on the trainer's tendon (Building the leg trainer, step 3); the frames until
  it lifts off give the contact time). An instrumented reflex hammer (a force sensor in the head) can replace the video
  for speed and contact time.

## Files

- `task.json`: instruction, setup, scoring, run limit, fidelity and scores
- `check_knee_jerk_with_reflex_hammer.py`: scene, reflex rule, scorer and scripted demonstration
- `yam_parts.py`, `yam_joints.py`: kit modules (compound rigid parts, hinged mechanisms), copied unchanged
- `test_check_knee_jerk_with_reflex_hammer.py`: generated offline checks of the scorer

No third-party assets: every shape is a primitive authored in the module.

## Reset

Let the lower leg settle onto its heel (it rests there by itself); stand the hammer head-down in its cup, handle end up, the head's broad end toward the robot.
