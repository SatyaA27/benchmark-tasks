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

Left and right are the robot's. Positions are in metres from the midpoint between the two arm bases: x forward, y to the robot's left; heights are above the bench top. Leg trainer: a life-size adult right-leg trainer cut at mid-thigh, lying on its back on the bench with its foot away to the robot's front right (the leg's line points 60 degrees right of straight ahead). Its thigh stub is clamped to the bench and rests on an 80 mm foam roll (250 mm long, fixed) 90 mm above the knee; the knee's hinge is at (0.338, -0.062), 150 mm up; the knee is bent about 18 degrees (the thigh rises 8 degrees toward the knee, the shin falls 10 degrees to the heel) and the heel rests on the bench. The lower leg and foot (2 kg) swing freely on the knee's hinge, lightly damped, and lie on the heel under their own weight. Kneecap: 50 mm across, its front 62 mm above the knee's axis. Patellar tendon: a strip 28 mm wide running 45 mm down the shin from the kneecap's lower pole (12 mm below the knee's axis) to the tibial tubercle, a bump 6 mm proud of it. The strike band is the middle of the tendon: 17 to 47 mm below the knee's axis along the shin and within 15 mm of the tendon's midline; its centre is at (0.358, -0.096), 189 mm up. Reflex hammer: an ADC Taylor percussion hammer, 190 mm long and about 70 g: a flat chrome handle 12 mm wide and 3 mm thick, and a triangular red rubber head 50 mm long, 28 mm tall and 13 mm thick, its broad striking end on one side of the handle and its apex on the other. It stands head-down in a rigid desk cup (80 mm inside, 60 mm tall, fixed to the bench) centred at (0.215, -0.400), the handle's end up at 194 mm, the head's broad end toward the robot. Both arms start in their ready pose.

Every collider is a box, a sphere or a cylinder (sizes in the module). The leg trainer is one articulation: its thigh,
kneecap and clamp are one fixed body; the lower leg and foot (one 2.0 kg body) turn on the knee's hinge with
0.3 N m s/rad of passive damping and rest on the heel pad under their own weight, the shin
10.2 deg below level and the knee bent 18.2 deg. The scored strike band's centre is at
(357.7, -96.2, 188.6) mm (x, y, height above the bench).

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

Rigid bodies, 240 Hz physics, 20 Hz control. No muscle or nerve is simulated: the knee jerk is a rule in the task's code (a timed knee-extension speed fired by a brisk tap on the tendon band), its speed thresholds set from Tham et al. 2013. The leg, skin and rubber head are rigid primitives; the thigh, roll and cup are fixed. Estimates: the hammer's mass and sizes, the trainer's sizes, lower-leg mass and knee damping, and the band's size.

- No muscle, spinal reflex or soft tissue is simulated: the knee jerk is the rule above, written in the task's code,
  with its thresholds set from the clinical sources (the speed for a visible jerk) and marked here as estimates.
- The leg trainer is rigid (spheres along thigh and shin, boxes for the tendon, the heel pad and the foot); a real trainer's skin
  gives a little under the hammer, which the simulation does not.
- Estimates, to measure on the real hammer and trainer: the hammer's mass (0.07 kg), head
  (50 x 28 x 13 mm) and handle (12 x 3 mm); the
  trainer's sizes and its lower leg's mass (2.0 kg); the tendon band's size; the roll's diameter.
- Real trial: a life-size leg trainer has no reflex, so the tap is judged by the same rule from a 240 fps side video
  (the head's travel over the last 4 frames before touching gives its speed; the frame of first touch shows where the
  face landed against a 30 mm band marked on the trainer's skin below the kneecap; the frames until it lifts off give
  the contact time). An instrumented reflex hammer (a force sensor in the head) can replace the video for speed and
  contact time.

## Files

- `task.json`: instruction, setup, scoring, run limit, fidelity and scores
- `check_knee_jerk_with_reflex_hammer.py`: scene, reflex rule, scorer and scripted demonstration
- `yam_parts.py`, `yam_joints.py`: kit modules (compound rigid parts, hinged mechanisms), copied unchanged
- `test_check_knee_jerk_with_reflex_hammer.py`: generated offline checks of the scorer

No third-party assets: every shape is a primitive authored in the module.

## Reset

Let the lower leg settle onto its heel (it rests there by itself); stand the hammer head-down in its cup, handle end up, the head's broad end toward the robot.
