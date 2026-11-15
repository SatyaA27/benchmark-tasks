# Check a car seat belt locks: draw the webbing out slowly, jerk it so the retractor locks and holds, then ease it back

Check the seat belt: hold its latch tongue and pull the webbing out slowly, at least 12 cm, then jerk it so the belt locks, keep pulling for a second to show it holds, then let the webbing wind back in slowly until the tongue rests on its stop, and let go.

Almost every adult knows both halves of this from buckling up: pull a seat belt out slowly and it runs free; pull it sharply and it locks. Owner's manuals give it as the check that a belt works: "With the webbing half unreeled, hold the tongue plate and pull forward quickly. The mechanism should lock automatically and prevent further unreeling." The belt's emergency-locking retractor senses how fast the webbing speeds up: under FMVSS 209 a webbing-sensitive retractor must not lock at 0.3 g or less and must lock within 25 mm at 0.7 g. So the robot has to do two opposite things with the same grip: draw the webbing out smoothly enough that it never catches, then give a sharp jerk and keep pulling so the lock visibly takes the load, and finally let the spring take the webbing back without letting the tongue fly into the trim.

## Sources

- Owner's check: Tesla Model Y Owner's Manual, Seat Belts, testing the seat belts: "With the webbing half unreeled,
  hold the tongue plate and pull forward quickly. The mechanism should lock automatically and prevent further
  unreeling." (https://www.tesla.com/ownersmanual/modely/en_us/GUID-3319C55D-F148-4D9C-98A6-AD09BA7EED61.html)
- FMVSS No. 209, Seat belt assemblies (49 CFR 571.209), S4.3(j)(2): an emergency-locking retractor "(ii) Shall lock
  before the webbing payout exceeds the maximum limit of 25 mm when the retractor is subjected to an acceleration of
  0.7 g ... The retractor is determined to be locked when the webbing belt load tension is at least 35 N. (iii) For a
  retractor sensitive to webbing withdrawal, shall not lock before the webbing payout extends to the minimum limit of
  51 mm when the retractor is subjected to an acceleration no greater than 0.3 g"; (i)(A) under zero acceleration it
  must "exert a retractive force of not less than 1 N and not more than 7 N when attached to a strap or webbing that
  restrains both the upper torso and the pelvis". S5.2(j)(2)(iii)(B): for a retractor sensitive to webbing withdrawal,
  "Accelerate the retractor in the direction of webbing retraction and measure the webbing payout."
- The belt: Wesco Performance's universal 3-point retractable seat belts for cars: "All of our 3 point retractable
  seat belts are ELR Retractable. Each is equipped with an Emergency Locking Retractor (ELR) inertia reel, which allows
  free movement while driving, but automatically locks in an emergency ("webbing sensitive")"; FMVSS 209 and 302;
  webbing 1-7/8 in (47.6 mm) wide (https://www.wescoperformance.com/resebe.html). Webbing-sensitive means the reel
  locks on how fast the webbing itself is pulled out, so a hand on the tongue can test it. The seller gives no sizes for
  the latch tongue and no wind-back force; those in this task are estimates (below).

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=check_seat_belt_locks_on_a_jerk.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Check the seat belt: hold its latch tongue and pull the webbing out slowly, at least 12 cm, then jerk it so the belt locks, keep pulling for a second to show it holds, then let the webbing wind back in slowly until the tongue rests on its stop, and let go." \
  --scorer task_score --max-steps 1200 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial, replace `--policy yam_scripted` with `--policy dream`. `--scorer task_score` reports the score at
the end of the run. Each trial allows 1,200 control steps (60 seconds at 20 Hz). There are no
seeds: the belt is fixed to its stand, so the layout is the same every trial. The scripted demonstration takes
16.4 s. `success()` requires a score of 1.0.

`pytest` in this folder runs the generated offline checks (no GPU): the scorer on the start, the goal and the named
partial and shortcut layouts (the retractor's readouts and the stages reached earlier are given as fixtures there; the
retractor rule runs only in the simulator).

## The scene

Left and right are the robot's. Positions are in metres from the midpoint between the two arm bases: x forward, y to the robot's left; heights are above the bench top. Stand (made for the trial, standing in for the car's door pillar and its trim): a closed grey-painted steel box 300 mm deep, 200 mm wide and 225 mm tall, bolted to the bench in front of the right arm, its face toward the robot at x = 0.355, centred on y = -0.250 (it spans x 0.355 to 0.655, y -0.350 to -0.150). On the face, a black moulded plastic trim 70 mm wide and 22 mm tall frames the belt's webbing slot, centred 155 mm above the bench. Seat belt: a car three-point seat belt with a webbing-sensitive emergency-locking retractor certified to FMVSS 209 (such as Wesco Performance's 3-point retractable belt), its retractor bolted inside the box; only the shoulder webbing and its latch tongue are used. The webbing, black-grey woven polyester 47.6 mm (1-7/8 in) wide and 1.2 mm thick, comes out of the slot level, toward the robot. A grey plastic stop button (10 x 16 x 7 mm) on the webbing keeps the tongue just outside the slot at rest. Latch tongue: a black moulded plastic body 60 mm wide, 40 mm tall and 12 mm thick with the steel tongue below it (26 mm wide, 30 mm long, 2.5 mm thick), 45 g in all; it hangs on the webbing, its top (where the webbing loops through it) at (0.310, -0.250), 155 mm up, its steel tip 80 mm above the bench. The retractor winds the webbing back in with about 2.5 N at the stop (2.9 N at 135 mm out) and locks when the webbing is pulled out sharply. Arms: both start in the raised ready pose with the fingers open; the left arm is not needed.

Every collider is a box (sizes in the module). The belt is one articulation: the stand is its fixed base; the
webbing swivels at the slot (side to side 20 deg either way, up and down
12 deg, both lightly sprung so a free belt comes straight out of the slot), then slides
out of the slot (240 mm of travel; the spool's inertia is given as 0.1 kg on the slide); the tongue
twists about the webbing (30 deg either way) and swings on it
(70 deg either way). The slide's lower stop is the stop button meeting the slot's trim.

## The retractor rule (code, standing in for the inertia sensor and pawl)

Before every physics step (240 Hz) `Retractor.update()` reads the webbing's payout x and its speed and works out the
payout acceleration, filtered over 15 ms as the retractor's light inertia disc lags the spool.

- Wind-back force: 2.5 N at the stop plus 3.0 N per metre of payout (so 2.9 N at
  135 mm), and 2.0 N s/m of drag; let go, the webbing winds in at about 1.2 m/s.
- Trip: when the filtered payout acceleration reaches 0.5 g while the webbing is going out, the
  sensor trips; the pawl then catches the ratchet after 5 mm more payout. If the reading falls below
  0.25 g for 30 ms before the pawl catches, the sensor resets.
- Locked: the webbing cannot pay out past the catch point (it is held by a stiff spring, 5000 N/m and
  28 N s/m, standing in for the webbing's stretch and the spool tightening); it can still wind in. Letting
  it back 2 mm frees the pawl, as on a real belt (let it retract a little and it pulls out freely again).
- The tension the stand sees is the wind-back force plus the drag plus, when locked, the spring's force; this is the
  pull along the webbing that the scorer reads.

The 0.5 g trip level is an estimate inside the window FMVSS 209 sets for a webbing-sensitive retractor
(no lock at 0.3 g or less; lock within 25 mm at 0.7 g); the maker does not publish the actual setting. With this rule
a steady 0.7 g pull locks after about 9 mm of payout and a 0.3 g pull never locks, so the model
meets both of the standard's limits.

## Scoring

Measured from the start of the run (the record is cleared only at reset). Payout: how far the webbing is out of the slot beyond its rest, read off 5 mm ticks printed on it at the slot. Tension: the pull along the webbing. Jerk onset: the moment the tongue's outward acceleration rises through 0.3 g (FMVSS 209's no-lock level), from its tracked position over 20 ms windows. Stages, in order: 0.25 when the webbing has paid out at least 120 mm without locking since it last left its stop; 0.5 when a jerk then locks it within 25 mm of payout after the jerk's onset (FMVSS 209's limit) and the locked belt then carries at least 6 N for 0.25 s (twice its 2.9 N wind-back force, so an unlocked belt would pay out; the jolt of the catch alone lasts about 0.1 s) with the tongue still held; 0.75 when, after that, it carries at least 6 N for a further 1.0 s while paying out no more than 5 mm more; 1.0 when the lock is then freed and the webbing winds back to its stop (within 3 mm) never faster than 0.5 m/s (the tongue is guided in, not let fly), the tongue is let go, and both grippers are at least 30 mm clear of it and the webbing is still, for 1 s. Anything less scores 0. A jerk that never locks the belt, or a lock the hand does not keep pulling against (stopping dead where it caught), stays at 0.25; locking it during the slow pull scores 0 until the webbing is wound back to its stop and pulled out slowly again. Real trial: a 240 fps video from above with a steel rule beside the webbing gives payout, speed, the jerk's onset and the catch point; tension comes from the arm's joint-torque log, checked once against a 0-20 N spring scale hooked in the tongue's latch window while the belt is locked.

## The wrong ways, measured

Three GPU-only copies of this module, whose demonstrations differ only after the slow pull; each ended with the webbing back on its stop and the tongue let go.

- A gentle pull instead of a jerk (45 mm more over 1.2 s, peak 0.29 g): the webbing paid out freely to 178 mm and never locked. Score 0.25 (belt-gentle-2).
- The jerk, then the hand stops dead where the webbing caught, instead of keeping on pulling (the shortcut of stopping rather than loading the lock): the catch's jolt gave 11 to 13 N for 0.1 s, then the tension fell to the 2.9 N wind-back force, the webbing drew back 2 mm and the lock freed 0.18 s after catching. Score 0.25 (belt-stopdead-2): the lock never carried 6 N for 0.25 s.
- The whole test, but the hand lets go of the tongue 140 mm out instead of guiding it home: the webbing wound in at up to 1.09 m/s and the tongue struck its stop. Score 0.75 (belt-flyback-2).

## Scripted demonstration

The right hand comes down over the latch tongue with its fingers open across the tongue's 60 mm width, pinches the moulded body 30 mm below the webbing, and draws the webbing straight out toward the robot: its target moves 150 mm in 2.2 s (peak 0.12 m/s), and the soft arm trails it by about 17 mm against the 2.9 N wind-back force, so the webbing pays out 133 mm. After a 0.4 s pause the hand's target jumps 45 mm outward in one control step: the jerk. Once the webbing has caught, the target is kept 48 mm beyond the catch point, so the arm keeps pulling; then it is let back 8 mm, and the hand guides the webbing home and lets go at the stop.

Measured on the GPU (demo-9773bead-a1 and an identical diagnostic copy of this module): the slow pull peaked at 0.29 g of filtered payout acceleration (the 20 Hz steps of the arm's targets; raw spikes up to 0.95 g) and paid out 133 mm with no lock. In the jerk the payout acceleration rose through 0.3 g at 133.2 mm, tripped the sensor at 0.73 g at 133.6 mm, and the pawl caught at 138.7 mm, 5.5 mm after the onset, with the webbing moving at 0.35 m/s; the catch jolted the tongue with up to 23 N for about 0.1 s. The locked belt then carried 9.0 N for 1.75 s and paid out no further (it sat 1.2 mm past the catch point, the lock's spring). Letting it back freed the lock after 2 mm; the hand guided the webbing home at no more than 0.12 m/s with 2.4 to 2.9 N on it, let go at the stop and rose clear. Score 1.0 from step 300 (15.0 s) of 319. The tongue stayed in the pinch throughout (it turns about 20 deg about the webbing under the pull). The same 45 mm jerk with the lock switched off (a probe copy, never submitted) reached 1.37 g raw and 1.00 g filtered at 0.41 m/s: the arm's jerk clears FMVSS 209's 0.7 g lock level, but could not reach UN R16's 3.0 g strap-sensitivity level.

| Time (s) | Step |
| --- | --- |
| 0.00 to 2.50 | hand over the latch tongue |
| 2.50 to 4.00 | fingers either side of the tongue |
| 4.00 to 4.60 | pinch the tongue |
| 4.60 to 5.00 | hold |
| 5.00 to 7.20 | pull the webbing out slowly |
| 7.20 to 7.60 | pause |
| 7.60 to 7.75 | jerk |
| 7.75 to 9.65 | keep pulling: the locked belt holds |
| 9.65 to 10.15 | let the webbing back a little: the lock frees |
| 10.15 to 12.15 | guide the webbing back in, slowly |
| 12.15 to 12.65 | tongue on its stop |
| 12.65 to 13.25 | let go |
| 13.25 to 14.45 | rise clear |
| 14.45 to 16.45 | hold clear |

## Simulation and the real trial

The lock is a rule standing in for the retractor's inertia disc and pawl: payout acceleration, filtered over 15 ms, trips it at 0.5 g (between FMVSS 209's 0.3 g no-lock and 0.7 g lock levels; the real setting is unpublished); the pawl catches 5 mm later; locked webbing is a 5,000 N/m spring; 2 mm let back frees it. The webbing is a rigid strip sliding through a swivel at the slot, its tail hidden in the stand. Wind-back force, drag, spool inertia, tongue size and mass are estimates.

- No inertia disc, pawl or ratchet is simulated as bodies: the lock is the rule above, written in the task's code. The
  webbing is a rigid strip that slides out of the slot and swivels there; the real webbing is cloth wound on a spool
  inside the retractor. While the belt is under tension (always, through the wind-back spring) the real webbing between
  slot and tongue is straight as well; a rigid strip cannot sag or fold if the tongue were pushed toward the slot.
- The stand is a closed box made for the trial; in a car the retractor sits in the door pillar behind the trim, the
  webbing runs up to a guide at shoulder height and a lap part runs on to a floor anchor. Here only the retractor and
  the shoulder webbing with its tongue are used, as on a retractor bench test.
- Estimates, to measure on the real belt: the wind-back force (2.5 N at the stop; FMVSS 209 allows 1 to 7 N),
  the drag, the spool's inertia, the trip level (0.5 g, between the standard's 0.3 and 0.7 g), the pawl's
  travel (5 mm), the tongue's size and mass (60 x 40 x 12 mm body,
  45 g), the webbing's thickness (1.2 mm).
- Real trial: a 240 fps video from above with a steel rule laid beside the webbing gives the payout frame by frame (5 mm ticks printed on the webbing, read at the slot), and from it the speed and acceleration over 20 ms windows: the jerk's onset (outward acceleration rising through 0.3 g), the catch (the payout stops while the arm still pulls) and the wind-in speed. The pull during the hold is the arm's force along the webbing from its joint-torque log (the YAM's motors report torque), checked once before the session by hooking a 0-20 N spring scale in the tongue's latch window and pulling the locked belt by hand to 6 N.

## Files

- `task.json`: instruction, setup, scoring, run limit, fidelity and scores
- `check_seat_belt_locks_on_a_jerk.py`: scene, retractor rule, scorer and scripted demonstration
- `yam_parts.py`, `yam_joints.py`: kit modules (compound rigid parts, jointed mechanisms), copied unchanged
- `test_check_seat_belt_locks_on_a_jerk.py`: generated offline checks of the scorer

No third-party assets: every shape is a primitive authored in the module.

## Reset

Nothing is consumed. Let the webbing wind fully in until the stop button meets the trim and the tongue hangs still. Before each session check the belt by hand: a slow pull of 150 mm runs free and a sharp tug locks it; let it back a little to free the lock, and check the wind-back force with a spring scale (about 2.5 to 3 N).
