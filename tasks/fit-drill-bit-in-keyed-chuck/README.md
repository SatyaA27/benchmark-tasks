# Tighten a 6 mm bit in a keyed drill chuck: keep the key's pinion meshed as the torque rises, then take the key out

Fit the 6 mm drill bit in the drill's chuck, centered and tightened with the chuck key, then take the key out.

General maintenance and repair workers (1.53 million in the US, BLS May 2025) "repair machines, equipment, or structures, using tools such as hammers, hoists, saws, drills, wrenches" (O*NET 49-9071.00). A keyed chuck holds nothing until it is tightened, and only its key tightens it: Bosch's steps are "Insert the tool. Insert the chuck key into the corresponding holes of the key type drill chuck and clamp the tool uniformly", and its warning "Remove any adjusting key or wrench before turning the power tool on. A wrench or a key left attached to a rotating part of the power tool may result in personal injury" (GBM 10 RE manual). The skill is the key: its 6 mm pilot in the hole and its pinion in the sleeve's teeth while the torque rises and the teeth push it out, the other hand holding the drill, until three jaws close centred on the shank and pass a holding test; then the key comes out. No bit is chosen and nothing is driven.

Sources: Bosch GBM 10 RE / GBM 10 operating instructions 1 609 92A 40X (keyed-chuck version 3 601 D73 5..: chuck
clamping range 1 to 10 mm, 1.3 kg, spindle collar 43 mm; "Changing the Tool", "Key Type Drill Chuck"; general power
tool safety warnings), https://www.bosch-professional.com/binary/manualsmedia/o273527v21_160992A40X_201711.pdf ; Bosch
replacement chuck key type D (1 607 950 045: 6 x 40 x 110 mm); Bosch keyed chucks 1 to 10 mm, 3/8"-24
(1 608 571 053); O*NET 49-9071.00 task statement; BLS OEWS May 2025 (49-9071: 1,529,700 employed).

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=fit_drill_bit_in_keyed_chuck.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Fit the 6 mm drill bit in the drill's chuck, centered and tightened with the chuck key, then take the key out." \
  --scorer task_score --max-steps 4200 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial, replace `--policy yam_scripted` with `--policy dream`. `--scorer task_score` reports the score at
the end of the run. Each trial allows 4,200 control steps (210 seconds at 20 Hz). There are no
seeds: the layout is fixed. The scripted demonstration's plan is 122.2 s long
(it stops turning the key once the chuck is tight, so some stroke slots are spent holding). `success()` requires a
score of 1.0.

`pytest` in this folder runs the generated offline checks (no GPU): the scorer on the start, the goal and the named
partial and shortcut layouts, and the chuck model's own tests (`test_chuck_model.py`).

## The scene

Millimetres from the midpoint between the arm bases: x forward, y to the robot's left, heights above the bench top. A Bosch GBM 10 RE corded drill, keyed-chuck version (3 601 D73 5..; 225 long, 200 high, 1.3 kg; blue plastic housing 62 x 155 x 70, black handle), unplugged, lies on its side on the bench: chuck axis along y, 35 above the bench, chuck nose at x 240, y -100 pointing to the robot's right, handle pointing toward the robot (its foot at x 71 to 91, y 30 to 75). Chuck: Bosch 10 mm keyed chuck (1 to 10 mm, 3/8"-24), steel, knurled sleeve 32 across and 30 long from the nose, body 33 across to 56 behind the nose, then the 43 mm spindle collar; its three round jaws (4 across) are opened to 7.5 mm; a fully pushed-in shank stops 34 in. One of its key holes (6.4 mm, 7 deep, 37 behind the nose) faces straight up. The bit: a 6 mm HSS twist drill (DIN 338, 93 long, 36 of plain shank, 14 g) lies across two wooden blocks (40 x 20, 32 high, centred at x 240, y -290 and y -225) on the chuck's axis line, shank end at y -207 toward the chuck. The key: the chuck's own steel T-handle chuck key (Bosch replacement key type D: 6 mm pilot, 10-tooth pinion 12 across, 40 from pilot tip to the 110 x 8 x 8 bar; 50 g) hangs by its bar in a wooden key rest (120 x 80, 48 high, with a 50 x 60 slot through it) centred at x 120, y -170, bar along x. README.md lists every collider.

### Collider table

Millimetres, same frame as above (heights above the bench). The drill is one free rigid body (1.3 kg); the
bit (14 g) and the key (50 g) are free rigid bodies; the bit rest and the key
rest are fixed. Each free body's centre of mass and inertia come from its colliders at uniform density. The drill is
listed where it lies (its frame: nose centre at x 240, y -100, 35 up; chuck axis along +y).

| Body | Shape | x | y | z |
| --- | --- | --- | --- | --- |
| drill | cylinder r 4, along y | 248 to 256 | -100 to -74 | 31 to 39 |
| drill | cylinder r 4, along y | 245.7 to 253.7 | -100 to -74 | 38.1 to 46.1 |
| drill | cylinder r 4, along y | 239.7 to 247.7 | -100 to -74 | 42.4 to 50.4 |
| drill | cylinder r 4, along y | 232.3 to 240.3 | -100 to -74 | 42.4 to 50.4 |
| drill | cylinder r 4, along y | 226.3 to 234.3 | -100 to -74 | 38.1 to 46.1 |
| drill | cylinder r 4, along y | 224 to 232 | -100 to -74 | 31 to 39 |
| drill | cylinder r 4, along y | 226.3 to 234.3 | -100 to -74 | 23.9 to 31.9 |
| drill | cylinder r 4, along y | 232.3 to 240.3 | -100 to -74 | 19.6 to 27.6 |
| drill | cylinder r 4, along y | 239.7 to 247.7 | -100 to -74 | 19.6 to 27.6 |
| drill | cylinder r 4, along y | 245.7 to 253.7 | -100 to -74 | 23.9 to 31.9 |
| drill | cylinder r 4, along y | 248 to 256 | -74 to -70 | 31 to 39 |
| drill | cylinder r 4, along y | 245.7 to 253.7 | -74 to -70 | 38.1 to 46.1 |
| drill | cylinder r 4, along y | 239.7 to 247.7 | -74 to -70 | 42.4 to 50.4 |
| drill | cylinder r 4, along y | 232.3 to 240.3 | -74 to -70 | 42.4 to 50.4 |
| drill | cylinder r 4, along y | 226.3 to 234.3 | -74 to -70 | 38.1 to 46.1 |
| drill | cylinder r 4, along y | 224 to 232 | -74 to -70 | 31 to 39 |
| drill | cylinder r 4, along y | 226.3 to 234.3 | -74 to -70 | 23.9 to 31.9 |
| drill | cylinder r 4, along y | 232.3 to 240.3 | -74 to -70 | 19.6 to 27.6 |
| drill | cylinder r 4, along y | 239.7 to 247.7 | -74 to -70 | 19.6 to 27.6 |
| drill | cylinder r 4, along y | 245.7 to 253.7 | -74 to -70 | 23.9 to 31.9 |
| drill | cylinder r 2, along y | 238 to 242 | -98.5 to -75 | 38.8 to 42.8 |
| drill | ball r 2 | 238 to 242 | -100.5 to -96.5 | 38.8 to 42.8 |
| drill | cylinder r 2, along y | 233 to 237 | -98.5 to -75 | 30.1 to 34.1 |
| drill | ball r 2 | 233 to 237 | -100.5 to -96.5 | 30.1 to 34.1 |
| drill | cylinder r 2, along y | 243 to 247 | -98.5 to -75 | 30.1 to 34.1 |
| drill | ball r 2 | 243 to 247 | -100.5 to -96.5 | 30.1 to 34.1 |
| drill | cylinder r 5.5, along y | 245.5 to 256.5 | -70 to -44 | 29.5 to 40.5 |
| drill | cylinder r 5.5, along y | 244 to 255 | -70 to -44 | 35 to 46 |
| drill | cylinder r 5.5, along y | 240 to 251 | -70 to -66.2 | 39 to 50 |
| drill | cylinder r 5.5, along y | 240 to 251 | -59.8 to -44 | 39 to 50 |
| drill | cylinder r 5.5, along y | 234.5 to 245.5 | -70 to -66.2 | 40.5 to 51.5 |
| drill | cylinder r 5.5, along y | 234.5 to 245.5 | -59.8 to -44 | 40.5 to 51.5 |
| drill | cylinder r 5.5, along y | 229 to 240 | -70 to -66.2 | 39 to 50 |
| drill | cylinder r 5.5, along y | 229 to 240 | -59.8 to -44 | 39 to 50 |
| drill | cylinder r 5.5, along y | 225 to 236 | -70 to -44 | 35 to 46 |
| drill | cylinder r 5.5, along y | 223.5 to 234.5 | -70 to -44 | 29.5 to 40.5 |
| drill | cylinder r 5.5, along y | 225 to 236 | -70 to -44 | 24 to 35 |
| drill | cylinder r 5.5, along y | 229 to 240 | -70 to -44 | 20 to 31 |
| drill | cylinder r 5.5, along y | 234.5 to 245.5 | -70 to -44 | 18.5 to 29.5 |
| drill | cylinder r 5.5, along y | 240 to 251 | -70 to -44 | 20 to 31 |
| drill | cylinder r 5.5, along y | 244 to 255 | -70 to -44 | 24 to 35 |
| drill | box | 243.2 to 249 | -66.2 to -59.8 | 44.5 to 51.5 |
| drill | box | 231 to 236.8 | -66.2 to -59.8 | 44.5 to 51.5 |
| drill | cylinder r 9.5, along y | 230.5 to 249.5 | -66 to -44 | 25.5 to 44.5 |
| drill | cylinder r 21.5, along y | 218.5 to 261.5 | -44 to -30 | 13.5 to 56.5 |
| drill | cylinder r 28, along y | 212 to 268 | -30 to -10 | 7 to 63 |
| drill | box | 209 to 271 | -30 to 125 | 0 to 70 |
| drill | box | 71 to 209 | 30 to 75 | 15 to 55 |
| drill | box | 185 to 209 | 18 to 30 | 27 to 43 |
| drill | box | 71 to 91 | 30 to 75 | 0 to 70 |
| bit (start) | cylinder r 3, along y | 237 to 243 | -243 to -207 | 32 to 38 |
| bit (start) | cylinder r 3, along y | 237 to 243 | -300 to -243 | 32 to 38 |
| key (start) | ball r 3 | 117 to 123 | -173 to -167 | 12 to 18 |
| key (start) | cylinder r 3, along z | 117 to 123 | -173 to -167 | 15 to 18.5 |
| key (start) | cylinder r 6, along z | 114 to 126 | -176 to -164 | 18.5 to 23.5 |
| key (start) | cylinder r 3, along z | 117 to 123 | -173 to -167 | 23.5 to 48 |
| key (start) | box | 65 to 175 | -174 to -166 | 48 to 56 |
| bit rest (fixed) | box | 220 to 260 | -300 to -280 | 0 to 32 |
| bit rest (fixed) | box | 220 to 260 | -235 to -215 | 0 to 32 |
| key rest (fixed) | box | 60 to 95 | -210 to -130 | 0 to 48 |
| key rest (fixed) | box | 145 to 180 | -210 to -130 | 0 to 48 |
| key rest (fixed) | box | 95 to 145 | -210 to -200 | 0 to 48 |
| key rest (fixed) | box | 95 to 145 | -140 to -130 | 0 to 48 |

## Scoring

Judged after the run with a ruler, a fixed pointer, a spring scale and a 100 mm bar clamped to the flutes. Key out: the key's pilot at least 30 mm from the key hole (not in any hole). Captured: at least 20 mm of the shank between the jaws (at most 73 mm of bit stands out of the nose). Centred: turned one turn by hand, the bit's axis 50 mm in front of the nose stays within 1 mm of the chuck's axis. Gripped: all three jaws on the shank. Secure: the bit does not turn in the jaws under 0.5 N m (5 N at the bar's end) for 2 s, nor slip under a 20 N pull; in the simulation, the chuck was tightened to at least 0.30 N m at the key. 1.0 (success): key out, bit captured, centred, gripped and secure, both grippers at least 3 cm clear of the drill, the bit and the key, drill and bit still (under 1 mm in 0.5 s), held 1 s. 0.5: key out, bit captured, centred and gripped, short of that (not secure, a hand not clear, or not yet still for 1 s). 0.25: key out, bit at least 5 mm in the chuck (its shank end between the jaws, within 5 mm of the chuck's axis; the bit within 15 degrees of the axis) and either not gripped (lying in the open jaws) or gripped centred with under 20 mm in. 0: the key left in the chuck, no bit in the chuck, the bit clamped off-centre (more than 1 mm out 50 mm in front: crooked), or, at the end, the drill fallen off the bench or out past its edge (its centre of mass below the bench top or outside the 1,100 x 1,300 mm top: x -190 to 910, y -650 to 650). Lifting, tipping or moving the drill over the bench is allowed and costs nothing by itself.

How each condition is read in `score()` (the chuck's state comes from the chuck model, `chuck_runtime.py`):

- **Key out:** the key's pilot tip at least 30 mm from the key hole's mouth, and the key not meshed.
- **In the chuck / captured:** the shank's end at least 5 mm / 20 mm past the nose face, along the
  chuck's axis (the jaws are 25 mm long), and between the jaws: within 5 mm of the
  chuck's axis (its largest opening is 10 mm) and no deeper than the bore's end
  (34 mm, with 2 mm for contact); the bit within 15 degrees of the axis. A bit lying beside the drill,
  parallel to it, is not in the chuck.
- **Centred:** the bit's axis 50 mm in front of the nose within 1 mm of the chuck's
  axis (half the reading of a full turn against a fixed pointer).
- **Gripped:** the model's three jaws have closed on the shank (opening = 6.0 mm with the shank between them).
  Gripped but not centred (clamped crooked) scores 0.
- **Secure:** the model's chuck was tightened to at least 0.30 N m at the key (by our estimate of the
  gearing, about twice that, 0.6 N m, holds the bit against turning; the real-world check is
  0.5 N m on the bit).
- **Let go:** both grippers' centre points at least 30 mm outside the drill's, the bit's and
  the key's axis-aligned bounding boxes. **Still:** the drill and the bit move under 1 mm over 0.5 s.
- **On the bench:** at the end, the drill's centre of mass (from its colliders at uniform density) over the bench
  top, x -190 to 910 mm and y -650 to 650 mm, and not
  below its surface; otherwise (fallen off, or carried out past the edge) the score is 0. Nothing about the drill is
  latched during the run: it may be lifted, tipped, turned over or moved anywhere over the bench.

Why the shortcuts fail: on the simulator, the bit's shank pushed only 8 mm into the open jaws and let go tipped 20 degrees in them, so the closing jaws never held it: the run scored 0 (had it lain within 15 degrees, the jaws would have clamped it at their front, crooked, which also scores 0). Tightened with the key left in the chuck scores 0 (the maker's warning). A bit laid on the bench beside the drill, parallel to it, is not in the chuck (its shank end is not between the jaws) and scores 0. Tried offline on 437 random layouts of the free parts (the drill included), the scorer never gave a non-zero score.

## Scripted demonstration

The left hand pinches the drill's handle foot (front and back faces) from about 5 s until the key is out: the hands
are then about 0.2 m apart. The right hand does the rest. The demonstration reads the simulator's state to line up:
the bit's and the key's start poses, the bit in the hand before it goes into the jaws (three times), the key's pilot
over the hole (twice), the T-bar before every regrip, and the chuck model's tightness (it aims for 0.34 N m;
near that load the teeth often lift the key and it skips first, and the next stroke tops the chuck up).
Each stroke turns the key clockwise (seen from its handle) by up to 205 degrees with the wrist alone
(the T-bar looks the same turned half a turn), pressing it down by commanding the hand 14 mm below the
seated bar.

| Start (s) | Step (right hand) |
| --- | --- |
| 0.0 | over the bit |
| 2.5 | pads down round the flutes |
| 4.0 | pinch the bit |
| 4.8 | settle |
| 5.1 | lift off the rest |
| 6.1 | carry to the chuck |
| 8.6 | down to the axis, 25 mm short |
| 9.6 | hold |
| 10.2 | line up 10 mm short (read) |
| 11.4 | hold |
| 12.0 | line up 4 mm short (read) |
| 12.8 | hold |
| 13.3 | line up 3 mm short (read) |
| 13.9 | hold |
| 14.3 | push the shank 24 mm into the jaws |
| 17.3 | hold |
| 17.6 | let go of the bit |
| 18.2 | rise |
| 19.2 | over the key |
| 21.7 | pads down round the bar |
| 23.2 | pinch the bar |
| 24.0 | settle |
| 24.3 | lift the key out of its rest |
| 25.8 | carry over the top hole |
| 28.8 | down, pilot 15 mm up |
| 29.8 | hold |
| 30.4 | line up the pilot (read) |
| 31.4 | hold |
| 32.0 | down, tip 2.5 mm over the mouth (read) |
| 33.0 | hold |
| 33.4 | lower the pilot in, pinion on the ring gear |
| 34.9 | press the key in |
| 35.5 | stroke 1: (re)grip the bar, press, turn up to 205 degrees clockwise, ease, open, rise, roll back |
| 42.0 | stroke 2: (re)grip the bar, press, turn up to 205 degrees clockwise, ease, open, rise, roll back |
| 50.7 | stroke 3: (re)grip the bar, press, turn up to 205 degrees clockwise, ease, open, rise, roll back |
| 59.4 | stroke 4: (re)grip the bar, press, turn up to 205 degrees clockwise, ease, open, rise, roll back |
| 68.1 | stroke 5: (re)grip the bar, press, turn up to 205 degrees clockwise, ease, open, rise, roll back |
| 76.8 | stroke 6: (re)grip the bar, press, turn up to 205 degrees clockwise, ease, open, rise, roll back |
| 85.5 | stroke 7: (re)grip the bar, press, turn up to 205 degrees clockwise, ease, open, rise, roll back |
| 94.2 | stroke 8: (re)grip the bar, press, turn up to 205 degrees clockwise, ease, open, rise, roll back |
| 102.9 | stroke 9: (re)grip the bar, press, turn up to 205 degrees clockwise, ease, open, rise, roll back |
| 108.1 | ease the press |
| 108.6 | lift the key out |
| 110.1 | turn the bar along x |
| 111.1 | carry to the rest |
| 114.1 | lower into the rest |
| 115.6 | let go |
| 116.2 | rise |
| 117.2 | park |
| 119.2 | hold clear |

On the simulator the scripted demonstration reaches full score at 117.9 s (step 2,357 of the 4,200 allowed): the bit held 23 mm deep, the chuck tightened past 0.30 N m, the key back in its rest and both hands parked.

## Simulation and the real parts

Rigid bodies, 240 Hz. PhysX cannot mesh the key's pinion or drive the jaws, so the chuck is modelled: the sleeve turns only with the pilot seated in the hole, 0.5 mm per turn, +0.6 N m/rad past jaw contact, the teeth pushing the key out 18 N per N m. The bit is let go in the open jaws (7.5 mm; it rests on two); the modelled jaws lift it onto the axis, straight only if 20 mm in, and hold its depth unless a hand is at it. Estimated: those rates; jaw, hole, key sizes. Spindle fixed; no cord.

The chuck model (`chuck_runtime.py`, applied before every physics substep):

- Meshed: the key's pinion face within 0.6 mm of its seat at the hole's mouth and the key within
  1 mm and 8 degrees of the hole's axis; it skips (unmeshes) when lifted more
  than 1 mm, tipped past 12 degrees or pushed 1.5 mm off. The pilot
  itself is a real collider in a real 6.4 mm hole.
- Turning: while meshed and held, the key drives the sleeve through a stiff torsional spring (4.0 N m/rad).
  The sleeve turns when that torque beats the chuck's: 0.03 N m before the jaws reach the bit, then
  +0.6 N m per radian of key turn (0.42 to back it off: self-locking). The jaws close
  0.5 mm of diameter per key turn (from 7.5 mm: three turns to a 6 mm shank). Let go, the chuck stays.
- The teeth push the key out of the hole with 18 N per N m they carry (a straight bevel pinion's
  separating force); the hand must press it in. Every force and torque on the key acts, equal and opposite, on the
  drill at the hole, so an unheld drill can turn on the bench.
- The jaws are three 4 mm round steel rods with rounded tips, fixed at the 7.5 mm opening for contact; their closing
  is modelled. At the start opening (7.5 mm) an unheld shank lies on the two lower jaws, 2.4 mm under the axis; the
  gap between two jaws (5.96 mm) is narrower than the shank, so it cannot drop between them. As the opening shrinks
  the jaws push the shank (where it lies between them) straight toward the axis; with at least
  20 mm of shank between them they also straighten it, with less they hold it at their front at the
  angle it lies (a shallow, tipped bit is clamped crooked and scores 0). Once they have begun to close, their friction
  keeps the shank at its depth along the axis (the model holds its depth) unless a gripper's centre is within
  40 mm of the bit outside the chuck, when a hand can push it further in or pull it out. At 6.0 mm the
  bit is gripped and moves with the drill. Loosened, they let go. The bit is not held by a hand while the jaws close:
  it is let go in the open jaws, as one hand steadies the drill and the other turns the key.
- Estimates, to measure on the real drill: the chuck's and key's dimensions beyond Bosch's (sleeve, body, jaw, hole
  and pinion sizes, the hole 37 mm behind the nose), the jaw travel per key turn (from a patent's ~25-30 turns to
  13 mm for bevel-pinion chucks), the torques above, the key's 50 g, the housing's and handle's shape (only 225 x 200
  mm and 1.3 kg are Bosch's), friction 0.5.
- Shortcuts: the spindle is fixed in the drill (a real GBM 10 RE has no spindle lock; its gearbox resists turning);
  only the top key hole is modelled (the other two face down); the power cord is not modelled; the hand's tightening
  is limited by its press, not by a torque spec (Bosch gives none for the key).

## Reset

Loosen the chuck with the key to a 7.5 mm opening (a 7.5 mm gauge pin just slides between the jaws), lay the bit back across its two blocks, hang the key in its rest, turn the chuck so a key hole faces straight up, and lay the drill back on its side where it was. Nothing is consumed.
