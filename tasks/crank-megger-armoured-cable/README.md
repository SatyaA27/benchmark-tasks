# Find the leaking core of an armoured cable: crank the insulation tester with one hand, probe each core with the other

A ZC25-3 hand-crank insulation tester makes its 500 V test voltage only while its generator handle turns. One core of
a 3-core steel-wire-armoured cable leaks to its armour (0.5 megohm; which core is set per trial and hidden). The robot
keeps the crank turning with one arm while the other holds the red line prod on each core's stud in turn, reads the
needle, tips the flag in front of the leaking core, puts the prod back in its rest and lets go.

## Scoring

`score()` and `success()` in `crank_megger_armoured_cable.py`; a person checks the same with a ruler, a protractor
and a stopwatch.

- **Tested core:** the prod's tip has rested on that core's stud (on the brass: within 6 mm of the stud's axis, at
  most 2.5 mm above the stud's top and not below the nut's top; prod within 30 degrees of vertical) for 1 s in a row
  while the crank turned at 0.8 rev/s or more, averaged over the last 0.5 s.
- **Tipped flag:** 25 degrees or more from upright (its stop is at 50).
- **0** if any flag is tipped before all three cores have been tested (it stays 0 for the trial), if a healthy core's
  flag is tipped, or if no core was tested.
- **0.25** one or two cores tested.
- **0.5** all three tested, the leaking core's flag not tipped.
- **0.75** all three tested and only the leaking core's flag tipped, but the prod is not back in its rest (tip
  through the floor's slot, finger guard within 4 mm of the floor, leaning 20 degrees or less: the cup holds it within about 16), or a gripper is within
  5 cm of the prod's grip, the crank handle or a flag's top, or something still moves (prod over 1 cm/s, crank over
  0.5 rad/s, a flag over 0.1 rad/s).
- **1.0** all of the above for 1 s (`success()`; the episode then ends).

The flag is scored against the trial's configured leaking core, so the needle has to be read: tipping a fixed flag,
or every flag, scores 0 on the other trials.

## Run a trial

From this folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=crank_megger_armoured_cable.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "One core of this armoured cable leaks to the armour. The tester's earth lead is already on the earth stud, which is bonded to the armour. Keep the tester's crank turning at least 0.8 turns a second with one hand, and with the other hold the red prod on each core's stud in turn until the needle settles: test all three cores before touching any flag. Then tip back the flag in front of the core whose needle stayed away from infinity, put the prod back in its slotted rest and let go of everything." \
  --scorer task_score --max-steps 1200 --epochs 3 --seed 17 --store-frames --log-dir ./trial
```

Trial configurations: the leaking core is the trial seed mod 3 (0 brown, 1 black, 2 grey). Inspect Robots derives each
epoch's trial seed from `--seed` and the epoch, so `--seed 17 --epochs 3` runs the grey, black and brown cores in turn
(`--seed 12`: black, brown, grey). The console prints `MEGGER_TRIAL seed=... leaking_core=...` at each reset. Report
the mean over the three. Each trial allows 1,200 control steps (60 s at 20 Hz). For the scripted demonstration use
`--policy yam_scripted`. `pytest` in this folder runs the offline checks of the scorer (start, goal, partial and
shortcut layouts, every leaking core).

## The circuit (a core-to-earth insulation test)

- The cable is 3-core 1.5 mm2 XLPE/SWA/PVC to BS 5467 (6943X; 12.6 mm across; brown, black, grey cores; galvanised
  steel wire armour). Its armour is clamped in a brass CW20S gland in a steel gland plate; the gland's brass earth tag
  (under the locknut) is joined by a green/yellow lead to the earth stud of the terminal block.
- The tester's EARTH lead is on that earth stud, so the tester's EARTH terminal is connected to the armour.
- The three cores, on crimped M6 ring lugs, are under the nuts of the three core studs. The tester's LINE terminal is
  on the red prod; the prod on a core's stud puts the tester across that core and the armour.
- The leak: in the closed box at the cable's far end the cores are cut back and kept apart, and a 0.5 megohm 1 W
  resistor joins the leaking core to the armour. A healthy core reads over 500 megohm (infinity on this scale); the
  leaking one reads 0.5 megohm, below the 1 megohm BS 7671 minimum for a 500 V test.
- Safe isolation: the cable is connected to no supply at either end; the only voltage is the tester's own. After each
  trial each core is discharged by touching its stud to the earth stud.

The ZC25 instructions describe this connection for a core-to-sheath test of a cable: the core on LINE, the
sheath on EARTH (GUARD, not used here, would go on the insulation between).

## The scene

Positions in cm from the midpoint between the arm bases (x forward, y left), heights above the bench. Full list in
`task.json` (setup).

| Object | What it is |
|---|---|
| `tester` | ZC25-3, 20.5 x 12 x 14.5 cm, fixed, centred x 30, y -8.75; crank on its right end face (6 cm, shaft 8.5 high, free-turning handle 1.4 x 5.5 cm); dial on top, needle pivot x 27, y -9.75 |
| `board` | terminal block 4 x 14 x 1.5 cm with four brass M6 studs at x 30 (earth y 5.5; cores y 9, 12.5, 16), gland plate, gland, cable, far box, earth bond and earth lead; three red flags on pins at x 26.2 |
| `rest` | prod rest, a cup with a 2.6 cm square bore and a 1 cm slot in its floor, at x 24, y 24.5 |
| `prod` | red test prod, 14 cm, 35 g, 2 mm steel tip, finger guard 1.6 cm above the tip |

The needle is drawn as one of 37 bars 2.5 degrees apart, only the current one visible; it moves toward the reading
of whatever the tip touches (infinity in the air or on a healthy core, 0.5 megohm on the leaking core, zero on the earth
stud) only while the crank turns at 0.8 rev/s or more, and otherwise stays where it was, like the free pointer of a
ratio-meter.

## Scripted demonstration

`demo()` (played by `--policy yam_scripted`): the right hand grips the crank handle from above and turns it at
1.0 rev/s for about 11 s (ramping up and down over 0.6 s); the hand follows the handle's 6 cm circle while the handle
turns on its pin inside the closed fingers. The left hand lifts the prod out of its rest, holds it 15 mm over the brown core's stud,
lowers its tip 2.5 mm into the stud top, holds 1.6 s, lifts, and does the same on the black and grey cores. It then
puts the prod back in its rest, lets go, and with closed fingers tips back one flag.

Which flag: at the end of each hold the demo reads the needle in the top camera image the policy receives (320 x 176):
for each candidate angle it compares the darkness of the needle line on the scale with the lines 15 degrees either
side, and keeps the darkest. The leaking core is the one whose reading stood at least 20 degrees below the highest,
when the highest is within 8 degrees of infinity. It never reads the configured core. It reads the prod's tip from
the simulator, as a person sees where a prod is, with the hand settled 15 mm over each stud and over the rest, and
corrects its aim.

## Results

- Scripted demonstration, `--seed 17` (grey core leaking): 1.0, success at step 556 (27.8 s). The crank turned at
  0.99 to 1.00 rev/s; the three cores counted as tested by 9, 13 and 17 s; the needle read from the camera image was
  44, 44 and -30 degrees (infinity is +45), so it tipped the grey core's flag.
- The same demonstration, `--seed 12 --epochs 3` (black, brown, grey leaking): 1.0 in all three; the leaking core
  read -30 or -31 degrees, the healthy ones 44.
- The same moves with the crank at 0.5 rev/s: 0, also when the flag it tips is the leaking core's. No core counts as
  tested and the needle never moves, so that flag counts as tipped before testing.
- Dream-1, one run (default seed, black core leaking): 0.0. Its left hand knocked the prod out of its rest onto the
  bench beside the terminal block and its right hand went to the tester's top, not the crank; the crank never turned,
  so no core was tested and no flag moved.

## Simulation and the real test

- Rigid bodies; the crank, the handle and the flags are hinges with friction. No current flows: the needle and
  whether a core counts as tested are computed from the crank's speed and where the tip rests, against the trial's
  configured leak.
- Crank speed: the ZC25-3's rated speed is 120 r/min (2 rev/s). This arm cranks a 6 cm crank at about 1 rev/s, so in
  simulation the needle answers from 0.8 rev/s. A real ZC25 cranked at 1 rev/s would give a lower test voltage.
- The needle is drawn 5 mm wide so the 320 x 176 top camera resolves it; the prod's lead to the LINE terminal is not
  simulated; the tester stands fixed (its 2 kg on rubber feet holds it in real use).
- Estimates (the maker publishes none of these): crank radius (6 cm, scaled from the maker's photo), shaft height,
  handle size, crank drag (0.04 N m friction, 0.05 to start it, plus 0.008 N m s/rad: about 0.09 N m at 1 rev/s), scale centre 5 megohm,
  the terminal block, flags, rest and prod.

## How this differs from the nearest board tasks

- Holding a hand coffee grinder steady while turning its crank: one hand only steadies; nothing is measured.
- Probing test points or terminals with a tester and probes: the meter reads from the contacts alone; nothing needs a
  second hand to keep a generator running, and nothing is read to choose a part.

## Files

| File | What |
|---|---|
| `crank_megger_armoured_cable.py` | the task: scene, tester model (crank speed, contact, needle), `score()`, `success()`, `demo()` |
| `yam_parts.py`, `yam_joints.py` | parts and hinged mechanisms built from boxes and cylinders |
| `test_crank_megger_armoured_cable.py` | generated offline checks of the scorer |
| `task.json` | the submission's fields |
