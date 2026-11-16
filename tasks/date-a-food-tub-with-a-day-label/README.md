# Date a prepped food tub: peel a day label off the dispenser at its bar and press it flat onto the tub's lid

Put today's label on the lid of the tub of cut lettuce. Today is Wednesday.

## Scoring

Judged at the end of the run with a ruler, a photo and a feeler (a business card). The label is read as four quarters, 12.7 wide bands across its length. Stuck quarter: it lies flat on the lid's top (no corner more than 2 above it) and was pressed there with a fingertip; in a real trial, the card pushed against its edges does not slip under it and nothing lifts. Let go: both grippers' centre points at least 30 from the label. Scores: 1.0 for the Wednesday label off its roll, wholly on the lid's flat top (inside its 186 x 186 panel), let go, all four quarters stuck, held for 1 s. 0.5 + 0.125 per stuck quarter (0.5 to 0.875) for the same with fewer quarters stuck. 0.25 for the Wednesday label off its roll but not wholly on the lid's top (over its edge, on its side, on the counter, on the dispenser) or still held (a label stuck to the gripper counts as held). 0 for: any part of the Wednesday label still on its backing; the label folded onto its own adhesive (any fold past 150 degrees); any other day's label pulled more than 10 out of place or peeled up 1.5 from its backing at any time, even if put back; the tub tipped more than 20 degrees at any time or more than 30 from its mark at the end; (real trials) the dispenser moved more than 10 or a roll lifted out. Possible scores: 0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=date_a_food_tub_with_a_day_label.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Put today's label on the lid of the tub of cut lettuce. Today is Wednesday." \
  --scorer task_score --max-steps 1800 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial, replace `--policy yam_scripted` with `--policy dream`. `--scorer task_score` reports the score at
the end of the run. Each trial allows 1,800 control steps (90 seconds at 20 Hz). With
`-E object_jitter=0.01` and `--seed N` the tub starts up to 10 mm off its mark in x and y; the dispenser and the
labels never move. The scripted demonstration takes 39.0 s, the last 3 s holding still.
`success()` requires a score of 1.0, held for 1 s.

`pytest` in this folder runs the generated offline checks (no GPU): the scorer on the start, the goal and the named
partial and shortcut layouts.

## How `score()` reads each condition

World frame: metres, x forward, y left, z up; the tub's frame for the lid.


- **Quarters:** the label is four 12.7 mm bands across its length (the bodies `base`, `q1`, `q2`, `q3`; `base` is the
  tab that stands out past the bar).
- **Off its roll:** no quarter is on the backing any longer (see the adhesive rule below).
- **Wholly on the lid's top:** every corner of every quarter's underside lies inside the lid's 186 mm square top
  panel and no more than 2 mm above it (106 mm above the counter).
- **Stuck:** a quarter whose state is stuck at the end (pressed and not peeled up again).
- **Let go:** both grippers' centre points at least 30 mm outside the label's bounding box.
- **Folded:** any of the label's three hinges past 150 degrees at any time.
- **Another label taken:** another day's label moved more than 10 mm at any time, or one of its quarters peeled
  1.5 mm off the deck.
- **Tub:** tilted more than 20 degrees at any time, or its origin more than 30 mm (in x, y) from its mark.
- `success()` is `score() >= 1.0`, held for 1 s. `SCORE_OVER = "end"`.

## The adhesive rule (simulated each control step)

A label's adhesive cannot be simulated by contact, so the task file applies it as forces after every control step
(`adhesion_step`), the same rule for all seven labels:

- **On the backing:** the three quarters on the deck start on the backing (the tab quarter, standing out past the bar,
  has none under it) and each is pulled onto the deck by 0.4 N (peeling one up
  takes about 0.2 N, the label stock's ~0.1 N per 25 mm). It leaves the backing for good once its middle has
  been drawn to within 2 mm of the bar's edge (where the backing turns down and the label goes straight on) or a
  corner of it has been peeled 1.5 mm up.
- **On the lid:** a quarter off the backing that lies flat on the lid (every underside corner within
  -0.8..1.2 mm of the lid's top and within 6 degrees of its plane) while a fingertip touches its top face
  (a point just inside either finger's tip within 1.5 mm above to 2.5 mm into it) becomes stuck: it is
  pulled onto the lid by 1.0 N, and the lid is pulled back by the same force, until a corner is peeled
  2 mm up. Stuck quarters stay stuck when the hand goes.
- The adhesive never sticks to the gripper pads, the counter or the tub's sides (a label laid there is simply not
  stuck).

## Scene

| Object | What it is | Simulated as |
|---|---|---|
| `dispenser` | DayMark IT111081 seven-slot label dispenser, 445 x 191 x 114 mm, front face at x 290 mm, centre line y -71.7 mm | fixed: floor, walls, 6 dividers, front panel, metal bar and deck (top 100 mm up), seven rolls (r 30 mm) with the next label on each strip |
| `label_<day>` x 7 | DayMark DissolveMark 2 x 2 in day label, first on each roll, 15 mm out past the bar | floating 4-body articulation: quarters 12.7 x 50.8 x 0.8 mm, 5 g, hinges 0.02 N m/rad (damping 0.0003, armature 5e-06 kg m^2), friction 0.6/0.5; rooted on its tab quarter; not shifted by the start jitter |
| `tub` | Cambro 2SFSCW 2 qt CamSquare with its SFC2 lid and cut lettuce, centred (165, -170) mm | free rigid body, 0.687 kg: body 184 x 184 x 98 mm, lid skirt 190 mm square, flat top panel 186 mm square 106 mm up |

Bay centres (y, mm): Monday 113.4, Tuesday 51.7, Wednesday -10, Thursday -71.7, Friday -133.4, Saturday -195.1, Sunday -256.9. Colours: Monday blue, Tuesday yellow, Wednesday red, Thursday green, Friday brown, Saturday orange, Sunday black.

## Demo (39.0 s)

The scripted demo reads where the tub stands, where the label sits in the pinch, and where each quarter lies before
it is pressed (scripted demos may read simulator state); it never moves an object.

| t (s) | arm | step |
|---|---|---|
|  0.0 | left | hand up, fingers level |
|  2.5 | left | in front of the tab |
|  4.5 | left | fingers either side of the tab |
|  6.0 | left | pinch the tab |
|  7.1 | left | peel: draw it straight out over the bar |
|  9.6 | left | lift it clear |
| 11.0 | left | turn it to point at the tub |
| 13.0 | left | carry it over the lid |
| 14.5 | left | settle over the lid |
| 15.0 | left | lower the far end onto the lid |
| 16.5 | left | press it down a little |
| 17.0 | left | let go, rising |
| 17.8 | left | slide the finger out from under the tab |
| 19.3 | left | rise clear |
| 20.3 | left | park |
| 21.3 | right | over the far quarter |
| 23.2 | right | dab the far quarter |
| 24.4 | right | lift off |
| 24.9 | right | over the next quarter |
| 25.9 | right | dab the next quarter |
| 27.1 | right | lift off |
| 27.6 | right | over the third quarter |
| 28.6 | right | dab the third quarter |
| 29.8 | right | lift off |
| 30.3 | right | over the tab quarter |
| 31.3 | right | dab the tab quarter |
| 32.5 | right | lift off |
| 33.0 | right | rise clear |
| 34.0 | right | park |

## Checked in simulation

In the YAM Arena simulation (PhysX at 240 Hz, 20 Hz control) the scripted demonstration scores 1.0. Drawing the tab
straight out takes quarters 1, 2 and 3 off the backing at 7.6, 8.0 and 8.4 s (score 0.25 from 8.4 s); the label
lies flat on the lid by 18.8 s and is let go (0.5 from 19.3 s); the four dabs stick its quarters at 23.9, 26.5, 29.2
and 31.9 s; 1.0 from 33.3 s, success at 34.2 s. The hand loads read from the arms' joint errors stay small:
under 2 N drawing the label off its backing, under 4 N pressing it down. With the tub started up to 10 mm off its mark
(`-E object_jitter=0.01`, three seeded trials) the demonstration also scored 1.0 each time; it reads where the tub
stands.

A wrong way, run in the simulation: the same demonstration aimed one bay to the left takes Tuesday's label. Tuesday's
label comes off its backing and is laid and pressed on the lid; Wednesday's stays on its roll. It scores 0 throughout
(another day's label taken; today's label still on its backing). A Dream-1 run of 1,800 steps scored 0: it did not
draw Wednesday's label off its backing.

## Why the shortcuts fail

- Taking any other day's label scores 0 (another label taken), whatever is done with it. 
- Pinching the tab and lifting it straight up peels the label too (as in life); dragging it sideways off its bay or
  pulling the backing out does not count as anything but a taken or still-stuck label.
- Laying the label on the lid without pressing it scores 0.5; pressing only part of it 0.625-0.875.
- Pressing it on the lid's edge or the tub's side leaves it not wholly on the top: 0.25.
- Pushing or tipping the tub to reach it: 0.

## Estimates (to be measured on the real items)

The dispenser's inner layout (bay width, bar and deck height 100 mm, roll size), the lid's size and height, the lettuce's
mass, the peel force from the backing (~0.1 N per 25 mm) and the pressed label's hold on the lid (1.0 N per quarter),
the frictions, and the label's bending stiffness. The label is simulated 0.8 mm thick and 5 g (real about 0.1 mm and 0.3 g).

## Files

`date_a_food_tub_with_a_day_label.py` (scene, adhesive rule, scorer, demo), `yam_parts.py` and `yam_joints.py` (rigid compound parts and passive
jointed parts), `task.json`, and the generated test `test_date_a_food_tub_with_a_day_label.py`.
