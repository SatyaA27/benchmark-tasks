# Date a prepped food tub: peel a day label off the dispenser at its bar and press it flat onto the tub's lid

Put today's label on the lid of the food tub. Today is Wednesday.

## Revision: review feedback and what changed

Review: "The simulated labels and tabs have colours but no readable weekdays, and the lettuce's type, temperature and
handling are unspecified. ... Please add readable weekday markings to the simulated labels or dispenser so the model
can identify Wednesday from its observations, as it can with the specified real day labels. Also specify the lettuce
type, temperature and handling for repeatable trials, or replace the sealed contents with a repeatable inert ballast
of the same mass." Changes:

- Day names (see "Day names in the camera images"): every label in the scene (the first on each roll and the next
  behind it) carries its day's full name in white capitals 6 mm tall on its coloured header, as the real DissolveMark
  labels do, and a printed paper strip under each bay names its day in black capitals 36 mm tall (MON ... SUN), large
  enough to read in the wrist camera's frames. Both are visual only (no collider, no mass).
- The tub holds an inert ballast of the same mass instead of food: 350 g of plastic stuffing beads sealed in a quart
  freezer bag laid flat on its floor (setup); it is drawn as clear plastic. The instruction says "the food tub".
- Thursday and Friday now have DayMark's colours (brown and green; they were swapped).
- Flag (adhesive on the pads): for real trials wrap each gripper pad in one layer of low-tack painter's masking tape,
  replaced when it turns tacky (the lower pad touches the tab's sticky underside); DissolveMark adhesive also washes
  off in water.

Physics, colliders, masses, the adhesive rule, the scorer and the demo's motion are unchanged; the demo, the wrong
way, the Dream-1 run and the media were redone for this version.

## Scoring

Judged at the end of the run with a ruler, a photo and a feeler (a business card). The label is read as four quarters, 12.7 wide bands across its length. Stuck quarter: it lies flat on the lid's top (no corner more than 2 above it) and was pressed there with a fingertip; in a real trial, the card pushed against its edges does not slip under it and nothing lifts. Let go: both grippers' centre points at least 30 from the label. Scores: 1.0 for the Wednesday label off its roll, wholly on the lid's flat top (inside its 186 x 186 panel), let go, all four quarters stuck, held for 1 s. 0.5 + 0.125 per stuck quarter (0.5 to 0.875) for the same with fewer quarters stuck. 0.25 for the Wednesday label off its roll but not wholly on the lid's top (over its edge, on its side, on the counter, on the dispenser) or still held (a label stuck to the gripper counts as held). 0 for: any part of the Wednesday label still on its backing; the label folded onto its own adhesive (any fold past 150 degrees); any other day's label pulled more than 10 out of place or peeled up 1.5 from its backing at any time, even if put back; the tub tipped more than 20 degrees at any time or more than 30 from its mark at the end; (real trials) the dispenser moved more than 10 or a roll lifted out. Possible scores: 0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=date_a_food_tub_with_a_day_label.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Put today's label on the lid of the food tub. Today is Wednesday." \
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
| `dispenser` | DayMark IT111081 seven-slot label dispenser, 445 x 191 x 114 mm, front face at x 290 mm, centre line y -71.7 mm | fixed: floor, walls, 6 dividers, front panel, metal bar and deck (top 100 mm up), seven rolls (r 30 mm) with the next label on each strip; printed (visual only, no collider): the day strip under each bay and each next label's name |
| `label_<day>` x 7 | DayMark DissolveMark 2 x 2 in day label, first on each roll, 15 mm out past the bar | floating 4-body articulation: quarters 12.7 x 50.8 x 0.8 mm, 5 g, hinges 0.02 N m/rad (damping 0.0003, armature 5e-06 kg m^2), friction 0.6/0.5; rooted on its tab quarter; not shifted by the start jitter; header (quarters 0-1) in the day's colour, the day's name printed in white on quarter 1 (visual only) |
| `tub` | Cambro 2SFSCW 2 qt CamSquare with its SFC2 lid and a sealed bag of 350 g of plastic beads, centred (165, -170) mm | free rigid body, 0.687 kg (spread evenly through it; the real beads lie on the floor, which only lowers its centre of mass): body 184 x 184 x 98 mm, lid skirt 190 mm square, flat top panel 186 mm square 106 mm up |

Bay centres (y, mm): Monday 113.4, Tuesday 51.7, Wednesday -10, Thursday -71.7, Friday -133.4, Saturday -195.1, Sunday -256.9. Colours: Monday blue, Tuesday yellow, Wednesday red, Thursday brown, Friday green, Saturday orange, Sunday black (DayMark's day colours).

### Day names in the camera images

The strip's capitals stand 6-42 mm above the counter on the dispenser's front face, under the colour tab, each word
51 mm long and centred on its bay; the labels' names lie across the header quarter next to the tab (quarter 1).
Measured in this version's images. In the setup image (the top camera at 1280 x 704) the strip's WED is about 12 px
tall and 45 px wide, legible though the front face is in shade; the labels' names are about 4 px tall there. In the
320 x 176 frames the policy receives, the top camera shows the strip about 3 px tall, too small to read (as a real
top camera at that distance would show the real labels). The left wrist camera reads them on the way to the tab: in
the demonstration's 320 x 176 wrist frames the strip's TUE WED THU stands 22-40 px tall from 2.0 to 3.5 s, clearly
readable; at 1.5 s the names on the nearest labels are 6-10 px tall (MONDAY, TUESDAY readable), while WEDNESDAY on
its own label, further off and at a slant, is 3-4 px.

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
(another day's label taken; today's label still on its backing).

Re-run for this version (day names printed; same physics and motion): the demonstration again scored 1.0, reaching
0.25, 0.5 and 1.0 at the same control steps as before (168, 385, 665); the wrong way again scored 0; a Dream-1 run of
1,800 steps scored 0 (its hands went toward the right-hand bays; Wednesday's label stayed on its backing).

## Why the shortcuts fail

- Taking any other day's label scores 0 (another label taken), whatever is done with it. 
- Pinching the tab and lifting it straight up peels the label too (as in life); dragging it sideways off its bay or
  pulling the backing out does not count as anything but a taken or still-stuck label.
- Laying the label on the lid without pressing it scores 0.5; pressing only part of it 0.625-0.875.
- Pressing it on the lid's edge or the tub's side leaves it not wholly on the top: 0.25.
- Pushing or tipping the tub to reach it: 0.

## Estimates (to be measured on the real items)

The dispenser's inner layout (bay width, bar and deck height 100 mm, roll size), the lid's size, height and mass (80 g), the peel force from the backing (~0.1 N per 25 mm) and the pressed label's hold on the lid (1.0 N per quarter),
the frictions, and the label's bending stiffness. The label is simulated 0.8 mm thick and 5 g (real about 0.1 mm and 0.3 g).

## Files

`date_a_food_tub_with_a_day_label.py` (scene, adhesive rule, scorer, demo), `yam_parts.py` and `yam_joints.py` (rigid compound parts and passive
jointed parts), `task.json`, and the generated test `test_date_a_food_tub_with_a_day_label.py`.

## Sources

- Day labels (2 x 2 in DissolveMark, header with the day's name, write-on lines) and colours: DayMark DissolveMark
  brochure, https://www.webstaurantstore.com/documents/PDF/brochure/dissolvemark_brochure_daymark.pdf ; Wednesday
  roll (red, 250 labels, about $14), https://chefstoys.com/products/daymark-1100533-wednesday-dissolve-labels-2-x-2-roll-of-250-2226 ;
  Thursday brown, https://www.amazon.com/DayMark-Dissolvable-Label-Thursday-Brown/dp/B009YM37FK ; Friday green,
  https://www.hubert.com/product/72364/DayMark-ToughMark-GreenWhite-Day-Of-The-Week-Labels-Friday---2-Square
- Ballast: Fairfield Poly-Fil Poly-Pellets, 32 oz bag, about $20,
  https://www.amazon.com/Fairfield-PP2B-Poly-Pellets-Weighted-Stuffing/dp/B0077AQ0W8 ; Ziploc Freezer quart bags,
  7 x 7 7/16 in (177 x 188 mm), https://ziploc.com/en-us/products/bags/freezer/medium , about $6.50 for 50,
  https://www.walmart.com/ip/Ziploc-Brand-Freezer-Bags-with-Grip-n-Seal-Technology-Quart-50-Count/256636026
