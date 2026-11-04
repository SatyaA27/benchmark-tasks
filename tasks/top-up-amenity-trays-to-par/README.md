# Top up two guest amenity trays to par and leave the rest on the cart

Each guest tray should end with one soap and one shampoo bottle. The left tray already has its soap and the right
tray is empty, so the robot has to add three items and leave the fourth, a spare soap, on the cart. Putting in an
extra item scores nothing, and so does leaving stock anywhere other than a tray or the cart. All the stock starts
on the cart in front of the left arm, and only the right arm reaches the right tray, so items for the right tray
are set down on the table between the arms and picked up again from there.

## Run a trial

On a machine with the official simulator (`yam-isaac-lab-arena`'s `setup.sh`, or the Brev launchable), with a Dream
API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=top_up_amenity_trays_to_par.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Top up both blue guest trays so each holds exactly one yellow soap lying flat and one green shampoo bottle standing upright, taking stock from the grey cart tray, and leave the spare stock on the cart tray." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/amenity-par-dream

# The scripted demo, which reaches full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=top_up_amenity_trays_to_par.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Top up both blue guest trays so each holds exactly one yellow soap lying flat and one green shampoo bottle standing upright, taking stock from the grey cart tray, and leave the spare stock on the cart tray." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/amenity-par-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/amenity-par-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. The trial ends with success once the score is 1.0 and every soap and bottle
has moved less than 1 cm/s (and turned less than 0.5 rad/s) for 2.5 s in a row; otherwise it scores the partial
credit below at the last step.

Offline checks of the scorer and the demo's reach (no GPU): `pytest test_top_up_amenity_trays_to_par.py` in an
environment with `yam_arena` installed.

## Scoring

An item is in a tray (a guest tray or the cart tray) when its centre is inside the tray's inner outline (or within
5 mm past it) and its bottom is between 5 mm below and 3 mm above that tray's floor. A bottle is upright when its
long axis is at most 15 degrees from vertical, standing on either end. A soap bar is upright when it lies flat on any
of its four long faces: its 3 cm or its 3.5 cm edge at most 15 degrees from vertical, so it stands 3 cm tall (on a
5 x 3.5 cm face) or 3.5 cm tall (on a 5 x 3 cm face). A soap stood on a 3.5 x 3 cm end, 5 cm tall, is not upright.

The handover zone is the table between the arms, 15 to 30 cm in front of the arm bases and within 10 cm of the
midline. An item is in transit there when it stands upright on the table (its bottom between 5 mm below and 3 mm
above the table top) with its centre in the zone.

| Credit | For |
|---|---|
| 0.25 | the left guest tray holds exactly one shampoo, upright |
| 0.25 | the right guest tray holds exactly one soap, upright |
| 0.25 | the right guest tray holds exactly one shampoo, upright |
| 0.25 | all three above, and the left guest tray also holds exactly one soap, upright (both trays at par) |
| 0 | an item in transit in the handover zone earns nothing, and doesn't zero the score while the guest trays lack at least that many of its kind (trays holding no soap, for soaps; trays holding no bottle, for bottles) |
| 0 overall | either guest tray holds two or more soaps, or two or more shampoos, in any pose |
| 0 overall | any soap or shampoo is not in a guest tray or the cart tray and not in transit as above: on the table or floor, on another item, on a wall, or with its bottom more than 3 mm above the surface under it (as when still held) |

Any soap or bottle counts, wherever it started: a tray's count is what is in it at the end. With both trays at par
no tray lacks anything, so no item may be in transit and the spare soap is necessarily on the cart tray. An item
left in the handover zone mid-route keeps the credit already earned; a spare taken off the cart beyond what the
trays lack scores 0. On a real table, a photo from above shows each item's centre inside a tray's outline and the
counts; a ruler checks that an item's bottom is on the tray floor and how tall a soap stands (3 or 3.5 cm counts,
5 cm doesn't), and a phone level app the tilt of a doubtful item.

## The scripted demo

The left arm picks one soap off the cart from above, gripping it across its 5 cm side, and sets it on the table
between the arms, 18 cm in front of the arm bases; then it does the same with the far shampoo, 26 cm in front of the
arm bases; then it stands the other shampoo in the left guest tray, 8 cm beyond the soap already there, and parks
in front of that tray. The right arm waits above its tray until the left arm has left the midline, then moves the
soap and the shampoo from the table into the right guest tray, 8 cm apart, and parks in front of it. Each item is
let go 8 mm above where it lands. The spare soap on the cart is never touched.

## Runtime and seeds

- The demo's plan takes about 41 s of simulated time (the right arm's 20 s wait, two 9.6 s pick-and-place moves,
  a 2 s park); in the offline kinematic replay, success, including the 2.5 s hold, is reached after 41.3 s (about
  830 steps at 20 Hz), well within the 1,800-step limit. On an L40S, once Isaac Sim has started, a scripted trial takes about 1 to 2 minutes of wall time
  and a full Dream-1 trial 3 to 4 minutes.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start.
