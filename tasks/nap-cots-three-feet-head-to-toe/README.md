# Set out nap cots three feet apart, head to toe

Before nap time a childcare worker sets the cots out so that every cot is at least 3 ft from every other on every
side and neighbouring children sleep head to toe. Here, at 1:10 scale, four cots start on storage strips outside a
nap mat, two on each side, and four pillows lie in front of the arms. The robot has to set the cots out in a row on
the mat, square to it and at least 9.2 cm (3 ft at 1:10) apart, and put a pillow on one end of each cot, alternating
ends along the row. Each arm can reach only its own side's cots and pillows, and the two middle cots, one placed by
each arm, have to be spaced and alternated relative to each other, so a full score takes both arms.

## Run a trial

On a machine with the official simulator (`yam-isaac-lab-arena`'s `setup.sh`, or the Brev launchable), with a Dream
API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=nap_cots_three_feet_head_to_toe.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set the two blue cots on your left and the two yellow cots on your right out in a row on the green mat, long sides pointing away from you and at least 9.2 cm apart, then put one pink pillow on an end of each cot, alternating ends from cot to cot so neighbours lie head to toe." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/nap-cots-dream

# The scripted demo, which reaches full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=nap_cots_three_feet_head_to_toe.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set the two blue cots on your left and the two yellow cots on your right out in a row on the green mat, long sides pointing away from you and at least 9.2 cm apart, then put one pink pillow on an end of each cot, alternating ends from cot to cot so neighbours lie head to toe." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/nap-cots-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/nap-cots-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. The trial ends at 1.0 once all four cots are set out, spaced and head to
toe, and everything has been still for 2.5 s; otherwise it scores the partial credit below at the last step.

Offline checks of the scorer and the demo's reach (no GPU): `pytest test_nap_cots_three_feet_head_to_toe.py` in an
environment with `yam_arena` installed.

## Scoring

**Tilt** is the angle between a box's top or bottom face and level: the cots and pillows are plain boxes, so one
lying upside down counts as flat. A cot is **set out** when it is tilted at most 10 degrees, its bottom is within
3 mm of the mat's top, its whole outline is inside the mat, and its long side is within 20 degrees of front to back.
A pillow is **on a cot** when its centre is inside the cot's outline, its bottom from 1 cm below to 3 mm above the
cot's top (a pillow held higher is not on it) and it is tilted at most 20 degrees; it is **on an end** when its
centre is at least 3 cm from the cot's centre along the cot's length (the far end, away from the arms, or the near
end). A cot's **outline** is its rectangle seen from above; a cot tilted more than 10 degrees counts as its
world-aligned bounding box seen from above. The set-out cots, taken from the robot's left to its right, form the
**row**, which sets the neighbours for the head-to-toe credit. For the heads-side-by-side rule, **neighbours** are
adjacent cots, left to right, among all cots whose centres are over the mat, set out or not. A cot still in the
gripper counts where it is.

| Credit | For |
|---|---|
| 0.05 | each cot set out |
| 0.1 | each set-out cot whose closest gap to every other cot (one still in the gripper included), outline edge to outline edge, is at least 9.2 cm (3 ft at 1:10) |
| 0.1 | each set-out cot with exactly one pillow, on an end, whose neighbours in the row (at least one) each have exactly one pillow on the opposite end |
| 0 overall | any cot or pillow off the table |
| 0 overall | any cot tilted more than 45 degrees (on its side or end) |
| 0 overall | a cot stacked on another: their outlines overlapping seen from above, its bottom from 1 cm below to 5 cm above the other's top (on it, or on one or two pillows on it), even while held |
| 0 overall | a cot lying on a pillow: their outlines overlapping seen from above, the cot's bottom within 1 cm of the pillow's top, even while held |
| 0 overall | two cots with their centres over the mat and their bottoms less than 3 cm above the mat's top (even while held) less than 3 cm (1 ft at 1:10) apart edge to edge: pushed together |
| 0 overall | two neighbouring cots (adjacent among all cots over the mat, set out or not) each with one pillow on the same end: heads side by side |

A cot carried more than 3 cm above the mat and more than 5 cm above any cot below it counts for neither the stacking
nor the pushed-together rule.

All four cots set out, spaced and head to toe scores 0.05 x 4 + 0.1 x 4 + 0.1 x 4 = 1.0.

On a real table, a tape measure checks the gaps between cots and the pillows' distance from each cot's centre, a
phone level app the tilts, and a photo from above shows each cot's outline inside the mat and its angle to the mat's
short side.

## The scripted demo

Both arms work at once, each on its own side. Each arm picks up its two cots from above, gripping them across their
width, and lays them on the mat with their long sides front to back, their centres 28 cm in front of the bases and,
left to right, 24 and 8 cm left of the middle (left arm) and 8 and 24 cm right of it (right arm): 10.5 cm between
neighbouring cots and 5.25 cm from the mat's ends. It lets each cot go 8 mm above the mat. Then each arm picks up its
two pillows, gripping them across their length, and sets one on an end of each of its cots, its centre 4.2 cm from
the cot's centre, letting go 8 mm above the cot: left to right on the far, near, far and near ends. Each arm then
parks above its emptied storage strip so the camera sees the row. The left arm's targets all lie at least 8 cm left
of the middle and the right arm's at least 8 cm right of it, so the tool points stay at least 16 cm apart.

## Runtime and seeds

- The demo's plan takes about 40 s of simulated time per arm; replayed offline, it reaches full score after about
  41 s, about 810 steps at 20 Hz including the 2.5 s hold, within the 2,400-step limit. On an L40S, once Isaac Sim
  has started, a scripted trial takes about 1 minute of wall time and a full Dream-1 trial 4 to 5 minutes (the first
  start on a new machine takes about 4 minutes more).
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start.
