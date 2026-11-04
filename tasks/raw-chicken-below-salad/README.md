# Store the raw chicken below the salad on a two-tier fridge rack

A tray of raw chicken and a container of ready-to-eat salad go onto a two-tier rack: a low tier in front and a high
tier, 7.5 cm higher, right behind it, both open from above like steps. Food codes say raw meat goes below
ready-to-eat food so its juice can't drip onto it, so the chicken belongs on the low tier and the salad on the high
one, even though the low tier is the nearer one for both. The right arm can reach the chicken and the left arm the
salad, so a full score takes both arms, one after the other.

## Run a trial

On a machine with the official simulator (`yam-isaac-lab-arena`'s `setup.sh`, or the Brev launchable), with a Dream
API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=raw_chicken_below_salad.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the red tray of raw chicken and the green salad container on the two-tier rack, with the raw chicken tray lying flat on the lower shelf, below the salad container standing upright on the upper shelf." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/raw-chicken-dream

# The scripted demo, which reaches full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=raw_chicken_below_salad.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the red tray of raw chicken and the green salad container on the two-tier rack, with the raw chicken tray lying flat on the lower shelf, below the salad container standing upright on the upper shelf." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/raw-chicken-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/raw-chicken-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. The trial ends at 1.0 once both items have been stored (below) for 2.5 s:
the chicken flat on the low tier and the salad flat on the high tier; otherwise it scores the partial credit below at
the last step.

Offline checks of the scorer and the demo's reach (no GPU): `pytest test_raw_chicken_below_salad.py` in an
environment with `yam_arena` installed.

## Scoring

An item is stored on a tier when all of these hold:

- its centre is inside that tier's inner outline (or within 5 mm past it, leaning on a lip);
- its bottom is between 5 mm below and 3 mm above that tier's deck, so it has been set down on the deck, not held
  above it;
- it is tilted at most 10 degrees;
- it is still: moving under 1 cm/s and turning under 0.5 rad/s.

| Credit | For |
|---|---|
| 0.5 | the chicken tray stored on the low tier |
| 0.5 | the salad container stored on the high tier |
| 1.0 | both; the trial ends in success once this has held for 2.5 s |
| 0 overall | the chicken on the high tier: its centre inside the high tier's outline (or within 5 mm past it) and its bottom no lower than 5 mm below the high deck, at any height |
| 0 overall | the chicken over the salad: the two items' footprints (each one's true outline seen from above, as turned and tilted) overlap and the chicken's bottom is at or above a level 1 cm below the salad's top |

On a real table, a ruler checks that each item's bottom sits on its tier's deck and a phone level app its tilt; a
photo from above shows that each centre is inside its tier and whether the chicken's outline overlaps the salad's;
items are scored after they have come to rest.

## The scripted demo

The right arm picks up the chicken tray from above, gripping it across its 7 cm width, and lays it flat on the right
half of the low tier, 13 cm right of centre and 18 cm in front of the arm bases, letting go 8 mm above the deck, then
parks back over the tray's starting mark. The left arm waits above the salad until then, picks it up across its width,
sets it down on the left half of the high tier, 13 cm left of centre and 31 cm in front of the arm bases, letting go 8
mm above the deck, lifts until its fingertips are 1.2 cm above the container's top, and parks back over the salad's
starting mark so the camera sees the stored rack.

## Runtime and seeds

- The demo's plan takes 23.4 s of simulated time (the right arm 11.1 s, then the left arm 11.9 s after an 11.5 s
  wait). In the offline kinematic replay the trial ends in success, the 2.5 s hold included, at 23.2 s (464 steps
  at 20 Hz), well within the 1,200-step limit.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start.
