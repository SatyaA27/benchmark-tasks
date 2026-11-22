# Rinse a training head's hair at a shampoo basin with a hand spray: scalp and hair tail wet, face, eyes and ears dry

**Instruction:** "Rinse the mannequin's hair with the hand spray: wet the six indicator patches on the scalp (one at the crown is under the hair) and the three tape bands on the hair tail, keep every patch on the face and ears dry, then lay the spray back face down in the basin."

Rinsing hair with a hand spray at a basin happens at every salon visit and at carers' bedside basins. The spray is
held, aimed and moved across the scalp and along the hair; the other hand shields the forehead, then lifts the hair
off the crown.

## Scoring

Read at the end from one photo of the tape, taken after the pump is switched off and the tail is lifted off the crown
with a comb (both sides of each band shown). The white water-contact tape turns red permanently wherever water touches
it, so the photo shows everywhere the water reached during the trial. Positions in mm (x forward, y to the robot's
left, z up from the table top).

Must wet (9 items):
- Six 15 x 15 mm scalp patches: HAIRLINE (406, 0, 207); TOP LEFT, TOP RIGHT (395, +-35, 187), 35 mm either side of
  the tail; CROWN (353, 0, 146), under the tail; CROWN LEFT, CROWN RIGHT (365, +-35, 146). Wet = at least half red.
- Three 15 mm bands wrapped round the tail, centred 42, 98 and 154 mm from its root (on the top of the head, down the
  back of the head, just above the basin floor). Wet = at least a quarter of the band's surface red (one side of a
  tail faces the spray).

Must stay dry (9 items; any red fails the item): a FOREHEAD STRIP 15 mm wide, about 150 mm long, across the brow
at x 440 from temple to temple (its ends at (440, +-55, 182)); each closed EYE (468, +-31, 203); each CHEEK
(515, +-42, 194); each EAR's outside (485, +-83, 111) and front (516, +-74, 111).

Spray laid back: at the end it lies inside the basin face down, still, and neither hand's fingertips are within
50 mm of its handle.

| At the end | Score |
|---|---|
| No red on the face or ears; all 9 must-wet items wet; spray laid back | 1.0 |
| No red on the face or ears; all 9 wet; spray not laid back (held, face up, outside the basin) | 0.75 |
| No red on the face or ears; 6, 7 or 8 wet | 0.5 |
| No red on the face or ears; 1 to 5 wet | 0.25 |
| Red on a face or ear item; all 9 wet | 0.25 |
| Anything else (nothing wet; or red on the face or ears with 8 or fewer wet) | 0 |

Doing nothing scores 0 (the face-down spray wets nothing). The rows do not overlap and cover every end state.
Wetting only the crown and the hairline scores 0.25. Success = score 1.0, held 1 s.

In simulation (`score()`, `patch_states()`, `update()`; jets in `spray_kit.py`): every control step 91 jets leave the
nozzle face (56 mm hole field, 8 degree rim spread); each straight ray stops at the first thing it meets: the tail (it
keeps a 40 mm strip dry), a gripper (22 mm capsule from the fingertips 120 mm up the hand, plus its 142 x 68 mm palm
block), the head or the basin floor, and wets a 5 mm spot there. Each patch's 5 x 5 grid and each band's 12 x 3 ring of points keeps every point
ever wetted. A scalp patch needs half its points, a band a quarter; a face or ear item fails on one.

## Rig protection, run-off and the dry check

- Water stays in the basin: the pump runs at 1 L/min, throttled with an inline hose valve (fills a 1 L jug in about
  60 s); the drain hose runs back into the pump's bucket; the operator keeps the pump switch in reach and switches it
  off if the spray leaves the basin or its face points up out of it (the trial is then scored as it stands). Both
  grippers wear nitrile glove fingers; each arm is in a clear sleeve from wrist to shoulder, with a clear film window
  over the wrist camera.
- Run-off and soak-through checks before the first trial (two minutes, fresh tape after): with the tail in place,
  shield the forehead with your hand and spray only the hairline patch for 5 s from 70 mm; then spray the crown patch
  through the lying tail for 5 s. Both times the crown patch must show less than half red and no face or ear item any
  red; if not, lower the flow and repeat.
- Drying between trials (reset): blot head and hair with a towel, then a hair dryer on its warm (not hot) setting,
  15 cm away, over the tail and the scalp for 3 minutes (estimate). Dry check: press a fresh 20 x 20 mm piece of the
  tape for 5 s on the tail at each band mark, on the scalp at the crown and the hairline, and on the forehead; every
  piece must stay white before the new patches and bands go on. Any red: dry 1 more minute and check again with new
  pieces. The whole reset takes about ten minutes (estimate).

## Real objects and their numbers

- Training head: a hairdressing practice head with 16 in human hair (https://www.amazon.com/YODIDI-Mannequin-Cosmetology-Hairdresser-Practice/dp/B0F1CJJDJX); here it
  lies face up without its clamp, the top hair trimmed to about 250 mm and combed into one loose tail with a small
  elastic hair tie near its ends.
- Basin: an inflatable bed shampoo basin, 660 x 610 x 150 mm, 725 g, with a drain hose
  (https://www.northeastmob.com/product/inflatable-shampoo-basin/).
- Hand spray and water: a battery camping-shower pump with its hand shower head, in a bucket of lukewarm water (for
  example https://www.amazon.com/Ejoyous-Portable-Vehicle-Mounted-Showerhead-Backpacking/dp/B08FMQMNTY).
- Tape: 3M Water Contact Indicator Tape 5557 (white, turns red on contact with water, not on humidity; 0.23 mm),
  https://multimedia.3m.com/mws/media/1007611O/3m-water-contact-indicator-tape-5557-tds.pdf.

Estimates (no published figure; each checked in a minute): head size (crown to chin 225 mm, 145 mm ear to ear, 185 mm
back of head to brow: calipers) and mass (1 kg: scale); tail (24 mm thick, 250 mm, 63 g; stiffness 1.5e-4 N m^2 from
the droop of a 100 mm end held level); the 40 mm dry width under the tail (spray a dry patch once); spray size and
mass (ruler, scale); its jets (91 holes, 56 mm field, 8 degrees; 5 mm wet spot: spray paper from 100 mm); drying and
reset times.

## The scripted runs

The scripted demo (`demo()`): the left hand pinches the spray's handle where it lies and lifts it. The right hand leans
sideways across the head as a wall, its closed fingertips 7 mm above the scalp 21 mm in front of the hairline patch.
The spray rinses the hairline from behind-left, aimed 14 mm behind it (30 degrees from vertical, 66 mm, 6 mm sway for
4.8 s), then the tail over the back of the head band by band (55 degrees, 85 mm), then points into the basin. The right
hand leaves the forehead, pinches the tail on the basin floor between its last two links (18 mm on 24 mm) and lifts it
up, back and to the right; the spray turns beside the head, left of the lifted tail, rinses the crown under it (64
degrees, 90 mm, 12 mm sway for 4.8 s), is laid back face down and let go; the right hand lays the hair down. Plan
50.9 s (1,018 steps).

Demo result: on the GPU, with no start shift, the demo scored 1.0: all six scalp patches and three bands wet, face
and ears dry, spray laid back; success at step 868 (43.4 s) of the 2,400 allowed. Three more runs with seeded start
shifts of up to 10 mm all scored 1.0.

Wrong ways (the same timing; only the targets differ), run on the GPU with seeded start shifts of up to 10 mm:
- No shield (no hand at the brow, the spray aimed at the hairline patch itself from 40 degrees): the forehead strip
  turns red, all 9 must-wet items wet: 0.25.
- Top only (the tail is never lifted): the crown patch under it stays dry, everything else wet, face dry: 0.5.
- Spots (only the crown and the hairline aimed at, 8 mm sway, no rinse along the tail): the 77 mm wide spray still
  wets the six scalp patches, but the middle band stays dry: 0.5. Only the two spots wet scores 0.25 (`pytest`).
With the demo's aim (14 mm behind the hairline, 30 degrees) a run without the shield also kept the face dry in
simulation; the shield is the margin for a less careful aim, as the no-shield run shows.

Scores: Dream-1 run: 0.0 (2,400 steps); demo: 1.0

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=rinse_hair_at_a_shampoo_basin_with_a_hand_spray.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Rinse the mannequin's hair with the hand spray: wet the six indicator patches on the scalp (one at the crown is under the hair) and the three tape bands on the hair tail, keep every patch on the face and ears dry, then lay the spray back face down in the basin." \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial, use `--policy dream`. Each trial allows 2,400 control steps (120 s at 20 Hz). The wrong ways run
with `DEMO_WAY = "noshield"`, `"toponly"` or `"spots"` at the top of the task file. `pytest` here runs the offline
checks (no GPU): scorer layouts, demo reach and length.

## Fidelity and shortcuts

- The water is a scored model, not a fluid: straight jets, stopped by the first thing they meet. Run-off, water
  soaking through hair, drips and splashes are not modelled; in a real trial run-off can only add red, so the real
  face and ear checks are at least as strict.
- The tail is one capsule chain fixed at its root; the hair over the sides is shape only. A band's points are laid
  out from its link's axis and the vertical (the tail's twist is not tracked).
- The head and basin are fixed rigid bodies; the hose's pull on the spray is left out; the pump runs the whole trial.

## Files

The task file (scene, scorer, demo); `spray_kit.py` (jets, patches, bands); `hair_kit.py` (tail, head); `yam_parts.py`
(basin, spray); the test file (offline checks).
