# Rinse a training head's hair at a shampoo basin with a hand spray: scalp wet at crown and hairline, face and ears dry

**Instruction:** "Rinse the mannequin's hair with the hand spray: wet the indicator patches on the scalp at the crown (under the hair) and at the front hairline, keep the patches on the forehead and ears dry, then lay the spray back face down in the basin."

Rinsing hair with a hand spray while the person lies back over a basin happens at every salon visit, and carers do
the same in bed with an inflatable basin for people who cannot get to a shower. The water goes where the spray
points, so the spray must be held, aimed and moved by hand; at the front hairline the eyes are a finger's width
away, so the other hand is held across the forehead as a shield; at the crown the hair lies over the scalp and takes
the water, so the other hand gathers the hair and lifts it away first.

## Scoring

The trial is read at the end from one photo of five 15 x 15 mm patches of white water-contact indicator tape on the
head, taken after the pump is switched off and the hair over the crown patch is lifted with a comb. The tape turns
red permanently wherever water touches it, so the photo shows everywhere the water reached during the whole trial.

- Must wet: CROWN (on the scalp at the back of the crown, facing the robot, under the hair that runs back over it)
  and HAIRLINE (on the scalp at the middle of the front hairline). Wet = at least half of the patch red.
- Must stay dry: FOREHEAD (34 mm in front of the hairline patch, above the eyes), LEFT EAR, RIGHT EAR (outside of
  each ear). Any red fails it.
- Spray laid back: at the end the spray lies inside the basin face down, still, and neither hand's fingertips are
  within 50 mm of its handle.

| At the end | Score |
|---|---|
| Both must-wet patches wet, no red on the three must-dry patches, spray laid back | 1.0 |
| Both must-wet wet, no red on the must-dry patches, spray not laid back (held, face up, outside the basin) | 0.75 |
| Exactly one must-wet patch wet, no red on the must-dry patches | 0.5 |
| Both must-wet wet, red on the forehead or an ear patch | 0.25 |
| Anything else: no must-wet patch wet, or one wet with red on the forehead or an ear | 0 |

Doing nothing scores 0 (the face-down spray wets nothing). The bands do not overlap and cover every end state. Success = score 1.0, held 1 s.

How the simulation reads the patches (`score()` and `success()` in
`rinse_hair_at_a_shampoo_basin_with_a_hand_spray.py`, the jets in `spray_kit.py`): every control step, 91 jets leave
the spray's nozzle face (a 56 mm field of holes, spreading 8 degrees at the rim); each jet is a straight ray that stops
at the first thing it meets: the hair (the tail keeps a 40 mm wide strip dry), either gripper (a 22 mm radius capsule
from the fingertips 120 mm up the hand), the head (scalp, face, ears) or the basin floor. Where a jet lands on the head
it wets a 5 mm spot, and each patch's 5 x 5 grid of points keeps every point that was ever wetted, as the tape does.
A must-wet patch needs half its grid points; a must-dry patch fails on one.

## Rig protection and run-off

- Water stays in the basin: the pump runs at 1 L/min, throttled with an inline hose valve (it fills a 1 L jug in
  about 60 s); the drain hose runs back into the pump's bucket; the operator keeps the pump switch in reach and
  switches it off if the spray leaves the basin or its face points up out of it (the trial is then scored as it
  stands). Both grippers wear nitrile glove fingers; each arm is in a clear plastic sleeve from wrist to shoulder,
  with a clear film window over the wrist camera.
- Run-off and soak-through checks before the first trial (two minutes, fresh patches after): with the tail in place,
  shield the forehead with your hand and spray only the hairline patch for 5 s from 75 mm; then spray the crown
  patch through the lying tail for 5 s. Both times the crown patch must show less than half red and the ear patches
  no red; if not, lower the flow and repeat. The half-patch threshold is there so that a trickle alone does not
  count as rinsing a patch.

## Real objects and their numbers

- Training head: a hairdressing practice head with human hair and a table clamp, for example the 16 in human-hair
  head sold for braiding and cutting practice (https://www.amazon.com/YODIDI-Mannequin-Cosmetology-Hairdresser-Practice/dp/B0F1CJJDJX). Here it lies face up without its clamp, the top
  hair trimmed to about 250 mm and combed back into one loose tail with a small elastic hair tie near its ends.
- Basin: an inflatable bed shampoo basin, 660 x 610 x 150 mm, 725 g, with a drain hose
  (https://www.northeastmob.com/product/inflatable-shampoo-basin/).
- Hand spray and water: a battery camping-shower pump with its hand shower head, the pump set in a bucket of
  lukewarm water (for example https://www.amazon.com/Ejoyous-Portable-Vehicle-Mounted-Showerhead-Backpacking/dp/B08FMQMNTY).
- Patches: 3M Water Contact Indicator Tape 5557 (white, turns red on contact with water; 0.23 mm),
  https://multimedia.3m.com/mws/media/1007611O/3m-water-contact-indicator-tape-5557-tds.pdf.

Estimates (no published figure; each checked in about a minute): the head's size (crown to chin 225 mm, 145 mm ear to
ear, 185 mm back of the head to the brow: ruler and calipers) and mass (about 1 kg: kitchen scale); the tail (24 mm
thick, 250 mm from root to ends, 63 g: ruler, scale; bending stiffness 1.5e-4 N m^2 from how far a 100 mm end droops
when held level); the width the lying tail keeps dry (40 mm: spray it once on a dry patch and look); the spray's size
and mass (handle 25 mm across, 130 mm; head 65 mm across, 25 mm thick; 120 g with water: ruler, scale); its jet
pattern (91 holes in a 56 mm field, 8 degree spread at the rim: spray onto paper from 100 mm and measure the wet
circle); the wet spot per jet (5 mm: the same paper).

## The scripted runs

The scripted demo (`demo()`): the left hand pinches the spray's handle near its centre of mass where the spray
lies (the grasp is read from the spray's pose) and lifts it. The right hand leans sideways across the head, its
closed fingertips 10 mm above the scalp between the hairline and forehead patches, as a wall. The left hand aims the
spray at the hairline patch from behind and to the robot's left (60 degrees from vertical, 75 mm away) and sways it
8 mm side to side for 4.8 s, then turns the spray down into the basin. The right hand leaves the forehead, passes up
and behind the head, opens there, goes down to the hair tail on the basin floor (fingers 94 mm open), closes to 18 mm
on the 24 mm tail between its last two links over 1 s, holds the pinch 0.6 s, lifts it and holds it up and back; the left hand rinses the crown patch from behind and to the left under the lifted hair (64 degrees from
vertical, 90 mm away, 4.8 s of sway), lays the spray down face down where it was, and lets go; the right hand lays the
hair back down. Plan length 47.9 s (958 steps).

Demo result: on the GPU, with no start shift, the demo scored 1.0: hairline patch wet, crown patch wet once the hair
was lifted (from step 625), forehead and ears dry, spray laid back face down; success at step 808 (40.4 s) of the
2,400 allowed; the run ends there, with the spray released and the right hand still holding the hair up. In two more runs with seeded start shifts of up to 10 mm, both scored 1.0.

Wrong ways (the same timing; only the hands' targets differ), run on the GPU with seeded shifts of up to 10 mm:
- No shield (the right hand never goes to the forehead): the jets aimed at the hairline also wet the forehead patch.
  In this run the lifted tail also slipped out of the pinch as the crown rinse began, so the crown stayed dry: 0
  (with the crown wet it would be 0.25).
- Top only (the hair is never lifted; the crown is rinsed through it): the tail takes the jets and the crown patch
  stays dry; hairline wet, face dry: 0.5.

Scores: Dream-1 run: 0.0 (2,400 steps); demo: 1.0

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=rinse_hair_at_a_shampoo_basin_with_a_hand_spray.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Rinse the mannequin's hair with the hand spray: wet the indicator patches on the scalp at the crown (under the hair) and at the front hairline, keep the patches on the forehead and ears dry, then lay the spray back face down in the basin." \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial, use `--policy dream`. Each trial allows 2,400 control steps
(120 s at 20 Hz). The wrong ways run with `DEMO_WAY = "noshield"` or `"toponly"` at the top of the task file.
`pytest` here runs the offline checks (no GPU): scorer layouts, demo reach and length.

## Fidelity and shortcuts

- The water is a scored model, not a fluid: straight jets, stopped by the first thing they meet. Run-off along the
  skin, water soaking through hair, drips and splashes are not modelled; in a real trial run-off can only add red, so
  the real forehead and ear checks are at least as strict.
- The hair over the crown is one capsule chain fixed at its root; the rest of the hair (over the sides) is shape only.
- The head and basin are fixed rigid bodies; the hose's pull on the spray is left out; the pump runs the whole trial.

## Files

- `rinse_hair_at_a_shampoo_basin_with_a_hand_spray.py`: scene, scorer and scripted demonstration
- `spray_kit.py`: the jets, ray tests and indicator patches
- `hair_kit.py`: the hair tail (capsule chain along a curved start shape) and the fixed head built from simple shapes
- `yam_parts.py`: compound rigid parts (the basin and the spray)
- `test_rinse_hair_at_a_shampoo_basin_with_a_hand_spray.py`: offline checks of the scorer and the demo
