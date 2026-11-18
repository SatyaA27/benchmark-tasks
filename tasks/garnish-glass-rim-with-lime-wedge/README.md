# Garnish a full glass of water with a lime wedge: push its slit onto the 2 mm rim without tipping or spilling the glass

Pick up the lime wedge by its two cut ends, turn it slit-down and push its slit onto the rim of the glass so it sits astride the rim, then let go, without tipping the glass or spilling water.

## Scoring

Judged at the end of the run from the trial video, a photo and a ruler. Seated: the glass wall is inside the slit and the wedge's lowest point is at least 7 mm below the rim's top (ruler stood on the counter against the glass); the wedge rests on the rim on its own, both grippers at least 3 cm from it, and has been still for 1 s. Spilled: in the real trial, any water on the paper or the counter, or running down the outside of the glass; in the simulation, which has no water, the glass tilted more than 10 degrees or moved sideways faster than 0.15 m/s (a jolt) at any time. Knocked over: the glass tilted more than 30 degrees at any time. Score 0: the glass was knocked over, or the wedge was never lifted 3 cm off the board (doing nothing scores 0). 1.0: seated, nothing spilled. 0.75: seated but spilled. 0.5: not seated but resting on the rim on its own and still (for example perched on the slit's mouth, or lying across the rim, its middle within 2 cm of the rim's line), nothing spilled. 0.25: every other outcome once the wedge was lifted: it fell into the drink, onto the board or counter, is still held, or rests on the rim after a spill.

How `score()` reads each condition (glass frame: origin at the centre of its base, z up its axis; wedge frame: origin
on the wedge's inner edge at mid-length, x along the wedge, z toward the peel):

- **Seated:** take the rim's top centre line (27.6 mm from the axis, 133.4 mm up) at the wedge's slit; in the
  wedge's frame it must lie inside the slit (within 2.2 mm of the slit's middle plane, between the mouth
  and the slit's root 14 mm up, inside the wedge's width), and the wedge's inner edge (its lowest line) must be at
  least 7 mm below the rim's top.
- **Resting on the rim:** not seated and not in the drink, the wedge's middle within 20 mm of the rim's line and between
  12 mm below and 30 mm above the rim's top. **In the drink:** its middle inside the glass, below the rim.
- **Let go:** both grippers' centre points at least 30 mm outside the wedge's bounding box. **Still:** over
  the last 1 s the wedge moved under 1 mm and turned under 2 degrees.
- **Spilled** (remembered for the rest of the run): the glass's axis tilted more than 10 degrees, or its base
  moved sideways faster than 0.15 m/s, measured over 0.1 s. **Knocked over:** tilted more than 30 degrees.
- **Lifted** (remembered): the wedge's middle 30 mm or more above the board's top.

`success()` is a score of 1.0. Why the shortcuts fail: the slit's cut faces press shut, so a wedge dropped or laid
on the rim stays perched at the mouth of the cut (0.5 at most); only a push of about 2.7 N
opens the cut and takes the rim in. Shoving or tipping the glass, or lifting the wedge and dropping it in the drink,
scores 0.25 or 0. Our wrong-way run lets go of the wedge slit-down 25 mm over the rim instead of pushing it on: it landed with the rim in the slit's chamfered mouth, the cut's faces never opened (0.0 mm), and it tipped 66 degrees in the slit's plane and stayed perched there, its lowest point 2 mm below the rim: 0.5.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=garnish_glass_rim_with_lime_wedge.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Pick up the lime wedge by its two cut ends, turn it slit-down and push its slit onto the rim of the glass so it sits astride the rim, then let go, without tipping the glass or spilling water." \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial use `--policy dream`. Each trial allows 2,400 control steps (120 s at 20 Hz);
the scripted demonstration takes 27.7 s. `pytest` here checks the scorer on named layouts.

## The scene

Products, sizes, the wedge's recipe and the positions are in task.json's `setup` (mm, from the midpoint between the
arm bases; x forward, y to the robot's left; heights above the counter). Both arms start at their ready pose.

- **Glass (one free body, 458 g):** Libbey 161, outer radius 31.8 at the base to
  28.6 at the rim (133.4 tall); a solid base 11.5 thick, 24 convex wall pieces 2 thick with
  a 0.4 mm chamfer each side of the rim's top, and the water (256 ml) as one convex body to
  20 mm below the rim, 0.5 mm inside the wall; the mass is spread evenly over them (centre of mass
  a few mm higher than a real glass's; the 1.1 N sideways push at the rim that tips it is the same). Friction 0.40/0.35.
- **Board:** a box 254 x 152.4 x 12.7, 467 g, friction 0.5/0.4.
- **Lime wedge (free body with two sliding links, 8 g):** an ellipsoid sector of a lime 51 across and
  60 long, 45 degrees wide, cut off 25 mm each side of its middle (the trimmed ends).
  Rigid: two flesh halves either side of the slit (3.4 mm apart below the slit's root, the slit's mouth chamfered
  4 mm at 45 degrees) and the peel bridge above the root (14 mm up); friction 0.5/0.4. The
  slit's two cut flesh faces are plates 1.4 mm thick, 0.6 mm apart at rest, each on a slide across the slit
  (travel 1.3 mm, spring 600 N/m with 1.6 mm preload, damping 2.0 N s/m, armature
  0.01 kg, 0.5 g, a 30-degree lead-in on the lower edge), friction 0.25/0.22 on glass.
  The 2 mm rim spreads each plate 0.7 mm: about 1.4 N on each face; opening the cut over the lead-ins takes about
  2.7 N of push, sliding on about 0.7 N, pulling off about 0.7 N (estimates).

## Scripted demonstration

One hand does the work; the left arm parks (it cannot reach the glass's lower body from the side). The demonstration
reads (never moves) where the wedge lies, the glass's position once, and the held wedge's pose twice over the rim,
and moves the hand so the slit is where the plan has it. The pads touch only the wedge's two trimmed ends (about 10 N of squeeze); after the lift the hand turns
40 degrees about its closing axis, so the wedge reaches the rim leaning 27.5 degrees in its slit's plane, inner half over
the drink; the lowest finger stays 7 mm above the rim's top and 13 mm above the water.

| Time (s) | Step |
| --- | --- |
| 0.0 to 2.5 | over the wedge |
| 2.5 to 4.0 | fingers down at its ends |
| 4.0 to 5.0 | pinch it by its trimmed ends |
| 5.0 to 6.2 | lift it |
| 6.2 to 8.2 | turn it slit-down |
| 8.2 to 9.7 | swing it toward the glass |
| 9.7 to 11.7 | carry it over the rim |
| 11.7 to 13.3 | line the slit up over the rim, hold |
| 13.3 to 14.7 | slit just over the rim, hold |
| 14.7 to 18.9 | push the slit onto the rim (8 steps of 0.45 s), hold |
| 18.9 to 19.5 | ease off the push |
| 19.5 to 20.3 | let go |
| 20.3 to 21.5 | lift the fingers straight up |
| 21.5 to 22.7 | back away |
| 22.7 to 24.7 | the hand to its parking point; then still |

In our recorded demonstration (no object jitter) the slit took the rim at 18.2 s and full credit came at 21.4 s: the rim 10.1 mm into the slit, each cut face pushed back 0.95 mm, the wedge leaning 27 degrees in the slit's plane; the glass never moved or tilted. The run ends there, both grippers over 3 cm clear. Three trials with every object shifted up to 10 mm: the same.

## Simulation and the real objects

Rigid bodies. No liquid: the water is a solid body in the glass (its mass and centre), so spilling is judged from the glass's tilt (>10 deg) or a jolt (>0.15 m/s). The slit's cut flesh is two sprung plates on 1.3 mm slides (estimates: ~2.7 N peak to open the cut, ~0.7 N to slide on or pull off); the alignment and the push are real. Wedge mass, glass wall and base thickness and mass, and all frictions are estimates; the paper sheet is not modelled.

- Glass: Libbey 161 Heavy Base Hi-Ball, 10.25 oz, 5 1/4 in tall, 2 1/4 in top, 2 1/2 in bottom
  (https://libbeyfoodservice.com/product/libbey/heavy-base-hi-ball/161,
  https://www.webstaurantstore.com/libbey-161-heavy-base-10-25-oz-hi-ball-glass-case/551161.html, $1.84 each).
- Lime: USDA FoodData Central 168155 "Limes, raw", 1 fruit (2 in across) = 67 g
  (https://fdc.nal.usda.gov/food-details/168155); Persian lime 1.5-2.5 in wide, 2-3 in long, seedless, thin peel
  (Morton, Fruits of Warm Climates, https://hort.purdue.edu/newcrop/morton/tahiti_lime.html).
- Board: Winco CBWT-610, 6 x 10 x 1/2 in, white polyethylene
  (https://www.bakedeco.com/product/winco-white-polyethylene-cutting-board-12-thick-6-x-10-1893, $2.85).
- Estimates, each checkable in a minute: the glass's wall (2.0 mm) and base (11.5 mm) thickness and its 200 g (ruler,
  kitchen scale); the lime's 60 mm length and the wedge's 8 g (ruler, scale); the push to seat a slit wedge and to pull
  it off (stand the glass on a kitchen scale and read the rise while pushing the wedge on); every friction. Reusable
  plastic wedges sold as props are hard and hollow and cannot grip a rim through a slit, so the trial uses a fresh
  wedge; the whole kit (glass, board, a bag of limes, a paring knife, nitrile finger-cots) costs under $40.

## Reset

About 2 minutes. Throw the used wedge away and dry the counter. Refill the glass to 255 ml (20 mm below the rim) with cold water, dry its outside, and stand it back at (220, 0) on a fresh dry sheet of white copy paper. Cut a fresh wedge to the recipe in the setup (keep limes in the fridge at about 4 C and cut each wedge no more than 30 minutes before its trial; one lime gives eight wedges) and lay it on one cut face at its start place on the board, inner edge toward the robot. Change the nitrile pad covers when they are wet. If the glass was knocked over, stop the trial, mop the plastic sheet dry and wipe the grippers before the next one. Return both arms to the ready pose.
