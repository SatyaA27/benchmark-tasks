# Clip a shoulder strap onto a laptop bag: squeeze each swivel snap hook open over its D-ring, the strap not twisted

Clip the shoulder strap onto the laptop bag: pick up each hook so your grip squeezes its trigger open, pass the open hook over the D-ring at its end of the bag, and let it snap shut, without twisting the strap.

## Scoring

Judged from a photo of the end of the run and the trial video. A hook is clipped on a ring when the ring's wire passes through the hook's loop and the gate is shut (back against its stop). Twisted: with both hooks left clipped, lift the strap by its middle and follow it from one hook to the other; it is twisted if the webbing turns over between them (120 degrees or more; a half turn is 180). Scores: 1.0 both rings each clipped by one of the two hooks, strap not twisted; 0.75 both clipped but the strap twisted; 0.5 one ring clipped; 0.25 no ring clipped but a hook hangs on a ring's wire with its gate open (jammed on the wire); 0 otherwise (hooks on the desk, dropped, hooked on the tab or the bag only). Doing nothing scores 0.

How `score()` reads each condition (from the simulator's link poses, the same things a photo shows):

- **Clipped:** the ring's wire (its closed centre-line: the D's curve and its straight bar) passes through the
  hook's closed loop (back, nose, tip and gate) an odd number of times, and the gate is within 4 degrees of its
  shut stop. A gate held open on the wire, or a hook merely lying beside or on the ring, is not clipped.
- **Each ring by one hook:** the two rings are clipped by the two different hooks (either hook on either ring).
- **Twisted:** the webbing's twist summed link to link from one hook's eyelet to the other's is 120 degrees or
  more either way (a half turn is 180).
- **Jammed open:** a hook's loop surrounds a ring's wire but its gate is not shut: 0.25 if no ring is clipped.

`success()` is a score of 1.0 held for 1 s. In the demonstration both hooks are clipped, gates back on their stops,
14.6 s in, and the run ends there on success with the grippers still beside the rings.

Why the shortcuts fail: the gate opens inward only, so pushing the hook's back or nose at the ring does nothing;
the wire has to come in through the open mouth, past the gate's end into the bend of the hook, before the hook is
let go. Run the wrong way (each hook let go with the wire only in the gate's mouth, short of the hook's bend), both gates snap shut, the hooks slip off the wires and come to rest beside their rings: score 0 (0.25 only while the open hooks were still held round the wires).

## Setup

Millimetres from the midpoint between the arm bases: x forward, y to the robot's left, heights above the desk (table top). 1) Bag: a laptop bag (Zign leather laptop bag, 393.7 x 266.7 x 63.5 mm) lying on its back, its top edge (zip, handles folded flat on its front) toward the robot at x 180, centred on y 0. 2) D-rings: a steel D-ring (38 mm inside width, 22 mm inside depth, 4 mm wire; estimate) on a short leather tab on each end panel, its straight bar upright, centre 30 mm from the top edge (x 210) at mid-height (z 31.75), bar centre at y +-210.9; each ring stands out square from its panel and swings on its bar if pushed. 3) Strap: a 38 mm webbing shoulder strap set to its shortest (Ytonet 52 in universal strap, pad taken off), lying flat and straight along y at x 145, its eyelet bars at y +-300; a swivel trigger snap hook at each end (BIKICOCO 1-1/2 in: 58 mm long, 38 mm eyelet, 8 mm opening, 15 g) lying flat in line with the strap, nose outward, both gates toward the back (the left hook lies on its other face). Start positions vary by up to 8 mm and 1.5 degrees for the strap.

## Real objects and sources

- Laptop bag: Zign leather laptop bag (Zalando article ZI154H01J-Q11), 15.5 x 10.5 x 2.5 in = 393.7 x 266.7 x
  63.5 mm, side D-rings on leather tabs, GBP 49:
  https://www.zalando.co.uk/zign-unisex-laptop-bag-black-zi154h01j-q11.html . Any laptop bag of this size with a
  D-ring on each end panel serves; the D-ring's size is an estimate (check with a ruler: 38 mm inside, 4 mm wire).
- Strap: Ytonet 52 in replacement laptop shoulder strap (Amazon B075ST5ZDQ): 1.5 in webbing, 50 in with its two
  metal hooks, adjustable 28-50 in, detachable pad, 4.2 oz, USD 8.99:
  https://www.amazon.com/Laptop-Shoulder-Strap-Adjustable-Briefcase/dp/B075ST5ZDQ . Set to its shortest (the
  simulated eyelet-to-eyelet run is 600 mm; check with a tape measure), pad taken off.
- Hooks: swivel trigger snap hooks of the BIKICOCO 1-1/2 in type (Amazon B0832F8595): 38 mm eyelet, 58 mm long,
  opens 8 mm, zinc alloy, 10 for USD 12.29 (15 g each):
  https://www.amazon.com/BIKICOCO-Swivel-Trigger-Lobster-Spring/dp/B0832F8595 . If the strap's own hooks are not
  trigger snaps, fit two of these. Hook thickness (6 mm), the gate's travel (33 degrees to its stop) and its spring
  (about 3 N at the trigger to start, 5 N fully open) are estimates: check with a ruler and a kitchen scale pressed
  on the trigger.
- Total cost: under USD 100.

## Simulation notes

True size. Bag: rigid, fixed (a strap pulls at most a few newtons). Webbing: 30 mm flat links joined by sprung joints (flat bend, stiff edge bend, twist); stiffness, 30 g/m and the doubled adjuster run are estimates. Hook: body, sprung gate and swivel; gate spring (about 3-5 N at the trigger), swivel friction, ring size and tab stiffness are estimates. The ring hinges on its bar only.

The webbing is 20 flat links of 30 x 38 mm (1.6 mm thick; links 3-12 doubled, the adjuster's run) joined by
sprung joints; the hook is three bodies (eyelet, body, gate) with the swivel and the gate's hinge as joints; each
D-ring is a fixed tab and a ring of 11 round-wire segments hinged on its bar. All shapes are boxes, cylinders and
capsules built here (`strap_kit.py`). The bag is rigid and does not move.

## Reset

Unclip both hooks (squeeze each trigger and lift the hook off its ring). Set the strap to its shortest and lay it flat and straight on its tape marks 145 mm in front of the bag's top edge, both hooks in line with it, nose outward, both gates toward the bag (turn the left hook onto its other face). Swing each D-ring out square from its end panel. Check the bag still lies on its back on its marks; double-sided tape under the bag keeps it from sliding when a hook pulls on its ring (the simulated bag does not move).

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=clip_shoulder_strap_onto_laptop_bag.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Clip the shoulder strap onto the laptop bag: pick up each hook so your grip squeezes its trigger open, pass the open hook over the D-ring at its end of the bag, and let it snap shut, without twisting the strap." \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

The scripted demonstration reads where each hook and ring actually are, picks each hook with a pinch across its
trigger tab and its back (the squeeze opens the gate), turns it a quarter turn, leans the hand back 25 degrees so
the fingers clear the ring's curve, brings the wire into the open mouth and past the gate's end, and lets go.
`pytest .` runs the offline scorer checks.
