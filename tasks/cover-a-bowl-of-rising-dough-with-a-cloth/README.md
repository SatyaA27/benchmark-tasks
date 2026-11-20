# Cover a bowl of rising dough with a cloth: draw it off a shelf with both hands so it hangs past the rim, off the dough

Cover the bowl of dough with the cloth so it can rise.

## Scoring

Judged at the end of the run from the top-camera video, a top photo, a side photo through the clear bowl with a ruler stood behind it, and a flexible tape. A part of the rim is covered when, in the top photo, the cloth hides it. Overhang: along the cloth from the rim's outer edge to the nearest cloth edge, where it is shortest. Score 0 if either gripper still holds the cloth or is within 2 cm of it, the cloth is still moving (any point more than 5 mm in the last 5 s; in simulation any node more than 1 cm in the last second), no part of the rim is covered (doing nothing scores 0), the bowl tipped more than 15 degrees at any time, or the bowl's centre moved more than 2 cm. Otherwise the lowest of: (1) rim covered: 1.0 if all of it; 0.5 if at least 75 % but not all; 0.25 if under 75 %. (2) shortest overhang: 1.0 if at least 2 cm; 0.75 if at least 1 cm and under 2 cm; 0.5 if under 1 cm. (3) bowl moved: 1.0 if at most 1 cm; 0.75 if more than 1 cm. (4) dough: 1.0 if the cloth does not touch the dough and its lowest point inside the rim is at most 1.5 cm below the rim's top; 0.75 if it does not touch but hangs lower; 0.5 if it touches. (5) shelf: 1.0 if no part of the cloth lies on the shelf insert's top; 0.25 if some does. Simulation checks the rim every 5 degrees; touching = a cloth node within 2.75 mm of the dome.

How `score()` reads each condition (bowl frame: origin at the centre of the bowl's foot, z up its axis; cloth: its
1369 node positions, `objects["cloth"].nodes`):

- **Rim covered:** 72 points on the rim's top centre line (67.8 mm from the axis, 55 mm up), one every
  5 degrees. A point is covered when a cloth triangle lies over it (seen from above) no lower than 3 mm under the rim's
  top. The fraction covered is band (1).
- **Overhang:** at each covered point, the cloth's own (flat, unstretched) coordinates of the covering point give the
  distance along the cloth to its nearest edge; less half the wall (2.2 mm) it is the overhang from the
  rim's outer edge. The smallest over the covered points is band (2).
- **Bowl moved / tipped:** the bowl's foot centre, horizontal distance from where it stood at the first step (kept as a
  running maximum); tipped = its axis more than 15 degrees from vertical at any step.
- **Dough:** the dome is a spherical cap 80 mm across and 26 mm high on the inside of the bottom; touching = a
  cloth node within half the cloth's thickness + 1.5 mm (2.8 mm) of it. Sag = the rim's top minus the
  lowest cloth node inside the opening (more than 3 mm inside the inner wall); full credit at most 15 mm.
- **On the shelf:** any cloth node over the shelf insert's top (beyond its front edge, within its width) and no more
  than 4 mm under the top's height.
- **Let go:** no contact-gated hold is on and both grippers' tool points are at least 20 mm from every cloth
  node. **Still:** no cloth node has moved more than 10 mm over the last 1 s (the scorer keeps
  the nodes of the last second; one step's node velocities carry solver jitter).
- `success()` is a score of 1.0; the episode ends once it has held for 1 s.

## Run it

```
inspect-robots run --embodiment yam_arena -E task=cover_a_bowl_of_rising_dough_with_a_cloth.py --policy yam_scripted   # scripted demonstration
inspect-robots run --embodiment yam_arena -E task=cover_a_bowl_of_rising_dough_with_a_cloth.py --policy dream           # model run
```

Run limit 1,200 control steps (20 Hz, 60 s). The scripted demonstration lasts 21.1 s
(422 steps). About 7 control steps per second on an L40S: a full run takes about three minutes after
start-up. No seed is needed: the start is fixed (`-E object_jitter` moves the bowl up to 10 mm).

## What is different

Unlike folding or storing towels, the cloth is carried spread by two hands and released so it settles centred over an
object without touching what is inside.

## Scene and real objects

Millimetres from the midpoint between the arm bases: x forward, y to the robot's left, heights above the counter. 1) Shelf insert: IKEA VARIERA, white powder-coated steel, 320 wide (y) x 280 deep (x) x 160 high, a perforated top on four legs, open at the front; about 0.9 kg (estimate). Front edge at x = 250, centred on y = 0; its front feet on two tape marks. Top sheet 1.5 thick with 12 deep turned-down front and back edges, legs 8 x 8 (estimates). 2) Bowl: Duralex Lys 14 cm, clear tempered glass, 140 across the rim, 55 high, 500 ml; foot 70 across, wall 4.5, bottom 6 (estimates that give the 500 ml); about 270 g (estimate). Foot centred at (150, 0), so its far rim is 30 from the shelf. 3) Dough stand-in: one 4 oz (112 g) can of white Play-Doh pressed into the bottom as a centred dome 80 across and 26 high, its top 23 below the rim (shape it by ruler). 4) Cloth: one Burt's Bees Baby organic cotton woven terry washcloth, Blue Solid, 254 x 254 (10 x 10 in), about 29 g and 2.5 thick (estimates). It lies flat and square on the shelf, smooth face up, hems parallel to x and y, centred on y = 0, its near edge 30 past the shelf's front edge (x = 220 before it droops); the lip's tip droops about 8 below the shelf top (estimate). Start: both arms at their ready pose, grippers open. Frictions are estimates (cloth 1.5 in the simulator, shelf 0.35, glass 0.45).

| Object | Real product | Source |
|---|---|---|
| Shelf insert | IKEA VARIERA shelf insert, white, 32 x 28 x 16 cm, steel, 601.366.23 | https://www.ikea.com/ca/en/p/variera-shelf-insert-white-60136623/ |
| Bowl | Duralex Lys bowl 14 cm / 500 ml, tempered glass, 14 x 5.5 cm, T542025AF | https://www.kitchenshop.eu/en/glass-bowl-14-cm-500-ml-lys-duralex |
| Cloth | Burt's Bees Baby Organic Cotton Woven Terry Solid Washcloths (3 pack), 10 x 10 in, Blue Solid | https://burtsbeesbaby.com/products/organic-cotton-woven-terry-solid-washcloths-3-pack-cloud ; https://www.babylist.com/gp/burt-s-bees-baby-organic-cotton-woven-terry-solid-washcloths-3-pack/71736/2309838 |
| Dough stand-in | Play-Doh single can, white, 4 oz (112 g) | https://www.kroger.com/p/play-doh-single-can-white/0063050946340 |

Cost: about USD 7 (shelf), 4 (bowl), 12 (three cloths), 3 (Play-Doh). Play-Doh contains wheat; it never touches the
grippers (the hands hold only the cloth's dry edge).

**Estimates** (not published; how to check one in a minute): shelf mass ~0.9 kg (kitchen scale), its sheet 1.5 mm,
turned-down edges 12 mm, legs 8 x 8 mm (ruler); bowl foot 70 mm, wall 4.5 mm, bottom 6 mm, mass 273 g (ruler,
kitchen scale; the model's inside holds 497 ml against the published 500); cloth mass 29 g and thickness
2.5 mm (kitchen scale, ruler on a folded stack); the lip's droop: lay the cloth with 30 mm over the shelf edge
and read the tip's height below the top with a ruler (estimate 8 mm); all frictions.

## How it is simulated

The cloth is true PhysX cloth (1,369 nodes); its bend stiffness makes the 30 mm lip droop about 8 mm at rest, as estimated for the real cloth. While both pads press the cloth it is pinned to the gripper (PhysX cloth has no static friction); opening releases it, and nothing else does. Once let go, its hem may tremble a few mm for seconds, so stillness is judged over 1 s. The dough is a rigid dome in the bowl; the shelf is fixed. Masses, thicknesses, droop and frictions are estimates.

- **Cloth** (`cloth_kit.py`, a reusable module): a PhysX surface deformable sheet 254 x 254 mm, 1369
  nodes about 7 mm apart, thickness 2.5 mm, 29 g (density 179.8 kg/m3), Young's modulus
  1e+06 Pa, Poisson 0.25, bend stiffness 4.5e+03, damping 1 / bend damping 5, friction 1.5,
  rest / contact offset 1.2 / 5.2 mm, 24 solver iterations, self-collision on. It starts
  flat with 30 mm over the shelf edge and droops under gravity before the robot moves.
- **Hold:** `cloth_kit.GripRule` (`ck.use_grip_rule("cloth")`, every control step, any policy) attaches a hand's cloth
  nodes only while both fingers have stopped on the cloth (under 3 mm/s), the drives press it with at least 0.5 N and
  at least 3 nodes lie between the pads, touching each, for 2 checks in a row; opening or easing the squeeze releases it.
- **Bowl:** one rigid body of 98 convex pieces (foot, 24 x 3 wall pieces, the dough dome), mass
  385 g (glass 273 g estimate + Play-Doh 112 g). Wall 4.5 mm, so the cloth (contact offset
  5.2 mm) cannot pass through the rim.
- **Shelf insert:** fixed boxes (top sheet, turned-down front and back edges, four legs); perforations not modelled.

## Demonstration

Both hands, 20 cm apart, level with their fingers one above the other, slide the lower pad 6 mm under the lip's tip
(read from the settled cloth), close while coming down so the lower pad barely moves, lift 15 mm, draw the cloth back
toward the robot while pitching down and leaning the wrists outward (they would touch at 20 cm otherwise), bring the
held edge to 25 mm above the counter 60 mm short of the near rim, wait, open and rise.

Phases (s): approach 0.0-2.5, in 2.5-4.0, close 4.0-5.4, settle 5.4-6.0, lift 6.0-7.0, draw 7.0-9.0, over 9.0-11.0, lower 11.0-13.0, pull 13.0-15.0, wait 15.0-16.5, open 16.5-17.1, away 17.1-18.6, rest 18.6-21.1.

**Scripted run (simulation, no jitter): 1.0**, first at 19.4 s. Both holds took the lip at 5.3 s and let go at 16.5 s; the cloth slid off the shelf onto the bowl and covered the rim all round. Three more runs of the same code with logging: 1.0 each, shortest overhang 40-46 mm, no sag, bowl moved 0 mm.

**Wrong way (one hand takes the lip, the other stays parked): 0.75.** The cloth reaches the bowl skewed and hangs only 12 mm past the rim at its shortest (band 2).
