# Cut a single-serve pizza in half with a pizza wheel pressed hard through the centre, one hand holding the board

**Instruction:** "Cut the pizza in half. Hold the pizza board still by its handle with one hand. With the other, lift the pizza wheel out of its block, press it down on the pizza and roll it straight across through the centre, from one crust edge right over the other, pressing hard enough to cut right through. Put the pizza wheel back in its block, then draw one half about 3 cm away from the other."

A small single-serve pizza (about 6 in) lies on a round serving board with a handle. One hand holds the board by its handle; the
other lifts a palm-held pizza wheel out of a utensil block, presses it down on the pizza and rolls it straight across
through the centre, crust edge to crust edge, so the pizza is halved to share, puts the wheel back and draws one
half away from the other.

## Scoring

Score 0 if the cut line is not cut through anywhere from the centre to a crust edge, or the board ends tipped over or off the table. Otherwise the lowest of these numbers. Cut (the line through the centre is two seams, centre to each crust edge; a stretch is cut only where the wheel's edge rolls over it within 6 mm of the line pressing down at least 8 N, so the halves are within 45/55 of equal): both seams cut crust to crust = 1.0; one = 0.5 (the halves stay joined). Half drawn (how far each half has moved from its place at the end; a half can move only once the whole line is cut through): one half 30 mm or more away with the other within 10 mm = 1.0; 10-30 mm = 0.75; otherwise = 0.5. Board held still (largest shift of its middle and largest turn): within 15 mm and 5 degrees = 1.0; 30 mm and 10 degrees = 0.75; 60 mm and 20 degrees = 0.5; more = 0.25. Pizza wheel let go = 1.0; still in the hand = 0.75. Doing nothing scores 0. Success: 1.0 held for 1 s. In a real trial: a half comes away only if the cut goes right through crust to crust, so the end photo shows the gap of the drawn half (ruler); the board against its tape outline (ruler; 5 degrees is 11 mm at the board's rim); the video shows the wheel let go.

| Outcome | Score |
|---|---|
| Whole line cut crust to crust, one half drawn 30 mm or more (the other within 10 mm), board within 15 mm / 5 degrees, wheel let go | 1.0 |
| As above, but the half drawn only 10-30 mm, or the board moved up to 30 mm / 10 degrees, or the wheel still held | 0.75 |
| Only one seam cut (the pass stopped short of a crust: halves joined), or no half drawn, or the board moved up to 60 mm / 20 degrees | 0.5 |
| Board shoved more than 60 mm / 20 degrees | 0.25 |
| Nothing cut through (doing nothing, a light pass, a pass more than 6 mm off the centre line), board tipped or off the table | 0 |

(The score is the lowest band reached on any line.)

**Real trial.** From the video and an end photo with a ruler: a half comes away only if the cut goes right through
crust to crust, so the drawn half's gap shows the cut; a cut left short shows as the halves staying joined and the
whole pizza and board moving together. The board against its tape outline (5 degrees is 11 mm at the board's 127 mm
rim). The video shows the wheel let go.

## The real objects and their numbers

- Pizza wheel: Zyliss Pizza Wheel (palm-held; the handle sits straight above the wheel over a guard), SKU E910081,
  13.4 x 2.9 x 13 cm, US$19.95 (https://us.zyliss.com/products/zyliss-pizza-wheel ); the earlier model 30810 is
  listed at 0.29 lb (https://www.walmart.com/ip/Zyliss-Pizza-Wheel-Handheld-Pizza-Cutter-with-Removable-Blade-Red/19330718 ).
  The palm handle matters: a top-down pinch across its 29 mm thickness pushes down straight through the axle, so the
  push needs no twisting strength from the pinch.
- Board: Acopa 10 in round acacia serving board with a 5 in handle, 1/2 in thick, US$10.99
  (https://www.webstaurantstore.com/acopa-10-round-acacia-wood-serving-board-with-5-handle/303SBR10.html ).
- Pizza: a single-serve frozen pizza, for example Celeste Pizza For One, pepperoni, 5 oz (142 g)
  (https://www.cub.com/store/cub/products/42538-celeste-pizza-for-one-pepperoni-pizza-5-oz ), heated as its box says
  (microwave about 2 min). A small pizza on a 10 in board, as served at home.
- Any wooden utensil or knife block with a slot that takes the wheel upright (the stand; not part of the job).
- Food-safe fingertip covers (disposable finger cots) for the half drawn by hand.
Total: about US$40 plus about US$2 of pizza per trial.

**Estimates** (not published; how to check in a minute): pizza 152 mm (6 in) across and 9 mm thick (ruler); wheel 96
mm across, 2.5 mm at the hub, guard and handle shape, 75 g guard + 56 g wheel (ruler, kitchen scale); board handle 40
mm wide, board 0.50 kg (ruler, kitchen scale); the down force that cuts a baked thin crust right through with a sharp
wheel, 8 N (stand the pizza on a kitchen scale and press the wheel down until it cuts through: about 0.8 kg);
wheel bearing drag 0.03 N m; board-on-table friction 0.35.

## How the simulation works

The board is one free articulated body. The pizza is two rigid halves (crust, sauce and cheese half-discs with
pepperoni), each on a slide along the board held locked by a stiff drive while the pizza is whole: a half cannot move
apart from the other, so pushing on it pushes the whole board, as a pizza that has not been cut
through does. The pizza wheel is a free articulated body: guard and palm handle, and a steel wheel on a free axle
that turns only by its contact with the pizza or the board.

The cut map: the two seams (the halves of the cut line, from the centre to 3 mm inside the crust edge) are each
eight stretches. Every control step the task file finds the wheel's lowest point on the board and the force through
its axle (the PhysX joint reaction force between the wheel and its guard, that is how hard the wheel is being pushed
onto what it stands on). A stretch is cut when the wheel's lowest point passes over it while: within 6 mm of the
seam line, on the pizza's top (within 6 mm), the wheel's plane within 15 degrees of the seam, the wheel turning on its
axle as it moves (rolling, not skidding), and the axle force at least 8 N. When both seams are cut along their whole
length (the line crust to crust) the slides are freed, the halves move 2 mm apart (the cut opening) and one can be
drawn away. A light pass, a pass more than 6 mm off the centre line (so the halves are within 45/55 of equal) or one
that stops short of the far crust leaves stretches uncut, and the halves stay joined.

Scripted demo: the left hand pinches the board's handle across its width, fingertips 3 mm above the table, and holds
it to the end. The right hand lifts the wheel out of the block by the palm handle (pads across its 29 mm thickness,
squeezed 5 mm). It turns the wheel over the crust edge on the left arm's side, reads how the wheel sits in its fingers and turns
the grip so the wheel's plane lies on the cut line, sets it on the crust, aims its edge 20 mm below the cheese (the
arm's give turns that into 10-16 N on the axle) and rolls it across through the centre in 4 s, reading the wheel's
position every 0.4 s and moving the hand sideways to keep the edge on the line. The cut rolls toward the robot's right. It
lowers the wheel into the block and lets go (in the recorded run it ends on the table beside the block), then presses its closed (covered) fingertips onto the far half and draws it 45 mm
away from the other. The demo reads object poses; a policy reads the cameras.

Demo result: demo-658a3e75-a1, full length, no jitter: 1.0. The wheel rolled crust to crust through the centre and cut both seams; the halves came free, the far half was drawn 43.6 mm away (the other stayed within 0.1 mm), the board moved under 1 mm, and the wheel was let go. The wheel was set down in its block's slot, but the rising hand pulled it back out and it ended on its side on the table beside the block. The scorer checks only that the wheel is let go.

Wrong way: pizzahalf-wrong-658a3e75 (GPU copy of the same file, seed 11, object jitter 0.01). A light pass (edge aimed 1 mm into the cheese, lowered slowly; at most 2.8 N on the axle) cut nothing: 0. A pass that stopped 30 mm short of the far crust cut one seam through and 3 of 8 stretches of the other, so the halves stayed joined and nothing could be drawn away: 0.5.

Scores: Dream-1 run: 0.0 (full length); demo: 1.0 (full length); wrong way: light pass 0, pass stopped short of the far crust 0.5

## Fidelity and shortcuts

- The pizza is rigid pieces held by locked slides, not a soft food; the cut is the rule above (force- and
  roll-gated); a freed half slides only along one straight line. Cheese stringing, crust crumbling, toppings
  dragging and the crust's give under the wheel are left out.
- The wheel's 8 N cutting force, its size and the drag of cutting (0.03 N m on the axle) are estimates.
- The utensil block is a stand for the start pose, not part of the job; built from simple shapes, no logos.
