# Cut a single-serve pizza in half with a pizza wheel pressed hard through the centre, one hand holding the board

**Instruction:** "Cut the pizza in half. Hold the pizza board still by its handle with one hand. With the other, lift the pizza wheel out of its block, press it down on the pizza and roll it straight across through the centre, from one crust edge right over the other, pressing hard enough to cut right through. Lay the pizza wheel down on the table in the taped rectangle to the right of the board, then draw one half about 3 cm away from the other."

A small single-serve pizza (about 6 in) lies on a round serving board with a handle. One hand holds the board by its handle; the
other lifts a palm-held pizza wheel out of a utensil block, presses it down on the pizza and rolls it straight across
through the centre, crust edge to crust edge, so the pizza is halved to share, lays the wheel down in a taped
rectangle and draws one half away from the other.

## Scoring

Score 0 if neither seam is cut through, or the board ends tipped over or off the table. Otherwise the lowest of these numbers. Cut (the line through the centre is two seams, each from the centre to 3 mm inside the crust edge, 73 mm; a stretch is cut only where the wheel's edge rolls over it within 6 mm of the line pressing down at least 8 N): both seams cut through = 1.0; one = 0.5 (the halves stay joined). Half drawn (how far each half has moved at the end; a half can move only once both seams are through): one half 30 mm or more away with the other within 10 mm = 1.0; 10-30 mm = 0.75; otherwise = 0.5. Board held still (largest shift of its middle and largest turn): within 15 mm and 5 degrees = 1.0; 30 mm and 10 degrees = 0.75; 60 mm and 20 degrees = 0.5; more = 0.25. Pizza wheel: let go, resting on the table with its centre (the axle) inside the taped rectangle = 1.0; let go anywhere else = 0.75; still in the hand = 0.5. Doing nothing scores 0. Success: 1.0 held for 1 s. In a real trial: at reset a dot on the board marks the pizza's centre and two pencil ticks on the board's rim mark the cut line through it. After the trial slide a thin spatula under each half and lift it (or turn the pizza over onto a second board) and look at the underside along the line. A seam counts as cut through when a card pushed down the cut reaches the board along the whole 73 mm from the centre dot (ruler along the line), with no joined bridge 9 mm or longer (the simulator splits each seam into eight 9 mm stretches and needs all eight cut), and every point of the cut lies within 6 mm of the line between the ticks (ruler square to the line); a part further off is not cut. Both seams = 1.0, one = 0.5, none = 0. End photo with a ruler: the drawn half's gap, the board against its tape outline (5 degrees is 11 mm at the rim), whether the wheel's centre lies inside the tape; the video shows it let go.

| Outcome | Score |
|---|---|
| Whole line cut crust to crust, one half drawn 30 mm or more (the other within 10 mm), board within 15 mm / 5 degrees, wheel let go, resting on the table with its centre (the axle) inside the taped rectangle | 1.0 |
| As above, but the half drawn only 10-30 mm, or the board moved up to 30 mm / 10 degrees, or the wheel let go outside the taped rectangle | 0.75 |
| Only one seam cut (the pass stopped short of a crust: halves joined), or no half drawn, or the board moved up to 60 mm / 20 degrees, or the wheel still in the hand | 0.5 |
| Board shoved more than 60 mm / 20 degrees | 0.25 |
| Nothing cut through (doing nothing, a light pass, a pass more than 6 mm off the centre line), board tipped or off the table | 0 |

(The score is the lowest band reached on any line.)

**Real trial.** At reset a dot on the board marks the pizza's centre and two pencil ticks on the board's rim mark
the cut line through it; the pizza is centred on the dot. After the trial:

- Cut. Slide a thin spatula under each half and lift it, or turn the pizza over onto a second board, and look at the
  underside along the line. Each seam runs from the centre dot to 3 mm inside the crust edge (73 mm). It counts as
  cut through when a card pushed down the cut reaches the board along that whole length (ruler along the line from
  the dot), with no joined bridge 9 mm or longer, and every point of the cut lies within 6 mm of the line between the
  ticks (ruler square to the line); a part further off counts as not cut. Both seams through = 1.0; one seam (centre
  to one crust edge) = 0.5; none = 0. The simulator matches this: each seam is eight 9.1 mm stretches, a stretch is
  cut only where the pressed, rolling wheel's edge passes within 6 mm of the line, and a seam counts only with all
  eight cut, so an uncut stretch is a joined bridge of up to 9 mm.
- Half drawn and board: end photo with a ruler (the drawn half's gap; the board against its tape outline, 5 degrees
  is 11 mm at the board's 127 mm rim).
- Wheel: the end photo shows whether the wheel's centre (its axle) lies inside the 140 mm taped rectangle, however
  it rests; the video shows it let go.

## The real objects and their numbers

- Pizza wheel: Zyliss Pizza Wheel (palm-held; the handle sits straight above the wheel over a guard), SKU E910081,
  13.4 x 2.9 x 13 cm, US$19.95 (https://us.zyliss.com/products/zyliss-pizza-wheel ).
  A top-down pinch across its 29 mm palm handle pushes straight down through the axle.
- Board: Acopa 10 in round acacia serving board with a 5 in handle, 1/2 in thick, US$10.99
  (https://www.webstaurantstore.com/acopa-10-round-acacia-wood-serving-board-with-5-handle/303SBR10.html ).
- Pizza: a single-serve frozen pizza, for example Celeste Pizza For One, pepperoni, 5 oz (142 g)
  (https://www.cub.com/store/cub/products/42538-celeste-pizza-for-one-pepperoni-pizza-5-oz ), heated as its box says
  (microwave about 2 min). A small pizza on a 10 in board, as served at home.
- Any wooden utensil or knife block with a slot that takes the wheel upright (the stand; not part of the job).
- Food-safe fingertip covers (disposable finger cots) for the half drawn by hand.
Total: about US$40, plus US$2 of pizza a trial.

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

The cut map: each of the two seams (centre to 3 mm inside the crust edge) is eight 9.1 mm stretches. Every control
step the task file finds the wheel's lowest point and the force through its axle (the PhysX joint reaction force
between the wheel and its guard). A stretch is cut when that point passes over it within 6 mm of the line, on the
pizza's top, the wheel's plane within 15 degrees of the line, the wheel turning on its axle (not skidding) and the
axle force at least 8 N. With both seams fully cut the halves are freed, part 2 mm and can be drawn away.

Scripted demo: the left hand pinches the board's handle across its width, fingertips 3 mm above the table, and holds
it to the end. The right hand lifts the wheel out of the block by the palm handle (pads across its 29 mm thickness,
squeezed 5 mm). It turns the wheel over the crust edge on the left arm's side, reads how the wheel sits in its fingers and turns
the grip so the wheel's plane lies on the cut line, sets it on the crust, aims its edge 20 mm below the cheese (the
arm's give turns that into 10-16 N on the axle) and rolls it across through the centre in 4 s, reading the wheel's
position every 0.4 s and moving the hand sideways to keep the edge on the line. The cut rolls toward the robot's right. It
lowers the wheel to 8 mm above the table in the taped rectangle and lets go, rises clear, then presses its closed
(covered) fingertips onto the far half and draws it 45 mm away from the other. The demo reads object poses; a policy reads the cameras.

Demo result: demo-fbc5b7ec-a1, full length: 1.0 (success from 31 s). Both seams cut, the halves freed and the far half drawn 31 mm away when success was reached, board moved 0.1 mm, wheel laid down with its centre inside the taped rectangle (resting tipped on its guard) and let go.

Wrong way: pizzahalf-wrong-fbc5b7ec (seed 11, jitter 0.01): a light pass (edge 1 mm into the cheese, at most 2.8 N on the axle) cut nothing: 0; a pass stopped 30 mm short of the far crust cut one seam and 3 of 8 stretches of the other, halves joined: 0.5.

Scores: Dream-1 run: 0.0 (full length); demo: 1.0 (full length); wrong way: light pass 0, pass stopped short of the far crust 0.5

## Fidelity and shortcuts

- The pizza is rigid pieces held by locked slides, not a soft food; the cut is the rule above (force- and
  roll-gated); a freed half slides only along one straight line. Cheese stringing, crust crumbling, toppings
  dragging and the crust's give under the wheel are left out.
- The wheel's 8 N cutting force, its size and the drag of cutting (0.03 N m on the axle) are estimates.
- The block is only a stand for the start pose (simple shapes, no logos); the wheel is laid in a taped rectangle,
  not stood back in its 12 mm slot, where it did not stay upright in probes.
