# Finish a game of Perfection against its timer: fit the last four shapes in their holes, then stop the clock

**Instruction:** "Finish this game of Perfection before its clock runs out: the timer is already running. Fit each of the four yellow shapes lying beside the game into its own hole in the blue tray, turned so it drops in, then slide the timer's switch to STOP. If the clock runs out first the tray pops up and throws the shapes out. Do not hold the tray down."

A Perfection game stands on the desk with its timer running and 40 seconds left. 21 of its 25 yellow shapes are
already in the blue tray; the last four (a square, a circle, a cross and a "hot dog", a bar with round ends) lie on
the desk beside the unit, two on each side, at loose angles. The robot fits each shape into the one hole cut for it,
turned so it drops in, then slides the timer's switch to STOP. If the clock runs out first, the tray pops up and
throws the shapes out, as in the real game.

## Scoring

Score 0 if the tray was held down when it popped, or the game unit was shoved more than 30 mm or tipped more than 10 degrees. Otherwise, if the clock ran out (the tray popped): 0.25 if all four shapes were down in their own holes at that moment, else 0. If the switch is at STOP at the end (the clock stopped by it, never run out), count the four shapes that were down in their own holes when the switch reached STOP and are still there at the end: 4 = 1.0; 3 = 0.75; 2 = 0.5; 1 = 0.25; 0 = 0. If the clock is still running at the end, 0. A shape is in only when it lies flat in the hole cut for its own shape, sunk to the pocket floor (within 1.5 mm, turned to fit); one resting on top of a hole is not in. Doing nothing scores 0: the clock runs out with no shape in. Success: 1.0 held 1 s. In a real trial: the video shows when the switch reached STOP and whether the tray popped; an end photo shows the switch at STOP and which holes are filled flush.

| Outcome | Score |
|---|---|
| Switch moved to STOP before time-out with all 4 shapes down in their own holes (and still there at the end) | 1.0 |
| Stopped with 3 of the 4 in | 0.75 |
| Stopped with 2 in | 0.5 |
| Stopped with 1 in; or all 4 in but the clock ran out (the tray popped) | 0.25 |
| Stopped with none in; the clock ran out with fewer than 4 in; the clock still running at the end; the tray held down when it popped; the unit shoved over 30 mm or tipped over 10 degrees | 0 |

Every outcome has exactly one score. The start scores 0: the clock is running and the four shapes are on the desk,
so doing nothing lets it run out with none in. Stopping the clock early earns only the shapes already in. A shape
counts only where it physically lies in its own hole, sunk to the pocket floor; the holes are cut to each shape
with 0.5 mm to spare a side, so a shape cannot go down into another's hole or into its own at the wrong angle, and
one resting on top of a hole is not in. Holding the tray down against its spring when it pops scores 0 (the real
game's rules forbid it). The clock stops only when the switch is physically slid to STOP.

**Real trial.** The trial video shows when the switch reached STOP and whether the tray popped; an end photo shows the
switch at STOP and which of the four holes are filled flush (a shape sitting proud of the tray is not in; a ruler
across the hole settles it). The unit sits on a tape outline (ruler for the 30 mm).

## The product and its numbers

- Hasbro Gaming Perfection (current edition), about US$26: "25 geometric shapes", a "game unit with timer", a
  "pop-up tray" and a storage compartment; listed size 13.11 x 10.51 x 2.99 in (333 x 267 x 76 mm), from a retailer
  listing (https://yeagerssportinggoods.com/hasbro-inc-perfection/). The standard round is 60 seconds for 25 shapes
  in a 5 x 5 grid; if time runs out the spring-loaded tray pops up and scatters the pieces
  (https://en.wikipedia.org/wiki/Perfection_(board_game)). The maker's instructions: press the tray down until it
  locks, set the dial, slide the switch to START, fit the shapes, slide it to STOP when done.
- The four shapes used here are in the modern set (players call them the square, the circle, the cross and the hot
  dog).

Estimates (not published; how to check in a minute): tray 225 mm square with holes 42 mm apart, its top 52 mm above
the desk, pockets 6 mm deep (ruler); shapes 6 mm thick with a knob 8 mm across and 11 mm tall; square 26 mm,
circle 32 mm, cross 34 mm with 12 mm arms, hot dog 36 x 16 mm (ruler); 0.5 mm clearance a side (feeler or the
shape's wobble in its hole); shapes 3.9-5.7 g (kitchen scale); the switch slides 12 mm and stops the clock after
8 mm, detent about 2 N (kitchen scale pushed against it); the tray's spring about 21 N when locked down and 6 N at
the top of its 25 mm pop (kitchen scale under the tray as it is pressed down). The layout of the timer housing on
the unit's right-hand side. Total cost of real objects: the game, about US$26.

## How the simulation works

The unit is one articulated body: a free base (the red frame and the timer housing), the tray on a vertical slide
(25 mm of travel, resting locked down), the switch on a horizontal slide with a 2 N detent, and the dial on a
hinge. The wind-up timer is a small script (`perfection_kit.py`): 40 s on the clock at the start, counting down
while the switch is short of 8 mm from START; at zero it releases the tray, and from then on the tray's spring
force (6 N + 600 N/m x the stroke still to go) pushes the tray up and the unit's base down, every physics step: the
tray jumps and throws loose and seated shapes out. The dial is turned to the clock's time by a small torque between
it and the base (a readout only). The tray counts as held down if it has not risen 15 mm within 1 s of its release. The game records which shapes sat
in their own holes at the moment the switch reached STOP and at the moment of the pop; the scorer reads those and the end state. A shape is in when, in the
tray's frame, its middle is within 1.5 mm of its hole's centre, its underside within 1.5 mm of the pocket floor,
it lies within 5 degrees of flat and within 5 degrees of the hole's angle (for its symmetry). The 21 shapes already
in are modelled as part of the tray's top (their yellow outlines and knobs, which the fingers must avoid). The four
loose shapes are rigid compounds (box and cylinder pieces); each hole is cut from the tray as wall blocks round
the shape grown by 0.5 mm, with the curved edges as 1.5 mm steps that never cut into that outline.

Demo: demo-364e6bd9-a1 scored 1.0 (full length, no jitter): all four shapes down in their own holes (0.2-0.5 mm off centre, flat, sunk to the floor), the switch at STOP about 26 s in with 14 s left on the clock. With 10 mm start jitter (three episodes) it fitted 4, 2 and 3 of the 4 shapes (1.0, 0.5, 0.75): a shape that lands 1-2.5 mm off rests on its hole's edge and does not count.

Wrong way: each shape tried in the other hole on its side and the switch never touched: the shapes rest on the wrong holes (no shape fits another's hole), the clock runs out at 40 s and the tray pops to its full 25 mm and throws them: 0.

Scores: Dream-1 run: 0.0; demo: 1.0

## Scripted demo

Each arm fits the two shapes on its side, one at a time; the right hand starts 4 s after the left, so the two never
reach in over the tray together. A top-down pinch on the shape's knob (jaws opened only 20 mm, so they clear the
neighbours' knobs 38 mm away), carried 40 mm above the tray to its own hole and turned the shortest way to fit; the
hand re-reads where the shape really is and corrects its aim twice with the shape 16 mm above the tray (clear of the
knobs) and three times at 1 mm, then lowers it into the hole and lets go. The right hand then pushes the switch to
STOP with its closed fingertips, about 26 s into the trial (the clock starts at 40 s). The demo reads the poses of
the shapes and the unit, as a person would see them, never the clock.
