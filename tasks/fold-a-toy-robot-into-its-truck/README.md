# Change a toy robot into its truck: legs together, arms in, then the bonnet halves over the head

## Scoring

Score 0 if the figure fell off the bench or was forced: its body moved more than 30 mm from its start mark or turned more than 20 degrees at any time. Otherwise the lowest of these numbers. (a) Parts home (each of the six hinged parts - two legs, two arms, two bonnet halves - within 6 degrees of its truck-form stop): 6 = 1.0; 5 = 0.75; 3-4 = 0.5; 1-2 = 0.25; 0 = 0. (b) On its wheels (tilted no more than 15 degrees) = 1.0, otherwise 0.5. (c) Hands away (no part of either gripper within 20 mm of the figure's outline) = 1.0, otherwise 0.75. Doing nothing scores 0. Success: 1.0 held 1 s. In a real trial: one end photo from above against the truck picture on the instruction sheet shows which parts are home and whether the truck stands on its four wheels inside its tape outline; the video shows any shove.

| Parts home (of 6) | Tier |
|---|---|
| 6 | 1.0 |
| 5 | 0.75 |
| 3-4 | 0.5 |
| 1-2 | 0.25 |
| 0 | 0 |

The result is capped at 0.5 if the truck is tilted more than 15 degrees (not standing on its wheels) and at 0.75 if a
gripper is still within 20 mm of the figure at the end. It is 0 if the body was shoved more than 30 mm or turned
more than 20 degrees at any time, or the figure fell off the bench. "Home" is within 6 degrees of the truck-form stop
of that hinge. The start state has no part home, so it scores 0.

## Instruction

The toy robot lies face down on the bench. Change it into its truck the way its picture instructions show, in this order: (1) swing both legs together until each clicks against the other; (2) swing both arms in flat along the sides of the body until they click; (3) swing the two shoulder plates forward over the head until they meet in front of it as the truck's bonnet. The arms will not go in while a leg is still spread. Do not force a part or shove the figure off its mark; leave the truck on its wheels and take both hands away.

## The toy

A plastic converting robot toy of the simple kind sold for children of three and up: a robot of about 16 cm that
becomes a 165 mm truck in a few big steps, with chunky parts on stiff hinges that stop at the robot form and at the
truck form. The model is procedural and generic (orange and grey rescue truck; no logo, character or faction
marks). Its six hinged parts:

| Part | Size (mm) | Robot form | Truck form | Hinge friction (N m) |
|---|---|---|---|---|
| Legs (2), double rear wheels on the outside | 70 x 27 x 26 | 32 deg spread outward | together, side by side | 0.025 |
| Arms (2), front wheel near the shoulder | 92 x 12 x 16 | 45 deg out from the body | flat along the body's sides | 0.020 |
| Bonnet halves (2), grille and headlight in front | 33 x 28 x 10 | straight out beside the head (shoulder plates) | meeting in front of the head | 0.015 |

Torso 60 x 56 x 30 mm, head 21 x 24 x 21 mm, whole figure 0.28 kg, wheels 26 mm across. All sizes, masses and
hinge stiffness are estimates of this kind of toy; on a real one they take a minute to check with a ruler, a kitchen
scale, and a luggage scale or a small spring balance hooked on a part's tip (0.02 N m on a 60 mm lever is about
0.3 N, about 35 g).

Why the order matters: with a leg still spread, the arm on that side meets the thigh about 19 degrees before its stop
(the parts collide in the simulation), so it cannot click home; once the legs are together the arms swing in freely,
and the bonnet halves close over the head last. Pushing a blocked arm harder drives the figure across the bench,
which the scorer counts as forcing.

## Setup

The figure lies face down in robot form inside a tape outline, torso centre at (0.13, 0.00) m, head toward the
robot, feet away; arm bases at (0, +/-0.305) m. Both arms start at the ready pose with grippers open.

## Scripted demo

The right hand folds the right-side parts, the left hand the left-side ones, one part at a time: legs, arms, then the
bonnet halves. For each part the hand comes down beside it with its fingers closed, then sweeps round the hinge in
3-degree steps, its fingertips 2 mm into the face it pushes and 3 degrees ahead of the part, ends 3 degrees past the
stop, backs off and lifts. Before each part the figure's pose and that part's angle are read again. The other hand
waits clear. The scripted demo scores 1.0 at full length: all six parts reach their stops (within 1.4 degrees) and success holds from 55.4 s; the figure moves under 1 mm.

## Wrong way

Folding the arms first: the right arm stops against the spread thigh about 19 degrees short of its stop and the figure is pushed 2-11 mm and turned 4-10 degrees; folding the legs afterwards frees the right arm, but the left arm stays jammed about 19 degrees short. Three jittered trials scored 0.75, 0.5 and 0.75 (4-5 parts home).

## Scores

Dream-1 run: 0.0; demo: 1.0

## Simulation notes

Rigid parts on vertical hinges with joint friction; each click is the hinge's end stop (no snap torque is modelled).
Self-collision between the parts is on, so a part blocks another by contact. Forcing is judged by the figure being
shoved or turned, as a real trial sees it on video. Left out: plastic flex, ball joints popping off, the toy's small
detail parts.
