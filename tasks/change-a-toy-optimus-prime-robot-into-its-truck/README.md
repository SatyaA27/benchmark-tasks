# Change a toy Optimus Prime robot into its truck: legs together, arms in, then the back pack swung over as its front

## Scoring

Score 0 if the figure fell off the bench or was forced: its body moved more than 30 mm from its start mark or turned more than 20 degrees at any time. Otherwise the lowest of these numbers. (a) Parts home (each of the six hinged parts - two legs, two arms, two halves of the truck front - within 6 degrees of its truck-form stop): 6 = 1.0; 5 = 0.75; 3-4 = 0.5; 1-2 = 0.25; 0 = 0. (b) On its wheels (tilted no more than 15 degrees) = 1.0, otherwise 0.5. (c) Hands away (no part of either gripper within 60 mm of the figure's outline) = 1.0, otherwise 0.75. Doing nothing scores 0. Success: 1.0 held 1 s. In a real trial: one end photo from above against the truck picture on the instruction sheet shows which parts are home, whether the truck stands on its wheels inside its tape outline and where the hands are; the video shows any shove.

| Parts home (of 6) | Tier |
|---|---|
| 6 | 1.0 |
| 5 | 0.75 |
| 3-4 | 0.5 |
| 1-2 | 0.25 |
| 0 | 0 |

The result is capped at 0.5 if the truck is tilted more than 15 degrees (not standing on its wheels) and at 0.75 if a
gripper is still within 60 mm of the figure's outline at the end. It is 0 if the body was shoved more than 30 mm or turned
more than 20 degrees at any time, or the figure fell off the bench. "Home" is within 6 degrees of the truck-form stop
of that hinge. The start state has no part home, so it scores 0.

## Instruction

The toy robot, a Transformers Authentics Optimus Prime, lies face down on the bench. Change it into its truck in the order of its instruction steps: (1) straighten the legs: swing both legs in until they stop side by side; (2) push both arms in flat along the sides of the body until they stop; (3) swing the back pack forward to form the front of the truck: swing its two halves round the sides of the head until they meet in front of it. Do not force a part or shove the figure off its mark; leave the truck on its wheels and take both hands well away.

## The toy

The product is the **Transformers Authentics Alpha Optimus Prime** by Hasbro (item E0771, 7-inch figure scale, ages
6 and up, robot mode to truck mode "in 4 steps"):
https://consumercare.hasbro.com/en-us/product/transformers-authentics-optimus-prime/1362184A-2B6C-4048-8576-F789421C5A11
Its steps from robot to truck, as printed on the card: straighten out the arms and legs; rotate the arms up, then
push them in; swing the "back pack" over the arms to form the front of the vehicle; swing the legs back up over the
thighs.

The model gives the figure's proportions and colours only (red cab-shaped torso and arms, blue helmet, legs and
fists, silver waist, knee plates, bumpers, grille and exhaust stacks, light-blue windscreen, black wheels), built
from plain boxes and cylinders: no logo, faction mark, printed text or sculpted detail. The figure lies face down so
that every step is a flat swing on the bench and the truck ends standing on its wheels. Two simplifications: the
one-piece back pack that is swung over to form the front of the truck is modelled as two halves that swing forward
round the sides of the head on vertical hinges and meet in front of it, because a hand from above cannot get under a
flat piece lying on the figure's back; and the last step (lower legs swung up over the thighs) is left out, the legs
being modelled straight. Its six hinged parts:

| Part | Size (mm) | Robot form | Truck form | Hinge friction (N m) |
|---|---|---|---|---|
| Legs (2), blue, rear wheel on the outside | 81 x 32 x 30 | 32 deg spread outward | together, side by side | 0.015 |
| Arms (2), red with blue fists, front wheel near the shoulder | 108 x 14 x 19 | 45 deg out from the body | flat along the body's sides | 0.020 |
| Truck-front halves (2), red, silver grille and amber headlight in front | 39 x 32 x 9 | straight out beside the head | meeting in front of the head | 0.015 |

Head to feet 178 mm (the published 7-inch scale), torso 70 x 66 x 35 mm, head 25 x 28 x 25 mm, 194 mm long as a
truck, whole figure 0.28 kg, wheels 30 mm across. Everything but the scale is an estimate; on the real toy it takes a
minute to check with a ruler, a kitchen scale, and a small spring balance hooked on a part's tip (0.02 N m on a
70 mm lever is about 0.3 N, about 30 g).

Why the card's order is the easy one: with a leg still spread, the arm on that side meets the thigh about 19 degrees
before its stop (the parts collide in the simulation), so an arm pushed in first has to drive the leg in with it;
with the legs together the arms swing in freely, and the truck-front halves close over the head last. The scorer
does not demand the order; it demands every part at its stop and no forcing: a figure pushed more than 30 mm or
turned more than 20 degrees scores 0.

## Setup

The Optimus Prime figure lies face down in robot form inside a tape outline, torso centre at (0.13, 0.00) m, head toward the
robot, feet away; arm bases at (0, +/-0.305) m. Both arms start at the ready pose with grippers open.

## Scripted demo

The right hand folds the right-side parts, the left hand the left-side ones, one part at a time: legs, arms, then the
truck-front halves. For each part the hand comes down beside it with its fingers closed, then sweeps round the hinge in
3-degree steps, its fingertips 0.5 mm into the face it pushes and 2 degrees ahead of the part, ends 3 degrees past the stop, holds there
0.3 s, backs off 12 mm and lifts. Before each part the figure's pose and that part's angle are read again. The other hand
waits clear. At the end both hands go back to their park points, about 0.2 m to either side of the truck, so no part
of a gripper is within 60 mm of it. In the full-length run the six parts reach their stops (each within 0.2 degrees), the figure ends 0.4 mm from its mark, and success is called 60 s in, once both hands are away.

## Wrong way

Folding the arms first: in a jittered run of the same code each arm, pushed in first, met its spread thigh and drove the leg most of the way in with it (the leg was at 7 degrees when the arm reached its stop); the later leg folds finished the legs, so this order also ended at 1.0, the figure moved 3.6 mm and turned 0.5 degrees. The scorer does not demand the order; pushing harder against a blocked part until the figure is moved more than 30 mm or turned more than 20 degrees scores 0.

## Scores

Dream-1 run: 0.25; demo: 1.0

## Simulation notes

Rigid parts on vertical hinges with joint friction; each click is the hinge's end stop (no snap torque is modelled).
Self-collision between the parts is on, so a part blocks another by contact. Forcing is judged by the figure being
shoved or turned, as a real trial sees it on video. Left out: plastic flex, ball joints popping off, the toy's small
detail parts.
