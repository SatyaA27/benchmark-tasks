# Buy a gumball from a coin gumball machine: drop the quarter in the slot, turn the handle a full turn, lift the door

Buy one gumball: drop the quarter into the coin slot on top of the gumball machine's coin mechanism, turn the handle one full turn clockwise, then lift the chute door so the gumball rolls out of the machine.

Countertop coin gumball machines stand in shops, waiting rooms and children's bedrooms, and working one is among the first purchases a child makes. It strings three different contacts together. A flat quarter has to be picked off the table, turned to stand on edge and dropped into a slot barely wider than the coin, through a mouth no bigger than a fingertip. The handle, a small T-bar, then has to be turned a whole turn clockwise; no wrist turns that far, so the hand lets go half way, turns back and grips again, and a ratchet stops the handle going back. The mechanism only turns if the coin is seated: without it a sprung pin stops the handle after about a quarter turn. Last, the gumball waits behind a hanging chrome door; lifting the door's lip lets it roll out. Nothing can be skipped: no coin, no turn; no full turn, no gumball.

## Scoring

Judged at the end of the run, from the video and a look at the machine (is the quarter gone into the slot, how far
has the handle turned, where are the gumballs).

| End state | Score |
|---|---|
| Quarter inside the machine, exactly one gumball out through the door onto the table, machine upright | 1.0 |
| Quarter inside, exactly one gumball dispensed but still behind the closed door | 0.75 |
| Quarter inside, handle turned at least a quarter turn, no gumball out | 0.5 |
| Quarter dropped in and seated in the coin dial, handle turned less than a quarter turn | 0.25 |
| Anything else: quarter not in the machine (whatever else happened), two gumballs out, machine tipped over 10 degrees | 0 |

Doing nothing scores 0: the quarter starts on the mat. A gumball got any other way than through the mechanism scores
0 because the quarter must be in the machine for any credit above 0, and one quarter buys exactly one gumball.
In the simulation `score()` reads the same things: the quarter's position in the machine's frame (in the coin dial's
channel or in the coin box), the dial's angle, each gumball's place (in the wheel or feed tube, in the chute behind
the door, or out of the machine's footprint on the table) and the machine's tilt.

## The machine and what is real

- **Gumball machine**: VEVOR 16 in gumball machine (model ak201-01 class; red, PC globe, aluminium-alloy base),
  published 215 x 200 x 410 mm, 2.95 kg, "insert a 25-cent coin, rotate the control lever clockwise 360 degrees",
  "the candy exit is fitted with a metal door guard", "fits 350 1" round candies", one gumball wheel, about 99 USD:
  https://www.vevor.com/candy-vending-machine-c_10566/vevor-16-h-gumball-machine-vending-coin-bank-vintage-candy-dispenser-pc-red-p_010498259659 . The maker's own picture shows the quarter dropped edge-down, face to the user, into the slot on top
  of the coin mechanism.
- **Estimated from the maker's photos** (scaled to the published 410 mm height; check with a ruler in a minute):
  coin mouth 182 mm up, knob axis 141 mm up, T-bar 41 x 13 mm standing 15 mm proud, mechanism plate 90 x 82 mm, exit
  door 46 x 47 mm hinged 89 mm up with a 16 mm turned-out lip.
- **Coin mechanism** (common rotary 25-cent arrangement; see the gumball-bank patent https://patents.google.com/patent/US3937314A/en for the same coin
  pin and anti-reverse idea): the quarter drops through a 28 x 2.8 mm slit with chamfered lips and stands in a 26.2 x 2.8 mm channel in the coin dial with 6 mm of it above
  the dial's rim; a sprung pin presses on the rim at its side. With the channel empty the pin drops into a 4 mm
  notch after about a quarter turn and the notch's wall bears on it: the handle stops; a seated coin fills the notch
  and pushes the pin out over it. An 8-tooth ratchet and pawl stop
  the handle turning back. The coin leaves the dial past the bottom and falls into the coin box.
- **Gumball**: 1 in (25.4 mm), 9 g (850 per 17 lb case: https://www.gumballs.com/orange-gumballs.html). **Quarter**: 24.26 mm, 1.75 mm, 5.67 g
  (https://www.usmint.gov/learn/coins-and-medals/circulating-coins/quarter).
- **Forces (estimates, labelled)**: handle turning torque 0.12-0.15 N m (check: push the T-bar's end with a kitchen
  scale's pan, 20 mm from the axis, while turning: 0.6-0.75 kg on the scale); door 20 g, closed by its own weight;
  pin and pawl springs about 0.2-0.6 N.

## Simulation shortcuts (disclosed)

- The globe's gumballs are fixed scenery. The vended gumball comes from a one-pocket wheel on the knob's own shaft,
  fed by a short tube holding the next gumball; the real machine's horizontal turntable is geared to the knob. Either
  way one gumball comes out per full turn, in its last third (here at 245-270 degrees).
- The coin pin, ratchet and coin path inside the base follow the common rotary mechanism, not a drawing of this
  model. All parts are rigid; the gumballs, the quarter and the mat are free bodies.
- The machine is fixed to the table (the real one weighs 2.95 kg and does not move under these pushes); the parts inside it
  (the two gumballs) start where they are, so runs use no start jitter.
- Gumballs are sealed candy and never touched by the gripper; only the quarter, the T-bar and the door's lip are.

## Setup

Millimetres from the midpoint between the arm bases: x forward, y to the robot's left, heights above the table top. A VEVOR 16 in gumball machine (red, 215 wide x 200 deep x 410 high, 2.95 kg, 25-cent coin mechanism, one gumball wheel, chrome exit door) stands at x 332, y -60, its coin mechanism facing the arms. On the mechanism: the coin slot's mouth on top, 182 high; the knob axis 141 high, with a 41 x 13 mm T-bar standing 15 mm proud, horizontal at the start; below it the exit door (46 x 47 mm, hinged along its top edge 89 high, a 16 mm turned-out lip at its bottom 42 high). The globe holds 1 in gumballs; one sits in the wheel's pocket, one in the feed tube. A US quarter (24.26 mm, 1.75 mm, 5.67 g) lies flat in the middle of a black rubber mat 110 x 110 x 2 mm at x 150, y -175. The table in front of the exit door is clear. In real trials the machine stands on a thin non-slip pad (the pushes stay under 1 N against its 2.95 kg; the simulation holds it fixed).

## Reset

Unlock the base's coin box with the machine's key and take the quarter out; put it back flat in the middle of the mat. Unlock the top cover, drop the gumball back into the globe, close it. If the handle stopped part way, finish the turn with a second quarter and take that coin and gumball back out the same way. About a minute; nothing is consumed (gumballs are sealed candy; a clean dry gumball is reused, or use a set of 1 in wooden balls, which the wheel vends the same way).

## Demonstration and checks

- Scripted demonstration (right arm; the left stays at rest): the right hand picks the quarter off the mat by its rim, swings level so the coin hangs face to the machine, lowers it into the slit's mouth and lets go: it drops into the coin dial's channel. It then grips the T-bar and turns it about 166 degrees, lets go, turns its wrist back, grips again and turns on to 346 degrees (the gumball drops into the chute in the last third of the turn; the quarter falls into the coin box). Last it slides a fingertip under the door's lip and lifts the door: the gumball rolls out onto the table. Full score at 38.3 s (766 of the 2,400 steps); the run stops there, once the score has held for 1 s, with the hand still holding the door open, so in the last frames the gumball lies on the table under the right hand, about 10 cm in front of the machine's base (the top view hides it; the scorer reads its position).
- Wrong way: the same motions without the quarter: the handle stops at 97.5 degrees (the coin pin has dropped into the empty notch), the door is lifted to 61 degrees and nothing comes out: 0.0.

The code is `buy_a_gumball_coin_in_handle_full_turn_lift_door.py` (layout, scoring, demonstration) and `gumball_machine.py` (the machine: body, coin dial, pin,
ratchet, gumball wheel, door; its docstring lists every number).
