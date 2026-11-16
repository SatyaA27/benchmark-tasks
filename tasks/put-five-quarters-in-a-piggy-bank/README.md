# Put five quarters in a piggy bank: pinch each flat coin off the table and turn it on edge to fit the slot

Put the five coins in the piggy bank.

Dropping spare change into a piggy bank is one of the first money habits children learn, and plenty of adults keep one on a shelf. It is easy for a person and hard for a two-finger gripper: a US quarter lying flat is only 1.75 mm thick, so it must be pinched by its rim right at the table's surface, then turned on edge in the fingers and lined up over a slot only 5 mm wide before it is let go. A coin carried flat lands across the slot and stays on the pig's back; one let go off line hits the slot's edge. The robot does it five times with one hand, and must not knock the loose ceramic bank over.

Sources: US Mint, "Quarter" (diameter 0.955 in / 24.26 mm, thickness 1.75 mm, weight 5.670 g,
cupronickel clad copper), https://www.usmint.gov/learn/coins-and-medals/circulating-coins/quarter ; Child to Cherish
Large Piggy Bank retail listings (8" L x 7" W x 7" H, item weight 2.0 lb, "a 1.5 inch coin slot at the top", "a
removable rubber stopper ... at the bottom", dolomite ceramic).

## Scoring

Judged at the end of the run (in the real world: pull the stopper and count). 0.2 for each quarter inside the bank: its centre inside the hollow body and the whole coin below the slot's inner mouth, so a coin stuck in the slot does not count. 1.0 means all five are in. The score is 0 if the bank is ever tipped more than 20 degrees from upright (knocked over, or laid on its side so coins could be pushed in flat) or leaves the table. The run ends early with success once all five have been in for 1 second.

| End state | Score |
|---|---|
| Five quarters in the bank, bank upright | 1.0 |
| Four / three / two / one in | 0.8 / 0.6 / 0.4 / 0.2 |
| A coin wedged in the slot, or lying on the bank's back | that coin earns nothing |
| Bank ever tipped past 20 degrees, or off the table | 0 |

"Inside" is measured in the bank's own frame: the coin's centre is inside the hollow (the inner ellipsoid, outer
semi-axes less the 5 mm wall) and lower than the inner mouth of the slot by more than the coin's radius plus 1 mm, so
the whole coin has passed through the wall. The tilt test reads the angle between the bank's vertical axis and the
world's, every control step; "ever" means once tipped, the run scores 0 even if the bank rocks back. `success()` is a
score of 1.0.

Why the shortcuts fail: measured in simulation, quarters carried flat (not turned on edge) and let go 25 mm over the slot all land across it and stay stacked on the pig's back, 1 to 8 mm above the slot's mouth: score 0, with the bank untouched. Tipping the bank or laying it on its side to push coins in flat scores 0 by the tilt rule, and a coin wedged in the slot earns nothing.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=put_five_quarters_in_a_piggy_bank.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Put the five coins in the piggy bank." \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial, replace `--policy yam_scripted` with `--policy dream`. Each trial allows 2,400 control steps
(120 seconds at 20 Hz). The layout is fixed. The scripted demonstration scores 1.0 on this layout: the fifth quarter is in at step 1,336 (66.8 s) and the run ends with success at step 1,355. With every object shifted at random by up to 10 mm it also scored 1.0 in 3 of 3 runs.

`pytest` in this folder runs the generated offline checks (no GPU): the scorer on the start, the goal and the named
partial and shortcut layouts.

## The scene

Millimetres from the midpoint between the arm bases: x forward, y to the robot's left, heights above the table top. A Child to Cherish Large Piggy Bank (glazed pink ceramic, 203 long, 178 wide, 178 high, 0.91 kg; hollow body with a 5 mm wall; a 38 x 5 mm coin slot along the top of its back, its mouth 176 above the table; a rubber stopper in its belly) stands loose on its four feet with its body centred at x 320, y 200, its snout pointing forward and to the left, 70 degrees from straight ahead, so the slot runs along that line. Five US quarters (24.26 across, 1.75 thick, 5.67 g, cupronickel) lie flat on the bare wooden table, centres at (235, 60), (190, 95), (130, 150), (200, 20) and (120, 70). Nothing else is on the table; the bank is empty. The left arm can do the whole task; the right arm is not needed.

The bank (one rigid body, 0.91 kg): a hollow ellipsoidal body 172 long, 172 wide and 152 high, centred 100 above
the table, with a 5 mm ceramic wall; the slot (38 x 5) runs along the top of the back through the wall; four legs
30 across; a snout 48 across ending 105 in front of the body's centre; two ear flaps rising to 178; a tail; a rubber
stopper 42 across in the belly. The body is one closed triangle mesh (about 11,800 triangles) with a signed-distance
collider, so coins really pass through the slot into the hollow and land on its floor; its shell volume times
ceramic's density (0.40 L x 2.3 g/cm3 = 0.92 kg) matches the listed weight. Quarters are 24.26 x 1.75 mm discs of
5.67 g (collider: a square-edged 32-sided prism of the same area). Friction (static/dynamic): coins 0.40/0.35, bank 0.50/0.45.

## Scripted demonstration

The left arm posts the coins one at a time (70.5 s plan). For each coin: the hand comes over it
pointing straight down, fingers closing along the bank's spine direction; it stops with the pad tips 8 mm above the
table, reads where it actually is from its joint angles, corrects its sideways position by that reading and lowers
the tips to 0.5 mm above the table, so the two flat pads close on the coin's 1.75 mm rim across its diameter (pads
commanded 21 mm apart, about 8 N). It lifts the coin flat, rises while turning, and ends level, pointing across the slot from 50 mm back,
so the coin now stands on edge in the pinch with its faces square to the slot's width. Over the slot it holds, reads
the coin's and the bank's poses and corrects position and face angle; lowers to 8 mm above the slot, reads again;
lowers until the coin's lower edge is 2 mm into the slot; opens; backs away. The coin is held 4.5 mm off the pads'
middle so about 10 mm of it hangs below the pads; the pad tips clear the bank's back by about 1 mm at let-go.
The scripted demo reads object poses from the simulator (privileged); a model under test sees only the cameras.

Measured: 1.0, all five quarters in by 66.8 s of the 120 s allowed; the bank never moved or tilted. Over 20 postings (this run and the three jittered ones) the first read over the slot found the coin as held within 2.4 mm and 5.7 degrees of its planned line, and the second, just above the slot, within 2.7 mm and 0.5 degrees; each read is corrected before the hand moves on.

## Simulation and the real objects

Rigid bodies, 240 Hz physics, 20 Hz control. Measured or published: the quarter's size, mass and alloy (US Mint);
the bank's overall size, weight, slot length, ceramic body and rubber stopper (maker's listings). Estimates, to be
measured on the real bank: slot width 5 mm (listings that give a width show 4.5 to 8 mm), wall 5 mm (a German maker
lists 3.5 to 5 mm for its ceramic pigs), the body's exact shape and the slot's position along the back, the legs,
snout and ears, and every friction value. The slot's edges are square (no chamfer), which is the harder case. The
quarters are plain discs (32-sided colliders): no relief, no reeded edge. The bank has no felt or rubber pads under its feet.

## Reset

Pull the rubber stopper out of the bank's belly, tip the coins out and refit the stopper. Stand the bank on its marks and lay the five quarters flat on theirs, either face up.
