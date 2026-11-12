# Cut an exact blackjack payout off a stack of real-size casino chips and set it beside the bet

Instruction: **"The player has blackjack. Pay the bet with chips from your stack, set beside it."**

**The difficulty is physical: cutting an exact count of thin chips off a stack.** A table-games dealer pays a winning
bet by taking chips off the top of the working stack and setting them down as their own stack beside the bet. The chips
here are true size (39.37 x 3.302 mm, 10 g, realistic friction), so the pinch has to land inside one chip's 3.3 mm
thickness (a window of about +/-1.6 mm) for exactly six to come away, and the six have to be carried and set down
without toppling and without touching the bet. The bet is never moved or capped (no chip placed on top of it), so
surveillance can still read what was wagered: a capped or moved bet, a chip tipped past 30 degrees or a chip off the
table is a procedure error (score 0), and a chip left loose (anywhere but the payout or the working stack) costs half
the credit.

The arithmetic is routine and printed on the layout: BLACKJACK PAYS 3 TO 2, paid in the bet's own colour, so the
player's 4-chip bet is paid 6 chips from the dealer's working stack of 20. Done by gambling dealers
([O*NET 39-3011.00](https://www.onetonline.org/link/summary/39-3011.00)).

## Run and score

From this task folder, with the YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=pay_winning_bet_with_casino_chips.py -E object_jitter=0 -E control_hz=20 \
  --instruction "The player has blackjack. Pay the bet with chips from your stack, set beside it." \
  --scorer task_score --max-steps 1200 --epochs 2 --store-frames --log-dir ./demo-run
```

For a model trial, replace `--policy yam_scripted` with `--policy dream`. The output of `--scorer task_score` is the
score.

Runtime per trial: at most 1,200 control steps (60 s at 20 Hz). The scripted demo plan lasts 22.2 s; the
`yam_scripted` policy then holds its last pose for 3 s and stops at 25.2 s (504 steps), unless the trial has already
ended with success (earliest 17.95 s: both grippers 30 mm clear of every chip at 15.95 s, plus 1 s still and the 1 s
success hold). The harness settles the scene for 0.5 s before the first observation. There are no seeds: start exactly
as authored, with `object_jitter=0`. The chips report themselves as not free, so a seeded reset never jitters a chip
out of its stack.

**The demo is open loop.** It uses only the known start layout, time and the robot's own kinematics, and reads nothing
from the simulator (no object poses, no scorer state, no contacts).

## Setup

World frame: x away from the robot (toward the player), y to the robot's left, z up; metres. Station table top
z = 0.7475 (`TOP`); the felt top is `FELT_TOP` = 0.7495. Arm bases at (0, +/-0.305, 0.75). All yaw 0.

### Colliders

| Object | Body | Collider (part frame) | Mass | Material, colour (sRGB) | Friction static / dynamic | Contact offset |
| --- | --- | --- | --- | --- | --- | --- |
| `felt_layout` | fixed | box, centre (0, 0, 0.001), size 0.360 x 0.550 x 0.002; origin (0.300, -0.175, 0.7475), so x 0.12..0.48, y -0.45..+0.10 | none | wool felt cloth, green (0.05, 0.33, 0.17) | 0.40 / 0.35 | 0.0005 |
| `bet_1` .. `bet_4` (bottom to top) | dynamic | cylinder, z axis, radius 0.019685 (1.55 in across), height 0.003302 (0.130 in), centred on the origin | 0.010 kg each | clay-composite chip (clay and resin), red (0.70, 0.06, 0.08) | 0.40 / 0.35 | 0.0005 |
| `stack_01` .. `stack_20` (bottom to top) | dynamic | the same cylinder | 0.010 kg each | the same | 0.40 / 0.35 | 0.0005 |

Chip k (0-based from the bottom of its stack) spawns with its centre at z = 0.7495 + (k + 0.5) x 0.003302 +
(k + 1) x 0.00002: a 0.02 mm spawn gap per level, closed by the 0.5 s settle. The bet's axis is (0.300, -0.240); the
working stack's axis is (0.200, -0.380). Every chip is its own dynamic rigid body with 32 position / 4 velocity solver
iterations, maximum depenetration velocity 1.0 m/s and rest offset 0. The station table and the two YAM arms are the
harness's own; the gripper pads keep the simulator's default material.

### Visual-only prints (no collider, no mass; nothing the scorer reads)

| Print | Geometry | Where | Colour |
| --- | --- | --- | --- |
| Chip edge stripe (every chip) | cylinder, radius 0.019785 (0.1 mm proud of the rim), height 0.0008 | centred on the chip | white (0.93, 0.93, 0.90) |
| Betting circle (the bet's) | white disc radius 0.045, 0.05 mm thick, on the felt; felt-green disc radius 0.042 on top of it (top 0.10 mm above the felt): a 3 mm white ring, 90 mm across | centred (0.300, -0.240) | white / green (0.05, 0.33, 0.17) |
| Betting circle (empty seat) | the same | centred (0.300, -0.090) | the same |
| Odds line "BLACKJACK PAYS 3 TO 2" | 86 white boxes 0.1 mm thick in a stroke font: stroke 1.6 mm, glyph cell 12 mm (x) by 8 mm (y), 3 mm letter spacing, 6 mm word spacing; 196 mm long | centred (0.240, -0.165); reads in +y with letter tops toward -x, i.e. from the player's side | white |

The edge stripe stands for the chip's edge inserts, drawn as one band at mid-thickness so each chip reads in a stack.

### Dimension sources

| Quantity | Value used | Source |
| --- | --- | --- |
| Chip diameter and thickness | 1.55 in (39.37 mm) x 0.130 in (3.302 mm) | Nevada Gaming Commission Regulation 12.040(1)(a): chips "must be disk-shaped, must be .130 inch thick, and must have a diameter of 1.55 inches, for chips used at games other than baccarat" ([Regulation 12](https://www.gaming.nv.gov/siteassets/content/home/features/Regulation12.pdf)). The same text is in Arkansas CGR 11.040, Louisiana LAC 42:III-4305 and South Dakota ARSD 20:18:20:04. |
| $5 chip colour | red | Louisiana LAC 42:III-4305 assigns red to $5 value chips; the common US convention. |
| Chip mass | 10.0 g (uniform density 2.49 g/cm3, no metal insert) | Chips in US casinos typically weigh about 10 g ([Casino chip](https://en.wikipedia.org/wiki/Casino_chip)); casino clay chips are sold at 10 g, some casino lines at 8.5 g. |
| Chip-chip and chip-felt friction | 0.40 static / 0.35 dynamic | Estimate: no published measurement for clay-composite chips was found; dry material pairs typically lie at 0.3-0.6, felt on smooth plastic at 0.24-0.34. Check on real chips by tilting a two-chip stack until the top chip slides (static friction = tan of the angle). |
| Felt | 2 mm wool felt cloth | Typical casino table cloth thickness (estimate). |
| Betting circle, odds lettering | 90 mm circle, 12 mm capitals | Typical blackjack layouts (estimate); print only. |
| Stack sizes | bet of 4, working stack of 20 | A 4-chip ($20) bet pays $30 = 6 red chips at 3 to 2; chips are racked and stacked in twenties. |

## Scoring

Each chip's centre is its body origin, its axis its body z axis, and its tilt the angle between that axis and vertical.
A chip is **flat** at tilt <= 10 degrees. A flat chip is on **level k** when its centre is within 1.5 mm of
0.7495 + (k + 0.5) x 0.003302 (level 0 rests on the felt).

A **stack** is a flat level-0 base chip plus flat chips on levels 1 .. K-1, exactly one per level, each with its centre
within 19.685 mm (one chip radius) horizontally of the chip on the level below, so every chip overlaps the one under
it. A flat chip within 10 mm horizontally of the base chip's axis that is not in the stack (off every level, or a
second chip on a level), or two chips on one level that both overlap the chip below, makes the stack malformed. A
well-formed stack is **neat** when every chip's centre is within 10 mm horizontally of the base chip's axis, and
**untidy** otherwise.

The **bet's axis** and **top face** are measured where the bet stands now: the centre of its top chip (`bet_4`) and
that chip's top face (13.208 mm above the felt at the start). The bet starts at (0.300, -0.240) and must stay within
2 mm of it, or the trial scores 0 (*bet disturbed*, below).

- **Payout.** Among the 20 non-bet chips, the flat level-0 chips whose axis is 39.37 to 54.37 mm from the bet's axis,
  i.e. an edge gap of 0 to +15 mm. There must be exactly one; its stack must be well formed; and no
  other non-bet chip may have its axis within 64.37 mm (54.37 + 10) of the bet's axis. Then the payout is its K chips,
  neat or untidy; otherwise there is no payout. Any side of the bet counts.
- **Working stack.** All non-bet chips not in the payout must form one neat stack whose base axis is within 30 mm of
  (0.200, -0.380). A non-bet chip in neither is **loose** (anywhere: on the felt, on the bare table, or still in a
  gripper).
- **Violations, for the whole trial.** They are checked at every control step; once seen, the trial scores 0 to the
  end, and putting things back never clears them:
  - *capped*: any part of a non-bet chip over the bet's top face and at most 2 mm above it (at most 15.208 mm above
    the felt for an unmoved bet), whatever the chip's tilt. Over the top face means horizontally less than 19.685 mm
    from the bet's axis (inside its rim) for a flat chip, and less than 20.185 mm (the rim plus 0.5 mm) for a chip
    tilted more than 10 degrees. This covers a chip resting on the bet, overhanging its top (a payout chip any amount
    over the bet's rim at the bet's top height), pressed onto it, or leaning on its top or rim from the felt or from
    another stack, even for one step. A chip carried more than 2 mm above the bet's top is not capping. Both
    distances are taken from the bet where it stands, so a bet nudged within its 2 mm tolerance is guarded just the
    same. (In code: 3,240 points through each chip's solid, 5 heights x 9 radii x 72 angles.)
  - *bet disturbed*: a bet chip's centre more than 2 mm (3-D) from its start centre (its resting centre on its level at
    the bet's axis), or tilted more than 5 degrees.
  - *knocked over*: any chip tilted more than 30 degrees (a toppled stack or a chip on its edge).
  - *off the table*: any chip centre below 0.7275 (20 mm under the table top) or outside x -0.19..0.91, y -0.65..0.65.
- **Released** means each gripper's centre point between the pads is at least 30 mm from every chip's bounding box,
  computed from the robot's joint state. **Still** means that over the last 1 s of consecutive control steps no chip's
  centre moved more than 1 mm and no chip's tilt changed more than 1 degree (spin about its own axis is ignored). It
  is judged from poses, not speeds: the contact solver makes the instantaneous speeds of stacked 3.3 mm chips flicker
  (up to 22 mm/s and 0.8 rad/s on the GPU while every centre stayed within 0.31 mm). A static layout without a history
  (the offline scorer tests) uses reported speeds instead: under 0.010 m/s and 0.5 rad/s.

| Score | End-state condition (no best-score latch) |
| --- | --- |
| 0.0 | Any violation during the trial; or no payout, a payout of 4 or fewer or of 8 or more chips, a malformed payout, two stacks beside the bet, or a stray chip within 64.37 mm of the bet's axis |
| 0.25 | Untidy payout of 6 chips (a chip more than 10 mm off the base chip's axis but still overlapping the chip below), or a payout of 5 or 7 chips, neat or untidy: mispaid by one chip, over or under, held or released |
| 0.5 | Neat payout of 6 chips, but still held, moving, or with a loose chip left anywhere |
| 1.0 | Neat payout of 6 chips, no loose chip, all chips released and still, held for 1 s |

Over- and under-payment are graded alike: both are mispays a supervisor would have corrected, so a stack one chip off
earns 0.25 and nothing more, and two or more chips off earns 0. An untidy stack of six is the right amount but has to
be squared up before the player can take it, so it earns the same 0.25, never less than a neat mispay. A policy may correct a mispay before the end (for
example, take one chip back to the working stack); only the end state counts, but violations never clear.
`success()` is `score() >= 1.0`; the trial ends after 1 more second of success (`SUCCESS_HOLD_S = 1.0`).

## Demo

The right arm cuts the top six chips with one pinch and sets them down 45 mm beside the bet (on the dealer's right);
the left arm parks. Every right-arm waypoint turns the grip 90 degrees (yaw pi/2), so the pads close along world x and
stay at least 18 mm from the bet while the payout is set down beside it.

Cut and set-down heights (`PAD_TIP_BELOW_TCP` = 12.84 mm: the deepest finger collider below the TCP, measured from the
YAM collision hulls, whose inner pad face is flat to within 0.05 mm from 0.25 mm above that point up to 16 mm; `Z_CAL`
= 0 is the pad-tip height calibration constant):

- `CUT_AIM` = 0.003302 / 2 = 1.651 mm: the pad tip at the middle of the 6th chip from the top.
- `CUT_TIP_Z` = 0.7495 + (20 - 6) x 0.003302 + 0.001651 = 0.797379; `CUT_TCP_Z` = `CUT_TIP_Z` + 0.01284 + `Z_CAL` =
  0.810219. The pad tip is 1.651 mm above the top of the 7th chip from the top, so, geometrically, a tip error of less
  than 1.651 mm either way still takes exactly six.
- `SET_TCP_Z` = 0.7495 + 0.0015 (`RELEASE_GAP`, slab bottom above the felt) + 0.001651 + 0.01284 + `Z_CAL` = 0.765491.
- Payout axis `P_XY` = (0.300, -0.240 - 0.045) = (0.300, -0.285): a 5.63 mm edge gap to the bet.
- `TRAVEL_Z` = 0.7475 + 0.130 = 0.8775; parks (0.100, -0.300, 0.9075) and (0.100, 0.270, 0.9075).

| Arm | # | Waypoint | Duration s | Ends at s | TCP (x, y, z) | Fingers | Yaw deg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| left | 1 | park | 1.5 | 1.5 | (0.100, 0.270, 0.9075) | open | 0 |
| left | 2 | hold clear | 20.7 | 22.2 | (0.100, 0.270, 0.9075) | open | 0 |
| right | 1 | park | 1.5 | 1.5 | (0.100, -0.300, 0.9075) | open | 90 |
| right | 2 | hover over the working stack | 1.5 | 3.0 | (0.200, -0.380, 0.8775) | open | 90 |
| right | 3 | lower beside the stack | 1.5 | 4.5 | (0.200, -0.380, 0.8102) | open | 90 |
| right | 4 | settle before the cut | 0.6 | 5.1 | (0.200, -0.380, 0.8102) | open | 90 |
| right | 5 | pinch six chips | 1.5 | 6.6 | (0.200, -0.380, 0.8102) | closed | 90 |
| right | 6 | pinch dwell | 0.4 | 7.0 | (0.200, -0.380, 0.8102) | closed | 90 |
| right | 7 | lift the cut clear | 0.8 | 7.8 | (0.200, -0.380, 0.8302) | closed | 90 |
| right | 8 | lift | 1.0 | 8.8 | (0.200, -0.380, 0.8775) | closed | 90 |
| right | 9 | carry | 2.0 | 10.8 | (0.300, -0.285, 0.8775) | closed | 90 |
| right | 10 | lower | 1.2 | 12.0 | (0.300, -0.285, 0.7855) | closed | 90 |
| right | 11 | set down | 1.0 | 13.0 | (0.300, -0.285, 0.7655) | closed | 90 |
| right | 12 | settle | 0.5 | 13.5 | (0.300, -0.285, 0.7655) | closed | 90 |
| right | 13 | release | 1.5 | 15.0 | (0.300, -0.285, 0.7655) | open | 90 |
| right | 14 | release dwell | 0.5 | 15.5 | (0.300, -0.285, 0.7655) | open | 90 |
| right | 15 | rise clear | 1.2 | 16.7 | (0.300, -0.285, 0.8775) | open | 90 |
| right | 16 | park | 1.5 | 18.2 | (0.100, -0.300, 0.9075) | open | 90 |
| right | 17 | hold clear | 4.0 | 22.2 | (0.100, -0.300, 0.9075) | open | 90 |

Every duration is multiplied by `DEMO_PACE` (1.0). Pinch and release take 1.5 s so the grip command stays within the
action guard's 0.05-per-step limit.

On the simulator the 20-chip stack settles within 0.13 mm of its nominal levels, the pad tip reaches the cut height
within 0.06 mm, exactly six chips come away and stay in the pinch (no slip over a 5 s hold and the carry), and the
payout lands aligned within 0.1 mm, 1.0 with success in both episodes of a two-episode run and at 0.8x speed. The same
plan with the pad tip 2.5 mm low takes seven chips, and 2.5 mm high takes five (0.25 each): an error under one chip's
thickness changes the payout, which is why the cut is the difficulty. The planned path stays within 0.323 m of each arm base (limit 0.36 m), with IK error
under 0.1 mm.

## Fidelity

| Simplification | Size | Why |
| --- | --- | --- |
| Chips are sharp-edged uniform cylinders | real clay chips have edges rounded by a few tenths of a mm and moulded edge inserts | the kit's primitives; the inserts are drawn as a visual stripe with no collider |
| Friction 0.40 / 0.35 | estimate within the dry-material range | no published chip measurement; not rubber-like |
| Felt is a rigid 2 mm sheet | real cloth over padding gives under a chip by a fraction of a mm | rigid-body simulation |
| Printing is visual only | real ink has no thickness | nothing can catch on it |
| Distances compressed | real betting circles sit 0.5-0.7 m from the dealer | the arm's 0.36 m working reach; every part keeps its true size |
| One player, no cards, no chip tray | a full table has several seats and a tray of chips | the instruction states the result; the payout comes from the working stack |

## Licence

All geometry is original primitives built with the supplied parts kit (`yam_parts.py`, unchanged); no third-party
assets, meshes or textures.
