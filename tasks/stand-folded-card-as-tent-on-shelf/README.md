# Stand a folded birthday card on a shelf as a tent: turn it crease-up by its front leaf and set both edges down open

Stand the birthday card up on the shelf where it lies, open like a tent with its crease on top and its picture facing you, and let go of it.

A Nitelemn white 5 x 7 in folded tent card (110 lb / 300 g/m2, 13.5 g), its front printed with a birthday picture,
lies closed on a short oak-effect desk shelf over a blue dot. Lying closed, its crease holds the front leaf 11.8 degrees
open. Both hands have to pinch the front leaf at its side edges, turn the card crease-up, set the hanging back leaf's
edge down, carry the crease over it until the front edge lands, and let go of a 13.5 g card standing on two thin
edges.

## Scoring

Judged 10 s after both hands have let go, as a person would with a ruler, a protractor and a
photo from above. In code: `card_terms()`, `released()`, `still()`, `tier()`, `score()` and `success()` in
`stand_folded_card_as_tent_on_shelf.py`.

| Score | Condition (all must hold) |
|---|---|
| 1.0 | standing, open 30 to 90 deg, picture facing the robot (-x) within 20 deg, crease's middle within 40 mm of the dot |
| 0.6 | standing, open 20 to 120 deg, any facing, crease's middle within 80 mm of the dot |
| 0.3 | standing on the shelf crease-up, outside the limits above |
| 0 | anything else: lying, on its side or face, crease down, off the shelf, held, touched or moving |

- Standing (`card_terms`): both free (bottom) edges rest on the shelf, each within 4 mm of its
  top at both ends (y = -88.9 and +88.9 mm along the edge), over the shelf; the crease's middle at least
  40 mm above both edges. Rigid leaves standing on both edges cannot touch the shelf
  anywhere else.
- Opening: the crease hinge's angle (0 = closed). Facing: the front leaf's outward normal projected on the shelf,
  its angle from -x. Crease's middle: the hinge's middle point, its horizontal distance from the dot.
- Let go (`released`): both grippers' centre points at least 10 cm from the card's
  bounding box. In a run, a failed gripper read counts as not let go.
- Still (`still`): over the last 1 s the crease's middle and the free edges' corners each moved under
  1 mm (positions, not solver velocity).
- The score counts only once the card has been standing, let go and still for 10 s in a row
  (`held_for`); before that, and if it falls, it is 0. `success()` is score 1.0, held 1 s
  (`SUCCESS_HOLD_S`); the run's score is the score at its end (`SCORE_OVER = "end"`).
- Real trials only: a new crease (a visible fold line, or a bend that stays over 10 deg when that leaf is laid flat
  on the shelf) scores 0. The simulated leaves are rigid, so this cannot happen in simulation.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=stand_folded_card_as_tent_on_shelf.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Stand the birthday card up on the shelf where it lies, open like a tent with its crease on top and its picture facing you, and let go of it." \
  --scorer task_score --max-steps 1800 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial use `--policy dream`. Each trial allows 1,800 control steps
(90 s at 20 Hz). The official layout is fixed (`object_jitter=0`); `-E object_jitter=0.01`
shifts the card up to 10 mm in x and y. `pytest` in this folder runs the offline checks of the scorer (no GPU).

## The scene

Metres in code, millimetres here; x forward from the arm bases, y to the robot's left, heights above the desk
(table top 0.7475 m). Every collider (box primitives; centres in world x, y and height above the desk):

| Part | Mass | Collider (mm) | Where (mm) |
|---|---|---|---|
| shelf board | fixed | box 220.0 x 220.0 x 18.0 | (285.0, 0.0, 111.0) |
| shelf back panel | fixed | box 18.0 x 220.0 x 102.0 | (386.0, 0.0, 51.0) |
| blue dot | fixed | box 12.0 x 12.0 x 0.1 | (285.0, 0.0, 120.0) |
| back leaf (card base) | 6.75 g | box 127.0 x 177.8 x 1.0 | lying flat, centre (285.0, 0.0, 120.5) |
| front leaf (link `front`) | 6.75 g | box 127.0 x 177.8 x 1.0 + 11 picture patches 0.1 mm | on the back leaf, open 11.8 deg about the crease |
| crease (hinge `front_joint`) | - | y axis at x = 348.5, height 121.0 | 0 to 178 deg; spring 0.0071 N m/rad toward 45 deg; damping 0.0003 N m s/rad |

The card's frame is its back leaf (the articulation is rooted there): origin at the middle of the back leaf's
underside, x toward the crease, y along it. Positions in `task.json`'s setup use the closed card's outline centre,
which is this origin. Friction: card 0.3 static / 0.25 moving, shelf
0.3 / 0.25 (PhysX averages: card on shelf 0.30); the grippers' pads use the
scene default.

## The card and its crease (estimates)

The product (Amazon B0GQXMQW8J) is a 10 x 7 in sheet of 110 lb / 300 g/m2 cover scored and folded to 5 x 7 in: two
leaves 127 x 178 mm, crease 178 mm long; 0.254 x 0.178 m x 300 g/m2 = 13.5 g. No figure exists for a card's crease,
so the crease is a hinge with a spring toward 45 deg at 0.0071 N m/rad, chosen so that:

- lying closed, the front leaf's weight (6.75 g at 63.5 mm: 0.0041 N m) balances the spring at 11.8 deg
  open, its free edge 26 mm up, as the closed card in the task's setup;
- standing, the card holds still between about 36 and 83 deg open on friction 0.30 (each foot's push outward
  against the spring's pull and its weight); outside that band the spring opens or closes it until friction holds;
- hanging from the front leaf turned 60 deg below level, the back leaf hangs 9.4 deg off vertical.

Damping 0.0003 N m s/rad (about a quarter of critical) stands for the card's internal losses.

## Scripted demonstration

Both hands pinch only the front leaf, each at one side edge, gripper pointing along the crease (pads on the leaf's
two faces), so the card stays level between them (one hand alone also managed it in simulation, the card sagging
about 4 degrees in its pinch; see below). The right hand slips its lower finger into the
gap under the front leaf at the -y edge, 105 mm from the crease, pinches and opens the card to 22 deg about its
crease; the left hand pinches the +y edge 75 mm from the crease; the right lets go and pinches again 75 mm from the
crease. Together they lift the card and turn the front leaf from +22 to -60 deg about the line between the pinches
(crease up); the back leaf's edge drags on the shelf, then swings under the crease and hangs. After 1.5 s they read
its angle, lower its edge onto the shelf 63.5 mm behind the dot, then carry the crease round that edge in four steps
until the front edge lands: open 60 deg, crease over the dot. They open, slide out 20 mm along the crease, rise and
return to the ready pose. The demo reads the card's pose (allowed for scripted demos) and never moves an object.
Plan 46.3 s long; the 10 s count reaches 1.0 at 41.3 s and the run stops there.

## Checked in simulation

Version `bc128a646467`, `yam_scripted`, official layout (`object_jitter=0`): score 1.0, success at step 825 of 844 (41.3 s). The card ended standing on both free edges, open 59.6 deg, its crease over the dot and its picture to the robot, both arms back at their ready pose.

The same plan on a diagnostic copy of this file (it only adds prints), three episodes with the card shifted up to 10 mm (`-E object_jitter=0.01 --seed 3`): 1.0 in all three. Open 59.6-59.8 deg, crease 0.3-0.5 mm from the dot and 110 mm above the shelf, standing, let go and still for 10.7-10.8 s at the check. The right hand opened the card to 21.2-21.7 deg before the left pinched; turned crease-up, the back leaf hung 8.9-9.0 deg off vertical (9.4 deg computed from the crease estimates).

Dream-1 (1,800 steps): 0.0. Both hands came to the card's sides and nudged it, but never opened, lifted or turned it; it ended lying closed on the shelf.

## Shortcuts tried in simulation

Run on a GPU-only copy of this file with the plan cut short (two episodes, official layout):

- Let go as soon as the card is lifted, front leaf level and crease 54 mm up with the back leaf's edge still on the shelf: score 0. With nothing holding the crease up, the back leaf folds under and the card falls back to lying closed (open 11.8 deg, crease 71 mm from the dot), not standing.
- Let go in the air once crease-up, the back leaf's edge hanging 12 mm above the shelf: score 1.0. The card drops onto that edge and tips forward onto the front edge, ending open 41 deg with the crease 23 mm from the dot. So the last centimetre need not be placed; turning the card crease-up over its back edge is the part that matters.

Also measured: one hand alone (the left let go after the regrip, two episodes) carried the card through the turn and stood it, 70 deg open, the crease 14-15 mm off the dot; the card sagged about 4 deg and 11 mm at its far end in the single pinch.

## Simulation and the real objects

The card is two rigid leaves (true size and mass) on a hinge with a spring and damper for the crease; the spring's rest angle and stiffness are estimates set so the card lies open 11.8 degrees. Real leaves bend a little; these do not, so no new crease can form in simulation. Each leaf's collider is 1.0 mm thick (real 0.36 mm) for stable pad and edge contacts. The picture is thin coloured patches. Card-on-melamine friction is an estimate. The shelf is scenery (fixed). It is 220 mm long, 21 mm longer than the card at each end: a hand pointing along the crease has its gripper body 51 mm below its fingers, and that body passes beyond the shelf's ends while the fingers pinch the card's side edges just above it. The robot's pads close with about 2.5 N on the 1.0 mm leaf collider
(the real 0.36 mm card would get about 0.9 N); the card's 13.5 g weight is far below that grip's limit, so the
pinch holds the card by friction either way. The card is white like the product (each leaf 6.75 g, its true
share of the 13.5 g); the shelf is oak-effect so the card shows against it in the top camera.

## Reset

Close the card, lay it on its mark (middle over the dot, crease along y away from the robot, picture up; within 1 cm and 5 degrees) and leave it 30 s so the crease settles to its lying gape. Use a new card from the pack when the lying gape falls under 8 degrees or the card is marked or bent. Zero object jitter.

## Files

- `stand_folded_card_as_tent_on_shelf.py`: scene, scorer, scripted demonstration.
- `task.json`: instruction, setup, scoring, run limit, fidelity.
- `test_stand_folded_card_as_tent_on_shelf.py`: offline scorer checks (generated layouts).
- `yam_parts.py`, `yam_joints.py`: the shared parts kit (unchanged).
