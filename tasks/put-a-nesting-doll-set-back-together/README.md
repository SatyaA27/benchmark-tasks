# Put a five-piece wooden nesting doll set back together, pressing each snug top home with its face lined up

This task builds on the task folder `nesting-doll-reassemble`: its scene, doll outline, sizes and score ladder are
reused with credit. What changed: each top is now a snug friction fit that stops on its spigot under its own weight
and goes home only when pressed (there the tops seated by their own weight in a 1.5 mm sliding fit), "closed" now
needs the top pressed home (seam gap 2 mm or less), the layout lies within one arm's reach, and the scripted demo
presses both tops home.

Instruction given to the policy:

`Put the nesting dolls back together, each closed with its own top pressed home and its face lined up, the smaller ones inside the largest.`

## Scoring

Everything is read at the end, from the video and a photo, with both grippers at least 30 mm from every piece.
Score 0 if any piece ever fell off the table, or if a gripper is still on the set. Otherwise the score is the highest
of these levels that holds:

- **Inside:** a doll, or a bottom half, stands on the floor of the next bottom half, within 3.5 mm of its axis, both
  upright within 8 degrees.
- **Closed:** a top sits on its own bottom with a seam gap of 2 mm or less at the waist (a ruler or feeler card in a
  side photo), centred within 2.5 mm and square within 5 degrees, its painted face within 15 degrees of the apron
  painted on the bottom (a protractor card on the top photo). A top resting on the spigot without being pressed
  stands about 4 mm up and is not closed.

| Score | Condition |
|---|---|
| 1.0 | doll 3 inside doll 2; doll 2 closed and inside doll 1; doll 1 closed and upright within 5 degrees; no piece moves more than 1 mm over the last 1 s |
| 0.75 | doll 2, closed with doll 3 in it, stands inside doll 1 (also a finished set still moving) |
| 0.5 | doll 2 is closed with doll 3 in it |
| 0.25 | doll 3 is inside doll 2's bottom half |
| 0 | anything else; a top on the other doll's bottom, or resting unpressed above its spigot, does not count as closed |

The levels nest, so every outcome has exactly one score, and the start (all pieces apart) scores 0. In a real
trial, doll 1 is opened after the run to confirm what is inside; the video shows the order.

## Scores

- Scripted demo: 1.0, success at step 1,480 (74 s) of 3,000. Before it, the same demo scored 1.0 in three of three
  episodes with every piece shifted up to 10 mm at the start: after the crown press the seam gaps were 1.6 mm or less
  and the faces within 2 degrees of the aprons.
- Wrong way (each top set on its own bottom, face lined up, and let go without pressing; doll 2 left out of doll 1):
  0.25. Both tops rest about 4 mm above home on the spigots, so neither counts as closed; only doll 3 inside doll 2
  scores.
- Dream-1 run (3,000 steps): 0.0.

## The fit

A real set's halves are turned to a snug friction fit: the top goes on by hand with a firm push and holds the bottom
and the dolls inside when the closed doll is lifted by its head. Nobody publishes the force. We estimate a few newtons
for a 15 cm set; a lab can check its own set in a minute by pressing a top home on a kitchen scale and reading the
peak. In the simulation each top carries four sprung pads inside its skirt (0.8 N each at rest, 1.0 N spread, with a
20 degree lead-in, friction 0.5): a top set on the spigot stops with its rim about 4 mm above the waist (the pads need
several newtons to spread, the top weighs 0.08-0.16 N), and goes home only when pressed, the pads spreading 0.8 mm. The task is unchanged for any
real fit between about 1 N and 8 N to press home (heavier than the top, lighter than the arm can push); the arm's
stiffness gives about 4-8 N on a 10 mm press. These forces are estimates from the design, not readings. A top pressed on crooked jams on the pads and the spigot instead of
going home. The closed doll 2 can be carried by its head because the fit holds about 2 N against the 0.4 N of the
bottom and doll 3.

## Run a trial

In the Isaac Lab-Arena environment, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream -E task=put_a_nesting_doll_set_back_together.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the nesting dolls back together, each closed with its own top pressed home and its face lined up, the smaller ones inside the largest." \
  --scorer task_score --max-steps 3000
```

The scripted demo is the module's `demo()` (policy `yam_scripted`, same command with `--policy yam_scripted`). It
reads each piece's pose when it first reaches for it and never moves an object. One hand does everything:

1. the third doll, pinched by its head, is lowered into doll 2's bottom half and let go 3 mm above the floor;
2. doll 2's top, pinched by its head, is turned so its face lies over the apron and set on the spigot, where it
   stops; the hand lets go, closes its fingertips and presses the crown 10 mm past home;
3. the closed doll 2 is lifted by its head (the fit carries its bottom and doll 3) into doll 1's bottom half;
4. doll 1's top is turned, set on and pressed home the same way.

Every pinch closes 3 mm inside the piece's width, with the fingertips above the wider shoulders.

## Runtime and seeds

3,000 control steps (2.5 minutes at 20 Hz) per trial; the scripted demo takes about 81 s. One fixed layout
(`-E object_jitter=0`); there is no other randomness.

## Files

| File | What |
|---|---|
| `put_a_nesting_doll_set_back_together.py` | scene (the dolls, reusable section), `score()`, `success()`, `demo()` |
| `yam_parts.py`, `yam_joints.py`, `yam_mesh.py` | the shared part, joint and mesh kit, unchanged |
| `test_put_a_nesting_doll_set_back_together.py` | offline scorer tests (no simulator) |

## Fidelity

- **Shape:** each half is a rigid lathed body (48 sides) of a classic outline with a signed-distance collider; walls
  2.5 mm, floors 4 mm, spigot 2 mm thick and 8 mm tall; the top's skirt is 1.5 mm clear of the spigot, the pads take
  up the gap.
- **Fit:** four sprung pads per top stand in for the wood's snug grip (estimate, above). Wood springback, wear and
  crushing are not modelled.
- **Mass:** linden at 0.45 g/cm3. Doll 3 and the two inside it are one solid body.
- **Paint:** the face, shawl, hair and apron are thin raised strips. **Friction:** 0.5 (varnished wood, estimate).

## Real-world build

- **Set:** Lindenhaus Imports 5 pc. Cinderella Matryoshka Nesting Doll Set, item 156058, 6 in tall, linden wood,
  hand painted, US$67.95 (https://lindenhausimports.com/5-piece-cinderella-matryoshka-nesting-doll-6/). A cheaper
  stand-in is their 5 pc. blank set, item 100053, 6 in, linden, US$24.95
  (https://lindenhausimports.com/5-piece-diy-blank-matryoshka-nesting-doll-6/), with a face and apron marked by paint
  pen. The maker publishes the height and the piece count only; the widths (71, 52, 34 mm), the inner dolls' heights
  (110, 72 mm), walls and masses are estimates: check them with a ruler and a kitchen scale.
- **Layout:** stand the pieces on printed outlines at the positions in `task.json`, the faces turned to the marks.
- **Checks:** a top photo for the face angles (protractor card), a side photo for each seam gap (2 mm feeler card),
  then open doll 1 to confirm the contents.
- **Hazards:** none. The pieces are smooth varnished wood; nothing sheds on the grippers.
