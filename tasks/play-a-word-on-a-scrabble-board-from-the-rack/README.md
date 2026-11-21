# Play a word on a Scrabble board from the rack: lay four tiles face up in their squares without jostling the others

Play the word PLANT down the board through the A that is already there: take the tiles P, L, N and T from the rack and lay each one face up, the right way round, in its square, without knocking the tiles already on the board out of place.

## Scoring

Judged at the end of the run from a top photo and a ruler (in simulation from the poses). A played tile counts when it lies flat on the board face up, its centre is inside its own square (the square named in the setup), its letter is turned no more than 20 degrees from upright as the player on the robot's side reads it, no gripper is within 3 cm of it and it has not moved in the last second. Score 0 if none of P, L, N, T counts. Otherwise the score is the lowest of three numbers. Tiles: 1.0 if all four count, 0.7 if three, 0.4 if two, 0.2 if one. Board: measure how far each of the ten tiles that started on the board is from where it started (a tile turned over or no longer flat on the board is over 19 mm): 1.0 if none is over 5 mm; 0.5 if exactly one is over 5 mm and it is not over 19 mm; 0.2 otherwise. Rack: 1.0 if O, E and I all still stand on the rack, 0.6 otherwise. Doing nothing scores 0; tiles laid face down do not count; four tiles laid correctly while a starting tile was pushed a square away scores 0.2.

How each check is made on the real rig: one photo from straight above at the end, compared with the start photo.
A played tile "lies flat face up" if its letter is visible and it rests on the board, not on another tile. "Inside
its own square" means the middle of the tile (where its diagonals cross) is inside the printed outline of the square
named in the setup; with 19 mm tiles in 23 mm squares that is up to 11.5 mm off centre either way. The letter's turn
is read with a protractor against the board's grid lines. For the board number, lay a ruler from each starting
tile's place in the start photo to its place now. A spare tile "stands on the rack" if it still leans in the rack's
groove. No force or sub-millimetre reading is needed. In simulation the same checks are read from the poses: a tile
is flat and face up when its printed face's normal is within 15 degrees of straight up and its centre is within 2
mm of its resting height on the board; a gripper is clear when its tool point is 30 mm or more from the tile's
centre; still means it moved under 0.5 mm and turned under 2 degrees over the last second.

In our recorded demonstration (no start shift) the right hand takes T, N, L and P off the rack one at a time by their side edges, turns each face up and lays it in its square; full credit at 62 s. At the end the four tiles lie flat with their centres 0.4, 0.9, 1.8 and 1.6 mm from their squares' centres (the limit is 11.5 mm each way), letters upright within 2 degrees, none of the ten starting tiles moved (0.0 mm), and O, E and I still stand on the rack (the rack itself ended 1.8 mm from its spot). Three seeded trials with the rack and its tiles shifted up to 5 mm and the board tiles up to 1.5 mm: all three reached full credit.

Wrong way, tried in the simulator: T and N were set down flat two squares beyond the word and slid along the board down column 11 to their squares, through the A. The T pushed the A ahead of it and left it 64 mm from its square, on the board's border; all four tiles ended in their own squares, so the tiles number is 1.0, but the board number is 0.2: score 0.2.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=play_a_word_on_a_scrabble_board_from_the_rack.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Play the word PLANT down the board through the A that is already there: take the tiles P, L, N and T from the rack and lay each one face up, the right way round, in its square, without knocking the tiles already on the board out of place." \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial use `--policy dream`. Each trial allows 2,400 control steps (120 s at 20 Hz). `pytest` checks the
scorer on named layouts. With a seed and object_jitter above 0, the rack and its seven tiles start shifted together
by up to 5 mm each way and every tile on the board by up to 1.5 mm inside its square.

## What the robot has to do

Seven tiles lean in the rack's groove, 8.5 mm apart, faces toward the robot. A tile is 19 mm square, 4 mm thick and
weighs about a gram; only its two side edges and the top few millimetres are free to hold, because its back lies on
the rack and its bottom edge sits in the groove. The scripted demonstration brings the hand in from the robot's
side with the fingers pointing square at the tile's face (30 degrees below level), slips the two fingertips past
the tile's side edges, between it and its neighbours, until their ends are just short of the rack, and pinches the
edges with the last 3 mm of the pads. It draws the tile off the rack, lifts it and turns the hand until the fingers
point straight down: the tile now hangs level across the fingertips, face up, its letter upright for the player.
Over the square the tile is lined up above the neighbouring tiles, lowered between them (2 mm to spare on each
side along the word) and let go on the board. T, N, L and P are played one at a time with one hand; the other arm
stays parked. Any other way of getting the tiles there is fine: only the end state is scored.

## Real objects (sources fetched; nothing here was measured)

- Hasbro Scrabble Classic Crossword Game (model F4204): gameboard, 100 wooden letter tiles, 4 wooden racks, bag
  (price: see the listings; an estimate is 20 to 25 US dollars). Retail listings: https://www.amazon.com/Hasbro-Gaming-Scrabble-Classic-Refresh/dp/B09D44JH4S
  and https://www.maziply.com/products/scrabble-classic-crossword-board-game .
- Tiles "19 by 19 millimetres square" and "4 mm thick, slightly smaller than the squares on the board"; a 15 x 15
  board with red triple word, pink double word, blue triple letter and light blue double letter squares and a
  centre star: https://en.wikipedia.org/wiki/Scrabble . Standard English tile values (P 3, H 4, X 8, the others
  here 1).
- Wooden racks for these tiles are listed at 7.5 x 0.7 x 0.9 in (190 x 18 x 23 mm):
  https://www.amazon.com/JUNWRROW-Wooden-Scrabble-Replacement-inches%EF%BC%89/dp/B09W153J7T . Another guide gives
  7 in for the rack and 1 in for a square: https://measuringstuff.com/scrabble-board-sizes-standard-and-travel/ .

## Estimates (not published; how to check each in a minute)

Square pitch 23 mm (ruler across ten squares; if the real pitch differs, the play squares move with it and the
gaps between tiles change, nothing else). Board 380 mm square, 2 mm thick (ruler). Tile mass 1.0 g (ten tiles on a
kitchen scale). Rack 190 x 23 x 18 mm, 31 g, groove floor 7 mm wide meeting the rest 10 mm behind the front face
and 4 mm up, tiles leaning 30 degrees back and standing about 3 mm above the rack's top (ruler, phone level on a
standing tile, kitchen scale). Friction of wood on the printed board and wood on wood 0.35 (tilt the board until a
tile slides: about 19 degrees); rack on the table 0.40.

## Simulation model

- Units metres and kilograms; x forward from the arm bases' midpoint, y left, z up; table top z = 0.7475.
- `board` (fixed): one 380 x 380 x 2 mm box. Its 225 squares, grid lines and premium colours are prints.
- 17 tiles (loose rigid bodies, 1.0 g each, convex 19 x 19 x 4 mm blocks, contact offset 0.5 mm): ten on the board
  (`board_<letter><row>_<column>`), seven on the rack (`tile_P`, `tile_L`, `tile_N`, `tile_T`, `spare_O`,
  `spare_E`, `spare_I`). Letters and point values are prints on the face: black strokes 0.05 mm proud, with no
  collider.
- `rack` (loose rigid body, 31 g): a 4 mm base slab plus two convex bars that form the groove.
- Module `tile_kit.py` (reusable): printed parts, loose bodies that the seeded reset does not shift one by one, the
  stroke font. `yam_parts.py`, `yam_joints.py`, `yam_mesh.py` are the kit modules, unchanged.
- The demo reads tile poses (allowed for scripted demos) to line the hand up; it never moves an object by any other
  means than the gripper.
- Credit: the run-context, gripper-reading and still-window helpers, the grip-matrix patch and the live waypoint
  class in the module are adapted from the task folder slide-a-mic-into-its-desk-stand-clip-and-aim-it.

## Fidelity

Rigid bodies. Tiles are plain 19 x 19 x 4 mm blocks with sharp edges (real ones are slightly rounded) and uniform density; letters, point values and the board's squares are print only. The board is one flat, fixed sheet: its fold and any surface texture are not modelled. The rack is a loose wooden bar with a straight groove. Square pitch, board and rack sizes, the groove's shape and lean, masses and frictions are estimates.

## Reset

About 2 minutes. Keep a printed top photo of the start as the card. Lift the four played tiles off the board. Put any starting tile that moved back in the centre of its square, face up, upright for the player on the robot's side: THRONE down column 7 from the centre star, EXTRA along row 2 from its E. Set the rack back on its two tape marks (front face along x = 105 mm, centred between the arms) and stand the seven tiles in its groove, faces to the robot, in the order P L N T O E I from the robot's left, 27.5 mm apart with the T in the middle (a paper strip with seven ticks laid along the rack gives the spacing). Check from above that the board's near edge is still on its tape line. Return both arms to the ready pose. Nothing is used up; the set has spare tiles for any that get lost (a set costs about 20 to 25 US dollars, an estimate).
