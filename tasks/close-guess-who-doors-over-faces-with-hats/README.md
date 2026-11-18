# Take a turn at Guess Who: shut the doors over the five faces wearing hats without shutting any other face

The other player has answered: 'My person is not wearing a hat.' Shut the door over every face that is wearing a hat and leave all the other doors open.

## Scoring

Judged at the end of the run from a photo of the board, taken once the robot has let go. A door is **shut** when it
lies against the sheet over its face. A door has only two resting places: open on its stop, or shut against the sheet; one lifted less than upright
falls back open by its own weight. Count H = hat faces shut (0 to 5) and W = other faces shut (0 to 19), and
net = H - 2 x W:

| net | score |
| --- | --- |
| 5 (all five hats shut, nothing else) | 1.0 |
| 4 | 0.75 |
| 3 | 0.5 |
| 1 or 2 | 0.25 |
| 0 or less | 0 |

Whatever the doors show, the score is 0 if the board was **pushed** (its front edge more than 5 cm from where it
started; mark the start with tape on the table) or **tipped** (its base more than 20 degrees off the table, i.e. it
fell on its back or face). At the start no door is shut, so doing nothing scores 0.

How `score()` reads each condition: a door is shut when its hinge has turned past upright, 90 degrees from open (it
then lies on its stop against the sheet at 115 degrees); the five hat doors are the persons in `HATS`. Pushed: the
board's origin (front edge, centre) more than 50 mm across the table from where it stood at the first scored step
of the run. Tipped: the board's vertical axis more than 20 degrees from vertical. `success()` is a score of 1.0,
held for 1 s. A door still touching a gripper when the run ends counts on whichever side of upright it is, which is
where it falls when released.

Why the shortcuts fail: every wrong face costs two right ones, so shutting a whole row, a whole board or the faces
around each hat scores 0; there is nothing to gain from pushing or tipping the board. A brushed neighbour that
falls back open is not a mistake, as in the game. Our wrong-way run shuts the four front-row doors in front of the
right hand in turn, one of them a hat face (person 24): H = 1, W = 3, net = -5, score 0.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=close_guess_who_doors_over_faces_with_hats.py -E object_jitter=0 -E control_hz=20 \
  --instruction "The other player has answered: 'My person is not wearing a hat.' Shut the door over every face that is wearing a hat and leave all the other doors open." \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial use `--policy dream`. Each trial allows 2,400 control steps (120 s at 20 Hz); the scripted
demonstration takes 54 s. `pytest` here checks the scorer on named layouts and the demo's reach.

## The scene

The product is one player's board from **Hasbro Guess Who?, item F6105**, the current edition in which the faces
are printed on two slide-in character sheets and 24 doors close over them (three rows of eight; a narrow selector
strip above them is not used here). Sources:

- Maker's instructions page and English rules (they show the board, the sheet and the doors):
  https://instructions.hasbro.com/en-us/instruction/guess-who-original-guessing-game-board-game-for-kids-ages-6-and-up-for-2-players
  and https://instructions.hasbro.com/api/download/F6105_en-us_guess-who-original-guessing-game-board-game-for-kids-ages-6-and-up-for-2-players.pdf
- Retail listings: https://www.nda-toys.com/product/17900/guess-who-wholesale (F6105, box 245 x 292 x 41 mm,
  RRP GBP 14.99) and https://www.argos.co.uk/product/3904323 (GBP 14.00).

The maker publishes no board or door dimensions, so every size below is an **estimate** scaled from the manual's
drawing inside the published box size. Each takes a minute to check on the real game:

| Figure (estimate) | Value | One-minute check |
| --- | --- | --- |
| sheet | 270 x 220 mm | ruler across the sheet |
| door | 29 wide x 38 tall x 2 mm | ruler across one door |
| gap between doors / column pitch | 3 mm / 32 mm | ruler across a door and a gap |
| row pitch along the sheet | 41 mm | ruler from one hinge line to the next |
| sheet tilt on its rear panel | 65 degrees from the table | side photo against a set square |
| lowest hinge above the table | 14 mm | ruler stood on the table |
| board mass / door mass | 250 g / 3 g | kitchen scale (board); door from its size in plastic |
| force to lift an open door at its free edge | about 0.016 N (the weight of 1.6 g) | it lifts at the lightest touch |
| open door rests on its stop sticking out level; no click; past upright it falls shut | from the manual's play picture | open one door and let go; lift another past upright and let go |

If the real doors click into place or rest at another angle, the model's two resting places and the "past upright"
rule are what to adjust.

Positions are in task.json's `setup` (mm, from the midpoint between the arm bases): the board's front edge is
centred 160 mm in front of the bases, the sheet faces the robot, all 24 doors start open (sticking out towards the
robot) and nothing else is on the table. The simulation's layout is fixed (`object_jitter=0` for the demo).
Seen from straight above (the top camera, as in the screenshot) the open doors hide the two lower rows of faces;
as for a player, they are seen from the front, here through the wrist cameras.

### The character sheet

The faces are our own plain drawing, not the game's artwork: each of 24 numbered people has a shirt, a face and
hair, except five who wear a brimmed hat instead of hair. Persons are numbered from the top row, left to right as
the robot sees the sheet. The five hats are chosen by a seeded shuffle, `random.Random(2).sample(range(24), 5)`
plus 1 in Python: persons **2, 3, 6, 12 and 24** (top row 2nd, 3rd and 6th; middle row 4th; bottom row 8th).
`python character_sheet.py --seed 2 --out sheet.svg` prints the same drawing the simulation uses at true size
(check its 100 mm bar with a ruler), with small person numbers in the cell corners that the doors hide; cut it to
the game's own sheet and slide it in. Other seeds give other layouts for extra practice, but a scored trial uses
seed 2, as in the screenshot.

## Scripted demonstration

Open loop, reading the layout from the task file. The hand points forward and 15 degrees down with its fingers
opened wide (94 mm, one above the other), so only the lower fingertip meets a door. For each hat door it comes in
from the front, puts that fingertip under the door 4 mm inside its free edge, lifts the door through upright in
10-degree steps until it falls shut against the sheet, and backs off towards the robot. The right hand shuts the two
hat doors on the right half (top row 6th from the left, bottom row 8th), parks, and then the left hand shuts the
three on the left half (top row 2nd and 3rd, middle row 4th) and parks.

In the recorded demonstration all five hat doors were shut and full score was reached at 47 s. At the end of the run
the hinges read 115 degrees (shut) for the five hat doors (top row 2nd, 3rd and 6th; middle row 4th; bottom row 8th)
and 0.0 degrees (open) for all 19 others; two other doors
brushed by the upper finger (up to 7 degrees) fell back open to 0, and the board moved 2.3 mm. In our earlier probe
of the motion (three episodes with 2 mm of seeded jitter) the fingertip shut a single back-row and a single
front-row door every time with the same brushes falling back. The wrong-way run shut its four doors as planned
(one hat, three others) and scored 0.

## Simulation and the real objects

- Rigid bodies. The board is one free articulation: the board body (sheet, rear panel, front edge) and 24 doors,
  each its own 3 g body on a passive hinge along its lower edge with light stiction (5e-5 N m) and damping, no
  spring and no detent. The doors' two resting places come from their weight and the hinge stops, as on the real
  board: open on the stop, shut against the sheet once past upright.
- The faces are thin coloured plates on the sheet (shirt, face, hair or hat), the same shapes the printable sheet
  draws; printed numbers are on paper only.
- In simulation "shut" is read from the hinge angle (past upright); in a real trial from the photo. A door the robot
  leaves exactly balanced at upright is not expected; the rule counts it as open below 90 degrees.
- The real board's selector strip, second character sheet and score counter are not modelled; they are not touched.

## Reset

About 1 minute. Open any shut door with a fingertip so that all 24 stick out towards the robot, check that every
face shows, and stand the board back on its tape marks. Nothing is consumed; the printed sheet stays in.
