# Play a cassette on a shoebox recorder: pop the door open, slide the tape into it side A up, click it shut, press PLAY

Put the cassette into the recorder so that side A will play, shut the cassette door and press PLAY.

Cassettes are sold new again, and homes, schools and libraries still play tapes and dictation on portable 'shoebox' recorders. Loading one is a chain of small mechanical steps. Pressing STOP/EJECT pops the sprung door up about 3 cm. The cassette must then be slid into the holder under the tilted door the right way round, side to play facing up and the open tape edge toward the keys (where the heads are), between rails that leave about 1 mm on each side. The door has to be pushed down until its catch clicks, and PLAY pressed all the way until it locks. A cassette left beside the deck, put in backwards, a door that has not clicked or a key not pressed home and the reels do not turn.

## Scoring

Judged at the end of the run. In the real world: look through the door's window and at the keys, and check the
recorder against its taped outline with a ruler.

| End state | Score |
|---|---|
| Cassette seated side A up, door shut, PLAY locked down (the reels turn) | 1.0 |
| The same with side B up; or seated side A up with the door shut but PLAY not locked | 0.75 |
| Cassette seated, any other state (door not shut; or side B up without PLAY) | 0.5 |
| Not seated, but the door is open at least 5 degrees; or the cassette is in the holder backwards | 0.25 |
| Anything else, doing nothing included | 0 |

If the recorder ends more than 3 cm from its taped mark, turned more than 15 degrees or tipped more than 10 degrees,
the score is at most 0.25 (a shove the real machine would not take).

- Seated: the cassette lies fully in the door's holder: its open tape edge faces the keys and is within 4 mm of the
  door's front edge, centred between the rails (3 mm) and resting on them (2.5 mm). Simulation: the cassette's centre
  in the door's own frame against the seated position; tape-edge and label directions within 25 degrees.
- Shut: the door is clicked down flush with the top and stays there with no hand on it (simulation: within 1 degree
  of shut, where the catch holds it).
- PLAY locked: the PLAY key stays down by itself (simulation: at least 6 mm of its 7 mm travel, where the lock holds).
- Side A / side B: which label faces up through the window. Backwards: tape edge toward the hinge.
- `success()` is a score of 1.0, held for one second. The tiers do not overlap and cover every end state; the start
  (door shut and empty, cassette on the table) scores 0.

Why shortcuts fail: PLAY pressed without a seated cassette, or with the door open, never reaches 1.0; a cassette
left on the table scores at most 0.25 (door opened); one put in backwards or a door that has not clicked stops at
0.25 or 0.5. Measured in simulation: the same demonstration with the door left open before PLAY is pressed ends with the cassette seated side A up and PLAY locked down, the door still up at 22 degrees: 0.5.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=play_a_cassette_on_a_shoebox_recorder.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Put the cassette into the recorder so that side A will play, shut the cassette door and press PLAY." \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model policy, replace `--policy yam_scripted` with the policy. A trial is at most 2,400 control steps (120 s at
20 Hz); the scripted demonstration takes about 36 s. `pytest` in this folder runs the offline checks of the scorer.
The demonstration reads where the recorder and the cassette are before acting, so it also runs with a small
`object_jitter` (it was probed at 0.01).

## The scene

Positions in cm from the midpoint between the two arm bases (bases 61 cm apart), x forward, y to the robot's left. (1) A Jensen MCR-100 portable cassette player/recorder (silver and black plastic, 24.0 x 14.2 x 4.6 cm, 1.8 lb plus four C alkaline cells, about 1.08 kg), standing on its four rubber feet on the bare table with its keys toward the robot, the middle of its footprint at x = 24, y = -5, its long side along x; door shut and empty, no key down, running on its batteries. Keys from the robot's left to right: RECORD, PLAY, REWIND, FAST FWD, STOP/EJECT, PAUSE. Its window door is hinged along the edge away from the keys: STOP/EJECT pops it up and the cassette slides into a holder under the door, open tape edge toward the keys (the published photos show the door shut; check this on the unit). Mark its outline on the table with tape. (2) A Maxell UR-90 compact cassette (black plastic IEC 60094 shell, 10.04 x 6.38 x 1.2 cm, about 42 g; side A's paper label covered with an orange sticky note, side B's with a blue one) standing upright on its long label edge directly on the table, its middle at x = 20, y = -21.5, its long side along y, side A facing away from the robot, open tape edge up.

The recorder (cassette_recorder.py) is one articulated body, loose on its feet: the case (lower shell, speaker end,
the sides of the cassette well, the label strip and the key surround), the window door on a hinge along its rear edge,
and six keys on slides. Under the door's 2 mm plate hangs the cassette holder: two side walls 102.4 mm apart (1.0 mm a
side around the shell, the front 4 mm flared out to 109 mm), two rails under the cassette's ends 13.2 mm below the
plate (the front 4 mm bent down 4 mm), open at the front and closed by a back stop 63.8 mm in. Seated, the cassette's
tape edge is level with the door's front edge. The door pops up to 22 degrees (its front edge about 28 mm up), so the
cassette goes in along a 22 degree slope, label edge first.

## Forces and how to check them on the real recorder

None of these is published; all are estimates. Each can be checked in about a minute with the recorder standing on a
kitchen scale (press slowly, read the peak, subtract the recorder's weight):

| What | Model | Check |
|---|---|---|
| Key springs | 300 N/m, 3.0 N at rest, 5.1 N at the 7 mm stop | press a key that does not lock (PAUSE) |
| STOP/EJECT | door released at 4.5 mm: about 4.4 N | press until the door pops |
| PLAY | lock ramp 4.5-6 mm, about 6.7 N peak; locked past 6 mm until STOP/EJECT | press until it locks |
| Door spring | 0.05 N m/rad: 0.061 N m shut, 0.042 N m at 22 degrees | the door must pop fully open with a cassette in it |
| Door catch | about 2.8 N at the front edge to click shut, 2.5 N hold | push the open door shut at its front edge |
| Door pop-up | 22 degrees, front edge about 28 mm up | ruler at the front edge after STOP/EJECT |
| Holder | 102.4 x 13.2 mm inside, 4 mm lead-ins | calipers inside the open door |
| Key pitch | 17 mm, keys 15 mm wide, 7 mm travel | ruler |
| Cassette | 42 g | kitchen scale |

The door spring, the door catch and the PLAY lock are force rules applied between the door or key and the case
before every physics step, as the real springs and catches act; the arm still has to press the key down, slide the
cassette in and push the door down with about the real forces to trigger them.

## Scripted demonstration

Right arm only; the left arm stays at its start. Run on this version (no jitter): full score at 31.5 s; the run stops once it has held for 1 s, with the hand still at the keys. The cassette slid home by itself on the rails when let go, the door was pushed shut in 1.2 s and caught, PLAY locked at 7 mm, and the recorder did not move. Without a hand: in the probe runs the door stayed at 0 degrees for 4 s while the hand went to PLAY, and in the wrong-way run, which goes on to the park pose, PLAY stayed down at 7 mm for 4.4 s with the hand 25 cm away. The same plan scored 1.0 in three episodes with the recorder and cassette shifted up to 10 mm (it reads both first).

1. Closed fingers come down on STOP/EJECT and press it to its stop; the door pops up.
2. The open hand comes down over the standing cassette with the pads either side of its faces 10 mm below the tape
   edge, pinches it (pads sent 2.5 mm inside the 12 mm shell), lifts it, and turns it 68 degrees on the way so that it
   lies along the door's slope, side A up, tape edge toward the keys.
3. It holds the label edge 15 mm in front of the holder, reads where the cassette really is and corrects, slides it in
   until the fingers are 2 mm short of the door, lets go, and pushes the tape edge home with the closed fingertips
   held below the door plate.
4. Closed fingers come down on the door's plate 10 mm behind its front edge and push it down past the click.
5. Closed fingers press PLAY to its stop; it stays down. The hand moves clear.

## Sources

- Jensen MCR-100 maker page (Spectra Merchandising): https://www.spectraintl.com/index.php/jensen-products/item/mcr-100
  ("shoebox" style, top loading cassette deck, four C cells or the AC adaptor; product photos for the key layout).
- MCR-100 user manual: https://www.spectraintl.com/images/stories/pdf/jn/UM/MCR-100_EIB.pdf (STOP/EJECT "Press to stop
  all deck operations. Press again to open the cassette door."; "Put in a cassette tape with the side to play facing
  out, and close the cassette door.").
- Size and weight: Best Buy listing Q&A, https://www.bestbuy.com/site/questions/jensen-cassette-player-recorder-silver/5657390
  (width 5.59 in, depth 9.45 in, 1.8 lb); height 1.8 in from the Beach Audio listing,
  https://www.beachaudio.com/jensen-mcr100-cassette-player-and-recorder-with-one-touch-mcr100_9/ (about 47 USD).
- Compact cassette shell 100.4 x 63.8 x 12 mm: IEC 60094-7. The Maxell UR-90 is a common blank tape; its price
  (about 2-8 USD a tape) and mass are estimates (retail pages could not be fetched; any blank C60/C90 serves).
- Cost of the real objects (estimate): recorder about 47 USD, cassette under 10 USD, four C cells and two sticky
  notes: under 80 USD.

Estimates, as listed above: the door's hinge side, pop-up angle and holder (the published photos show the door shut;
built as the common shoebox design: hinged at the rear, the cassette sliding into the door label edge first), the key
pitch and travel, every force, the four C cells' mass (about 66 g each), the cassette's 42 g, and all frictions
(holder 0.25, cassette 0.35, case 0.7).

## Simulation shortcuts

- The tape, reels, motor, heads and pinch roller are not simulated: "playing" is PLAY locked with the door shut on a
  seated cassette, which on the real machine turns the reels.
- RECORD, REWIND, FAST FWD and PAUSE spring back without locking; one STOP/EJECT press both stops and opens the door
  (the real recorder needs a second press while playing).
- The cassette is a rigid shell with its two labels and the tape across its open edge; its hub holes and the deck's
  spindles are not modelled (the seated cassette rests on the holder's rails).
- The case's parts do not collide with each other; the door and keys are held by their joints and limits.
