# Play a cassette on a shoebox recorder: pop the door open, slide the tape into it side A up, click it shut, press PLAY

Put the cassette into the recorder so that side A will play, shut the cassette door and press PLAY.

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
- Side A / side B: which marked note faces up through the window: black A on orange or black B on blue. The letters, not the colors alone, identify side A to the robot. Backwards: tape edge toward the hinge.
- `success()` is a score of 1.0, held for one second. The tiers do not overlap and cover every end state; the start
  (door shut and empty, cassette on the table) scores 0.

PLAY alone cannot reach 1.0. The original physical wrong-way run seated side A and locked PLAY but left the door at 22 degrees: score 0.5.

## Run and score

From this folder with YAM Arena installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=play_a_cassette_on_a_shoebox_recorder.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Put the cassette into the recorder so that side A will play, shut the cassette door and press PLAY." \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

Replace `yam_scripted` for a model policy. Limit: 2,400 steps (120 s at 20 Hz); demo about 36 s. Run `pytest` for scorer checks.
The demo reads both object poses before acting; original jitter probes used 0.01.

## The scene

Coordinates from the midpoint of arm bases 61 cm apart: x forward, y left. Jensen MCR-100 silver/black plastic recorder, 240 x 142 x 46 mm, 1.08 kg including four C cells (estimate), centre (24, -5) cm, long axis x, feet on bare table, keys toward robot; mark its outline with tape. Start shut and empty, no key down. Maxell UR-90 black IEC cassette, 100.4 x 63.8 x 12 mm, about 42 g, stands on its label edge at (20, -21.5) cm, long axis y, tape edge up. Side A faces away from the robot, with an orange note marked A; side B faces the robot, with a blue note marked B. Each note has its own side's black marker capital at least 8 mm tall. The letters tell the robot which side is A. Keys from robot left to right: RECORD, PLAY, REWIND, FAST FWD, STOP/EJECT, PAUSE.

The recorder is one articulated body, loose on its feet: the case (lower shell, speaker end,
the sides of the cassette well, the label strip and the key surround), the window door on a hinge along its rear edge,
and six keys on slides. Under the door's 2 mm plate hangs the cassette holder: two side walls 102.4 mm apart (1.0 mm a
side around the shell, the front 4 mm flared out to 109 mm), two rails under the cassette's ends 13.2 mm below the
plate (the front 4 mm bent down 4 mm), open at the front and closed by a back stop 63.8 mm in. Seated, the cassette's
tape edge is level with the door's front edge. The door pops up to 22 degrees (its front edge about 28 mm up), so the
cassette goes in along a 22 degree slope, label edge first.

Reset: keep black marker A on orange side A and B on blue side B, capitals at least 8 mm tall; renew faded letters. Simulated ink: about 32 mm high, 3.5 mm strokes, 0.15 mm proud, visual-only without collision/mass APIs. Physics, pose, scorer and demo path are unchanged.

No door dimension is a maker figure: the spec gives overall size (9.45 x 5.59 x 1.78 in); the manual says playing side facing out. Rear hinge, 22-degree opening, 112 x 73 mm door and 102.4 x 13.2 mm holder cross-section, depth 63.8 mm, are estimates. Before each trial, spend one minute with a ruler checking hinge side, door rise/run (opening angle), holder depth/width/gap against the 100.4 x 63.8 x 12 mm cassette, and confirm slide-in loading without forcing. If any check fails or loading differs, do not run a scored trial on that unit; do not force it or substitute another model. No hardware verification is claimed. At native 320 x 176 resolution, the upright notes are too edge-on to read in the start top view and outside both start wrist views. Bring a camera face-on to read a note; the existing demo turn makes A identifiable in the top view.

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

Door spring, catch and PLAY lock apply internal forces before each physics step. The arm must still grasp, insert and press with roughly the real forces to trigger them.

## Scripted demonstration

Right arm only; left stays at start. Revision demo: 1.0 after 648 executed steps, first full score at call 629. It ends during the one-second success hold with the hand over PLAY; its video alone does not show a retreat. Final recorded state: door 0 degrees, PLAY 7 mm, cassette seated, recorder unmoved. With the unchanged physics, original probes held the door shut without contact for 4 s; the original wrong-way run held PLAY down for 4.4 s with the hand parked 25 cm away. Original 10 mm-jitter probes: 1.0 in all three episodes.

1. Press STOP/EJECT with closed fingers; the door pops up.
2. Pinch the standing cassette 10 mm below the tape edge (12 mm shell, jaws 2.5 mm inside), lift and turn 68 degrees to match the door slope, side A up.
3. Read the held cassette pose and correct; slide label-edge first until fingers are 2 mm short of the door, release, then push the tape edge home with closed fingertips below the plate.
4. Press the door 10 mm behind its front edge until it catches.
5. Press PLAY fully; release and move clear.

## Sources

- Jensen MCR-100 maker page (Spectra Merchandising): https://www.spectraintl.com/index.php/jensen-products/item/mcr-100
  ("shoebox" style, top loading cassette deck, four C cells or the AC adaptor; product photos for the key layout).
- Maker spec: https://www.spectraintl.com/images/stories/pdf/jn/MCR-100.pdf (overall size 9.45 x 5.59 x 1.78 in; no door or holder dimensions).
- MCR-100 user manual: https://www.spectraintl.com/images/stories/pdf/jn/UM/MCR-100_EIB.pdf (STOP/EJECT "Press to stop
  all deck operations. Press again to open the cassette door."; "Put in a cassette tape with the side to play facing
  out, and close the cassette door.").
- Size and weight: Best Buy listing Q&A, https://www.bestbuy.com/site/questions/jensen-cassette-player-recorder-silver/5657390
  (width 5.59 in, depth 9.45 in, 1.8 lb); height 1.8 in from the Beach Audio listing,
  https://www.beachaudio.com/jensen-mcr100-cassette-player-and-recorder-with-one-touch-mcr100_9/ (about 47 USD).
- Compact cassette shell 100.4 x 63.8 x 12 mm: IEC 60094-7. Maxell UR-90 price (about 2-8 USD) and mass are estimates.
- Cost of the real objects (estimate): recorder about 47 USD, cassette under 10 USD, four C cells and two sticky
  notes: under 80 USD.

Other estimates: four C cells about 66 g each, cassette 42 g, frictions holder 0.25, cassette 0.35, case 0.7. No physical measurements are claimed.

## Simulation shortcuts

- The tape, reels, motor, heads and pinch roller are not simulated: "playing" is PLAY locked with the door shut on a
  seated cassette, which on the real machine turns the reels.
- RECORD, REWIND, FAST FWD and PAUSE spring back without locking; one STOP/EJECT press both stops and opens the door
  (the real recorder needs a second press while playing).
- The cassette is a rigid shell with its two labels and the tape across its open edge; its hub holes and the deck's
  spindles are not modelled (the seated cassette rests on the holder's rails).
- The case's parts do not collide with each other; the door and keys are held by their joints and limits.
