# Push a wooden toy train round a bend by its last wagon, keeping all four vehicles coupled and in the grooves

Push the toy train counter-clockwise round the track only by the back face of its last wagon, without gripping it or touching the other vehicles, until the engine passes the red finish mark, keeping all four vehicles coupled and every wheel in its groove, then take your hand away.

## Scoring

Judged from the trial video and a look at the end. Coupled: each pair of neighbouring vehicles has its coupling magnets touching. In the grooves: every wheel down in its groove (none on a rail top or off the track) and every vehicle upright. Lifted: at any time a vehicle was in the air, its lowest point more than 1 cm above the rail tops (picked up or carried). Pushing rule: the train may be moved only by pushing on the back face of the last wagon; gripping any vehicle, or touching the engine, the load wagons or the last wagon anywhere but its back face (with that face's top edge, up to 8 mm onto the roof), breaks it; touches are judged on the wrist-camera views of the trial video. Hands clear: both grippers at least 3 cm from every vehicle at the end. The engine's progress is where its middle (half way between its axles, marked by a small pencil tick on its outer side) stands against the paper marks. Score the first that applies. 0: a vehicle was lifted, or the pushing rule broken, at any time. 1.0: the engine's middle at or past the red finish mark, the train coupled, every wheel in its groove and hands clear. 0.75: past the finish, coupled and every vehicle still on the track, but a wheel out of its groove or a hand still touching. 0.5: the engine's wheels in their grooves and either past the finish with a coupling open, or the train coupled and past the half-way mark. 0.25: the engine's wheels in their grooves at or past the 10 cm mark. 0: anything else (doing nothing scores 0).

How `score()` reads each condition (angles about the circle's centre, counter-clockwise; a vehicle's middle is half
way between its axles):

- **Progress:** the engine's middle's angle from its start: finish 70.0 degrees (262.5 mm of track),
  half way 35.0 degrees, the 10 cm mark 26.7 degrees.
- **Coupled:** for each neighbouring pair, the two coupling magnets within 1.5 mm of touching.
- **In the grooves:** every wheel's middle within 2.5 mm of its groove's centre line (groove 6 mm,
  wheel 4 mm), its bottom within 2 mm of the groove floor (3 mm deep), and the
  vehicle tipped less than 14 degrees.
- **Lifted** (remembered for the rest of the run): a vehicle's lowest point more than 10 mm above the rail tops.
- **Pushing rule** (remembered for the rest of the run): points 3 mm apart on each gripper's outline (from the
  robot's collision model: fingers 4 mm thick at the tip, 24 mm at the root; the palm) within 2.5 mm of a
  vehicle's shapes touch it. Broken by a touch on the engine or a load wagon, on the cargo wagon more than
  8 mm forward of its back face, or by both fingers' inner faces on one vehicle (a pinch).
- **Hands clear:** both fingertips at least 30 mm from every vehicle's bounding box.
- **On the track:** a vehicle's middle within 10 mm of the track's centre line.

`success()` is a score of 1.0. Why the shortcuts fail: picking the train up and setting it down at the finish
counts as lifted (0); towing it by the engine, or pushing a wagon's side, breaks the pushing rule (0); pushing it so fast that the front wagons run on and part leaves a coupling open (0.5 at most);
knocking the engine off the track gives no progress at all (every partial tier needs the engine's wheels in its
grooves, or the whole coupled train on the track). A wrong-way run that pushes in a straight line from the start point to the end point, cutting the bend, scored 0 (run before the pushing rule was added, which can only lower a score): the hand slid off the wagon's back after moving the train about 5 cm, then ran into the train's side, which also breaks the pushing rule, and knocked the engine and both load wagons out of their grooves.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=push_a_wooden_toy_train_round_a_bend.py -E object_jitter=0 -E control_hz=20 \
  --instruction "$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')" \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial use `--policy dream`. Each trial allows 2,400 control steps (120 s at 20 Hz);
the scripted demonstration takes 20.0 s. `pytest` here checks the scorer on named layouts. Keep
`object_jitter=0`: the vehicles start with their wheels in the grooves, and a sideways jitter would put them on the
rails.

## The scene

Products, sizes and positions are in task.json's `setup`. In the simulation (all sizes estimates unless the setup
gives a maker figure; `train_kit.py` builds them and lists the numbers):

- **Track:** eight fixed pieces, each a 45-degree arc of centre-line radius 216 mm: a bed 9 mm thick and
  40 mm wide under three rails 3 mm high, which leave two grooves 6 mm wide with their centres
  26 mm apart. Exact triangle meshes; friction 0.05/0.04.
- **Vehicles:** four free bodies, each 70 mm long on four wheels 44 mm apart along and 26 mm
  across: engine 60 g (boiler, cab, chimney), two load wagons 45 g each with
  their containers, the cargo wagon 55 g with its 34 mm wide back face. Each wheel is a fixed
  13 mm disc with an 8 x 4 mm runner under it in the groove; friction
  0.05/0.04 (so 0.04 sliding where wheel meets track, standing for rolling).
- **Couplings:** a 9 mm ball at each end, 37 mm from the middle, 13 mm above the wheel bottoms;
  coupled vehicles' middles are 83 mm apart (four: 332 mm, maker 333 mm). Each coupling
  pulls with F(d) = 1.0 N x ((1 + 9 / 6) / (1 + d / 6))^4 between the ball centres
  (d in mm), faded to zero between 14 and 18 mm: 1.0 N to pull apart, 0.48 N at a
  3 mm gap, 0.22 N at 6 mm.
- **Marks:** thin paper squares beside the track (no collision effect on the train).

## Scripted demonstration

The right hand alone pushes; the left arm stays at its ready pose. With its fingers closed and turned so the closing
direction runs along the track, the hand comes down 12 mm behind the back face of the cargo wagon, its fingertips
34 mm above the table (above the rear coupling, below the wagon's roof), and pushes that face with the
front finger's outer face, which slopes outward as it rises and so bears on the face's top edge. It then moves round the circle at the wagon's own radius, turning with the
track, so the push always runs along the track at the wagon's back: 80.0 degrees in 11 s (about
27 mm/s on average), sent 3 mm into the face so the hand's small lag still keeps contact.
Nothing holds the train: the hand only pushes, and the grooves steer the wheels.

| Time (s) | Step |
| --- | --- |
| 0.0 to 2.0 | fingers closed, over the back of the last wagon |
| 2.0 to 3.5 | down behind its back face |
| 3.5 to 4.0 | settle |
| 4.0 to 15.0 | push it round the bend |
| 15.0 to 16.0 | back off along the track |
| 16.0 to 17.0 | rise clear |
| 17.0 to 18.5 | park clear of the train |
| 18.5 to 20.0 | hold, the train in view |

Simulation run of this demonstration: full score from 16.6 s, held to the end of the run at 19.6 s with the hand parked to the right, where the top camera sees the whole train; the engine went 79 degrees round (finish at 70), coupled, every wheel in its groove, and the pushing rule was never broken. An earlier test copy of the same push, with the whole hand path shifted 0, 4 mm out and 4 mm in (standing for the arm's path error), scored 1.0 each time.

## Simulation and the real objects

Rigid bodies. Each vehicle is one body; its wheels do not turn but slide in the grooves on a low friction (0.04, estimate) that stands for rolling, held sideways only by the groove walls (exact track meshes). The magnetic couplings are forces between points (1.0 N pull-off, estimate; the real ones attract only one way round). Containers and motor load are part of their wagons, as on their pins. Groove, wheel and vehicle sizes and masses are estimates.

- Train: BRIO 33982 Three-Wagon Cargo Train, 7 pieces (wooden cargo engine, cargo wagon with sliding doors, silver
  motor load, two wagons for loads, orange and light green containers), 333 x 34 x 50 mm, magnetic connections and
  pins (BRIO 2024 catalogue, https://www.kayes.co.uk/BRIOSS24.pdf, page 194); $34.99, shipping weight 0.29 kg
  (https://www.trainworld.com/brio-33982-wood-world-three-wagon-cargo-train.html).
- Track: BRIO 33342 Large Curved Tracks, 4 pieces, track length 170 mm (same catalogue, page 210); $14.99 a pack
  (https://www.trainshack.com/wooden-trains/curved-track.html). Eight make one circle.
- Estimates, each checkable in a minute with a ruler or a kitchen scale: the circle's radius (about 432 mm between
  the centre lines across the circle), track, groove, wheel and coupling sizes, the masses, the coupling's pull-off
  (lift a coupled wagon slowly off one held on the scale), the friction standing for rolling (tilt a track piece until
  a wagon rolls) and the 0.1 N push (a 10 g weight on a thread over the table's edge).

## Reset

About 30 seconds; no consumables beyond a few strips of masking tape. Lift the train off the track, couple it in order (engine, orange-container wagon, green-container wagon, cargo wagon), engine facing counter-clockwise and each wagon the way round in which it pulls onto the vehicle ahead (the magnets attract only one way round; turn a wagon end for end if the two push apart), check that the containers sit on their pins and the motor load is inside the cargo wagon with its doors shut, and set the train down with every wheel in the grooves and the engine's middle at the start mark. Check that the tape still holds every track joint and that no piece has moved (re-tape a loose one). Return both arms to their ready pose. Kit: one BRIO 33982 ($34.99) and two packs of BRIO 33342 ($14.99 each), masking tape and paper marks; under $70.
