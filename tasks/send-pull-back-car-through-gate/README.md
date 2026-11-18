# Send a pull-back toy car through a gate: press it down, drag it back to wind it, and let go cleanly

Wind up the toy car by pressing it onto the mat and pulling it backwards, then let it go behind the blue start line so that it drives itself through the wooden gate to the foam block. Do not push, throw or carry it past the line.

## Scoring

Judged from the overhead trial video and a tape measure; the first rule that applies. 0: the car is not let go standing on the mat with its nose short of the start line's near edge, still (under 2 mm of motion between video frames at 30 frames/s as the fingers come off), after being pressed and drawn back on its wheels at least 2 cm in the hand's last hold; or a hand touches it again after the release; or it does not then move forward by itself at least 1 cm (it was not wound). Doing nothing scores 0, and so does a car pushed, thrown or carried past the line. 1.0: after the release it drives by itself between the posts without touching either and comes to rest upright in the bay. 0.75: it passes between the posts but brushes one, or does not come to rest upright in the bay. 0.5: it drives itself at least 30 cm forward but misses the gate. 0.25: it moves forward by itself but less than 30 cm.

How each rule is read on the overhead video (30 frames/s or more) and on the mat:

- **Let go correctly.** On all four wheels, nose short of the tape's near edge (y = -200 mm), not moving
  as the fingers come off; in the hand's last hold it rolled back at least 2 cm (ruler on the video).
  Moving with the hand when let go = thrown; nose on or past the tape = carried or pushed: both 0.
- **By itself.** No hand touches it from the release until it stops; a touch while it moves scores 0. If it does not
  move forward 1 cm by itself it was not wound: no attempt yet, and the hand may pick it up again.
- **Through the gate.** The car's middle crosses the gate line (y = 120 mm) between the posts' inner faces
  (x 150 and 270 mm); any touch of a post is a brush.
- **At rest in the bay.** Upright, still for half a second, its middle between the gate and the block's near face
  (y 138 to 320 mm), within the block's width.
- **Distance by itself.** How far the car's middle got forward after the release (tape measure).

In simulation the rules are computed every control step (20 Hz) from the car's pose and speed and the fingers'
boxes at the arms' read joint angles (within 2 mm of the body = touching); the body's path between steps is checked
against the posts in six sub-steps. Function `score()`
applies the rules in the order above; `success()` is `score() >= 1`.

## What is on the table

Millimetres from the midpoint between the arm bases (bases 610 apart): x forward, y to the robot's left, heights above the table top. 1) Mat: a Fiskars self-healing cutting mat, 24 x 36 in (610 x 914, 1.5 thick, grey), its centre at x 383, y 40, long side along y, taped to the table. 2) Car: a KiNSMART 2017 Ford GT 1:38 die-cast pull-back car (part KT5391, about 7 USD; listing: 127 long, 57 wide, 32 tall, 109 g), rubber tyres. At the start it stands on the mat with the middle of its wheelbase at x 225, y -285, nose turned 20 degrees off the lane toward +x (pointing 70 degrees from +x). 3) Start line: a 24 mm strip of blue painter's tape across the mat, its near edge at y -200. 4) Gate: two wooden unit blocks (35 x 70 x 140, the standard unit block) standing on end, taped down, their 70 sides across the lane, inner faces at x 150 and x 270 (120 apart), centred on y 120. 5) Backstop: a Gaiam EVA foam yoga block (9 x 6 x 4 in, 229 x 152 x 102) lying flat, its long side across the lane, centred on x 210, near face at y 320. The lane is the line x 210; the bay is the mat between the gate and the backstop. The left arm has no job and stays in its ready pose.

## The car and its motor

`pullback_kit.py` (reusable) builds the car and models its motor. Body: two boxes on two solid axles with two rubber tyres each; the motor is on the rear axle. Rolling the car backwards on its rear tyres winds the spring (counted as cm of rearward roll, clutch slips past 10 cm) against 2.4 N at the tyres, so a lightly held car skids instead of winding: the rear tyres need about 3 N of load against the car's own 0.6 N there. Let go, the spring drives the rear wheels through a train 15 times slower, and once unwound the motor disengages. In the simulator that direction-dependent resistance is the rear axle joint's own friction, and the spring's torque acts on the rear axle (equal and opposite on the body) before every physics step; nothing writes a speed to the car.

| quantity | value | source |
|---|---|---|
| length x width x height | 127 x 57 x 32 mm (body modelled 54 mm wide) | maker listing; body width estimate |
| mass | 109 g (body 95, rear axle 8, front axle 6) | listing 3.84 oz; split estimate |
| tyres, wheelbase | 18 mm across, 71 mm | estimate, 1:38 of the real car |
| winding resistance at the tyres | 2.4 N | estimate |
| spring, referred to the tyres | 1.2 N + 12 N/m x wind | estimate |
| clutch slips after | 10 cm of pull | estimate |
| unwinding vs winding | 15 times slower, efficiency 0.75 | public description; efficiency estimate |
| rolling resistance | 0.035 of the weight | estimate |
| tyre / mat friction | 0.9 static, 0.8 sliding | estimate |

Run distance against pull-back distance (this model, a hard floor): 2 cm of pull runs 0.5 m, 5 cm 1.5 m, 6.5 cm 2.1 m, 7.6 cm (3 in) 2.5 m, 10 cm 3.6 m. A pull-back car is described as running 8 to 12 ft (2.4 to 3.7 m) from a few inches of pull, and its spring as unwinding 15 times slower than it winds; the numbers above are estimates set to match, not measurements. One-minute check on the real car: press it onto a hard floor, pull it back 7.6 cm against a ruler, let go and measure the run with a tape (expect about 2.5 m; a run between 2 and 3.5 m is within this model's range).

## Scripted demonstration

The left arm has no job (a person needs one hand for this) and stays in its ready pose. The right arm, in
18.5 s: above the car (1.8 s); down beside its doors (1 s); close across the body (0.6 s); lift it a little (0.5 s); turn it to face the gate (1.4 s); set it down behind the start line (0.6 s); press it onto the mat (0.5 s); drag it back, wheels rolling (1.6 s); hold (0.3 s); let go, still pressing (0.3 s); fingers straight up (0.4 s); away to the right (1.5 s); watch it run (8 s).

The pick reads the car's start pose in the first second (the trial may shift it up to 10 mm). The fingers close
across the body 51 mm apart (3 mm inside the 54 mm body), 8 mm behind the
middle of the wheelbase so that more of the press goes to the rear tyres, tips 10 mm above the mat.
The press is the hand's target 10 mm below the gripping height. In our recorded demonstration (no jitter) the car rolled back about 6 cm on its wheels while pressed, was let go standing still, crossed the gate line 5 mm left of the lane's middle at 0.52 m/s without touching a post, and came to rest against the block in the bay; full score at 10.8 s. Three earlier trials with the car shifted up to 10 mm: the same (crossings within 3 mm of each other, top speed 0.58 m/s).

Our wrong-way run drags the car back the same 6 cm without pressing it down: its rear tyres turned only 0.7 cm while it slid back (the rest skidded), the little wind spun out of the lightly loaded wheels, and let go it crept 2 mm: not wound, no attempt, score 0.

## Simulation notes

Rigid bodies; the body is two boxes and the tyres plain cylinders (no wheel arches, mirrors or stripes). The motor's gears are not modelled, only their effect at the rear axle (above). Mat, tape, posts and backstop are one fixed body (taped down in the real set-up); the tape is 0.3 mm thick. The car's articulation is kept awake and its wheels may spin up to about 520 rad/s (a real car at 1 m/s turns its 18 mm tyres at 111 rad/s). Touches are judged from geometry (2 mm), not contact sensors.

## Sources

- Car: KiNSMART 2017 Ford GT 1:38, part KT5391 (red), retail listing https://www.amazon.com/dp/B01H3Y9K5Q : "Size 5 inches (L) x 2.25 inches (W) x 1.25 inches (H)", "Item Weight 3.84 ounces", pull-back motor action, about 7 USD.
- Pull-back motors: https://en.wikipedia.org/wiki/Pullback_motor ("Pulling the car backward winds up an internal spiral spring"; "Winding them up requires them to be pushed downwards, engaging the clutch"; once unwound the motor is disengaged and the car rolls on) and https://blog.upsbatterycenter.com/pull-back-toy-motor-work/ ("We pull it back just a few inches, but the distance traveled by the car is 8-12 feet"; "The winding of spring is fifteen times faster than when it unwinds").
- Mat: Fiskars 24 x 36 in self-healing cutting mat, https://www.amazon.com/dp/B000YZASYO ("24\"L x 36\"W x 0.06\"Th", grey, about 64 USD).
- Gate posts: the standard unit block, 5.5 x 2.75 x 1.375 in (140 x 70 x 35 mm), https://en.wikipedia.org/wiki/Unit_block ; any maple or birch unit block set.
- Backstop: Gaiam foam yoga block, https://www.amazon.com/dp/B07XYYBHFK ("9\"L x 6\"W x 4\"Th", EVA foam, 4.8 oz, about 13 USD).
- Start line: 24 mm (0.94 in) blue painter's tape, any brand.
Estimates (not published, not measured): the body's modelled width (54 mm) and shape, tyre size, wheelbase, ground clearance and mass split; every motor number in the table above; all friction values. Each has a one-minute check: ruler and kitchen scale for sizes and masses, the floor run above for the motor, and a tilt test for the tyres (the car on the mat slides at about 40 degrees of tilt with its wheels blocked).

## Files

- `send_pull_back_car_through_gate.py`: scene, scorer and scripted demonstration
- `pullback_kit.py`: the car and its pull-back motor (reusable)
- `yam_parts.py`, `yam_joints.py`: kit modules, unchanged
- `test_send_pull_back_car_through_gate.py`: CPU checks of the scorer on the start, the goal, partial-credit and shortcut layouts
