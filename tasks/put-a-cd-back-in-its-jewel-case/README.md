# Put a CD back in its jewel case: press it onto the hub until the teeth grip, then shut the lid

Put the CD back in its case, pressed onto the hub, and close the case.

Anyone who has owned CDs, DVDs or game discs has put one back in its case. The disc is a 1.2 mm plate, 120 mm across, wider than the gripper opens, so it can only be held by its edge. Its 15 mm hole has to be centred on the tray's hub, a ring of twelve sprung teeth whose rounded tips give less than 2 mm of lead-in, then pressed down until the teeth flex in and snap back over the hole (about 3 N), and finally the lid is swung over and pressed until its catch clicks. A disc only laid on the hub fits under the shut lid but rides loose and gets scratched as it rattles: that scores nothing.

## Scoring

Judged at the end of the run, in the case's own frame (a case that has slid a little is judged where it is).

| End state | Score |
|---|---|
| Disc seated on the hub; lid shut on its catch and still caught 1 s after both hands are clear | 1.0 |
| Disc seated, lid down within 10 degrees but not caught (or only held down by a hand) | 0.8 |
| Disc seated, lid left open wider | 0.6 |
| Disc in the tray within 12 mm of the hub, not seated, lid open | 0.2 |
| Lid down (within 10 degrees) over a disc that is not seated | 0 |
| Disc fell off the player; case tipped over 10 degrees or knocked off the player | 0 |
| Case pushed more than 50 mm along the player | capped at 0.5 |

"Seated" needs all of: the disc's centre within 0.6 mm of the hub's axis; its centre within 0.35 mm of the height it
has lying on the support ring (a disc resting on top of the teeth sits about 1.7 mm higher, so it fails); its axis
within 1.5 degrees of the case's; and each of the hub's four sprung tooth groups back within 0.12 mm of rest (a group
still held in by the hole's edge is not back). In a real case a seated disc stays on when the case is turned over.
"Caught" means the lid is at or below 0.8 degrees from shut, past the catch's crest (below it the catch pulls the lid
shut; above it, up to 3 degrees, it pushes the lid back open), and it is still there 1 s after both grippers' centre
points are at least 30 mm outside the case's bounding box, so a lid held down by a hand does not count until the hand
has left and the catch alone holds it. The fall and tip checks run every control step
("ever": once it happens the run scores 0); the 50 mm cap compares the case's final position with its start.
`success()` is a score of 1.0; the run ends early once 1.0 has held for 1 second.

Why the shortcuts fail: on the GPU, a scripted wrong way laid the disc on the hub without pressing it, then shut the lid; the disc stayed on top of the teeth (centre 7.3 mm up; seated is 5.6), the lid clicked and stayed caught, and the run scored 0. A lid let fall from 12 degrees rests on the catch at about 3 degrees (all four runs measured), so shutting it takes a press. A lid shut on a disc off the hub, an empty case shut, a dropped disc or a case knocked off the player all score 0 in the offline checks.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=put_a_cd_back_in_its_jewel_case.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Put the CD back in its case, pressed onto the hub, and close the case." \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial, replace `--policy yam_scripted` with `--policy dream`. Each trial allows 2,400 control steps
(120 seconds at 20 Hz). The scripted demo scored 1.0 on the GPU (success at 49 s), and 1.0 in each of three runs with the objects jittered by up to 10 mm. `pytest` in this folder runs the offline checks (no GPU): the scorer on the
start, the goal and the named partial and shortcut layouts.

## The scene

Millimetres from the midpoint between the arm bases: x forward, y to the robot's left, heights above the table top. A Denon DCD-600NE CD player (434 wide, 273 deep, 107 high, black, tray shut) stands on the table with its front face at x 235, centred on y 0. On its top lies an open standard CD jewel case (142 x 125 x 10.4 shut, clear polystyrene plastic, 70 g, black tray): base flat, its 142 mm length along y from y -71 (spine, right) to +71, its front long side 2 mm over the player's front edge, so the hub's centre is at x 296, y 0, 107 up. The lid stands open at 100 degrees on the spine (leaning 10 degrees past upright, away from the tray) and holds a 120 x 120 booklet. The tray has a well for the disc, two thumb cut-outs at the ends and the usual hub: 12 teeth on a 15.3 mm crown (14.85 mm below it) standing on a 33 mm support ring. A CD (120 across, 1.2 thick, 15 mm hole, 16 g, polycarbonate plastic) lies label up on the player's top centred at x 268, y 145, its rim 27 mm past the front edge: just taken out of the player and put down where a person (or a gripper) can take it by the edge without sliding it. Nothing else is on the table. The right arm can do the disc and the lid's swing; the left presses the lid shut.

Sources: IEC 60908 (120 mm disc, 1.2 mm thick, 15 mm hole); Wikipedia, "Optical disc packaging" (jewel case 142 x 125
x 10.4 mm, three polystyrene parts); cdrom2go and Walvis product pages (whole case 68-100 g; without its tray 44-48 g);
US 2002/0112976 A1, "Disc hub retainer" ("eight to twelve upwardly projecting prongs"; a conventional 12-prong post
15.25-15.35 mm across holds a 15.00 mm hole by friction); US 5,896,986 (the rosette's free ends deflect slightly
inward; a disc is mounted by a slight downward pressure); Denon DCD-600NE information sheet (434 x 107 x 273 mm, 4.0 kg).

The case (70 g, one body with jointed parts): a 1.5 mm back plate, 3.4 mm long walls, 8.9 mm end walls and a black tray
with its deck 4.0 mm up, a well 121 mm across (rim 0.4 mm above a seated disc), two 28 mm thumb cut-outs at the disc's
3 and 9 o'clock, and the support ring (33 mm across, 5.0 mm up) a seated disc lies on. The hub: 12 teeth, 20 degrees
wide with 10 degree slits, 14.85 mm across below the crown and 15.30 mm at it (the conventional post), the crown
starting just over a seated disc (0.15 mm of lip over the 15.00 mm hole), sloping at 52 then 27 degrees and ending
8.2 mm up, 11.2 mm across, so a hole set down within about 1.9 mm of the axis is steered onto it. The teeth move as
four sprung groups of three (each tooth 1,100 N/m: a polystyrene cantilever 6.5 x 3.1 x 0.5 mm; a group 2,750 N/m
along its line, at most 0.3 mm in). Pushing the hole over the lip takes about 3 N (12 x 1,100 N/m x 0.15 mm = 2.0 N
inward, through the 52 degree slope with friction 0.35), some twenty times the disc's weight, so a disc only laid on
the hub stays on top of the teeth. The lid (24 g, a booklet under it) turns on pins at the spine 5.2 mm up with 0.012 /
0.008 N m of static / sliding friction, so it stays where it is put; shut, its walls rest on the base's walls and the
booklet is 0.1 mm above the teeth. Its catch: on a real case two small tabs at the lid's free corners snap past bumps
on the tray. Here they are one detent at the hinge, a torque between lid and base set by the lid's angle: none above 3
degrees; from 3 to 1.6 degrees it pushes the lid back open, rising to 3 N at the free edge (142 mm from the hinge; a
jewel case needs roughly 2-5 N to click shut); it falls to nothing at the crest, 0.8 degrees; past the crest it pulls
the lid shut and holds it with 2 N at the free edge, so about 2 N opens it. A lid let go on the bump springs back to
about 3 degrees open. The disc: 16 g, its hole's flats exactly
on 15.00 mm. Friction (static / sliding): disc 0.35 / 0.30, case 0.45 / 0.40, player 0.45 / 0.40.

## Scripted demonstration

The right arm does the disc and swings the lid; the left presses the lid shut (51.9 s plan). The right hand
comes in level, fingers one above the other, and closes its pads 8 mm onto the rim where it passes the player's front
edge (about 3 N squeeze). It lifts the disc level, carries it over the hub, reads the disc's and the case's poses twice
(25 mm up, then at let-go height) and moves so the disc's centre is on the hub's axis and its face square to it. It
opens the pads to 5 mm: the hole drops onto the crown and centres while the rim rests on the lower pad, which slides
straight back out. The hand then points straight down over the hub with its fingers open 24 mm, so its two fingertips
press the disc's clamping ring 12-16 mm either side of the axis at once: sent 32 mm past the disc's top over 2.5 s,
the arm's stiffness pushes the disc evenly down (about 11 N, read from the arm's drive torques) until the teeth snap
over the hole; it holds 1.5 s, lifts, and repeats with the hand turned 90 degrees. Then the closed fingertips push the
standing lid over in six steps from the angle it is read at to 12 degrees, and the left hand presses the lid's free end
down through the catch, holds, and lifts away. The demo reads object poses from the simulator (privileged); a model under test sees only the
cameras.

In three jittered GPU runs the disc ended 0.06-0.07 mm from the hub's axis, flat on the support ring, every tooth back; the lid came to rest on the catch at 2.9-3.0 degrees and the left hand pressed it through to 0 within about 0.6 s; it stayed caught with both hands parked; the case moved 5 mm or less.

## Simulation and the real objects

Rigid bodies, 240 Hz physics, 20 Hz control. Published: the disc's size and hole, the case's outside size, weight
range and material, the hub's tooth count and diameters, the player's size and weight. Estimates, to be measured on a
real case: wall, deck, well and support-ring heights; the teeth's profile above the lip, their stiffness and 0.3 mm
travel (so the ~3 N push); the catch's forces (3 N to click shut, 2 N to open) and angles; the lid's pin friction and
its 24 g share; every friction value. Shortcuts: the lid's two catch tabs are modelled as one detent at the hinge (the
torque above, applied between lid and base before every physics step) instead of two sprung tabs; the 12 teeth move as
four groups of three, because the simulation's sliding joints run along the case's own axes (each group's side teeth
sit 30 degrees off its line); the tooth groups (2 g) are heavier than the real plastic so the 240 Hz solver passes
the push through them; the disc collider is 48
flat-sided sectors; the clear case is drawn tinted so the cameras see it; the 4 kg player is fixed to the table; the
label is two printed colours. A disc left resting on the crown fits under the shut lid with about 0.4 mm to spare, as
in a real case: it rides loose rather than holding the lid open.

## Reset

Open the case, press the hub's centre and lift the disc off by its edge; lay the disc label up on its mark with its rim over the player's front edge, put the case on its marks and stand the lid open at 100 degrees. Loose items may be off their marks by up to 10 mm. The 4 kg player stands free on the table (fixed in the simulation; the few newtons of the task do not move it).
