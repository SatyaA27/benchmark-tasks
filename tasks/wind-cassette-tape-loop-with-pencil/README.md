# Wind a cassette's loose loop of tape back in with a pencil: hold the cassette up, push the pencil through a hub, turn it

A loop of tape hangs out of a cassette. The robot lifts the cassette by a short edge so its hubs and the loop are
free, pushes a sharpened hexagonal pencil point-first through one reel hub until the six-sided barrel sits among the
hub's six teeth, and turns the pencil the way that draws the tape in, letting go and re-gripping because a wrist turns
only so far, until the tape runs straight across the opening. It must not wind on past straight, which stretches the
tape, and must not push the tape in with its fingers.

## Scoring

Judged at the end from the trial video, a photo and a ruler.

- **Loop**: tape outside the shell beyond the straight path across the opening, read at the end with the cassette
  as it is (still held, or as it lies if set down or dropped): hold a ruler upright against the shell's opening edge
  and read how far the loop hangs below it; the loop is twice that.
- **Engaged turn**: a reel hub turned with the pencil's six-sided barrel through it; on the video the hub's spoke and
  the pencil turn together. A hub turned with only the cone in it, or by anything else, is not an engaged turn.
- **Damaged**: the tape is creased, curled, stretched or snapped, or the reel was turned more than a quarter turn on
  after the tape first came straight: mark one spoke of each hub with a paint pen before the trial; on the trial
  video (20 frames a second, or a phone at 30) the mark shows through the clear shell window; count from the frame
  the tape first lies straight.

| Score | Outcome |
|---|---|
| 1.0 | No loop left: the tape runs straight across the opening, not damaged |
| 0.75 | An engaged turn; loop under 20 mm |
| 0.5 | An engaged turn; loop 20 to 60 mm |
| 0.25 | Any other outcome after an engaged turn: loop over 60 mm, a loop longer than at the start (wound the wrong way), the cassette dropped |
| 0 | No engaged turn of at least 10 degrees (doing nothing, tape pushed in with the fingers, only the point in a hub), or damaged |

Damage overrides every positive tier; the tiers do not overlap and the start (120 mm loop, no turn) scores 0.

**In the simulation** (`score()`, `tracking()` in `wind_cassette_tape_loop_with_pencil.py`): the loop is 120 mm minus
12 mm per radian wound in. Wound-in angle counts only the steps in which (a) the hub centre lies within 1.1 mm of the
pencil's axis, on the hex barrel (not the cone), with the pencil within 15 degrees of the hub axis (a barrel through
both 11.1 mm shell openings cannot tilt further), (b) that reel and the pencil both turn, the same way, relative to
the shell, and (c) the step is under 0.35 rad. Each reel counts in its own take-up sense: the robot's left reel draws
tape in turning clockwise as the robot sees it, the right reel anticlockwise; the other way pays tape out. Wound in
more than 10 rad + a quarter turn marks the tape stretched for the rest of the trial. `success()` = score 1.0 with the
cassette and the pencil moving under 2 cm/s, held 1 s.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=wind_cassette_tape_loop_with_pencil.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Pick up the cassette by its short edge and hold it up so its loop of loose tape hangs free. Push the sharpened pencil point-first through one reel hub until the six-sided barrel catches the hub's teeth, and turn it the way that draws the loop in. Stop once the tape runs straight across the opening; never push the tape in with the fingers." \
  --scorer task_score --max-steps 4000 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial use `--policy dream`. Each trial allows 4,000 control steps (200 s at 20 Hz); the scripted
demonstration plans at most 109 s and reached full score at 54.5 s. `pytest` here checks the scorer on named layouts.

## The scene

Millimetres from the midpoint between the arm bases; x forward, y to the robot's left, heights above the table.

| Object | Real item | Simulation |
|---|---|---|
| Desk mat | dry rubber desk mat, 360 x 360 x 2 (cut from a larger mat) | fixed box, centred (120, 0) |
| Cassette | RTM C60 blank compact cassette, IEC 60094-7 shell | 100.4 x 63.8 x 12 shell (1 mm skins), two free-turning reels 42.5 apart, 40 g in all (estimate), standing on its opening edge, centre (150, -42.5, 31.9) |
| Reel hubs | six teeth, 60 degrees apart, 7.9 tooth-tip circle, 10.2 root circle; shell openings 11.1 | the same; teeth 1.3 wide and 4 deep (estimates); reel drag 0.002 N m static, 0.0015 N m sliding (IEC bound, not a measurement) |
| Pencil | STAEDTLER Noris 120 HB, hexagonal, 7 across the flats, 175 long, sharpened in a STAEDTLER 510 20 sharpener (23 degree point) | hex barrel 8.08 across the corners, 17.4 cone from 8.08 to a 1 mm nose (estimate), 5 g (estimate), lying along x, point +x, flat end at (25, -140), tip at (200, -140) |
| Tape loop | 120 mm drawn out of the opening | drawn curve only (no collider), length as in Scoring; it lies on the mat at the start |

Frictions (estimates): shell 0.5, pencil 0.6. Free objects start jittered by up to 10 mm. The pads touch only the
dry plastic shell and the wooden barrel; nothing wet or powdery is involved.

## Scripted demonstration

The left hand pinches the shell's 12 mm thickness 8 mm in from its left short edge (fingers 2 mm inside the width),
lifts it 68 mm straight up and holds it there, hubs and loop free. The right hand picks the pencil across its barrel 43 mm from
the flat end (fingers 2 mm inside the flats), lifts it, points it along +x at the left reel's hub, and pushes the cone
through until the barrel is among the teeth (hub 30 mm behind the tip); it draws the pencil back first so the point
passes in front of the held shell. Then it winds: roll the wrist 0 to 60
degrees about the pencil, open, back off and rise, look where the pencil now rests (it droops about 13 degrees in the
hub once let go), come straight down on it, close, lift it level on the hub axis, and repeat. Each cycle the reel
turns about 1 rad; about ten cycles take the 120 mm loop in. The demo looks at the loop at the start of each cycle and
stops turning once the tape is straight. The cassette rolls up to about 20 degrees in the left pinch while it is
wound; each re-level lifts it back.

GPU runs on this version: the scripted demonstration scored 1.0 (full score at 54.1 s, success at 54.5 s, 1,109
steps); the cassette rolled under 3 degrees in the pinch. A wrong-way copy that pushes the point through the hub but
stops the barrel 20 mm short of the teeth scored 0: the cone turned the reel 14 degrees by friction (not counted),
then the pencil, hanging by its point, fell to the mat. Dream-1 scored 0.0 over the full 4,000 steps: it lifted the
cassette, but never got the pencil into a hub.

## Simulation and the real objects

- The tape is not simulated as a body. A rule stands in for the tape's response, but the robot must still do the
  real work: hold the cassette up, find and grip a 7 mm pencil, guide its cone through the real 7.9 mm toothed opening
  until the barrel's corners sit among the teeth, and turn a free reel with it through about 1.6 turns.
- Wound tape 24 mm across (estimate) gives 12 mm per radian; 120 mm needs 1.59 turns. Check in a minute: ruler across
  the tape pack through the clear window, or mark a spoke, turn once and measure how much the loop shortened.
- Pencil cone and nose, tooth width and depth, shell skins, masses, frictions and reel drag are estimates (check the
  cone with a ruler and both masses with a kitchen scale). The pencil sheet gives "7 mm diameter" without saying
  flats or corners; measure across the flats before a trial.
- Over-winding is a rule (a quarter turn past straight), not tape mechanics; creases are not simulated.
- Clear plastic is drawn grey; screws, label and the tape inside the shell are left out. All shapes are built here.

## Sources

- Cassette: RTM C60, maker page (tape 3.81 mm wide, 16.7 micrometre, clear shell; ten-pack USD 65):
  https://us.recordingthemasters.com/products/rtm-c60
- Shell and hub dimensions, IEC 60094-7:1986 Figure 2 and section 12.5 (reel torque bound):
  https://previewnorm.com/iec/IEC%2060094-7-1986%20PDF.pdf ; scope: https://webstore.iec.ch/en/publication/728 ;
  hub spacing 42.5 mm: https://www.ecma-international.org/wp-content/uploads/ECMA-34_2nd_edition_july_1973.pdf ;
  7.9 mm hub bore: https://patents.google.com/patent/KR960006402Y1/en
- Pencil: STAEDTLER Noris 120 product sheet (hexagonal, 7 mm, 175 mm):
  https://www.itrip.it/site/itrip_webprofessional_it/ebusiness_pdf/STA105309_PDBPDF.pdf
- Sharpener: STAEDTLER 510 20 (23 degree angle, pencils up to 8.2 mm):
  https://www.staedtler.com/intl/en/products/pencils-and-accessories/sharpeners/staedtler-510-20-metal-double-hole-sharpener-510-20/
- The everyday technique (pencil pushed through the hub, point out the far side), a player manual, printed page 16:
  https://m.media-amazon.com/images/I/91z90WJ0TiL.pdf

## Cost and reset

One cassette per trial at USD 6.50 (ten-pack USD 65); pencil, sharpener and mat about USD 20 together (estimate).
Reset in about 2 minutes: take a fresh cassette, wind side A to its start, turn the robot's-left reel on until its
tape pack is about 24 mm across, draw out a 120 mm loop with a fingernail without creasing it (pull it to a 120 mm
mark on a strip of card), stand the cassette at its start place with the loop flat on the mat, lay the pencil
(re-sharpened if its point broke) at its place, and return the arms to the ready pose. Set aside any cassette whose
tape is creased or stretched.
