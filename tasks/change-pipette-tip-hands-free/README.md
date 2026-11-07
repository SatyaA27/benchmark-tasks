# Change a pipette tip hands-free

Eject the pipette's used orange tip into the bin, fit a fresh yellow tip from the rack, and return the pipette to its stand. The task tests retaining a contaminated tip during transport, ejecting it into a bin, and pressing on a clean tip without touching either tip with a gripper.

Medical and clinical laboratory technicians and biological technicians change tips this way to prevent contamination between samples. The left arm handles the pipette throughout; the right arm parks away with its gripper open.

A robot gripper has no thumb to work the pipette's ejector. Instead, the sleeve's square flange passes under a fixed stainless steel fork on the bin. Raising the pipette against that fork pushes the passive sleeve down onto the used tip's rim. The nozzle then presses into a fresh tip in the rack, and passive spring-ball contact retains it during the return to the stand.

## Run a trial

On a machine with the official simulator and its mechanism support installed, with a Dream API key in `~/.dreamlabs/credentials`, run from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=change_pipette_tip_hands_free.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Eject the pipette's used orange tip into the bin, fit a fresh yellow tip from the rack, and return the pipette to its stand." \
  --scorer task_score --max-steps 1400 --store-frames --log-dir /tmp/pipette-dream

# The scripted demo, which reaches full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=change_pipette_tip_hands_free.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Eject the pipette's used orange tip into the bin, fit a fresh yellow tip from the rack, and return the pipette to its stand." \
  --scorer task_score --max-steps 1400 --store-frames --log-dir /tmp/pipette-demo

# One MP4 per camera, next to each run's frames
for log in /tmp/pipette-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end, reflecting the current end state and any earlier zero-out violation. Full score requires all goal conditions to hold continuously for 1 s; success then holds for a further 1 s to end the trial. Otherwise the trial ends at 1,400 steps and reports the partial credit below.

Offline checks of the scorer and demo geometry in an environment with `yam_arena` installed: `pytest -q -p no:cacheprovider .`

## Scoring

The tiers are cumulative. Losing a requirement removes its credit; each outcome earns the highest satisfied tier, otherwise zero.

| Score | Required state |
|---|---|
| 0.25 | Used orange tip at least 20 mm off the nozzle by origin distance, at rest, with the footprint of its world-aligned box inside the bin interior x=0.085 to 0.255 m, y=-0.023 to 0.073 m (2 mm tolerance), and its lowest point from 2 mm below to 10 mm above the bin floor top z=0.7505 m. It may lie flat or lean on the eject fork's inner plate. Any part over a wall or rim, half outside the bin, or perched on the fork earns zero. |
| 0.50 | Also either fresh yellow tip seated on the nozzle. |
| 0.75 | Also the pipette in its stand: the shoulder point, local (0,0,0.130) m, within 2 mm vertically of stand fork top z=0.8935 m; the shoulder y from 22 mm toward the slot mouth (-y) to 7 mm toward the back of y=0.180 m (the 56 mm sleeve flange meets the back wall's inner face at 6 mm, so no deeper park is reachable); for each rail, part of the yawed 44 x 12 mm shoulder bar's footprint lying at least 18 mm from x=0.260 m on that rail's side (1 mm onto the rail; rail inner edges 17 mm out) and from 28 mm toward the slot mouth to 40 mm toward the back of y=0.180 m (rail end to the back of the crossbar). At yaw 0 this means the shoulder within 4 mm of x=0.260 m; a yawed bar whose only reach onto a rail is a corner past the rail end earns no stand credit; pipette tilt under 5 degrees; and the other fresh tip standing in its own hole. |
| 1.00 | Also released and all parts at rest, with every full-score condition held continuously for 1 s. |

Four violations force zero for the rest of the episode:

- (a) Either gripper TCP comes less than 15 mm from any tip's world-aligned box.
- (b) The used tip is at least 3 mm off its seat, measured by origin distance in the nozzle frame, and the ejector sleeve has not been pushed at least 6 mm at any step since that tip was last seated.
- (c) Any fresh tip is neither seated, in its own rack hole, nor engaged on the nozzle. An engaged tip that later falls outside all three conditions triggers this rule. A nudge within the in-hole limits may settle or seat; lifting out and dropping onto the rack plate, table or bin forces zero.
- (d) Pipette tilt exceeds 45 degrees.

These violations clear only on episode reset.

**Seated:** absolute nozzle-frame axial origin error at most 1 mm, radial error at most 1.5 mm, and relative axis angle under 5 degrees.

**Engaged:** absolute axial origin error at most 6 mm, radial error at most 3 mm, and relative axis angle under 15 degrees.

**In its own hole:** origin xy within 6 mm of its own hole, collar underside (local z=4 mm) between 2 mm below and 5 mm above rack plate top z=0.7675 m, and tilt under asin(5/14) degrees (approximately 20.925 degrees, the largest physical bearing tilt within that height band).

**Standing:** for the 0.75 tier, origin xy within 3 mm of its own hole, collar underside (local z=4 mm) within 2 mm of rack plate top z=0.7675 m, and tilt under 10 degrees.

**At rest:** pipette reported speed below 10 mm/s and spin below 0.5 rad/s; each tip changes origin by less than 0.5 mm and full orientation by less than 1.5 degrees between consecutive 20 Hz control steps. Tip rest uses pose change because the clamped tip's reported velocity is solver noise.

**Released:** each gripper TCP at least 30 mm from the pipette's and every tip's box.

A halfway-pressed fresh tip 2.5 mm short of its seat earns disposal credit only (0.25). A seated fresh tip carried above the stand earns 0.5; parked while held or moving earns 0.75. A used tip still on the nozzle, swapped used/fresh identities, or the starting layout earns zero. Success is score 1.0 held for a further 1 s.

On a real bench, a ruler or calipers checks the tip's seat depth and centring, and a glance at the bin and rack confirms that the orange tip lies wholly inside the bin walls with its low end on the floor and the spare yellow tip remains upright in its own hole. Check the pipette's shoulder against the stand fork and observe the released parts over the hold period; the trial video also records any earlier contact, fall or improper removal.

## The scripted demo

The left arm grips the pipette's handle, lifts its shoulder out of the stand, and withdraws before travelling. It carries the used tip to the bin, slides the sleeve flange under the eject fork, and raises the pipette 16 mm so the fork pushes the sleeve down and ejects the tip. It waits, lowers 2 mm, withdraws from the fork, and travels around the rack to clear the other fresh tip.

Above the nearer yellow tip (`tip_a`), the arm settles for 0.8 s before descending. The approach stops 29 mm above the seat, then a slow insertion travels 27 mm in 2.25 s at 12 mm/s. The final 10 mm press takes 0.5 s, commanding 8 mm of overdrive beyond nominal seating, and holds for 1.2 s against the tip's physical stop, supported by the rack. The arm lifts vertically, carries the seated tip clear of the rack, and holds it in the air for 2 s to demonstrate retention.

The arm returns the pipette's shoulder to the stand, opens the gripper, raises the pads clear, and backs off. Both gripper TCPs finish at least 45 mm from every movable object's box, with a final 3.5 s hold. The right arm stays parked with its gripper open. The entire demo uses only the known starting layout and elapsed time.

## Runtime and seeds

The scripted plan lasts 40.65 s; adding the 1 s success hold gives 41.65 s, below 60% of the 1,400-step limit (70 s at 20 Hz). On the official simulator stack, the scripted demo ended in success after 777 steps (38.85 s at 20 Hz). Earlier runs of the same scripted motion, scored with an earlier version of the parking and bin checks, succeeded after 758 steps in each of three runs, and a copy with every demo segment 0.8x as long also succeeded. Wall-clock runtime depends on the host.

Use `object_jitter=0`. Seeds do not change this fixed starting layout.

## Setup and reset

Coordinates are in metres: x forward, y to the robot's left, z up, with the table top at z=0.7475. There are eight named objects:

- A blue pipette (RGB 0.12, 0.36, 0.75), initially upright in the stand at origin (0.260, 0.180, 0.7635). Its free base weighs 100 g. The body is polypropylene, the nozzle PVDF, and the passive ejector sleeve and four spring balls stainless steel. The handle is 24 mm across; its upper cap is shortened by 4 mm to give 5.5 mm palm clearance at full press. The shoulder is 130 mm above the nozzle origin.
- Three identical polypropylene tips, each 3 g and 64 mm long, with a 49 mm upper funnel and 15 mm lower stem tapering to 1 mm radius. The used tip starts seated at the pipette origin. It is dark orange, the stain of use (RGB 0.70, 0.35, 0.08). The fresh tips are saturated yellow (RGB 0.95, 0.78, 0.10), standing at origins (0.200, 0.105, 0.7635) and (0.130, 0.185, 0.7635).
- A dark grey acrylic stand (RGB 0.32, 0.35, 0.40), anchored at (0.260, 0.180, 0.7475). Its two fork rails support the pipette's shoulder at z=0.8935.
- A grey polypropylene rack (RGB 0.65, 0.66, 0.68), anchored at (0.200, 0.105, 0.7475), with plate top z=0.7675. Its four holes are at xy (0.200, 0.105), (0.130, 0.185), (0.130, 0.105), and (0.200, 0.185); the latter two start empty. The neck's 6.5 mm half-width in an 8 mm half-width aperture gives 1.5 mm side play.
- A purple HDPE biohazard bin (RGB 0.32, 0.25, 0.35), anchored at (0.170, 0.025, 0.7475), with a 170 by 96 mm interior and 27 mm walls above its inner floor.
- A metallic grey stainless steel eject fork (RGB 0.35, 0.40, 0.45), anchored on the bin at (0.170, 0.025, 0.7475). Its slotted plate catches the sleeve flange while the pipette rises.

The passive retention mechanism has a 6.5 mm square bore half-width, 2 mm radius balls, 275 N/m ball springs with 3 mm outward preload, 2 mm inward slide travel, and static/dynamic ball friction 0.8/0.6. The sleeve spring is 100 N/m with 1 mm outward rest, 14 mm travel, and a 6 mm initial rim gap (sleeve bottom 55 mm vs tip mouth top 49 mm above the nozzle origin). Radial capture is 3.5 mm: the tip mouth's 11.5 mm inner half-width minus the balls' 8 mm outer reach (6 mm radial position plus 2 mm radius). All task contact offsets are 0.4 mm. Fixtures have no simulated mass and remain anchored. Complete local shape centres, dimensions, materials, colours, masses, springs and starting coordinates are listed in the [Collider table](#collider-table) below. No other task props or marks are needed.

Reset the pipette upright in its stand with the used orange tip seated at the nozzle origin, both fresh yellow tips upright in their own holes, and every passive joint at zero as in the task module. Start all origins at yaw zero and let contact establish the ball preload. Keep the stand, rack, bin and eject fork anchored. For real trials, build the pipette (with its sliding ejector sleeve and four spring balls), the three tips, the stand, the rack, the bin and the eject fork to the [Collider table](#collider-table) below, for example machined or 3D-printed. Commercial pipette tips do not fit the 16 mm square rack holes or the scored seat geometry. Colour represents contamination; no biological hazard is needed.

## Collider table

### Conventions

All dimensions/coordinates are metres and masses kg. x forward, y left, z up; table top z=0.7475. All origins
start upright at yaw zero, passive joints q=0. Shape centres below are local to the named origin. Cylinders
have vertical axes. R(a,b,l,h) is a square annulus of four boxes: at (+/-(a+b)/2,0,(l+h)/2), size
(b-a,2*b,h-l), and at (0,+/-(a+b)/2,(l+h)/2), size (2*a,b-a,h-l).

### Pipette

Pipette origin (0.26, 0.18, 0.7635), blue RGB (0.12, 0.36, 0.75); polypropylene body with PVDF nozzle, 0.100
kg free base; base geometry: ball c=(0.0, 0.0, 0.004) r=0.004; cylinder c=(0.0, 0.0, 0.00995) r=0.004
h=0.0041; cylinder c=(0.0, 0.0, 0.016) r=0.0019 h=0.008; cylinder c=(0.0, 0.0, 0.022) r=0.004 h=0.004;
cylinder c=(0.0, 0.0, 0.0445) r=0.006 h=0.041; box c=(0.0, 0.0, 0.0965) size=(0.02, 0.02, 0.063); cylinder
c=(0.0, 0.0, 0.129) r=0.006 h=0.002; box c=(0.0, 0.0, 0.135) size=(0.044, 0.012, 0.01); cylinder c=(0.0, 0.0,
0.147) r=0.006 h=0.014; box c=(0.0, 0.0, 0.172) size=(0.024, 0.024, 0.036). The shoulder bottom is local
z=.130; the handle is the last box, with its upper cap shortened 4 mm to leave 5.5 mm palm clearance at the
deeper press while retaining the same lower grip surface.

Stainless steel ejector sleeve (0.008 kg): R(.007,.0155,.055,.063) and collar flange R(.0155,.028,.060,.064);
passive z slide anchored at (0,0,.055), travel [-.014,0], spring 100 N/m, rest +.001 m, damping .3 N*s/m.

Four stainless steel balls (0.001 kg each), radius .002, centres (+/-.006,0,.016) and (0,+/-.006,.016); each
has a passive radial slide anchored at its centre, inward travel .002, spring 275 N/m, outward rest .003,
damping .15 N*s/m. Positive-axis slides span [-.002,0] with rest +.003; negative-axis slides span [0,.002]
with rest -.003. Ball friction .8/.6 static/dynamic, base/sleeve/tip friction .1/.08; material combination is
average. The used tip compresses balls about 1.5 mm at spawn; they relax through passive joint travel. No task
object is attached or actuated.

### Tips (`tip_used`, `tip_a`, `tip_b`)

Three polypropylene tips: tip_used starts seated at (0.26, 0.18, 0.7635), RGB (0.70, 0.35, 0.08) dark orange
from stained liquid residue (no liquid simulated); tip_a at (0.2, 0.105, 0.7635), tip_b at (0.13, 0.185,
0.7635), both saturated yellow polypropylene RGB (0.95, 0.78, 0.10). Each mass .003 kg versus about .001 kg
real, to improve conditioning of the light tips against stiff spring slides. Identical tip geometry: cylinder
c=(0.0, 0.0, -0.0005) r=0.006 h=0.001; cylinder c=(0.0, 0.0, -0.003) r=0.0055 h=0.004; cylinder c=(0.0, 0.0,
-0.007) r=0.0045 h=0.004; cylinder c=(0.0, 0.0, -0.011) r=0.0028 h=0.004; cylinder c=(0.0, 0.0, -0.014)
r=0.001 h=0.002; R(.0045,.0065,0,.004); R(.0065,.014,.004,L) with L=0.01742449979984; 15 ramp rings for
k=0..14: R(.0065+(k+1)*.0001,.014,L+k*.00035,L+(k+1)*.00035); R(.0088,.014,L+.00525,.032);
R(.0095,.0105,.032,.037); R(.0105,.0115,.037,.042); R(.0115,.0125,.042,.049). The stop plate ends at local
z=0. Lower stem length is .015; the 225 mm TCP envelope plus 5 mm carried clearance prevents a 25-30 mm stem
with this unchanged 49 mm retention funnel. Radial capture (mouth inner half-width .0115 minus ball outer reach .008) is 3.5 mm and sleeve-to-rim gap 6 mm;
eight-millimetre commanded seat overdrive brings the nozzle onto the physical stop plate, with the rack plate
carrying the reaction and the arm/pads accommodating the overtravel.

### Stand

Fixed acrylic stand origin (0.26, 0.18, 0.7475), dark grey RGB (0.32, 0.35, 0.4): box c=(0.0, 0.038, 0.071)
size=(0.054, 0.008, 0.142); box c=(0.0, 0.034, 0.144) size=(0.054, 0.012, 0.004); box c=(-0.022, 0.0, 0.144)
size=(0.01, 0.056, 0.004); box c=(0.022, 0.0, 0.144) size=(0.01, 0.056, 0.004).

### Rack

Fixed polypropylene rack origin (0.2, 0.105, 0.7475), grey RGB (0.65, 0.66, 0.68). Four holes at world xy
((0.2, 0.105), (0.13, 0.185), (0.13, 0.105), (0.2, 0.185)), the latter two empty. Each has
R(.008,.025,.017,.020) translated by its xy offset from the rack origin. Connecting plate bridges and legs:
box c=(-0.035, 0.0, 0.0185) size=(0.02, 0.05, 0.003); box c=(-0.035, 0.08, 0.0185) size=(0.02, 0.05, 0.003);
box c=(-0.07, 0.04, 0.0185) size=(0.05, 0.03, 0.003); box c=(0.0, 0.04, 0.0185) size=(0.05, 0.03, 0.003); box
c=(-0.093, -0.023, 0.0085) size=(0.004, 0.004, 0.017); box c=(-0.093, 0.103, 0.0085) size=(0.004, 0.004,
0.017); box c=(0.023, -0.023, 0.0085) size=(0.004, 0.004, 0.017); box c=(0.023, 0.103, 0.0085) size=(0.004,
0.004, 0.017). Neck half-width 6.5 mm in an 8 mm half-width aperture gives 1.5 mm side play; collar underside
local z=.004 rests on plate top z=.7675.

### Waste bin

Fixed HDPE biohazard bin origin (0.17, 0.025, 0.7475), purple RGB (0.32, 0.25, 0.35): box c=(0.0, 0.0, 0.0015)
size=(0.176, 0.102, 0.003); box c=(-0.0865, 0.0, 0.0165) size=(0.003, 0.102, 0.027); box c=(0.0865, 0.0,
0.0165) size=(0.003, 0.102, 0.027); box c=(0.0, -0.0495, 0.0165) size=(0.17, 0.003, 0.027); box c=(0.0,
0.0495, 0.0165) size=(0.17, 0.003, 0.027).

### Eject fork

Fixed stainless steel eject_fork on bin at (0.17, 0.025, 0.7475), metallic grey RGB (0.35, 0.4, 0.45): box
c=(0.047, 0.0, 0.0485) size=(0.008, 0.072, 0.091); box c=(0.04275, 0.0, 0.096) size=(0.0165, 0.072, 0.004);
box c=(0.00175, -0.0285, 0.096) size=(0.0655, 0.015, 0.004); box c=(0.00175, 0.0285, 0.096) size=(0.0655,
0.015, 0.004).

### Contact and robot setup

Fixtures have no simulated mass and use the arena default friction. All task shapes have .0004 contact
offsets. No other task props or marks; 8 named objects total. The right robot parks with its gripper open at
(.13,-.19,.8675); the left grips the handle .150 above the nozzle origin. All fixtures avoid the camera strip,
and every commanded TCP stays within .36 planar of its own base.

## Fidelity

The pipette and tips are oversized mock-ups of the real tools. The tip mouth is 23 mm across inside, where real 200-1000 uL tips are about 5-8 mm. The tips have a 28 mm square seating collar. A square nozzle bore with four preloaded spring balls replaces the conical friction fit of a real nozzle. The ejector sleeve has a 56 mm square flange, and the fixed eject fork on the bin stands in for the thumb-operated ejector button, since a gripper has no thumb.

Primitive rigid bodies model the pipette and fixtures. Four passive spring balls stand in for the tip cone's interference fit; friction retains the tip, and sleeve-to-rim contact ejects it without attachments. Tips weigh 3 g rather than about 1 g for real tips to improve solver conditioning. No fluids, contamination chemistry, flex, wear or damage are simulated; colour represents contamination. Tip stillness uses pose changes because the clamped tip's reported velocity jitters.

No third-party assets; all geometry is primitives from the included kit.
