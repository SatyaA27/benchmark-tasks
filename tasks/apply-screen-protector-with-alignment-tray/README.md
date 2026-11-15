# Fit a glass screen protector with its alignment tray: seat the tray square round the phone, press, lift it off empty

Put the screen protector on the phone with its tray: set the tray down over the phone, press the middle of the glass, then lift the tray off and put it down beside the phone.

Screen protectors are the second most common phone accessory after cases: in a 2023 YouGov survey of 1,104 US adults, 46% said they use one. Spigen's Glas.tR EZ Fit comes with the glass already mounted in a rigid tray; in Spigen's words: "Simply place the tray over the phone, press the center and watch the glass adhere in place." Its installation video adds: "Place the tray flat on the screen without tilting." Its support page: "please remove the phone case before proceeding." The tray lines the glass up only if it goes on square: it fits closely round the phone, so it has to come down level and within about 2 mm. Set down further off, its rim lands on the phone's edge: it sits proud and crooked and the glass goes down off true. And the glass sticks only once it is pressed down flat: lift the tray without pressing and the glass comes away with it. Done right, the glass is on true and the empty tray comes off.

Sources: Apple, iPhone 16 Tech Specs (147.6 x 71.6 x 7.80 mm, 170 g). Spigen's GLAS.tR EZ Fit product description
("Less time, guaranteed alignment. ... comes with an innovative tray designed to eliminate the time wasted to achieve the
greatest alignment. Simply place the tray over the phone, press the center and watch the glass adhere in place.");
Spigen support: "For any screen protectors that utilize an EZ FIT tray or aligner, please remove the phone case before
proceeding." Spigen's installation video "Glas.tR EZ FIT for Apple iPhone" (its own installation-guide site), in its
captions: "Peel off the back film of the screen protector", "Hold the tray handles and keep it level", "Place the tray
flat on the screen without tilting", "Press & slide, then wait for 30 seconds", "Remove the sticker from the tray"; then
the tray is lifted off. One published review: lifting the tray before the glass had stuck pulled the glass away with
it. YouGov, "American consumers and cell phone accessories" (survey of 1,104 US adults, 28-29 November 2023): 46% use a
screen protector or guard.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=apply_screen_protector_with_alignment_tray.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Put the screen protector on the phone with its tray: set the tray down over the phone, press the middle of the glass, then lift the tray off and put it down beside the phone." \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial, replace `--policy yam_scripted` with `--policy dream`. `--scorer task_score` reports the score at
the end of the run. Each trial allows 2,400 control steps (120 seconds at 20 Hz). There are no
seeds: the layout is fixed. With `-E object_jitter` the phone and the tray (with its glass) are each moved by up to that
much; the glass is captive in its tray, so the task file moves the two together (the stock reset would move each body on
its own and pull the glass out of the tray). The scripted demonstration takes 47.1 s, the last
2 s holding still. `success()` requires a score of 1.0.

`pytest` in this folder runs the generated offline checks (no GPU): the scorer on the start, the goal and the named
partial and shortcut layouts.

## The scene

A desk in front of two arms (bases 61 cm apart), seen by one overhead camera and a camera on each wrist. Distances are measured forward from the line between the arm bases and sideways from the midpoint between them. Three objects. (1) An Apple iPhone 16 (147.6 x 71.6 x 7.80 mm, 170 g; teal aluminium frame, glass front and back), switched off, no case, screen wiped clean, lying screen up on the bare desk, its long side across the robot: centred 27.0 cm forward and 12.0 cm right of centre, its top end (camera) toward the left arm, rocked slightly on its camera bump. (2) Its Spigen Glas.tR EZ Fit installation tray (sizes estimated), clear moulded plastic (drawn dark grey so it shows), 76.4 x 152.4 mm (164.4 mm over its two flat end handles), 25 g: a 2 mm deck with a 54 x 120 mm press window and a 2 mm rim round its edge, and under the deck a skirt 4.5 mm deep that fits round the phone with 0.4 mm to spare a side, its bottom edge chamfered 1.5 mm as a lead-in; seated, the rim stands 5.5 mm above the screen, so the outside of each long wall is 10 mm tall. It lies skirt down on the desk 12.5 cm nearer the robot than the phone, in line with it, its back film already peeled. (3) The tempered-glass protector, 68 x 144 x 0.4 mm with its films, 8 g (drawn blue-tinted so it shows), held adhesive side down in the tray's recess under the window by the tray's strip sticker; with the tray sitting on the phone it hangs 1 mm above the screen.

### Parts and colliders

Three free rigid bodies. The phone's frame: origin in the middle of its back plane, x across the phone, y along it
(its top end, with the camera, toward +y), z out of the screen; it starts at (270, -120)
mm, its back 1.6 mm above the desk on its camera plateau. The tray's frame: origin in the middle of the deck's
underside (the screen plane when the tray is seated), axes as the phone's. The glass's frame: its centre.
Masses spread at uniform density over each part's colliders: phone 0.17 kg, tray 0.025 kg, glass
0.008 kg.

| Part | Collider | Centre (part frame, mm) | Size (mm) |
| --- | --- | --- | --- |
| phone | box | 0, 0, 3.9 | 53.6 x 147.6 x 7.8 |
| phone | box | 0, 0, 3.9 | 71.6 x 129.6 x 7.8 |
| phone | cylinder | 26.8, 64.8, 3.9 | radius 9, length 7.8 along z |
| phone | cylinder | 26.8, -64.8, 3.9 | radius 9, length 7.8 along z |
| phone | cylinder | -26.8, 64.8, 3.9 | radius 9, length 7.8 along z |
| phone | cylinder | -26.8, -64.8, 3.9 | radius 9, length 7.8 along z |
| phone | box | 0, 0, 7.5 | 67.6 x 143.6 x 0.8 |
| phone | box | 12, 52, -0.8 | 22 x 36 x 1.6 |
| tray | box | 37.2, 0, -1.5 | 2 x 152.4 x 3 |
| tray | box | -37.2, 0, -1.5 | 2 x 152.4 x 3 |
| tray | box | 0, 75.2, -1.5 | 72.4 x 2 x 3 |
| tray | box | 0, -75.2, -1.5 | 72.4 x 2 x 3 |
| tray | box | 36.25, 0, 0.7 | 3.9 x 152.4 x 1.4 |
| tray | box | -36.25, 0, 0.7 | 3.9 x 152.4 x 1.4 |
| tray | box | 0, 74.25, 0.7 | 68.6 x 3.9 x 1.4 |
| tray | box | 0, -74.25, 0.7 | 68.6 x 3.9 x 1.4 |
| tray | box | 32.6, 0, 1.7 | 11.2 x 152.4 x 0.6 |
| tray | box | -32.6, 0, 1.7 | 11.2 x 152.4 x 0.6 |
| tray | box | 0, 68.1, 1.7 | 54 x 16.2 x 0.6 |
| tray | box | 0, -68.1, 1.7 | 54 x 16.2 x 0.6 |
| tray | box | 37.2, 0, 3.75 | 2 x 152.4 x 3.5 |
| tray | box | -37.2, 0, 3.75 | 2 x 152.4 x 3.5 |
| tray | box | 0, 75.2, 3.75 | 72.4 x 2 x 3.5 |
| tray | box | 0, -75.2, 3.75 | 72.4 x 2 x 3.5 |
| tray | box | 0, 79.2, 1 | 30 x 6 x 2 |
| tray | box | 0, -79.2, 1 | 30 x 6 x 2 |
| tray | convex wedge (lead_xp) | 37.2, 0, -3.75 | 2 x 152.4 x 1.5 bounding box: the skirt's 45 degree lead-in |
| tray | convex wedge (lead_xm) | -37.2, 0, -3.75 | 2 x 152.4 x 1.5 bounding box: the skirt's 45 degree lead-in |
| tray | convex wedge (lead_yp) | 0, 75.2, -3.75 | 72.4 x 2 x 1.5 bounding box: the skirt's 45 degree lead-in |
| tray | convex wedge (lead_ym) | 0, -75.2, -3.75 | 72.4 x 2 x 1.5 bounding box: the skirt's 45 degree lead-in |
| glass | box | 0, 0, 0 | 68 x 144 x 0.4 |

## Scoring

Judged at the end; measure with a ruler or calliper and a photo. The glass is on when it lies flat on the screen (no corner lifted more than 0.3 mm). Its offset is the most any of its four corners lies from where it belongs (the border between the glass's edge and the phone's edge equal on opposite sides). 1.0: glass on, every corner within 1.5 mm, the tray off the phone (no part of it within 10 mm of the phone below 20 mm up), the hands at least 3 cm from the tray and the phone, everything still for 1 s. 0.5: glass on within 1.5 mm with the tray still on the phone, or glass on within 2 mm with the tray off. 0.25: glass on within 2 mm, tray still on. 0: anything else (glass still in the tray, lifted, tilted, or more than 2 mm off), or the phone pushed more than 30 mm from where it lay or turned over.

How each condition is read in `score()` (in the phone's own frame, so a phone nudged or rocked on its camera plateau is
judged the same):

- **Glass on:** no corner of the glass's underside more than 0.3 mm above the screen (the display's
  surface, drawn 0.1 mm proud of the frame so the two faces render apart; and, a simulation
  allowance, none more than 1 mm into it: pressed hard, the thin glass collider can sink a little).
- **Offset:** the largest distance, across the screen, of a glass corner from where it would be with the glass dead
  centre and square (1.8 mm border on the long sides, 1.8 mm at
  the ends). Full credit allows 1.5 mm: the tray's own play (0.4 mm round the phone, 0.3 mm round
  the glass, both estimates) lets a correctly seated tray put a corner up to 1.05 mm off once the small turns that
  play allows are added (worst case), and a glass 1.5 mm off still lies wholly on the screen with 0.3 mm of its
  1.8 mm border left. In the GPU runs the seated tray put it 0.4-0.95 mm off; a tray caught on its lead-in or
  set down crooked puts it further off (partial credit up to 2 mm, then 0).
- **Tray off:** no corner of any of the tray's boxes within 10 mm (across) of the phone's outline at
  less than 20 mm above its screen.
- **Hands clear:** both grippers' centre points at least 3 cm outside the tray's and the
  phone's bounding boxes. **Still:** reference points on all three parts moved less than 1 mm over the last 0.5 s.
- **Phone in place:** its middle within 30 mm of where it lay at the start, screen up (tilted under
  10 degrees), on the desk; once it fails, the trial scores 0.

## Scripted demonstration

The right hand does the work; the left stays parked (the maker's video holds the tray by its two end handles with two
hands, but the robot's hands cannot work that close together). As people also do, the right hand holds the tray
across its width: fingers down either side of its middle, the pads (opened to 93.9 mm, as wide as they go) closing on its
two long walls with about 15 N (each pad's target 3 mm inside the wall), so the tray hangs centred in the grip; on the
desk the pads' tips stop 2 mm above the desk (8 mm of the 10 mm wall held), on the phone 0.5 mm below the skirt's
edge (all of it). It lifts the tray, reads it and levels it in the hand, carries it over the phone, reads where the tray
and the phone are and lines the tray up with the phone (at most 40 mm and
10 degrees from the plan), lowers it until its skirt is 3 mm above the phone, reads again, lowers
it round the phone, lets go, presses the middle of the glass through the window with its closed fingertips (commanded
10 mm below the glass's top, about 6 N), then takes the tray by its long walls again, lifts it straight
up and puts it down where it started. Timeline:

| Time (s) | Step |
| --- | --- |
| 0.0 to 2.5 | hand over the tray's middle |
| 2.5 to 4.0 | pads down beside the long walls |
| 4.0 to 5.0 | hold the tray across its width |
| 5.0 to 8.0 | lift the tray |
| 8.0 to 8.6 | hold lifted |
| 8.6 to 9.6 | level the tray in the hand |
| 9.6 to 12.6 | carry it over the phone |
| 12.6 to 13.2 | hold over the phone |
| 13.2 to 14.4 | line up over the phone |
| 14.4 to 16.9 | lower to just above the phone |
| 16.9 to 17.5 | hold just above |
| 17.5 to 20.0 | lower round the phone |
| 20.0 to 20.6 | hold seated |
| 20.6 to 21.4 | let go of the walls |
| 21.4 to 22.4 | rise off the tray |
| 22.4 to 23.9 | over the middle of the glass, fingers closed |
| 23.9 to 25.4 | fingertips onto the glass |
| 25.4 to 26.4 | press the centre |
| 26.4 to 28.4 | hold the press |
| 28.4 to 29.4 | lift the fingertips |
| 29.4 to 31.4 | wait |
| 31.4 to 32.9 | hand over the tray again |
| 32.9 to 34.1 | pads down beside the long walls |
| 34.1 to 35.1 | hold the tray across its width |
| 35.1 to 37.1 | lift the tray straight up |
| 37.1 to 39.6 | carry it clear |
| 39.6 to 41.1 | set it down |
| 41.1 to 41.9 | let go |
| 41.9 to 43.1 | rise clear |
| 43.1 to 45.1 | park |
| 45.1 to 47.1 | hold clear |

## Simulation and the real kit

Rigid bodies; glass, adhesive and films are one 0.4 mm plate. The strip sticker holding the glass in the tray is a 0.1 N lift at each end with a 1.5 mm stretch limit. Adhesion is a rule: flat on the screen for 0.5 s, the glass stays where it lies and the strip lets go, so Spigen's 30 s wait and peeling the strip off the tray are not done. Starts after the back-film peel. Tray sizes are estimates (rim height read off Spigen's install video, the rest guessed), as are the masses.

- How the strip sticker and the adhesive are modelled (`_Model` in the task file, run before every physics step): the
  glass hangs in the tray's recess, its top against the recess's ceiling, held there by a lift of 0.1 N at each
  end (the strip sticker's hold on the tray); it can be pressed down at most 1.5 mm (the strip's
  stretch), so it reaches the screen only with the tray seated. Once the glass has lain flat on the screen (no corner
  more than 0.3 mm above it, every corner over the phone) for 0.5 s, the adhesive holds it flat on the screen where it lies (its
  position and turn on the screen kept): from then on it moves with the phone, and the strip lets go of the tray.
  Pressed only at an end, the light hold lets the rigid glass go flat anyway (a real glass would bend and still
  stick); lifted before it was pressed flat, the glass stays in the tray and comes away with it.
- Not simulated: the back film's peel (the trial starts after it); the maker's 30 s wait and its step "Remove the
  sticker from the tray" before the tray is lifted (a film: here the strip lets go by itself once the glass has
  stuck); the press is a press at the centre (Spigen's listing) rather than the video's press and slide along the
  strip; after the trial, the front film's peel and pushing out bubbles; dust, the phone's side buttons (the real
  tray's skirt clears them) and the screen's slight edge curve. The real tray is clear plastic; it is drawn dark grey.
- From the maker: the phone's size and mass. Read off frames of the maker's video (about 0.7 mm a pixel, with the
  phone's 147.6 mm length as the scale): the seated rim's top about 5-7 mm above the screen (modelled
  5.5 mm). Estimates, to measure on the real kit: the tray's pocket clearance (0.4 mm a side), wall
  (2 mm), skirt depth (4.5 mm; the clear skirt's edge is not visible in the video) and lead-in
  (1.5 mm at 45 degrees), deck (2 mm), recess (1.4 mm), press window
  (54 x 120 mm), the flat end handles (30 x 6 x
  2 mm) and mass (0.025 kg); the glass's size (68 x 144 mm), thickness with its films
  (0.4 mm) and mass (0.008 kg); the strip's hold and stretch and the adhesive's tack time; the
  phone's corner radius (9 mm) and camera plateau (22 x 36 mm,
  1.6 mm proud); frictions (phone on the desk 0.4, tray 0.35).

## Reset

Each trial uses up one pre-loaded tray: peel the glass off the screen and wipe the screen, then lay out a fresh tray (the kit has two; Spigen sells EZ Fit refill packs) with its back film peeled. Lay the phone and the tray at their taped marks.

A real trial needs no fixture: an iPhone 16 with no case, an EZ Fit kit for it, a desk. Wipe the screen and peel the
tray's back film just before the trial starts.
