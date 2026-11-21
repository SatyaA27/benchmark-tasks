# Mount a flash on a camera: hold the camera, slide the flash's foot into the hot shoe to the stop and lock its lever

Hold the camera still by its grip with one hand. With the other, pick up the flash, slide its foot into the camera's hot shoe from the back until it stops, let go, and slide the lock lever on the flash's foot across to LOCK. Then let go of the camera.

## Scoring

Judged at the end of the run from the trial video, a photo and a ruler. Score 0 if the camera fell over (tilted more than 30 degrees at any time) or the flash's foot is not in the shoe (its front edge less than 3 mm inside the rails' rear end, or off the rails); doing nothing scores 0. Otherwise the lowest of: foot: 1.0 if home (its front edge within 1.5 mm of the shoe's front stop: the foot then sticks out of the shoe's back by 3.5 mm or less, 2.0 when fully home), 0.5 if 1.5 to 8 mm short, 0.25 if more than 8 mm short; lever: 1.0 at LOCK (slid 6 mm or more of its 7 mm travel; ruler against the tab), 0.75 if slid 2 to 6 mm, 0.5 if less; camera: 1.0 if within 20 mm of its taped outline and turned 10 degrees or less, else 0.5; hands: 1.0 if both grippers are 30 mm or more from the camera and the flash and the flash has been still for 1 s, else 0.5.

How `score()` reads each condition (camera frame: origin on the table under the lens axis in the plane of the
camera's back, x toward the lens, z up; flash frame: origin at the middle of the foot's underside, x toward the
window, z up):

- **In the shoe:** the foot's front edge at least 3 mm past the shoe's rear end (4 mm in front of the
  back), the foot's middle within 2 mm of the shoe's axis and of its seated height, the flash upright within 20
  degrees. **Short of the stop:** 21 mm (the stop) less the foot's front edge, along the camera's x.
- **Lever:** how far the lever has slid from UNLOCK (0) toward LOCK (7 mm, its end stop); LOCK is 6 mm or more.
- **Camera:** its origin's distance from its start, and its turn about the vertical; **fell over:** tilted more than
  30 degrees at any time (remembered).
- **Hands clear:** both grippers' centre points at least 30 mm outside the flash's and the camera's bounding
  boxes; **still:** over the last 1 s the flash moved under 1 mm and turned under 2 degrees.

`success()` is a score of 1.0. Shortcuts do not pay: a flash stood beside the camera, or resting on the shoe's
rear edge, scores 0; a foot left short of the stop caps the score at 0.5 or 0.25 however the lever is set; an
unlocked flash caps it at 0.5; a camera shoved off its mark caps it at 0.5. Wrong way (the foot left short): the left hand lets go 6 mm short of the stop and still locks the lever. The foot then sits about 10 mm short, tilted 8 degrees, so the score is 0.25 (foot more than 8 mm short) although the lever is at LOCK.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=mount_flash_on_camera_hot_shoe_and_lock.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Hold the camera still by its grip with one hand. With the other, pick up the flash, slide its foot into the camera's hot shoe from the back until it stops, let go, and slide the lock lever on the flash's foot across to LOCK. Then let go of the camera." \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial use `--policy dream`. Each trial allows 2,400 control steps (120 s at 20 Hz);
the scripted demonstration takes 43.8 s. `pytest` here checks the scorer on named layouts.

## The scene

Products, sizes and positions are in task.json's `setup`. Colliders are convex pieces built from the published sizes
and the estimates below.

- **Camera (one free body with two links, 485 g, friction 0.45/0.4):** body
  64 deep, 78 tall, from 75 mm right to 54 mm left of the lens axis; grip to
  77.6 deep on the right; prism hump to 96.2 mm; eyecup 36 wide, 18 deep, 72-95 mm up; lens mount and body
  cap 60 mm across. Hot shoe: floor at 98.2 mm, slot 18.6 x 2.1, lips 1 thick and 12.5 apart, outside
  20.7 wide, front stop 21 mm in front of the back, 1 mm 45-degree lead-ins at the rails' rear ends.
  The shoe's spring: a plate 10 mm long in the floor on a slide (spring 3000 N/m, preload 0.3 mm,
  travel 0.8 mm), 0.5 mm proud at rest; it presses the foot up against the lips with about
  0.9 N (1.8 N with the foot over it; a 3.5 mm ramp at its rear end), so the foot drags about 1 N (friction 0.2/0.15; estimates).
  The shoe's rails are a link of their own on a slide that a 20 N preload holds at its stop (0.1 mm travel), in
  effect fixed: the kit gives each body one friction, and this lets the metal rails have metal's friction.
- **Flash (one free body with two links, 387 g, friction 0.55/0.45):** foot plate
  18.2 x 19 x 1.9 with its front corners cut 2 mm and its front edges 0.6 mm; a neck 11 mm
  wide through the lips (foot and neck: a link held like the shoe's rails, metal's friction); foot housing 36 x 28 mm,
  4.5-14 mm up; body (battery box) 70.5 x 50 mm, 14-69 mm up; head
  70.5 x 98.2 mm to 113.8 mm up, a 2.5 mm shell, reaching 48 mm in front of the body. The mass is spread
  by volume (body about 280 g with the cells, head shell about 90 g): centre of mass about 53 mm above the foot.
  Lever: a bar across the housing's back, 4 mm proud, with a tab 6 mm proud and 8.4 mm tall at its left
  end, on a slide along the flash's width (travel 7 mm, friction 2 N to start, 1.5 N moving; estimates).

## Scripted demonstration

The right hand comes in from the camera's right, pitched 45 degrees, and pinches the grip front to back (so it
holds the camera along the push) for the whole insertion and the lever. The left hand pinches the lying flash across
its body (front and back faces), lifts it, stands it upright by turning about the pinch, carries it behind the camera,
holds it still 30 mm over the shoe's rear end (the arm sags under the flash, so the hand's aim is corrected from
a reading there), lowers the foot to the shoe's rear end, slides it in aiming 10 mm past the stop in 8 steps
(the arm's give sets the push; once more if the foot stopped short), eases off, rests the flash in the shoe, lets go,
backs its fingers out to the left and, fingers closed and pitched 45 degrees, pushes the lever's tab
across to LOCK. The demonstration reads (never moves) the camera's pose once and the flash's pose at the pick, over
the shoe, at its entry, at the stop and before the lever.

| Time (s) | Left hand |
| --- | --- |
| 0.0 to 8.7 | pick up the flash, stand it upright |
| 8.7 to 12.5 | carry it behind the camera, hold it over the shoe |
| 12.5 to 21.7 | line the foot up, check, slide it in (in steps), push again if short |
| 21.7 to 25.0 | ease off, rest, let go, back the fingers out |
| 25.0 to 32.1 | close the fingers, go behind the tab |
| 32.1 to 37.3 | push the lever to LOCK (in steps), hold, back off, lift away |

Demo (demo run, no jitter): 1.0, full credit at 39 s of 120. The foot ended at the stop, the lever at LOCK, the camera within 2 mm of its mark.

## Simulation and the real objects

Rigid bodies. The shoe's leaf spring is a sprung plate (est. ~1.8 N on the foot; metal-on-metal friction 0.2, est.); lead-in chamfers on the foot and rails are estimates. The lever is a 7 mm slide held by friction (est. 2 N) with no click, and the locking pin it drives is not modelled: LOCK is read from its travel. The flash's head is a hollow shell over a solid battery box (centre of mass low, est.); head joints locked. Shapes beyond the published sizes, masses and frictions are estimates.

- Camera: Canon EOS Rebel T7 instruction manual, "Dimensions (W x H x D): Approx. 129.0 x 101.3 x 77.6 mm",
  "Approx. 475 g (Including battery and card)", hot shoe, eyecup, grip and body cap on pp. 26-27
  (https://gdlp01.c-wss.com/gds/0/0300029810/01/eos-rebelt7-1500d-im-en.pdf). Price: new with a kit lens about
  $479, Canon-refurbished kits $219-299
  (https://community.usa.canon.com/t5/EOS-DSLR-Mirrorless-Cameras/EOS-Rebel-T7-New-vs-Refurbished/td-p/483861);
  a used body that no longer takes pictures works for this task.
- Flash: Canon Speedlite 430EX III-RT instruction manual, "Dimensions (W x H x D): Approx. 70.5 x 113.8 x 98.2 mm",
  "Approx. 295 g (Speedlite only, excluding batteries)", p. 17: "Slip the Speedlite's mounting foot all the way into
  the camera's hot shoe ... Slide the mounting foot lock lever to the right. When the lock lever clicks in place, it
  is locked." (https://gdlp01.c-wss.com/gds/0/0300019910/02/sl430exiii-rt-im2-en.pdf); $299.99
  (https://www.usa.canon.com/shop/p/speedlite-430ex-iii-rt).
- Hot shoe: ISO 518:2006, inside 18.6 (+0.2/-0) x 16.0 (+3/-2) x 2 (+0.15/-0) mm, teeth 12.5 (+0.4/-0) mm apart
  (https://en.wikipedia.org/wiki/Hot_shoe).
- Estimates, each checkable in a minute: the lever's travel and tab size and the foot's size (ruler or calipers); the
  flash with cells (kitchen scale); the force to slide the foot in (stand the camera on its front, body cap down, on
  a kitchen scale and read the rise while pushing the foot down into the shoe) and to slide the lever (press the
  scale's edge against the tab); the camera's shape beyond its published box; every friction.

## Reset

About 1 minute. Press the flash's lock-release button, slide the lever back to UNLOCK (left) and slide the flash off the shoe toward the back. Lay it on its right side at its start place (window away from the robot, foot toward the robot's left, the middle of the foot's underside at x = 190, y = 232). Stand the camera back in its taped outline, back to the robot, body cap on. Both stay switched off (the task needs no working electronics, so a used body works). If the camera fell over, check the shoe and the foot for bends before the next trial. Return both arms to the ready pose.
