# Clean a tapered battery post with a terminal brush

This task is **post-only**; no cable clamp is present or scored.

Score first: P is the visibly clean fraction of the initially opaque-coated negative post taper. Compare close start/end photos and the trial video, using three height bands and 24 angular sectors (72 approximate equal-area patches). View hidden sides in video. Every nonzero tier first requires cup seating for at least one continuous second: cup rim 129–142 mm above the bench (at least 8 mm below the post top), within 3 mm coaxial offset and 10 degrees tilt. Judge this from the trial video with a ruler reference. Any contact between a gripper link and the coated taper during the run is a fail, even after that patch is clean; score 0. Apply the disjoint rules in order:

| Score | End condition |
|---|---|
| 0 | No qualifying cup seating, any gripper/taper contact, brush or cap below the bench/floor, or P=0. |
| 1.0 | P>= 95%, brush released and still on bench within 40 mm of its start mark, both hands 30 mm clear. |
| 0.75 | Not 1.0 and P>= 75%. |
| 0.5 | 50%<=P<75%. |
| 0.25 | 0<P<50%. |

Full cleaning without return is 0.75; doing nothing is 0 because all scored patches start black. Still means no visible motion for the last second (simulation speeds <10 mm/s and <0.1 rad/s). Only the post is scored: there is no cable-clamp objective or cable in the scene. The ordinary two-ended tool retains its unused full-size cone. The dust cap starts detached on the bench; cap removal is outside the trial.

## Setup, sources and cost

Bench service uses an intact retired sealed battery, strapped in a tray and isolated from any circuit. Its other terminal stays covered. Use clean brass SAE adapters and dry chalk marker as the non-lead corrosion stand-in. Never brush bare lead or actual corrosion on this shared rig. The [maker technical manual](https://www.odysseybattery.com/wp-content/uploads/AMER-EN-TM-ODY.pdf) identifies its SAE terminal type as brass. All meshes are procedural; published facts below were fetched, not physically measured.

| Object | Published facts and source |
|---|---|
| Osborn 0008493100 / EUMBHB018 | Plastic body, internal scratch strip, external steel cone, protective cap; 120 mm overall, 65 mm trim, cone 15–25 mm. [Maker catalogue, p134](https://www.osborn.com/downloads/literature/emea/main-catalog/osborn-main-catalog-en.pdf). [Retail](https://binnisf.com/battery-terminal-brushes-osborn-germany) KWD2.950. USD15 budget estimate. |
| ODYSSEY ODS-AGM28L (PC925), Group 28 | New USD279.99. [Maker](https://www.odysseybattery.com/products/ods-agm28l-battery-pc925/). [Maker metric datasheet](https://cdn.tayna.com/datasheets/Odyssey%20PC925%20Battery%20Datasheet.pdf): 168 x 175 x 127 mm, 119 mm receptacles, 10.8 kg, 134 mm terminal spacing in drawing.  |
| Odyssey 3217-0006 SAE adapters | 28 mm overall, 30 mm black base, M6 mounting, USD19.95 at [Battery Mart](https://www.batterymart.com/p-odyssey-sae-terminals.html); [maker](https://www.odysseybattery.com/products/sae-terminal-kit/) USD14.20 and lists PC925 compatibility. Maker size is packaging. |
| Tapered SAE post | 19.5/17.9 mm base diameters, 1:9 diameter taper; 20 mm overall including 2 mm shoulder in [Discover maker drawing](https://assets.discoverbattery.com/documents/09_discover-general/885_product_application_literature/gui-terminal-torque-specifications-guide.pdf). 18 mm taper gives 17.5/15.9 mm tops. [Exide guide](https://exide.sk/asset/uploads/na%20stiahnutie/NEW_Exide_Edition5_TechnicalGuid.pdf) also shows these top sizes. |
| Attwood 9092-5 tray | Strap holder, 7 x 11 in internal, [published catalogue](https://images.carid.com/attwood/items/pdf/attwood-catalog.pdf). [Retail](https://www.marineoutfitters.ca/index.cfm?category=10002%7C10081&code=022697909251&product=83310906) CAD14.99; use catalogue dimensions (retail text is for larger tray). USD20 estimate. |
| edding 4095 black marker | Water-based chalk, 2–3 mm nib, removable from nonporous surfaces: [maker](https://www.edding.com/products/edding-4095-chalk-marker/). [Black marker price](https://www.yourartsupplies.com/drawing/markers-and-pens/chalk-markers/edding-chalk-markers/edding-4095-chalk-marker-black.html) USD5.15. |

Procure an intact sealed retired PC925 from a battery-service shop: USD 50 allowance, explicitly an estimate to confirm with the shop. Add USD 15 brush, USD 20 adapters, USD 20 tray, USD 5.15 marker and USD 8 covers/fasteners: about USD 118 total. A new battery would raise the total to about USD348. Never open or drain the battery; reject damaged/leaking cases. Marker/covers are consumables; estimated coating cost under USD 0.10/trial plus covers.

Coordinates in mm from the midpoint between arm bases: x forward, y left, z above the bench. Battery centre (307,-147), long axis along x; negative taper base (240,-220,132), top 150. Attwood tray base on bench, strap crosses battery centre away from posts. Uncapped brush stands cup-down at (230,-365), origin 1 mm above bench before settling. Detached cap lies beside it: open-end centre (310,-340,19.5), axis +x. Both arms ready/open. Grip only the plastic body; left hand parks. The battery is removed for isolated bench service.

Estimates below have one-minute checks:

- Brush body diameter 35 mm, length 55 mm (120 overall minus 65 trim); cup depth 20 mm, mouth 25 mm then 17.2–15.2 mm inner bristle taper, core 3 mm: ruler. Confirm the accessible plastic grip band and >= 3 mm gripper hardware clearance. Cap 38 mm diameter x 75 mm, overlap 5 mm: ruler. No cap retention is modelled.
- Brush 90 g, detached cap 30 g: kitchen scale. Battery geometry uses published dimensions; adapter pedestal 10 mm including shoulder, recess details and 73 mm fore/aft terminal offset are estimates, ruler check. Tray floor/strap 3 mm: ruler. Terminal spacing 134 mm is published.
- Force/stiffness/friction values are effective estimates: rehearse loaded brushing over a kitchen scale (roughly 200–1000 g-equivalent = 2–10 N); use a ruler-marked 1 mm bristle deflection against the scale to check aggregate stiffness. Tilt a dry board until the plastic body slides to check friction. Calibration only, not real scoring.
- Pad covers: sacrificial 0.25 mm polyurethane under nitrile finger covers. Check thickness on packaging or ten stacked layers against a ruler. Covers keep wire, lead and dry debris off the pads; the grip contacts plastic only.

## Preparation and reset

Apply one opaque coat of edding 4095 black to the complete taper (about 1 minute). Dry at least 1 minute and until a clean swab picks up no wet ink; drying time is an estimated protocol, not a maker quick-dry claim. Photograph the fully coated start. Keep the rig dry; no wet marker, cleaner, loose powder or debris touches pads.

Reset in about 3 minutes: brush debris into the tray, damp-wipe the post away from the robot, dry, recoat and dry-check. Keep the cap off beside the uncapped upright tool at the specified poses. Keep the battery strapped and other post covered. Replace torn/dirty covers, wipe the tray, arms ready, photograph the new start. Use no actual corrosion, bare lead, loose chalk powder or liquid cleaning step during the trial. Catch residue in the tray, damp-wipe it away from the robot and replace dirty pad covers.

## Fidelity and physical cleaning rule

`battery_kit.py` is the self-contained reusable part module. Four radial spring sectors approximate individual cup bristles:5,000 N/m each,3 mm outward travel, damping 3 N s/m. The unused cone also has four sectors:800 N/m,4 mm inward travel, damping 1.2 N s/m. Helper carriers are 3 g for solver stability; total brush 90 g. Frictions are estimates: cup 0.7/0.6, cone 0.65/0.5, plastic 0.8/0.7. The 30 g cap is a free shell; no retention or scripted part motion.

Each post patch clears only after 20 mm accumulated tangential bristle sweep while its sector carries >= 0.25 N and total radial bristle load is >= 2 N, calculated from physical spring deflection. The cup axis must be within 10 degrees of the post axis, with actual bristle overlap. A loaded stationary press removes nothing; unloaded motion removes nothing. Only material colours change, never colliders, masses, joints or poses. Sector loads and coarse patches approximate local wire contact. This rule models a dry repeatable coating, not the chemistry or required force of every corrosion deposit. The robot still performs the grasp, coaxial placement, loaded twist and withdrawal. Seating history gates every tier. In simulation only this bristle rule clears patches, so fingers cannot clean the post. The gripper/taper contact fail is judged from video; the code's PhysX contact observer got no events on GPU and is not relied on.

Expert cup-height feedback compensates arm give with <=11 mm command lead (estimated 4.4–11 N), reading tool pose and driving ordinary gripper targets. Jaw target 32 mm on estimated 35 mm body is 3 mm inward. Twist while withdrawing slowly, then move clear of the case before lowering. Maximum planned grip height 219 mm; no added handle, tab or jig.

Offline cone/core proxies stop at 95 mm because the generic palm proxy starts 50 mm behind TCP; actual GPU meshes extend the full 120 mm. Real central gripper hardware starts 70.5 mm behind TCP:55 mm body grip leaves 5.5 mm cone-tip clearance. Paint uses one shared offline post proxy; GPU colliders retain every surface patch. Sparse proxies do not change physical geometry.

## Run and score

From this folder in Arena's environment:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=clean_battery_post_and_cable_clamp_with_terminal_brush.py -E object_jitter=0 -E control_hz=20 --instruction "Clean the post and return the brush" --scorer task_score --max-steps 3200 --store-frames
```

Use `--policy dream` for Dream-1. Limit 3,200 steps = 160 seconds at 20 Hz; end-result scoring. The folder/module identifier is retained; the task is post-only.
