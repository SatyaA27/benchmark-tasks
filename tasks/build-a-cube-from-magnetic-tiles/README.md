# Build a cube from six magnetic building tiles: stand four walls on the base so their edges catch, then lower the lid on

Build a closed cube from the six square tiles: stand four of the tiles from the rack on the edges of the flat tile so their magnets catch, then put the last one on top as the lid.

## Scoring

Judged after the run from the trial video and a photo, then a lift test. An edge is joined when the two tiles meet along it with no gap (in simulation: both magnets of that edge within 3 mm of the other tile's). A wall stands: within 10 degrees of upright, joined to the base. Tiles count only when both grippers are at least 3 cm from them and no tile has moved over the last second. Lift test (real trial, after the robot has let go): pinch the cube's top at two opposite edges and lift it 5 cm for 2 s; it stays together only if every edge is joined. It is done once, after the photo: if only the lid comes away, the score is 0.75; if a wall or the base comes away, that wall's edges count as not joined (at most 0.25). 1.0: closed cube: base, four walls and lid, all 12 edges joined (it passes the lift test). 0.75: base, four walls and all 8 of their edges joined, with the lid lying on top within 15 mm of centre but not joined on all four edges (it fails the lift test). 0.5: base and four walls with all 8 edges joined, no lid on top. 0.25: one to four walls standing joined to the base, but not all 8 wall edges joined. A lid that misses, falls off or falls inside leaves the score of the walls still standing (0.5 if all four stand joined). 0: no wall standing joined to the base, or the base no longer lying flat on the table. Doing nothing scores 0.

How `score()` reads it (each tile's own frame: centre, normal along local z):

- **Joined edge:** each tile carries eight magnet points (two per edge, 1.5 mm in from the edge's ridge, 19 mm
  either side of its middle). Tiles A and B are joined along an edge when at least two of A's points have a point of
  B within 3 mm (two tiles joined at a right angle sit 2.12 mm apart point to point).
- **Base:** a tile within 10 degrees of flat with its centre within 4 mm of lying on the table. **Wall:** a tile
  within 10 degrees of upright joined to the base; its side is the one its centre lies toward. **Corner:** two
  neighbouring walls joined. **Lid on top:** a flat tile whose centre is within 6 mm of one tile-width above the base
  and within 15 mm of the base's centre sideways; its joined edges are counted against the walls.
- **Counted only when let go and at rest:** a tile counts only if both grippers' centre points are at least
  30 mm outside its bounding box; every tile must have moved less than 1 mm over the last
  1 s. The structure that scores best over all flat low tiles is taken.

`success()` is a score of 1.0. Why the shortcuts fail: a wall let go a few millimetres too far out, tilted or
skewed is not caught and falls; tiles leaned against the base or each other without their edges meeting are not
joined; a lid dropped or set down crooked rests on the walls without catching all four edges (0.75 at most), and
one that lands hard can push a wall out. Our wrong-way run lets each wall go 8 mm outward of its place instead of 2 mm: three of the four were not caught and fell outward, lying flat beside the base; the fourth was pulled in and stood: 0.25.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=build_a_cube_from_magnetic_tiles.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Build a closed cube from the six square tiles: stand four of the tiles from the rack on the edges of the flat tile so their magnets catch, then put the last one on top as the lid." \
  --scorer task_score --max-steps 3000 --epochs 1 --store-frames --log-dir ./demo-run
```

For a model trial use `--policy dream`. Each trial allows 3,000 control steps (150 s at 20 Hz);
the scripted demonstration takes 83.2 s. `pytest` here checks the scorer on named layouts.

## The scene

Products, sizes and positions are in task.json's `setup` (mm from the midpoint between the arm bases; x forward,
y to the robot's left; heights above the table). Both arms start at their ready pose.

- **Six tiles (free bodies, 32 g each):** a slab 76.2 mm square ridge to ridge and 6 mm thick, one convex
  body: flat faces 70.2 mm square, every edge two 45-degree chamfers meeting at a ridge on the mid-plane (the real
  rim is rounded), so two tiles meet edge to edge at any angle and six close into a cube with every edge
  chamfer-to-chamfer. Friction 0.45/0.35.
- **Magnets (`magnet_kit.py`):** between any two magnet points on different tiles an attraction
  F = 0.70 N x ((1 + 2.12/5) / (1 + d/5 mm))^4, faded to zero between 12 and 15 mm,
  applied to both tiles (equal and opposite, with its torque) before every physics step; no joints, pins or
  grooves. A right-angle edge (two pairs) takes 1.40 N to pull straight off; two tiles joined flat sit
  3 mm apart and take 0.88 N; one edge pulls with a tile's weight (0.31 N) from a 3.2 mm gap and
  with 0.1 N from 6.7 mm. The points sit inside the rim, so a lone wall has a small righting pull and
  tips over past about 7 degrees of lean.
- **Rack (fixed):** two bamboo bars 18 mm wide, 30 mm tall, 30 mm apart, slots 8 mm wide every
  20.3 mm down to 6 mm above the table; friction 0.5/0.4. A seeded reset moves free objects by up to the
  jitter; a tile in a slot keeps its slot's x (it can only shift along its slot).

## Scripted demonstration

The right hand does all the work; the left arm parks. For each wall it reads where the rack tile stands (never moves
it), comes down with the fingers either side of the tile's top edge, pinches the 6 mm frame 8 mm under the top
ridge (pads aimed 3 mm apart), slides the tile up out of its slot, stands it upright, carries it high over the
walls already up, lowers it over the base's edge, reads the tile's pose in the hand and corrects, lowers it to
2 mm outward and 1 mm above the joined place and lets go: the magnets pull it the rest of the way and catch the
base and the wall beside it. The lid is turned flat held by one edge; its far edge goes onto the far wall's top
first, then the near edge comes down to 18 mm over its seat and is let go (the lower finger would otherwise hit
the near wall), and the lid swings shut on the far edge.

| Tile | Becomes | Let go and clear by (s) |
| --- | --- | --- |
| tile_1 | wall px | 15.2 |
| tile_2 | wall py | 30.4 |
| tile_3 | wall nx | 45.6 |
| tile_4 | wall ny | 60.8 |
| tile_5 | lid | 78.2 |

In our recorded demonstration (no object jitter) every wall was let go 2 mm outward and 1 mm above its place and the magnets pulled it the rest of the way; the lid swung shut on its far edge from 18 mm. Full credit came at 78.2 s: all 12 edges joined, the lid's centre at its planned height; the run ends once the closed cube has stayed still for 1 s with both grippers more than 3 cm from every tile (the scorer's own rule), so the video stops while the hand is still lifting away. In earlier probe runs of the first two walls with every object shifted up to 10 mm, both walls caught the base and each other in 2 of 3 trials; in the third the second wall came down 6 mm inside its line and knocked the first over (the demonstration now reads the tile in the hand before that descent).

## Simulation and the real objects

Tiles are rigid slabs at true size and mass; the rounded rim is two 45-degree chamfers. Each edge's two magnets are two attracting points, applied as forces every physics step, no joints: a corner edge takes 1.4 N to pull off and pulls a tile's weight from a 3 mm gap (estimates from published hang tests). The rack's slot sizes and all frictions are estimates.

- Tiles: MAGNA-TILES Classic 32-Piece Set (item 02132, $49.99 at the maker; 14 squares among its 32 pieces). FAQ: "Each
  MAGNA-TILES piece has a 3-inch base measurement", food-grade MABS plastic, ceramic (ferrite) magnets
  (https://magnatiles.com/pages/faq; set: https://magnatiles.com/products/magna-tiles-classic-32-piece-set).
  One square weighed 32 g (https://twomamabears.com/connetix-vs-magna-tiles/). Two magnets per edge, eight per
  square, arranged so any two edges attract (https://rothmanshore.com/2013/03/03/the-secret-of-how-magna-tiles-work/).
- Magnet strength: no maker figure. Two published hang tests (https://craftykidsplay.com/connetix-vs-magna-tiles/,
  https://celebratingwithkids.com/magna-tiles-vs-picasso-tiles/) hang a chain of 5 squares from a held one and lose it
  at 6: one edge holds 4 x 32 g (1.26 N) and lets go below 5 x 32 g (1.57 N). The model's corner edge takes 1.4 N, its
  flat joint 0.88 N (weaker than the tests). One-minute check: lay a tile flat on a kitchen scale, join a second at a
  right angle on its edge, pull the second straight up and read the drop when it lets go; repeat with a 1, 2 and 3 mm
  card shim between the edges for the catch.
- Estimates: tile thickness 6 mm (ruler), the chamfered edge shape, the rack's slot sizes (any bamboo plate rack with
  0.8 in slots; ruler), all frictions. Cost: one set and a rack, under $80.

## Reset

About 1 minute. Pull the cube apart (wall by wall; the magnets let go with a light pull) and pick up any fallen tile. Lay one tile flat at the base's place, edges along x and y, and stand the other five on edge in the five alternate rack slots, faces toward the robot. Use the same six squares every trial. Return both arms to the ready pose.
