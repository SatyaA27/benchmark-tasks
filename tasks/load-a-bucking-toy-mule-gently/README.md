# Load the bucking toy mule: hang a rope, a canteen and a pan on its saddle so gently that it does not buck

**Instruction:** "Hang the three pieces of gear on the toy mule's saddle without making it buck: the coiled rope over the saddle horn, the canteen and the frying pan over the two hangers at the back of the saddle. Set each one down gently and do not touch the mule."

The spring-loaded bucking-mule game stands on the desk, set and locked, its skill switch on the most sensitive
setting. A saddle rests on the mule's blanket with a horn post at the front and two hanger posts at the back. Three
of the game's pieces lie on the desk: the coiled rope and the canteen on the robot's right, the frying pan on its
left. The robot hangs all three on the saddle, one at a time, so gently that the mule does not buck.

## Scoring

Score 0 if the mule bucked at any time (the catch let go and it kicked), or it was knocked over (tilted over 20 degrees) or shoved more than 30 mm. Otherwise by the pieces hanging on the saddle at the end: a piece counts when a hanger post passes through its hole or loop, it lies on the seat within 30 degrees of flat, no part of a gripper is within 30 mm of it, and the saddle is still on the mule's back. Any hanger counts. Three pieces 1.0; two 0.6; one 0.3; none 0 (includes doing nothing). Success: 1.0 held 1 s more. In the real trial: the video (did the mule kick at any point) and one photo at the end (which pieces are on the posts).

| Outcome | Score |
|---|---|
| Mule never bucked; rope, canteen and pan all on hanger posts, hands away | 1.0 |
| Mule never bucked; two pieces on posts | 0.6 |
| Mule never bucked; one piece on a post | 0.3 |
| Nothing on the posts (pieces on the desk, in the hand, beside a post, or nothing done) | 0 |
| The mule bucked at any time, or it was knocked over or shoved more than 30 mm | 0 |

A buck scores 0 even if pieces end up back on the posts, as in the game ("If he bucks while you're placing the piece
... you're out of the game"). The start scores 0: all three pieces are on the desk.

**Real trial, read from the video and one photo.** A buck is unmistakable on video: the hindquarters kick up and the
saddle and gear fly. The end photo shows which posts have a piece threaded on them and that the hands are away.
Mule movement: a tape outline of its base and a ruler. Reset is about 30 s (the maker's own reset: push the bottom
down until the back legs touch the base, push the tail down to lock).

## The product and its numbers

- The classic bucking-mule game (Hasbro, item 48380), sold now for about US$15-25 (for example
  https://www.walmart.com/ip/Open-Box-Hasbro-Buckaroo-Classic-Spring-Loaded-Mule-Board-Game-Multicolor-10-5/17688664238,
  https://www.amazon.co.uk/Hasbro-HAS48380-Buckaroo/dp/B00VQ6UAQ6). Mule size from retailer listings: 22 cm long,
  8.5 cm wide, 23 cm tall (https://www.drtoymaster.co.uk/products/buckaroo, https://grennansonline.ie/products/buckaroo).
- The maker's instructions (https://www.hasbro.com/common/instruct/buckaroo.pdf). Contents: "Plastic mule with base;
  Plastic saddle, blanket, shovel, canteen, pan, bedroll, lantern, hat, holster, rope, dynamite, crate, guitar".
  Setting: "Slide the side switch ... to the skill level you want: 1, 2 or 3. The higher the skill level, the more
  sensitive [the mule] will be"; "Push [his] bottom down until his back legs touch the base. Then push his tail down
  to lock him into place." A turn: "Carefully hang or place one piece of gear (your choice) on [the mule]. Place the
  piece on the saddle— either on one of the hooks, or on another piece of gear that's already there. As you place
  the piece, don't touch [him]."
- How it works (the original patent, US 3,630,520, "Game with bucking animal",
  https://patents.google.com/patent/US3630520): the saddle carries upright hangers; the body's weight and load are
  taken by a spring-biased catch, and "as weights accumulate" it slides until it lets go; the body drops and frees a
  spring that drives a kicker, which strikes the saddle first, so the saddle and its load are thrown; the hind legs
  "do not provide any support for the body whatsoever". The sensitivity setting moves the catch's starting point.
  A published review (https://www.geekyhobbies.com/buckaroo-board-game-review-and-rules/) notes that on the hardest
  setting "the mule bucked once but it was after most of the items were already placed" and that players must "lay
  it down softly as to not make the mule buck".

Estimates (no published figure; how to check in a minute): the trip load at the hardest setting, about 60 g of static
load on the saddle (from the review above: most of the twelve pieces), checked by adding coins on the saddle until it
bucks; the catch's give (2.9 mm at the trip) and its damping; piece sizes (rope 36 mm ring with a 20 mm hole, canteen
34 mm with a 16 mm strap loop, pan 36 mm with a 14 mm handle hole: ruler) and masses (rope 8 g, canteen 9 g, pan 6 g;
whole saddle 25 g, hindquarters 50 g, mule and base 0.30 kg: kitchen scale); body proportions and post size (6 mm,
18 mm tall: ruler). Total cost of real objects: the game, about US$15-25.

## How the simulation works

The mule is three rigid bodies: the front body on its base, the hindquarters with the blanket, and the loose saddle
with its three posts. The hindquarters are held at their set pose by a damped spring hold that is soft vertically
(200 N/m) and stiff sideways and in rotation, preloaded with their own and the saddle's weight; the only way to load
it is contact: pieces on the saddle, the saddle on the blanket, or a hand. When the hindquarters have given 2.9 mm
(60 g of static load, or less load arriving with a jolt) the catch lets go for the rest of the episode: the
hindquarters swing rear-up 35 degrees and the saddle is struck upward, and the pieces are thrown by contact. With the
same law, offline: an 8 g piece set down still gives 0.6 mm; three pieces at rest about 1.1 mm; with two on, a third
dropped from 10 cm reaches the trip; a hand pressing a piece down with 1.5 N or more trips at once.

Scripted demo: each piece is pinched top-down across its body, carried over its post, its hole read and lined up,
lowered slowly over the post, read again just above the seat, and let go about 2 mm above it; the fingers open only a
few millimetres, the hand rises straight up and leaves. The right hand hangs the canteen, the left the pan, the right
the rope.

Demo result: On the GPU (no seeded shift) the demo scored 1.0: all three pieces on their posts and the mule never bucked, success at step 804 (40.2 s). In three runs with seeded shifts of up to 10 mm for the mule and the pieces, each scored 1.0; the hole reads were within 1.2 mm, and the catch's largest give was 1.44-1.49 mm (about 0.3 N) against the 2.9 mm trip; at rest with all three on, 1.13 mm.

Wrong way: each piece pushed 8 mm down into the saddle before letting go (a hand that keeps pressing): the first piece, the canteen, tripped the catch at 3.0 mm of give (0.6 N); the hindquarters kicked and the saddle and canteen were thrown off: 0. The scripted hand then carried on with its other pieces and pushed the bucked mule over: still 0.

Scores: Dream-1 run: 0.0; demo: 1.0

## Fidelity and shortcuts

- The catch is a force law standing in for the real latch, pawl and spring (stated above, with estimated numbers);
  the robot still has to do the real grasp, the real placement over a 6 mm post and the real gentle release.
- Rigid bodies only; pieces are simplified flat shapes in the game's colours (no lettering); rings are built from
  small blocks, so their holes are slightly polygonal.
- Seeded start shifts move the whole mule and the pieces; the saddle and hindquarters are re-seated on the mule
  before the catch is armed.
