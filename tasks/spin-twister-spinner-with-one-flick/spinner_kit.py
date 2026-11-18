"""A family-game card spinner: a loose card with a plastic arrow on a free rivet (reusable module).

Models the spinner of a current Hasbro Twister game (item 98831): a round white card, 250 mm across and 2 mm thick,
90 g, lying loose on the table, and a black plastic arrow, 185 x 14 x 4 mm and 8 g, turning on a rivet at the card's
centre. The rivet sits 80 mm from the arrow's short end and 105 mm from its long (pointing) end; the arrow's underside
is 4 mm above the card. All sizes and masses are estimates from product photos (a ruler and a kitchen scale check
them in a minute).

The rivet is a passive hinge with friction only: no drive, spring or injected speed. Its friction torque is fixed at
the arrow's own inertia about the rivet times 4.5 rad/s^2, the slowing fitted to a person's flick in the maker's
retail video (1.88 turns in 2.4 s from about 10.3 rad/s; an estimate from video, not a bench measurement). Static
friction is 10 % above the sliding value. Tuned numbers: INERTIA 2.42e-5 kg m^2, BEARING_TORQUE 1.09e-4 N m; in the
simulator the free arrow slows at about 4.7 rad/s^2 (the bearing plus the simulator's own small damping). Do not lower
the friction to make a weak strike enough.

spinner(pos, angle) returns the Mechanism: `pos` is the card's centre, 1 mm above the table top (the card's middle
plane); `angle` (degrees) is the arrow's long end, measured from +x toward +y. Metres, kilograms.
"""
from yam_parts import box, cyl
from yam_joints import Mechanism, Hinge, Link

ALPHA = 4.5  # rad/s^2, the free arrow's slowing (retail-video estimate)
BOARD_RADIUS = .125
BOARD_THICK = .002
BOARD_MASS = .090
POINTER_MASS = .008
POINTER_LENGTH = .185
POINTER_WIDTH = .014
POINTER_THICK = .004
POINTER_OFFSET = .0125  # the arrow's middle lies this far from the rivet, toward the long end
SHORT_ARM = POINTER_LENGTH / 2 - POINTER_OFFSET  # 80 mm, rivet to the short end
LONG_ARM = POINTER_LENGTH / 2 + POINTER_OFFSET  # 105 mm, rivet to the pointing end
POINTER_Z = .007  # the arrow's middle plane above the card's middle plane
INERTIA = POINTER_MASS * ((POINTER_LENGTH**2 + POINTER_WIDTH**2) / 12 + POINTER_OFFSET**2)
BEARING_TORQUE = INERTIA * ALPHA


def spinner(pos=(.16, 0, .7485), angle=90):
    return Mechanism('spinner', pos,
        [cyl((0, 0, 0), BOARD_RADIUS, BOARD_THICK, color=(.91, .90, .82)),
         cyl((0, 0, .003), .004, .006, color=(.08, .08, .09))],
        [Link('arrow', [box((POINTER_OFFSET, 0, POINTER_Z),
                            (POINTER_LENGTH, POINTER_WIDTH, POINTER_THICK), color=(.04, .04, .05))],
              Hinge('z', (0, 0, POINTER_Z), -100000, 100000, initial=angle,
                    static_friction=BEARING_TORQUE * 1.1, dynamic_friction=BEARING_TORQUE),
              POINTER_MASS)], mass=BOARD_MASS, friction=(.45, .35), contact_offset=.0005)
