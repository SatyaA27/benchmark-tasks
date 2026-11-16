// Printed parts of the trial colorimeter and its cuvette rack (README.md, "Real instrument: build").
// OpenSCAD source. Set PART, then render (F6) and export STL. Millimetres.
// Housing frame: origin at the housing centre on the table, x forward (away from the robot), y to the robot's
// left, z up from the table. The outlines, well, lip rods, hood, caps and rack are the simulation's sizes; the
// cavities, slits, bores and holes are the build's own.

PART = "upper";  // "lower", "upper", "hood", "cap", "led_holder", "rack"
$fn = 48;

W = [22, -40];   // well centre (world: 190 mm forward, 45 mm right of the midline)
WW = 14.5;       // well width; floor 30 above the table, mouth 65
BEAM_Z = 45;     // light path height: 15 mm above the well floor
LAMPS = [[-27, 31], [-9, 31], [9, 31], [27, 31], [36, 43]];  // 430, 470, 565, 635 nm lamps; CAL lamp
BUTTONS = [-36, 0, 36];                                       // down, up, CAL; all at y 60
CAP_BOSS = 4;    // stem boss under each cap: set so the cap's underside stands 7 mm above the housing top

module box(lo, hi) { translate(lo) cube([hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]); }

// Four R5 rods round a square opening of width w centred at c, axes at height z (the simulation's lead-ins).
module lip(c, w, z) {
    h = w / 2 + 5;
    for (s = [-1, 1]) {
        translate([c[0] + s * h, c[1], z]) rotate([90, 0, 0]) cylinder(r = 5, h = w + 10, center = true);
        translate([c[0], c[1] + s * h, z]) rotate([0, 90, 0]) cylinder(r = 5, h = w + 10, center = true);
    }
}

// Lower tray, z 0 to 27, open at the top; holds the microcontroller, LED driver and wiring.
module housing_lower() {
    difference() {
        box([-55, -80, 0], [55, 80, 27]);
        box([-52, -77, 3], [52, 77, 28]);
        box([51, 30, 8], [56, 42, 16]);  // USB cable slot in the far wall
    }
}

// Upper part, z 27 to 65: well, lip, optics, panel holes and the two rail pillars. Print with supports.
module housing_upper() {
    difference() {
        union() {
            box([-55, -80, 27], [55, 80, 65]);
            lip(W, WW, 65);
            for (x = [-51, 45], y = [W[1] - 35, W[1] + 35]) box([x - 2, y - 4, 64], [x + 2, y + 4, 95]);
        }
        // hollow inside (3 mm walls and top), except the solid optics block round the well
        difference() {
            box([-52, -77, 26], [52, 77, 62]);
            box([-2, -58, 26], [53, -22, 66]);
        }
        // the well
        box([W[0] - WW / 2, W[1] - WW / 2, 30], [W[0] + WW / 2, W[1] + WW / 2, 80]);
        // source side (+x): 2 x 6 mm slit, LED pocket (LED holder and diffuser), wire channel down to the tray
        box([W[0] + WW / 2 - 1, W[1] - 1, BEAM_Z - 3], [34, W[1] + 1, BEAM_Z + 3]);
        box([33.25, W[1] - 7.5, BEAM_Z - 7.5], [50, W[1] + 7.5, BEAM_Z + 7.5]);
        box([40, W[1] - 7.5, 26], [50, W[1] + 7.5, BEAM_Z]);
        // detector side (-x): 2 x 6 mm slit, slot for the light-sensor board, wire channel
        box([11, W[1] - 1, BEAM_Z - 3], [W[0] - WW / 2 + 1, W[1] + 1, BEAM_Z + 3]);
        box([8.8, W[1] - 13.5, BEAM_Z - 10], [11.2, W[1] + 13.5, BEAM_Z + 10]);
        box([1, W[1] - 13.5, 26], [11.2, W[1] + 13.5, BEAM_Z - 9]);
        // key-switch cutouts in a 1.5 mm skin
        for (x = BUTTONS) {
            box([x - 7, 53, 60], [x + 7, 67, 70]);
            box([x - 10, 50, 26], [x + 10, 70, 63.5]);
        }
        // 3 mm lamp holes and the display window (30 x 14 mm face, long side across the panel)
        for (p = LAMPS) translate([p[0], p[1], 60]) cylinder(d = 3.1, h = 10);
        box([-47, 3, 60], [-33, 33, 70]);
        // 4 mm steel rails through the pillars, axes 92 above the table
        for (y = [W[1] - 35, W[1] + 35]) translate([-60, y, 92]) rotate([0, 90, 0]) cylinder(d = 4.1, h = 120);
    }
}

// Hood, drawn closed (centre over the well): roof, stem, grip tab, skirts as simulated; runners bored for the rails.
module hood() {
    translate([W[0], W[1], 0]) difference() {
        union() {
            box([-16, -16, 98], [16, 16, 102]);
            box([-4, -4, 102], [4, 4, 111]);
            box([-12, -9, 111], [12, 9, 131]);
            for (s = [-1, 1]) box([-16, s > 0 ? 16 : -18, 70], [16, s > 0 ? 18 : -16, 98]);
            for (s = [-1, 1]) box([-3, s > 0 ? 16 : -39, 88], [3, s > 0 ? 39 : -16, 96]);
        }
        for (s = [-1, 1]) translate([-10, s * 35, 92]) rotate([0, 90, 0]) cylinder(d = 4.3, h = 20);
    }
}

// Button cap, 16 x 16 x 12, upside down for printing, with a Cherry MX cross socket in its stem boss.
module cap() {
    difference() {
        union() {
            box([-8, -8, 0], [8, 8, 12]);
            translate([0, 0, 12]) cylinder(d = 5.5, h = CAP_BOSS);
        }
        translate([0, 0, 12 + CAP_BOSS - 3.8]) {
            box([-2.1, -0.7, 0], [2.1, 0.7, 4]);
            box([-0.7, -2.1, 0], [0.7, 2.1, 4]);
        }
    }
}

// LED holder: a 2 x 2 cluster (7 mm pitch) for the four 5 mm LEDs, facing -x toward the slit; pushed up the
// wire channel into the LED pocket with a 1 mm white diffuser between it and the slit.
module led_holder() {
    difference() {
        box([0, -7.4, -7.4], [10, 7.4, 7.4]);
        for (y = [-3.5, 3.5], z = [-3.5, 3.5]) translate([-1, y, z]) rotate([0, 90, 0]) cylinder(d = 5.9, h = 12);
    }
}

// Cuvette rack, 44 x 164 x 24 (rack frame: centre at the origin), two 15.5 mm square pockets at y -60 and +60 with
// floors 4 above the table and the same R5 lip.
module rack() {
    difference() {
        union() {
            box([-22, -82, 0], [22, 82, 24]);
            for (y = [-60, 60]) lip([0, y], 15.5, 24);
        }
        for (y = [-60, 60]) box([-7.75, y - 7.75, 4], [7.75, y + 7.75, 40]);
    }
}

if (PART == "lower") housing_lower();
else if (PART == "upper") housing_upper();
else if (PART == "hood") hood();
else if (PART == "cap") cap();
else if (PART == "led_holder") led_holder();
else if (PART == "rack") rack();
