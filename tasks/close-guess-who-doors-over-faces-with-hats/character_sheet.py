"""Print the character sheet for one trial at true size (SVG, 270 x 220 mm), the same drawing as the simulation.

    python character_sheet.py --seed 7 --out sheet-7.svg

The five people with hats are random.Random(seed).sample(range(24), 5), plus 1 (person numbers 1-24, top row
first, each row left to right as the robot sees the sheet). The simulated trial uses seed 2. Print at 100 %, check
the 100 mm scale bar with a ruler, cut to the game's own sheet and slide it in. Small person numbers in each
cell's corner are for the person running the trial (the doors hide them); the simulation does not draw them.
"""
import argparse
import random
from pathlib import Path

import numpy as np

import guesswho_kit as G

WIDTH, HEIGHT = 270, 220


def hats(seed):
    return {n + 1 for n in random.Random(int(seed)).sample(range(24), 5)}


def _hex(rgb):
    return "#" + "".join(f"{round(255 * c):02x}" for c in rgb)


def svg(seed):
    chosen = hats(seed)
    bottom = G.SHEET_MID - .110 * G.UP                    # the sheet's lower edge on its mid-plane
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}mm" height="{HEIGHT}mm" '
           f'viewBox="0 0 {WIDTH} {HEIGHT}">', f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{_hex(G.SHEET)}"/>']
    for row in range(G.ROWS):
        for col in range(G.COLS):
            n = G.person(row, col)
            h = G.hinge(row, col)
            on_sheet = h - np.dot(h - G.SHEET_MID, G.N) * G.N
            b = float(np.dot(on_sheet - bottom, G.UP)) * 1000   # hinge line, mm up from the lower edge
            cx = WIDTH / 2 - h[1] * 1000                        # the robot's left (+y) prints on the left
            for _, outline, color in G.portrait_shapes(n, n in chosen):
                pts = " ".join(f"{cx + x:.2f},{HEIGHT - (b + y):.2f}" for x, y in outline)
                out.append(f'<polygon points="{pts}" fill="{_hex(color)}"/>')
            out.append(f'<text x="{cx - 13.5:.2f}" y="{HEIGHT - b - 1:.2f}" font-family="sans-serif" '
                       f'font-size="3" fill="black">{n}</text>')
    out.append('<path d="M10 8 H110 M10 5 V11 M110 5 V11" fill="none" stroke="black" stroke-width=".3"/>')
    out.append(f'<text x="114" y="9" font-family="sans-serif" font-size="3">100 mm; layout seed {seed}; '
               f'hats: {", ".join(map(str, sorted(chosen)))}</text></svg>')
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("character-sheet-seed2.svg"))
    a = ap.parse_args()
    a.out.write_text(svg(a.seed))
    print(a.out, "hats", sorted(hats(a.seed)))
