"""A photo-free picture of the reel's graphics for the README (no children's photos in the repo).

Usage: python -I edits/demo.py -> docs/demo.png
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content import END, SHOTS, TITLE  # noqa: E402
from render import (CREAM, DEEP, GOLD, INK, MAGENTA, ORANGE, PURPLE, VIOLET, WHITE, bat_image, blit,  # noqa: E402
                    fit_width, font, label_sprite, logo_image, pill_image, sparkle_image, text_image, to_sprite)

ROOT = Path(__file__).resolve().parent.parent
DW, DH = 1600, 900


def main():
    y = np.linspace(0, 1, DH)[:, None, None]
    img = (np.array(DEEP) * (1 - y) + np.array(PURPLE) * 0.55 * y) / 255.0
    img = np.repeat(img, DW, axis=1).astype(np.float32)
    rng = np.random.default_rng(3)
    sparks = [to_sprite(sparkle_image(r, c)) for r, c in [(22, GOLD), (16, CREAM), (18, (255, 150, 205))]]
    for _ in range(26):
        blit(img, sparks[rng.integers(0, 3)], rng.uniform(20, DW - 20), rng.uniform(20, DH - 20), rng.uniform(0.4, 1))
    blit(img, to_sprite(logo_image(230)), 400, 170)
    blit(img, to_sprite(fit_width(text_image(TITLE["big"].upper(), font("Creepster-Regular.ttf", 170),
                                             fill=(GOLD, ORANGE), stroke_w=8, extrude=12, extrude_col=PURPLE), 760)),
         400, 400, 1, -2)
    blit(img, to_sprite(text_image(TITLE["sub"].upper(), font("LuckiestGuy-Regular.ttf", 84), fill=(WHITE, CREAM),
                                   stroke_w=7, extrude=9, extrude_col=MAGENTA)), 400, 560, 1, 1.5)
    cta = text_image(TITLE["cta"], font("Baloo2.ttf", 46, [800]), fill=(WHITE, WHITE), stroke_w=0, shadow=0, pad=2)
    blit(img, to_sprite(pill_image(cta, [VIOLET, MAGENTA], pad_x=34, pad_y=6)), 400, 690)
    for k, i in enumerate((0, 3, 6, 21)):
        blit(img, to_sprite(label_sprite(i + 1, SHOTS[i]["label"])), 1180, 150 + k * 150, 0.82, -2 + k % 2 * 2.5)
    end = text_image(END["cta"].upper() + "!", font("LuckiestGuy-Regular.ttf", 48), fill=(GOLD, (255, 214, 90)),
                     stroke_w=5, extrude=4, extrude_col=INK)
    blit(img, to_sprite(end), 1180, 790)
    blit(img, to_sprite(bat_image(0.6)), 760, 120, 0.7, -10)
    blit(img, to_sprite(bat_image(-0.4)), 120, 800, 0.5, 12)
    (ROOT / "docs").mkdir(exist_ok=True)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(ROOT / "docs" / "demo.png", optimize=True)
    print("wrote docs/demo.png")


if __name__ == "__main__":
    main()
