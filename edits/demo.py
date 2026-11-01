"""A photo-free picture of the reel's graphics for the README (no children's photos in the repo).

Usage: python -I edits/demo.py -> docs/demo.png
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content import SHOTS, TITLE  # noqa: E402
from render import (BLOOD, DEEP, F_DRIP, F_HORROR, GOLD, INK, ORANGE, PURPLE, SLIME, WHITE, bat_image, blit,  # noqa: E402
                    bolt_image, cobweb_image, drippy_pill, fit_width, font, label_sprite, logo_image, periodic_noise,
                    text_image, to_sprite)

ROOT = Path(__file__).resolve().parent.parent
DW, DH = 1600, 900


def main():
    y = np.linspace(0, 1, DH)[:, None, None]
    img = (np.array(DEEP) * (1 - y) + np.array(PURPLE) * 0.5 * y) / 255.0
    img = np.repeat(img, DW, axis=1).astype(np.float32)
    fog = np.array(Image.fromarray((periodic_noise(DH // 4, DW // 4, 3) * 255).astype(np.uint8)).resize((DW, DH)))
    a = (np.clip(fog / 255.0 - 0.35, 0, 1) * 0.7 * np.clip((y[..., 0] - 0.45) / 0.55, 0, 1))[..., None]
    img = img * (1 - a) + np.array([0.6, 0.55, 0.75], np.float32) * a
    blit(img, to_sprite(bolt_image(1, 600)), 760, 330, 0.8, -8, 0.8)
    blit(img, to_sprite(cobweb_image()), DW - 165, 165, 1.0, 0, 0.8)
    blit(img, to_sprite(logo_image(230)), 400, 150)
    blit(img, to_sprite(fit_width(text_image(TITLE["big"].upper(), font(F_DRIP, 112), fill=(GOLD, BLOOD), stroke=INK,
                                             stroke_w=7, extrude=10, extrude_col=(60, 10, 70),
                                             glow=((255, 90, 20), 18, 1.2)), 780)), 400, 400, 1, -2)
    blit(img, to_sprite(text_image(TITLE["sub"].upper(), font(F_HORROR, 78), fill=(WHITE, (220, 210, 235)), stroke=INK,
                                   stroke_w=6, extrude=7, extrude_col=PURPLE, glow=((130, 70, 255), 14, 1.1))),
         400, 560, 1, 1.5)
    cta = text_image(TITLE["cta"].upper(), font(F_HORROR, 42), fill=(GOLD, ORANGE), stroke=INK, stroke_w=4, shadow=0, pad=4)
    blit(img, to_sprite(drippy_pill(cta, SLIME, pad_x=30, pad_y=8, seed=5)), 400, 700)
    for k, i in enumerate((0, 3, 6, 21)):
        blit(img, to_sprite(label_sprite(i + 1, SHOTS[i]["label"])), 1170, 150 + k * 165, 0.85, -2 + k % 2 * 2.5)
    blit(img, to_sprite(bat_image(0.6)), 1450, 820, 0.6, -10)
    blit(img, to_sprite(bat_image(-0.4)), 120, 800, 0.5, 12)
    (ROOT / "docs").mkdir(exist_ok=True)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(ROOT / "docs" / "demo.png", optimize=True)
    print("wrote docs/demo.png")


if __name__ == "__main__":
    main()
