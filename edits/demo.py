"""A photo-free picture of the reel's graphics for the README (no children's photos in the repo).

Usage: python -I edits/demo.py -> docs/demo.png
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content import SHOTS, TITLE  # noqa: E402
from render import (CREAM, DEEP, F_GOTH, F_SANS, NEON, PEACH, PURPLE, WHITE, Glass, bat_image, blit,  # noqa: E402
                    bolt_image, cobweb_image, costume_label, fit_width, flanked, font, halo_image, lettering,
                    logo_image, periodic_noise, place_label, to_sprite)

ROOT = Path(__file__).resolve().parent.parent
DW, DH = 2000, 1000  # drawn large, saved at 1600x800


def main():
    y = np.linspace(0, 1, DH)[:, None, None]
    img = (np.array(DEEP) * (1 - y) + np.array(PURPLE) * 0.5 * y) / 255.0
    img = np.repeat(img, DW, axis=1).astype(np.float32)
    fog = np.array(Image.fromarray((periodic_noise(DH // 4, DW // 4, 3) * 255).astype(np.uint8)).resize((DW, DH)))
    a = (np.clip(fog / 255.0 - 0.35, 0, 1) * 0.7 * np.clip((y[..., 0] - 0.45) / 0.55, 0, 1))[..., None]
    img = img * (1 - a) + np.array([0.6, 0.55, 0.75], np.float32) * a
    blit(img, to_sprite(bolt_image(1, 600)), 960, 360, 0.8, -8, 0.7)
    blit(img, to_sprite(cobweb_image()), DW - 165, 165, 1.0, 0, 0.6)
    blit(img, to_sprite(halo_image(520, 420)), 470, 175, 1.0, 0, 0.8)
    blit(img, to_sprite(logo_image(240)), 470, 175)
    blit(img, to_sprite(fit_width(lettering(TITLE["big"], font(F_GOTH, 200, 900), fill=(CREAM, PEACH),
                                            glow=(NEON, 22, 1.1), shadow=0.8), 820)), 470, 450)
    blit(img, to_sprite(flanked(lettering(TITLE["sub"].upper(), font(F_SANS, 38, 600), 18, fill=(WHITE, WHITE),
                                          shadow=0.8), 80, 24)), 470, 620)
    cta = lettering(TITLE["cta"].upper(), font(F_SANS, 26, 700), 8, fill=(PEACH, PEACH), shadow=0)
    Glass(cta.width + 70, 66, 33).draw(img, 470, 730)
    blit(img, to_sprite(cta), 470, 730)
    for k, i in enumerate((0, 3, 5, 11)):
        place_label(img, costume_label(i + 1, SHOTS[i]["label"]), 1490, 215 + k * 190)
    blit(img, to_sprite(bat_image(0.6)), 1880, 950, 0.6, -10)
    blit(img, to_sprite(bat_image(-0.4)), 130, 900, 0.5, 12)
    (ROOT / "docs").mkdir(exist_ok=True)
    out = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((1600, 800), Image.LANCZOS)
    out.save(ROOT / "docs" / "demo.png", optimize=True)
    print("wrote docs/demo.png")


if __name__ == "__main__":
    main()
