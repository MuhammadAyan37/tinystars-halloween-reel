"""Still posts to go with the reel (1080x1350, the 4:5 feed size for Instagram and Facebook).

  exports/post/carousel_01_cover.jpg ... carousel_10_end.jpg  cover, 8 favourite costumes, sign-off
  exports/post/all_costumes_collage.jpg                       every costume, numbered as in the reel
  exports/reel_cover.jpg                                      1080x1920 cover frame for the reel

Numbers match the reel, so "comment the number" works across the reel and the posts.
Usage: python -I edits/posts.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content import BRAND, SHOTS  # noqa: E402
from render import (F_GOTH, F_SANS, NEON, OPENGEN, PEACH, WHITE, CREAM, Renderer, blit, costume_label,  # noqa: E402
                    fit_width, flanked, font, lettering, photo_tile, place_label, to_sprite)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "exports" / "post"
PW, PH = 1080, 1350
HEROES = ["IMG_8302", "IMG_8187", "IMG_8030", "IMG_8386", "IMG_8113", "IMG_8309", "IMG_8335", "IMG_8274"]


def still_bg(name, darken=0.0):
    """An OpenGen still, upscaled to 1080 wide and cropped to 4:5 from the middle."""
    im = cv2.cvtColor(cv2.imread(str(OPENGEN / name)), cv2.COLOR_BGR2RGB)
    h = round(im.shape[0] * PW / im.shape[1])
    im = cv2.resize(im, (PW, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    y0 = (h - PH) // 2
    im = im[y0:y0 + PH]
    return im * (1 - darken)


def vignette(img, k=0.5):
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((x - w / 2) / (w * 0.62)) ** 2 + ((y - h / 2) / (h * 0.62)) ** 2)
    a = np.clip(r - 0.5, 0, 1)[..., None] * k
    return img * (1 - a) + np.array([0.07, 0.02, 0.11], np.float32) * a


def top_band(img, k=0.6, hgt=460):
    y = np.arange(img.shape[0], dtype=np.float32)[:, None, None]
    a = k * (1 - np.clip(y / hgt, 0, 1)) ** 1.6
    return img * (1 - a) + np.array([0.07, 0.02, 0.11], np.float32) * a


def save(img, path):
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, quality=93, subsampling=0)


def hero_slide(r, idx):
    shot = SHOTS[idx]
    img = vignette(top_band(r.photo(shot).crop(PW, PH, 0.82)), 0.55)
    blit(img, r.web, PW - 165, 165, 1.0, 0, 0.5, pivot=(330, 0))
    place_label(img, costume_label(idx + 1, shot["label"]), PW / 2, 250)
    blit(img, r.logo_small, 100, 96, 0.85)
    return img


def cover_slide(r):
    img = vignette(still_bg("intro_night_scene.jpg", 0.4), 0.5)
    blit(img, r.halo, PW / 2, 230, 1.0, 0, 0.8)
    blit(img, r.logo_mid, PW / 2, 230, 0.95)
    blit(img, r.t_big, PW / 2, 560, 0.95)
    blit(img, r.t_sub, PW / 2, 735)
    glass, txt = r.t_cta
    glass.draw(img, PW / 2, 850)
    blit(img, txt, PW / 2, 850)
    swipe = to_sprite(flanked(lettering("SWIPE TO MEET OUR LITTLE MONSTERS", font(F_SANS, 30, 700), 8,
                                        fill=(PEACH, PEACH), shadow=0.8), 60, 22))
    blit(img, swipe, PW / 2, 1235)
    blit(img, r.bats[2], 880, 380, 0.7, 12)
    blit(img, r.bats[5], 190, 980, 0.5, -10)
    return img


def end_slide(r):
    img = vignette(still_bg("outro_haunted_bg.jpg", 0.2), 0.5)
    blit(img, r.halo, PW / 2, 175, 1.0, 0, 0.9)
    blit(img, r.o_logo, PW / 2, 175, 0.72)
    blit(img, r.o_l1, PW / 2, 375, 0.9)
    blit(img, r.o_l2, PW / 2, 490, 0.85)
    blit(img, r.o_tag, PW / 2, 605)
    y = 670
    for glass, icon, text in r.o_cards:
        x0, y0 = glass.draw(img, PW / 2, y + glass.h / 2)
        blit(img, icon, x0 + 72, y0 + glass.h / 2)
        blit(img, text, x0 + 150 + text.shape[1] / 2, y0 + glass.h / 2 + 1)
        y += glass.h + 20
    blit(img, r.o_stay, PW / 2, y + 28)
    blit(img, r.o_web, PW / 2, y + 82)
    blit(img, r.bats[3], 900, 330, 0.55, 10)
    return img


def collage(r):
    img = vignette(still_bg("intro_night_scene.jpg", 0.7), 0.4)
    head = to_sprite(fit_width(lettering("Halloween Costume Parade", font(F_GOTH, 92, 800), fill=(CREAM, PEACH),
                                         glow=(NEON, 16, 1.0), shadow=0.8), 960))
    blit(img, head, PW / 2, 92)
    sub = to_sprite(flanked(lettering("COMMENT YOUR FAVOURITE NUMBER", font(F_SANS, 28, 700), 9,
                                      fill=(WHITE, WHITE), shadow=0.8), 60, 22))
    blit(img, sub, PW / 2, 188)
    cols, margin, gap, top = 4, 40, 16, 236
    tw = (PW - 2 * margin - gap * (cols - 1)) // cols
    th = int(tw * 4 / 3)
    for i, shot in enumerate(SHOTS):
        c, rr = i % cols, i // cols
        x, y = margin + c * (tw + gap) + tw / 2, top + rr * (th + gap) + th / 2
        blit(img, to_sprite(photo_tile(r.photo(shot).crop(tw, th, 0.72), i + 1)), x, y)
    foot = to_sprite(lettering(f"{BRAND['web']}   ·   {BRAND['phone']}   ·   Grande Prairie, AB",
                               font(F_SANS, 28, 600), 2, fill=(CREAM, CREAM), shadow=0.8))
    blit(img, foot, PW / 2, PH - 40)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("carousel_*.jpg"):
        old.unlink()
    r = Renderer()
    save(cover_slide(r), OUT / "carousel_01_cover.jpg")
    for k, name in enumerate(HEROES):
        idx = next(i for i, s in enumerate(SHOTS) if s["photo"] == name)
        save(hero_slide(r, idx), OUT / f"carousel_{k + 2:02d}_{SHOTS[idx]['label'].lower().replace(' ', '_')}.jpg")
    save(end_slide(r), OUT / "carousel_10_end.jpg")
    save(collage(r), OUT / "all_costumes_collage.jpg")
    # reel cover: the finished title card
    title = r.seg("title")
    save(r.frame(title["t1"] - 0.35), ROOT / "exports" / "reel_cover.jpg")
    print("posts written to", OUT)


if __name__ == "__main__":
    main()
