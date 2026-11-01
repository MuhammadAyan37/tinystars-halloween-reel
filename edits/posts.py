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
from render import (F_BODY, F_HORROR, GOLD, INK, OPENGEN, ORANGE, SLIME, CREAM, Renderer, blit,  # noqa: E402
                    drippy_pill, fit_width, font, label_sprite, pumpkin_badge, stack, text_image, to_sprite)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "exports" / "post"
PW, PH = 1080, 1350
HEROES = ["IMG_8302", "IMG_8187", "IMG_8030", "IMG_8309", "IMG_8386", "IMG_8202", "IMG_8335", "IMG_8274"]


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
    img = r.photo(shot).crop(PW, PH, 0.82)
    img = vignette(top_band(img), 0.55)
    blit(img, r.web, PW - 165, 165, 1.0, 0, 0.75, pivot=(330, 0))
    lab = to_sprite(label_sprite(idx + 1, shot["label"]))
    blit(img, lab, PW / 2 + 10, 250, min(1.0, 1000 / lab.shape[1]), -1.5)
    blit(img, r.logo_small, 110, 92, 0.85)
    return img


def cover_slide(r):
    img = vignette(still_bg("intro_night_scene.jpg", 0.35), 0.5)
    blit(img, r.logo_mid, PW / 2, 220, 0.95)
    blit(img, r.t_big, PW / 2, 560, 0.95, -2)
    blit(img, r.t_sub, PW / 2, 735, 1.0, 1.5)
    blit(img, r.t_cta, PW / 2, 880)
    swipe = to_sprite(text_image("SWIPE TO MEET OUR LITTLE MONSTERS  >>", font(F_HORROR, 40), fill=(GOLD, ORANGE),
                                 stroke=INK, stroke_w=4, shadow=0.8, pad=10))
    blit(img, swipe, PW / 2, 1240)
    blit(img, r.bats[2], 880, 380, 0.7, 12)
    blit(img, r.bats[5], 190, 980, 0.5, -10)
    return img


def end_slide(r):
    img = vignette(still_bg("outro_haunted_bg.jpg", 0.12), 0.5)
    blit(img, r.o_logo, PW / 2, 190, 0.78)
    blit(img, r.o_l1, PW / 2, 420, 0.9)
    blit(img, r.o_l2, PW / 2, 545, 0.9)
    blit(img, r.o_tag, PW / 2, 650)
    for spr, y in zip(r.o_pills, (790, 935, 1078)):
        blit(img, spr, PW / 2, y)
    blit(img, r.o_stay, PW / 2, 1215)
    blit(img, r.bats[3], 900, 330, 0.55, 10)
    return img


def collage(r):
    img = vignette(still_bg("intro_night_scene.jpg", 0.72), 0.4)
    head = to_sprite(fit_width(text_image("HALLOWEEN COSTUME PARADE", font(F_HORROR, 60), fill=(GOLD, ORANGE),
                                          stroke=INK, stroke_w=5, shadow=0.7, pad=10, glow=((255, 90, 20), 12, 1.0)),
                               880))
    blit(img, head, PW / 2 + 70, 72)
    blit(img, r.logo_small, 90, 74, 0.62)
    cols, gap, top = 5, 8, 136
    tw = (PW - gap * (cols + 1)) // cols
    th = (PH - top - 92 - gap * 5) // 5
    for i, shot in enumerate(SHOTS):
        c, rr = i % cols, i // cols
        x, y = gap + c * (tw + gap), top + rr * (th + gap)
        img[y:y + th, x:x + tw] = r.photo(shot).crop(tw, th, 0.7)
        badge = to_sprite(fit_width(pumpkin_badge(i + 1, 92), 64))
        blit(img, badge, x + 34, y + 34)
    x0 = gap + (len(SHOTS) % cols) * (tw + gap)
    y0 = top + (len(SHOTS) // cols) * (th + gap)
    lines = [text_image(t, font(F_HORROR, 40), fill=(GOLD, ORANGE), stroke=INK, stroke_w=4, shadow=0, pad=2)
             for t in ("COMMENT YOUR", "FAVE NUMBER!")]
    cta = to_sprite(fit_width(drippy_pill(stack(lines, -6), SLIME, pad_x=24, pad_y=10, seed=13), PW - x0 - 20))
    blit(img, cta, (x0 + PW) / 2, y0 + th / 2)
    foot = to_sprite(text_image(f"{BRAND['web']}  •  {BRAND['phone']}  •  Grande Prairie, AB",
                                font(F_BODY, 36, [800]), fill=(CREAM, CREAM), stroke=INK, stroke_w=3, shadow=0, pad=6))
    blit(img, foot, PW / 2, PH - 46)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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
