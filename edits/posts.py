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
from content import BRAND, END, SHOTS, TITLE  # noqa: E402
from render import (CREAM, DEEP, GOLD, INK, MAGENTA, ORANGE, PURPLE, VIOLET, WHITE, Photo, Renderer,  # noqa: E402
                    blit, fit_width, font, label_sprite, logo_image, pill_image, text_image, to_sprite)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "exports" / "post"
PW, PH = 1080, 1350
HEROES = ["IMG_8302", "IMG_8187", "IMG_8030", "IMG_8309", "IMG_8386", "IMG_8202", "IMG_8335", "IMG_8274"]
DEEPF = np.array(DEEP, np.float32) / 255


def crop(p, w, h, zoom=0.82):
    """Crop a framed photo to w:h around the kid(s) and resize to w x h (float RGB 0..1)."""
    ch = min(p.win_h * zoom, p.hs)
    cw = ch * w / h
    if cw > p.ws:
        cw, ch = p.ws, p.ws * h / w
    cx = min(max(p.cx, cw / 2), p.ws - cw / 2)
    cy = min(max(p.cy, ch / 2), p.hs - ch / 2)
    x0, y0 = int(cx - cw / 2), int(cy - ch / 2)
    im = p.img[y0:y0 + int(ch), x0:x0 + int(cw)]
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def darken(img, top=0.55, top_h=480, bottom=0.0, bottom_h=300):
    h = img.shape[0]
    y = np.arange(h, dtype=np.float32)[:, None, None]
    a = top * (1 - np.clip(y / top_h, 0, 1)) ** 1.6
    a = 1 - (1 - a) * (1 - bottom * np.clip((y - (h - bottom_h)) / bottom_h, 0, 1) ** 1.4)
    return img * (1 - a) + DEEPF * a


def save(img, path):
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, quality=93, subsampling=0)


def numbered(i):
    return i + 1, SHOTS[i]


def hero_slide(r, idx, photos):
    num, shot = numbered(idx)
    img = darken(crop(photos[shot["photo"]], PW, PH), 0.6, 520, 0.35, 260)
    blit(img, to_sprite(label_sprite(num, shot["label"])), PW / 2 + 10, 250, min(1.0, 990 / 1100), -2)
    blit(img, r.logo_small, 110, 92, 0.8)
    return img


def wall_bg(r, w, h, bright=0.34):
    wh, ww = r.wall.shape[:2]
    y0, x0 = int(wh * 0.25), int((ww - w) / 2)
    return r.wall[y0:y0 + h, x0:x0 + w] * bright + DEEPF * (1 - bright)


def cover_slide(r):
    img = wall_bg(r, PW, PH, 0.42)
    blit(img, r.title_logo, PW / 2, 270, 1.05)
    blit(img, r.t_big, PW / 2, 600, 1.0, -2)
    blit(img, r.t_sub, PW / 2, 800, 1.0, 1.5)
    blit(img, r.t_cta, PW / 2, 950, 1.15)
    swipe = to_sprite(text_image("SWIPE TO MEET OUR LITTLE MONSTERS  >>", font("LuckiestGuy-Regular.ttf", 44),
                                 fill=(GOLD, (255, 214, 90)), stroke_w=5, extrude=4, extrude_col=INK, pad=8))
    blit(img, swipe, PW / 2, 1200, 1.0)
    for x, y, k, s in [(120, 470, 0, 1.0), (960, 520, 2, 0.8), (150, 1080, 4, 0.7), (930, 1090, 1, 0.9),
                       (900, 180, 3, 1.0)]:
        blit(img, r.sparkles[k], x, y, s)
    blit(img, r.bats[2], 880, 330, 0.75, 12)
    blit(img, r.bats[5], 200, 760, 0.55, -10)
    return img


def end_slide(r):
    img = wall_bg(r, PW, PH, 0.30)
    blit(img, r.logo_big, PW / 2, 230, 0.95)
    blit(img, r.e_happy, PW / 2, 470)
    blit(img, r.e_hall, PW / 2, 610)
    blit(img, r.e_from, PW / 2, 755)
    blit(img, r.e_cta, PW / 2, 900)
    blit(img, r.e_enrol, PW / 2, 1065)
    blit(img, r.e_phone, PW / 2, 1150)
    blit(img, r.e_addr, PW / 2, 1222)
    for x, y, k, s in [(110, 400, 0, 0.9), (980, 360, 2, 0.8), (90, 1000, 1, 0.8), (990, 980, 4, 0.7)]:
        blit(img, r.sparkles[k], x, y, s)
    blit(img, r.bats[3], 900, 120, 0.6, 10)
    return img


def collage(r, photos):
    img = np.ones((PH, PW, 3), np.float32) * DEEPF
    head = to_sprite(fit_width(text_image("HALLOWEEN COSTUME PARADE", font("LuckiestGuy-Regular.ttf", 64),
                                          fill=(GOLD, ORANGE), stroke_w=6, extrude=6, extrude_col=PURPLE,
                                          pad=10), 900))
    blit(img, head, PW / 2 + 60, 72)
    blit(img, r.logo_small, 92, 76, 0.62)
    cols, rows, gap, top = 5, 5, 8, 136
    tw = (PW - gap * (cols + 1)) // cols
    th = (PH - top - 92 - gap * rows) // rows
    badge_font = font("LuckiestGuy-Regular.ttf", 34)
    for i, shot in enumerate(SHOTS):
        c, rr = i % cols, i // cols
        x, y = gap + c * (tw + gap), top + rr * (th + gap)
        img[y:y + th, x:x + tw] = crop(photos[shot["photo"]], tw, th, 0.7)
        num = to_sprite(pill_image(text_image(str(i + 1), badge_font, fill=(INK, INK), stroke_w=0, shadow=0, pad=0),
                                   [GOLD, (255, 214, 90)], pad_x=12, pad_y=4, border=3))
        blit(img, num, x + 34, y + 30, 0.9)
    # last row: the spare cells get the call to action
    x0 = gap + (len(SHOTS) % cols) * (tw + gap)
    y0 = top + (len(SHOTS) // cols) * (th + gap)
    cta = to_sprite(fit_width(pill_image(text_image("COMMENT YOUR FAVE #", font("LuckiestGuy-Regular.ttf", 40),
                                                    fill=(WHITE, CREAM), stroke_w=4, extrude=3, extrude_col=INK,
                                                    shadow=0, pad=4), [VIOLET, MAGENTA], pad_x=24, pad_y=12),
                              PW - x0 - 20))
    blit(img, cta, (x0 + PW) / 2, y0 + th / 2)
    foot = to_sprite(text_image(f"{BRAND['web']}  •  {BRAND['phone']}  •  Grande Prairie, AB",
                                font("Baloo2.ttf", 36, [800]), fill=(CREAM, CREAM), stroke_w=0, shadow=0, pad=6))
    blit(img, foot, PW / 2, PH - 46)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = Renderer()
    photos = {s["photo"]: Photo(s) for s in SHOTS}
    save(cover_slide(r), OUT / "carousel_01_cover.jpg")
    for k, name in enumerate(HEROES):
        idx = next(i for i, s in enumerate(SHOTS) if s["photo"] == name)
        save(hero_slide(r, idx, photos), OUT / f"carousel_{k + 2:02d}_{SHOTS[idx]['label'].lower().replace(' ', '_')}.jpg")
    save(end_slide(r), OUT / "carousel_10_end.jpg")
    save(collage(r, photos), OUT / "all_costumes_collage.jpg")
    # reel cover: the title card, sharp photo behind it
    r.prev_frame = None
    title = next(s for s in r.segs if s["kind"] == "title")
    save(r.frame(title["t1"] - 0.45), ROOT / "exports" / "reel_cover.jpg")
    print("posts written to", OUT)


if __name__ == "__main__":
    main()
