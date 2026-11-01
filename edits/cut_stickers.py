"""Cut the sticker graphics out of Tiny Stars' latest "Thanks for Watching!" card for the animated outro.

The logo is cut by flood-filling the background up to its closed white outline. The other stickers
(the title cluster and the three contact pills) are cut with GrabCut from a box, plus a few hand seeds
where the background and the sticker share colours (the orange pumpkin icon on the phone pill).

Usage: python -I edits/cut_stickers.py
  source/brand_ref/tinystars_thanks_for_watching.png -> work/stickers/<name>.png (RGBA) + check.jpg
Each PNG covers its box plus PAD px, so it sits back at the same place on a 1080x1920 frame.
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "source" / "brand_ref" / "tinystars_thanks_for_watching.png"
OUT = ROOT / "work" / "stickers"
PAD = 40

# name: (box x0, y0, x1, y1 on the card, method, seeds [(kind, shape, params)])
STICKERS = {
    "logo": ((330, 50, 745, 450), "flood", []),
    "title": ((95, 462, 985, 1160), "grabcut", []),
    "pill_phone": ((268, 1160, 826, 1318), "grabcut",
                   [("circle", (357, 1238, 58)), ("rect", (450, 1205, 790, 1270))]),
    "pill_address": ((255, 1322, 858, 1468), "grabcut", []),
    "pill_email": ((268, 1468, 840, 1608), "grabcut", []),
}


def largest_filled(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    keep = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
    ff = keep.copy()
    cv2.floodFill(ff, np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8), (0, 0), 255)
    return keep | cv2.bitwise_not(ff)


def flood_cut(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] > 200) & (hsv[..., 1] < 60)).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    h, w = white.shape
    flood = white.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    edge = [(x, y) for x in range(w) for y in (0, h - 1)] + [(x, y) for y in range(h) for x in (0, w - 1)]
    for x, y in edge:
        if flood[y, x] == 0:
            cv2.floodFill(flood, mask, (x, y), 128)
    return largest_filled((flood != 128).astype(np.uint8) * 255)


def grabcut_cut(roi, box, seeds, origin):
    x0, y0, x1, y1 = box
    ox, oy = origin
    m = np.full(roi.shape[:2], cv2.GC_BGD, np.uint8)
    m[y0 - oy:y1 - oy, x0 - ox:x1 - ox] = cv2.GC_PR_FGD
    for kind, p in seeds:
        if kind == "circle":
            cv2.circle(m, (p[0] - ox, p[1] - oy), p[2], cv2.GC_FGD, -1)
        else:
            cv2.rectangle(m, (p[0] - ox, p[1] - oy), (p[2] - ox, p[3] - oy), cv2.GC_FGD, -1)
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(roi, m, None, bgd, fgd, 8, cv2.GC_INIT_WITH_MASK)
    return largest_filled(np.where((m == 1) | (m == 3), 255, 0).astype(np.uint8))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(SRC))
    tiles, placement = [], {}
    for name, (box, method, seeds) in STICKERS.items():
        x0, y0, x1, y1 = box
        X0, Y0, X1, Y1 = x0 - PAD, y0 - PAD, x1 + PAD, y1 + PAD
        roi = img[Y0:Y1, X0:X1]
        mask = flood_cut(roi) if method == "flood" else grabcut_cut(roi, box, seeds, (X0, Y0))
        rgba = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
        rgba[..., 3] = cv2.GaussianBlur(mask, (0, 0), 0.8)
        cv2.imwrite(str(OUT / f"{name}.png"), rgba)
        placement[name] = [(X0 + X1) / 2, (Y0 + Y1) / 2]
        a = rgba[..., 3:4].astype(np.float32) / 255
        tiles.append((roi * a + np.array([180, 0, 180]) * (1 - a)).astype(np.uint8))
    (OUT / "placement.json").write_text(json.dumps(placement, indent=1))
    W = max(t.shape[1] for t in tiles)
    sheet = np.vstack([cv2.copyMakeBorder(t, 4, 4, 0, W - t.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))
                       for t in tiles])
    cv2.imwrite(str(OUT / "check.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("stickers:", ", ".join(STICKERS))


if __name__ == "__main__":
    main()
