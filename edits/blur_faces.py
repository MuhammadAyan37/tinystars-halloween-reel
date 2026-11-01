"""Blur every face listed in work/faces.json so no child or parent can be recognised.

Each face gets a padded, feather-edged ellipse that is pixelated first and then heavily
Gaussian-blurred (pixelating first stops the blur from keeping a recognisable outline).
The ellipse is sized from the face box, so it also covers the hairline and chin.

Faces are found on the 2400 px working copies; the blur is applied to the full-resolution
original (boxes scaled up), so the reel is cut from full-quality pixels.

Usage: python -I edits/detect_faces.py && python -I edits/blur_faces.py
  -> work/blurred/IMG_xxxx.jpg (full resolution, upright)
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "source" / "drive_raw"
JPG = ROOT / "work" / "jpg"
OUT = ROOT / "work" / "blurred"
FACES = ROOT / "work" / "faces.json"
PAD_W, PAD_H = 1.55, 1.75   # ellipse size relative to the face box
LIFT = 0.04                 # move the ellipse up a little to cover the forehead/hair


def blur_face(img, box):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2 - LIFT * h
    ax, ay = w * PAD_W / 2, h * PAD_H / 2
    H, W = img.shape[:2]
    x0, x1 = int(max(0, cx - ax * 1.6)), int(min(W, cx + ax * 1.6))
    y0, y1 = int(max(0, cy - ay * 1.6)), int(min(H, cy + ay * 1.6))
    roi = img[y0:y1, x0:x1].astype(np.float32)
    rh, rw = roi.shape[:2]
    # pixelate, then blur hard
    cell = max(6, int(w / 7))
    small = cv2.resize(roi, (max(1, rw // cell), max(1, rh // cell)), interpolation=cv2.INTER_AREA)
    pix = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    sigma = max(8.0, w * 0.16)
    soft = cv2.GaussianBlur(pix, (0, 0), sigma)
    # feathered elliptical mask
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2
    mask = np.clip((1.25 - d) / 0.35, 0, 1)[..., None]
    img[y0:y1, x0:x1] = (soft * mask + roi * (1 - mask)).astype(np.uint8)


def load_original(name):
    src = next(p for p in RAW.rglob("*") if p.stem.upper() == name)
    rgb = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def main():
    pillow_heif.register_heif_opener()
    OUT.mkdir(parents=True, exist_ok=True)
    faces = json.loads(FACES.read_text())
    for name, boxes in faces.items():
        img = load_original(name)
        with Image.open(JPG / f"{name}.jpg") as small:
            k = img.shape[1] / small.width
        for b in boxes:
            blur_face(img, [v * k for v in b])
        cv2.imwrite(str(OUT / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"{len(faces)} photos, {sum(len(b) for b in faces.values())} faces blurred")


if __name__ == "__main__":
    main()
