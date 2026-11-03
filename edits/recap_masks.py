"""Cut the star of each "cast card" out of its frozen frame, so the background can drop away behind them.

Runs in its own environment because rembg brings a lot with it, and only ever reads the blurred working copies:
  uv venv --python 3.12 work/recap/mask_env
  uv pip install --python work/recap/mask_env/Scripts/python.exe "rembg[cpu]" opencv-python-headless

Usage: work/recap/mask_env/Scripts/python.exe -I edits/recap_masks.py IMG_xxxx:FRAME[:x0,y0,x1,y1][:-x0,y0,x1,y1 ...][:+x0,y0,x1,y1 ...][:cN] ...
  (a box, as fractions of the frame, looks only at that part of the picture; a box starting with "-" is cut out of the
  result, for when a second child or a chair is in shot; a box starting with "+" is filled in; more than two corners
  make it a polygon; cN closes gaps up to N pixels wide)
  -> work/recap/masks/IMG_xxxx_fFRAME.png  (8-bit matte at 1620x2880)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from rembg import new_session, remove
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "work" / "recap" / "clean"
OUT = ROOT / "work" / "recap" / "masks"
SIZE = (1620, 2880)
MODEL = "isnet-general-use"


def read_frame(name, frame):
    cap = cv2.VideoCapture(str(CLEAN / f"{name}.mp4"))
    for _ in range(frame):
        cap.grab()
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"{name} has no frame {frame}")
    return cv2.resize(fr, SIZE, interpolation=cv2.INTER_AREA)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    session = new_session(MODEL)
    for arg in sys.argv[1:]:
        parts = arg.split(":")
        name, frame = parts[0], int(parts[1])
        fr = read_frame(name, frame)
        crops = [q for q in parts[2:] if not q.startswith(("-", "+", "c"))]
        x0, y0, x1, y1 = [float(v) for v in crops[0].split(",")] if crops else (0, 0, 1, 1)
        x0, y0, x1, y1 = int(x0 * SIZE[0]), int(y0 * SIZE[1]), int(x1 * SIZE[0]), int(y1 * SIZE[1])
        part = Image.fromarray(cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        matte = np.zeros((SIZE[1], SIZE[0]), np.uint8)
        matte[y0:y1, x0:x1] = np.asarray(remove(part, session=session, only_mask=True))
        for q in parts[2:]:
            if q.startswith(("-", "+")):
                v = np.array([float(n) for n in q[1:].split(",")]).reshape(-1, 2) * SIZE
                if len(v) == 2:  # two corners make a rectangle
                    (a0, b0), (a1, b1) = v
                    v = np.array([[a0, b0], [a1, b0], [a1, b1], [a0, b1]])
                cv2.fillPoly(matte, [v.astype(np.int32)], 0 if q[0] == "-" else 255)
        for q in parts[2:]:
            if q.startswith("c"):
                size = int(q[1:]) | 1
                matte = cv2.morphologyEx(matte, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        # keep only the biggest connected shape, so stray furniture does not come along
        n, labels, stats, _ = cv2.connectedComponentsWithStats((matte > 128).astype(np.uint8))
        if n > 1:
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            region = cv2.dilate((labels == big).astype(np.uint8), np.ones((15, 15), np.uint8))
            matte = (matte * region).astype(np.uint8)
        cv2.imwrite(str(OUT / f"{name}_f{frame}.png"), matte)
        print(name, frame, "covers", round(float((matte > 128).mean()) * 100, 1), "% of the frame", flush=True)


if __name__ == "__main__":
    main()
