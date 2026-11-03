"""Privacy check on the finished reel: look through every frame for anything that still reads as a face.

Runs YuNet at normal and double size with a low threshold, so it errs on the side of raising alarms; decorations with
faces (ghosts, pumpkins, owls) will show up too. Every hit is cut out onto a sheet for checking by eye.

Usage: python -I edits/recap_verify.py [video]
  -> work/recap/verify_N.jpg  (each hit enlarged, labelled with its frame number and score)
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
W, H = 1080, 1920
SCORE = 0.55


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "exports" / "recap_last_week.mp4")
    det1 = cv2.FaceDetectorYN.create(str(MODEL), "", (W, H), SCORE, 0.3, 200)
    det2 = cv2.FaceDetectorYN.create(str(MODEL), "", (W * 3 // 2, H * 3 // 2), SCORE, 0.3, 200)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE, bufsize=W * H * 3)
    tiles, i = [], 0
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        hits = []
        _, f1 = det1.detect(fr)
        if f1 is not None:
            hits += [(*f[:4], f[-1]) for f in f1]
        if i % 2 == 0:
            big = cv2.resize(fr, (W * 3 // 2, H * 3 // 2), interpolation=cv2.INTER_CUBIC)
            _, f2 = det2.detect(big)
            if f2 is not None:
                hits += [(*(f[:4] / 1.5), f[-1]) for f in f2]
        for x, y, w, h, s in hits:
            m = max(w, h) * 0.9
            x0, y0 = int(max(0, x + w / 2 - m)), int(max(0, y + h / 2 - m))
            x1, y1 = int(min(W, x + w / 2 + m)), int(min(H, y + h / 2 + m))
            if x1 <= x0 or y1 <= y0:
                continue
            tile = cv2.resize(fr[y0:y1, x0:x1], (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.putText(tile, f"f{i} {s:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        i += 1
    p.wait()
    print(f"{i} frames, {len(tiles)} hits")
    per = 13 * 13
    for page in range(0, len(tiles), per):
        part = tiles[page:page + per]
        sheet = np.full((((len(part) + 12) // 13) * 152, 13 * 152, 3), 255, np.uint8)
        for k, t in enumerate(part):
            sheet[(k // 13) * 152:(k // 13) * 152 + 150, (k % 13) * 152:(k % 13) * 152 + 150] = t
        cv2.imwrite(str(ROOT / "work" / "recap" / f"verify_{page // per + 1}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
