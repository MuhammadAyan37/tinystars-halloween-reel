"""Find every face in every frame of the recap clips, plus the camera's motion between frames.

YuNet runs at three scales (the clips are 4K, so far-away kids are tiny and close-up kids are huge):
half and quarter size on every frame, nearly full size on every third frame.

Usage: python -I edits/recap_detect.py [IMG_xxxx ...]
  -> work/recap/faces/IMG_xxxx.json  {"n", "w", "h", "dets": {frame: [[x, y, w, h, score], ...]}, "motion": [[2x3], ...]}
     (pixels in the upright 2160x3840 frame; motion[i] maps frame i-1 to frame i). Finished clips are skipped.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "source" / "recap_raw"
OUT = ROOT / "work" / "recap" / "faces"
MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
W, H = 2160, 3840
SCORE = 0.4
# (scale, run on every Nth frame)
SCALES = ((0.75, 3), (0.5, 1), (0.25, 1))


def clip_path(name):
    return next(p for p in RAW.iterdir() if p.stem.upper() == name and p.suffix.lower() == ".mov")


def frames(path):
    """Upright BGR frames, one at a time, so a 4K clip never sits in memory."""
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE, bufsize=W * H * 3)
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    p.wait()


def iou(a, b):
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def merge(boxes):
    kept = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        if all(iou(b, k) < 0.3 for k in kept):
            kept.append(b)
    return kept


def camera_motion(prev, gray):
    """How the whole picture moved from prev to gray (quarter-size grayscale), in full-frame pixels."""
    pts = cv2.goodFeaturesToTrack(prev, 400, 0.01, 12)
    if pts is None or len(pts) < 8:
        return [[1, 0, 0], [0, 1, 0]]
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, winSize=(21, 21), maxLevel=3)
    good = st[:, 0] == 1
    M, _ = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None:
        return [[1, 0, 0], [0, 1, 0]]
    M[:, 2] *= 4
    return M.tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or sorted(p.stem.upper() for p in RAW.iterdir() if p.suffix.lower() == ".mov")
    dets = {s: cv2.FaceDetectorYN.create(str(MODEL), "", (int(W * s), int(H * s)), SCORE, 0.3, 500) for s, _ in SCALES}
    for name in names:
        out = OUT / f"{name}.json"
        if out.exists():
            continue
        t0 = time.time()
        result, motion, prev, i = {}, [], None, -1
        for i, fr in enumerate(frames(clip_path(name))):
            found = []
            for s, every in SCALES:
                if i % every:
                    continue
                im = cv2.resize(fr, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
                _, faces = dets[s].detect(im)
                if faces is not None:
                    found += [[float(v) / s for v in f[:4]] + [float(f[-1])] for f in faces]
            result[str(i)] = [[round(v, 1) for v in b[:4]] + [round(b[4], 3)] for b in merge(found)]
            gray = cv2.cvtColor(cv2.resize(fr, (W // 4, H // 4), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            motion.append([[1, 0, 0], [0, 1, 0]] if prev is None else camera_motion(prev, gray))
            prev = gray
        out.write_text(json.dumps({"n": i + 1, "w": W, "h": H, "dets": result, "motion": motion}))
        hits = sum(len(v) for v in result.values())
        print(f"{name}: {i + 1} frames, {hits} detections, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
