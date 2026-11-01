"""Find every face in the chosen photos so they can be blurred.

Runs OpenCV's YuNet detector at two scales and merges the boxes. Hits scoring 0.8+ are
kept; weaker ones (mostly the inflatable ghost and the little ghosts on the tree) are
dropped unless edits/face_fixes.json keeps them. That file also adds boxes the detector
missed and drops confident false hits: {"IMG_xxxx": {"keep": [i], "drop": [i], "add": [[x, y, w, h]]}}.

Usage: python -I edits/detect_faces.py [IMG_xxxx ...]
  -> work/faces.json  {"IMG_xxxx": [[x, y, w, h], ...]}  (pixels in work/jpg/IMG_xxxx.jpg)
  -> work/review/IMG_xxxx.jpg  photo with numbered boxes, for checking by eye
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content import FAMILY, SHOTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
JPG = ROOT / "work" / "jpg"
MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
FIXES = ROOT / "edits" / "face_fixes.json"
OUT = ROOT / "work" / "faces.json"
REVIEW = ROOT / "work" / "review"
SCORE = 0.45
AUTO_KEEP = 0.8


def yunet(img, scale):
    h, w = img.shape[:2]
    im = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    det = cv2.FaceDetectorYN.create(str(MODEL), "", (im.shape[1], im.shape[0]), SCORE, 0.3, 5000)
    _, faces = det.detect(im)
    boxes = []
    if faces is not None:
        for f in faces:
            x, y, bw, bh = f[:4] / scale
            boxes.append([float(x), float(y), float(bw), float(bh), float(f[-1])])
    return boxes


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def merge(boxes):
    boxes = sorted(boxes, key=lambda b: -b[4])
    kept = []
    for b in boxes:
        if all(iou(b, k) < 0.3 for k in kept):
            kept.append(b)
    return kept


def main():
    names = sys.argv[1:] or [s["photo"] for s in SHOTS + FAMILY["shots"]]
    fixes = json.loads(FIXES.read_text()) if FIXES.exists() else {}
    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    REVIEW.mkdir(parents=True, exist_ok=True)
    for name in names:
        img = cv2.imread(str(JPG / f"{name}.jpg"))
        found = merge(yunet(img, 1.0) + yunet(img, 0.6))
        fx = fixes.get(name, {})
        # confident hits are kept unless dropped by hand; weak ones only when kept by hand
        drop = {i for i, b in enumerate(found)
                if (b[4] < AUTO_KEEP and i not in fx.get("keep", [])) or i in fx.get("drop", [])}
        boxes = [b[:4] for i, b in enumerate(found) if i not in drop]
        boxes += fx.get("add", [])
        result[name] = [[round(v) for v in b] for b in boxes]
        vis = img.copy()
        for i, b in enumerate(found):
            x, y, w, h = map(int, b[:4])
            col = (0, 0, 255) if i in drop else (0, 255, 0)
            cv2.rectangle(vis, (x, y), (x + w, y + h), col, 4)
            cv2.putText(vis, f"{i}:{b[4]:.2f}", (x, max(30, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
        for b in fx.get("add", []):
            x, y, w, h = map(int, b)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 4)
        cv2.imwrite(str(REVIEW / f"{name}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(name, len(found), "found ->", len(result[name]), "kept", [list(map(int, b[:4])) + [round(b[4], 2)] for b in found])
    OUT.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
