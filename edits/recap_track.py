"""Turn the per-frame face detections of the recap clips into smooth tracks, one per person.

Detections are linked from frame to frame (allowing for the camera's own motion), short gaps are filled,
each track is carried a little past its first and last hit, and the boxes are smoothed so the blur glides.
edits/recap_faces.json holds what was decided by eye:
  {"IMG_xxxx": {"drop": [track ids that are not faces], "keep": [weak track ids that are],
                "add": [{"keys": {"frame": [x, y, w, h], ...}}]}}   # hand-placed tracks, interpolated between keys

Usage: python -I edits/recap_track.py [--sheets] [IMG_xxxx ...]
  -> work/recap/tracks/IMG_xxxx.json  {"n", "tracks": [{"id", "start", "boxes": [[x, y, w, h], ...]}]}
  -> work/recap/track_sheets/IMG_xxxx.jpg  each track's best face, numbered, for checking by eye
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "source" / "recap_raw"
FACES = ROOT / "work" / "recap" / "faces"
TRACKS = ROOT / "work" / "recap" / "tracks"
SHEETS = ROOT / "work" / "recap" / "track_sheets"
FIXES = ROOT / "edits" / "recap_faces.json"
MAX_GAP = 24         # frames a track may go unseen and still be the same person
CARRY = 15           # frames a track is carried past its first and last hit
AUTO_KEEP = 0.7      # best score a track needs to count as a face without being kept by hand
MIN_HITS = 3


def link(dets, motion, n):
    """Greedy frame-to-frame linking; a track's position follows the camera while it goes unseen."""
    tracks, live = [], []
    for f in range(n):
        M = np.array(motion[f])
        for t in live:
            cx, cy = t["x"] + t["w"] / 2, t["y"] + t["h"] / 2
            nx, ny = M @ np.array([cx, cy, 1.0])
            t["x"] += nx - cx
            t["y"] += ny - cy
        boxes = dets.get(str(f), [])
        pairs = []
        for i, b in enumerate(boxes):
            bx, by = b[0] + b[2] / 2, b[1] + b[3] / 2
            for t in live:
                d = np.hypot(bx - (t["x"] + t["w"] / 2), by - (t["y"] + t["h"] / 2))
                size = max(b[2], t["w"])
                if d < 1.1 * size and 0.5 < b[2] / t["w"] < 2.0:
                    pairs.append((d / size, i, id(t), t))
        pairs.sort(key=lambda p: p[0])
        used_b, used_t = set(), set()
        for _, i, tid, t in pairs:
            if i in used_b or tid in used_t:
                continue
            used_b.add(i)
            used_t.add(tid)
            b = boxes[i]
            t.update(x=b[0], y=b[1], w=b[2], h=b[3], last=f)
            t["hits"][f] = b
        for i, b in enumerate(boxes):
            if i not in used_b:
                t = dict(x=b[0], y=b[1], w=b[2], h=b[3], last=f, hits={f: b})
                tracks.append(t)
                live.append(t)
        live = [t for t in live if f - t["last"] <= MAX_GAP]
    return tracks


def to_boxes(hits, motion, n):
    """A box for every frame from first hit - CARRY to last hit + CARRY: interpolated across gaps, then
    carried along with the camera past the ends, then smoothed."""
    keys = sorted(hits)
    start, end = max(0, keys[0] - CARRY), min(n - 1, keys[-1] + CARRY)
    arr = np.array([hits[k][:4] for k in keys], np.float64)
    cx, cy = arr[:, 0] + arr[:, 2] / 2, arr[:, 1] + arr[:, 3] / 2
    fr = np.arange(start, end + 1)
    out = np.stack([np.interp(fr, keys, cx), np.interp(fr, keys, cy), np.interp(fr, keys, arr[:, 2]),
                    np.interp(fr, keys, arr[:, 3])], 1)
    for f in range(keys[-1] + 1, end + 1):  # forwards past the last hit
        x, y = np.array(motion[f]) @ np.array([out[f - 1 - start, 0], out[f - 1 - start, 1], 1.0])
        out[f - start, :2] = x, y
    for f in range(keys[0] - 1, start - 1, -1):  # backwards before the first hit
        M = np.vstack([motion[f + 1], [0, 0, 1]])
        x, y, _ = np.linalg.solve(M, np.array([out[f + 1 - start, 0], out[f + 1 - start, 1], 1.0]))
        out[f - start, :2] = x, y
    k = np.exp(-0.5 * (np.arange(-4, 5) / 1.6) ** 2)
    k /= k.sum()
    pad = np.pad(out, ((4, 4), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(pad[:, c], k, "valid") for c in range(4)], 1)
    smooth[:, 0] -= smooth[:, 2] / 2
    smooth[:, 1] -= smooth[:, 3] / 2
    return start, smooth.round(1).tolist()


def hand_track(keys, n):
    fr = sorted(int(k) for k in keys)
    arr = np.array([keys[str(k)] for k in fr], np.float64)
    span = np.arange(fr[0], min(n - 1, fr[-1]) + 1)
    return fr[0], np.stack([np.interp(span, fr, arr[:, c]) for c in range(4)], 1).round(1).tolist()


def grab(path, frame):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
    return np.frombuffer(raw[:2160 * 3840 * 3], np.uint8).reshape(3840, 2160, 3)


def sheet(name, candidates):
    """One numbered tile per track: green = kept as a face, red = dropped."""
    if not candidates:
        return
    path = next(p for p in RAW.iterdir() if p.stem.upper() == name and p.suffix.lower() == ".mov")
    size, cols = 200, 8
    rows = (len(candidates) + cols - 1) // cols
    out = np.full((rows * (size + 34), cols * size, 3), 255, np.uint8)
    frames = {}
    for i, c in enumerate(candidates):
        fr = frames.setdefault(c["best_frame"], grab(path, c["best_frame"]))
        x, y, w, h = c["best"][:4]
        m = max(w, h) * 1.2
        x0, y0 = int(max(0, x + w / 2 - m)), int(max(0, y + h / 2 - m))
        tile = cv2.resize(fr[y0:int(y0 + 2 * m), x0:int(x0 + 2 * m)], (size, size))
        col = (0, 170, 0) if c["kept"] else (0, 0, 220)
        cv2.rectangle(tile, (0, 0), (size - 1, size - 1), col, 5)
        ty, tx = (i // cols) * (size + 34), (i % cols) * size
        out[ty:ty + size, tx:tx + size] = tile
        cv2.putText(out, f"{c['id']} {c['score']:.2f} n{c['hits']} f{c['best_frame']}", (tx + 2, ty + size + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    SHEETS.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SHEETS / f"{name}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    TRACKS.mkdir(parents=True, exist_ok=True)
    fixes = json.loads(FIXES.read_text()) if FIXES.exists() else {}
    names = [a for a in sys.argv[1:] if a != "--sheets"] or sorted(p.stem for p in FACES.glob("*.json"))
    for name in names:
        data = json.loads((FACES / f"{name}.json").read_text())
        n, motion, fx = data["n"], data["motion"], fixes.get(name, {})
        raw = link(data["dets"], motion, n)
        raw.sort(key=lambda t: min(t["hits"]))
        tracks, cands = [], []
        for tid, t in enumerate(raw):
            best_f = max(t["hits"], key=lambda f: t["hits"][f][4])
            best = t["hits"][best_f]
            auto = best[4] >= AUTO_KEEP and len(t["hits"]) >= MIN_HITS
            kept = (auto or tid in fx.get("keep", [])) and tid not in fx.get("drop", [])
            cands.append(dict(id=tid, kept=kept, score=best[4], hits=len(t["hits"]), best=best, best_frame=best_f))
            if kept:
                start, boxes = to_boxes(t["hits"], motion, n)
                tracks.append(dict(id=tid, start=start, boxes=boxes))
        for j, a in enumerate(fx.get("add", [])):
            start, boxes = hand_track(a["keys"], n)
            tracks.append(dict(id=f"hand{j}", start=start, boxes=boxes))
        (TRACKS / f"{name}.json").write_text(json.dumps({"n": n, "tracks": tracks}))
        if "--sheets" in sys.argv:
            sheet(name, sorted(cands, key=lambda c: -c["score"])[:48])
        print(f"{name}: {n} frames, {len(raw)} candidates, {len(tracks)} kept", flush=True)


if __name__ == "__main__":
    main()
