"""Render single frames for checking layout without encoding the whole reel.

Usage: python -I edits/stills.py            -> work/stills/sheet_NN.jpg (middle of every segment + title/end steps)
       python -I edits/stills.py 1.2 30.5   -> work/stills/t_1.20.png ... (full size, chosen times)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import Renderer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "stills"


def still(r, t):
    r.prev_frame = None
    fr = r.frame(t)
    return cv2.cvtColor((np.clip(fr, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = Renderer()
    if len(sys.argv) > 1:
        for a in sys.argv[1:]:
            t = float(a)
            cv2.imwrite(str(OUT / f"t_{t:.2f}.png"), still(r, t))
        return
    times = [0.25, 0.8, 1.3, 1.75]
    times += [(s["t0"] + s["t1"]) / 2 for s in r.segs if s["kind"] in ("shot", "family")]
    end = r.segs[-1]
    times += [end["t0"] + d for d in (0.5, 1.4, 2.6, 5.0)]
    tiles = []
    for t in times:
        im = cv2.resize(still(r, t), (360, 640), interpolation=cv2.INTER_AREA)
        cv2.putText(im, f"{t:.2f}s", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(im)
    per = 10
    for s in range(0, len(tiles), per):
        group = tiles[s:s + per]
        while len(group) < per:
            group.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(group[:5]), np.hstack(group[5:])])
        cv2.imwrite(str(OUT / f"sheet_{s // per + 1:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(len(times), "stills")


if __name__ == "__main__":
    main()
