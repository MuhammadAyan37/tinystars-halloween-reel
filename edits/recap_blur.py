"""Bake the face blur into the recap clips, so everything after this step only ever sees blurred faces.

Each clip is decoded one frame at a time, every tracked face is blurred the same way as in the photo reel
(pixelated, then heavily blurred, inside a padded feathered ellipse), and the result is written as a
high-quality 4K working copy. Clips with no faces are copied through the same way.

Usage: python -I edits/recap_track.py && python -I edits/recap_blur.py [IMG_xxxx ...]
  -> work/recap/clean/IMG_xxxx.mp4  (2160x3840, 30 fps, no sound; finished clips are skipped unless named)
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blur_faces import blur_face  # noqa: E402
from recap_detect import H, RAW, W, clip_path, frames  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRACKS = ROOT / "work" / "recap" / "tracks"
OUT = ROOT / "work" / "recap" / "clean"


def writer(path):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                             "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "13",
                             "-pix_fmt", "yuv420p", "-threads", "4", "-x264-params", "rc-lookahead=10",
                             "-g", "15", str(path)], stdin=subprocess.PIPE)


def bake(name):
    tracks = json.loads((TRACKS / f"{name}.json").read_text())["tracks"]
    out = writer(OUT / f"{name}.mp4")
    n = 0
    for i, fr in enumerate(frames(clip_path(name))):
        img = fr.copy()
        for t in tracks:
            k = i - t["start"]
            if 0 <= k < len(t["boxes"]):
                x, y, w, h = t["boxes"][k]
                if x + w * 1.5 > 0 and y + h * 1.5 > 0 and x - w * 0.5 < W and y - h * 0.5 < H:
                    blur_face(img, [x, y, w, h])
        out.stdin.write(img.tobytes())
        n = i + 1
    out.stdin.close()
    out.wait()
    print(f"{name}: {n} frames, {len(tracks)} tracks", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    named = sys.argv[1:]
    names = named or sorted(p.stem.upper() for p in RAW.iterdir() if p.suffix.lower() == ".mov")
    for name in names:
        if named or not (OUT / f"{name}.mp4").exists():
            bake(name)


if __name__ == "__main__":
    main()
