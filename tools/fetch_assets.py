"""Download the fonts, the face-detection model and the Tiny Stars logo the pipeline needs.

Usage: python -I tools/fetch_assets.py
  -> brand/fonts/*.ttf (Google Fonts, OFL / Apache 2.0), models/face_detection_yunet_2023mar.onnx (OpenCV Zoo),
     brand/logo.png (from tinystars.ca)
"""
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = {
    "brand/fonts/GrenzeGotisch.ttf":
        "https://github.com/google/fonts/raw/main/ofl/grenzegotisch/GrenzeGotisch%5Bwght%5D.ttf",
    "brand/fonts/Cinzel.ttf": "https://github.com/google/fonts/raw/main/ofl/cinzel/Cinzel%5Bwght%5D.ttf",
    "brand/fonts/Montserrat.ttf": "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",
    "models/face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "brand/logo.png": "https://tinystars.ca/assets/images/logo/logo.png",
}


def main():
    for rel, url in FILES.items():
        dest = ROOT / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r:
            dest.write_bytes(r.read())
        print("fetched", rel)


if __name__ == "__main__":
    main()
