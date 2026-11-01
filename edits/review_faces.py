"""Put the face-check images for every photo in the reel on 2x2 sheets for checking by eye.

Usage: python -I edits/review_faces.py  -> work/review/sheet_NN.jpg
Green = blurred, red = detector hit ignored, magenta = box added by hand.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content import FAMILY, SHOTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REVIEW = ROOT / "work" / "review"
CELL = 1000


def main():
    names = [s["photo"] for s in SHOTS] + [s["photo"] for s in FAMILY["shots"]]
    font = ImageFont.truetype("arialbd.ttf", 36)
    for s in range(0, len(names), 4):
        sheet = Image.new("RGB", (2 * CELL, 2 * CELL), "white")
        d = ImageDraw.Draw(sheet)
        for i, n in enumerate(names[s:s + 4]):
            im = Image.open(REVIEW / f"{n}.jpg")
            im.thumbnail((CELL - 10, CELL - 50))
            x, y = (i % 2) * CELL, (i // 2) * CELL
            sheet.paste(im, (x + 5, y + 46))
            d.text((x + 8, y + 4), n, fill="black", font=font)
        sheet.save(REVIEW / f"sheet_{s // 4 + 1:02d}.jpg", quality=85)


if __name__ == "__main__":
    main()
