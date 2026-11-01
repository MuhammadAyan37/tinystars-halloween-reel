"""Decode the iPhone HEIC originals to upright JPGs and build numbered contact sheets.

Usage: python -I edits/prep_photos.py
  source/drive_raw/*.HEIC -> work/jpg/IMG_xxxx.jpg (EXIF orientation applied, long side 2400 px)
                          -> work/sheets/sheet_NN.jpg (4x3 thumbnails with file names)
"""
from pathlib import Path

import pillow_heif
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "source" / "drive_raw"
JPG = ROOT / "work" / "jpg"
SHEETS = ROOT / "work" / "sheets"
LONG_SIDE = 2400
COLS, ROWS, THUMB = 4, 3, 480

pillow_heif.register_heif_opener()


def main():
    JPG.mkdir(parents=True, exist_ok=True)
    SHEETS.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW.rglob("*"), key=lambda p: p.stem.upper())
    files = [f for f in files if f.suffix.lower() in (".heic", ".jpg", ".jpeg", ".png")]
    names = []
    for f in files:
        out = JPG / f"{f.stem.upper()}.jpg"
        if not out.exists():
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            im.thumbnail((LONG_SIDE, LONG_SIDE), Image.LANCZOS)
            im.save(out, quality=94)
        names.append(out)
    font = ImageFont.truetype("arialbd.ttf", 28)
    per = COLS * ROWS
    for s in range(0, len(names), per):
        sheet = Image.new("RGB", (COLS * THUMB, ROWS * THUMB), "white")
        d = ImageDraw.Draw(sheet)
        for i, n in enumerate(names[s:s + per]):
            im = Image.open(n)
            im.thumbnail((THUMB - 8, THUMB - 40))
            x, y = (i % COLS) * THUMB, (i // COLS) * THUMB
            sheet.paste(im, (x + (THUMB - im.width) // 2, y + 36))
            d.text((x + 8, y + 4), n.stem.replace("IMG_", ""), fill="black", font=font)
        sheet.save(SHEETS / f"sheet_{s // per + 1:02d}.jpg", quality=85)
    print(f"{len(names)} photos, {(len(names) + per - 1) // per} sheets")


if __name__ == "__main__":
    main()
