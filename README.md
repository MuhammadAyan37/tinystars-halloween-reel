# Tiny Stars Halloween Costume Parade reel

Code that turns a folder of Halloween photo-booth pictures from [Tiny Stars Daycare](https://tinystars.ca) (Grande Prairie, Alberta) into a 38-second vertical reel for Instagram, Facebook and TikTok, plus a matching carousel and collage post. **Every face is blurred**, kids and parents alike, and every cut lands on the beat of the music.

![The reel's graphics: title, numbered costume labels and call to action](docs/demo.png)

*The graphics on their own. The finished reel puts these over the costume photos.*

## What it makes

- **The reel** (1080×1920, 30 fps, H.264 + AAC, normalised to −14 LUFS): an animated title, then 22 numbered costumes with Ken Burns motion, beat-synced punch and whip transitions, a "grown-ups dressed up too" section, and an end card with a "comment your favourite number" call to action and enrolment details.
- **A no-music version**, for adding a trending sound inside the app.
- **A 10-slide 4:5 carousel**, a numbered collage of all 22 costumes, and a reel cover.

## Privacy first

The photos, the face boxes and everything rendered from them stay out of git (see `.gitignore`). Faces are handled in three steps:

1. `detect_faces.py` runs OpenCV's YuNet detector at two scales. Confident hits are kept automatically.
2. Weaker hits, and anything the detector missed, are checked by eye on review sheets. Fixes go in `edits/face_fixes.json`.
3. `blur_faces.py` pixelates each face, then heavily blurs it inside a padded, feathered ellipse that also covers the hairline and chin. It works on the full-resolution originals.

## Run it

You need Python 3.12, [uv](https://docs.astral.sh/uv/) and `ffmpeg` on your PATH. Put the HEIC/JPG photos in `source/drive_raw/` and the music file in `source/music/`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -I tools/fetch_assets.py     # fonts, face model, logo
.venv/Scripts/python.exe -I edits/prep_photos.py      # HEIC -> upright JPGs + contact sheets
.venv/Scripts/python.exe -I edits/detect_faces.py     # face boxes -> work/faces.json, review images
.venv/Scripts/python.exe -I edits/review_faces.py     # 2x2 sheets for checking faces by eye
.venv/Scripts/python.exe -I edits/blur_faces.py       # full-resolution blurred photos
.venv/Scripts/python.exe -I edits/stills.py           # layout check frames
.venv/Scripts/python.exe -I edits/render.py           # exports/tinystars_halloween_reel.mp4
.venv/Scripts/python.exe -I edits/render.py --no-music
.venv/Scripts/python.exe -I edits/posts.py            # carousel, collage, reel cover
```

The photo order, the costume labels, the timing (in beats) and the end-card text all live in `edits/content.py`.

## Tech stack

- **Python** with NumPy, OpenCV (YuNet face detection, warping, compositing) and Pillow (lettering and graphics, `pillow-heif` for iPhone photos)
- **FFmpeg** for H.264/AAC encoding and two-pass EBU R128 loudness normalisation
- **Fonts:** Creepster, Luckiest Guy and Baloo 2 from Google Fonts (OFL / Apache 2.0)
- **Music:** "Ghosts of the Past" by melodyayresgriffiths on [Pixabay](https://pixabay.com/music/scary-childrens-tunes-ghosts-of-the-past-instrumental-halloween-autumn-spooky-upbeat-boppy-171666/) (Pixabay Content License, no Content ID)

## License

The code is MIT licensed (see `LICENSE`). The Tiny Stars name and logo belong to Tiny Stars Daycare. The fonts, the face model and the music keep their own licenses.
