"""The finished recap laid out for a TV: 1920x1080, H.264 + AAC.

The footage was filmed vertically, so filling a wide screen would crop off the kids' heads and costumes. Instead the film
sits in the middle at full height, on a heavily blurred, darkened copy of itself, with the logo, the title and the contact
details in the two side panels. It only ever reads the finished, face-blurred film.

Usage: python -I edits/recap_tv.py [video]
  -> exports/recap_last_week_tv.mp4
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recap as C  # noqa: E402
from content import BRAND  # noqa: E402
from render import CREAM, WHITE, fit_width, icon_image, lettering, logo_image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TV_W, TV_H = 1920, 1080
PANEL_W = 608  # the 9:16 film at full height
OUT = ROOT / "exports" / "recap_last_week_tv.mp4"
OVERLAY = ROOT / "work" / "recap" / "tv_overlay.png"


def put(canvas, sprite, cx, cy):
    canvas.alpha_composite(sprite, (int(cx - sprite.width / 2), int(cy - sprite.height / 2)))


def overlay():
    """Everything that stays still: the side panels' words and the shadow the film casts on the background."""
    x0, x1 = (TV_W - PANEL_W) // 2, (TV_W + PANEL_W) // 2
    # darken the sides a little, more towards the film so it seems to float above them
    a = np.full((TV_H, TV_W), 0.26, np.float32)
    dist = np.maximum(np.maximum(x0 - np.arange(TV_W), np.arange(TV_W) - x1), 0).astype(np.float32)
    a = 1 - (1 - a) * (1 - 0.6 * np.exp(-dist / 34))[None, :]
    canvas = Image.fromarray((np.dstack([np.zeros((TV_H, TV_W, 3)), a]) * 255).astype(np.uint8), "RGBA")
    left, right = x0 / 2, (x1 + TV_W) / 2
    # left: who and what
    put(canvas, logo_image(340), left, 375)
    put(canvas, fit_width(lettering("Recap", C.display(170), 0, fill=(CREAM, CREAM), shadow=0.8), 540), left, 675)
    put(canvas, lettering("OF LAST WEEK", C.small(38, 800), 10, fill=(C.GOLD_LIGHT, C.GOLD), shadow=0.8), left, 805)
    # right: how to reach us
    rows = [("phone", [BRAND["phone"]]), ("pin", ["10629 West Side Drive", "Grande Prairie, AB"]), ("mail", [BRAND["email"]])]
    texts = [[lettering(t, C.small(36, 700 if len(lines) == 1 else 600), 0, fill=(WHITE, WHITE), shadow=0.7) for t in lines]
             for _, lines in rows]
    width = max(s.width for row in texts for s in row)
    x = right - (110 + width) / 2
    y = 320
    for (kind, _), row in zip(rows, texts):
        icon = icon_image(kind, 50, C.GOLD)
        put(canvas, icon, x + 40, y)
        for k, s in enumerate(row):
            ty = y + (k - (len(row) - 1) / 2) * 48
            canvas.alpha_composite(s, (int(x + 110), int(ty - s.height / 2)))
        y += 150
    put(canvas, lettering(BRAND["web"], C.small(44, 800), 2, fill=(C.GOLD_LIGHT, C.GOLD), shadow=0.8), right, y + 30)
    OVERLAY.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OVERLAY)


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "exports" / "recap_last_week.mp4")
    overlay()
    x0 = (TV_W - PANEL_W) // 2
    # worked out in RGB, so the logo's colours stay exact; a touch of still noise stops the soft background banding on a big screen
    graph = ("[0:v]scale=in_color_matrix=bt709:in_range=tv,format=gbrp,split[a][b];"
             "[a]scale=480:854:flags=area,crop=480:270:0:292,gblur=sigma=26,"
             f"scale={TV_W}:{TV_H}:flags=bicubic,eq=saturation=1.2,noise=alls=3:allf=u[bg];"
             f"[b]scale={PANEL_W}:{TV_H}:flags=lanczos[film];"
             "[bg][1:v]overlay=format=gbrp[side];"
             f"[side][film]overlay={x0}:0:format=gbrp,"
             "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[v]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", str(OVERLAY), "-filter_complex", graph,
                    "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
                    "-maxrate", "16M", "-bufsize", "32M", "-profile:v", "high", "-level", "4.2",
                    "-x264-params", "rc-lookahead=20", "-threads", "4", "-g", "60",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-bsf:v", "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
                    "-c:a", "copy", "-movflags", "+faststart", str(OUT)], check=True)
    print("wrote", OUT, f"({OUT.stat().st_size / 2 ** 20:.1f} MiB)")


if __name__ == "__main__":
    main()
