"""Render the Tiny Stars Halloween costume-parade reel: 1080x1920, 30 fps, cut on the beat.

Timeline (beats of the 128 BPM track, see content.py for the shots):
  title (4 beats)  -> costume parade, one numbered label per costume
  -> "even the grown-ups dressed up" family shots -> end card (Happy Halloween, comment CTA, contact)

Usage: python -I edits/render.py [--preview] [--no-music]
  -> exports/tinystars_halloween_reel.mp4         (with music)
  -> exports/tinystars_halloween_reel_nomusic.mp4 (--no-music: silent track, for adding an in-app sound)
  --preview renders every 3rd frame at half size for a quick look (work/preview.mp4)
"""
import json
import math
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content import BRAND, END, FAMILY, SHOTS, TITLE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "brand" / "fonts"
BLURRED = ROOT / "work" / "blurred"
JPG = ROOT / "work" / "jpg"
FACES_JSON = ROOT / "work" / "faces.json"
MUSIC = ROOT / "source" / "music" / "ghosts_of_the_past_171666.mp3"
EXPORTS = ROOT / "exports"

W, H, FPS = 1080, 1920, 30
BEAT = 60 / 128          # the track is a steady 128 BPM
T0 = 0.0704              # first beat in the track (seconds)
TITLE_BEATS, END_BEATS = 4, 12
TAIL = 0.9               # hold the end card while the music fades

# brand palette (RGB, from tinystars.ca and the logo)
DEEP = (30, 11, 46)
INK = (43, 11, 63)
PURPLE = (100, 42, 115)
VIOLET = (98, 75, 255)
MAGENTA = (232, 62, 140)
ORANGE = (255, 122, 26)
GOLD = (246, 199, 37)
CREAM = (255, 246, 224)
WHITE = (255, 255, 255)


def bt(k):
    """Video time of beat k (beat 0 is the start of the video)."""
    return 0.0 if k <= 0 else T0 + k * BEAT


# ---------- easing ----------
def clamp01(x):
    return max(0.0, min(1.0, x))


def ease_out_cubic(p):
    p = clamp01(p)
    return 1 - (1 - p) ** 3


def ease_in_cubic(p):
    p = clamp01(p)
    return p ** 3


def ease_out_back(p, s=1.9):
    p = clamp01(p) - 1
    return 1 + (s + 1) * p ** 3 + s * p ** 2


def ease_in_out(p):
    return 0.5 - 0.5 * math.cos(math.pi * clamp01(p))


# ---------- sprites (premultiplied float RGBA) ----------
def to_sprite(img):
    a = np.asarray(img.convert("RGBA")).astype(np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    return a


def font(name, size, axes=None):
    f = ImageFont.truetype(str(FONTS / name), size)
    if axes:
        f.set_variation_by_axes(axes)
    return f


def vgrad(w, h, top, bottom, y0=0, y1=None):
    y1 = h if y1 is None else y1
    t = np.clip((np.arange(h)[:, None] - y0) / max(1, y1 - y0), 0, 1)
    rgb = np.array(top) * (1 - t[..., None]) + np.array(bottom) * t[..., None]
    rgb = np.repeat(rgb, w, axis=1)
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def hgrad(w, h, stops):
    xs = np.linspace(0, 1, w)
    pos = np.linspace(0, 1, len(stops))
    rgb = np.stack([np.interp(xs, pos, [s[c] for s in stops]) for c in range(3)], -1)
    return Image.fromarray(np.repeat(rgb[None], h, axis=0).astype(np.uint8), "RGB")


def text_image(text, f, fill=(WHITE, CREAM), stroke=INK, stroke_w=8, extrude=0, extrude_col=PURPLE,
               shadow=0.55, pad=40):
    """Chunky 'sticker' lettering like the Tiny Stars logo: gradient fill, outline, 3D extrude, soft shadow."""
    l, t, r, b = f.getbbox(text, stroke_width=stroke_w)
    w, h = r - l + 2 * pad, b - t + 2 * pad + extrude
    ox, oy = pad - l, pad - t
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if shadow:
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).text((ox, oy + extrude + 10), text, font=f, fill=255, stroke_width=stroke_w, stroke_fill=255)
        m = m.filter(ImageFilter.GaussianBlur(12)).point(lambda v: int(v * shadow))
        sh = Image.new("RGBA", (w, h), (10, 0, 20, 0))
        sh.putalpha(m)
        out = Image.alpha_composite(out, sh)
    d = ImageDraw.Draw(out)
    dark = tuple(int(c * 0.55) for c in extrude_col)
    for i in range(extrude, 0, -1):
        d.text((ox, oy + i), text, font=f, fill=extrude_col if i > 2 else dark,
               stroke_width=stroke_w, stroke_fill=dark if i > 2 else INK)
    d.text((ox, oy), text, font=f, fill=stroke, stroke_width=stroke_w, stroke_fill=stroke)
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).text((ox, oy), text, font=f, fill=255)
    _, ty0, _, ty1 = f.getbbox(text)
    out.paste(vgrad(w, h, fill[0], fill[1], oy + ty0, oy + ty1), (0, 0), m)
    return out


def pill_image(content, stops, pad_x=46, pad_y=22, border=6, radius=None, left_room=0):
    """Rounded gradient pill with a cream border and drop shadow, holding an RGBA image."""
    cw, ch = content.size
    pw, ph = cw + 2 * pad_x + left_room, ch + 2 * pad_y
    radius = radius or ph // 2
    m = 30
    out = Image.new("RGBA", (pw + 2 * m, ph + 2 * m + 12), (0, 0, 0, 0))
    sh = Image.new("L", out.size, 0)
    ImageDraw.Draw(sh).rounded_rectangle((m, m + 12, m + pw, m + ph + 12), radius, fill=150)
    sh = sh.filter(ImageFilter.GaussianBlur(10))
    shadow = Image.new("RGBA", out.size, (10, 0, 20, 0))
    shadow.putalpha(sh)
    out = Image.alpha_composite(out, shadow)
    mask = Image.new("L", out.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((m, m, m + pw, m + ph), radius, fill=255)
    body = Image.new("RGBA", out.size, (0, 0, 0, 0))
    body.paste(hgrad(pw, ph, stops), (m, m))
    out.paste(body, (0, 0), mask)
    ImageDraw.Draw(out).rounded_rectangle((m, m, m + pw, m + ph), radius, outline=CREAM, width=border)
    # soft gloss on the top half
    gloss = Image.new("L", out.size, 0)
    ImageDraw.Draw(gloss).rounded_rectangle((m + 10, m + 6, m + pw - 10, m + ph // 2), radius, fill=38)
    out = Image.alpha_composite(out, Image.merge("RGBA", (*[Image.new("L", out.size, 255)] * 3, gloss)))
    out.alpha_composite(content, (m + pad_x + left_room, m + pad_y))
    return out


def label_sprite(num, text):
    name = text_image(text.upper(), font("LuckiestGuy-Regular.ttf", 74), fill=(WHITE, CREAM), stroke_w=7,
                      extrude=6, extrude_col=INK, shadow=0, pad=8)
    badge_d = 168
    pill = pill_image(name, [ORANGE, MAGENTA, VIOLET], pad_x=34, pad_y=20, left_room=badge_d - 52)
    out = Image.new("RGBA", (pill.width + 40, max(pill.height, badge_d + 60)), (0, 0, 0, 0))
    py = (out.height - pill.height) // 2
    out.alpha_composite(pill, (40, py))
    # gold number badge overlapping the pill's left end
    b = Image.new("RGBA", (badge_d + 40, badge_d + 50), (0, 0, 0, 0))
    sh = Image.new("L", b.size, 0)
    ImageDraw.Draw(sh).ellipse((20, 30, 20 + badge_d, 30 + badge_d), fill=150)
    b.putalpha(sh.filter(ImageFilter.GaussianBlur(9)))
    b = Image.composite(Image.new("RGBA", b.size, (10, 0, 20, 255)), Image.new("RGBA", b.size, 0), b.getchannel("A"))
    disc = Image.new("RGBA", b.size, (0, 0, 0, 0))
    dm = Image.new("L", b.size, 0)
    ImageDraw.Draw(dm).ellipse((20, 20, 20 + badge_d, 20 + badge_d), fill=255)
    disc.paste(vgrad(b.width, b.height, (255, 226, 92), (240, 160, 20), 20, 20 + badge_d), (0, 0), dm)
    ImageDraw.Draw(disc).ellipse((20, 20, 20 + badge_d, 20 + badge_d), outline=INK, width=8)
    b = Image.alpha_composite(b, disc)
    n = text_image(str(num), font("LuckiestGuy-Regular.ttf", 92), fill=(PURPLE, INK), stroke=CREAM,
                   stroke_w=0, extrude=0, shadow=0, pad=4)
    b.alpha_composite(n, (20 + (badge_d - n.width) // 2, 20 + (badge_d - n.height) // 2 + 6))
    out.alpha_composite(b, (0, (out.height - b.height) // 2 + 4))
    return out


def sparkle_image(r, color):
    s = int(r * 2 + 40)
    c = s / 2
    pts = []
    for i in range(8):
        ang = math.pi / 4 * i - math.pi / 2
        rad = r if i % 2 == 0 else r * 0.2
        pts.append((c + rad * math.cos(ang), c + rad * math.sin(ang)))
    glow = Image.new("L", (s, s), 0)
    ImageDraw.Draw(glow).ellipse((c - r * 0.6, c - r * 0.6, c + r * 0.6, c + r * 0.6), fill=120)
    glow = glow.filter(ImageFilter.GaussianBlur(r * 0.35))
    out = Image.new("RGBA", (s, s), color + (0,))
    out.putalpha(glow)
    star = Image.new("L", (s, s), 0)
    ImageDraw.Draw(star).polygon(pts, fill=255)
    out.paste(Image.new("RGBA", (s, s), color + (255,)), (0, 0), star)
    core = Image.new("L", (s, s), 0)
    ImageDraw.Draw(core).ellipse((c - r * 0.12, c - r * 0.12, c + r * 0.12, c + r * 0.12), fill=255)
    out.paste(Image.new("RGBA", (s, s), (255, 255, 255, 255)), (0, 0), core)
    return out


def bat_image(flap, size=220):
    """Bat silhouette; flap in [-1, 1] folds the wings up/down."""
    w, h = size, int(size * 0.62)
    cx, cy = w / 2, h * 0.48
    wing = [(0.06, -0.04), (0.2, -0.16), (0.34, -0.22), (0.47, -0.12), (0.43, 0.02), (0.36, -0.01),
            (0.31, 0.12), (0.22, 0.04), (0.15, 0.17), (0.08, 0.06)]
    k = 0.25 + 0.75 * (0.5 + 0.5 * flap)
    pts_r = [(cx + x * w, cy + y * h * 2.2 * (k if y < 0 else 0.6 + 0.4 * k) - (1 - k) * x * h * 0.9) for x, y in wing]
    pts_l = [(2 * cx - x, y) for x, y in pts_r]
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    col = (22, 8, 32, 255)
    d.polygon([(cx, cy - 6)] + pts_r + [(cx, cy + 10)], fill=col)
    d.polygon([(cx, cy - 6)] + pts_l + [(cx, cy + 10)], fill=col)
    d.ellipse((cx - 13, cy - 16, cx + 13, cy + 22), fill=col)
    d.polygon([(cx - 11, cy - 10), (cx - 9, cy - 28), (cx - 2, cy - 14), (cx + 2, cy - 14), (cx + 9, cy - 28),
               (cx + 11, cy - 10)], fill=col)
    for ex in (-5, 5):
        d.ellipse((cx + ex - 2.5, cy - 9, cx + ex + 2.5, cy - 4), fill=(255, 200, 60, 255))
    return img


def fit_width(img, maxw):
    if img.width <= maxw:
        return img
    return img.resize((maxw, round(img.height * maxw / img.width)), Image.LANCZOS)


def logo_image(width):
    im = Image.open(ROOT / "brand" / "logo.png").convert("RGBA")
    im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    rgb = im.convert("RGB").filter(ImageFilter.UnsharpMask(2, 80, 2))
    out = rgb.convert("RGBA")
    out.putalpha(im.getchannel("A"))
    # white sticker outline + shadow so it reads on photos
    pad = 26
    a = Image.new("L", (out.width + 2 * pad, out.height + 2 * pad), 0)
    a.paste(im.getchannel("A"), (pad, pad))
    ring = a.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.GaussianBlur(1.5))
    sh = a.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.GaussianBlur(10)).point(lambda v: v * 0.5)
    base = Image.new("RGBA", a.size, (10, 0, 20, 0))
    base.putalpha(sh)
    white = Image.new("RGBA", a.size, (255, 255, 255, 0))
    white.putalpha(ring)
    base = Image.alpha_composite(base, white)
    base.alpha_composite(out, (pad, pad))
    return base


# ---------- compositing ----------
def blit(frame, spr, cx, cy, scale=1.0, rot=0.0, alpha=1.0):
    if scale <= 0.01 or alpha <= 0.004:
        return
    h, w = spr.shape[:2]
    if abs(scale - 1) < 1e-3 and abs(rot) < 1e-3:
        warped = spr
    else:
        rad = math.radians(rot)
        bw = int(abs(w * scale * math.cos(rad)) + abs(h * scale * math.sin(rad))) + 2
        bh = int(abs(w * scale * math.sin(rad)) + abs(h * scale * math.cos(rad))) + 2
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, scale)
        M[0, 2] += bw / 2 - w / 2
        M[1, 2] += bh / 2 - h / 2
        warped = cv2.warpAffine(spr, M, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    hh, ww = warped.shape[:2]
    x0, y0 = int(round(cx - ww / 2)), int(round(cy - hh / 2))
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(frame.shape[1], x0 + ww), min(frame.shape[0], y0 + hh)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    s = warped[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    region = frame[fy0:fy1, fx0:fx1]
    a = s[..., 3:4] * alpha
    region *= (1 - a)
    region += s[..., :3] * alpha


def grade(img):
    """Gentle Halloween grade: a touch more contrast and colour, warm highlights, purple-leaning shadows."""
    x = img.astype(np.float32) / 255.0
    x = np.clip((x - 0.5) * 1.07 + 0.5, 0, 1)
    lum = (x @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    x = lum + (x - lum) * 1.14
    x[..., 0] += 0.025 * lum[..., 0]
    x[..., 2] += 0.035 * (1 - lum[..., 0]) * (1 - lum[..., 0])
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def make_overlay():
    """Darkening for legibility: top band (labels), bottom band (app UI), soft vignette."""
    y = np.arange(H, dtype=np.float32)[:, None]
    x = np.arange(W, dtype=np.float32)[None, :]
    top = 0.62 * (1 - np.clip(y / 640, 0, 1)) ** 1.8
    bottom = 0.5 * np.clip((y - 1430) / 490, 0, 1) ** 1.6
    r = np.sqrt(((x - W / 2) / (W * 0.62)) ** 2 + ((y - H * 0.52) / (H * 0.6)) ** 2)
    vig = 0.32 * np.clip(r - 0.55, 0, 1) ** 1.5 / 0.45 ** 1.5
    a = 1 - (1 - top) * (1 - bottom) * (1 - np.clip(vig, 0, 0.32))
    return a[..., None].astype(np.float32)


# ---------- photos ----------
@lru_cache(maxsize=1)
def faces():
    return json.loads(FACES_JSON.read_text())


@lru_cache(maxsize=None)
def jpg_width(name):
    with Image.open(JPG / f"{name}.jpg") as im:
        return im.width


class Photo:
    """A face-blurred photo, framed for 9:16 around the kid(s), graded and pre-scaled for warping."""

    def __init__(self, shot):
        name = shot["photo"]
        img = cv2.cvtColor(cv2.imread(str(BLURRED / f"{name}.jpg")), cv2.COLOR_BGR2RGB)
        hs, ws = img.shape[:2]
        k = ws / jpg_width(name)
        boxes = np.array([shot["frame_box"]] if shot.get("frame_box") else faces()[name], np.float64) * k
        fx = boxes[:, 0] + boxes[:, 2] / 2
        fy = boxes[:, 1] + boxes[:, 3] / 2
        fh = float(np.median(boxes[:, 3]))
        spread = float((boxes[:, 0] + boxes[:, 2]).max() - boxes[:, 0].min())
        win_h = max(11.0 * fh, (spread + 5 * fh) * 16 / 9) * shot.get("zoom", 1.0)
        # leave 10% spare so the Ken Burns zoom-out never runs off the photo
        win_h = min(win_h, hs / 1.1, ws * 16 / 9 / 1.1)
        face_pos = shot.get("face_pos", 0.40)
        cx = (fx.min() + fx.max()) / 2 + shot.get("dx", 0.0) * win_h * 9 / 16
        cy = float(fy.mean()) + (0.5 - face_pos) * win_h
        # pre-scale so the largest window used maps close to 1:1 onto the frame
        f = min(1.0, (H * 1.12) / (win_h * 0.86))
        if f < 1:
            img = cv2.resize(img, (round(ws * f), round(hs * f)), interpolation=cv2.INTER_AREA)
        self.img = grade(img)
        self.f = f
        self.win_h = win_h * f
        self.hs, self.ws = self.img.shape[:2]
        self.cx, self.cy = self.clamp(cx * f, cy * f, self.win_h)

    def clamp(self, cx, cy, wh):
        ww = wh * 9 / 16
        cx = min(max(cx, ww / 2), self.ws - ww / 2) if ww < self.ws else self.ws / 2
        cy = min(max(cy, wh / 2), self.hs - wh / 2) if wh < self.hs else self.hs / 2
        return cx, cy

    def render(self, zoom=1.0, dx=0.0, offset_x=0.0, motion_blur=0):
        """zoom > 1 shows more of the photo; dx shifts the window (fraction of its width);
        offset_x moves the picture on screen (pixels, for whip transitions)."""
        wh = self.win_h * zoom
        cx, cy = self.clamp(self.cx + dx * wh * 9 / 16, self.cy, wh)
        s = H / wh
        M = np.array([[s, 0, W / 2 - s * cx + offset_x], [0, s, H / 2 - s * cy]], np.float32)
        out = cv2.warpAffine(self.img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
        if motion_blur > 1:
            out = cv2.blur(out, (int(motion_blur), 1))
        return out


# ---------- timeline ----------
def build_timeline():
    segs = [{"kind": "title", "b0": 0, "b1": TITLE_BEATS, "shot": SHOTS[0]}]
    b = TITLE_BEATS
    for i, s in enumerate(SHOTS):
        segs.append({"kind": "shot", "b0": b, "b1": b + s["beats"], "shot": s, "num": i + 1,
                     "tin": "reveal" if i == 0 else ("whip" if s["beats"] >= 4 else "punch")})
        b += s["beats"]
    for j, s in enumerate(FAMILY["shots"]):
        segs.append({"kind": "family", "b0": b, "b1": b + s["beats"], "shot": s, "fam": j, "tin": "whip"})
        b += s["beats"]
    segs.append({"kind": "end", "b0": b, "b1": b + END_BEATS, "tin": "fade"})
    for i, sg in enumerate(segs):
        sg["t0"], sg["t1"] = bt(sg["b0"]), bt(sg["b1"])
        sg["i"] = i
    segs[-1]["t1"] += TAIL
    return segs


class Renderer:
    def __init__(self):
        self.segs = build_timeline()
        self.total = self.segs[-1]["t1"]
        self.overlay = make_overlay()
        self.deep = np.array(DEEP, np.float32) / 255
        self.photos = {}
        self.labels = {sg["i"]: to_sprite(label_sprite(sg["num"], sg["shot"]["label"]))
                       for sg in self.segs if sg["kind"] == "shot"}
        self.logo_small = to_sprite(logo_image(170))
        self.logo_big = to_sprite(logo_image(330))
        self.title_logo = to_sprite(logo_image(300))
        self.t_big = to_sprite(fit_width(text_image(TITLE["big"].upper(), font("Creepster-Regular.ttf", 236),
                                                    fill=(GOLD, ORANGE), stroke_w=10, extrude=16,
                                                    extrude_col=PURPLE), 1060))
        self.t_sub = to_sprite(fit_width(text_image(TITLE["sub"].upper(), font("LuckiestGuy-Regular.ttf", 112),
                                                    fill=(WHITE, CREAM), stroke_w=9, extrude=12,
                                                    extrude_col=MAGENTA), 1040))
        cta = text_image(TITLE["cta"], font("Baloo2.ttf", 60, [800]), fill=(WHITE, WHITE), stroke_w=0,
                         shadow=0, pad=2)
        self.t_cta = to_sprite(pill_image(cta, [VIOLET, MAGENTA], pad_x=40, pad_y=8))
        fam = Image.new("RGBA", (1, 1))
        lines = ["EVEN THE GROWN-UPS", "DRESSED UP!"]
        imgs = [text_image(t, font("LuckiestGuy-Regular.ttf", 76), fill=(WHITE, CREAM), stroke_w=7, extrude=6,
                           extrude_col=INK, shadow=0, pad=6) for t in lines]
        fam = Image.new("RGBA", (max(i.width for i in imgs), sum(i.height for i in imgs) - 10), (0, 0, 0, 0))
        y = 0
        for im in imgs:
            fam.alpha_composite(im, ((fam.width - im.width) // 2, y))
            y += im.height - 10
        self.fam_banner = to_sprite(pill_image(fam, [ORANGE, MAGENTA], pad_x=44, pad_y=18, radius=60))
        self.build_end()
        self.sparkles = [to_sprite(sparkle_image(r, c)) for r, c in
                         [(26, GOLD), (18, CREAM), (22, (255, 150, 205)), (14, GOLD), (30, CREAM)]]
        self.bats = [to_sprite(bat_image(math.cos(2 * math.pi * i / 8))) for i in range(8)]
        rng = np.random.default_rng(7)
        self.side_sparks = []
        for i in range(16):
            x = rng.uniform(30, 190) if i % 2 == 0 else rng.uniform(890, 1050)
            self.side_sparks.append((x, rng.uniform(560, 1480), rng.uniform(0.35, 0.9), rng.uniform(0.6, 1.4),
                                     rng.uniform(0, 1), int(rng.integers(0, 5))))
        self.all_sparks = [(rng.uniform(40, 1040), rng.uniform(160, 1700), rng.uniform(0.35, 1.0),
                            rng.uniform(0.6, 1.5), rng.uniform(0, 1), int(rng.integers(0, 5))) for _ in range(26)]
        self.bat_flights = self.plan_bats()

    # ----- end card -----
    def build_end(self):
        self.e_happy = to_sprite(text_image("Happy", font("LuckiestGuy-Regular.ttf", 120), fill=(WHITE, CREAM),
                                            stroke_w=9, extrude=10, extrude_col=MAGENTA))
        self.e_hall = to_sprite(fit_width(text_image("HALLOWEEN!", font("Creepster-Regular.ttf", 206),
                                                     fill=(GOLD, ORANGE), stroke_w=10, extrude=14,
                                                     extrude_col=PURPLE), 1060))
        self.e_from = to_sprite(text_image(END["sub"], font("Baloo2.ttf", 52, [800]), fill=(CREAM, CREAM),
                                           stroke_w=0, shadow=0.7, pad=10))
        q = text_image("Which one's your favourite?", font("Baloo2.ttf", 52, [800]), fill=(WHITE, WHITE),
                       stroke_w=0, shadow=0, pad=2)
        c = text_image("COMMENT THE NUMBER!", font("LuckiestGuy-Regular.ttf", 70), fill=(GOLD, (255, 214, 90)),
                       stroke_w=6, extrude=5, extrude_col=INK, shadow=0, pad=4)
        box = Image.new("RGBA", (max(q.width, c.width), q.height + c.height - 4), (0, 0, 0, 0))
        box.alpha_composite(q, ((box.width - q.width) // 2, 0))
        box.alpha_composite(c, ((box.width - c.width) // 2, q.height - 4))
        self.e_cta = to_sprite(pill_image(box, [VIOLET, MAGENTA], pad_x=46, pad_y=16, radius=56))
        self.e_enrol = to_sprite(text_image(END["enrol"].upper(), font("LuckiestGuy-Regular.ttf", 60),
                                            fill=(GOLD, (255, 214, 90)), stroke_w=6, extrude=6, extrude_col=INK,
                                            shadow=0.6, pad=10))
        self.e_phone = to_sprite(text_image(f"{BRAND['phone']}  •  {BRAND['web']}", font("Baloo2.ttf", 56, [800]),
                                            fill=(WHITE, WHITE), stroke_w=0, shadow=0.8, pad=10))
        self.e_addr = to_sprite(text_image(BRAND["address"], font("Baloo2.ttf", 40, [700]), fill=(CREAM, CREAM),
                                           stroke_w=0, shadow=0.8, pad=10))
        self.wall = self.make_wall()

    def make_wall(self):
        """A tilted wall of every costume photo, for the end card background."""
        tw, th, gap = 330, 440, 26
        cols, rows = 5, 8
        canvas = Image.new("RGB", (cols * (tw + gap) + gap, rows * (th + gap) + gap), DEEP)
        names = [s for s in SHOTS] + [s for s in FAMILY["shots"]]
        cache = {}
        for i in range(cols * rows):
            shot = names[(i * 7) % len(names)]
            if shot["photo"] not in cache:
                cache[shot["photo"]] = Photo(shot)
            p = cache[shot["photo"]]
            wh = p.win_h
            ww = wh * 9 / 16
            # tile = the photo's 9:16 window cropped to 3:4 around the kid
            th_src = ww * 4 / 3
            x0, y0 = int(p.cx - ww / 2), int(p.cy - th_src / 2)
            crop = p.img[max(0, y0):y0 + int(th_src), max(0, x0):x0 + int(ww)]
            tile = Image.fromarray(cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA))
            c, r = i % cols, i // cols
            canvas.paste(tile, (gap + c * (tw + gap), gap + r * (th + gap) + (th // 2 if c % 2 else 0) - th // 4))
        big = canvas.rotate(-9, resample=Image.BICUBIC, expand=True, fillcolor=DEEP)
        return np.asarray(big).astype(np.float32) / 255.0

    def plan_bats(self):
        flights = []  # (start, duration, y0, y1, left_to_right, scale)
        flights += [(0.05, 1.6, 380, 230, True, 0.9), (0.35, 1.5, 1480, 1340, False, 0.65)]
        fam = next(s for s in self.segs if s["kind"] == "family")
        flights += [(fam["t0"] + 0.1, 1.4, 700, 520, True, 0.7)]
        end = self.segs[-1]
        flights += [(end["t0"] + 0.2, 2.2, 1500, 1100, True, 0.8), (end["t0"] + 1.4, 2.0, 300, 180, False, 0.55),
                    (end["t0"] + 3.0, 2.4, 1650, 1300, False, 0.7)]
        return flights

    def photo(self, shot):
        key = shot["photo"]
        if key not in self.photos:
            if len(self.photos) > 3:
                self.photos.pop(next(iter(self.photos)))
            self.photos[key] = Photo(shot)
        return self.photos[key]

    # ----- per-frame drawing -----
    def kb(self, sg, t):
        """Ken Burns zoom + drift for a photo segment, plus the punch/whip on the way in and out."""
        p = (t - sg["t0"]) / (sg["t1"] - sg["t0"])
        e = ease_in_out(p)
        zoom = 1.07 - 0.07 * e if sg["i"] % 2 == 0 else 1.0 + 0.06 * e
        dx = (((sg["i"] * 37) % 7) - 3) / 3 * 0.025 * (e - 0.5)
        off, blur = 0.0, 0
        since = t - sg["t0"]
        if sg.get("tin") == "punch":
            zoom *= 1 - 0.085 * (1 - ease_out_cubic(since / 0.3))
        if sg.get("tin") == "whip" and since < 0.15:
            q = 1 - ease_out_cubic(since / 0.15)
            off, blur = W * 0.7 * q, 1 + 140 * q
        nxt = self.segs[sg["i"] + 1] if sg["i"] + 1 < len(self.segs) else None
        until = sg["t1"] - t
        if nxt and nxt.get("tin") == "whip" and until < 0.12:
            q = ease_in_cubic(1 - until / 0.12)
            off, blur = -W * 0.7 * q, 1 + 140 * q
        return zoom, dx, off, blur

    def flash(self, sg, t):
        since = t - sg["t0"]
        if sg.get("tin") == "punch":
            return 0.30 * math.exp(-since / 0.05)
        if sg.get("tin") == "whip":
            return 0.18 * math.exp(-max(0, since - 0.1) / 0.06) if since > 0.08 else 0.0
        return 0.0

    def draw_sparks(self, frame, t, sparks, strength=1.0):
        for x, y, sc, fq, ph, kind in sparks:
            tw = math.sin(2 * math.pi * (t * fq * 0.7 + ph))
            a = max(0.0, tw) ** 2 * strength
            if a < 0.02:
                continue
            yy = (y - 22 * t) % 1600 + 150
            blit(frame, self.sparkles[kind], x, yy, sc * (0.6 + 0.4 * a), 20 * t * fq, a)

    def draw_bats(self, frame, t):
        for t0, dur, y0, y1, ltr, sc in self.bat_flights:
            p = (t - t0) / dur
            if not 0 <= p <= 1:
                continue
            x = -160 + (W + 320) * p
            x = x if ltr else W - x
            y = y0 + (y1 - y0) * p + 40 * math.sin(p * 6.5)
            spr = self.bats[int(t * 22) % len(self.bats)]
            blit(frame, spr, x, y, sc * (0.85 + 0.3 * p), (-12 if ltr else 12) + 8 * math.sin(p * 9), 1.0)

    def base_photo(self, sg, t):
        ph = self.photo(sg["shot"])
        zoom, dx, off, blur = self.kb(sg, t)
        img = ph.render(zoom, dx, off, blur).astype(np.float32) / 255.0
        img *= (1 - self.overlay)
        img += self.deep * self.overlay
        return img

    def frame(self, t):
        sg = next(s for s in self.segs if s["t0"] <= t < s["t1"]) if t < self.total else self.segs[-1]
        kind = sg["kind"]
        if kind == "title":
            return self.frame_title(sg, t)
        if kind == "end":
            return self.frame_end(sg, t)
        fr = self.base_photo(sg, t)
        self.draw_sparks(fr, t, self.side_sparks, 0.85)
        since, until = t - sg["t0"], sg["t1"] - t
        if kind == "shot":
            lab = self.labels[sg["i"]]
            p = (since - 0.04) / 0.34
            sc = min(1.0, 990 / lab.shape[1]) * (ease_out_back(p) if p < 1 else 1.0)
            out = clamp01(1 - until / 0.07)
            rot = -2.0 + 6 * (1 - ease_out_cubic(p)) + 0.8 * math.sin(2 * math.pi * since / 1.4)
            blit(fr, lab, W / 2 + 10, 395 + 5 * math.sin(2 * math.pi * since / 1.1), sc * (1 - 0.1 * out),
                 rot, clamp01(p * 3) * (1 - out))
        else:
            p = (t - self.segs[[s["kind"] for s in self.segs].index("family")]["t0"] - 0.05) / 0.4
            sc = ease_out_back(p) if p < 1 else 1.0
            blit(fr, self.fam_banner, W / 2, 400 + 5 * math.sin(2 * math.pi * t / 1.2), sc,
                 -1.5 + 0.8 * math.sin(2 * math.pi * t / 1.5), clamp01(p * 3))
        blit(fr, self.logo_small, 118, 228, 1.0, 0, 0.96)
        self.draw_bats(fr, t)
        fl = self.flash(sg, t)
        if fl > 0.003:
            fr += (1 - fr) * fl
        self.prev_frame = fr
        return fr

    def frame_title(self, sg, t):
        first = next(s for s in self.segs if s["kind"] == "shot")
        ph = self.photo(first["shot"])
        z0, dx0, _, _ = self.kb(first, first["t0"])  # end exactly where the first costume shot starts
        sharp = ph.render(z0 * (1.10 - 0.10 * ease_in_out(t / sg["t1"])), dx0, 0, 0)
        small = cv2.resize(sharp, (W // 6, H // 6), interpolation=cv2.INTER_AREA)
        soft = cv2.resize(cv2.GaussianBlur(small, (0, 0), 6), (W, H), interpolation=cv2.INTER_LINEAR)
        reveal = ease_in_out((t - (sg["t1"] - 0.32)) / 0.32)
        img = (soft.astype(np.float32) * (1 - reveal) + sharp.astype(np.float32) * reveal) / 255.0
        dim = 0.62 * (1 - reveal)
        img = img * (1 - dim) + self.deep * dim
        img *= (1 - self.overlay * reveal)
        img += self.deep * self.overlay * reveal
        self.draw_sparks(img, t, self.all_sparks, 1.0 - reveal)
        b = lambda k: k * BEAT  # noqa: E731 (beats from the start of the title)
        exitp = ease_in_cubic((t - (sg["t1"] - 0.3)) / 0.3)
        grow, fade = 1 + 0.35 * exitp, 1 - exitp
        float_y = 6 * math.sin(2 * math.pi * t / 1.3)
        p = (t - 0.0) / 0.38
        blit(img, self.title_logo, W / 2, 580 + float_y, ease_out_back(p) * grow, -3 + 3 * ease_out_cubic(p),
             clamp01(p * 3) * fade)
        p = (t - b(0.6)) / 0.4
        y = 870 - 260 * (1 - ease_out_back(p, 1.4))
        blit(img, self.t_big, W / 2, y, (0.9 + 0.1 * ease_out_cubic(p)) * grow, -2, clamp01(p * 2.5) * fade)
        p = (t - b(1.6)) / 0.34
        blit(img, self.t_sub, W / 2, 1070, ease_out_back(p) * grow, 1.5, clamp01(p * 3) * fade)
        p = (t - b(2.5)) / 0.32
        blit(img, self.t_cta, W / 2, 1230 + 60 * (1 - ease_out_cubic(p)), grow, 0, clamp01(p * 2.5) * fade)
        self.draw_bats(img, t)
        return img

    def frame_end(self, sg, t):
        since = t - sg["t0"]
        wall = self.wall
        wh, ww = wall.shape[:2]
        y0 = int(wh * 0.18 + 60 * since)
        x0 = int((ww - W) / 2)
        img = wall[y0:y0 + H, x0:x0 + W].copy()
        img = img * 0.34 + self.deep * 0.66
        # fade in from the last family shot
        if since < 0.35:
            last = getattr(self, "prev_frame", None)
            if last is None:  # rendering this frame on its own (QA stills)
                prev = self.segs[sg["i"] - 1]
                last = self.base_photo(prev, prev["t1"] - 1e-3)
            k = ease_in_out(since / 0.35)
            img = last * (1 - k) + img * k
        self.draw_sparks(img, t, self.all_sparks, 1.0)
        fl = 5 * math.sin(2 * math.pi * since / 1.4)
        items = [  # (sprite, y, start, floats)
            (self.logo_big, 400, 0.10, True), (self.e_happy, 700, 0.35, False), (self.e_hall, 850, 0.50, True),
            (self.e_from, 1000, 0.75, False), (self.e_cta, 1160, 1.15, False), (self.e_enrol, 1330, 1.70, False),
            (self.e_phone, 1410, 1.90, False), (self.e_addr, 1480, 2.05, False)]
        for spr, y, st, floats in items:
            p = (since - st) / 0.36
            if p <= 0:
                continue
            sc = ease_out_back(p, 1.6) if p < 1 else 1.0
            if spr is self.logo_big:
                sc *= 1 + 0.02 * math.sin(2 * math.pi * since / 1.6)
            blit(img, spr, W / 2, y + (fl if floats else 0), sc, 0, clamp01(p * 3))
        self.draw_bats(img, t)
        return img


def music_mix(dur):
    """Cut the track to the reel's length, fade it out, and normalise to -14 LUFS in two passes."""
    shape = (f"atrim=0:{dur:.3f},asetpts=PTS-STARTPTS,afade=t=in:d=0.02,"
             f"afade=t=out:st={dur - 1.7:.3f}:d=1.7")
    target = "I=-14:TP=-1.5:LRA=11"
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(MUSIC), "-af",
                            f"{shape},loudnorm={target}:print_format=json", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
    m = json.loads(probe[probe.rindex("{"):probe.rindex("}") + 1])
    out = ROOT / "work" / "music_mix.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(MUSIC), "-af",
                    f"{shape},loudnorm={target}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
                    f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
                    f"offset={m['target_offset']}:linear=true,aresample=48000",
                    "-ar", "48000", "-c:a", "pcm_s16le", str(out)], check=True)
    return out


def encode(r, out_path, music=True, preview=False):
    step = 3 if preview else 1
    w, h = (W // 2, H // 2) if preview else (W, H)
    fps = FPS / step
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-"]
    dur = r.total
    if music:
        cmd += ["-i", str(music_mix(dur))]
    else:
        cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
    cmd += ["-filter_complex", "[0:v]scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[v]",
            "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", "medium" if preview else "slow",
            "-crf", "24" if preview else "16", "-profile:v", "high", "-g", str(int(fps * 2)),
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-bsf:v", "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{dur:.3f}",
            "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(math.ceil(dur * FPS))
    for i in range(0, n, step):
        fr = r.frame(i / FPS)
        u8 = (np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8)
        if preview:
            u8 = cv2.resize(u8, (w, h), interpolation=cv2.INTER_AREA)
        proc.stdin.write(u8.tobytes())
        if i % 150 == 0:
            print(f"  frame {i}/{n}", flush=True)
    proc.stdin.close()
    proc.wait()
    if proc.returncode:
        raise SystemExit(f"ffmpeg failed ({proc.returncode})")
    print("wrote", out_path)


def main():
    preview = "--preview" in sys.argv
    music = "--no-music" not in sys.argv
    EXPORTS.mkdir(exist_ok=True)
    r = Renderer()
    print(f"{len(r.segs)} segments, {r.total:.2f} s")
    if preview:
        out = ROOT / "work" / "preview.mp4"
    else:
        out = EXPORTS / ("tinystars_halloween_reel.mp4" if music else "tinystars_halloween_reel_nomusic.mp4")
    encode(r, out, music, preview)


if __name__ == "__main__":
    main()
