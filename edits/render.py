"""Render the Tiny Stars Halloween costume-parade reel: 1080x1920, 30 fps, cut on the beat.

Timeline:
  cold open (thunder, no music: "Something spooky happened at..." + logo over the haunted-house clip)
  -> title slam on the first beat (HALLOWEEN / COSTUME PARADE)
  -> 22 numbered costumes, each joined by a horror transition (lightning, glitch, slime drip, ghost,
     flicker, fog, zoom punch)
  -> "even the grown-ups dressed up" family shots
  -> numbered vote grid ("which one wins? comment the number")
  -> closing card on the haunted classroom: logo, THANKS FOR WATCHING, contact stickers from the
     Tiny Stars "Thanks for Watching" card, all flickering, swaying and glitching in

Usage: python -I edits/render.py [--no-music]
  -> exports/tinystars_halloween_reel.mp4 (music + sound effects) or ..._nomusic.mp4 (sound effects only)
"""
import json
import math
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content import COLD, FAMILY, OUTRO, SHOTS, TITLE, VOTE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "brand" / "fonts"
BLURRED = ROOT / "work" / "blurred"
JPG = ROOT / "work" / "jpg"
STICKERS = ROOT / "work" / "stickers"
FACES_JSON = ROOT / "work" / "faces.json"
OPENGEN = ROOT / "source" / "opengen"
SFX = ROOT / "source" / "sfx"
MUSIC = ROOT / "source" / "music" / "ghosts_of_the_past_171666.mp3"
EXPORTS = ROOT / "exports"

W, H, FPS = 1080, 1920, 30
BEAT = 60 / 128          # the track is a steady 128 BPM
T0 = 0.0704              # first beat in the music file (seconds); the music starts here
COLD_LEN = 2.4           # cold open before the music
TITLE_BEATS = 4
TAIL = 1.2               # hold the closing card while the music fades

# brand palette (RGB)
DEEP = (18, 6, 28)
INK = (38, 9, 56)
PURPLE = (100, 42, 115)
SLIME = (74, 22, 108)
VIOLET = (98, 75, 255)
MAGENTA = (232, 62, 140)
ORANGE = (255, 122, 26)
BLOOD = (214, 58, 20)
GOLD = (246, 199, 37)
CREAM = (255, 246, 224)
WHITE = (255, 255, 255)
GHOST = (210, 230, 255)

# fonts: dripping horror for the big words, horror-movie lettering for names, clean for small print
F_DRIP = "Nosifer-Regular.ttf"
F_HORROR = "Eater-Regular.ttf"
F_WET = "RubikWetPaint-Regular.ttf"
F_BODY = "Baloo2.ttf"


def bt(k):
    """Video time of beat k; beat 0 is the title slam right after the cold open."""
    return COLD_LEN + k * BEAT


# ---------- easing / noise ----------
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


def hash01(*k):
    """Deterministic pseudo-random number in [0, 1) from integers."""
    h = 2166136261
    for v in k:
        h = ((h ^ (int(v) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 1274126177) & 0xFFFFFFFF
    return (h & 0xFFFFFF) / float(0x1000000)


def flicker(t, seed, base=1.0, depth=0.85, rate=24):
    """A failing-bulb flicker: mostly on, with sudden dips."""
    f = int(t * rate)
    r = hash01(f, seed)
    return base * (1 - depth) if r < 0.18 else base * (0.75 + 0.25 * hash01(f, seed, 7))


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


def fit_width(img, maxw):
    if img.width <= maxw:
        return img
    return img.resize((maxw, round(img.height * maxw / img.width)), Image.LANCZOS)


def text_image(text, f, fill=(WHITE, CREAM), stroke=INK, stroke_w=8, extrude=0, extrude_col=PURPLE,
               shadow=0.55, pad=40, glow=None):
    """Lettering with gradient fill, outline, optional 3D extrude, soft shadow and an optional coloured glow."""
    gpad = glow[1] * 2 if glow else 0
    pad += gpad
    l, t, r, b = f.getbbox(text, stroke_width=stroke_w)
    w, h = r - l + 2 * pad, b - t + 2 * pad + extrude
    ox, oy = pad - l, pad - t
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if glow:
        col, rad, strength = glow
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).text((ox, oy), text, font=f, fill=255, stroke_width=stroke_w + 4, stroke_fill=255)
        m = m.filter(ImageFilter.GaussianBlur(rad)).point(lambda v: min(255, int(v * strength)))
        g = Image.new("RGBA", (w, h), col + (0,))
        g.putalpha(m)
        out = Image.alpha_composite(out, g)
    if shadow:
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).text((ox, oy + extrude + 10), text, font=f, fill=255, stroke_width=stroke_w, stroke_fill=255)
        m = m.filter(ImageFilter.GaussianBlur(12)).point(lambda v: int(v * shadow))
        sh = Image.new("RGBA", (w, h), (6, 0, 12, 0))
        sh.putalpha(m)
        out = Image.alpha_composite(out, sh)
    d = ImageDraw.Draw(out)
    dark = tuple(int(c * 0.55) for c in extrude_col)
    for i in range(extrude, 0, -1):
        d.text((ox, oy + i), text, font=f, fill=extrude_col if i > 2 else dark,
               stroke_width=stroke_w, stroke_fill=dark if i > 2 else INK)
    if stroke_w:
        d.text((ox, oy), text, font=f, fill=stroke, stroke_width=stroke_w, stroke_fill=stroke)
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).text((ox, oy), text, font=f, fill=255)
    _, ty0, _, ty1 = f.getbbox(text)
    out.paste(vgrad(w, h, fill[0], fill[1], oy + ty0, oy + ty1), (0, 0), m)
    return out


def stack(images, gap=-8):
    w = max(i.width for i in images)
    h = sum(i.height for i in images) + gap * (len(images) - 1)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    y = 0
    for im in images:
        out.alpha_composite(im, ((w - im.width) // 2, y))
        y += im.height + gap
    return out


def drippy_pill(content, fill=SLIME, pad_x=40, pad_y=18, left_room=0, seed=1, drips=True, outline=CREAM):
    """A pill like the ones on the Tiny Stars Halloween card: dark purple with slime drips and a cream outline."""
    cw, ch = content.size
    pw, ph = cw + 2 * pad_x + left_room, ch + 2 * pad_y
    r = ph // 2
    m = 34
    W_, H_ = pw + 2 * m, ph + 2 * m + 70
    mask = Image.new("L", (W_, H_), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle((m, m, m + pw, m + ph), r, fill=255)
    rng = np.random.default_rng(seed)
    if drips:
        xs = np.sort(rng.uniform(m + r, m + pw - r, max(4, pw // 70)))
        for x in xs:
            dw = rng.uniform(12, 24)
            dl = rng.uniform(16, 58)
            y0 = m + ph - 6
            d.rectangle((x - dw / 2, y0, x + dw / 2, y0 + dl - dw / 2), fill=255)
            d.ellipse((x - dw / 2, y0 + dl - dw, x + dw / 2, y0 + dl), fill=255)
            # little fillets so drips flow out of the edge
            d.ellipse((x - dw, y0 - dw * 0.4, x + dw, y0 + dw * 0.6), fill=255)
    out = Image.new("RGBA", (W_, H_), (0, 0, 0, 0))
    ring = mask.filter(ImageFilter.MaxFilter(11))
    sh = ring.filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.6))
    shadow = Image.new("RGBA", (W_, H_), (4, 0, 10, 0))
    shadow.putalpha(sh)
    out = Image.alpha_composite(out, shadow.transform(out.size, Image.AFFINE, (1, 0, 0, 0, 1, -8)))
    col = Image.new("RGBA", (W_, H_), outline + (255,))
    out.paste(col, (0, 0), ring)
    body = vgrad(W_, H_, tuple(min(255, int(c * 1.35)) for c in fill), tuple(int(c * 0.7) for c in fill), m, m + ph + 50)
    out.paste(body, (0, 0), mask)
    gloss = Image.new("L", (W_, H_), 0)
    ImageDraw.Draw(gloss).rounded_rectangle((m + 14, m + 7, m + pw - 14, m + ph * 0.45), r, fill=46)
    out = Image.alpha_composite(out, Image.merge("RGBA", (*[Image.new("L", out.size, 255)] * 3, gloss)))
    out.alpha_composite(content, (m + pad_x + left_room, m + pad_y))
    return out


def pumpkin_badge(num, d=150):
    """Orange jack-o-lantern style badge with the costume number."""
    s = d + 60
    out = Image.new("RGBA", (s, s + 20), (0, 0, 0, 0))
    m = Image.new("L", out.size, 0)
    md = ImageDraw.Draw(m)
    cx, cy = s / 2, s / 2 + 12
    for dx, rw in ((-d * 0.26, 0.62), (d * 0.26, 0.62), (0, 0.7)):
        md.ellipse((cx + dx - d * rw / 2, cy - d * 0.46, cx + dx + d * rw / 2, cy + d * 0.46), fill=255)
    ring = m.filter(ImageFilter.MaxFilter(9))
    sh = ring.filter(ImageFilter.GaussianBlur(9)).point(lambda v: int(v * 0.6))
    shadow = Image.new("RGBA", out.size, (4, 0, 10, 0))
    shadow.putalpha(sh)
    out = Image.alpha_composite(out, shadow.transform(out.size, Image.AFFINE, (1, 0, 0, 0, 1, -7)))
    out.paste(Image.new("RGBA", out.size, (30, 8, 40, 255)), (0, 0), ring)
    out.paste(vgrad(s, s + 20, (255, 170, 60), (220, 80, 10), int(cy - d * 0.46), int(cy + d * 0.46)), (0, 0), m)
    dd = ImageDraw.Draw(out)
    for dx in (-d * 0.26, d * 0.26):
        dd.arc((cx + dx - d * 0.3, cy - d * 0.44, cx + dx + d * 0.3, cy + d * 0.44), 100, 260, fill=(200, 70, 10), width=4)
    dd.rounded_rectangle((cx - 9, cy - d * 0.62, cx + 9, cy - d * 0.4), 6, fill=(70, 120, 40), outline=(30, 8, 40), width=3)
    n = text_image(str(num), font(F_HORROR, int(d * 0.5)), fill=(WHITE, CREAM), stroke=(40, 8, 30), stroke_w=5,
                   shadow=0, pad=4)
    out.alpha_composite(n, (int(cx - n.width / 2), int(cy - n.height / 2 + 4)))
    return out


def label_sprite(num, text):
    name = text_image(text.upper(), font(F_HORROR, 64), fill=(GOLD, ORANGE), stroke=INK, stroke_w=5,
                      shadow=0, pad=6)
    badge = pumpkin_badge(num, 150)
    pill = drippy_pill(name, SLIME, pad_x=30, pad_y=14, left_room=96, seed=num)
    out = Image.new("RGBA", (pill.width + 30, max(pill.height, badge.height) + 10), (0, 0, 0, 0))
    out.alpha_composite(pill, (30, (out.height - pill.height) // 2 + 14))
    out.alpha_composite(badge, (0, (out.height - badge.height) // 2 - 6))
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
    col = (10, 4, 16, 255)
    d.polygon([(cx, cy - 6)] + pts_r + [(cx, cy + 10)], fill=col)
    d.polygon([(cx, cy - 6)] + pts_l + [(cx, cy + 10)], fill=col)
    d.ellipse((cx - 13, cy - 16, cx + 13, cy + 22), fill=col)
    d.polygon([(cx - 11, cy - 10), (cx - 9, cy - 28), (cx - 2, cy - 14), (cx + 2, cy - 14), (cx + 9, cy - 28),
               (cx + 11, cy - 10)], fill=col)
    for ex in (-5, 5):
        d.ellipse((cx + ex - 2.5, cy - 9, cx + ex + 2.5, cy - 4), fill=(255, 60, 30, 255))
    return img


def cobweb_image(size=330):
    """A corner cobweb (anchored top-right)."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = size, 0
    col = (235, 235, 245, 150)
    angs = np.linspace(math.pi / 2, math.pi, 7)
    for a in angs:
        d.line((cx, cy, cx + size * math.cos(a) * 1.05, cy + size * math.sin(a) * 1.05), fill=col, width=2)
    for r in np.linspace(40, size * 0.95, 7):
        pts = []
        for a in np.linspace(math.pi / 2, math.pi, 7):
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            sag = r * 0.06
            d.line([(x1, y1), (mx + sag * 0.5, my - sag * 0.3), (x2, y2)], fill=col, width=2, joint="curve")
    return img.filter(ImageFilter.GaussianBlur(0.4))


def bolt_image(seed, length=760):
    """A jagged lightning bolt with a few branches and a glow."""
    rng = np.random.default_rng(seed)
    w, h = 520, length + 40
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    core = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(core)

    def branch(x, y, n, step, width):
        pts = [(x, y)]
        for _ in range(n):
            x += rng.uniform(-38, 38)
            y += step * rng.uniform(0.7, 1.2)
            pts.append((x, y))
            if rng.random() < 0.18 and width > 2:
                branch(x, y, n // 3, step * 0.8, width - 2)
        d.line(pts, fill=255, width=width, joint="curve")

    branch(w / 2, 10, 16, length / 16, 7)
    glow = core.filter(ImageFilter.GaussianBlur(14)).point(lambda v: min(255, v * 3))
    img.paste(Image.new("RGBA", (w, h), (170, 150, 255, 255)), (0, 0), glow)
    img.paste(Image.new("RGBA", (w, h), (255, 255, 255, 255)), (0, 0), core)
    return img


def sticker(name, width=None):
    im = Image.open(STICKERS / f"{name}.png").convert("RGBA")
    if width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    return im


def logo_image(width):
    """The high-resolution Tiny Stars logo sticker (cut from their latest card), falling back to logo.png."""
    if (STICKERS / "logo.png").exists():
        return sticker("logo", width)
    im = Image.open(ROOT / "brand" / "logo.png").convert("RGBA")
    return im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)


# ---------- compositing ----------
def blit(frame, spr, cx, cy, scale=1.0, rot=0.0, alpha=1.0, pivot=None):
    """Draw a premultiplied sprite centred at (cx, cy). pivot=(px, py) in sprite coords swings it around that point."""
    if scale <= 0.01 or alpha <= 0.004:
        return
    h, w = spr.shape[:2]
    if abs(scale - 1) < 1e-3 and abs(rot) < 1e-3:
        warped, ox, oy = spr, w / 2, h / 2
    else:
        rad = math.radians(rot)
        bw = int(abs(w * scale * math.cos(rad)) + abs(h * scale * math.sin(rad))) + 4
        bh = int(abs(w * scale * math.sin(rad)) + abs(h * scale * math.cos(rad))) + 4
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, scale)
        M[0, 2] += bw / 2 - w / 2
        M[1, 2] += bh / 2 - h / 2
        warped = cv2.warpAffine(spr, M, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        ox, oy = bw / 2, bh / 2
        if pivot is not None:
            # keep the pivot point fixed: move the centre by how far the pivot travelled
            px, py = pivot[0] - w / 2, pivot[1] - h / 2
            rx = scale * (px * math.cos(rad) + py * math.sin(rad))
            ry = scale * (-px * math.sin(rad) + py * math.cos(rad))
            cx += px * scale - rx
            cy += py * scale - ry
    hh, ww = warped.shape[:2]
    x0, y0 = int(round(cx - ox)), int(round(cy - oy))
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
    """Moody Halloween grade: more contrast, warm highlights, cold purple shadows."""
    x = img.astype(np.float32) / 255.0
    x = np.clip((x - 0.5) * 1.1 + 0.48, 0, 1)
    lum = (x @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    x = lum + (x - lum) * 1.12
    sh = (1 - lum[..., 0]) ** 2
    x[..., 0] += 0.03 * lum[..., 0] - 0.01 * sh
    x[..., 1] -= 0.012 * sh
    x[..., 2] += 0.05 * sh
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def make_overlay(top=0.6, bottom=0.5, vig=0.45):
    """Darkening for legibility: top band (labels), bottom band (app UI), heavy horror vignette."""
    y = np.arange(H, dtype=np.float32)[:, None]
    x = np.arange(W, dtype=np.float32)[None, :]
    t = top * (1 - np.clip(y / 640, 0, 1)) ** 1.8
    b = bottom * np.clip((y - 1430) / 490, 0, 1) ** 1.6
    r = np.sqrt(((x - W / 2) / (W * 0.6)) ** 2 + ((y - H * 0.52) / (H * 0.58)) ** 2)
    v = vig * np.clip(r - 0.45, 0, 1) ** 1.4 / 0.55 ** 1.4
    a = 1 - (1 - t) * (1 - b) * (1 - np.clip(v, 0, vig))
    return a[..., None].astype(np.float32)


def periodic_noise(h, w, seed, octaves=((6, 1.0), (14, 0.5), (32, 0.25))):
    """Tileable cloudy noise in [0, 1] (FFT-filtered, so it wraps seamlessly)."""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    out = np.zeros((h, w), np.float64)
    for cycles, amp in octaves:
        spec = np.fft.fft2(rng.standard_normal((h, w)))
        filt = np.exp(-(f * max(h, w) / cycles) ** 2)
        layer = np.real(np.fft.ifft2(spec * filt))
        out += amp * layer / (layer.std() + 1e-9)
    out = (out - out.min()) / (out.max() - out.min())
    return out.astype(np.float32)


# ---------- photos and clips ----------
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
        # leave spare room so the Ken Burns zoom-out and shakes never run off the photo
        win_h = min(win_h, hs / 1.12, ws * 16 / 9 / 1.12)
        face_pos = shot.get("face_pos", 0.40)
        cx = (fx.min() + fx.max()) / 2 + shot.get("dx", 0.0) * win_h * 9 / 16
        cy = float(fy.mean()) + (0.5 - face_pos) * win_h
        f = min(1.0, (H * 1.12) / (win_h * 0.86))
        if f < 1:
            img = cv2.resize(img, (round(ws * f), round(hs * f)), interpolation=cv2.INTER_AREA)
        self.img = grade(img)
        self.win_h = win_h * f
        self.hs, self.ws = self.img.shape[:2]
        self.cx, self.cy = self.clamp(cx * f, cy * f, self.win_h)

    def clamp(self, cx, cy, wh):
        ww = wh * 9 / 16
        cx = min(max(cx, ww / 2), self.ws - ww / 2) if ww < self.ws else self.ws / 2
        cy = min(max(cy, wh / 2), self.hs - wh / 2) if wh < self.hs else self.hs / 2
        return cx, cy

    def render(self, zoom=1.0, dx=0.0, offset=(0.0, 0.0), motion_blur=0):
        wh = self.win_h * zoom
        cx, cy = self.clamp(self.cx + dx * wh * 9 / 16, self.cy, wh)
        s = H / wh
        M = np.array([[s, 0, W / 2 - s * cx + offset[0]], [0, s, H / 2 - s * cy + offset[1]]], np.float32)
        out = cv2.warpAffine(self.img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
        if motion_blur > 1:
            out = cv2.blur(out, (int(motion_blur), 1))
        return out

    def crop(self, w, h, zoom=0.82):
        """A w x h crop around the kid(s), for grids and posts (float RGB 0..1)."""
        ch = min(self.win_h * zoom, self.hs)
        cw = ch * w / h
        if cw > self.ws:
            cw, ch = self.ws, self.ws * h / w
        cx = min(max(self.cx, cw / 2), self.ws - cw / 2)
        cy = min(max(self.cy, ch / 2), self.hs - ch / 2)
        x0, y0 = int(cx - cw / 2), int(cy - ch / 2)
        im = self.img[y0:y0 + int(ch), x0:x0 + int(cw)]
        return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


class Clip:
    """An OpenGen clip, played back slowed to fill `length` seconds with frame blending."""

    def __init__(self, path, length):
        cap = cv2.VideoCapture(str(path))
        self.frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if fr.shape[1] != W or fr.shape[0] != H:
                fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_CUBIC)
            self.frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        self.fps = 24.0
        self.speed = min(1.0, (len(self.frames) - 1) / self.fps / length)

    def at(self, t, zoom=1.0, offset=(0, 0)):
        fi = max(0.0, min(len(self.frames) - 1.001, t * self.speed * self.fps))
        i = int(fi)
        k = fi - i
        a = self.frames[i].astype(np.float32)
        img = a if k < 0.02 else a * (1 - k) + self.frames[i + 1].astype(np.float32) * k
        img /= 255.0
        if zoom != 1.0 or offset != (0, 0):
            M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, zoom)
            M[0, 2] += offset[0]
            M[1, 2] += offset[1]
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        return img


# ---------- horror transitions ----------
FX_WINDOWS = {  # (seconds before the cut, seconds after)
    "lightning": (0.0, 0.42), "glitch": (0.10, 0.16), "slime": (0.24, 0.24), "ghost": (0.0, 0.45),
    "flicker": (0.07, 0.17), "fog": (0.30, 0.34), "zoom": (0.0, 0.3),
}


def fx_glitch(img, amt, seed):
    if amt <= 0.01:
        return img
    out = img.copy()
    dx = int(34 * amt)
    out[..., 0] = np.roll(img[..., 0], dx, axis=1)
    out[..., 2] = np.roll(img[..., 2], -dx, axis=1)
    for i in range(int(10 * amt) + 2):
        y = int(hash01(seed, i, 1) * H)
        hh = int(12 + hash01(seed, i, 2) * 140)
        s = int((hash01(seed, i, 3) - 0.5) * 260 * amt)
        out[y:y + hh] = np.roll(out[y:y + hh], s, axis=1)
        if hash01(seed, i, 4) < 0.3:
            out[y:y + hh] = out[y:y + hh] * 0.6 + np.array([0.35, 0.05, 0.6], np.float32) * 0.4 * amt
    out[::3] *= (1 - 0.3 * amt)
    return out


def fx_chroma(img, amt):
    if amt <= 0.01:
        return img
    out = img.copy()
    d = int(16 * amt)
    out[..., 0] = np.roll(img[..., 0], d, axis=1)
    out[..., 2] = np.roll(img[..., 2], -d, axis=1)
    return out


def shake(img, dx, dy):
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return img
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


class SlimeWipe:
    """Purple slime that pours down over the old shot and slides away to reveal the new one."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        x = np.arange(W, dtype=np.float32)
        L = 30 + 18 * np.sin(x / 47 + seed) + 12 * np.sin(x / 19 + seed * 2)
        for _ in range(16):
            c, w, ln = rng.uniform(0, W), rng.uniform(34, 86), rng.uniform(90, 300)
            prof = np.sqrt(np.clip(1 - ((x - c) / (w / 2)) ** 2, 0, 1))
            L = np.maximum(L, ln * prof ** 0.6)
        self.L = L[None, :]
        self.top = (26 * np.sin(x / 63 + seed) + 14 * np.sin(x / 23))[None, :]
        self.y = np.arange(H, dtype=np.float32)[:, None]
        self.body = H * 1.2
        self.col = np.array(SLIME, np.float32) / 255

    def apply(self, img, p):
        Y = -320 + p * (H + self.body + 380)
        yb = Y + self.L
        yt = Y - self.body + self.top
        a = np.clip((yb - self.y) / 3, 0, 1) * np.clip((self.y - yt) / 3, 0, 1)
        if a.max() <= 0:
            return img
        rim = np.clip(1 - (yb - self.y) / 26, 0, 1) * a
        shade = 0.75 + 0.35 * np.clip((self.y - yt) / (self.body + 1), 0, 1)
        slime = self.col[None, None, :] * shade[..., None] + rim[..., None] * np.array([0.35, 0.2, 0.45], np.float32)
        a3 = a[..., None]
        return img * (1 - a3) + slime * a3


# ---------- timeline ----------
def build_timeline():
    segs = [{"kind": "cold", "t0": 0.0, "t1": COLD_LEN}]
    segs.append({"kind": "title", "b0": 0, "b1": TITLE_BEATS, "fx": None})
    b = TITLE_BEATS
    for i, s in enumerate(SHOTS):
        segs.append({"kind": "shot", "b0": b, "b1": b + s["beats"], "shot": s, "num": i + 1, "fx": s.get("fx", "zoom")})
        b += s["beats"]
    for j, s in enumerate(FAMILY["shots"]):
        segs.append({"kind": "family", "b0": b, "b1": b + s["beats"], "shot": s, "fam": j, "fx": s.get("fx", "fog")})
        b += s["beats"]
    segs.append({"kind": "vote", "b0": b, "b1": b + VOTE["beats"], "fx": VOTE["fx"]})
    b += VOTE["beats"]
    segs.append({"kind": "outro", "b0": b, "b1": b + OUTRO["beats"], "fx": OUTRO["fx"]})
    for i, sg in enumerate(segs):
        if "b0" in sg:
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
        self.scene_cache = {}
        self.clips = {}
        self.labels = {sg["i"]: to_sprite(label_sprite(sg["num"], sg["shot"]["label"]))
                       for sg in self.segs if sg["kind"] == "shot"}
        self.logo_small = to_sprite(logo_image(170))
        self.logo_mid = to_sprite(logo_image(330))
        self.logo_big = to_sprite(logo_image(400))
        self.title_logo = self.logo_mid
        # cold open + title
        self.c1 = to_sprite(text_image(COLD["line1"].upper(), font(F_HORROR, 92), fill=(CREAM, (200, 190, 220)),
                                       stroke=INK, stroke_w=5, shadow=0.8, glow=((150, 90, 255), 16, 1.6)))
        self.c2 = to_sprite(text_image(COLD["line2"].upper(), font(F_HORROR, 70), fill=(CREAM, (200, 190, 220)),
                                       stroke=INK, stroke_w=5, shadow=0.8, glow=((150, 90, 255), 14, 1.4)))
        self.t_big = to_sprite(fit_width(text_image(TITLE["big"].upper(), font(F_DRIP, 150), fill=(GOLD, BLOOD),
                                                    stroke=INK, stroke_w=8, extrude=12, extrude_col=(60, 10, 70),
                                                    glow=((255, 90, 20), 22, 1.3)), 1060))
        self.t_sub = to_sprite(fit_width(text_image(TITLE["sub"].upper(), font(F_HORROR, 108), fill=(WHITE, (220, 210, 235)),
                                                    stroke=INK, stroke_w=7, extrude=8, extrude_col=PURPLE,
                                                    glow=((130, 70, 255), 16, 1.2)), 1020))
        cta = text_image(TITLE["cta"].upper(), font(F_HORROR, 50), fill=(GOLD, ORANGE), stroke=INK, stroke_w=4,
                         shadow=0, pad=4)
        self.t_cta = to_sprite(drippy_pill(cta, SLIME, pad_x=34, pad_y=10, seed=5))
        # family banner
        lines = [text_image(t, font(F_HORROR, 66), fill=(GOLD, ORANGE), stroke=INK, stroke_w=5, shadow=0, pad=4)
                 for t in ("EVEN THE GROWN-UPS", "DRESSED UP!")]
        self.fam_banner = to_sprite(drippy_pill(stack(lines, -10), SLIME, pad_x=40, pad_y=16, seed=9))
        self.build_vote()
        self.build_outro()
        self.sparkles = [to_sprite(sparkle_image(r, c)) for r, c in
                         [(22, GOLD), (16, CREAM), (18, (200, 160, 255)), (12, GOLD), (26, CREAM)]]
        self.bats = [to_sprite(bat_image(math.cos(2 * math.pi * i / 8))) for i in range(8)]
        self.web = to_sprite(cobweb_image())
        self.bolts = [to_sprite(bolt_image(s)) for s in range(4)]
        self.fog = periodic_noise(H // 4, W // 4, 3)
        self.fog2 = periodic_noise(H // 4, W // 4, 11, ((4, 1.0), (10, 0.6)))
        self.grain = [np.random.default_rng(s).normal(0, 1, (H // 2, W // 2)).astype(np.float32) for s in range(6)]
        self.slimes = {}
        self.bat_flights = self.plan_bats()
        self.wall = self.make_wall()

    # ----- builders -----
    def make_wall(self):
        """A tilted wall of every costume photo (used by the posts)."""
        tw, th, gap = 330, 440, 26
        cols, rows = 5, 8
        canvas = Image.new("RGB", (cols * (tw + gap) + gap, rows * (th + gap) + gap), DEEP)
        names = [s for s in SHOTS] + [s for s in FAMILY["shots"]]
        for i in range(cols * rows):
            shot = names[(i * 7) % len(names)]
            tile = Image.fromarray((self.photo(shot).crop(tw, th, 0.86) * 255).astype(np.uint8))
            c, r = i % cols, i // cols
            canvas.paste(tile, (gap + c * (tw + gap), gap + r * (th + gap) + (th // 2 if c % 2 else 0) - th // 4))
        big = canvas.rotate(-9, resample=Image.BICUBIC, expand=True, fillcolor=DEEP)
        return np.asarray(big).astype(np.float32) / 255.0

    def build_vote(self):
        self.v_big = to_sprite(fit_width(text_image(VOTE["big"].upper(), font(F_DRIP, 92), fill=(GOLD, BLOOD), stroke=INK,
                                                    stroke_w=7, extrude=9, extrude_col=(60, 10, 70),
                                                    glow=((255, 90, 20), 18, 1.2)), 1040))
        self.v_sub = to_sprite(text_image(VOTE["sub"].upper(), font(F_HORROR, 54), fill=(WHITE, (220, 210, 235)),
                                          stroke=INK, stroke_w=5, shadow=0.7, glow=((130, 70, 255), 12, 1.2)))
        cols, gap = 6, 10
        self.cell_w = (W - gap * (cols + 1)) // cols
        self.cell_h = int(self.cell_w * 4 / 3)
        self.cells = []
        for i, s in enumerate(SHOTS):
            crop = self.photo(s).crop(self.cell_w, self.cell_h, 0.75)
            im = Image.fromarray((crop * 255).astype(np.uint8)).convert("RGBA")
            framed = Image.new("RGBA", (self.cell_w + 8, self.cell_h + 8), CREAM + (255,))
            framed.alpha_composite(im, (4, 4))
            badge = fit_width(pumpkin_badge(i + 1, 92), 70)
            framed.alpha_composite(badge, (0, 0))
            self.cells.append(to_sprite(framed))
        tag = text_image("COMMENT", font(F_HORROR, 44), fill=(GOLD, ORANGE), stroke=INK, stroke_w=4, shadow=0, pad=2)
        tag2 = text_image("YOUR FAVE #", font(F_HORROR, 44), fill=(GOLD, ORANGE), stroke=INK, stroke_w=4, shadow=0, pad=2)
        self.cell_cta = to_sprite(fit_width(drippy_pill(stack([tag, tag2], -6), SLIME, pad_x=22, pad_y=10, seed=13),
                                            self.cell_w * 2 + 20))
        self.grid_gap = gap
        # haunted-house night scene, blurred and darkened, behind the grid
        bg = cv2.cvtColor(cv2.imread(str(OPENGEN / "intro_night_scene.jpg")), cv2.COLOR_BGR2RGB)
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        self.vote_bg = cv2.GaussianBlur(bg, (0, 0), 14) * 0.38

    def build_outro(self):
        self.o_logo = self.logo_big
        self.o_l1 = to_sprite(text_image(OUTRO["line1"].upper(), font(F_HORROR, 96), fill=(CREAM, (205, 195, 225)),
                                         stroke=INK, stroke_w=6, shadow=0.8, glow=((140, 80, 255), 18, 1.4)))
        self.o_l2 = to_sprite(fit_width(text_image(OUTRO["line2"].upper(), font(F_DRIP, 132), fill=(GOLD, BLOOD),
                                                   stroke=INK, stroke_w=8, extrude=12, extrude_col=(60, 10, 70),
                                                   glow=((255, 90, 20), 24, 1.4)), 1050))
        self.o_tag = to_sprite(text_image(OUTRO["tag"], font(F_HORROR, 46), fill=(GHOST, (170, 190, 230)), stroke=INK,
                                          stroke_w=4, shadow=0.8, glow=((120, 160, 255), 12, 1.1)))
        self.o_pills = [to_sprite(sticker(n, w)) for n, w in
                        (("pill_phone", 600), ("pill_address", 640), ("pill_email", 610))]
        self.o_stay = to_sprite(text_image(OUTRO["stay"], font(F_BODY, 44, [800]), fill=(CREAM, CREAM), stroke=INK,
                                           stroke_w=5, shadow=0.8, pad=10))

    def plan_bats(self):
        flights = []  # (start, duration, y0, y1, left_to_right, scale)
        flights += [(0.2, 1.8, 420, 240, True, 0.9), (0.9, 1.6, 1500, 1300, False, 0.65)]
        title = self.seg("title")
        flights += [(title["t0"] + 0.05, 1.4, 600, 420, False, 0.8)]
        fam = self.seg("family")
        flights += [(fam["t0"] + 0.1, 1.4, 700, 520, True, 0.7)]
        vote = self.seg("vote")
        flights += [(vote["t0"] + 1.0, 1.6, 300, 200, False, 0.6)]
        end = self.seg("outro")
        flights += [(end["t0"] + 0.4, 2.0, 520, 300, True, 0.85), (end["t0"] + 2.6, 2.2, 260, 140, False, 0.55),
                    (end["t0"] + 4.2, 2.0, 980, 760, True, 0.6)]
        return flights

    def seg(self, kind):
        return next(s for s in self.segs if s["kind"] == kind)

    def photo(self, shot):
        key = shot["photo"]
        if key not in self.photos:
            if len(self.photos) > 26:
                self.photos.pop(next(iter(self.photos)))
            self.photos[key] = Photo(shot)
        return self.photos[key]

    def clip(self, name):
        if name not in self.clips:
            if name == "intro":
                length = self.seg("title")["t1"]
                self.clips[name] = Clip(OPENGEN / "intro_scene.mp4", length)
            else:
                end = self.seg("outro")
                path = OPENGEN / "outro_haunted.mp4"
                self.clips[name] = Clip(path if path.exists() else OPENGEN / "outro_bg.mp4", end["t1"] - end["t0"])
        return self.clips[name]

    def drop_clip(self, name):
        self.clips.pop(name, None)

    # ----- shared overlays -----
    def darken(self, img, k=1.0):
        a = self.overlay * k
        return img * (1 - a) + self.deep * a

    def draw_fog(self, img, t, density=0.5, y_from=0.55, speed=26, color=(0.78, 0.74, 0.9)):
        sh = int(t * speed) % self.fog.shape[1]
        f = np.roll(self.fog, -sh, axis=1)
        f2 = np.roll(self.fog2, int(t * speed * 0.6) % self.fog2.shape[1], axis=1)
        tex = np.clip((f * 0.6 + f2 * 0.4 - 0.35) * 1.9, 0, 1)
        tex = cv2.resize(tex, (W, H), interpolation=cv2.INTER_LINEAR)
        y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        band = np.clip((y - y_from) / (1 - y_from), 0, 1) ** 1.2
        a = (tex * band * density)[..., None]
        return img * (1 - a) + np.array(color, np.float32) * a

    def draw_grain(self, img, t, amount=0.012):
        g = self.grain[int(t * FPS) % len(self.grain)]
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)
        return img + g[..., None] * amount

    def draw_sparks(self, img, t, n=14, strength=1.0, sides_only=True):
        for i in range(n):
            x = hash01(i, 1) * 160 + 30 if (sides_only and i % 2 == 0) else (
                W - 30 - hash01(i, 1) * 160 if sides_only else hash01(i, 1) * (W - 80) + 40)
            y0 = hash01(i, 2) * 1500 + 300
            fq = 0.6 + hash01(i, 3) * 0.8
            tw = math.sin(2 * math.pi * (t * fq * 0.7 + hash01(i, 4)))
            a = max(0.0, tw) ** 2 * strength
            if a < 0.02:
                continue
            y = (y0 - 26 * t) % 1600 + 150
            blit(img, self.sparkles[i % 5], x, y, (0.4 + 0.4 * hash01(i, 5)) * (0.6 + 0.4 * a), 20 * t * fq, a)

    def draw_bats(self, img, t):
        for t0, dur, y0, y1, ltr, sc in self.bat_flights:
            p = (t - t0) / dur
            if not 0 <= p <= 1:
                continue
            x = -160 + (W + 320) * p
            x = x if ltr else W - x
            y = y0 + (y1 - y0) * p + 40 * math.sin(p * 6.5)
            spr = self.bats[int(t * 22) % len(self.bats)]
            blit(img, spr, x, y, sc * (0.85 + 0.3 * p), (-12 if ltr else 12) + 8 * math.sin(p * 9), 1.0)

    def lightning_flash(self, img, dt, strength=1.0, seed=0, bolt=True):
        """dt = time since the strike. White-blue flash, a second flicker, one inverted frame and a bolt."""
        if dt < 0 or dt > 0.45:
            return img
        f1 = math.exp(-dt / 0.045)
        f2 = 0.65 * math.exp(-abs(dt - 0.13) / 0.03) if dt > 0.08 else 0.0
        fl = max(f1, f2) * strength
        if dt < 1 / FPS and strength > 0.6:
            img = (1 - img) * np.array([0.55, 0.5, 0.85], np.float32)
        if bolt and (dt < 0.1 or 0.11 < dt < 0.17):
            b = self.bolts[seed % len(self.bolts)]
            blit(img, b, 200 + 680 * hash01(seed, 9), 330 + 120 * hash01(seed, 8), 1.0,
                 -12 + 24 * hash01(seed, 7), min(1.0, fl * 1.4))
        return img + (np.array([0.92, 0.94, 1.0], np.float32) - img) * fl * 0.85

    # ----- scenes (everything belonging to one segment) -----
    def kb(self, sg, t):
        p = (t - sg["t0"]) / (sg["t1"] - sg["t0"])
        e = ease_in_out(p)
        zoom = 1.07 - 0.07 * e if sg["i"] % 2 == 0 else 1.0 + 0.06 * e
        dx = (((sg["i"] * 37) % 7) - 3) / 3 * 0.025 * (e - 0.5)
        since = t - sg["t0"]
        if sg.get("fx") == "zoom":
            zoom *= 1 - 0.1 * (1 - ease_out_cubic(since / 0.32))
        if sg.get("fx") == "lightning":
            zoom *= 1 - 0.05 * (1 - ease_out_cubic(since / 0.5))
        return zoom, dx

    def photo_scene(self, sg, t):
        t = min(max(t, sg["t0"]), sg["t1"] - 1e-4)
        zoom, dx = self.kb(sg, t)
        img = self.photo(sg["shot"]).render(zoom, dx).astype(np.float32) / 255.0
        img = self.darken(img)
        img = self.draw_fog(img, t, 0.32, 0.72)
        blit(img, self.web, W - 165, 165, 1.0, 2 * math.sin(t * 1.3), 0.75, pivot=(330, 0))
        self.draw_sparks(img, t, 10, 0.7)
        since, until = t - sg["t0"], sg["t1"] - t
        if sg["kind"] == "shot":
            lab = self.labels[sg["i"]]
            p = (since - 0.06) / 0.38
            fit = min(1.0, 1000 / lab.shape[1])
            drop = 1 - ease_out_back(p, 1.6)
            swing = 3.5 * math.sin(2 * math.pi * since / 1.6) * (0.4 + 0.6 * (1 - ease_out_cubic(p)))
            blit(img, lab, W / 2 + 10, 395 - 220 * drop, fit, swing, clamp01(p * 3), pivot=(lab.shape[1] / 2, 0))
        else:
            fam0 = self.seg("family")["t0"]
            p = (t - fam0 - 0.05) / 0.4
            blit(img, self.fam_banner, W / 2, 400 - 200 * (1 - ease_out_back(p, 1.6)), 1.0,
                 2.5 * math.sin(2 * math.pi * t / 1.7), clamp01(p * 3), pivot=(self.fam_banner.shape[1] / 2, 0))
        blit(img, self.logo_small, 112, 222, 1.0, 0, 0.95 * flicker(t, 3, depth=0.25))
        return img

    def cold_scene(self, sg, t):
        clip = self.clip("intro")
        img = clip.at(t, 1.06 - 0.03 * t / COLD_LEN)
        img = self.darken(img, 1.3)
        img = img * 0.62  # night
        img = self.draw_fog(img, t, 0.55, 0.6, speed=34)
        # appear out of black on the first thunder strike
        img = img * clamp01((t - 0.08) / 0.12) if t < 0.2 else img
        img = self.lightning_flash(img, t - 0.12, 1.0, seed=1)
        img = self.lightning_flash(img, t - 1.55, 0.55, seed=2, bolt=False)
        p1 = (t - 0.45) / 0.3
        a1 = clamp01(p1 * 2) * (flicker(t, 11, depth=0.9) if p1 < 1.4 else 1.0)
        blit(img, self.c1, W / 2, 760, 1.0 + 0.04 * (1 - ease_out_cubic(p1)), 0, a1)
        p2 = (t - 0.95) / 0.3
        a2 = clamp01(p2 * 2) * (flicker(t, 12, depth=0.9) if p2 < 1.4 else 1.0)
        blit(img, self.c2, W / 2, 870, 1.0, 0, a2)
        p3 = (t - 1.45) / 0.4
        a3 = clamp01(p3 * 2) * (flicker(t, 13, depth=0.8) if p3 < 1.2 else 1.0)
        blit(img, self.logo_mid, W / 2, 1110 + 6 * math.sin(t * 3), 0.85 + 0.15 * ease_out_back(p3), 0, a3)
        self.draw_bats(img, t)
        # glitch build-up into the music
        g = clamp01((t - (COLD_LEN - 0.35)) / 0.35)
        if g > 0:
            img = fx_glitch(img, g * 0.9, int(t * FPS))
        return img

    def title_scene(self, sg, t):
        since = t - sg["t0"]
        clip = self.clip("intro")
        img = clip.at(t, 1.03 - 0.03 * since / (sg["t1"] - sg["t0"]))
        img = self.darken(img, 1.2) * 0.7
        img = self.draw_fog(img, t, 0.55, 0.6, speed=34)
        self.draw_sparks(img, t, 18, 0.8, sides_only=False)
        b = lambda k: k * BEAT  # noqa: E731
        # logo rises from the cold open to the top
        p = ease_in_out(since / 0.45)
        blit(img, self.logo_mid, W / 2, 1110 + (330 - 1110) * p, 1.0 - 0.18 * p, 0, 1.0)
        # HALLOWEEN slams on the first beat
        ps = since / 0.22
        blit(img, self.t_big, W / 2, 860, 1.0 + 0.7 * (1 - ease_out_cubic(ps)), -2, clamp01(ps * 3))
        p2 = (since - b(1)) / 0.3
        blit(img, self.t_sub, W / 2, 1040, ease_out_back(p2), 1.5, clamp01(p2 * 3))
        p3 = (since - b(2.4)) / 0.3
        blit(img, self.t_cta, W / 2, 1200 + 50 * (1 - ease_out_cubic(p3)), 1.0, 0, clamp01(p3 * 2.5))
        self.draw_bats(img, t)
        img = self.lightning_flash(img, since, 1.0, seed=3)
        sx = 26 * math.exp(-since / 0.12) * math.sin(since * 95)
        sy = 18 * math.exp(-since / 0.12) * math.cos(since * 80)
        return shake(img, sx, sy)

    def vote_scene(self, sg, t):
        since = t - sg["t0"]
        img = self.vote_bg * (0.92 + 0.08 * flicker(t, 22, depth=0.5, rate=10))
        img = self.draw_fog(img, t, 0.5, 0.35, speed=20, color=(0.36, 0.24, 0.46))
        p = since / 0.3
        blit(img, self.v_big, W / 2, 330, 1.0 + 0.5 * (1 - ease_out_cubic(p)), -1.5, clamp01(p * 3))
        p = (since - 0.25) / 0.3
        blit(img, self.v_sub, W / 2, 455, 1.0, 0, clamp01(p * 2) * flicker(t, 21, depth=0.3))
        gap, cw, ch = self.grid_gap, self.cell_w, self.cell_h
        top = 545
        for i, cell in enumerate(self.cells):
            c, r = i % 6, i // 6
            x = gap + c * (cw + gap) + cw / 2
            y = top + r * (ch + gap) + ch / 2
            q = (since - 0.35 - i * 0.03) / 0.28
            if q <= 0:
                continue
            pulse = 1 + 0.03 * math.sin(2 * math.pi * (since * 1.2 + i * 0.13))
            blit(img, cell, x, y, ease_out_back(q) * pulse, 3 * math.sin(i * 1.7 + since * 2) * 0.4, clamp01(q * 3))
        q = (since - 1.1) / 0.35
        x = gap + 4 * (cw + gap) + cw + gap / 2
        y = top + 3 * (ch + gap) + ch / 2
        blit(img, self.cell_cta, x, y + 6 * math.sin(since * 4), ease_out_back(q) * (1 + 0.04 * math.sin(since * 6)),
             -2, clamp01(q * 3))
        self.draw_bats(img, t)
        return img

    def outro_scene(self, sg, t):
        since = t - sg["t0"]
        clip = self.clip("outro")
        img = clip.at(since, 1.0 + 0.04 * since / (sg["t1"] - sg["t0"]))
        img = self.darken(img, 0.9)
        img = img * (0.9 + 0.1 * flicker(t, 31, depth=0.4, rate=12))
        img = self.draw_fog(img, t, 0.45, 0.62, speed=22)
        self.draw_sparks(img, t, 16, 0.6, sides_only=False)
        strike = 0.42
        # logo flickers on like a failing sign, then floats
        p = (since - 0.12) / 0.5
        a = clamp01(p * 2) * (flicker(t, 41, depth=0.95) if p < 1.3 else 1.0)
        blit(img, self.o_logo, W / 2, 300 + 8 * math.sin(since * 2.2), 1.0, 1.5 * math.sin(since * 1.4), a)
        # THANKS FOR / WATCHING arrive on the lightning strike and keep swaying
        q = (since - strike) / 0.25
        jit = (since > strike + 1.2) and (int(since * FPS) % 37 < 2)
        sw = 2.2 * math.sin(2 * math.pi * since / 2.2)
        blit(img, self.o_l1, W / 2 + (8 if jit else 0), 620, 1.0 + 0.35 * (1 - ease_out_cubic(q)), sw * 0.6,
             clamp01(q * 3), pivot=(self.o_l1.shape[1] / 2, 0))
        blit(img, self.o_l2, W / 2 - (10 if jit else 0), 760, 1.0 + 0.6 * (1 - ease_out_cubic(q)), -sw,
             clamp01(q * 3), pivot=(self.o_l2.shape[1] / 2, 0))
        q = (since - 0.95) / 0.5
        blit(img, self.o_tag, W / 2, 885 + 4 * math.sin(since * 2), 1.0, 0, clamp01(q) * flicker(t, 44, depth=0.35))
        # contact stickers glitch in one by one, then bob
        for k, (spr, y) in enumerate(zip(self.o_pills, (1035, 1185, 1332))):
            q = (since - 1.3 - 0.22 * k) / 0.2
            if q <= 0:
                continue
            off = 0 if q >= 1 else (hash01(k, int(since * FPS)) - 0.5) * 120 * (1 - q)
            blit(img, spr, W / 2 + off, y + 5 * math.sin(since * 2.4 + k), 1.0, 1.2 * math.sin(since * 1.7 + k * 1.3),
                 clamp01(q * 2) * (flicker(t, 50 + k, depth=0.7) if q < 1.5 else 1.0))
        q = (since - 2.1) / 0.5
        blit(img, self.o_stay, W / 2, 1462, 1.0, 0, clamp01(q))
        self.draw_bats(img, t)
        img = self.lightning_flash(img, since - strike, 1.0, seed=5)
        img = self.lightning_flash(img, since - 3.9, 0.45, seed=6, bolt=False)
        sx = 20 * math.exp(-(since - strike) / 0.12) * math.sin(since * 95) if since > strike else 0
        return shake(img, sx, 0)

    def scene(self, sg, t):
        kind = sg["kind"]
        if kind == "cold":
            return self.cold_scene(sg, t)
        if kind == "title":
            return self.title_scene(sg, t)
        if kind in ("shot", "family"):
            return self.photo_scene(sg, t)
        if kind == "vote":
            return self.vote_scene(sg, t)
        return self.outro_scene(sg, t)

    def frozen(self, sg):
        """The last frame of a segment (for the ghost transition), cached."""
        if sg["i"] not in self.scene_cache:
            self.scene_cache = {sg["i"]: self.scene(sg, sg["t1"] - 1e-3)}
        return self.scene_cache[sg["i"]]

    def slime(self, i):
        if i not in self.slimes:
            self.slimes = {i: SlimeWipe(i)}
        return self.slimes[i]

    # ----- one frame -----
    def frame(self, t):
        t = min(t, self.total - 1e-4)
        idx = next(i for i, s in enumerate(self.segs) if s["t0"] <= t < s["t1"])
        sg = self.segs[idx]
        nxt = self.segs[idx + 1] if idx + 1 < len(self.segs) else None
        prv = self.segs[idx - 1] if idx > 0 else None
        if sg["kind"] not in ("cold", "title") and "intro" in self.clips:
            self.drop_clip("intro")
        img = self.scene(sg, t)
        # outgoing half of the next cut's transition
        if nxt and nxt.get("fx"):
            pre, _ = FX_WINDOWS[nxt["fx"]]
            until = nxt["t0"] - t
            if pre > 0 and until < pre:
                img = self.fx(nxt, img, -until, pre, prv=sg)
        # incoming half of this segment's transition
        if sg.get("fx"):
            pre, post = FX_WINDOWS[sg["fx"]]
            since = t - sg["t0"]
            if since < post:
                img = self.fx(sg, img, since, post, prv=prv)
        img = self.draw_grain(img, t)
        return img

    def fx(self, sg, img, dt, dur, prv):
        """Apply sg's transition. dt < 0 before the cut (on the outgoing shot), dt >= 0 after it."""
        kind = sg["fx"]
        seed = sg["i"]
        if kind == "lightning":
            img = self.lightning_flash(img, dt, 1.0, seed=seed)
            sx = 24 * math.exp(-dt / 0.12) * math.sin(dt * 95)
            sy = 16 * math.exp(-dt / 0.12) * math.cos(dt * 80)
            return shake(img, sx, sy)
        if kind == "glitch":
            pre, post = FX_WINDOWS["glitch"]
            amt = 1 - (abs(dt) / (pre if dt < 0 else post))
            return fx_glitch(img, clamp01(amt), seed * 100 + int((dt + 1) * FPS))
        if kind == "slime":
            pre, post = FX_WINDOWS["slime"]
            p = 0.5 + (dt / pre if dt < 0 else dt / post) * 0.5
            return self.slime(seed).apply(img, clamp01(p))
        if kind == "ghost":
            q = clamp01(dt / dur)
            old = self.frozen(prv)
            lum = old @ np.array([0.3, 0.55, 0.15], np.float32)
            ghost = lum[..., None] * np.array([0.8, 0.95, 1.05], np.float32)
            sc = 1 + 0.22 * q
            M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, sc)
            M[1, 2] -= 90 * q
            ghost = cv2.warpAffine(ghost, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
            a = 0.6 * (1 - q) ** 1.4
            img = img * (1 - a * 0.55) + ghost * a
            return fx_chroma(img, (1 - q) * 0.6)
        if kind == "flicker":
            f = int(round(dt * FPS))
            mult = {-2: 0.5, -1: 0.18, 0: 1.7, 1: 0.12, 2: 1.35, 3: 0.45, 4: 1.15, 5: 0.85}.get(f, 1.0)
            tint = np.array([0.85, 1.0, 0.95], np.float32) if mult < 0.6 else np.ones(3, np.float32)
            return np.clip(img * mult * tint, 0, 1.4)
        if kind == "fog":
            pre, post = FX_WINDOWS["fog"]
            p = 0.5 + (dt / pre if dt < 0 else dt / post) * 0.5
            dens = math.sin(math.pi * clamp01(p)) ** 0.8
            f = np.roll(self.fog, int(dt * 900), axis=1)
            tex = cv2.resize(np.clip(f * 1.3, 0, 1), (W, H), interpolation=cv2.INTER_LINEAR)
            a = np.clip(tex * 0.5 + 0.55, 0, 1)[..., None] * dens
            return img * (1 - a) + np.array([0.7, 0.66, 0.82], np.float32) * a
        if kind == "zoom":
            amt = 1 - dt / dur
            img = fx_chroma(img, amt)
            return img + (1 - img) * 0.28 * math.exp(-dt / 0.05)
        return img


# ---------- audio ----------
def decode(path, start=0.0, dur=None):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(path)]
    if dur:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-f", "f32le", "-ac", "2", "-ar", "48000", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()


def sfx_events(r):
    """(sound, time, gain dB, trim start, trim length) for every sound effect in the reel."""
    ev = [("thunder", 0.08, -2, 0.0, 3.4), ("ghost_whoosh", 0.42, -8, 0, None), ("riser", COLD_LEN - 2.0, -6, 1.6, None),
          ("glitch", COLD_LEN - 0.35, -10, 0.0, 0.4), ("impact", COLD_LEN, -3, 0, None)]
    sounds = {"lightning": ("thunder", -9, 0.0, 1.6), "glitch": ("glitch", -13, 0.0, 0.35),
              "slime": ("whoosh", -9, 0, None), "ghost": ("ghost_whoosh", -9, 0, None), "fog": ("whoosh", -12, 0, None)}
    for sg in r.segs:
        fx = sg.get("fx")
        if fx in sounds and sg["kind"] != "title":
            name, g, s0, ln = sounds[fx]
            lead = FX_WINDOWS[fx][0] if fx in ("slime", "fog") else 0.0
            ev.append((name, sg["t0"] - lead - (0.15 if name == "whoosh" else 0), g, s0, ln))
    end = r.seg("outro")
    ev += [("thunder", end["t0"] + 0.42, -5, 0.0, 3.0), ("witch", end["t0"] + 2.4, -9, 0, None)]
    return ev


def build_audio(r, music=True):
    sr = 48000
    n = int(math.ceil(r.total * sr))
    mix = np.zeros((n, 2), np.float32)
    if music:
        dur = r.total - COLD_LEN
        m = decode(MUSIC, T0, dur)
        fade_n = int(1.9 * sr)
        m[-fade_n:] *= np.linspace(1, 0, fade_n)[:, None] ** 1.5
        m[:240] *= np.linspace(0, 1, 240)[:, None]
        s0 = int(COLD_LEN * sr)
        mix[s0:s0 + len(m)] += m[:n - s0] * 0.95
    cache = {}
    for name, t, g, s0, ln in sfx_events(r):
        key = (name, s0, ln)
        if key not in cache:
            snd = decode(SFX / f"{name}.mp3", s0, ln)
            k = min(len(snd), int(0.08 * sr))
            if ln:
                snd[-k:] *= np.linspace(1, 0, k)[:, None]
            cache[key] = snd
        snd = cache[key] * (10 ** (g / 20))
        i = int(max(0, t) * sr)
        j = min(n, i + len(snd))
        mix[i:j] += snd[:j - i]
    raw = EXPORTS.parent / "work" / ("mix_raw.wav" if music else "mix_raw_nomusic.wav")
    sf.write(str(raw), mix, sr, subtype="FLOAT")
    target = "I=-14:TP=-1.5:LRA=11"
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(raw), "-af",
                            f"loudnorm={target}:print_format=json", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
    mm = json.loads(probe[probe.rindex("{"):probe.rindex("}") + 1])
    out = raw.with_name(raw.stem.replace("_raw", "") + ".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-af",
                    f"loudnorm={target}:measured_I={mm['input_i']}:measured_TP={mm['input_tp']}:"
                    f"measured_LRA={mm['input_lra']}:measured_thresh={mm['input_thresh']}:"
                    f"offset={mm['target_offset']}:linear=true,aresample=48000",
                    "-ar", "48000", "-c:a", "pcm_s16le", str(out)], check=True)
    return out


def encode(r, out_path, music=True):
    audio = build_audio(r, music)
    dur = r.total
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", str(audio),
           "-filter_complex", "[0:v]scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[v]",
           "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high",
           "-g", str(FPS * 2), "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-bsf:v", "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{dur:.3f}", "-movflags", "+faststart",
           str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(math.ceil(dur * FPS))
    for i in range(n):
        fr = r.frame(i / FPS)
        proc.stdin.write((np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
        if i % 150 == 0:
            print(f"  frame {i}/{n}", flush=True)
    proc.stdin.close()
    proc.wait()
    if proc.returncode:
        raise SystemExit(f"ffmpeg failed ({proc.returncode})")
    print("wrote", out_path)


def main():
    music = "--no-music" not in sys.argv
    EXPORTS.mkdir(exist_ok=True)
    r = Renderer()
    print(f"{len(r.segs)} segments, {r.total:.2f} s")
    out = EXPORTS / ("tinystars_halloween_reel.mp4" if music else "tinystars_halloween_reel_nomusic.mp4")
    encode(r, out, music)


if __name__ == "__main__":
    main()
