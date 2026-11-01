"""Render the Tiny Stars Halloween costume-parade reel: 1080x1920, 30 fps, cut on the beat, under 25 seconds.

Timeline:
  cold open (thunder, no music: "Something spooky happened at..." + logo over the haunted-house clip)
  -> title slam on the first beat (neon gothic "Halloween", COSTUME PARADE)
  -> 12 numbered costumes on frosted-glass labels, each joined by a horror transition (lightning, glitch,
     flicker, ghost, fog, zoom punch)
  -> numbered vote grid ("which one wins? comment the number")
  -> closing card on the haunted classroom: logo, neon THANKS FOR WATCHING, glass contact cards,
     all flickering, drifting and glitching in

Usage: python -I edits/render.py
  -> exports/tinystars_halloween_reel.mp4 (music + sound effects)
     exports/tinystars_halloween_reel_nomusic.mp4 (same picture, sound effects only)
"""
import gc
import json
import math
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content import BRAND, COLD, OUTRO, SHOTS, TITLE, VOTE  # noqa: E402

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
COLD_LEN = 1.5           # cold open before the music
TITLE_BEATS = 4
TAIL = 1.0               # hold the closing card while the music fades

# palette (RGB)
DEEP = (18, 6, 28)
INK = (38, 9, 56)
PURPLE = (100, 42, 115)
VIOLET = (150, 90, 255)
ORANGE = (255, 122, 26)
NEON = (255, 92, 22)       # the glow around the big words
CREAM = (255, 240, 222)
PEACH = (255, 196, 140)
WHITE = (255, 255, 255)
GHOST = (214, 226, 255)

# fonts: modern blackletter for the big words, film-poster serif for names, tracked caps for small print
F_GOTH = "GrenzeGotisch.ttf"
F_SERIF = "Cinzel.ttf"
F_SANS = "Montserrat.ttf"


def bt(k):
    """Video time of beat k; beat 0 is the title slam right after the cold open."""
    return COLD_LEN + k * BEAT


# ---------- easing / noise ----------
def clamp01(x):
    return max(0.0, min(1.0, x))


def ease_out_cubic(p):
    p = clamp01(p)
    return 1 - (1 - p) ** 3


def ease_out_back(p, s=1.7):
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


# ---------- lettering and graphics (PIL, then premultiplied float RGBA sprites) ----------
def to_sprite(img):
    a = np.asarray(img.convert("RGBA")).astype(np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    return a


def font(name, size, weight=None):
    f = ImageFont.truetype(str(FONTS / name), size)
    if weight:
        f.set_variation_by_axes([weight])
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


def text_width(text, f, track=0):
    if not track:
        return f.getlength(text)
    return sum(f.getlength(c) for c in text) + track * (len(text) - 1)


def lettering(text, f, track=0, fill=(CREAM, CREAM), glow=None, shadow=0.6, alpha=1.0):
    """Text with letter-spacing, a top-to-bottom gradient fill, a soft drop shadow and an optional neon glow.

    glow = (colour, radius, strength): a wide halo plus a tight bright one, like a neon tube.
    The image is cropped to what is drawn, so it centres on the lettering."""
    pad = 30 + (glow[1] * 3 if glow else 0)
    asc, desc = f.getmetrics()
    w, h = int(text_width(text, f, track)) + 2 * pad, asc + desc + 2 * pad
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    if track:
        x = pad
        for c in text:
            d.text((x, pad), c, font=f, fill=255)
            x += f.getlength(c) + track
    else:
        d.text((pad, pad), text, font=f, fill=255)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if shadow:
        sh = mask.filter(ImageFilter.GaussianBlur(9)).point(lambda v: int(v * shadow))
        layer = Image.new("RGBA", (w, h), (4, 0, 10, 0))
        layer.putalpha(sh)
        out = Image.alpha_composite(out, layer.transform(out.size, Image.AFFINE, (1, 0, 0, 0, 1, -6)))
    if glow:
        col, rad, k = glow
        for r, kk in ((rad, 1.5 * k), (rad / 3.5, 1.7 * k)):
            g = mask.filter(ImageFilter.GaussianBlur(r)).point(lambda v, kk=kk: min(255, int(v * kk)))
            layer = Image.new("RGBA", (w, h), col + (0,))
            layer.putalpha(g)
            out = Image.alpha_composite(out, layer)
    _, y0, _, y1 = mask.getbbox()
    body = mask if alpha >= 1 else mask.point(lambda v: int(v * alpha))
    out.paste(vgrad(w, h, fill[0], fill[1], y0, y1), (0, 0), body)
    return out.crop(out.getbbox())


def stack(images, gap=0):
    w = max(i.width for i in images)
    h = sum(i.height for i in images) + gap * (len(images) - 1)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    y = 0
    for im in images:
        out.alpha_composite(im, ((w - im.width) // 2, y))
        y += im.height + gap
    return out


def rule(width, col=CREAM, alpha=0.6, thick=2):
    """A thin horizontal line that fades out at both ends."""
    a = np.sin(np.linspace(0, math.pi, width)) ** 0.7 * alpha * 255
    img = Image.new("RGBA", (width, thick + 4), col + (0,))
    m = np.zeros((thick + 4, width), np.uint8)
    m[2:2 + thick] = a.astype(np.uint8)
    img.putalpha(Image.fromarray(m))
    return img


def flanked(text_img, line_w=90, gap=26, col=ORANGE):
    """Text with a short fading rule either side: ---- TEXT ----"""
    r = rule(line_w, col, 0.9)
    w = text_img.width + 2 * (line_w + gap)
    h = max(text_img.height, r.height)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.alpha_composite(r, (0, h // 2 - r.height // 2))
    out.alpha_composite(text_img, (line_w + gap, (h - text_img.height) // 2))
    out.alpha_composite(r, (w - line_w, h // 2 - r.height // 2))
    return out


def rounded_mask(w, h, r):
    """Anti-aliased rounded rectangle, float 0..1 (drawn at 4x, then scaled down)."""
    big = Image.new("L", (w * 4, h * 4), 0)
    ImageDraw.Draw(big).rounded_rectangle((0, 0, w * 4 - 1, h * 4 - 1), r * 4, fill=255)
    return np.asarray(big.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0


def outline_image(w, h, r, stops=(NEON, PEACH, VIOLET), width=2, alpha=0.85, glow=0):
    """A rounded-rectangle outline with a colour gradient (and an optional soft glow), as RGBA."""
    pad = glow * 2
    big = Image.new("L", ((w + 2 * pad) * 4, (h + 2 * pad) * 4), 0)
    ImageDraw.Draw(big).rounded_rectangle((pad * 4 + 2, pad * 4 + 2, (pad + w) * 4 - 3, (pad + h) * 4 - 3), r * 4,
                                          outline=255, width=width * 4)
    m = big.resize((w + 2 * pad, h + 2 * pad), Image.LANCZOS)
    if glow:
        m = Image.fromarray(np.maximum(np.asarray(m), np.asarray(m.filter(ImageFilter.GaussianBlur(glow)).point(
            lambda v: min(255, v * 4)))))
    m = m.point(lambda v: int(v * alpha))
    out = hgrad(w + 2 * pad, h + 2 * pad, list(stops)).convert("RGBA")
    out.putalpha(m)
    return out


def frost(frame, x0, y0, mask, alpha=1.0, dark=0.42):
    """Frosted dark glass: blur and darken what is behind `mask` (top-left corner at x0, y0), in place."""
    if alpha <= 0.004:
        return
    h, w = mask.shape
    fh, fw = frame.shape[:2]
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(fw, x0 + w), min(fh, y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    m = 36
    X0, Y0, X1, Y1 = max(0, fx0 - m), max(0, fy0 - m), min(fw, fx1 + m), min(fh, fy1 + m)
    blur = cv2.GaussianBlur(frame[Y0:Y1, X0:X1], (0, 0), 16)[fy0 - Y0:fy1 - Y0, fx0 - X0:fx1 - X0]
    glass = blur * dark + np.array(DEEP, np.float32) / 255 * 0.5
    a = mask[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0, None] * alpha
    region = frame[fy0:fy1, fx0:fx1]
    region *= 1 - a
    region += glass * a


def wipe(spr, q, soft=60):
    """Reveal a sprite left to right (q in 0..1) with a soft edge."""
    if q >= 1:
        return spr
    w = spr.shape[1]
    x = np.arange(w, dtype=np.float32)
    ramp = np.clip((q * (w + soft) - x) / soft, 0, 1)
    return spr * ramp[None, :, None]


class Glass:
    """A frosted-glass card with a thin gradient outline. Content is drawn on top by the caller."""

    def __init__(self, w, h, r, stops=(NEON, PEACH, VIOLET), dark=0.42):
        self.w, self.h, self.dark = w, h, dark
        self.mask = rounded_mask(w, h, r)
        self.edge = to_sprite(outline_image(w, h, r, stops))

    def draw(self, frame, cx, cy, alpha=1.0):
        """Draws the card centred at (cx, cy) and returns its top-left corner."""
        x0, y0 = int(round(cx - self.w / 2)), int(round(cy - self.h / 2))
        frost(frame, x0, y0, self.mask, alpha, self.dark)
        blit(frame, self.edge, x0 + self.w / 2, y0 + self.h / 2, 1.0, 0, alpha)
        return x0, y0


def costume_label(num, name):
    """Glass name tag: a big neon number, a hairline divider and the costume name in poster capitals."""
    f_num = font(F_GOTH, 124, 800)
    num_txt = f"{num:02d}"
    size = 62
    while True:
        f_name = font(F_SERIF, size, 800)
        name_w = text_width(name.upper(), f_name, 6)
        num_w = f_num.getlength(num_txt)
        pad, gap = 40, 34
        w = int(pad + num_w + 2 * gap + 2 + name_w + pad + 6)
        if w <= W - 110 or size <= 40:
            break
        size -= 4
    w = max(w, 520)
    h = 150
    num_spr = to_sprite(lettering(num_txt, f_num, fill=(CREAM, PEACH), glow=(NEON, 16, 1.0), shadow=0))
    name_spr = to_sprite(lettering(name.upper(), f_name, 6, fill=(WHITE, CREAM), glow=((255, 120, 50), 9, 0.55),
                                   shadow=0))
    div = np.zeros((h - 56, 2), np.float32)
    div[:] = (np.sin(np.linspace(0, math.pi, h - 56)) ** 0.8 * 0.45)[:, None]
    div_spr = np.dstack([div, div * 0.95, div * 0.9, div])
    x_div = pad + num_w + gap
    extra = (w - (pad + num_w + 2 * gap + 2 + name_w + pad)) / 2
    return {
        "glass": Glass(w, h, 30),
        "num": (num_spr, pad + num_w / 2 + extra * 0.4, h / 2 + 2),
        "div": (div_spr, x_div + extra * 0.7, h / 2),
        "name": (name_spr, x_div + gap + 2 + name_w / 2 + extra, h / 2 + 2),
    }


def place_label(img, lab, cx, cy, alpha=1.0, num_alpha=1.0, div_alpha=1.0, reveal=1.0):
    """Draw a costume_label centred at (cx, cy); the name slides in as `reveal` goes 0 -> 1."""
    x0, y0 = lab["glass"].draw(img, cx, cy, alpha)
    for key, a in (("div", div_alpha), ("num", num_alpha)):
        spr, px, py = lab[key]
        blit(img, spr, x0 + px, y0 + py, 1.0, 0, a)
    spr, px, py = lab["name"]
    blit(img, wipe(spr, reveal), x0 + px, y0 + py, 1.0, 0, 1.0)


def ember_image(r=7):
    s = r * 8
    c = s / 2
    g = Image.new("L", (s, s), 0)
    ImageDraw.Draw(g).ellipse((c - r * 2.4, c - r * 2.4, c + r * 2.4, c + r * 2.4), fill=110)
    g = g.filter(ImageFilter.GaussianBlur(r * 1.3))
    core = Image.new("L", (s, s), 0)
    ImageDraw.Draw(core).ellipse((c - r * 0.6, c - r * 0.6, c + r * 0.6, c + r * 0.6), fill=255)
    core = core.filter(ImageFilter.GaussianBlur(r * 0.3))
    out = Image.new("RGBA", (s, s), (255, 120, 30, 0))
    out.putalpha(g)
    hot = Image.new("RGBA", (s, s), (255, 226, 170, 255))
    out.paste(hot, (0, 0), core)
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


def icon_image(kind, d=84, col=ORANGE):
    """Line icons for the contact cards (phone, pin, mail) in a glowing ring, drawn at 4x for smooth edges."""
    s = d * 4
    m = Image.new("L", (s, s), 0)
    dr = ImageDraw.Draw(m)
    lw = 18
    c = s / 2
    if kind == "phone":
        dr.rounded_rectangle((c - 62, c - 104, c + 62, c + 104), 26, outline=255, width=lw)
        dr.rounded_rectangle((c - 22, c + 64, c + 22, c + 78), 7, fill=255)
    elif kind == "pin":
        dr.ellipse((c - 78, c - 112, c + 78, c + 44), outline=255, width=lw)
        dr.line([(c - 64, c - 4), (c, c + 108), (c + 64, c - 4)], fill=255, width=lw, joint="curve")
        dr.ellipse((c - 26, c - 60, c + 26, c - 8), outline=255, width=lw - 4)
    else:
        dr.rounded_rectangle((c - 104, c - 72, c + 104, c + 72), 18, outline=255, width=lw)
        dr.line([(c - 92, c - 58), (c, c + 12), (c + 92, c - 58)], fill=255, width=lw, joint="curve")
    m = m.resize((d, d), Image.LANCZOS)
    pad = 30
    ring_d = d + 2 * pad
    out = Image.new("RGBA", (ring_d + 40, ring_d + 40), (0, 0, 0, 0))
    ring = Image.new("L", (ring_d * 4, ring_d * 4), 0)
    ImageDraw.Draw(ring).ellipse((6, 6, ring_d * 4 - 7, ring_d * 4 - 7), outline=255, width=10)
    ring = ring.resize((ring_d, ring_d), Image.LANCZOS)
    fillc = Image.new("L", (ring_d * 4, ring_d * 4), 0)
    ImageDraw.Draw(fillc).ellipse((6, 6, ring_d * 4 - 7, ring_d * 4 - 7), fill=46)
    fillc = fillc.resize((ring_d, ring_d), Image.LANCZOS)
    full = Image.new("L", out.size, 0)
    full.paste(fillc, (20, 20))
    lay = Image.new("RGBA", out.size, col + (0,))
    lay.putalpha(full)
    out = Image.alpha_composite(out, lay)
    lines = Image.new("L", out.size, 0)
    lines.paste(ring, (20, 20))
    icon = Image.new("L", out.size, 0)
    icon.paste(m, (20 + pad, 20 + pad))
    lines = Image.fromarray(np.maximum(np.asarray(lines), np.asarray(icon)))
    glow = lines.filter(ImageFilter.GaussianBlur(8)).point(lambda v: min(255, v * 2))
    lay = Image.new("RGBA", out.size, NEON + (0,))
    lay.putalpha(glow)
    out = Image.alpha_composite(out, lay)
    lay = Image.new("RGBA", out.size, PEACH + (0,))
    lay.putalpha(lines)
    return Image.alpha_composite(out, lay)


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


def halo_image(w, h, col=VIOLET, strength=0.55):
    """A soft oval glow to sit behind a logo."""
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).ellipse((w * 0.2, h * 0.2, w * 0.8, h * 0.8), fill=int(255 * strength))
    m = m.filter(ImageFilter.GaussianBlur(min(w, h) * 0.14))
    out = Image.new("RGBA", (w, h), col + (0,))
    out.putalpha(m)
    return out


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
    """Moody Halloween grade: more contrast, warm highlights, cold purple shadows (in strips, to keep memory low)."""
    out = np.empty_like(img)
    for y in range(0, img.shape[0], 256):
        x = img[y:y + 256].astype(np.float32) / 255.0
        x = np.clip((x - 0.5) * 1.1 + 0.48, 0, 1)
        lum = (x @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
        x = lum + (x - lum) * 1.12
        sh = (1 - lum[..., 0]) ** 2
        x[..., 0] += 0.03 * lum[..., 0] - 0.01 * sh
        x[..., 1] -= 0.012 * sh
        x[..., 2] += 0.05 * sh
        out[y:y + 256] = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    return out


def sharpen(img, sigma=1.6, amount=0.7):
    """Gentle unsharp mask (uint8 in, uint8 out): the photos were taken indoors on a phone and are a little soft."""
    return cv2.addWeighted(img, 1 + amount, cv2.GaussianBlur(img, (0, 0), sigma), -amount, 0)


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
        self.img = grade(sharpen(img))
        self.win_h = win_h * f
        self.hs, self.ws = self.img.shape[:2]
        self.cx, self.cy = self.clamp(cx * f, cy * f, self.win_h)

    def clamp(self, cx, cy, wh):
        ww = wh * 9 / 16
        cx = min(max(cx, ww / 2), self.ws - ww / 2) if ww < self.ws else self.ws / 2
        cy = min(max(cy, wh / 2), self.hs - wh / 2) if wh < self.hs else self.hs / 2
        return cx, cy

    def render(self, zoom=1.0, dx=0.0, offset=(0.0, 0.0)):
        wh = self.win_h * zoom
        cx, cy = self.clamp(self.cx + dx * wh * 9 / 16, self.cy, wh)
        s = H / wh
        M = np.array([[s, 0, W / 2 - s * cx + offset[0]], [0, s, H / 2 - s * cy + offset[1]]], np.float32)
        return cv2.warpAffine(self.img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)

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


def photo_tile(crop, num, r=22, num_size=78):
    """A grid cell: rounded photo, dark fade at the bottom, hairline outline and a neon number (RGBA, 20 px margin)."""
    h, w = crop.shape[:2]
    im = Image.fromarray((crop * 255).astype(np.uint8)).convert("RGBA")
    shade = Image.new("RGBA", (w, h), DEEP + (0,))
    a = np.clip((np.arange(h) - h * 0.5) / (h * 0.5), 0, 1) ** 1.3 * 220
    shade.putalpha(Image.fromarray(np.repeat(a[:, None], w, 1).astype(np.uint8)))
    im = Image.alpha_composite(im, shade)
    im.putalpha(Image.fromarray((rounded_mask(w, h, r) * 255).astype(np.uint8)))
    out = Image.new("RGBA", (w + 40, h + 40), (0, 0, 0, 0))
    out.alpha_composite(im, (20, 20))
    out.alpha_composite(outline_image(w, h, r, (CREAM, CREAM), 2, 0.55), (20, 20))
    n = lettering(f"{num:02d}", font(F_GOTH, num_size, 800), fill=(CREAM, PEACH), glow=(NEON, 12, 1.0), shadow=0)
    out.alpha_composite(n, (20 + 14 - 20, 20 + h - n.height + 22))
    return out


class Clip:
    """An OpenGen clip, played back slowed to fill `length` seconds with frame blending.

    Frames are decoded as they are needed (the reel plays forward), so only a couple sit in memory."""

    def __init__(self, path, length):
        self.path = path
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.fps = 24.0
        self.speed = min(1.0, (total - 1) / self.fps / length)
        self.count = total
        self.cap, self.next_i, self.cache = None, 0, {}

    def frame(self, i):
        if i in self.cache:
            return self.cache[i]
        if self.cap is None or i < self.next_i:
            if self.cap is not None:
                self.cap.release()
            self.cap, self.next_i = cv2.VideoCapture(str(self.path)), 0
        while self.next_i <= i:
            ok, fr = self.cap.read()
            if not ok:
                return self.cache.get(self.next_i - 1, next(iter(self.cache.values())))
            if self.next_i >= i - 1:
                if fr.shape[1] != W or fr.shape[0] != H:
                    fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_CUBIC)
                self.cache[self.next_i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                for old in [k for k in self.cache if k < self.next_i - 2]:
                    del self.cache[old]
            self.next_i += 1
        return self.cache[i]

    def at(self, t, zoom=1.0, offset=(0, 0)):
        fi = max(0.0, min(self.count - 1.001, t * self.speed * self.fps))
        i = int(fi)
        k = fi - i
        a = self.frame(i).astype(np.float32)
        img = a if k < 0.02 else a * (1 - k) + self.frame(i + 1).astype(np.float32) * k
        img /= 255.0
        if zoom != 1.0 or offset != (0, 0):
            M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, zoom)
            M[0, 2] += offset[0]
            M[1, 2] += offset[1]
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        return img


# ---------- horror transitions ----------
FX_WINDOWS = {  # (seconds before the cut, seconds after)
    "lightning": (0.0, 0.42), "glitch": (0.10, 0.16), "ghost": (0.0, 0.45),
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
            out[y:y + hh] = out[y:y + hh] * 0.6 + np.array([0.45, 0.12, 0.05], np.float32) * 0.4 * amt
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


# ---------- timeline ----------
def build_timeline():
    segs = [{"kind": "cold", "t0": 0.0, "t1": COLD_LEN}]
    segs.append({"kind": "title", "b0": 0, "b1": TITLE_BEATS, "fx": None})
    b = TITLE_BEATS
    for i, s in enumerate(SHOTS):
        segs.append({"kind": "shot", "b0": b, "b1": b + s["beats"], "shot": s, "num": i + 1, "fx": s.get("fx", "zoom")})
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
        self.labels = {sg["i"]: costume_label(sg["num"], sg["shot"]["label"])
                       for sg in self.segs if sg["kind"] == "shot"}
        self.logo_small = to_sprite(logo_image(150))
        self.logo_mid = to_sprite(logo_image(330))
        self.halo = to_sprite(halo_image(620, 520))
        # cold open
        self.c1 = to_sprite(lettering(COLD["line1"].upper(), font(F_SERIF, 74, 800), 10, fill=(WHITE, CREAM),
                                      glow=(VIOLET, 14, 0.9), shadow=0.8))
        self.c2 = to_sprite(lettering(COLD["line2"].upper(), font(F_SERIF, 50, 600), 16, fill=(CREAM, GHOST),
                                      glow=(VIOLET, 10, 0.7), shadow=0.8))
        # title
        self.t_big = to_sprite(fit_width(lettering(TITLE["big"], font(F_GOTH, 290, 900), fill=(CREAM, PEACH),
                                                   glow=(NEON, 26, 1.1), shadow=0.8), 1040))
        sub = lettering(TITLE["sub"].upper(), font(F_SANS, 46, 600), 22, fill=(WHITE, WHITE), shadow=0.8)
        self.t_sub = to_sprite(flanked(sub, 110, 30))
        cta = lettering(TITLE["cta"].upper(), font(F_SANS, 30, 700), 9, fill=(PEACH, PEACH), shadow=0)
        self.t_cta = (Glass(cta.width + 80, 76, 38), to_sprite(cta))
        self.build_vote()
        self.build_outro()
        self.ember = to_sprite(ember_image())
        self.bats = [to_sprite(bat_image(math.cos(2 * math.pi * i / 8))) for i in range(8)]
        self.web = to_sprite(cobweb_image())
        self.bolts = [to_sprite(bolt_image(s)) for s in range(4)]
        self.fog = periodic_noise(H // 4, W // 4, 3)
        self.fog2 = periodic_noise(H // 4, W // 4, 11, ((4, 1.0), (10, 0.6)))
        self.bat_flights = self.plan_bats()

    # ----- builders -----
    def build_vote(self):
        self.v_big = to_sprite(fit_width(lettering(VOTE["big"], font(F_GOTH, 168, 800), fill=(CREAM, PEACH),
                                                   glow=(NEON, 22, 1.0), shadow=0.8), 1020))
        sub = lettering(VOTE["sub"].upper(), font(F_SANS, 32, 700), 11, fill=(WHITE, WHITE), shadow=0.8)
        self.v_sub = to_sprite(flanked(sub, 70, 24))
        cols, margin, gap = 4, 44, 18
        self.cell_w = (W - 2 * margin - gap * (cols - 1)) // cols
        self.cell_h = int(self.cell_w * 4 / 3)
        self.grid = (margin, gap, cols)
        self.cells = [to_sprite(photo_tile(self.photo(s).crop(self.cell_w, self.cell_h, 0.75), i + 1))
                      for i, s in enumerate(SHOTS)]
        self.cell_glow = to_sprite(outline_image(self.cell_w, self.cell_h, 22, (NEON, PEACH, NEON), 3, 1.0, glow=10))
        # haunted-house night scene, blurred and darkened, behind the grid
        bg = cv2.cvtColor(cv2.imread(str(OPENGEN / "intro_night_scene.jpg")), cv2.COLOR_BGR2RGB)
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        self.vote_bg = cv2.GaussianBlur(bg, (0, 0), 16) * 0.34

    def build_outro(self):
        self.o_logo = to_sprite(logo_image(380))
        self.o_l1 = to_sprite(lettering(OUTRO["line1"].upper(), font(F_SERIF, 62, 700), 16, fill=(WHITE, CREAM),
                                        glow=(VIOLET, 12, 0.8), shadow=0.8))
        self.o_l2 = to_sprite(fit_width(lettering(OUTRO["line2"], font(F_GOTH, 250, 900), fill=(CREAM, PEACH),
                                                  glow=(NEON, 26, 1.1), shadow=0.8), 1000))
        self.o_tag = to_sprite(flanked(lettering(OUTRO["tag"].upper(), font(F_SANS, 27, 600), 8, fill=(GHOST, GHOST),
                                                 shadow=0.8), 60, 20, GHOST))
        f_main = font(F_SANS, 48, 700)
        rows = [
            ("phone", [lettering(BRAND["phone"], f_main, 1, fill=(WHITE, WHITE), shadow=0)]),
            ("pin", [lettering("10629 West Side Drive", font(F_SANS, 42, 700), 1, fill=(WHITE, WHITE), shadow=0),
                     lettering("Grande Prairie, Alberta T8V 8E6", font(F_SANS, 33, 500), 1, fill=(CREAM, CREAM),
                               shadow=0, alpha=0.8)]),
            ("mail", [lettering(BRAND["email"], f_main, 1, fill=(WHITE, WHITE), shadow=0)]),
        ]
        texts = [lines[0] if len(lines) == 1 else self.left_stack(lines, 10) for _, lines in rows]
        card_w = max(t.width for t in texts) + 150 + 50  # icon column + right padding, same width for all three
        self.o_cards = []
        for (kind, lines), text in zip(rows, texts):
            glass = Glass(card_w, 124 if len(lines) == 1 else 160, 32)
            self.o_cards.append((glass, to_sprite(icon_image(kind, 52)), to_sprite(text)))
        self.o_stay = to_sprite(lettering(OUTRO["stay"].upper(), font(F_SANS, 25, 600), 6, fill=(CREAM, CREAM),
                                          shadow=0.8, alpha=0.85))
        self.o_web = to_sprite(lettering(f"{BRAND['web']}   ·   {BRAND['instagram']}", font(F_SANS, 34, 700), 3,
                                         fill=(PEACH, PEACH), glow=(NEON, 8, 0.5), shadow=0.6))

    @staticmethod
    def left_stack(images, gap):
        w = max(i.width for i in images)
        h = sum(i.height for i in images) + gap * (len(images) - 1)
        out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        y = 0
        for im in images:
            out.alpha_composite(im, (0, y))
            y += im.height + gap
        return out

    def plan_bats(self):
        flights = []  # (start, duration, y0, y1, left_to_right, scale)
        flights += [(0.15, 1.5, 420, 240, True, 0.9), (0.55, 1.4, 1500, 1300, False, 0.65)]
        title = self.seg("title")
        flights += [(title["t0"] + 0.05, 1.4, 600, 420, False, 0.8)]
        vote = self.seg("vote")
        flights += [(vote["t0"] + 0.5, 1.5, 250, 160, False, 0.55)]
        end = self.seg("outro")
        flights += [(end["t0"] + 0.4, 2.0, 520, 300, True, 0.8), (end["t0"] + 2.5, 2.2, 230, 130, False, 0.5),
                    (end["t0"] + 3.9, 1.8, 980, 760, True, 0.55)]
        return flights

    def seg(self, kind):
        return next(s for s in self.segs if s["kind"] == kind)

    def photo(self, shot):
        key = shot["photo"]
        if key not in self.photos:
            if len(self.photos) > 3:
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
                self.clips[name] = Clip(OPENGEN / "outro_haunted.mp4", end["t1"] - end["t0"])
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

    def draw_embers(self, img, t, n=16, strength=1.0):
        """Glowing embers drifting up and swaying, twinkling as they go."""
        for i in range(n):
            speed = 70 + 90 * hash01(i, 1)
            y = H + 60 - ((t * speed + hash01(i, 2) * (H + 120)) % (H + 120))
            x = hash01(i, 3) * W + 34 * math.sin(t * (0.8 + hash01(i, 4)) + i)
            tw = 0.55 + 0.45 * math.sin(t * (3 + 4 * hash01(i, 5)) + i * 1.7)
            life = clamp01((H + 60 - y) / 300) * clamp01(y / 500)
            a = tw * life * strength
            if a > 0.02:
                blit(img, self.ember, x, y, 0.45 + 0.7 * hash01(i, 6), 0, a)

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

    def neon(self, t, seed, on_at):
        """Brightness of a neon word: stutters on like a failing tube, then hums with the odd dip."""
        since = t - on_at
        if since < 0:
            return 0.0
        if since < 0.4:
            return flicker(t, seed, depth=0.95, rate=30)
        return 1.0 if hash01(int(t * 12), seed, 3) > 0.04 else 0.55

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

    def draw_label(self, img, lab, t, since, seed):
        p = (since - 0.03) / 0.3
        place_label(img, lab, W / 2, 400 + 34 * (1 - ease_out_cubic(p)), alpha=clamp01(p * 2.2),
                    num_alpha=self.neon(t, seed, t - since + 0.06), div_alpha=clamp01((since - 0.1) / 0.2),
                    reveal=ease_out_cubic((since - 0.12) / 0.42))

    def photo_scene(self, sg, t):
        t = min(max(t, sg["t0"]), sg["t1"] - 1e-4)
        zoom, dx = self.kb(sg, t)
        img = self.photo(sg["shot"]).render(zoom, dx).astype(np.float32) / 255.0
        img = self.darken(img)
        img = self.draw_fog(img, t, 0.3, 0.74)
        blit(img, self.web, W - 165, 165, 1.0, 2 * math.sin(t * 1.3), 0.5, pivot=(330, 0))
        self.draw_embers(img, t, 12, 0.8)
        self.draw_label(img, self.labels[sg["i"]], t, t - sg["t0"], sg["i"] * 7 + 1)
        blit(img, self.logo_small, 104, 200, 1.0, 0, 0.95 * flicker(t, 3, depth=0.25))
        return img

    def cold_scene(self, sg, t):
        clip = self.clip("intro")
        img = clip.at(t, 1.06 - 0.03 * t / COLD_LEN)
        img = self.darken(img, 1.3)
        img = img * 0.6  # night
        img = self.draw_fog(img, t, 0.55, 0.6, speed=34)
        # appear out of black on the first thunder strike
        img = img * clamp01((t - 0.06) / 0.1) if t < 0.16 else img
        img = self.lightning_flash(img, t - 0.1, 1.0, seed=1)
        img = self.lightning_flash(img, t - 1.0, 0.5, seed=2, bolt=False)
        p1 = (t - 0.28) / 0.25
        a1 = clamp01(p1 * 2) * (flicker(t, 11, depth=0.9) if p1 < 1.3 else 1.0)
        blit(img, self.c1, W / 2, 760, 1.0 + 0.05 * (1 - ease_out_cubic(p1)), 0, a1)
        p2 = (t - 0.55) / 0.25
        a2 = clamp01(p2 * 2) * (flicker(t, 12, depth=0.9) if p2 < 1.3 else 1.0)
        blit(img, self.c2, W / 2, 860, 1.0, 0, a2)
        p3 = (t - 0.85) / 0.35
        a3 = clamp01(p3 * 2) * (flicker(t, 13, depth=0.8) if p3 < 1.2 else 1.0)
        blit(img, self.halo, W / 2, 1110, 1.0, 0, a3 * 0.8)
        blit(img, self.logo_mid, W / 2, 1110 + 6 * math.sin(t * 3), 0.85 + 0.15 * ease_out_back(p3), 0, a3)
        self.draw_bats(img, t)
        # glitch build-up into the music
        g = clamp01((t - (COLD_LEN - 0.3)) / 0.3)
        if g > 0:
            img = fx_glitch(img, g * 0.9, int(t * FPS))
        return img

    def title_scene(self, sg, t):
        since = t - sg["t0"]
        clip = self.clip("intro")
        img = clip.at(t, 1.03 - 0.03 * since / (sg["t1"] - sg["t0"]))
        img = self.darken(img, 1.2) * 0.62
        img = self.draw_fog(img, t, 0.55, 0.6, speed=34)
        self.draw_embers(img, t, 18, 1.0)
        b = lambda k: k * BEAT  # noqa: E731
        # logo rises from the cold open to the top
        p = ease_in_out(since / 0.45)
        y = 1110 + (330 - 1110) * p
        blit(img, self.halo, W / 2, y, 1.0 - 0.18 * p, 0, 0.8)
        blit(img, self.logo_mid, W / 2, y, 1.0 - 0.18 * p, 0, 1.0)
        # "Halloween" slams on the first beat, then hums like neon
        ps = since / 0.2
        blit(img, self.t_big, W / 2, 830, 1.0 + 0.6 * (1 - ease_out_cubic(ps)), 0,
             clamp01(ps * 3) * (self.neon(t, 61, sg["t0"] + 0.5) if since > 0.5 else 1.0))
        p2 = (since - b(1)) / 0.35
        blit(img, wipe(self.t_sub, ease_out_cubic(p2)), W / 2, 1010, 1.0, 0, 1.0)
        p3 = (since - b(2.2)) / 0.3
        if p3 > 0:
            glass, txt = self.t_cta
            cy = 1120 + 30 * (1 - ease_out_cubic(p3))
            glass.draw(img, W / 2, cy, clamp01(p3 * 2))
            blit(img, txt, W / 2, cy, 1.0, 0, clamp01(p3 * 2) * flicker(t, 62, depth=0.2))
        self.draw_bats(img, t)
        img = self.lightning_flash(img, since, 1.0, seed=3)
        sx = 26 * math.exp(-since / 0.12) * math.sin(since * 95)
        sy = 18 * math.exp(-since / 0.12) * math.cos(since * 80)
        return shake(img, sx, sy)

    def vote_scene(self, sg, t):
        since = t - sg["t0"]
        img = self.vote_bg * (0.92 + 0.08 * flicker(t, 22, depth=0.5, rate=10))
        img = self.draw_fog(img, t, 0.5, 0.35, speed=20, color=(0.36, 0.24, 0.46))
        self.draw_embers(img, t, 10, 0.6)
        p = since / 0.25
        blit(img, self.v_big, W / 2, 285, 1.0 + 0.35 * (1 - ease_out_cubic(p)), 0,
             clamp01(p * 3) * (self.neon(t, 71, sg["t0"] + 0.4) if since > 0.4 else 1.0))
        p = (since - 0.18) / 0.35
        blit(img, wipe(self.v_sub, ease_out_cubic(p)), W / 2, 405, 1.0, 0, 1.0)
        margin, gap, cols = self.grid
        cw, ch = self.cell_w, self.cell_h
        top = 470
        # a glowing frame hops between costumes on every half beat, like a roulette
        hop = int(since / (BEAT / 2)) if since > 0.6 else -1
        pick = int(hash01(hop, 5) * len(self.cells)) if hop >= 0 else -1
        for i, cell in enumerate(self.cells):
            c, r = i % cols, i // cols
            x = margin + c * (cw + gap) + cw / 2
            y = top + r * (ch + gap) + ch / 2
            q = (since - 0.15 - (r * cols + c) * 0.035) / 0.25
            if q <= 0:
                continue
            a = clamp01(q * 2.5) * (flicker(t, 80 + i, depth=0.8) if q < 1.4 else 1.0)
            blit(img, cell, x, y + 30 * (1 - ease_out_cubic(q)), 1.0, 0, a)
            if i == pick:
                blit(img, self.cell_glow, x, y, 1.0, 0, 0.95)
        self.draw_bats(img, t)
        return img

    def outro_scene(self, sg, t):
        since = t - sg["t0"]
        clip = self.clip("outro")
        img = clip.at(since, 1.0 + 0.05 * since / (sg["t1"] - sg["t0"]))
        img = self.darken(img, 0.95) * 0.82
        img = img * (0.9 + 0.1 * flicker(t, 31, depth=0.4, rate=12))
        img = self.draw_fog(img, t, 0.45, 0.62, speed=22)
        self.draw_embers(img, t, 20, 0.9)
        strike = 0.42
        # logo flickers on like a failing sign, then floats
        p = (since - 0.1) / 0.45
        a = clamp01(p * 2) * (flicker(t, 41, depth=0.95) if p < 1.3 else 1.0)
        ly = 330 + 8 * math.sin(since * 2.2)
        blit(img, self.halo, W / 2, ly, 1.15, 0, a * 0.9)
        blit(img, self.o_logo, W / 2, ly, 1.0, 0, a)
        # THANKS FOR / Watching arrive on the lightning strike; the neon then hums and glitches now and then
        q = (since - strike) / 0.22
        jit = since > strike + 1.0 and int(since * FPS) % 41 < 2
        blit(img, self.o_l1, W / 2 + (6 if jit else 0), 625, 1.0 + 0.25 * (1 - ease_out_cubic(q)), 0, clamp01(q * 3))
        blit(img, self.o_l2, W / 2 - (9 if jit else 0), 765, 1.0 + 0.5 * (1 - ease_out_cubic(q)), 0,
             clamp01(q * 3) * (self.neon(t, 42, sg["t0"] + strike + 0.5) if since > strike + 0.5 else 1.0))
        q = (since - 0.85) / 0.4
        blit(img, wipe(self.o_tag, ease_out_cubic(q)), W / 2, 905 + 3 * math.sin(since * 2), 1.0, 0, 1.0)
        # glass contact cards slide in from alternating sides, glitching as they land, then drift
        y = 1030
        for k, (glass, icon, text) in enumerate(self.o_cards):
            q = (since - 1.15 - 0.2 * k) / 0.32
            cy = y + glass.h / 2
            y += glass.h + 22
            if q <= 0:
                continue
            e = ease_out_cubic(q)
            side = -1 if k % 2 == 0 else 1
            jitter = 0 if q >= 1 else (hash01(k, int(since * FPS)) - 0.5) * 60 * (1 - q)
            cx = W / 2 + side * 260 * (1 - e) + jitter
            cyk = cy + 4 * math.sin(since * 2.1 + k)
            al = clamp01(q * 2) * (flicker(t, 50 + k, depth=0.6) if q < 1.3 else 1.0)
            x0, y0 = glass.draw(img, cx, cyk, al)
            blit(img, icon, x0 + 72, y0 + glass.h / 2, 1.0, 0, al)
            blit(img, text, x0 + 150 + text.shape[1] / 2, y0 + glass.h / 2 + 1, 1.0, 0, al)
        q = (since - 2.0) / 0.45
        blit(img, self.o_stay, W / 2, y + 30, 1.0, 0, clamp01(q))
        q = (since - 2.25) / 0.45
        blit(img, self.o_web, W / 2, y + 86, 1.0, 0, clamp01(q) * (self.neon(t, 43, sg["t0"] + 2.25)))
        self.draw_bats(img, t)
        img = self.lightning_flash(img, since - strike, 1.0, seed=5)
        img = self.lightning_flash(img, since - 3.7, 0.4, seed=6, bolt=False)
        sx = 20 * math.exp(-(since - strike) / 0.12) * math.sin(since * 95) if since > strike else 0
        return shake(img, sx, 0)

    def scene(self, sg, t):
        kind = sg["kind"]
        if kind == "cold":
            return self.cold_scene(sg, t)
        if kind == "title":
            return self.title_scene(sg, t)
        if kind == "shot":
            return self.photo_scene(sg, t)
        if kind == "vote":
            return self.vote_scene(sg, t)
        return self.outro_scene(sg, t)

    def frozen(self, sg):
        """The last frame of a segment (for the ghost transition), cached."""
        if sg["i"] not in self.scene_cache:
            self.scene_cache = {sg["i"]: self.scene(sg, sg["t1"] - 1e-3)}
        return self.scene_cache[sg["i"]]

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
    riser_at = 0.2  # the riser peaks 3.6 s into its file; trim it so the peak lands on the title slam
    ev = [("thunder", 0.06, -2, 0.0, 3.4), ("ghost_whoosh", 0.3, -8, 0, None),
          ("riser", riser_at, -6, 3.6 - (COLD_LEN - riser_at), None),
          ("glitch", COLD_LEN - 0.3, -10, 0.0, 0.35), ("impact", COLD_LEN, -3, 0, None)]
    sounds = {"lightning": ("thunder", -9, 0.0, 1.6), "glitch": ("glitch", -13, 0.0, 0.35),
              "ghost": ("ghost_whoosh", -9, 0, None), "fog": ("whoosh", -12, 0, None)}
    for sg in r.segs:
        fx = sg.get("fx")
        if fx in sounds and sg["kind"] != "title":
            name, g, s0, ln = sounds[fx]
            lead = FX_WINDOWS[fx][0] if fx == "fog" else 0.0
            ev.append((name, sg["t0"] - lead - (0.15 if name == "whoosh" else 0), g, s0, ln))
    end = r.seg("outro")
    ev += [("thunder", end["t0"] + 0.42, -5, 0.0, 3.0), ("witch", end["t0"] + 2.2, -10, 0, None)]
    return ev


def build_audio(r, music=True):
    sr = 48000
    n = int(math.ceil(r.total * sr))
    mix = np.zeros((n, 2), np.float32)
    if music:
        dur = r.total - COLD_LEN
        m = decode(MUSIC, T0, dur)
        fade_n = int(1.6 * sr)
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
    # nothing rings past the end
    tail = int(0.25 * sr)
    mix[-tail:] *= np.linspace(1, 0, tail)[:, None]
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


def out_of_memory(e):
    return isinstance(e, MemoryError) or (isinstance(e, cv2.error) and "Insufficient memory" in str(e))


def render_frames(r, raw_path):
    """Render every frame into a raw RGB file on disk, so a crash can resume where it stopped and the
    encoder never runs at the same time as the renderer. Waits out moments when the PC is short of memory."""
    n = int(math.ceil(r.total * FPS))
    fb = W * H * 3
    done = min(n, raw_path.stat().st_size // fb) if raw_path.exists() else 0
    with open(raw_path, "r+b" if raw_path.exists() else "wb") as fh:
        fh.seek(done * fb)
        fh.truncate()
        for i in range(done, n):
            for attempt in range(60):
                try:
                    fr = r.frame(i / FPS)
                    data = (np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes()
                    break
                except (MemoryError, cv2.error) as e:
                    if not out_of_memory(e):
                        raise
                    # drop everything that reloads on demand, then try the frame again
                    r.photos.clear()
                    r.clips.clear()
                    r.scene_cache.clear()
                    gc.collect()
                    time.sleep(5)
            else:
                raise SystemExit("still out of memory after 5 minutes; close some programs and run again to resume")
            fh.write(data)
            if i % 150 == 0:
                print(f"  frame {i}/{n}", flush=True)
    return n


def encode(raw_path, out_path, audio, dur):
    cmd =["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", str(raw_path), "-i", str(audio),
           "-filter_complex", "[0:v]scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[v]",
           "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high",
           "-maxrate", "20M", "-bufsize", "40M", "-x264-params", "rc-lookahead=20", "-threads", "4",
           "-g", str(FPS * 2), "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-bsf:v", "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{dur:.3f}", "-movflags", "+faststart",
           str(out_path)]
    for attempt in range(5):
        if subprocess.run(cmd).returncode == 0:
            print("wrote", out_path)
            return
        time.sleep(10)
    raise SystemExit("ffmpeg failed")


def phone_copy(src, out_path, mib=29.0):
    """The same full-HD reel as a two-pass encode sized to fit under phone and web file limits (30 MiB)."""
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(src)], capture_output=True, text=True, check=True).stdout)
    kbps = int(mib * 1024 * 1024 * 8 / dur / 1000) - 192 - 100  # leave room for the audio and the container
    log = str(ROOT / "work" / "x264pass")
    common = ["-c:v", "libx264", "-preset", "slow", "-b:v", f"{kbps}k", "-maxrate", f"{kbps * 2}k",
              "-bufsize", f"{kbps * 4}k", "-profile:v", "high", "-pix_fmt", "yuv420p", "-g", str(FPS * 2),
              "-x264-params", "rc-lookahead=20", "-threads", "4", "-passlogfile", log]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *common, "-pass", "1", "-an", "-f", "null",
                    "-"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *common, "-pass", "2", "-c:a", "copy",
                    "-movflags", "+faststart", str(out_path)], check=True)
    print("wrote", out_path, f"({out_path.stat().st_size / 2 ** 20:.1f} MiB)")


def remux(video, audio, out_path):
    """Same picture, different soundtrack (no re-encode of the video)."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio), "-map", "0:v",
                    "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest",
                    "-movflags", "+faststart", str(out_path)], check=True)
    print("wrote", out_path)


def main():
    EXPORTS.mkdir(exist_ok=True)
    r = Renderer()
    print(f"{len(r.segs)} segments, {r.total:.2f} s")
    raw = ROOT / "work" / "frames.rgb"
    render_frames(r, raw)
    music, sfx_only = build_audio(r, music=True), build_audio(r, music=False)
    dur = r.total
    del r  # free the renderer's memory before the encoder starts
    gc.collect()
    reel = EXPORTS / "tinystars_halloween_reel.mp4"
    encode(raw, reel, music, dur)
    remux(reel, sfx_only, EXPORTS / "tinystars_halloween_reel_nomusic.mp4")
    phone_copy(reel, EXPORTS / "tinystars_halloween_reel_phone.mp4")
    raw.unlink()


if __name__ == "__main__":
    main()
