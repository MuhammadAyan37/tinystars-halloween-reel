"""Render "Recap of Last Week": the Tiny Stars Halloween week as a 25-second film trailer, 1080x1920, 30 fps.

The story runs dark to bright. It opens in a dark classroom where a flashlight finds a hanging skeleton and lights up the
title, then walks the haunted halls at dusk while the song is still quiet. A dive through the wreath lands on the band
coming in, the lights come up and the kids take over: real footage, then each star freezes into a cut-out sticker with a
movie credit. A wall of costumes scrolls on the beat, a star wipe opens the closing card, and the cast hop in for a
curtain call on the song's last notes.

Only the blurred working copies from recap_blur.py are ever read here, so no effect can touch an unblurred face.

Usage: python -I edits/recap.py [--stills t t t ...]
  -> exports/recap_last_week.mp4, recap_last_week_nomusic.mp4, recap_cover.jpg, and recap_last_week_phone.mp4 if the
     master is too big to send to a phone (--stills only writes check frames to work/recap/stills/)
"""
import gc
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render as R  # noqa: E402
from content import BRAND, FAMILY, SHOTS as PHOTO_SHOTS  # noqa: E402
from recap_content import CAPTIONS, END, MUSIC, SHOTS, TITLE, WALL  # noqa: E402
from render import (CREAM, FPS, Glass, H, W, WHITE, blit, clamp01, decode, ease_in_out, ease_out_back,  # noqa: E402
                    ease_out_cubic, encode, fit_width, halo_image, hash01, icon_image, lettering, logo_image, phone_copy,
                    remux, render_frames, rounded_mask, sf, shake, text_width, to_sprite)

ROOT = R.ROOT
CLEAN = ROOT / "work" / "recap" / "clean"
FACES = ROOT / "work" / "recap" / "faces"
TRACKS = ROOT / "work" / "recap" / "tracks"
MASKS = ROOT / "work" / "recap" / "masks"
SFX = R.SFX
EXPORTS = R.EXPORTS

BEAT = 60 / MUSIC["bpm"]
DROP_BEAT = MUSIC["lift"]  # the song's quiet opening ends here and the band comes in

# palette: the daycare's own purple and gold, plus pumpkin
AUBERGINE = (30, 12, 44)
PLUM = (100, 42, 115)
GOLD = (246, 199, 37)
GOLD_LIGHT = (255, 226, 130)
PUMPKIN = (255, 122, 26)
LILAC = (196, 160, 255)
F_DISPLAY = "Fraunces-Italic.ttf"
F_SMALL = "InterTight.ttf"


def bt(k):
    return k * BEAT


def display(size, weight=900):
    """Fraunces italic: optical size, weight, softness, 'wonk' (the slightly crooked, Halloween-ish letters)."""
    f = R.ImageFont.truetype(str(R.FONTS / F_DISPLAY), size)
    f.set_variation_by_axes([144, weight, 100, 1])
    return f


def small(size, weight=700):
    return R.font(F_SMALL, size, weight)


# ---------- the camera path of a clip, for steadying the footage ----------
class Path2D:
    """Where the camera went in each frame of a clip, from the motion measured by recap_detect.py."""

    def __init__(self, name):
        motion = json.loads((FACES / f"{name}.json").read_text())["motion"]
        self.raw = np.zeros((len(motion), 3), np.float64)  # x, y, angle
        for i, m in enumerate(motion):
            m = np.array(m)
            self.raw[i] = (self.raw[i - 1] if i else 0) + np.array([m[0, 2], m[1, 2], math.atan2(m[1, 0], m[0, 0])])

    def correction(self, sigma):
        """How far each frame sits from a smooth version of the camera's path."""
        r = int(sigma * 3)
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        k /= k.sum()
        pad = np.pad(self.raw, ((r, r), (0, 0)), mode="edge")
        smooth = np.stack([np.convolve(pad[:, c], k, "valid") for c in range(3)], 1)
        return smooth - self.raw


class Source:
    """A baked clip, read one frame at a time and scaled down cleanly, keeping just a few frames in memory."""

    def __init__(self, name, width):
        self.name, self.w, self.h = name, int(width), int(width * 16 / 9)
        self.cap, self.pos, self.cache = None, 0, {}
        self.n = json.loads((FACES / f"{name}.json").read_text())["n"]

    def frame(self, i):
        i = int(max(0, min(self.n - 1, i)))
        if i in self.cache:
            return self.cache[i]
        if self.cap is None or i < self.pos:
            if self.cap is not None:
                self.cap.release()
            self.cap, self.pos = cv2.VideoCapture(str(CLEAN / f"{self.name}.mp4")), 0
        while self.pos < i:
            self.cap.grab()
            self.pos += 1
        ok, fr = self.cap.read()
        self.pos += 1
        if not ok:
            raise SystemExit(f"could not read frame {i} of {self.name}")
        fr = cv2.cvtColor(cv2.resize(fr, (self.w, self.h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self.cache[i] = fr
        for k in sorted(self.cache)[:-5]:
            del self.cache[k]
        return fr

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.cap, self.cache = None, {}


# ---------- picture helpers ----------
def blend_flow(a, b, q):
    """A frame between a and b (uint8 RGB) by optical flow, for slow motion."""
    if q < 0.02:
        return a
    if q > 0.98:
        return b
    ga = cv2.cvtColor(cv2.resize(a, (540, 960), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, (540, 960), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    f_ab = dis.calc(ga, gb, None)
    f_ba = dis.calc(gb, ga, None)
    h, w = a.shape[:2]
    k = w / 540.0
    f_ab = cv2.resize(f_ab, (w, h), interpolation=cv2.INTER_LINEAR) * k
    f_ba = cv2.resize(f_ba, (w, h), interpolation=cv2.INTER_LINEAR) * k
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    wa = cv2.remap(a, xx + f_ba[..., 0] * q, yy + f_ba[..., 1] * q, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wb = cv2.remap(b, xx + f_ab[..., 0] * (1 - q), yy + f_ab[..., 1] * (1 - q), cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(wa, 1 - q, wb, q, 0)


def make_curve():
    """A gentle film S-curve for the whole reel."""
    x = np.arange(256, dtype=np.float32) / 255
    s = x + 0.09 * (x - 0.5) * (1 - np.abs(2 * x - 1))
    return (np.clip(s, 0, 1) * 255).astype(np.uint8)


CURVE = make_curve()
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
_rr = np.sqrt(((_xx - W / 2) / (W * 0.72)) ** 2 + ((_yy - H * 0.48) / (H * 0.72)) ** 2)
VIG = np.clip(_rr - 0.45, 0, 1)[..., None] ** 1.6
del _yy, _xx, _rr
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def grade(img, mood=0.0):
    """uint8 RGB -> float RGB. mood 0 is bright, warm daylight; mood 1 is dusk: low light and purple shadows."""
    x = cv2.LUT(img, CURVE).astype(np.float32) / 255
    lum = (x @ LUMA)[..., None]
    x = lum + (x - lum) * (1.14 - 0.1 * mood)
    x = x * np.array([1.025, 1.0, 0.965], np.float32)  # warm up the fluorescent classroom light
    if mood > 0:
        dusk = np.power(np.clip(x, 0, 1), 1.45) * 0.72
        shadow = (1 - np.clip(lum * 1.6, 0, 1)) * np.array([0.05, 0.012, 0.11], np.float32)
        x = x * (1 - mood) + (dusk + shadow) * mood
    x = x * (1 - VIG * (0.28 + 0.42 * mood))
    return x


def add_flash(img, k, col=(1.0, 0.97, 0.9)):
    return img + (np.array(col, np.float32) - img) * k


def dir_blur(img, length, vertical=False):
    if length < 3:
        return img
    length = int(length) | 1
    return cv2.blur(img, (1, length) if vertical else (length, 1), borderType=cv2.BORDER_REFLECT)


def zoom_blur(img, cx, cy, amount, n=6):
    """A burst of speed towards (cx, cy)."""
    if amount < 0.004:
        return img
    out = img * (1.0 / n)
    for i in range(1, n):
        s = 1 + amount * i / (n - 1)
        M = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]], np.float32)
        out = out + cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101) / n
    return out


def light_leak(img, q, strength=1.0):
    """A warm film-style light leak sweeping across the frame (q 0..1)."""
    if not 0 < q < 1:
        return img
    x = np.linspace(0, 1, W // 8, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, H // 8, dtype=np.float32)[:, None]
    c = -0.3 + 1.6 * q + 0.25 * (y - 0.5)
    band = np.exp(-((x - c) / 0.3) ** 2) * math.sin(math.pi * q) * strength
    band = cv2.resize(band, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    col = np.array([1.0, 0.5, 0.16], np.float32) * (1 - 0.45 * band) + np.array([0.9, 0.24, 0.5], np.float32) * 0.45 * band
    return 1 - (1 - img) * (1 - np.clip(band * col, 0, 1))


def star_image(size, col):
    """A soft five-pointed star, like the one in the logo."""
    k = 4
    s = size * k
    pts = []
    for i in range(10):
        r = (0.5 if i % 2 == 0 else 0.23) * s
        a = -math.pi / 2 + i * math.pi / 5
        pts.append((s / 2 + r * math.cos(a), s / 2 + r * math.sin(a)))
    m = Image.new("L", (s, s), 0)
    R.ImageDraw.Draw(m).polygon(pts, fill=255)
    m = m.filter(R.ImageFilter.GaussianBlur(k * 0.6)).resize((size, size), Image.LANCZOS)
    out = Image.new("RGBA", (size, size), col + (0,))
    out.putalpha(m)
    return out


class Words:
    """A line of text that arrives word by word, each word rising into place."""

    def __init__(self, text, f, cx, y, fill=(CREAM, CREAM), max_w=930, shadow=0.75, gap=None, align="center", lead=1.08):
        gap = gap if gap is not None else int(f.size * 0.25) - 44  # sprites carry their shadow's padding on both sides
        mid_x = (f.getbbox("x")[1] + f.getbbox("x")[3]) / 2
        lines = []
        for part in text.split("|"):  # a | is a line break
            cur, cur_w = [], 0
            for w in part.split():
                s = lettering(w, f, 0, fill=fill, shadow=shadow)
                if cur and cur_w + gap + s.width > max_w:
                    lines.append(cur)
                    cur, cur_w = [], 0
                cur_w += (gap if cur else 0) + s.width
                # sprites are cropped to their ink, so "up" and "They" need different offsets to share a baseline
                top, bottom = f.getbbox(w)[1], f.getbbox(w)[3]
                cur.append((s, (top + bottom) / 2 - mid_x))
            lines.append(cur)
        self.items = []
        line_h = f.size * lead
        self.width = max(sum(s.width for s, _ in line) + gap * (len(line) - 1) for line in lines)
        self.mid_y = y + (len(lines) - 1) * line_h / 2
        for li, line in enumerate(lines):
            lw = sum(s.width for s, _ in line) + gap * (len(line) - 1)
            x = cx - lw / 2 if align == "center" else cx
            yy = y + li * line_h
            for s, dy in line:
                self.items.append((to_sprite(s), x + s.width / 2, yy + dy))
                x += s.width + gap

    def draw(self, img, since, until=None, stagger=0.055):
        out = clamp01((until - since) / 0.16) if until is not None else 1.0
        for i, (spr, x, y) in enumerate(self.items):
            p = (since - i * stagger) / 0.3
            if p <= 0:
                continue
            blit(img, spr, x, y + 44 * (1 - ease_out_cubic(p)), 1.0, 0, clamp01(p * 1.8) * out)


# ---------- timeline ----------
def build_timeline():
    segs, beat = [], 0
    for i, sh in enumerate(SHOTS):
        segs.append(dict(kind="shot", i=i, shot=sh, b0=beat, b1=beat + sh["beats"]))
        beat += sh["beats"]
    segs.append(dict(kind="wall", b0=beat, b1=beat + WALL["beats"]))
    beat += WALL["beats"]
    segs.append(dict(kind="end", b0=beat, b1=beat + END["beats"]))
    for s in segs:
        s["t0"], s["t1"] = bt(s["b0"]), bt(s["b1"])
    return segs


class Renderer:
    def __init__(self):
        self.segs = build_timeline()
        self.total = self.segs[-1]["t1"]
        # render_frames() clears these when the PC runs short of memory
        self.photos, self.clips, self.scene_cache = {}, {}, {}
        self.paths = {}
        self.logo_small = to_sprite(logo_image(150))
        self.caps = [dict(c, w=Words(c["text"], display(c.get("size", 98)), W / 2, c["y"], max_w=c.get("max_w", 930)))
                     for c in CAPTIONS]
        self.title_big = lettering(TITLE["big"], display(TITLE["size"]), 0, fill=(CREAM, CREAM), shadow=0.35,
                                   glow=(PUMPKIN, 30, 0.3))
        self.title_big_alpha = np.asarray(self.title_big.convert("RGBA"), np.float32) / 255
        self.title_sub = to_sprite(lettering(TITLE["sub"], small(62, 800), 13, fill=(GOLD_LIGHT, GOLD), shadow=0.8))
        self.title_kick = to_sprite(lettering(TITLE["kicker"], small(30, 700), 10, fill=(CREAM, CREAM), shadow=0.8))
        self.stars = [to_sprite(star_image(s, c)) for s in (26, 40, 58) for c in (GOLD, PUMPKIN, LILAC, CREAM)]
        self.sparks = [to_sprite(star_image(s, GOLD_LIGHT)) for s in (24, 36, 52)]
        self.confetti = self.plan_confetti()
        self.cards = {}
        for sg in self.segs:
            if sg["kind"] == "shot" and sg["shot"].get("cast"):
                c = sg["shot"]["cast"]
                self.cards[sg["i"]] = dict(
                    kick=to_sprite(lettering(c["kicker"], small(36, 800), 13, fill=(GOLD_LIGHT, GOLD), shadow=0.8)),
                    name=to_sprite(fit_width(lettering(c["name"], display(c.get("size", 158)), 0,
                                                       fill=(CREAM, CREAM), shadow=0.85), 980)))
        self.wall_strips = None
        d = cv2.getGaussianKernel(25, 5)
        d = (d @ d.T / (d @ d.T).max()).astype(np.float32)[..., None]
        self.mote = np.dstack([np.repeat(d, 3, axis=2) * np.array([1.0, 0.92, 0.78], np.float32), d])
        self.build_end()

    # ----- a shot's camera -----
    def path(self, name):
        if name not in self.paths:
            self.paths[name] = Path2D(name)
        return self.paths[name]

    def source(self, sh):
        name = sh["clip"]
        if name not in self.clips:
            zmax = 2.0 if sh.get("through") else max(sh.get("zoom", (1.0, 1.0)))
            self.clips[name] = Source(name, max(1296, min(2160, 1080 * zmax * 1.15)))
        return self.clips[name]

    @staticmethod
    def src_frame(sh, tau):
        """Which frame of the clip is showing tau seconds into the shot (speed may change partway)."""
        speed = sh.get("speed", 1.0)
        if not isinstance(speed, (list, tuple)):
            return sh.get("start", 0) + tau * FPS * speed
        # a list of (beat in the shot, speed): integrate the piecewise-linear speed
        ts = np.array([p[0] * BEAT for p in speed])
        vs = np.array([p[1] for p in speed], np.float64)
        fine = np.linspace(0, tau, max(2, int(tau * 120) + 2))
        return sh.get("start", 0) + float(np.trapezoid(np.interp(fine, ts, vs), fine)) * FPS

    def matrix(self, sh, src, z, cx, cy, sw, extra=(0.0, 0.0)):
        """Clip-pixels to screen, including the steadying of the camera's shake."""
        s = W / sw * z
        ang, ox, oy = 0.0, 0.0, 0.0
        if sh.get("steady"):
            cor = self.path(sh["clip"]).correction(sh["steady"])
            c = cor[int(min(len(cor) - 1, max(0, src)))]
            k = sw / 2160.0
            ox, oy, ang = c[0] * k, c[1] * k, c[2]
        ca, sa = math.cos(ang) * s, math.sin(ang) * s
        h = sw * 16 / 9
        px, py = cx * sw, cy * h
        # never reveal the edge of the frame
        half_w, half_h = W / 2 / s, H / 2 / s
        ox = max(px + half_w - sw, min(px - half_w, ox)) if half_w * 2 <= sw else 0.0
        oy = max(py + half_h - h, min(py - half_h, oy)) if half_h * 2 <= h else 0.0
        return np.array([[ca, -sa, W / 2 - ca * px + sa * py + (ox * s) + extra[0]],
                         [sa, ca, H / 2 - sa * px - ca * py + (oy * s) + extra[1]]], np.float32)

    def camera(self, sh, tau, dur):
        """Zoom and centre for this moment: an eased push, a punch on the beat, an optional follow of a face."""
        z0, z1 = sh.get("zoom", (1.04, 1.0))
        z = z0 + (z1 - z0) * ease_in_out(tau / dur)
        if sh.get("punch"):
            z *= 1 + sh["punch"] * math.exp(-tau / 0.14)
        cx, cy = sh.get("center", (0.5, 0.5))
        follow = sh.get("follow")
        if follow is not None:
            f = self.follow_path(sh)
            k = min(len(f) - 1, max(0, self.src_frame(sh, tau)))
            i0 = int(k)
            i1 = min(len(f) - 1, i0 + 1)
            fx, fy = f[i0] + (f[i1] - f[i0]) * (k - i0)
            cx, cy = fx, fy
        half_w, half_h = 0.5 / z, 0.5 / z
        return z, max(half_w, min(1 - half_w, cx)), max(half_h, min(1 - half_h, cy))

    def follow_path(self, sh):
        key = ("follow", sh["clip"], sh["follow"])
        if key not in self.paths:
            data = json.loads((TRACKS / f"{sh['clip']}.json").read_text())
            t = next(t for t in data["tracks"] if t["id"] == sh["follow"])
            boxes = np.array(t["boxes"])
            face = np.stack([(boxes[:, 0] + boxes[:, 2] / 2) / 2160, (boxes[:, 1] + boxes[:, 3] / 2) / 3840], 1)
            full = np.zeros((data["n"], 2))
            for i in range(data["n"]):
                full[i] = face[min(len(face) - 1, max(0, i - t["start"]))]
            r = 30
            k = np.exp(-0.5 * (np.arange(-r, r + 1) / 10.0) ** 2)
            k /= k.sum()
            pad = np.pad(full, ((r, r), (0, 0)), mode="edge")
            smooth = np.stack([np.convolve(pad[:, c], k, "valid") for c in range(2)], 1)
            # keep the face where the shot wants it on screen, rather than dead centre
            off = sh.get("follow_offset", (0.0, 0.14))
            self.paths[key] = smooth + np.array(off)
        return self.paths[key]

    def sample(self, sh, tau, dur):
        """The picture for this moment of the shot, as uint8 RGB, at screen size."""
        src = self.source(sh)
        f = self.src_frame(sh, tau)
        speed = sh.get("speed", 1.0)
        if isinstance(speed, (list, tuple)):
            speed = float(np.interp(tau, [p[0] * BEAT for p in speed], [p[1] for p in speed]))
        z, cx, cy = self.camera(sh, tau, dur)
        M = self.matrix(sh, f, z, cx, cy, src.w)
        if speed >= 1.6:
            n = min(4, int(round(speed)))  # fast footage: blend the frames it skips, for real motion blur
            acc = None
            for j in range(n):
                fr = src.frame(f + j * speed / n)
                acc = fr.astype(np.float32) if acc is None else acc + fr
            pic = (acc / n).astype(np.uint8)
        elif abs(f - round(f)) < 0.04:
            pic = src.frame(round(f))
        else:
            i0 = int(math.floor(f))
            pic = blend_flow(src.frame(i0), src.frame(i0 + 1), f - i0)
        return cv2.warpAffine(pic, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)

    # ----- cast cards: the star freezes and becomes a sticker -----
    def cast_layers(self, sg):
        key = ("cast", sg["i"])
        if key in self.scene_cache:
            return self.scene_cache[key]
        for k in [k for k in self.scene_cache if k[0] == "cast"]:
            del self.scene_cache[k]
        sh, c = sg["shot"], sg["shot"]["cast"]
        dur = sg["t1"] - sg["t0"]
        tau = dur - c["beats"] * BEAT
        src = self.source(sh)
        f = c["frame"]
        z, cx, cy = c["camera"] if c.get("camera") else self.camera(sh, tau, dur)
        M = self.matrix(sh, f, z, cx, cy, src.w)
        fr = src.frame(f)
        pic = cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
        matte = cv2.imread(str(MASKS / f"{sh['clip']}_f{c['frame']}.png"), cv2.IMREAD_GRAYSCALE)
        if matte is None:
            raise SystemExit(f"no matte for {sh['clip']} frame {c['frame']}: run recap_masks.py")
        matte = cv2.resize(matte, (src.w, src.h), interpolation=cv2.INTER_AREA)
        m = cv2.warpAffine(matte, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32) / 255
        m = np.clip((m - 0.25) / 0.5, 0, 1)
        fg = grade(pic, 0.0)
        R_OUT = 11
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * R_OUT + 1, 2 * R_OUT + 1))
        edge = cv2.GaussianBlur(cv2.dilate(m, kernel), (0, 0), 1.3)
        shadow = np.roll(cv2.GaussianBlur(edge, (0, 0), 20), 16, axis=0) * 0.5
        # the sticker as premultiplied RGBA: shadow, then a cream border, then the child
        rgb = np.zeros((H, W, 3), np.float32)
        a = shadow.copy()
        rgb_edge = np.array(CREAM, np.float32) / 255
        rgb = rgb * (1 - edge[..., None]) + rgb_edge * edge[..., None]
        a = a * (1 - edge) + edge
        rgb = rgb * (1 - m[..., None]) + fg * m[..., None]
        sticker = np.dstack([rgb, a])
        back = cv2.GaussianBlur(fg, (0, 0), 16)
        lum = (back @ LUMA)[..., None]
        back = (back * 0.35 + lum * 0.65) * 0.42 * np.array([0.85, 0.72, 1.0], np.float32) \
            + np.array(AUBERGINE, np.float32) / 255 * 0.5
        layers = dict(fg=fg, back=back, sticker=sticker, tau=tau)
        self.scene_cache[key] = layers
        return layers

    def draw_cast(self, sg, tau):
        L = self.cast_layers(sg)
        c = sg["shot"]["cast"]
        since = tau - L["tau"]
        p = 1.0 if c.get("camera") else clamp01(since / 0.16)  # a pull-back cuts straight to the card, under the flash
        img = L["fg"] * (1 - p) + L["back"] * p
        s = 1 + 0.05 * ease_out_back(since / 0.26)
        cx, cy = W / 2, c.get("pivot_y", H * 0.6)
        M = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s) - 14 * ease_out_cubic(since / 0.26)]], np.float32)
        st = cv2.warpAffine(L["sticker"], M, (W, H), flags=cv2.INTER_LINEAR)
        img = img * (1 - st[..., 3:4] * p) + st[..., :3] * p
        if since < 0.14:
            img = add_flash(img, 0.55 * (1 - since / 0.14))
        card = self.cards[sg["i"]]
        y = c.get("text_y", 400)
        pk = (since - 0.06) / 0.22
        if pk > 0:
            blit(img, card["kick"], W / 2, y - 92 + 18 * (1 - ease_out_cubic(pk)), 1.0, 0, clamp01(pk * 2))
        pn = (since - 0.1) / 0.28
        if pn > 0:
            blit(img, card["name"], W / 2, y + 40 * (1 - ease_out_cubic(pn)), 0.92 + 0.08 * ease_out_back(pn), 0,
                 clamp01(pn * 2))
        for i in range(9):
            a0 = hash01(sg["i"], i, 1) * 2 * math.pi
            d = 250 + 260 * hash01(sg["i"], i, 2)
            q = (since - 0.08 - 0.05 * i) / 0.75
            if 0 < q < 1:
                x = W / 2 + math.cos(a0) * d * (0.8 + 0.2 * q)
                y2 = c.get("pivot_y", H * 0.6) - 150 + math.sin(a0) * d * 1.25 * (0.8 + 0.2 * q)
                blit(img, self.sparks[i % 3], x, y2, 0.5 + 0.8 * math.sin(math.pi * q), q * 90, math.sin(math.pi * q))
        return img

    # ----- the opening: a flashlight in the dark -----
    @staticmethod
    def beam(t):
        """Where the flashlight is at time t: (x, y, radius, how far the room lights have come up 0..1)."""
        b = t / BEAT
        skull = (215 - 170 * min(t, 0.4), 840)
        keys = ((0.0, *skull, 280), (0.7, *skull, 280), (1.9, 930, 800, 300), (2.4, 540, 870, 440), (3.1, 540, 900, 2300))
        for (b0, x0, y0, r0), (b1, x1, y1, r1) in zip(keys, keys[1:]):
            if b <= b1:
                q = ease_in_out((b - b0) / (b1 - b0))
                return x0 + (x1 - x0) * q, y0 + (y1 - y0) * q, r0 + (r1 - r0) * q, clamp01((b - 2.4) / 0.7)
        return 540, 900, 2300, 1.0

    def flashlight(self, img, t):
        """Darken the room to what the beam touches. Returns the picture, the beam's mask and how far right it has reached."""
        x, y, r, up = self.beam(t)
        yy, xx = np.ogrid[0:H // 4, 0:W // 4]
        d = np.sqrt((xx * 4.0 - x) ** 2 + ((yy * 4.0 - y) * 0.9) ** 2).astype(np.float32) / r
        q = np.clip((1.05 - d) / 0.5, 0, 1)
        low = q * q * (3 - 2 * q)
        # it clicks on with a stutter
        on = 0.0 if t < 0.03 else 0.85 if t < 0.07 else 0.3 if t < 0.1 else 1.0
        m = cv2.resize(low * on, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
        room = img * 0.07 * np.array([0.8, 0.72, 1.2], np.float32)
        out = room * (1 - m) + img * (1.14 - 0.14 * up) * m
        if up < 1:
            for i in range(34):
                px = hash01(i, 41) * W + 26 * math.sin(t * 0.8 + i)
                py = (hash01(i, 42) * (H - 200) + 100 - t * 24) % H
                a = float(low[int(py) // 4 % (H // 4), int(px) // 4 % (W // 4)]) * on * (1 - up) * 0.75
                blit(out, self.mote, px, py, 0.3 + 0.5 * hash01(i, 43), 0, a)
        reach = x + r * 0.62 if t / BEAT < 1.9 else 1e6
        return out, reach

    def draw_title(self, img, base, since, src_f, reach):
        """base is the ungraded picture: the skeleton is found in it by its darkness and kept in front of the word."""
        name = SHOTS[0]["clip"]
        path = self.path(name).raw
        i = int(min(len(path) - 1, max(0, src_f)))
        # the word drifts with the ceiling, a little slower than the camera, so it feels painted up there
        k = W / 2160 * 0.08
        dx, dy = (path[i, 0] - path[0, 0]) * k, (path[i, 1] - path[0, 1]) * k * 0.6
        big = self.title_big_alpha
        h, w = big.shape[:2]
        x0, y0 = int(W / 2 - w / 2 + dx + TITLE.get("dx", 0)), int(TITLE["y"] - h / 2 + dy)
        X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if X1 > X0 and Y1 > Y0:
            region = big[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
            lum = cv2.cvtColor(base[Y0:Y1, X0:X1], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
            dark = (lum < 0.27).astype(np.uint8)
            dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
            # only the skeleton counts: not the specks and vents in the ceiling
            _, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
            dark = np.isin(labels, 1 + np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= 2500)).astype(np.uint8)
            keep = 1 - cv2.GaussianBlur(dark.astype(np.float32), (0, 0), 1.0)
            # the letters come alight as the beam crosses them, and stay lit
            lit = np.clip((reach - np.arange(X0, X1, dtype=np.float32)) / 140, 0, 1)[None, :, None]
            a = region[..., 3:4] * keep[..., None] * lit
            img[Y0:Y1, X0:X1] = img[Y0:Y1, X0:X1] * (1 - a) + region[..., :3] * a
        ps = (since - BEAT * 2) / 0.24
        if ps > 0:
            blit(img, self.title_sub, W / 2 + dx * 0.6, TITLE["y"] + TITLE["sub_dy"] + dy * 0.6,
                 1.12 - 0.12 * ease_out_cubic(ps), 0, clamp01(ps * 2.5))
        pk = (since - BEAT * 0.9) / 0.3
        if pk > 0:
            blit(img, self.title_kick, W / 2, TITLE["y"] + TITLE["kick_dy"], 1.0, 0, clamp01(pk * 2))

    # ----- star confetti when the music lifts -----
    def plan_confetti(self):
        out = []
        for i in range(46):
            a = hash01(i, 11) * 2 * math.pi
            v = 700 + 1500 * hash01(i, 12)
            out.append(dict(a=a, v=v, spr=self.stars[i % len(self.stars)], spin=(hash01(i, 13) - 0.5) * 720,
                            rot=hash01(i, 14) * 360, delay=hash01(i, 15) * 0.1))
        return out

    def draw_confetti(self, img, since):
        for c in self.confetti:
            t = since - c["delay"]
            if t <= 0 or t > 1.7:
                continue
            drag = (1 - math.exp(-3.0 * t)) / 3.0
            x = W / 2 + math.cos(c["a"]) * c["v"] * drag
            y = H * 0.48 + math.sin(c["a"]) * c["v"] * drag + 340 * t * t
            alpha = clamp01((1.7 - t) / 0.5)
            blit(img, c["spr"], x, y, 1.0, c["rot"] + c["spin"] * t, alpha)

    # ----- the wall of costumes -----
    def build_wall(self):
        """Three long strips of costume tiles, already rounded and laid on the plum background."""
        if self.wall_strips is not None:
            return
        tw, th, gap = WALL["tile"][0], WALL["tile"][1], WALL["gap"]
        shots = {s["photo"]: s for s in PHOTO_SHOTS + FAMILY["shots"]}
        bg = np.array(AUBERGINE, np.float32) / 255
        mask = rounded_mask(tw, th, 30)[..., None]
        strips = []
        for col in WALL["columns"]:
            strip = np.empty(((th + gap) * len(col), tw, 3), np.float32)
            strip[:] = bg
            for j, name in enumerate(col):
                p = R.Photo(shots.get(name, {"photo": name}))
                crop = p.crop(tw, th, zoom=WALL.get("crop_zoom", 0.74))
                del p
                y = j * (th + gap)
                strip[y:y + th] = crop * mask + bg * (1 - mask)
                gc.collect()
            strips.append(strip)
        self.wall_strips = strips

    def wall_picture(self, t, seg):
        self.build_wall()
        tw, th, gap = WALL["tile"][0], WALL["tile"][1], WALL["gap"]
        img = np.empty((H, W, 3), np.float32)
        img[:] = np.array(AUBERGINE, np.float32) / 255
        beats = (t - seg["t0"]) / BEAT
        n = int(math.floor(beats))
        frac = beats - n
        moved = n + ease_in_out(clamp01(frac / 0.62))
        blur = abs(math.sin(math.pi * clamp01(frac / 0.62)))
        # after the wall's own beats it keeps sliding, slowly, behind the closing card
        extra = max(0.0, beats - WALL["beats"])
        if extra > 0:
            moved = WALL["beats"] + extra * 0.1
            blur = 0
        total_w = 3 * tw + 2 * gap
        x0 = (W - total_w) // 2
        for c, strip in enumerate(self.wall_strips):
            step = (th + gap)
            d = 1 if c % 2 == 0 else -1
            off = WALL["start"][c] * step + d * moved * step
            sh = strip.shape[0]
            top = off % sh
            idx0 = int(top)
            fr = top - idx0
            rows = (np.arange(H + 1) + idx0) % sh
            colimg = strip[rows]
            if fr > 1e-3:
                colimg = colimg[:-1] * (1 - fr) + colimg[1:] * fr
            else:
                colimg = colimg[:-1]
            if blur > 0.02:
                colimg = cv2.blur(colimg, (1, int(blur * 34) | 1))
            img[:, x0 + c * (tw + gap):x0 + c * (tw + gap) + tw] = colimg
        return img

    # ----- the closing card: a star wipe, then a curtain call -----
    def build_end(self):
        self.e_logo = to_sprite(logo_image(300))
        self.e_halo = to_sprite(halo_image(820, 640, GOLD, 0.5))
        self.e_l1 = to_sprite(lettering(END["line1"], display(88, 500), 0, fill=(CREAM, CREAM), shadow=0.8))
        self.e_l2 = to_sprite(fit_width(lettering(END["line2"], display(186, 900), 0, fill=(GOLD_LIGHT, GOLD), shadow=0.85), 940))
        self.e_tag = to_sprite(lettering(END["tag"], small(33, 600), 2, fill=(CREAM, CREAM), shadow=0.7))
        f_main = small(40, 700)
        f_addr = small(35, 600)
        rows = [("phone", [lettering(BRAND["phone"], f_main, 1, fill=(WHITE, WHITE), shadow=0)]),
                ("pin", [lettering("10629 West Side Drive", f_addr, 0, fill=(WHITE, WHITE), shadow=0),
                         lettering("Grande Prairie, AB", f_addr, 0, fill=(CREAM, CREAM), shadow=0)]),
                ("mail", [lettering(BRAND["email"], f_main, 1, fill=(WHITE, WHITE), shadow=0)])]
        card_w = max(max(t.width for t in txt) for _, txt in rows) + 196
        self.e_cards = []
        for kind, txt in rows:
            h = 112 if len(txt) == 1 else 146
            self.e_cards.append(dict(glass=Glass(card_w, h, 34, stops=(GOLD, GOLD_LIGHT, PLUM), dark=0.55),
                                     icon=to_sprite(icon_image(kind, 50, GOLD)), h=h, w=card_w,
                                     text=[to_sprite(t) for t in txt]))
        self.e_foot = to_sprite(lettering(f"{BRAND['web']}   ·   {BRAND['instagram']}", small(36, 700), 2,
                                          fill=(GOLD_LIGHT, GOLD), shadow=0.8))
        # a dark theatre, with a pool of light on the stage where the cast lands
        bg = np.empty((H, W, 3), np.float32)
        pos = np.linspace(0, 1, H)
        stops = [0, 0.4, 0.78, 1]
        for c, vals in enumerate(zip((22, 9, 34), (60, 24, 86), (36, 14, 56), (24, 9, 36))):
            bg[..., c] = (np.interp(pos, stops, vals) / 255)[:, None]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        pool = np.exp(-(((xx - W / 2) / 800) ** 2 + ((yy - 1800) / 330) ** 2) * 1.3)[..., None]
        self.e_bg = bg + pool * np.array([0.2, 0.11, 0.27], np.float32)
        del yy, xx
        ry, rx = np.mgrid[0:H // 6, 0:W // 6].astype(np.float32)
        self.e_theta = np.arctan2(ry * 6 - 300, rx * 6 - W / 2)
        self.e_r = np.hypot(ry * 6 - 300, rx * 6 - W / 2)
        self.e_cast = None

    def curtain(self):
        """The cut-outs from the cast cards, as stickers for the curtain call."""
        if self.e_cast is not None:
            return self.e_cast
        self.e_cast = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
        for c in END["cast"]:
            src = Source(c["clip"], 1620)
            fr = src.frame(c["frame"])
            src.close()
            matte = cv2.imread(str(MASKS / f"{c['clip']}_f{c['frame']}.png"), cv2.IMREAD_GRAYSCALE)
            ys, xs = np.nonzero(matte > 40)
            y0, y1, x0, x1 = max(0, ys.min() - 40), ys.max() + 40, max(0, xs.min() - 40), xs.max() + 40
            m = np.clip((matte[y0:y1, x0:x1].astype(np.float32) / 255 - 0.25) / 0.5, 0, 1)
            edge = cv2.GaussianBlur(cv2.dilate(m, kernel), (0, 0), 2.0)
            x = cv2.LUT(fr[y0:y1, x0:x1], CURVE).astype(np.float32) / 255
            lum = (x @ LUMA)[..., None]
            x = np.clip((lum + (x - lum) * 1.14) * np.array([1.025, 1.0, 0.965], np.float32), 0, 1)
            cream = np.array(CREAM, np.float32) / 255
            rgb = cream * (edge * (1 - m))[..., None] + x * m[..., None]
            k = c["height"] / rgb.shape[0]
            spr = cv2.resize(np.dstack([rgb, edge]), None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            self.e_cast.append(dict(c, spr=spr))
        return self.e_cast

    def burst(self, img, since, cx, cy, seed, n=14, radius=430):
        for i in range(n):
            q = (since - 0.012 * i) / 0.75
            if 0 < q < 1:
                a = hash01(seed, i, 1) * 2 * math.pi
                d = radius * (0.45 + 0.55 * hash01(seed, i, 2)) * ease_out_cubic(q)
                blit(img, self.sparks[i % 3], cx + math.cos(a) * d, cy + math.sin(a) * d, 0.5 + 0.7 * math.sin(math.pi * q),
                     q * 120, math.sin(math.pi * q))

    @staticmethod
    def shine(img, spr, cx, cy, q):
        """A bright diagonal glint crossing a sprite (q 0..1)."""
        h, w = spr.shape[:2]
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        band = np.exp(-(((xx + 0.5 * yy) - (-0.2 + 1.4 * q) * (w + 0.5 * h)) / 46) ** 2)[..., None]
        img[y0:y0 + h, x0:x0 + w] += band * spr[..., 3:4] * 0.75

    @staticmethod
    def star_mask(radius, rot):
        pts = []
        for i in range(10):
            r = radius if i % 2 == 0 else radius * 0.46
            a = math.radians(rot) - math.pi / 2 + i * math.pi / 5
            pts.append((W / 4 + r / 2 * math.cos(a), H * 0.42 / 2 + r / 2 * math.sin(a)))
        m = np.zeros((H // 2, W // 2), np.uint8)
        cv2.fillPoly(m, [np.array(pts, np.float32).round().astype(np.int32)], 255, lineType=cv2.LINE_AA)
        return m

    def end_picture(self, t, seg):
        since = t - seg["t0"]
        rays = (0.5 + 0.5 * np.sin(11 * self.e_theta + 0.5 * t)) ** 6 + 0.6 * (0.5 + 0.5 * np.sin(17 * self.e_theta - 0.35 * t + 1.3)) ** 8
        rays = cv2.resize((rays * np.exp(-self.e_r / 850)).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        img = self.e_bg + rays[..., None] * np.array([1.0, 0.78, 0.45], np.float32) * 0.3 * clamp01(since / 0.5)
        for i in range(14):
            q = (since * 0.09 + hash01(i, 31)) % 1
            blit(img, self.sparks[i % 3], hash01(i, 32) * W + 30 * math.sin(since * 0.8 + i), H - q * (H + 80),
                 0.45 + 0.4 * hash01(i, 33), since * 40 + i * 30, 0.55 * math.sin(math.pi * q) * clamp01(since / 0.5))
        at = lambda beat: (beat - seg["b0"]) * BEAT  # noqa: E731
        button = at(END["button"])
        # the cast, behind the words: each hops up from below the frame, then bobs on the beat
        for c in self.curtain():
            land = at(c["hop"])
            p = (since - land) / 0.32
            if p <= 0:
                continue
            spr = c["spr"]
            h, w = spr.shape[:2]
            y = c["bottom"] - h / 2 + (1 - ease_out_back(p, 1.3)) * h * 1.05
            beats = (since - land) / BEAT
            y -= 9 * abs(math.sin(math.pi * beats))
            jump = (since - button) / 0.42
            if 0 < jump < 1:
                y -= 66 * math.sin(math.pi * jump)
            blit(img, spr, c["x"], y, 1.0, 1.6 * math.sin(math.pi * beats) * clamp01(p - 1), 1.0, pivot=(w / 2, h))
        # the logo and the words
        pl = since / 0.42
        ly = 300 - 4 * math.sin(since * 2.4)
        if pl > 0:
            blit(img, self.e_halo, W / 2, ly, 1.0, 0, 0.75 * clamp01(pl * 2))
            blit(img, self.e_logo, W / 2, ly, 0.8 + 0.2 * ease_out_back(pl), 0, clamp01(pl * 2.2))
        for start in (at(40), button):
            q = (since - start) / 0.5
            if 0 < q < 1 and pl >= 1:
                self.shine(img, self.e_logo, W / 2, ly, q)
        self.burst(img, since - 0.05, W / 2, ly, 5)
        p1 = (since - 0.28) / 0.3
        if p1 > 0:
            blit(img, self.e_l1, W / 2, 528 + 24 * (1 - ease_out_cubic(p1)), 1.0, 0, clamp01(p1 * 2))
        p2 = (since - 0.42) / 0.34
        if p2 > 0:
            blit(img, self.e_l2, W / 2, 648 + 30 * (1 - ease_out_cubic(p2)), 0.94 + 0.06 * ease_out_back(p2), 0, clamp01(p2 * 2))
        p3 = (since - 0.8) / 0.3
        if p3 > 0:
            blit(img, self.e_tag, W / 2, 778, 1.0, 0, clamp01(p3 * 2))
        y = 846
        for i, c in enumerate(self.e_cards):
            pc = (since - at(40.07) - i * 0.26) / 0.3  # one card per half beat, from the song's first closing hit
            cy = y + c["h"] / 2
            if pc > 0:
                a = clamp01(pc * 2)
                dy = 36 * (1 - ease_out_cubic(pc))
                x0, _ = c["glass"].draw(img, W / 2, cy + dy, a)
                blit(img, c["icon"], x0 + 70, cy + dy, 1.0, 0, a)
                n = len(c["text"])
                for k, spr in enumerate(c["text"]):
                    ty = cy + dy + (k - (n - 1) / 2) * 50
                    blit(img, spr, x0 + 138 + spr.shape[1] / 2, ty, 1.0, 0, a)
            y += c["h"] + 14
        pf = (since - at(41.6)) / 0.3
        if pf > 0:
            blit(img, self.e_foot, W / 2, y + 30, 1.0, 0, clamp01(pf * 2))
        # the song's last hit
        if button < since < button + 0.3:
            img = add_flash(img, 0.16 * math.exp(-(since - button) / 0.09))
        self.burst(img, since - button, W / 2, 1500, 9, 18, 620)
        # it all opens from a star growing out of the wall of costumes
        if since < 0.42:
            q = since / 0.42
            m = self.star_mask(2750 * q ** 2.4 + 1, -20 + 70 * q)
            rim = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).astype(np.float32) - m
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255
            rim = cv2.resize(rim, (W, H), interpolation=cv2.INTER_LINEAR)[..., None] / 255
            img = self.wall_picture(t, self.seg("wall")) * (1 - m) + img * m
            img = img + (np.array(GOLD_LIGHT, np.float32) / 255 - img) * rim
        return img

    def seg(self, kind):
        return next(s for s in self.segs if s["kind"] == kind)

    # ----- one frame -----
    def shot_picture(self, sg, t):
        sh = sg["shot"]
        tau = t - sg["t0"]
        dur = sg["t1"] - sg["t0"]
        c = sh.get("cast")
        if c and tau >= dur - c["beats"] * BEAT:
            return self.draw_cast(sg, tau)
        base = self.sample(sh, tau, dur)
        img = grade(base, sh.get("mood", 0.0))
        if sg["i"] == 0:
            img, reach = self.flashlight(img, tau)
            self.draw_title(img, base, tau, self.src_frame(sh, tau), reach)
        return img

    def wreath_through(self, sg, nxt, t):
        """The last moments of the wreath: the camera dives through the hole in it, into the next shot."""
        sh = sg["shot"]
        W_ = sh["through"]
        dur = sg["t1"] - sg["t0"]
        tau = t - sg["t0"]
        q = clamp01((tau - (dur - W_["time"])) / W_["time"])
        if q <= 0:
            return self.shot_picture(sg, t)
        e = q ** 3
        z = 1 + (W_["scale"] - 1) * e
        half = 0.5 / z
        cx, cy = max(half, min(1 - half, W_["x"])), max(half, min(1 - half, W_["y"]))
        src = self.source(sh)
        f = self.src_frame(sh, tau)
        M = self.matrix(sh, f, z, cx, cy, src.w)
        pic = cv2.warpAffine(src.frame(round(f)), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        img = grade(pic, sh.get("mood", 0.0))
        hx, hy = W / 2 + (W_["x"] - cx) * W * z, H / 2 + (W_["y"] - cy) * H * z
        img = zoom_blur(img, hx, hy, 0.5 * e)
        # the hole opens onto the next shot
        yy, xx = np.ogrid[0:H, 0:W]
        d = np.sqrt(((xx - hx) / (W_["rx"] * W * z)) ** 2 + ((yy - hy) / (W_["ry"] * H * z)) ** 2).astype(np.float32)
        m = np.clip((1.0 - d) / 0.12, 0, 1)[..., None]
        behind = self.shot_picture(nxt, nxt["t0"])
        return img * (1 - m) + behind * m

    def frame(self, t):
        t = min(t, self.total - 1e-4)
        i = next(k for k, s in enumerate(self.segs) if s["t0"] <= t < s["t1"])
        sg = self.segs[i]
        prv = self.segs[i - 1] if i else None
        nxt = self.segs[i + 1] if i + 1 < len(self.segs) else None
        if sg["kind"] == "shot":
            # free clips the reel has finished with
            for name in [n for n in self.clips if n != sg["shot"]["clip"]
                         and not (nxt and nxt["kind"] == "shot" and n == nxt["shot"]["clip"])]:
                self.clips.pop(name).close()
            if sg["shot"].get("through"):
                img = self.wreath_through(sg, nxt, t)
            else:
                img = self.shot_picture(sg, t)
        elif sg["kind"] == "wall":
            self.clips.clear()
            img = self.wall_picture(t, sg)
            cap = WALL["caption"]
            if "w" not in cap:
                cap["w"] = Words(cap["text"], display(cap.get("size", 104)), W / 2, cap["y"], max_w=1000)
            pill = cap.get("glass")
            if pill is None:
                pill = cap["glass"] = Glass(int(cap["w"].width) + 70, 170, 60, stops=(GOLD, GOLD_LIGHT, PLUM), dark=0.3)
            since = t - sg["t0"]
            if since > 0.05:
                pill.draw(img, W / 2, cap["y"], clamp01((since - 0.05) / 0.2))
                cap["w"].draw(img, since - 0.1)
        else:
            img = self.end_picture(t, sg)
        # how each shot arrives
        if sg["kind"] == "shot" and sg["shot"].get("in"):
            img = self.arrive(img, sg["shot"]["in"], t - sg["t0"])
        if nxt is not None and nxt["kind"] == "shot" and nxt["shot"].get("in", {}).get("kind") == "whip":
            until = nxt["t0"] - t
            if until < 3.5 / FPS:
                q = 1 - until / (3.5 / FPS)
                d = nxt["shot"]["in"].get("dir", 1)
                vert = nxt["shot"]["in"].get("vertical", False)
                sx = d * 0.34 * W * q * q
                img = shake(img, 0 if vert else sx, sx if vert else 0)
                img = dir_blur(img, 1 + q * 260, vert)
        if sg["kind"] == "wall" and t - sg["t0"] < 3.5 / FPS:
            q = 1 - (t - sg["t0"]) / (3.5 / FPS)
            img = dir_blur(shake(img, 0, -0.34 * W * q * q), 1 + q * 260, True)
        if nxt is not None and nxt["kind"] == "wall" and nxt["t0"] - t < 3.5 / FPS:
            q = 1 - (nxt["t0"] - t) / (3.5 / FPS)
            img = dir_blur(shake(img, 0, 0.34 * W * q * q), 1 + q * 260, True)
        # the music's lift
        drop = bt(DROP_BEAT)
        if drop <= t < drop + 0.26:
            img = add_flash(img, 0.85 * math.exp(-(t - drop) / 0.075))
        if drop <= t < drop + 0.9:
            img = light_leak(img, (t - drop) / 0.9, 0.9)
        if drop <= t < drop + 1.8:
            self.draw_confetti(img, t - drop)
        # captions, on a soft dark band so cream words read on the bright floors
        for c in self.caps:
            t0, t1 = bt(c["b0"]), bt(c["b1"])
            if t0 <= t < t1:
                a = 0.5 * clamp01((t - t0) / 0.2) * clamp01((t1 - t) / 0.14)
                band = np.exp(-((np.arange(H, dtype=np.float32) - c["w"].mid_y) / 190) ** 2)[:, None, None]
                img = img * (1 - a * band)
                c["w"].draw(img, t - t0, t1 - 0.02)
        # the logo stays in the corner until the closing card
        if sg["kind"] == "shot":
            cast = sg["shot"].get("cast")
            in_cast = cast and t - sg["t0"] >= sg["t1"] - sg["t0"] - cast["beats"] * BEAT
            if not in_cast and sg["shot"].get("logo", True):
                # in the first shot it waits for the lights to come up
                blit(img, self.logo_small, 112, 208, 1.0, 0, 0.96 if sg["i"] else 0.96 * clamp01((t / BEAT - 2.6) / 0.4))
        return img

    def arrive(self, img, how, since):
        kind = how["kind"]
        if kind == "whip" and since < 3.5 / FPS:
            q = 1 - since / (3.5 / FPS)
            d = how.get("dir", 1)
            vert = how.get("vertical", False)
            sx = -d * 0.34 * W * q * q
            img = shake(img, 0 if vert else sx, sx if vert else 0)
            return dir_blur(img, 1 + q * 260, vert)
        if kind == "flash" and since < 0.2:
            return add_flash(img, how.get("k", 0.5) * math.exp(-since / 0.06))
        return img


# ---------- sound ----------
def synth_click(sr):
    """A flashlight's switch: a tiny tick and a small low thump."""
    n = int(0.09 * sr)
    t = np.arange(n, dtype=np.float32) / sr
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, n).astype(np.float32) * np.exp(-t * 300).astype(np.float32) * 0.8
    x = x + np.sin(2 * np.pi * 95 * t) * np.exp(-t * 45).astype(np.float32) * 0.6
    return np.stack([x, x], 1) * 0.6


def synth_pop(sr):
    """A soft cartoon pop for each child landing in the curtain call."""
    n = int(0.11 * sr)
    t = np.arange(n, dtype=np.float32) / sr
    x = np.sin(2 * np.pi * (520 * t - 1500 * t * t)) * np.exp(-t * 32).astype(np.float32)
    return np.stack([x, x], 1) * 0.5


def synth_shutter(sr):
    """A camera shutter: two short clicks of filtered noise, for the freeze frames."""
    rng = np.random.default_rng(4)
    n = int(0.14 * sr)
    x = np.zeros(n, np.float32)
    for at, g in ((0.0, 1.0), (0.055, 0.7)):
        i = int(at * sr)
        m = int(0.03 * sr)
        burst = rng.normal(0, 1, m).astype(np.float32) * np.exp(-np.linspace(0, 7, m)).astype(np.float32) * g
        x[i:i + m] += burst
    x = np.diff(x, prepend=0)
    return np.stack([x, x], 1) * 0.5


def synth_ting(sr, f=2093.0):
    """A small bright 'ting' for the sparkles."""
    n = int(0.5 * sr)
    t = np.arange(n, dtype=np.float32) / sr
    x = (np.sin(2 * np.pi * f * t) + 0.45 * np.sin(2 * np.pi * f * 1.5 * t) + 0.2 * np.sin(2 * np.pi * f * 2 * t)) \
        * np.exp(-t * 9).astype(np.float32)
    return np.stack([x, x], 1) * 0.3


def sfx_events(r):
    drop = bt(DROP_BEAT)
    ev = [("@click", 0.02, -7, 0, None), ("impact", 0.02, -18, 0, None), ("riser", drop - 3.6, -14, 0, None),
          ("ghost_whoosh", drop - 0.5, -10, 0, None), ("impact", drop, -4, 0, None),
          ("@laugh", drop + 0.45, 0, 0, None)]
    for sg in r.segs:
        if sg["kind"] != "shot":
            continue
        sh = sg["shot"]
        if sh.get("in", {}).get("kind") == "whip":
            ev.append(("whoosh", sg["t0"] - 0.22, -14, 0, 0.9))
        if sh.get("cast"):
            at = sg["t1"] - sh["cast"]["beats"] * BEAT
            ev.append(("@shutter", at, -8, 0, None))
            ev.append(("@ting", at + 0.09, -14, 0, None))
    wall, end = r.seg("wall"), r.seg("end")
    ev.append(("whoosh", wall["t0"] - 0.22, -11, 0, 0.9))
    for b in range(1, WALL["beats"]):
        ev.append(("whoosh", wall["t0"] + b * BEAT - 0.1, -19, 0, 0.6))
    ev += [("whoosh", end["t0"] - 0.2, -12, 0, 0.9), ("impact", end["t0"], -9, 0, None), ("@ting", end["t0"] + 0.1, -13, 0, None)]
    for c in END["cast"]:
        ev.append(("@pop", end["t0"] + (c["hop"] - end["b0"]) * BEAT + 0.15, -12, 0, None))
    button = end["t0"] + (END["button"] - end["b0"]) * BEAT
    ev += [("@ting", button + 0.02, -11, 0, None), ("witch", button + 0.3, -13, 0, None)]
    return ev


def build_audio(r, music=True):
    sr = 48000
    n = int(math.ceil(r.total * sr))
    mix = np.zeros((n, 2), np.float32)
    if music:
        song = R.ROOT / "source" / "music" / MUSIC["file"]
        when = lambda beat: MUSIC["first_beat"] + beat * BEAT  # noqa: E731
        # the song from its quiet opening, then a short cross-fade into its last bars under the star wipe
        join, xf = int(bt(MUSIC["join"]) * sr), int(0.03 * sr)
        m = decode(song, when(MUSIC["start"]), bt(MUSIC["join"]) + 0.03)[:join + xf]
        m[-xf:] *= np.linspace(1, 0, xf)[:, None]
        mix[:len(m)] += m * 0.95
        m = decode(song, when(MUSIC["ending"]), r.total - bt(MUSIC["join"]))[:n - join]
        m[:xf] *= np.linspace(0, 1, xf)[:, None]
        mix[join:join + len(m)] += m * 0.95
    # a real laugh from the tiger's clip, under the moment the kids arrive (no names are spoken in it)
    laugh = decode(next((ROOT / "source" / "recap_raw").glob("IMG_7994.*")), 0.95, 0.85)
    fade = int(0.08 * sr)
    laugh[:fade] *= np.linspace(0, 1, fade)[:, None]
    laugh[-fade:] *= np.linspace(1, 0, fade)[:, None]
    cache = {"@shutter": synth_shutter(sr), "@ting": synth_ting(sr), "@laugh": laugh, "@click": synth_click(sr),
             "@pop": synth_pop(sr)}
    for name, t, g, s0, ln in sfx_events(r):
        key = (name, s0, ln)
        if name not in cache and key not in cache:
            snd = decode(SFX / f"{name}.mp3", s0, ln)
            kk = min(len(snd), int(0.08 * sr))
            if ln:
                snd[-kk:] *= np.linspace(1, 0, kk)[:, None]
            cache[key] = snd
        snd = (cache[name] if name in cache else cache[key]) * (10 ** (g / 20))
        i = int(t * sr)
        off = 0
        if i < 0:
            off, i = -i, 0
        j = min(n, i + len(snd) - off)
        if j > i:
            mix[i:j] += snd[off:off + j - i]
    tail = int(0.25 * sr)
    mix[-tail:] *= np.linspace(1, 0, tail)[:, None]
    work = ROOT / "work" / "recap"
    raw = work / ("mix_raw.wav" if music else "mix_raw_nomusic.wav")
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


def main():
    r = Renderer()
    print(f"{len(r.segs)} segments, {r.total:.2f} s", flush=True)
    if "--stills" in sys.argv:
        out = ROOT / "work" / "recap" / "stills"
        out.mkdir(parents=True, exist_ok=True)
        for arg in sys.argv[sys.argv.index("--stills") + 1:]:
            t = float(arg)
            fr = (np.clip(r.frame(t), 0, 1) * 255 + 0.5).astype(np.uint8)
            Image.fromarray(fr).save(out / f"t{t:06.2f}.jpg", quality=90)
            print("wrote", t, flush=True)
        return
    EXPORTS.mkdir(exist_ok=True)
    cover = (np.clip(r.frame(1.2), 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(cover).save(EXPORTS / "recap_cover.jpg", quality=95)
    raw = ROOT / "work" / "recap" / "frames.rgb"
    render_frames(r, raw)
    music, sfx_only = build_audio(r, music=True), build_audio(r, music=False)
    dur = r.total
    del r
    gc.collect()
    reel = EXPORTS / "recap_last_week.mp4"
    encode(raw, reel, music, dur)
    remux(reel, sfx_only, EXPORTS / "recap_last_week_nomusic.mp4")
    if reel.stat().st_size > 29 * 2 ** 20:  # when the master already fits, a second encode would only lose quality
        phone_copy(reel, EXPORTS / "recap_last_week_phone.mp4")
    raw.unlink()


if __name__ == "__main__":
    main()
