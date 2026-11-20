#!/usr/bin/env python3
"""
ASCII Banqiao - a winter night in Banqiao drawn stage by stage with coloured
ASCII characters: the Lin Family Garden, Banqiao Station, New Taipei City
Hall, the Crescent Bridge over the Dahan River, and the Christmasland light
show with its hot-air balloon, lasers and snow.

    pip install numpy
    python banqiao.py                      # Ctrl+C to quit
    python banqiao.py --speed 1.5 --fps 24
    python banqiao.py --once --at 32       # print one frame and exit

    pip install numpy pillow               # plus ffmpeg, to record a video
    python banqiao.py --record banqiao.mp4 --tag @yourname
    python banqiao.py --record banqiao.mp4 --code   # with the source typing out below
    python banqiao.py --record still.png --at 32
    python banqiao.py --record music.wav           # just the soundtrack

How it works
    1. The scene is built once for the terminal size as a stack of layers.
       Every cell of a layer holds a character, a colour and a reveal
       "order" between 0 and 1.
    2. A timeline splits the 39-second loop into stages. A stage's progress
       p shows the cells whose order <= p, so the hills sweep in from the
       left, buildings grow upwards and City Hall rises floor by floor. The
       newest cells flash white like a pen tip.
    3. Moving things (stars, moon, balloon, lasers, bridge lights, meteor,
       snow) are drawn on top each frame, City Hall's facade becomes a
       projection screen, and the river mirrors whatever is lit above it.
    4. The frame is printed with 24-bit ANSI colour codes.

Needs a terminal with 24-bit colour (Windows Terminal, VS Code, iTerm2,
macOS Terminal, most Linux terminals). Make the window big (120x40 or more)
and the font small for the best look.
"""

import argparse
import colorsys
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

ASPECT = 2.0  # a terminal cell is about twice as tall as it is wide

# ------------------------------------------------------------------ timeline
STAGES = [  # name, start, end (seconds)
    ("sky", 0.0, 3.0),
    ("mountains", 1.5, 5.0),
    ("river", 4.0, 7.0),
    ("city", 6.0, 11.0),
    ("lin_garden", 10.0, 13.5),
    ("city_hall", 13.0, 18.0),
    ("lights", 18.0, 21.5),
    ("moon_bridge", 21.5, 24.0),
    ("countdown", 24.0, 29.0),
    ("light_show", 29.0, 36.5),
    ("merry_xmas", 30.5, 36.5),
]
STAGE_T = {name: (a, b) for name, a, b in STAGES}
LOOP = 39.0
FADE = 37.0  # everything fades out from here, then the loop starts again
SHOW = 29.0  # the light show starts on the hour
METEOR = 22.2

# ------------------------------------------------------------------ palette
STAR = (0.80, 0.85, 1.00)
MOON = (1.00, 0.93, 0.70)
RIDGE = (0.36, 0.44, 0.68)
HILLSIDE = (0.13, 0.16, 0.28)
WALL = (0.40, 0.45, 0.64)
WIN_OFF = (0.20, 0.23, 0.36)
WIN_ON = [(1.00, 0.80, 0.42), (1.00, 0.90, 0.62), (0.92, 0.95, 1.00), (0.55, 0.80, 1.00)]
LEAVES = (0.10, 0.32, 0.18)
TRUNK = (0.35, 0.22, 0.12)
LIME = (0.62, 0.60, 0.56)      # the garden's whitewashed walls
TILE = (0.80, 0.34, 0.18)      # terracotta roof tiles
BRICK = (0.72, 0.24, 0.16)
RIDGE_TOP = (0.98, 0.58, 0.22)
AMBER_OFF = (0.36, 0.22, 0.10)
AMBER_ON = (1.00, 0.72, 0.32)
HALL = (0.46, 0.50, 0.64)
HALL_GLASS = (0.20, 0.24, 0.36)
HALL_ON = (0.95, 0.92, 0.80)
STEEL = (0.62, 0.66, 0.74)
DECK = (0.95, 0.88, 0.70)
SNOW = (0.86, 0.92, 1.00)
HUD = (0.45, 0.50, 0.64)
WHITE = (1.00, 1.00, 1.00)
BLACK = (0.0, 0.0, 0.0)
WATER = np.array([0.10, 0.22, 0.45], np.float32)
BALLOON = [(0.98, 0.55, 0.70), (1.00, 0.86, 0.62), (0.62, 0.82, 1.00), (1.00, 0.70, 0.80), (0.80, 0.95, 0.70)]

DIGITS = {  # 3x5 font for the countdown on City Hall's facade
    "1": [".#.", "##.", ".#.", ".#.", "###"],
    "2": ["###", "..#", "###", "#..", "###"],
    "3": ["###", "..#", ".##", "..#", "###"],
    "4": ["#.#", "#.#", "###", "..#", "..#"],
    "5": ["###", "#..", "###", "..#", "###"],
}


def clamp01(v):
    return min(max(v, 0.0), 1.0)


def smooth(v):
    v = clamp01(v)
    return v * v * (3 - 2 * v)


def hash01(a, b):
    """Repeatable pseudo-random number in [0, 1); index.html uses the same formula."""
    v = np.sin(np.asarray(a) * 12.9898 + b * 78.233) * 43758.5453
    return v - np.floor(v)


def stage_progress(name, t):
    a, b = STAGE_T[name]
    return clamp01((t - a) / (b - a))


def hsv(h, s=1.0, v=1.0):
    return colorsys.hsv_to_rgb(h % 1.0, s, v)


def dir_char(deg):
    """Line character for a direction in degrees (0 = horizontal)."""
    deg %= 180
    return "-" if deg < 22.5 or deg >= 157.5 else ("\\" if deg < 67.5 else ("|" if deg < 112.5 else "/"))


# ------------------------------------------------------------------ layers
class Layer:
    """A character, a colour and a reveal order (0..1) per cell; inf = empty.
    A space with an order is solid: it hides whatever was drawn behind it."""

    def __init__(self, h, w):
        self.h, self.w = h, w
        self.ch = np.full((h, w), " ", "<U1")
        self.rgb = np.zeros((h, w, 3), np.float32)
        self.order = np.full((h, w), np.inf, np.float32)

    def put(self, y, x, c, rgb, order):
        if 0 <= y < self.h and 0 <= x < self.w:
            self.ch[y, x] = c
            self.rgb[y, x] = rgb
            self.order[y, x] = order


class Canvas:
    def __init__(self, h, w):
        self.h, self.w = h, w
        self.ch = np.full((h, w), " ", "<U1")
        self.rgb = np.zeros((h, w, 3), np.float32)

    def blit(self, layer, p, tip=0.05):
        """Show the cells of a layer whose order <= p; the newest ones glow."""
        if p <= 0:
            return
        m = layer.order <= p
        self.ch[m] = layer.ch[m]
        self.rgb[m] = layer.rgb[m]
        if p < 1 and tip > 0:
            fresh = m & (layer.order > p - tip)
            self.rgb[fresh] = self.rgb[fresh] * 0.3 + 0.85

    def put(self, y, x, c, rgb):
        y, x = int(round(y)), int(round(x))
        if 0 <= y < self.h and 0 <= x < self.w:
            self.ch[y, x] = c
            self.rgb[y, x] = rgb

    def spark(self, y, x, c, rgb):
        """Like put, but only if brighter than what is already there (light adds up)."""
        y, x = int(round(y)), int(round(x))
        if 0 <= y < self.h and 0 <= x < self.w and max(rgb) >= self.rgb[y, x].max():
            self.ch[y, x] = c
            self.rgb[y, x] = rgb

    def text(self, y, x, s, rgb):
        for i, c in enumerate(s):
            if c != " ":
                self.put(y, x + i, c, rgb)


def edge_char(top, left, right):
    """Character for a ridge cell given its own and its neighbours' top rows."""
    if left > top and right > top:
        return "^"
    if left > top:
        return "/"
    if right > top:
        return "\\"
    return "_"


# ------------------------------------------------------------------ scene
class Scene:
    def __init__(self, w, h, tag="", seed=101, hud=True):
        self.w, self.h, self.tag, self.hud = w, h, tag, hud
        self.hz = int(round(h * 0.70))   # first row of the river
        self.S = h * ASPECT               # scene height measured in columns
        self.Sx = min(self.S, 1.1 * w)    # size scale that still fits a narrow (phone) screen
        self.xlin = int(round(w * 0.17))
        self.xtree = int(round(w * 0.335))
        self.xballoon = int(round(w * 0.46))
        self.xhall = int(round(w * 0.70))
        self.xstation = int(round(w * 0.92))
        self.year = time.localtime().tm_year
        rng = np.random.default_rng(seed)
        self._stars(rng)
        self.mountains = self._mountains(rng)
        self.city, self.city_win = self._city(rng)
        self.lin, self.lin_win = self._lin_garden(rng)
        self.hall, self.hall_win = self._city_hall(rng)
        self._balloon_shape()
        self._bridge()
        n = int(w * h * 0.025)
        i = np.arange(n)
        self.snow = [hash01(i, k) for k in range(1, 6)]

    # ---------------------------------------------------------- static parts
    def _stars(self, rng):
        n = int(self.w * self.hz * 0.025)
        self.star_y = rng.integers(0, self.hz - 2, n)
        self.star_x = rng.integers(0, self.w, n)
        self.star_phase = rng.uniform(0, 2 * math.pi, n)
        self.star_rate = rng.uniform(1.0, 3.0, n)
        self.star_order = rng.uniform(0, 1, n)

    def _mountains(self, rng):
        """Guanyin Mountain and the Linkou plateau, far across the river."""
        W, H, hz = self.w, self.h, self.hz
        L = Layer(H, W)
        x = np.linspace(0, 1, W)

        def bump(c, s, a):
            return a * np.exp(-((x - c) / s) ** 2)

        prof = (bump(0.08, 0.08, 0.20) + bump(0.20, 0.07, 0.27) + bump(0.34, 0.09, 0.18)
                + bump(0.60, 0.25, 0.07) + bump(0.95, 0.10, 0.10) + 0.03)
        ridge = np.clip(np.round(hz - 1 - prof * self.Sx / ASPECT).astype(int), 1, hz - 1)
        for xi in range(W):
            top = ridge[xi]
            left = ridge[xi - 1] if xi > 0 else top
            right = ridge[xi + 1] if xi < W - 1 else top
            c = edge_char(top, left, right)
            low = max(top, max(left, right) - 1)
            for y in range(top, hz):
                o = 0.92 * xi / W + 0.08 * (y - top) / max(1, hz - top)
                if y <= low:
                    cc = c if y == top or c != "^" else ("/" if left >= right else "\\")
                    L.put(y, xi, cc, RIDGE, o)
                elif rng.random() < 0.16:
                    k = 1 - 0.4 * (y - top) / max(1, hz - top)
                    L.put(y, xi, rng.choice([".", ":", "'"]), tuple(v * k for v in HILLSIDE), o)
                else:
                    L.put(y, xi, " ", BLACK, o)
        return L

    def _building(self, L, win, rng, x0, bw, bh, st):
        hz = self.hz
        top, x1 = hz - bh, x0 + bw - 1
        for y in range(top, hz):
            o = st + 0.36 * (hz - 1 - y) / max(bh - 1, 1)
            for x in range(x0, x1 + 1):
                if y == top:
                    if x0 < x < x1:
                        L.put(y, x, "_", WALL, o)
                elif x in (x0, x1):
                    L.put(y, x, "|", WALL, o)
                elif (x - x0) % 2 == 1 and (y - top) % 2 == 0:
                    L.put(y, x, ":", WIN_OFF, o)
                    if rng.random() < 0.8:
                        rgb = WIN_ON[rng.choice(4, p=[0.45, 0.25, 0.18, 0.12])]
                        win.put(y, x, "#" if rng.random() < 0.7 else "=", rgb, rng.random())
                else:
                    L.put(y, x, " ", BLACK, o)
        if bh >= 0.15 * self.h and rng.random() < 0.7:
            xc = x0 + bw // 2
            L.put(top - 1, xc, "|", WALL, st + 0.37)
            self.blinkers.append((top - 2, xc, L, top - 1, rng.uniform(0, 1)))

    def _city(self, rng):
        """The towers of the Xinban district, tallest around City Hall, and Banqiao Station."""
        W, H, hz = self.w, self.h, self.hz
        L, win = Layer(H, W), Layer(H, W)
        self.blinkers = []
        x = 0
        while x < W:
            bw = int(rng.integers(3, 8))
            near = math.exp(-(((x + bw / 2) / W - self.xhall / W) / 0.18) ** 2)
            hmax = 0.10 + 0.14 * near
            bh = max(3, int(round(rng.uniform(0.4, 1.0) * hmax * H)))
            self._building(L, win, rng, x, bw, bh, rng.uniform(0, 0.62))
            x += bw + int(rng.integers(0, 2))
        # Banqiao Station: a long, low block with glass bands and its name on top
        bw, bh = max(12, round(0.17 * self.Sx)), max(5, round(0.15 * H))
        x0 = self.xstation - bw // 2
        x1, top = x0 + bw - 1, hz - bh
        sign = "BANQIAO" if bw >= 11 else ""
        sx = x0 + (bw - len(sign)) // 2
        for y in range(top, hz):
            o = 0.55 + 0.4 * (hz - 1 - y) / max(bh - 1, 1)
            for x in range(x0, x1 + 1):
                if y == top:
                    if x0 < x < x1:
                        L.put(y, x, "_", WALL, o)
                elif x in (x0, x1):
                    L.put(y, x, "|", WALL, o)
                elif y == top + 1:
                    L.put(y, x, "=", WALL, o)
                    if sign and sx <= x < sx + len(sign):
                        L.put(y, x, sign[x - sx], WIN_OFF, o)
                        win.put(y, x, sign[x - sx], WHITE, 0.2)
                elif (y - top) % 2 == 0:
                    L.put(y, x, ":", WIN_OFF, o)
                    win.put(y, x, ":" if x % 3 else "|", (0.75, 0.85, 1.0), rng.uniform(0.1, 0.9))
                else:
                    L.put(y, x, "=", WALL, o)
        return L, win

    def _lin_garden(self, rng):
        """The Lin Family Mansion and Garden: banyans, a whitewashed wall with a
        moon gate, and a two-storey pavilion with a swallowtail ridge."""
        W, H, hz = self.w, self.h, self.hz
        L, win = Layer(H, W), Layer(H, W)
        xc = self.xlin
        a = max(8, round(0.12 * self.Sx))
        for dx, r in ((-0.85, 0.30), (0.85, 0.32)):   # banyan crowns either side
            cx, rx = xc + dx * a, max(3.0, r * a)
            ry = rx / ASPECT * 1.1
            cy = hz - 3 - ry
            for y in range(int(cy - ry) - 1, hz - 2):
                for x in range(int(cx - rx) - 1, int(cx + rx) + 2):
                    u, v = (x - cx) / rx, (y - cy) / ry
                    if u * u + v * v <= 1:
                        k = 0.7 + 0.3 * rng.random()
                        L.put(y, x, rng.choice(["@", "%", "&", "8"]), tuple(c * k for c in LEAVES), 0.25 * (1 - (y - cy + ry) / (2 * ry + 1)) + 0.05 * rng.random())
                    elif abs(x - cx) < 0.6 and y > cy:
                        L.put(y, x, "|", TRUNK, 0.05)
        hw = max(5, round(0.075 * self.Sx))
        spec = ([("ground", hw - 1)] * 3 + [("eave", hw + 2), ("slope", hw), ("upper", hw - 2),
                                             ("eave", hw), ("ridge", hw - 1)])
        y, n = hz - 1, len(spec)
        for i, (kind, half) in enumerate(spec):
            xl, xr = xc - half, xc + half
            for x in range(xl, xr + 1):
                o = 0.35 + 0.6 * i / n + 0.02 * abs(x - xc) / half
                if kind in ("ground", "upper"):
                    if x in (xl, xr) or (x - xl) % 2 == 0:
                        L.put(y, x, "|", BRICK, o)
                    elif kind == "ground" and abs(x - xc) <= 1:
                        L.put(y, x, " ", BLACK, o)   # the doorway
                    else:
                        L.put(y, x, ":", AMBER_OFF, o)
                        win.put(y, x, "#", AMBER_ON, rng.uniform(0, 0.7))
                elif kind == "eave":
                    L.put(y, x, "`" if x == xl else ("'" if x == xr else "~"), TILE, o)
                elif kind == "slope":
                    L.put(y, x, "/" if x == xl else ("\\" if x == xr else "="), TILE, o)
                else:   # the curved ridge, ends swept up like a swallow's tail
                    L.put(y, x, "\\" if x == xl else ("/" if x == xr else "_"), RIDGE_TOP, o)
            if kind == "ridge":
                L.put(y - 1, xl - 1, "'", RIDGE_TOP, 0.99)
                L.put(y - 1, xr + 1, "'", RIDGE_TOP, 0.99)
            y -= 1
        gx = xc - round(a * 0.6)   # moon gate in the garden wall
        for x in range(xc - a, xc + a + 1):
            o = 0.28 + 0.12 * (x - xc + a) / (2 * a)
            for yy in (hz - 2, hz - 1):
                if abs(x - gx) <= 1:
                    L.put(yy, x, " ", BLACK, o)
                elif abs(x - gx) == 2:
                    L.put(yy, x, ("(" if x < gx else ")") if yy == hz - 2 else "|", LIME, o)
                elif yy == hz - 2:
                    L.put(yy, x, "=", TILE, o)
                elif (x - xc) % 6 == 3:   # lattice "flower" windows
                    L.put(yy, x, ":", AMBER_OFF, o)
                    win.put(yy, x, "#", AMBER_ON, rng.uniform(0, 0.7))
                else:
                    L.put(yy, x, "." if (x * 7 + yy) % 5 == 0 else " ", LIME, o)
        return L, win

    def _city_hall(self, rng):
        """New Taipei City Hall: a podium and two square towers joined at the top."""
        W, H, hz = self.w, self.h, self.hz
        L, win = Layer(H, W), Layer(H, W)
        xc = self.xhall
        ytop = max(4, int(round(0.15 * H)))
        yb = hz - 1
        h = yb - ytop + 1
        th = max(6, round(0.12 * self.Sx))            # half width of the pair of towers
        sw = max(1, round(th * 0.12))                  # half width of the slot between them
        pod = max(3, round(h * 8 / 33))                # the 8-storey low block
        ph = th + max(2, round(0.2 * th))
        crown = max(2, round(h * 0.07))
        face = np.zeros((H, W), bool)
        u = np.zeros((H, W), np.float32)
        v = np.zeros((H, W), np.float32)
        y_tower_bottom = yb - pod
        for i, y in enumerate(range(yb, ytop - 1, -1)):
            o = i / (h - 1)
            if y > y_tower_bottom:   # podium
                for x in range(xc - ph, xc + ph + 1):
                    if x in (xc - ph, xc + ph):
                        L.put(y, x, "|", HALL, o)
                    elif y == y_tower_bottom + 1:
                        L.put(y, x, "=", HALL, o)
                    elif (x - xc) % 2 == 0:
                        L.put(y, x, ":", HALL_GLASS, o)
                        win.put(y, x, ":", HALL_ON, rng.uniform(0, 0.8))
                    else:
                        L.put(y, x, " ", BLACK, o)
                continue
            in_crown = y < ytop + crown
            for x in range(xc - th, xc + th + 1):
                d = abs(x - xc)
                if x in (xc - th, xc + th):
                    c, rgb = "|", HALL
                elif in_crown:
                    c, rgb = ("_" if y == ytop else "="), HALL
                    if y > ytop:
                        win.put(y, x, "=", HALL_ON, rng.uniform(0, 0.2))
                elif d <= sw:
                    c, rgb = ".", tuple(k * 0.6 for k in HALL_GLASS)
                elif d == sw + 1:
                    c, rgb = "|", HALL
                else:
                    c, rgb = ":", HALL_GLASS
                    if rng.random() < 0.85:
                        win.put(y, x, "#" if rng.random() < 0.6 else ":", HALL_ON, rng.uniform(0.1, 1.0))
                L.put(y, x, c, rgb, o + 0.01 * d / th)
                if x not in (xc - th, xc + th):
                    face[y, x] = True
                    u[y, x] = (x - (xc - th)) / (2 * th)
                    v[y, x] = (y - ytop) / max(1, y_tower_bottom - ytop)
        for dx in (-th // 2, th // 2):   # antennas with red lights on the roof
            L.put(ytop - 1, xc + dx, "|", STEEL, 1.0)
            self.blinkers.append((ytop - 2, xc + dx, L, ytop - 1, 0.3 if dx < 0 else 0.8))
        self.face_idx = np.flatnonzero(face)
        self.face_u, self.face_v = u.ravel()[self.face_idx], v.ravel()[self.face_idx]
        self.hall_top = ytop
        self.digit_y = (ytop + crown + y_tower_bottom) // 2
        self.digit_pw = max(1, (2 * th + 1) // 8)
        self.digit_ph = max(1, round(self.digit_pw / 2))
        return L, win

    def _balloon_shape(self):
        """The giant hot-air balloon that is this year's main light."""
        R = max(5.0, 0.075 * self.Sx)
        self.b_r, self.b_ry = R, R * 1.25 / ASPECT
        self.b_base = self.hz - 2                   # bottom of the basket, on the plaza

    def _bridge(self):
        """The Crescent Bridge: two unequal steel arches tied to one deck."""
        W, H, hz = self.w, self.h, self.hz
        self.deck_y = hz + max(3, round(0.4 * (H - hz)))   # out on the water, in front of everything
        self.arch = []   # (y, x, char, position along the bridge 0..1)
        for a, b, hp in ((0.03, 0.60, 0.13), (0.60, 0.97, 0.08)):
            xa, xb = a * W, b * W
            c, half = (xa + xb) / 2, (xb - xa) / 2
            peak = min(hp * self.Sx / ASPECT, self.deck_y - hz - 1)

            def yf(x):
                return self.deck_y - peak * max(0.0, 1 - ((x - c) / half) ** 2)
            for x in range(int(math.ceil(xa)), int(xb) + 1):
                y0, y1 = yf(x - 0.5), yf(x + 0.5)
                slope = y1 - y0
                lo, hi = int(round(min(y0, y1))), int(round(max(y0, y1)))
                if abs(slope) > 2.5:
                    ch = "|"
                elif abs(slope) < 0.25:
                    ch = "_" if abs(x - c) < half * 0.2 else "-"
                else:
                    ch = "/" if slope < 0 else "\\"
                for y in range(lo, hi + 1):
                    if y < self.deck_y:
                        self.arch.append((y, x, ch, x / W))
                if x % 4 == 0:   # hanger cables down to the deck
                    for y in range(hi + 1, self.deck_y):
                        self.arch.append((y, x, ":", -1.0 - x / W))

    # ---------------------------------------------------------- per frame
    def frame(self, t):
        t %= LOOP
        cv = Canvas(self.h, self.w)
        P = {name: stage_progress(name, t) for name in STAGE_T}
        self._draw_stars(cv, t, P["sky"])
        self._draw_moon(cv, t)
        cv.blit(self.mountains, P["mountains"])
        cv.blit(self.city, P["city"])
        cv.blit(self.city_win, P["lights"], tip=0)
        self._draw_balloon(cv, t, P)
        self._draw_tree(cv, t, P)
        cv.blit(self.lin, P["lin_garden"])
        cv.blit(self.lin_win, P["lights"], tip=0)
        cv.blit(self.hall, P["city_hall"])
        cv.blit(self.hall_win, P["lights"], tip=0)
        self._light_hall(cv, t, P)
        self._draw_blinkers(cv, t, P)
        self._draw_lasers(cv, t)
        self._draw_meteor(cv, t)
        self._draw_river(cv, t, P["river"])
        self._draw_arches(cv, t, P["moon_bridge"])
        self._draw_deck(cv, t, P["moon_bridge"])
        self._draw_snow(cv, t)
        self._draw_greeting(cv, t, P["merry_xmas"])
        if t > FADE:
            cv.rgb *= 1 - smooth((t - FADE) / (LOOP - FADE))
        if self.hud:
            self._draw_hud(cv, t)
        return cv

    def _draw_stars(self, cv, t, p):
        tw = 0.55 + 0.45 * np.sin(t * self.star_rate + self.star_phase)
        for y, x, b, o in zip(self.star_y, self.star_x, tw, self.star_order):
            if o <= p:
                c = "*" if b > 0.92 else ("+" if b > 0.7 else ".")
                cv.put(y, x, c, tuple(v * (0.35 + 0.65 * b) for v in STAR))

    def _draw_moon(self, cv, t):
        r = max(4.0, 0.075 * self.Sx)
        xm = 0.30 * self.w
        y_end = 0.20 * self.h
        ym = self.hz - 1 - (self.hz - 1 - y_end) * smooth(t / 6.0)
        y0, y1 = int(ym - r / ASPECT) - 1, int(ym + r / ASPECT) + 2
        for y in range(max(0, y0), min(self.hz, y1)):
            for x in range(int(xm - r) - 1, int(xm + r) + 2):
                dx, dy = (x - xm) / r, (y - ym) * ASPECT / r
                d = dx * dx + dy * dy
                shadow = (dx + 0.62) ** 2 + (dy - 0.10) ** 2
                if d > 1 or shadow < 0.72:
                    continue
                depth = min(1 - d, shadow - 0.72)   # distance from either edge
                c = "@" if depth > 0.22 else ("#" if depth > 0.12 else ("*" if depth > 0.05 else "."))
                k = 0.75 + 0.25 * min(1.0, depth * 5)
                cv.put(y, x, c, tuple(v * k for v in MOON))

    def _draw_balloon(self, cv, t, P):
        if P["city"] < 0.7:
            return
        lit = smooth((t - SHOW) / 0.8) if SHOW <= t else 0.0
        R, ry = self.b_r, self.b_ry
        bob = 0.5 * math.sin((t - SHOW) * 1.3) * lit
        base = self.b_base + bob
        cy = base - 4 - ry * 0.8
        xc = self.xballoon
        for y in range(int(cy - ry) - 1, int(cy + ry) + 2):
            dy = (y - cy) / ry
            if abs(dy) > 1:
                continue
            half = R * math.sqrt(1 - dy * dy) * (1 - 0.45 * max(0.0, dy))   # tapers to the mouth
            for x in range(int(xc - half), int(xc + half) + 1):
                dx = (x - xc) / max(half, 0.5)
                if abs(dx) > 1:
                    continue
                gore = int((dx + 1) / 2 * 6 - 1e-9)
                edge = abs(dx) > 0.82
                c = ("(" if dx < 0 else ")") if edge else ("_" if dy < -0.9 else ("|" if abs((dx + 1) * 3 % 1) < 0.18 else ":"))
                if lit > 0:
                    pulse = 0.7 + 0.3 * math.sin(t * 3 + gore + dy * 2)
                    k = lit * pulse
                    base_rgb = BALLOON[gore % len(BALLOON)]
                    rgb = tuple(0.2 * (1 - lit) + v * k for v in base_rgb)
                    if not edge and hash01(x * 31 + y, int(t * 6)) > 0.93:
                        c, rgb = "*", WHITE
                else:
                    rgb = (0.24, 0.22, 0.30)
                cv.put(y, x, c, rgb)
        mouth = cy + ry
        for k in (1, 2):   # ropes and the basket
            cv.put(mouth + k, xc - 2 + k * 0.5, "\\", STEEL)
            cv.put(mouth + k, xc + 2 - k * 0.5, "/", STEEL)
        for dx in range(-1, 2):
            cv.put(mouth + 3, xc + dx, "#", (0.8, 0.55, 0.3) if lit else (0.35, 0.25, 0.18))
        cv.put(mouth + 3, xc - 2, "[", (0.6, 0.45, 0.3))
        cv.put(mouth + 3, xc + 2, "]", (0.6, 0.45, 0.3))

    def _draw_tree(self, cv, t, P):
        if P["city"] < 0.8:
            return
        ht = max(6, round(0.13 * self.h))
        xc, base = self.xtree, self.hz - 2
        on = P["lights"] > 0.5
        show = t >= SHOW
        cv.put(base, xc, "|", TRUNK)
        for r in range(ht - 1):
            y = base - 1 - (ht - 2 - r)
            half = int(round((r + 1) * 0.55))
            for x in range(xc - half, xc + half + 1):
                if x == xc - half:
                    c = "/"
                elif x == xc + half:
                    c = "\\"
                else:
                    c = "^"
                rgb = (0.12, 0.42, 0.20)
                hsh = hash01(x * 13 + y, 3)
                if on and c == "^" and hsh > 0.6:
                    blink = (t * (1.5 + hsh) + hsh * 9) % 1 < (0.75 if show else 0.55)
                    if blink:
                        c, rgb = "o", hsv(hsh * 5, 0.7, 1.0)
                cv.put(y, x, c, rgb)
        cv.put(base - ht, xc, "*", (1.0, 0.85, 0.3) if on else (0.5, 0.45, 0.25))

    def _light_hall(self, cv, t, P):
        """City Hall's facade as a screen: countdown digits, then the projection show."""
        cd = P["countdown"]
        idx, u, v = self.face_idx, self.face_u, self.face_v
        vis = self.hall.order.ravel()[idx] <= P["city_hall"]
        idx, u, v = idx[vis], u[vis], v[vis]
        ch = cv.ch.reshape(-1)
        rgb = cv.rgb.reshape(-1, 3)
        if 0 < cd < 1:
            ch[idx] = ":"
            rgb[idx] = (0.06, 0.07, 0.11)
            n = 5 - min(4, int(cd * 5))
            pulse = 1 - 0.55 * ((cd * 5) % 1)
            pw, ph = self.digit_pw, self.digit_ph
            x0 = self.xhall - (3 * pw) // 2
            y0 = self.digit_y - (5 * ph) // 2
            for r, line in enumerate(DIGITS[str(n)]):
                for c, bit in enumerate(line):
                    if bit == "#":
                        for qy in range(ph):
                            for qx in range(pw):
                                cv.put(y0 + r * ph + qy, x0 + c * pw + qx, "#", (pulse, 0.95 * pulse, 0.8 * pulse))
            return
        if t < SHOW:
            return
        s = t - SHOW
        mode = int(s // 2) % 4   # a new picture on every chord change
        k = smooth(s / 0.6)
        if mode == 0:     # a rainbow washing across
            hue = u * 0.6 + v * 0.3 - s * 0.4
            col = np.array([hsv(h, 0.6, 1.0) for h in hue]).reshape(-1, 3)
            chars = np.full(len(idx), "#")
        elif mode == 1:   # a giant snowflake
            dx, dy = u - 0.5, (v - 0.5) * 1.6
            r, a = np.hypot(dx, dy), np.arctan2(dy, dx)
            on = ((np.cos(6 * a) > 0.82) & (r < 0.48)) | (np.abs(r - 0.28) < 0.03) | (r < 0.06)
            col = np.where(on[:, None], np.array(SNOW), np.array([0.10, 0.18, 0.42]))
            chars = np.where(on, "*", ":")
        elif mode == 2:   # red, green and gold candy stripes
            band = np.floor(u * 6 + v * 9 - s * 2).astype(int) % 3
            pal = np.array([[1.0, 0.25, 0.25], [0.30, 0.90, 0.45], [1.0, 0.80, 0.30]])
            col = pal[band]
            chars = np.full(len(idx), "#")
        else:             # warm gold with twinkles
            tw = hash01(idx, int(t * 8)) > 0.88
            col = np.where(tw[:, None], np.array(WHITE), np.array([1.0, 0.72, 0.30]) * (0.6 + 0.3 * v[:, None]))
            chars = np.where(tw, "*", ":")
        ch[idx] = chars
        rgb[idx] = rgb[idx] * (1 - k) + col * k

    def _draw_blinkers(self, cv, t, P):
        for y, x, layer, below, ph in self.blinkers:
            p = P["city_hall"] if layer is self.hall else P["city"]
            if layer.order[below, x] <= p and (t * 1.1 + ph) % 1 < 0.45:
                cv.put(y, x, "*", (1.0, 0.18, 0.15))

    def _draw_lasers(self, cv, t):
        """Beams from the plaza sweeping the sky behind everything."""
        if t < SHOW:
            return
        k = smooth((t - SHOW) / 1.0)
        for j, xf in enumerate((0.40, 0.53, 0.64)):
            ang = math.radians(90 + 38 * math.sin(t * 0.8 + j * 2.1))
            dx, dy = math.cos(ang), -math.sin(ang) / ASPECT
            x, y = xf * self.w, self.hz - 2.0
            c = dir_char(math.degrees(math.atan2(-dy * ASPECT, dx)) + 180)
            rgb0 = hsv(0.33 + 0.28 * j + 0.05 * math.sin(t), 0.8, 0.55 * k)
            for step in range(int(self.h * 2.5)):
                x += dx
                y += dy
                yi, xi = int(round(y)), int(round(x))
                if not (0 <= yi < self.hz and 0 <= xi < self.w):
                    break
                if cv.ch[yi, xi] == " ":
                    f = 1 - step / (self.h * 2.5)
                    cv.put(yi, xi, c, tuple(v * f for v in rgb0))

    def _draw_meteor(self, cv, t):
        a = t - METEOR
        if not 0 <= a < 1.3:
            return
        x0, y0 = 0.62 * self.w, 0.04 * self.h
        x1, y1 = 0.36 * self.w, 0.30 * self.h
        q = a / 1.3
        hx, hy = x0 + (x1 - x0) * q, y0 + (y1 - y0) * q
        dx, dy = x1 - x0, (y1 - y0) * ASPECT
        ln = math.hypot(dx, dy)
        ux, uy = dx / ln, dy / ln / ASPECT
        c = dir_char(math.degrees(math.atan2(dy, dx)))
        fade = 1 - q * q
        for j in range(1, 12):
            k = (1 - j / 12) * fade
            cv.put(hy - uy * j, hx - ux * j, c, (0.75 * k, 0.85 * k, 1.0 * k))
        cv.put(hy, hx, "*", tuple(v * fade for v in WHITE))

    def _draw_arches(self, cv, t, p):
        """The bridge's arches, lit in colours that flow along them, and their
        reflection under the deck: together they make the "crescent moons"."""
        if p <= 0:
            return
        flip = {"/": "\\", "\\": "/", "_": "-"}
        for y, x, c, s in self.arch:
            pos = s if s >= 0 else -1 - s
            if pos > p * 1.02:
                continue
            rgb = (0.30, 0.32, 0.40) if s < 0 else hsv(pos * 1.5 - t * 0.25, 0.6, 1.0)
            cv.put(y, x, c, rgb)
            ry = 2 * self.deck_y - y
            if ry < self.h and s >= 0 and hash01(x * 7 + ry, int(t * 4)) > 0.25:
                cv.put(ry, x, flip.get(c, c), tuple(v * 0.38 for v in rgb))

    def _draw_deck(self, cv, t, p):
        if p <= 0:
            return
        y = self.deck_y
        for x in range(min(self.w, int(p * self.w * 1.02) + 1)):
            lamp = x % 6 == 0
            k = 0.85 + 0.15 * math.sin(t * 2 + x * 0.4) if lamp else 0.55
            cv.put(y, x, "*" if lamp else "=", tuple(v * k for v in DECK))

    def _draw_river(self, cv, t, p):
        """Waves on the Dahan River plus the reflection of everything above."""
        if p <= 0:
            return
        H, W, hz = self.h, self.w, self.hz
        ys = np.arange(hz, H)[:, None]
        xs = np.arange(W)[None, :]
        d = (ys - hz) / max(H - hz - 1, 1)
        wave = np.sin(xs * 0.42 + ys * 1.7 - t * 2.0) + 0.6 * np.sin(xs * 0.17 - ys * 0.9 + t * 1.3)
        ch = np.where(wave > 1.0, "~", np.where(wave > 0.25, "-", np.where(wave > -0.5, ".", " ")))
        rgb = WATER[None, None, :] * (0.55 + 0.3 * wave[..., None]) * (1 - 0.45 * d[..., None])
        my = np.clip(2 * hz - 1 - ys, 0, hz - 1)
        dx = np.round(np.sin(ys * 1.3 + t * 2.4 + xs * 0.05) * (0.5 + 1.6 * d)).astype(int)
        mx = np.clip(xs + dx, 0, W - 1)
        src = cv.rgb[my, mx]
        b = src.max(-1)
        refl = (b > 0.32) & (wave + 0.7 * np.sin(xs * 1.1 + t * 3.0) > -0.9)
        rch = np.where(b > 0.75, "|", np.where(b > 0.5, "!", ":"))
        ch = np.where(refl, rch, ch)
        rgb = np.where(refl[..., None], src * (0.75 - 0.4 * d[..., None]), rgb)
        edge = p * (W + 6) + 2 * np.sin(np.arange(hz, H) * 1.7)[:, None]
        show = xs < edge
        cv.ch[hz:] = np.where(show, ch, " ")
        cv.rgb[hz:] = np.where(show[..., None], rgb, 0)

    def _draw_snow(self, cv, t):
        if t < SHOW:
            return
        h1, h2, h3, h4, h5 = self.snow
        s = t - SHOW
        ramp = clamp01(s / 2.5)
        speed = 1.5 + 2.5 * h2
        y = (speed * s - h3 * (self.h + 4)) % (self.h + 4) - 2
        x = h1 * self.w + 1.2 * np.sin(t * 0.7 + np.arange(len(h1)) * 1.3)
        for i in np.flatnonzero(h4 < ramp):
            if y[i] < self.deck_y:
                c = "*" if h2[i] > 0.7 else ("'" if h2[i] > 0.35 else ".")
                cv.put(y[i], x[i], c, tuple(v * (0.55 + 0.45 * h5[i]) for v in SNOW))

    def _draw_greeting(self, cv, t, p):
        if p <= 0:
            return
        msg = "MERRY CHRISTMAS"
        n = int(len(msg) * clamp01(p * 4))
        y = max(2, int(round(0.06 * self.h)))
        x = (self.w - len(msg)) // 2
        cv.ch[y: y + 3, max(0, x - 1): x + len(msg) + 1] = " "
        pal = [(1.0, 0.35, 0.35), (0.45, 1.0, 0.55), (1.0, 0.82, 0.35)]
        for i, c in enumerate(msg[:n]):
            k = 0.8 + 0.2 * math.sin(t * 5 + i * 0.7)
            cv.put(y, x + i, c, tuple(v * k for v in pal[(i + int(t * 2)) % 3]))
        if n < len(msg) and (t * 3) % 1 < 0.6:
            cv.put(y, x + n, "_", (1.0, 0.82, 0.35))
        if p * 4 > 1.3:
            sub = f"BANQIAO {self.year}"
            cv.text(y + 1, (self.w - len(sub)) // 2, sub, (0.75, 0.80, 0.92))
        if self.tag and p * 4 > 1.6:
            cv.text(y + 2, (self.w - len(self.tag)) // 2, self.tag, HUD)

    def _draw_hud(self, cv, t):
        name = "fade" if t >= FADE else max((s for s in STAGES if s[1] <= t), key=lambda s: s[1])[0]
        pct = 100 if name == "fade" else int(100 * stage_progress(name, t))
        n = int((cv.ch != " ").sum())
        lines = [f"banqiao.py  stage {name:<11} {pct:3d}%", f"chars {n:6,d}   t {t:5.2f}s"]
        for i, s in enumerate(lines):
            cv.ch[i, : len(s) + 2] = " "
            cv.text(i, 1, s, HUD)


# ------------------------------------------------------------------ terminal
def to_ansi(cv, c256=False):
    q = (np.clip(cv.rgb, 0, 1) * 255).astype(np.uint8) & 0xF8
    if c256:
        q6 = (q.astype(np.int32) * 6 // 256)
        code = (16 + 36 * q6[..., 0] + 6 * q6[..., 1] + q6[..., 2]).tolist()
    else:
        code = q.tolist()
    chars = cv.ch.tolist()
    out = []
    for y in range(cv.h):
        last, parts = None, []
        for c, col in zip(chars[y], code[y]):
            if c != " " and col != last:
                parts.append(f"\x1b[38;5;{col}m" if c256 else f"\x1b[38;2;{col[0]};{col[1]};{col[2]}m")
                last = col
            parts.append(c)
        out.append("".join(parts))
    return "\n".join(out) + "\x1b[0m"


def terminal_size(args):
    cols, rows = shutil.get_terminal_size((120, 40))
    return max(60, cols), max(24, rows - 1)


def run_terminal(args):
    if os.name == "nt":
        os.system("")   # turn on ANSI escape codes in the Windows console
    if args.once:
        w, h = terminal_size(args)
        sys.stdout.write(to_ansi(Scene(w, h, args.tag, hud=not args.no_hud).frame(args.at), args.c256) + "\n")
        return
    write = sys.stdout.write
    write("\x1b[?1049h\x1b[?25l\x1b[2J")
    scene, t0 = None, time.perf_counter()
    try:
        while True:
            start = time.perf_counter()
            w, h = terminal_size(args)
            if scene is None or (scene.w, scene.h) != (w, h):
                scene = Scene(w, h, args.tag, hud=not args.no_hud)
                write("\x1b[2J")
            t = args.at + (start - t0) * args.speed
            write("\x1b[H" + to_ansi(scene.frame(t), args.c256))
            sys.stdout.flush()
            time.sleep(max(0.0, 1 / args.fps - (time.perf_counter() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        write("\x1b[0m\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()


# ------------------------------------------------------------------ music
# An original soundtrack, synthesised from scratch and timed to the stages.
# Warm and unhurried: felt piano, a music box and soft strings on the
# I-V-vi-iii-IV-I-IV-V progression in D major (60 BPM, one chord every
# two seconds), a gentle heartbeat instead of drums, and
# jingling sleigh bells once the light show starts.
SR = 44100
D, A_CS, BM, FSM = ([62, 66, 69, 76], 38), ([61, 64, 69, 73], 37), ([59, 62, 66, 69], 35), ([57, 61, 64, 69], 42)
G, D_FS, GADD9, A = ([55, 59, 62, 66], 43), ([57, 62, 66, 69], 42), ([55, 59, 62, 69], 43), ([57, 61, 64, 69], 45)
CHORDS = [(0, D), (2, A_CS), (4, BM), (6, FSM), (8, G), (10, D_FS), (12, GADD9), (14, A),
          (16, D), (18, A_CS), (20, BM), (22, FSM), (24, G), (26.5, A),
          (29, D), (31, A_CS), (33, G), (35, A), (37, ([62, 66, 69, 73, 76], 38))]
MELODY = [(0, 69, 1), (1, 74, 1), (2, 76, 1.5), (3.5, 73, .5), (4, 74, 1), (5, 71, .5), (5.5, 69, .5),
          (6, 71, .5), (6.5, 73, .5), (7, 76, 1), (8, 74, 2.5)]   # seconds after midnight, note, length


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tline(dur):
    return np.arange(int(dur * SR)) / SR


def band(x, lo=None, hi=None):
    """Soft low/high-pass in the frequency domain."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR) + 1e-3
    if hi:
        X /= 1 + (f / hi) ** 4
    if lo:
        X /= 1 + (lo / f) ** 4
    return np.fft.irfft(X, len(x))


def piano(m, dur=3.0, bright=0.55):
    """A soft felt piano: decaying harmonics, the high ones fading first."""
    f, t = hz(m), tline(dur)
    s = np.zeros_like(t)
    for k in range(1, 9):
        for det, amp in ((1.0, 1.0), (1.0016, 0.35)):
            s += amp * np.sin(2 * np.pi * f * k * det * (1 + 0.0004 * k * k) * t) * bright ** (k - 1) / k * np.exp(-t * (0.7 + 0.6 * k))
    return s * np.minimum(1, t / 0.008) * np.clip((dur - t) / 0.3, 0, 1)


def musicbox(m, dur=2.0):
    f, t = hz(m), tline(dur)
    s = (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.9) + 0.22 * np.sin(2 * np.pi * f * 4.07 * t) * np.exp(-t / 0.2)
         + 0.12 * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t / 0.45))
    return s * np.minimum(1, t / 0.002)


def harp(m, dur=2.0):
    f, t = hz(m), tline(dur)
    s = sum(np.sin(2 * np.pi * f * k * t) * np.exp(-t * (1.2 + 1.5 * k)) / k ** 1.5 for k in range(1, 6))
    return s * np.minimum(1, t / 0.003)


def strings(notes, dur, attack=1.2, release=1.5):
    """Soft string section: detuned, mellow saw-like voices with a slow swell."""
    t = tline(dur + release)
    s = np.zeros_like(t)
    for m in notes:
        for det in (-0.003, 0.0, 0.0027):
            w = 2 * np.pi * hz(m) * (1 + det) * t + det * 900
            s += sum(np.sin(k * w) / k ** 1.8 for k in range(1, 6))
    env = np.minimum(1, t / attack) * np.clip((dur + release - t) / release, 0, 1)
    vib = 1 + 0.05 * np.sin(2 * np.pi * 0.25 * t)
    return s * env * vib / (3 * len(notes))


def bass(m, dur):
    t = tline(dur + 0.4)
    w = 2 * np.pi * hz(m) * t
    env = np.minimum(1, t / 0.03) * (0.7 + 0.3 * np.exp(-t / 0.6)) * np.clip((dur + 0.4 - t) / 0.4, 0, 1)
    return (np.sin(w) + 0.2 * np.sin(2 * w)) * env


def heartbeat():
    t = tline(0.5)
    f = 42 + 40 * np.exp(-t / 0.05)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16)


def noise_hit(rng, dur, tau, lo=None, hi=None):
    t = tline(dur)
    n = band(rng.standard_normal(len(t)), lo, hi)
    return n / (np.abs(n).max() + 1e-9) * np.exp(-t / tau)


def reverb(x, rng, decay=2.6):
    t = tline(decay * 1.3)
    ir = band(rng.standard_normal(len(t)), hi=4000) * np.exp(-t * 6.9 / decay)
    N = 1 << (len(x) + len(ir) - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(ir, N), N)[: len(x)]
    return y * (np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-12))


def sleigh(rng):
    """A shake of sleigh bells: a few bright, slightly out-of-tune pings and a hiss."""
    t = tline(0.35)
    s = sum(np.sin(2 * np.pi * f * t) * np.exp(-t / 0.06) for f in rng.uniform(2600, 6800, 6)) / 6
    return 0.8 * s + 0.4 * noise_hit(rng, 0.35, 0.05, lo=6000)


def make_music(scene, seed=8):
    """One loop (LOOP seconds) of stereo audio, as a (2, n) float array."""
    rng = np.random.default_rng(seed)
    n = int(LOOP * SR)
    bus = np.zeros((2, n))

    def add(t, sig, gain=1.0, pan=0.0):
        i = int(round(t * SR))
        if i >= n:
            return
        if i < 0:
            sig, i = sig[-i:], 0
        j = min(n, i + len(sig))
        a = (max(-1.0, min(1.0, pan)) + 1) * math.pi / 4
        bus[0, i:j] += sig[: j - i] * gain * math.cos(a)
        bus[1, i:j] += sig[: j - i] * gain * math.sin(a)

    spans = [(t0, (CHORDS[k + 1][0] if k + 1 < len(CHORDS) else LOOP) - t0, ch) for k, (t0, ch) in enumerate(CHORDS)]
    for t0, dur, (notes, root) in spans:
        # strings all the way through, swelling towards the light show
        if t0 < 6:
            g = 0.10 + 0.03 * t0
        elif t0 < 24:
            g = 0.26
        elif t0 < SHOW:
            g = 0.30 if t0 < 26 else 0.38
        else:
            g = 0.40
        add(t0, strings(notes, dur, attack=1.6 if t0 in (24, 26.5) else 1.0), g)
        if t0 >= 8:
            add(t0, bass(root, dur - 0.1), 0.22 if t0 < SHOW else 0.26)
        # felt piano: a slow broken chord, like a lullaby
        if 6 <= t0 < 24:
            seq = [notes[0] - 12, notes[1], notes[2], notes[3]]
            for k, m in enumerate(seq):
                add(t0 + 0.5 * k, piano(m, 2.6), (0.30, 0.22, 0.22, 0.20)[k], -0.25 + 0.15 * k)
        if t0 >= SHOW:   # during the show: the whole chord rolled, then broken
            for k, m in enumerate(notes):
                add(t0 + 0.04 * k, piano(m - 12, 3.0), 0.12, -0.3 + 0.15 * k)
            for k, m in enumerate([notes[1], notes[2], notes[3]]):
                add(t0 + 0.5 + 0.5 * k, piano(m, 2.2), 0.13, 0.2 - 0.15 * k)

    # night sky: a music box picks out the stars
    for k in range(11):
        t = 0.4 + k * 0.5
        notes = [ns for s, (ns, _) in CHORDS if s <= t][-1]
        add(t, musicbox(rng.choice(notes) + 24, 2.2), 0.07, rng.uniform(-0.7, 0.7))
    # the Lin Family Garden: a little plucked pentatonic phrase, like a zither
    for k, m in enumerate([62, 64, 66, 69, 71, 69, 74]):
        add(10.0 + k * 0.18, harp(m + 12, 1.8), 0.10, -0.6 + k * 0.05)
    # City Hall rises: one music-box note per few floors, climbing
    for k, m in enumerate([69, 71, 74, 76, 78, 81, 83, 86, 88, 90]):
        add(13.0 + k * 0.5, musicbox(m, 1.6), 0.10, 0.2 + k * 0.05)
    # Pachelbel's falling line (F# E D C#) as the city lights up
    for t, m in ((16, 78), (18, 76), (20, 74), (22, 73)):
        add(t, piano(m, 2.4, bright=0.6), 0.26, 0.1)
        add(t, musicbox(m + 12, 2.0), 0.05, 0.3)
    for k, m in enumerate([62, 64, 66, 69, 71, 74, 76, 78, 81, 83, 86]):   # a harp run when the lights come on
        add(18.0 + k * 0.07, harp(m, 2.2), 0.10, -0.7 + k * 0.14)
    for k in range(int((24 - 18) / 0.5)):        # a quiet shaker keeps it moving
        add(18.0 + k * 0.5 + 0.25, noise_hit(rng, 0.09, 0.035, lo=5000, hi=11000), 0.035, 0.35)
    # the Crescent Bridge: a harp glides up one arch and down the other
    for k, m in enumerate([69, 71, 74, 76, 78, 81, 78, 76, 74]):
        add(21.6 + k * 0.12, harp(m + 12, 1.6), 0.07, -0.8 + k * 0.2)
    for k, m in enumerate([93, 90, 88, 86, 81]):   # a meteor
        add(METEOR + 0.1 + k * 0.09, musicbox(m, 1.5), 0.05, 0.3 - k * 0.12)
    # countdown to the show: a music-box tick and a heartbeat each second, climbing
    for k, m in enumerate([81, 83, 85, 86, 88]):
        add(24.0 + k, musicbox(m, 1.2), 0.14)
        add(24.0 + k, heartbeat(), 0.22 + 0.04 * k)
    # the show: the balloon lights up with a sparkle, then the melody and sleigh bells
    for k, m in enumerate([86, 90, 93, 98]):
        add(SHOW + k * 0.06, musicbox(m, 2.5), 0.08, -0.2 + k * 0.15)
    for b, m, d in MELODY:
        add(SHOW + b, piano(m, d + 1.6, bright=0.62), 0.30, 0.05)
        add(SHOW + b, musicbox(m + 12, d + 1.0), 0.06, -0.2)
    for k in range(4):
        add(SHOW + 2 * k, heartbeat(), 0.22)
    for k in range(32):   # sleigh bells: shaken on the beat, lighter in between
        add(SHOW + 0.5 + k * 0.25, sleigh(rng), 0.05 if k % 2 else 0.09, 0.4)
    for _ in range(14):   # snowflakes
        add(rng.uniform(SHOW + 1.0, FADE), musicbox(int(rng.choice([86, 88, 90, 93, 95])), 1.5), 0.035, rng.uniform(-0.8, 0.8))
    for k, m in enumerate([62, 66, 69, 73, 76, 81, 86]):   # the last chord, rolled up into the stars
        add(37.0 + k * 0.12, musicbox(m + 12, 2.5) if k > 3 else piano(m, 2.5), 0.08, -0.4 + k * 0.13)

    out = np.zeros((2, n))
    for ch in range(2):
        out[ch] = bus[ch] + 0.55 * reverb(bus[ch], rng)
    t = np.arange(n) / SR
    fade = np.where(t > FADE, 1 - np.clip((t - FADE) / (LOOP - FADE), 0, 1) ** 1.5, 1.0)
    out *= fade * np.minimum(1, t / 0.05)
    out /= np.abs(out).max() + 1e-9
    return np.tanh(1.3 * out) / np.tanh(1.3) * 0.95


def write_wav(path, stereo):
    import wave
    pcm = (np.clip(stereo.T, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())


def music_for(scene, start, seconds):
    loop = make_music(scene)
    idx = (np.round((start + np.arange(int(seconds * SR)) / SR) * SR).astype(np.int64)) % loop.shape[1]
    return loop[:, idx]


# ------------------------------------------------------------------ video
FONTS = ["DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
         "/System/Library/Fonts/Menlo.ttc", "/Library/Fonts/Menlo.ttc",
         "C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf"]
KEYWORDS = set("""def class return for in if elif else import from while not and or None True
False with as lambda try except finally break continue pass yield global raise is""".split())
BUILTINS = set("range len int float min max abs zip enumerate round sum tuple list print np math".split())
TOKEN = re.compile(r"(#.*)|(\"[^\"]*\"|'[^']*')|(\b\d+\.?\d*\b)|([A-Za-z_]\w*)|(\s+)|(.)")


def load_font(size, path=None):
    from PIL import ImageFont
    for p in ([path] if path else []) + FONTS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    sys.exit("No monospace font found; pass one with --font path/to/font.ttf")


def blur(a, r):
    """Box blur along both image axes, twice (close to a gaussian)."""
    for _ in range(2):
        for ax in (0, 1):
            c = np.cumsum(np.pad(a, [(r + 1, r) if i == ax else (0, 0) for i in range(3)], mode="edge"), axis=ax)
            hi = np.take(c, range(2 * r + 1, c.shape[ax]), axis=ax)
            lo = np.take(c, range(0, c.shape[ax] - 2 * r - 1), axis=ax)
            a = (hi - lo) / (2 * r + 1)
    return a


class CodePanel:
    """The source of this file, typed out and scrolling, editor style."""

    def __init__(self, width, height, font_path=None):
        from PIL import Image, ImageDraw
        self.w, self.h = width, height
        self.font = font = load_font(max(12, width // 60), font_path)
        self.cw = font.getbbox("M" * 10)[2] / 10
        self.lh = int(font.size * 1.45)
        self.head = int(self.lh * 1.6)
        self.gut = int(self.cw * 5)
        with open(os.path.abspath(__file__), encoding="utf-8") as f:
            self.lines = f.read().rstrip("\n").split("\n")
        self.starts = np.cumsum([0] + [len(s) + 1 for s in self.lines])
        img = Image.new("RGB", (width, self.lh * len(self.lines)), (0, 0, 0))
        dr = ImageDraw.Draw(img)
        colours = [(110, 128, 110), (232, 160, 96), (160, 220, 150), None, None, (205, 210, 225)]
        for i, line in enumerate(self.lines):
            y = i * self.lh
            dr.text((self.cw * 0.5, y), f"{i + 1:>3}", font=font, fill=(70, 78, 98))
            x, prev = self.gut, ""
            for m in TOKEN.finditer(line):
                s = m.group(0)
                kind = m.lastindex - 1
                fill = colours[kind]
                if kind == 3:
                    fill = ((235, 120, 170) if s in KEYWORDS else (110, 200, 235) if s in BUILTINS
                            else (240, 210, 120) if prev in ("def", "class") else (205, 210, 225))
                if kind != 4:
                    dr.text((x, y), s, font=font, fill=fill)
                    prev = s
                x += self.cw * len(s)
        self.code = np.asarray(img, np.uint8)
        head = Image.new("RGB", (width, self.head), (22, 25, 36))
        dh = ImageDraw.Draw(head)
        r = self.head // 7
        for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            cx = int(self.cw * 1.5 + i * r * 3.2)
            dh.ellipse([cx - r, self.head // 2 - r, cx + r, self.head // 2 + r], fill=c)
        tx = int(self.cw * 1.5 + r * 10)
        dh.rectangle([tx - 6, 4, tx + self.cw * 11, self.head], fill=(12, 14, 22))
        dh.text((tx + 4, self.head // 2 - font.size // 2 - 2), "banqiao.py", font=font, fill=(220, 225, 240))
        self.header = np.asarray(head, np.float32) / 255

    def render(self, t, loop):
        out = np.zeros((self.h, self.w, 3), np.float32)
        out[:] = (12 / 255, 14 / 255, 22 / 255)
        out[: self.head] = self.header
        typed = (t % loop) / (loop * 0.97) * self.starts[-1]
        li = int(np.searchsorted(self.starts, min(typed, self.starts[-1] - 1), side="right") - 1)
        col = int(typed - self.starts[li])
        view = self.h - self.head
        vis = view / self.lh
        frac = li + min(col / max(len(self.lines[li]), 1), 1)
        top_px = int(max(0.0, frac - 0.72 * vis) * self.lh)
        crop = self.code[top_px: top_px + view].astype(np.float32) / 255
        cur_y = li * self.lh - top_px
        x_cut = int(self.gut + col * self.cw)
        if 0 <= cur_y < crop.shape[0]:
            crop[max(0, cur_y): cur_y + self.lh, x_cut:] = 0
            crop[max(0, cur_y): cur_y + self.lh, :] += (0.05, 0.06, 0.10)
            if (t * 2.5) % 1 < 0.6:
                crop[max(0, cur_y + 2): cur_y + self.lh - 2, x_cut: x_cut + int(self.cw)] = (0.9, 0.9, 1.0)
        crop[max(0, cur_y + self.lh):] = 0
        np.maximum(out[self.head: self.head + crop.shape[0]], crop, out=out[self.head: self.head + crop.shape[0]])
        return out


class Renderer:
    def __init__(self, width, height, cols, tag, font_path=None, code=False, hud=True):
        from PIL import Image, ImageDraw
        self.W, self.H = width, height
        self.cw = width // cols
        self.chh = self.cw * 2
        font = load_font(int(self.cw / 0.6), font_path)
        scene_px = int(height * (0.62 if code else 1.0))
        rows = scene_px // self.chh
        self.scene = Scene(cols, rows, tag, hud=hud)
        self.sw, self.sh = cols * self.cw, rows * self.chh
        self.x0 = (width - self.sw) // 2
        atlas = np.zeros((95, self.chh, self.cw), np.float32)
        for i in range(95):
            g = Image.new("L", (self.cw, self.chh), 0)
            ImageDraw.Draw(g).text((0, (self.chh - font.size) // 2 - 1), chr(32 + i), font=font, fill=255)
            atlas[i] = np.asarray(g, np.float32) / 255
        self.atlas = atlas
        y = np.linspace(0, 1, self.sh)[:, None, None]
        hz = self.scene.hz / rows
        sky = np.array([0.012, 0.016, 0.045]) * (1 - y / hz) + np.array([0.055, 0.040, 0.085]) * (y / hz)
        water = np.array([0.010, 0.018, 0.040]) * np.ones_like(y)
        self.bg = np.where(y < hz, sky, water).astype(np.float32) * np.ones((1, self.sw, 1), np.float32)
        self.panel = CodePanel(width, height - scene_px, font_path) if code else None
        self.scene_px = scene_px

    def frame(self, t):
        cv = self.scene.frame(t)
        codes = np.clip(cv.ch.view(np.uint32).reshape(cv.h, cv.w).astype(np.int64) - 32, 0, 94)
        g = self.atlas[codes][..., None] * cv.rgb[:, :, None, None, :]
        img = g.transpose(0, 2, 1, 3, 4).reshape(self.sh, self.sw, 3)
        f = 4
        small = img[: self.sh // f * f, : self.sw // f * f].reshape(self.sh // f, f, self.sw // f, f, 3).mean((1, 3))
        glow = np.repeat(np.repeat(blur(small, 3), f, 0), f, 1)
        fade = 1 - smooth((t % LOOP - FADE) / (LOOP - FADE)) if t % LOOP > FADE else 1.0
        scene = self.bg * fade + img
        scene[: glow.shape[0], : glow.shape[1]] += 1.6 * glow
        out = np.zeros((self.H, self.W, 3), np.float32)
        out[: self.sh, self.x0: self.x0 + self.sw] = scene
        if self.panel:
            out[self.scene_px:] = self.panel.render(t, LOOP)
            out[self.scene_px: self.scene_px + 2] = (0.16, 0.18, 0.26)
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def record(args):
    w, h = (int(v) for v in args.size.lower().split("x"))
    r = Renderer(w, h, args.cols, args.tag, args.font, code=args.code, hud=not args.no_hud)
    if args.record.lower().endswith(".png"):
        from PIL import Image
        Image.fromarray(r.frame(args.at)).save(args.record)
        print(f"saved {args.record}")
        return
    if args.record.lower().endswith(".wav"):
        write_wav(args.record, music_for(r.scene, args.at, args.seconds))
        print(f"saved {args.record}")
        return
    frames = int(args.seconds * args.fps)
    audio = []
    if not args.no_music:
        import tempfile
        wav = os.path.join(tempfile.mkdtemp(), "music.wav")
        write_wav(wav, music_for(r.scene, args.at, args.seconds))
        audio = ["-i", wav, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd = (["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-"] + audio
           + ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
              "-movflags", "+faststart", args.record])
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.perf_counter()
    for i in range(frames):
        proc.stdin.write(r.frame(args.at + i / args.fps).tobytes())
        if i % args.fps == 0:
            print(f"\r{i / args.fps:5.1f}s / {args.seconds:.0f}s", end="", flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"\rsaved {args.record} ({frames} frames, {time.perf_counter() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description="ASCII Banqiao Christmasland")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--speed", type=float, default=1.0, help="animation speed multiplier")
    ap.add_argument("--at", type=float, default=None, help="start time in seconds (32 = light show)")
    ap.add_argument("--once", action="store_true", help="print one frame and exit")
    ap.add_argument("--256", dest="c256", action="store_true", help="256-colour mode for older terminals")
    ap.add_argument("--tag", default="", help="text under the greeting, e.g. @yourname")
    ap.add_argument("--record", metavar="FILE",
                    help="render to .mp4 (needs pillow + ffmpeg), one .png, or just the music as .wav")
    ap.add_argument("--no-music", action="store_true", help="video without the soundtrack")
    ap.add_argument("--seconds", type=float, default=LOOP)
    ap.add_argument("--size", default="1080x1920", help="video size, WIDTHxHEIGHT")
    ap.add_argument("--cols", type=int, default=90, help="character columns in the video")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--font", help="monospace .ttf/.ttc for the video")
    ap.add_argument("--code", action="store_true", help="add a panel that types out this file under the view")
    ap.add_argument("--no-hud", action="store_true", help="hide the stage / time readout in the corner")
    args = ap.parse_args()
    if args.at is None:
        args.at = 32.0 if args.once or (args.record or "").lower().endswith(".png") else 0.0
    if args.record:
        record(args)
    else:
        run_terminal(args)


if __name__ == "__main__":
    main()
