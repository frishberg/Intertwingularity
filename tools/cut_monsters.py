"""
The monsters, as the team drew them, cut off their maps.

  python tools/cut_monsters.py

Reads  assets/drawings/<key>.jpeg         (each monster drawn over a Google Maps screenshot)
       assets/drawings/<key>-happy.jpeg   (the same monster smiling, for the run home at the end)
writes assets/monster-<key>.png           (just the monster, on transparency, cropped to it)
       assets/monster-<key>-happy.png

The map is taken from the picture's border: a handful of map colors by a tiny k-means.
Then a flood from the border runs through anything that looks like map and isn't ink;
whatever it can't reach (everything inside the black outlines, and the parts drawn
off the map, like the razorback's horns) is the monster.

It prints, for each one, where the eye and the mouth are in the cut-out, as fractions of its
width and height. Those go in DRAWN in index.html.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / 'assets'

# eye and mouth, in the drawing's own pixels, found by eye; and how tall to keep the cut-out
MONSTERS = {
    'ar':   {'eye': (303, 95),   'mouth': (310, 185),  'max_h': 400},
    'sc':   {'eye': (625, 207),  'mouth': (630, 265),  'max_h': 600},
    'ny':   {'eye': (185, 180),  'mouth': (143, 215),  'max_h': 400},
    'in':   {'eye': (318, 226),  'mouth': (350, 280),  'max_h': 500},
    'head': {'eye': (562, 790),  'mouth': (318, 1203), 'max_h': 1000},
}
EDGE = 4          # the screenshots have a stray line along an edge or two
TOL = 34          # how far from every map color a pixel must be to count as drawing


def map_palette(a, n=10):
    ring = np.concatenate([a[:12].reshape(-1, 3), a[-12:].reshape(-1, 3), a[:, :12].reshape(-1, 3), a[:, -12:].reshape(-1, 3)])
    ring = ring[ring.max(-1) >= 95]                     # ink isn't map
    rng = np.random.default_rng(1)
    cents = ring[rng.choice(len(ring), n, replace=False)].copy()
    for _ in range(20):
        lab = np.argmin(((ring[:, None] - cents[None]) ** 2).sum(-1), 1)
        for k in range(n):
            if (lab == k).any():
                cents[k] = ring[lab == k].mean(0)
    return cents


def flood_from_border(passable):
    """True where the border can reach through passable pixels"""
    h, w = passable.shape
    m = Image.fromarray(np.where(passable, 255, 0).astype(np.uint8)).copy()   # fromarray can hand back a read-only view
    px = m.load()
    seeds = [(x, 0) for x in range(w)] + [(x, h - 1) for x in range(w)] + [(0, y) for y in range(h)] + [(w - 1, y) for y in range(h)]
    for xy in seeds:
        if px[xy] == 255:
            ImageDraw.floodfill(m, xy, 128)
    return np.asarray(m) == 128


def keep_big(mask, frac=0.01):
    """drop specks: keep only the pieces at least frac of the biggest"""
    m = Image.fromarray(mask.astype(np.uint8)).copy()   # 1 = unvisited monster
    px = m.load()
    h, w = mask.shape
    sizes, label = [], 2
    arr = np.asarray(m)
    ys, xs = np.nonzero(arr == 1)
    for y, x in zip(ys, xs):
        if px[x, y] != 1:
            continue
        ImageDraw.floodfill(m, (int(x), int(y)), label)
        sizes.append((label, int((np.asarray(m) == label).sum())))
        label += 1
        if label > 250:
            break
    if not sizes:
        return mask
    big = max(s for _, s in sizes)
    keep = [l for l, s in sizes if s >= big * frac]
    return np.isin(np.asarray(m), keep)


def cut(key, cfg, happy=False):
    name = f'{key}-happy' if happy else key
    src = Image.open(A / 'drawings' / f'{name}.jpeg').convert('RGB')
    a = np.asarray(src).astype(np.float32)[EDGE:-EDGE, EDGE:-EDGE]
    pal = map_palette(a)
    d = np.sqrt(((a[:, :, None] - pal[None, None]) ** 2).sum(-1)).min(-1)
    ink = a.max(-1) < 95
    passable = (d < TOL) & ~ink
    # close one-pixel gaps in the outlines before flooding
    passable = np.asarray(Image.fromarray(passable.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3))) > 0
    monster = ~flood_from_border(passable)
    # a pixel's worth of opening, so lone map specks don't survive, then the specks that are left go
    monster = np.asarray(Image.fromarray(monster.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))) > 0
    monster = keep_big(monster)
    # soft edge, a pixel in from the outline's outside so none of the map comes along
    alpha = Image.fromarray(monster.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.7))
    ys, xs = np.nonzero(monster)
    pad = 6
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(a.shape[1], xs.max() + 1 + pad), min(a.shape[0], ys.max() + 1 + pad)
    rgba = np.dstack([a.astype(np.uint8), np.asarray(alpha)])[y0:y1, x0:x1]
    out = Image.fromarray(rgba, 'RGBA')
    if out.height > cfg['max_h']:
        out = out.resize((round(out.width * cfg['max_h'] / out.height), cfg['max_h']), Image.LANCZOS)
    out.save(A / f'monster-{name}.png', optimize=True)
    if happy:
        print(f"  {name}: {out.width}x{out.height}")
        return
    w, h = int(x1 - x0), int(y1 - y0)
    fr = lambda p: (round(float(p[0] - EDGE - x0) / w, 3), round(float(p[1] - EDGE - y0) / h, 3))
    print(f"  {key}: {{ eye: {list(fr(cfg['eye']))}, mouth: {list(fr(cfg['mouth']))} }},   // {out.width}x{out.height}")


def main():
    for k, cfg in MONSTERS.items():
        cut(k, cfg)
        if (A / 'drawings' / f'{k}-happy.jpeg').exists():
            cut(k, cfg, happy=True)


if __name__ == '__main__':
    main()
