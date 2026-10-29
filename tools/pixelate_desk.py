"""
Turn the desk photo into game art.

  python tools/pixelate_desk.py

Reads  assets/desk-photo.jpg  and writes
  assets/desk-bg.png        the desk with the movable objects lifted off and the desk painted in under them
  assets/desk-objects.png   the movable objects alone, in place, on a transparent layer
  assets/desk-preview.png   both together, 4x, to look at

Both layers are downsampled to the game's art grid, graded to 3 a.m., and ordered-dithered
(the same 4x4 Bayer matrix the street uses) into one shared palette, so the objects sit
pixel-for-pixel on top of the background they were cut from.
"""
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'assets' / 'desk-photo.jpg'
W, H = 400, 237

# movable objects, in photo pixels (x0, y0, x1, y1), and either how different from the surface
# under them a pixel must be to count as the object (a flood fill finds the outline), or, where
# paper sits on a pale wall and a flood fill can't tell them apart, the outline drawn by hand.
# Keep in sync with OBJ in index.html.
SHAPES = {
    'calendar': [('poly', [(118, 30), (140, 20), (280, 4), (304, 6), (316, 20), (319, 160), (317, 268), (156, 288), (124, 50)])],
    'receipt':  [('poly', [(938, 221), (1044, 198), (1101, 401), (1002, 439)])],
    'cards':    [('poly', [(1063, 516), (1152, 495), (1258, 499), (1263, 643), (1099, 651)])],
    'cd':       [('ell', 256, 503, 107, 38, 13), ('ell', 249, 602, 113, 74, 4), ('poly', [(146, 520), (352, 548), (354, 595), (142, 595)])],
}
OBJECTS = {
    'calendar': ((112, 0, 344, 292), 0),
    'clock':    ((125, 300, 362, 456), 46),
    'cd':       ((136, 462, 364, 674), 0),
    'receipt':  ((934, 194, 1104, 444), 0),
    'mug':      ((783, 422, 950, 594), 60),
    'mouse':    ((952, 526, 1060, 690), 60),
    'cards':    ((1058, 490, 1266, 656), 0),
    'postit':   ((944, 766, 1064, 890), 60),
    'flash':    ((1068, 793, 1250, 957), 60),
    'napkin':   ((1253, 756, 1452, 947), 60),
    'crumple':  ((1503, 686, 1667, 910), 56),
}

BAYER = np.array([0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]).reshape(4, 4)

# the street's palette; the photo's own colors are added to it below
STREET = ['#07070f', '#0b0b16', '#12142c', '#1d2146', '#35182a', '#5c2233', '#7a333b', '#8d8497', '#aca3b2',
          '#5f5870', '#221a28', '#34344c', '#6f6f8c', '#fff4c4', '#ffc21a', '#c98a00', '#e8323a', '#7a1a26',
          '#e38c46', '#ffc670', '#4f6fd8', '#8fb4ff', '#a44ee0', '#e08cff', '#34a8a0']


def hex2rgb(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def shape_mask(shape, size, rect):
    from PIL import ImageDraw
    m = Image.new('L', size, 0)
    d = ImageDraw.Draw(m)
    for sh in shape:
        if sh[0] == 'poly':
            d.polygon(sh[1], fill=255)
        else:
            _, cx, cy, rx, ry, deg = sh
            t = np.radians(deg)
            pts = [(cx + rx * np.cos(a) * np.cos(t) - ry * np.sin(a) * np.sin(t),
                    cy + rx * np.cos(a) * np.sin(t) + ry * np.sin(a) * np.cos(t)) for a in np.linspace(0, 2 * np.pi, 90)]
            d.polygon(pts, fill=255)
    out = np.asarray(m) > 0
    x0, y0, x1, y1 = rect
    clip = np.zeros_like(out)
    clip[y0:y1, x0:x1] = True
    return out & clip


def object_mask(img, rect, tol):
    """flood from the rect's border through pixels that look like the surface; whatever it can't reach is the object"""
    x0, y0, x1, y1 = rect
    a = img[y0:y1, x0:x1].astype(np.float32)
    h, w, _ = a.shape
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    # a few surface colors (desk, mousepad, wall, carpet) from the border, by a tiny k-means
    cents = border[np.linspace(0, len(border) - 1, 4).astype(int)].copy()
    for _ in range(12):
        lab = np.argmin(((border[:, None] - cents[None]) ** 2).sum(-1), 1)
        for k in range(len(cents)):
            if (lab == k).any():
                cents[k] = border[lab == k].mean(0)
    d = np.sqrt(((a[:, :, None] - cents[None, None]) ** 2).sum(-1)).min(-1)
    surface = d < tol
    seen = np.zeros((h, w), bool)
    stack = [(y, x) for y in range(h) for x in (0, w - 1)] + [(y, x) for x in range(w) for y in (0, h - 1)]
    stack = [(y, x) for y, x in stack if surface[y, x]]
    for y, x in stack:
        seen[y, x] = True
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and surface[yy, xx]:
                seen[yy, xx] = True
                stack.append((yy, xx))
    m = np.zeros(img.shape[:2], bool)
    m[y0:y1, x0:x1] = ~seen
    return m


def inpaint(img, hole):
    """onion-peel fill: each pass, hole pixels at the edge take the mean of their known neighbors"""
    out = img.astype(np.float32).copy()
    known = ~hole
    out[hole] = 0
    while not known.all():
        acc = np.zeros_like(out)
        cnt = np.zeros(known.shape, np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
            k = np.roll(np.roll(known, dy, 0), dx, 1)
            v = np.roll(np.roll(out, dy, 0), dx, 1)
            acc += v * k[..., None]
            cnt += k
        grow = (~known) & (cnt > 0)
        out[grow] = acc[grow] / cnt[grow][:, None]
        known = known | grow
    # soften the streaks the peel leaves: blur inside the hole only, a few times over
    for _ in range(6):
        b = out.copy()
        for ax in (0, 1):
            acc = np.zeros_like(b)
            for d in range(-6, 7):
                acc += np.roll(b, d, ax)
            b = acc / 13
        out[hole] = b[hole]
    # a little grain so the patch doesn't read as a smear
    rng = np.random.default_rng(3)
    out[hole] += rng.normal(0, 5, (hole.sum(), 1))
    return np.clip(out, 0, 255)


def downsample(a):
    return np.asarray(Image.fromarray(a.astype(np.uint8)).resize((W, H), Image.BOX)).astype(np.float32)


def grade(a):
    """3 a.m.: a little darker and cooler everywhere, the wall most of all; colors kept saturated"""
    yy, xx = np.mgrid[0:H, 0:W]
    lum = a.mean(-1, keepdims=True)
    a = lum + (a - lum) * 1.18                                    # a touch more saturation
    a = 255 * (np.clip(a, 0, 255) / 255) ** 1.12                  # gamma down
    wall = np.clip((70 - yy) / 70, 0, 1)[..., None]               # the wall above the desk, falling into the dark
    a = a * (1 - 0.32 * wall) + np.array([18, 10, 34]) * 0.32 * wall
    vig = ((xx - W * 0.55) / (W * 0.7)) ** 2 + ((yy - H * 0.55) / (H * 0.75)) ** 2
    a = a * (1 - 0.30 * np.clip(vig, 0, 1)[..., None])
    a = a * np.array([0.95, 0.94, 1.04])                          # toward the street's blue-violet night
    return np.clip(a, 0, 255)


def build_palette(a, n=40):
    px = a.reshape(-1, 3)
    rng = np.random.default_rng(1)
    cents = px[rng.choice(len(px), n, replace=False)].copy()
    for _ in range(20):
        lab = np.argmin(((px[::3, None] - cents[None]) ** 2).sum(-1), 1)
        for k in range(n):
            sel = px[::3][lab == k]
            if len(sel):
                cents[k] = sel.mean(0)
    return np.concatenate([cents, np.array([hex2rgb(h) for h in STREET], np.float32)])


def dither(a, pal):
    yy, xx = np.mgrid[0:H, 0:W]
    t = (BAYER[yy & 3, xx & 3][..., None] / 16 - 0.47) * 22
    b = np.clip(a + t, 0, 255)
    wts = np.array([0.30, 0.59, 0.11]) * 3
    idx = np.argmin((((b[:, :, None] - pal[None, None]) ** 2) * wts).sum(-1), -1)
    return pal[idx].astype(np.uint8)


def build_palette_any(a, n):
    """the same palette builder, for an image of any size"""
    px = a.reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(1)
    cents = px[rng.choice(len(px), n, replace=False)].copy()
    for _ in range(16):
        lab = np.argmin(((px[::3, None] - cents[None]) ** 2).sum(-1), 1)
        for k in range(n):
            sel = px[::3][lab == k]
            if len(sel):
                cents[k] = sel.mean(0)
    return cents


def dither_any(a, pal):
    """the same Bayer dither, for an image of any size"""
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    t = (BAYER[yy & 3, xx & 3][..., None] / 16 - 0.47) * 22
    b = np.clip(a + t, 0, 255)
    wts = np.array([0.30, 0.59, 0.11]) * 3
    idx = np.argmin((((b[:, :, None] - pal[None, None]) ** 2) * wts).sum(-1), -1)
    return pal[idx].astype(np.uint8)


def main():
    img = np.asarray(Image.open(SRC).convert('RGB'))
    size = (img.shape[1], img.shape[0])
    masks = {k: shape_mask(SHAPES[k], size, r) if k in SHAPES else object_mask(img, r, tol) for k, (r, tol) in OBJECTS.items()}
    hole = np.zeros(img.shape[:2], bool)
    for m in masks.values():
        hole |= m
    # grow the hole a couple of pixels so no object edge is left on the desk
    for _ in range(5):
        hole = hole | np.roll(hole, 1, 0) | np.roll(hole, -1, 0) | np.roll(hole, 1, 1) | np.roll(hole, -1, 1)
    bg_full = inpaint(img, hole)

    bg = grade(downsample(bg_full))
    fg = grade(downsample(img))
    objs = np.any(list(masks.values()), 0)
    alpha = np.asarray(Image.fromarray(objs.astype(np.uint8) * 255).resize((W, H), Image.BOX)) > 110
    pal = build_palette(fg)
    bg_q, fg_q = dither(bg, pal), dither(fg, pal)

    Image.fromarray(bg_q).save(ROOT / 'assets' / 'desk-bg.png', optimize=True)
    rgba = np.dstack([fg_q, (alpha * 255).astype(np.uint8)])
    Image.fromarray(rgba, 'RGBA').save(ROOT / 'assets' / 'desk-objects.png', optimize=True)
    comp = bg_q.copy()
    comp[alpha] = fg_q[alpha]
    Image.fromarray(comp).resize((W * 4, H * 4), Image.NEAREST).save(ROOT / 'assets' / 'desk-preview.png')

    s = W / img.shape[1]
    print('OBJ rects in art pixels:')
    for k, ((x0, y0, x1, y1), _) in OBJECTS.items():
        print(f"  {k}: [{int(x0 * s)}, {int(y0 * s)}, {int(np.ceil(x1 * s))}, {int(np.ceil(y1 * s))}],")


if __name__ == '__main__':
    main()
