"""
Assets for the look-up and level 2, from the Germany head piece.

  python tools/make_map_assets.py

Reads  assets/germany-head.png  (the satellite map with the face drawn over Germany) and writes
  assets/map.jpg        the map, cropped clear of Google's UI, full resolution, face and all
  assets/map-bare.jpg   the same map with the face lifted off and the land painted back in
  assets/face.png       just the face, on transparency, cropped to its box (offset printed below)
  assets/wall.png       the bedroom wall above the desk, in the desk's palette, with the
                        bare map hung on it as a poster (poster rect printed below)
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import pixelate_desk as PD

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / 'assets'
CROP = (45, 0, 1652, 985)            # clear of the side tab and the minimap thumbnail
WALL_W, WALL_H = 400, 250
POSTER = (124, 70, 276, 163)         # x0, y0, x1, y1 on the wall; keep in sync with WALL in index.html


def face_mask(img):
    """the face: everything inside its yellow outline, outline included"""
    a = img.astype(int)
    yellow = (a[..., 0] > 180) & (a[..., 1] > 150) & (a[..., 2] < 110) & (a[..., 0] - a[..., 2] > 110)
    # the black ink of the hat brim and zigzag closes the outline where it crosses it
    ys, xs = np.nonzero(yellow)
    box = np.zeros_like(yellow)
    box[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    ink = box & (a.max(-1) < 60)
    m = Image.fromarray(((yellow | ink) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(11))
    m = m.convert('L')
    ImageDraw.floodfill(m, (0, 0), 128)
    inside = np.asarray(m) != 128
    return np.asarray(Image.fromarray((inside * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(13))) > 0


def main():
    src = np.asarray(Image.open(A / 'germany-head.png').convert('RGB'))
    x0, y0, x1, y1 = CROP
    img = src[y0:y1, x0:x1].copy()
    Image.fromarray(img).save(A / 'map.jpg', quality=88)

    mask = face_mask(img)
    ys, xs = np.nonzero(mask)
    fx0, fy0, fx1, fy1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    rgba = np.dstack([img, (mask * 255).astype(np.uint8)])[fy0:fy1, fx0:fx1]
    Image.fromarray(rgba, 'RGBA').save(A / 'face.png', optimize=True)

    # the land under the face, painted back: the peel-and-blur fill, at a quarter size for speed
    small = np.asarray(Image.fromarray(img).resize((img.shape[1] // 4, img.shape[0] // 4), Image.BOX))
    sm = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(small.shape[1::-1], Image.BOX)) > 20
    filled = PD.inpaint(small, sm).astype(np.uint8)
    up = np.asarray(Image.fromarray(filled).resize(img.shape[1::-1], Image.BICUBIC))
    # a little of the real texture back on top, so the patch isn't a smear at full size
    tex = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(1))).astype(float)
    bare = img.copy()
    soft = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))).astype(float)[..., None] / 255
    # and the fine grain of real land, borrowed from France below and to the left
    det = tex - np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(7))).astype(float)
    det = np.roll(np.roll(det, 150, 0), -380, 1)
    bare = np.clip(img * (1 - soft) + (up + det * 1.1) * soft, 0, 255).astype(np.uint8)
    Image.fromarray(bare).save(A / 'map-bare.jpg', quality=88)

    # the wall: the top of the desk photo's wall, carried up and falling off into the dark
    desk = np.asarray(Image.open(A / 'desk-bg.png').convert('RGB')).astype(float)
    top = desk[0:6].mean(0)                                            # one color per column
    top = np.array([np.convolve(top[:, c], np.ones(15) / 15, mode='same') for c in range(3)]).T
    yy, xx = np.mgrid[0:WALL_H, 0:WALL_W]
    up_t = (WALL_H - 1 - yy) / WALL_H                                    # 0 at the seam, 1 at the ceiling
    wall = top[None, :, :] * (1 - 0.55 * up_t[..., None] ** 0.8)
    lamp = np.exp(-(((xx - 360) / 140) ** 2 + ((yy - WALL_H) / 170) ** 2))[..., None]   # the lamp, low and right
    wall = wall * (0.85 + 0.35 * lamp)
    wall = np.clip(wall + np.array([4, 2, 10]) * up_t[..., None], 0, 255)
    # the poster: the bare map, in a thin white border, a little darker for the hour
    px0, py0, px1, py1 = POSTER
    pw, ph = px1 - px0, py1 - py0
    art = np.asarray(Image.fromarray(bare).resize((pw - 4, ph - 4), Image.BOX)).astype(float)
    art = art * 0.78 + np.array([4, 2, 12])
    shadow_off = 3
    wall[py0 + shadow_off:py1 + shadow_off, px0 + shadow_off:px1 + shadow_off] *= 0.55
    wall[py0:py1, px0:px1] = np.array([206, 200, 214])
    wall[py0 + 2:py1 - 2, px0 + 2:px1 - 2] = art
    wall[py0:py1, px0:px1] *= (0.82 + 0.3 * lamp[py0:py1, px0:px1])
    # one dithered palette for the whole wall, poster included
    pal = np.concatenate([PD.build_palette_any(wall, 36), np.array([PD.hex2rgb(h) for h in PD.STREET], np.float32)])
    q = PD.dither_any(wall, pal)
    # tape at the poster's corners
    q = q.copy()
    for cx, cy in ((px0, py0), (px1 - 1, py0), (px0, py1 - 1), (px1 - 1, py1 - 1)):
        q[max(0, cy - 2):cy + 3, max(0, cx - 3):cx + 4] = (q[max(0, cy - 2):cy + 3, max(0, cx - 3):cx + 4] * 0.4 + np.array([230, 226, 200]) * 0.6).astype(np.uint8)
    Image.fromarray(q).save(A / 'wall.png', optimize=True)

    print('map size', img.shape[1], img.shape[0])
    print('face box', fx0, fy0, fx1 - fx0, fy1 - fy0)
    print('poster', POSTER)


if __name__ == '__main__':
    main()
