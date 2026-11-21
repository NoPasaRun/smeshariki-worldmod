"""Общий каркас правок рельефа сохранённого мира: рамка чанков в памяти, поверхность, запись изменённых секций.

Frame(world, x0, z0, x1, z1) читает блоки рамки (границы кратны 16) в vol[x, y, z] с общей палитрой idx; surface() даёт
G (индекс верхнего твёрдого блока, деревья и растения не считаются), GY (то же в мировых y), SURF (имя блока), WAT (над землёй вода).
write() пишет только изменённые секции, сбрасывает свет (isLightOn=0) и пересчитывает карты высот. Используется landscape_2026_11_20.py.
"""
import os, sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca
import snow_and_clearing as S

NY, YMIN, SEA = S.NY, S.Y_MIN, 62
SOFTN = set(S.PLANTS) | {'air', 'cave_air', 'void_air', 'water', 'vine', 'sugar_cane', 'lily_pad', 'kelp', 'kelp_plant', 'seagrass',
                          'tall_seagrass', 'bubble_column', 'cocoa', 'bamboo', 'cactus', 'snow'}
WATERS = ('water', 'kelp', 'kelp_plant', 'seagrass', 'tall_seagrass', 'bubble_column')


def sstep(a, b, x):
    t = np.clip((x - a)/(b - a), 0, 1); return t*t*(3 - 2*t)


class Frame:
    def __init__(s, world, x0, z0, x1, z1):
        assert x0 % 16 == 0 and z0 % 16 == 0 and x1 % 16 == 0 and z1 % 16 == 0
        s.world, s.X0, s.Z0, s.X1, s.Z1 = world, x0, z0, x1, z1
        s.sx, s.sz = x1 - x0, z1 - z0
        s.reg = os.path.join(world, 'dimensions/minecraft/overworld/region')
        s.idx = {'minecraft:air': 0}; s.chunks = {}; s.REGS = {}
        s.vol = np.zeros((s.sx, NY, s.sz), np.uint16)
        for cx in range(x0 >> 4, x1 >> 4):
            for cz in range(z0 >> 4, z1 >> 4):
                p = os.path.join(s.reg, f'r.{cx >> 5}.{cz >> 5}.mca')
                if p not in s.REGS: s.REGS[p] = mca.Region(p)
                raw = s.REGS[p].raw(cx, cz)
                if raw is None: sys.exit(f'нет чанка {cx},{cz}')
                nm, c = mca.parse(raw); s.chunks[(cx, cz)] = (p, nm, c)
                names, v = S._volume(c)
                mp = np.array([s.ID(k) for k in names])
                a, b = (cx - (x0 >> 4))*16, (cz - (z0 >> 4))*16
                s.vol[a:a + 16, :, b:b + 16] = mp[v]
        s.orig = s.vol.copy()
        s.XX, s.ZZ = np.meshgrid(np.arange(s.sx) + x0, np.arange(s.sz) + z0, indexing='ij')

    def ID(s, k): return s.idx.setdefault(k, len(s.idx))

    def names(s):
        a = [None]*len(s.idx)
        for k, i in s.idx.items(): a[i] = k
        return a

    def bn(s): return np.array([S.bname(k) for k in s.names()])

    def surface(s):
        bn = s.bn()
        soft = np.array([b in SOFTN or b.endswith('leaves') or b.endswith('_log') or b.endswith('_wood') for b in bn])
        isw = np.array([b in WATERS for b in bn])
        s.G = np.zeros((s.sx, s.sz), np.int32); s.WAT = np.zeros((s.sx, s.sz), bool); s.SURF = np.empty((s.sx, s.sz), object)
        J = np.arange(s.sz)
        for i in range(s.sx):
            sl = s.vol[i]
            top = NY - 1 - np.argmax((~soft[sl])[::-1, :], axis=0)
            s.G[i] = top; s.SURF[i] = bn[sl[top, J]]; s.WAT[i] = isw[sl[np.minimum(top + 1, NY - 1), J]]
        s.GY = s.G + YMIN
        return s

    def mask(s, img, value, ox=-1008, oz=-848):
        """Маска рамки из картинки масок (пиксель = мир − (ox, oz))."""
        px, pz = s.XX - ox, s.ZZ - oz
        ok = (px >= 0) & (px < img.shape[1]) & (pz >= 0) & (pz < img.shape[0])
        out = np.zeros((s.sx, s.sz), bool); out[ok] = img[pz[ok], px[ok]] == value
        return out

    def noise(s, sigma, rng):
        n = ndimage.gaussian_filter(rng.standard_normal((s.sx, s.sz)), sigma, mode='wrap')
        return (n - n.min())/(n.max() - n.min())

    def changed(s): return s.vol != s.orig

    def write(s):
        nl = s.names(); chg = s.changed(); touched = 0
        for (cx, cz), (p, nm, c) in s.chunks.items():
            a, b = (cx - (s.X0 >> 4))*16, (cz - (s.Z0 >> 4))*16
            sub = chg[a:a + 16, :, b:b + 16]
            if not sub.any(): continue
            have = {int.from_bytes(sec[b'Y'][1], 'big', signed=True) + 4 for sec in c[b'sections'][1][1] if sec.get(b'block_states') is not None}
            ch = {y >> 4 for y in np.nonzero(sub.any(axis=(0, 2)))[0]}
            if ch - have: sys.exit(f'нет секций {sorted(ch - have)} в чанке {cx},{cz}: записано {touched} чанков, вернуть копию')
            v = s.vol[a:a + 16, :, b:b + 16]
            S.write_sections(c, v, nl, ch, True)
            S.set_heightmaps(c, v, nl)
            s.REGS[p].put(cx, cz, mca.dump(nm, c)); touched += 1
        for r in s.REGS.values(): r.save()
        return touched

    def top_view(s, v=None, heights=True):
        """Картинка рамки сверху (z вниз, x вправо): упрощённые цвета, тень по высоте."""
        v = s.vol if v is None else v
        bn = s.bn()
        COL = {'grass_block': (95, 150, 60), 'sand': (222, 208, 160), 'red_sand': (190, 110, 50), 'dirt': (130, 100, 60), 'stone': (125, 125, 125),
               'andesite': (135, 135, 135), 'tuff': (110, 110, 100), 'gravel': (140, 135, 130), 'sandstone': (210, 195, 140),
               'snow_block': (245, 248, 252), 'water': (50, 90, 190), 'dirt_path': (160, 120, 70), 'calcite': (220, 220, 215),
               'coarse_dirt': (110, 85, 60), 'podzol': (90, 65, 35), 'moss_block': (90, 120, 45), 'packed_ice': (150, 180, 230),
               'snow': (240, 244, 250), 'ice': (170, 200, 240)}
        nonair = ~np.isin(bn, ['air', 'cave_air', 'void_air'])
        img = np.zeros((s.sz, s.sx, 3)); T = np.zeros((s.sx, s.sz))
        J = np.arange(s.sz)
        for i in range(s.sx):
            sl = v[i]; t = NY - 1 - np.argmax(nonair[sl][::-1, :], axis=0); T[i] = t
            for j, b in enumerate(bn[sl[t, J]]):
                img[j, i] = COL[b] if b in ('snow', 'ice') else ((30, 85, 35) if b == 'spruce_leaves' else (60, 125, 40) if b.endswith('leaves') else
                             (100, 75, 45) if b.endswith('_log') else COL.get(b, (95, 150, 60) if b in SOFTN else (150, 140, 130)))
        if heights:
            sh = np.clip(1 + 0.08*(T - np.roll(T, 1, 1) + T - np.roll(T, 1, 0)), 0.65, 1.3).T
            img *= sh[..., None]
        from PIL import Image
        return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
