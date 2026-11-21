"""Снимок мира строго сверху: верхний непрозрачный блок каждого столбца, цвет — средний цвет текстуры блока, тень по высоте, вода с глубиной.
Запуск: python3 top_map.py <папка с распакованными assets/*/textures/block из 26.2.jar и jar модов> <out.png>. Мир только читается."""
import sys, os, glob, numpy as np
from PIL import Image
sys.path.insert(0, '/home/bogdan/Документы/IdeaProjects/SmesharikiProject/WorldMod/tools')
import mca
W = os.path.expanduser('~/.minecraft/saves/Ромашковая Долина/dimensions/minecraft/overworld/region')
TEX = sys.argv[1]; OUT = sys.argv[2]
AIR = {'minecraft:air', 'minecraft:cave_air', 'minecraft:void_air', 'minecraft:barrier', 'minecraft:light', 'minecraft:structure_void'}
SKIP = ('grass', 'short_grass', 'tall_grass', 'fern', 'large_fern', 'torch', 'wall_torch', 'dandelion', 'poppy', 'rail', 'redstone_wire',
        'tripwire', 'string', 'dead_bush', 'seagrass', 'tall_seagrass', 'button', 'pressure_plate', 'sign', 'banner', 'cobweb', 'vine')
WATERISH = {'minecraft:water', 'minecraft:bubble_column', 'minecraft:kelp', 'minecraft:kelp_plant', 'minecraft:seagrass', 'minecraft:tall_seagrass'}
GRASS, FOLIAGE, WATER = np.array([0x91, 0xBD, 0x59]), np.array([0x77, 0xAB, 0x2F]), np.array([0x3F, 0x76, 0xE4])
cache = {}
def avg(path):
    try: a = np.array(Image.open(path).convert('RGBA')).astype(float)
    except Exception: return None
    a = a[:a.shape[1]]  # анимированные — первый кадр
    m = a[..., 3] > 20
    return a[..., :3][m].mean(0) if m.any() else None
def color(name):
    if name in cache: return cache[name]
    if '[' in name:  # крашеный блок BuildMod: цвет в свойствах r/g/b
        pr = dict(kv.split('=') for kv in name[:-1].split('[')[1].split(','))
        top = 7 if any(t in name for t in ('_stairs', '_slab', 'carpet')) else 15
        c = np.array([int(pr.get(k, top)) for k in 'rgb'], float) * 255 / top * 0.82
        cache[name] = c; return c
    ns, b = name.split(':')
    b = {'snow_block': 'snow', 'magma_block': 'magma', 'chest': 'oak_planks', 'trapped_chest': 'oak_planks'}.get(b, b)
    if b == 'dark_veil': cache[name] = np.array([20., 20., 24.]); return cache[name]
    if ns != 'minecraft' and not os.path.exists(f'{TEX}/assets/{ns}/textures/block/{b}.png'):
        mat = b.split('_')[-1]
        if os.path.exists(f'{TEX}/assets/minecraft/textures/block/{mat}.png'): ns, b = 'minecraft', mat
    d = f'{TEX}/assets/{ns}/textures/block/'
    base = b
    for suf in ('_stairs', '_slab', '_wall', '_fence_gate', '_fence', '_pressure_plate', '_button', '_trapdoor', '_door', '_carpet', '_pane'):
        if base.endswith(suf): base = base[:-len(suf)]; break
    cands = [b + '_top', b, base + '_top', base, base + 's', base + '_planks', base.replace('_wood', '_log') + '_top',
             base.replace('_hyphae', '_stem') + '_top', base.replace('waxed_', ''), base.replace('smooth_', '') + '_top', base.replace('smooth_', ''),
             base.replace('_bricks', '_brick'), base + '_block']
    if base.endswith('_brick'): cands.append(base + 's')
    if b == 'water' or b == 'bubble_column': cands = ['water_still']
    if b == 'lava': cands = ['lava_still']
    c = None
    for k in cands:
        c = avg(d + k + '.png')
        if c is not None: break
    if c is None:
        hits = sorted(glob.glob(d + base + '*.png'))
        if hits: c = avg(hits[0])
    if c is None:
        h = abs(hash(name)); c = np.array([90 + h % 120, 90 + h // 7 % 120, 90 + h // 49 % 120], float); print('нет текстуры', name)
    g = c.mean()
    if b in ('grass_block',) or b.endswith('_leaves') and b not in ('cherry_leaves', 'azalea_leaves', 'flowering_azalea_leaves') or b in ('lily_pad',):
        c = g * (FOLIAGE if b.endswith('_leaves') or b == 'lily_pad' else GRASS) / 255 * 1.0
        if b == 'spruce_leaves': c = g * np.array([0x61, 0x99, 0x61]) / 255
        if b == 'birch_leaves': c = g * np.array([0x80, 0xA7, 0x55]) / 255
    if b in ('water', 'bubble_column'): c = g * WATER / 255
    cache[name] = c; return c

files = glob.glob(W + '/r.*.mca')
rs = [tuple(map(int, os.path.basename(f).split('.')[1:3])) for f in files]
x0, z0 = min(r[0] for r in rs) * 512, min(r[1] for r in rs) * 512
x1, z1 = (max(r[0] for r in rs) + 1) * 512, (max(r[1] for r in rs) + 1) * 512
NX, NZ = x1 - x0, z1 - z0
top = np.full((NZ, NX), -999, np.int32); col = np.zeros((NZ, NX, 3)); depth = np.zeros((NZ, NX), np.int32); have = np.zeros((NZ, NX), bool)
for f, (rx, rz) in zip(files, rs):
    reg = mca.Region(f)
    for cz in range(32):
        for cx in range(32):
            raw = reg.raw(cx, cz)
            if raw is None: continue
            _, ch = mca.parse(raw)
            secs = ch.get(b'sections')
            if not secs: continue
            if ch.get(b'Status', (8, b'minecraft:full'))[1] not in (b'minecraft:full', b'full'): continue
            stack = []
            for s in secs[1][1]:
                bs = s.get(b'block_states')
                if not bs: continue
                pal = [mca.key(p) for p in bs[1][b'palette'][1][1]]
                if pal == ['minecraft:air']: continue
                ids = mca.unpack(bs[1][b'data'][1], len(pal)) if b'data' in bs[1] else np.zeros(4096, np.int32)
                stack.append((int.from_bytes(s[b'Y'][1], 'big', signed=True), pal, ids.reshape(16, 16, 16)))
            stack.sort(key=lambda t: -t[0])
            gx, gz = rx * 512 + cx * 16 - x0, rz * 512 + cz * 16 - z0
            done = np.zeros((16, 16), bool); wd = np.zeros((16, 16), np.int32); wtop = np.full((16, 16), -999)
            for Y, pal, ids in stack:
                pal = [p if p.startswith('buildmod:painted') else p.split('[')[0] for p in pal]
                kind = np.array([0 if p in AIR or p.split(':')[1] in SKIP or p.split(':')[1].endswith(('_sign', '_banner', '_button', 'pressure_plate', '_torch', '_flower'))
                                 else 1 if p in WATERISH or p == 'minecraft:water' else 2 for p in pal])
                for y in range(15, -1, -1):
                    k = kind[ids[y]]
                    w = (k == 1) & ~done
                    wd[w] += 1; wtop[w & (wtop == -999)] = Y * 16 + y
                    hit = (k == 2) & ~done
                    if hit.any():
                        zz, xx = np.nonzero(hit)
                        for z, x in zip(zz, xx):
                            col[gz + z, gx + x] = color(pal[ids[y, z, x]])
                        top[gz:gz+16, gx:gx+16][hit] = Y * 16 + y
                        done |= hit
                    if done.all(): break
                if done.all(): break
            depth[gz:gz+16, gx:gx+16] = wd
            have[gz:gz+16, gx:gx+16] = True
            wt = wtop > -999
            top[gz:gz+16, gx:gx+16][wt] = wtop[wt]
    print('регион', rx, rz, flush=True)

# тень по высоте (свет с северо-запада), вода — смесь дна и цвета воды по глубине
h = top.astype(float)
sh = np.clip(1 + 0.12 * ((h - np.roll(h, 1, 0)) + (h - np.roll(h, 1, 1))) / 2, 0.7, 1.25)
img = col * sh[..., None]
water = color('minecraft:water')
a = np.clip(0.45 + 0.08 * depth, 0, 0.92)[..., None] * (depth > 0)[..., None]
img = img * (1 - a) + water * a
img[~have] = (24, 26, 32)
Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(OUT)
ys, xs = np.nonzero(have)
print('мир X', x0, '..', x1, 'Z', z0, '..', z1, 'пиксель = блок; заполнено X', xs.min()+x0, xs.max()+x0, 'Z', ys.min()+z0, ys.max()+z0)
