#!/usr/bin/env python3
"""Левый (по течению, северный) берег реки в сохранённом мире: на него можно запрыгнуть с бревна.

Запуск: river_bank.py <папка мира> [--write <папка копии>]. Без --write мир не меняется, только отчёт. Игра должна быть закрыта.
Верх бревна на 1,59 выше верхней клетки воды, прыжок 1,25: верх блока берега не выше воды + 2, то есть блок на воде + 1.
Крайний ряд — на воде + 1, пятнами на уровне воды, дальше лестница с шагом 1 до прежнего рельефа. Берег только понижается.
Столбец с обрывом выше CLIFF не трогается, стволы продлеваются вниз, трава и бамбук на понижённых столбцах снимаются.
"""
import math, os, random, re, shutil, sys
from collections import deque
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca

X_FROM, X_TO = 371, 600      # от падения до начала каскада к морю
ROWS, CLIFF = 8, 7           # сколько рядов лестницы; столбцы выше воды + CLIFF не срезаются
SEED = 26110202
PATCH, GAP = (6, 16), (12, 30)  # длина пятен на уровне воды и промежутков, блоков вдоль реки
PLANTS = ('short_grass', 'tall_grass', 'fern', 'large_fern', 'bamboo', 'snow', 'azure_bluet', 'oxeye_daisy', 'dead_bush', 'poppy', 'dandelion',
          'bush', 'firefly_bush', 'leaf_litter', 'wildflowers', 'short_dry_grass', 'tall_dry_grass', 'sapling', 'melon', 'bamboo_sapling', 'jungle_sapling')
GROUND = ('stone', 'dirt', 'grass_block', 'gravel', 'andesite', 'sand', 'tuff', 'sandstone', 'calcite', 'coarse_dirt', 'podzol', 'mud', 'clay', 'moss_block', 'snow_block')
KEEP = ('air', 'vine', 'cocoa', 'water')  # над прежней землёй не мешают
Y_MIN = -64

world = sys.argv[1]
backup = sys.argv[sys.argv.index('--write') + 1] if '--write' in sys.argv else None
REG = os.path.join(world, 'dimensions/minecraft/overworld/region/')
import log_routes_dat
# маршрут брёвен из данных мира (пишет log_route.py)
nodes = np.array(log_routes_dat.read(world)['река'], float)
arc = np.r_[0, np.cumsum(np.hypot(*np.diff(nodes[:, [0, 2]], axis=0).T))]
NODE_FROM, NODE_TO = int(np.argmax(nodes[:, 0] >= X_FROM)), int(np.argmax(nodes[:, 0] >= X_TO))

X0, X1, Z0, Z1 = 352, 624, -240, -112
cx0, cx1, cz0, cz1 = X0 >> 4, (X1 - 1) >> 4, Z0 >> 4, (Z1 - 1) >> 4
ox, oz = cx0*16, cz0*16
sx, sz, NY = (cx1 - cx0 + 1)*16, (cz1 - cz0 + 1)*16, 384
vol = np.zeros((sx, NY, sz), np.uint16); names = {'minecraft:air': 0}; lst = ['minecraft:air']; regs = {}; chunks = {}
def gid(k):
    if k not in names: names[k] = len(lst); lst.append(k)
    return names[k]
for cx in range(cx0, cx1 + 1):
    for cz in range(cz0, cz1 + 1):
        p = REG + f'r.{cx >> 5}.{cz >> 5}.mca'
        regs.setdefault(p, mca.Region(p)); b = regs[p].raw(cx, cz)
        if b is None: continue
        nm, c = mca.parse(b); chunks[(cx, cz)] = (p, nm, c)
        for sec in c[b'sections'][1][1]:
            y = int.from_bytes(sec[b'Y'][1], 'big', signed=True); bs = sec.get(b'block_states')
            if bs is None or not -4 <= y < 20: continue
            pal = np.array([gid(mca.key(q)) for q in bs[1][b'palette'][1][1]])
            ids = pal[mca.unpack(bs[1][b'data'][1], len(pal))].reshape(16, 16, 16) if b'data' in bs[1] else np.full((16, 16, 16), pal[0])
            vol[(cx - cx0)*16:(cx - cx0 + 1)*16, (y + 4)*16:(y + 5)*16, (cz - cz0)*16:(cz - cz0 + 1)*16] = ids.transpose(2, 0, 1)
orig = vol.copy()
def bn(i): return lst[i].split('[')[0].replace('minecraft:', '')
def ids_of(names_): return [i for i in range(len(lst)) if bn(i) in names_]

isg = np.isin(vol, ids_of(GROUND)); G = NY - 1 - np.argmax(isg[:, ::-1, :], axis=1) + Y_MIN  # верхний блок земли
isw = np.isin(vol, ids_of(('water',))); anyw = isw.any(1)
W = np.where(anyw, NY - 1 - np.argmax(isw[:, ::-1, :], axis=1) + Y_MIN, -1000)  # верхняя клетка воды
hasw = anyw & (W > G)

# русло: связная вода от узлов маршрута; прыжки уровня ±2 внутри — пороги
river = np.zeros((sx, sz), bool); q = deque()
for n in nodes[NODE_FROM:NODE_TO + 1]:
    i, j = int(math.floor(n[0])) - ox, int(math.floor(n[2])) - oz
    if hasw[i, j] and not river[i, j]: river[i, j] = True; q.append((i, j))
while q:
    i, j = q.popleft()
    for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        u, v = i + a, j + b
        if 0 <= u < sx and 0 <= v < sz and hasw[u, v] and not river[u, v] and abs(W[u, v] - W[i, j]) <= 2 and W[u, v] >= 114:
            river[u, v] = True; q.append((u, v))

def nearest(x, z):
    """Ближайшая точка маршрута: путь вдоль реки, номер узла и «левизна» точки."""
    best = None
    for k in range(len(nodes) - 1):
        ax, az, bx, bz = nodes[k, 0], nodes[k, 2], nodes[k + 1, 0], nodes[k + 1, 2]
        dx, dz = bx - ax, bz - az; t = np.clip(((x - ax)*dx + (z - az)*dz)/(dx*dx + dz*dz), 0, 1)
        px, pz = ax + dx*t, az + dz*t; d = (x - px)**2 + (z - pz)**2
        if best is None or d < best[0]:
            # лево при взгляде по течению: (dz, −dx), ось z смотрит на юг
            best = (d, arc[k] + t*math.hypot(dx, dz), k, (x - px)*dz - (z - pz)*dx)
    return best[1], best[2], best[3] > 0

# пятна на уровне воды вдоль реки
rnd = random.Random(SEED); patches = []; s = arc[NODE_FROM] + rnd.uniform(*GAP)
while s < arc[NODE_TO]:
    l = rnd.uniform(*PATCH); patches.append((s, s + l)); s += l + rnd.uniform(*GAP)
def base(s): return 0 if any(a <= s < b for a, b in patches) else 1

# ряды: от кромки левого берега шагами по четырём соседям; у каждого столбца наименьшая цель
target = np.full((sx, sz), 10**6); row = np.zeros((sx, sz), int); ref = np.full((sx, sz), -1000); q = deque()
for i in range(sx):
    for j in range(sz):
        if river[i, j] or hasw[i, j]: continue
        ws = [W[i + a, j + b] for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= i + a < sx and 0 <= j + b < sz and river[i + a, j + b]]
        if not ws: continue
        s, k, left = nearest(i + ox + .5, j + oz + .5)
        if not left or not NODE_FROM <= k < NODE_TO: continue
        w = min(ws)
        target[i, j] = w + base(s); row[i, j] = 1; ref[i, j] = w; q.append((i, j))
while q:
    i, j = q.popleft()
    if row[i, j] >= ROWS: continue
    for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        u, v = i + a, j + b
        if not (0 <= u < sx and 0 <= v < sz) or river[u, v] or hasw[u, v] or row[u, v]: continue
        target[u, v] = target[i, j] + 1; row[u, v] = row[i, j] + 1; ref[u, v] = ref[i, j]; q.append((u, v))

new = G.copy(); skipped = {}; free = np.zeros((sx, sz), bool)
for i, j in zip(*np.nonzero(row)):
    if G[i, j] - ref[i, j] > CLIFF: continue          # скала остаётся
    # над землёй только известное: чужую постройку не срезаем
    yb = G[i, j] - Y_MIN; strange = {bn(c) for c in vol[i, yb + 1:yb + 4, j]} - set(PLANTS) - set(KEEP)
    strange = {b for b in strange if not b.endswith(('_leaves', '_log'))}
    if strange:
        for b in strange: skipped[b] = skipped.get(b, 0) + 1
        continue
    free[i, j] = True
    new[i, j] = min(target[i, j], G[i, j])
# шаг между соседями не выше 1 (стык пятна и рядов, прежние уступы), и столбец не ниже соседней воды: иначе вода вытечет
for _ in range(2*ROWS):
    moved = 0
    for i, j in zip(*np.nonzero(free)):
        limit = new[i, j]; floor = -1000
        for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            u, v = i + a, j + b
            if not (0 <= u < sx and 0 <= v < sz): continue
            if hasw[u, v]: floor = max(floor, W[u, v])
            elif row[u, v]: limit = min(limit, new[u, v] + 1)
        limit = min(G[i, j], max(limit, floor))
        if limit != new[i, j]: new[i, j] = limit; moved += 1
    if not moved: break

# запись в объём
plants = ids_of(PLANTS)
trunks = 0
for i, j in zip(*np.nonzero(new < G)):
    g, n = G[i, j] - Y_MIN, new[i, j] - Y_MIN
    top = vol[i, g, j]; above = vol[i, g + 1, j]
    vol[i, n + 1:g + 1, j] = above if bn(above).endswith('_log') else 0
    trunks += bn(above).endswith('_log')
    vol[i, n, j] = top
    y = g + 1
    while y < NY and vol[i, y, j] in plants: vol[i, y, j] = 0; y += 1
changed = vol != orig

# отчёт: по длине берега, можно ли запрыгнуть и уйти вглубь по шагам не выше 1
first = np.argwhere(row == 1); rel = new - ref; was = G - ref
print('рядов 1:', len(first), 'понижено столбцов:', int((new < G).sum()), 'стволов продлено:', trunks, 'блоков изменено:', int(changed.sum()))
print('верх крайнего ряда над водой, было:', dict(zip(*[a.tolist() for a in np.unique(np.clip(was[row == 1], -3, 9), return_counts=True)])))
print('стало:', dict(zip(*[a.tolist() for a in np.unique(np.clip(rel[row == 1], -3, 9), return_counts=True)])))
print('пятен на уровне воды:', len(patches), 'сумма длин:', round(sum(b - a for a, b in patches)), 'из', round(arc[NODE_TO] - arc[NODE_FROM]))
if skipped: print('столбцы не тронуты из-за блоков:', skipped)
bad = 0
for i, j in zip(*np.nonzero(row)):
    for a, b in ((1, 0), (0, 1)):
        u, v = i + a, j + b
        if u < sx and v < sz and (free[i, j] or free[u, v]) and row[u, v] and not (hasw[u, v] or hasw[i, j]) and abs(int(new[i, j]) - int(new[u, v])) > 1:
            bad += 1; print('уступ', i + ox, j + oz, new[i, j] - ref[i, j], '->', u + ox, v + oz, new[u, v] - ref[i, j])
print('уступов выше 1 внутри рядов:', bad)
per = [[] for _ in range(len(nodes))]
for i, j in first:
    per[nearest(i + ox + .5, j + oz + .5)[1]].append(rel[i, j])
print('узел x z: мин/макс крайнего ряда')
for k in range(NODE_FROM, NODE_TO, 8):
    print(k, round(nodes[k, 0]), round(nodes[k, 2]), (min(per[k]), max(per[k])) if per[k] else '-')
if not backup:
    print('без --write мир не изменён'); sys.exit()

# ---------- запись
os.makedirs(backup, exist_ok=False)
for p in {chunks[k][0] for k in chunks}: shutil.copy2(p, backup)
leaf = ids_of(('jungle_leaves', 'oak_leaves', 'spruce_leaves'))
noblock = ids_of(('air', 'vine', 'tall_grass', 'short_grass', 'fern', 'large_fern', 'cocoa', 'snow', 'azure_bluet', 'oxeye_daisy', 'dead_bush'))
fluid = ids_of(('water',))
TYPES = {b'WORLD_SURFACE': 0, b'MOTION_BLOCKING': 1, b'MOTION_BLOCKING_NO_LEAVES': 2, b'OCEAN_FLOOR': 3}
def hm(col):
    r = []; n = col != 0; mb = ~np.isin(col, noblock); water = np.isin(col, fluid)
    for m in (n, mb | water, (mb & ~np.isin(col, leaf)) | water, mb & ~water):
        idx = np.nonzero(m)[0]; r.append(int(idx[-1]) + 1 if len(idx) else 0)
    return r
touched = 0
for (cx, cz), (p, nm, c) in chunks.items():
    a, b = (cx - cx0)*16, (cz - cz0)*16
    if not changed[a:a + 16, :, b:b + 16].any(): continue
    touched += 1
    for sec in c[b'sections'][1][1]:
        y = int.from_bytes(sec[b'Y'][1], 'big', signed=True); bs = sec.get(b'block_states')
        sec.pop(b'SkyLight', None); sec.pop(b'BlockLight', None)
        if bs is None or not -4 <= y < 20: continue
        oy = (y + 4)*16
        if not changed[a:a + 16, oy:oy + 16, b:b + 16].any(): continue
        arr = vol[a:a + 16, oy:oy + 16, b:b + 16].transpose(1, 2, 0).reshape(4096)
        u, inv = np.unique(arr, return_inverse=True)
        bs[1][b'palette'] = (9, (10, [mca.state(lst[i]) for i in u]))
        if len(u) == 1: bs[1].pop(b'data', None)
        else: bs[1][b'data'] = (12, mca.pack(inv, len(u)))
    c[b'isLightOn'] = (1, b'\0')
    hs = np.zeros((4, 256), int)
    for lx in range(16):
        for lz in range(16): hs[:, lz*16 + lx] = hm(vol[a + lx, :, b + lz])
    for nmk, t in TYPES.items():
        w = np.zeros(math.ceil(256/7), np.uint64)
        for q_ in range(256): w[q_//7] |= np.uint64(int(hs[t, q_]) << ((q_ % 7)*9))
        c[b'Heightmaps'][1][nmk] = (12, w.astype('>u8').tobytes())
    regs[p].put(cx, cz, mca.dump(nm, c))
for r in regs.values(): r.save()
print('записано чанков:', touched, 'копия:', backup)
