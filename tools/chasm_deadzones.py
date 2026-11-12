#!/usr/bin/env python3
"""Мёртвые зоны (gameplaymod:deadzone) на дне расщелины в «Ромашковой Долине»: аккуратная цепочка ящиков вдоль щели на дне.

Запуск: chasm_deadzones.py <мир> [--write <новая папка копии>], игра закрыта. Без --write только печатает план. С --write все мёртвые зоны
в рамке FRAME убираются и ставятся заново (прежние файлы сущностей — в папку копии), поэтому повторный запуск даёт то же.

Зона берёт только непустые блоки ящика, поэтому ящик подбирается так, чтобы из открытых сверху блоков в него попадали только дно (гравий
и мох щели, y дна по x) и стены не выше дна + M_HI, а воды не было вовсе (у восточного конца расщелина выходит к морю). Каждый столбец дна
накрыт хотя бы до дна + M_LO. Ящики идут по x подряд, без перекрытий, не длиннее 64 (половина стороны у маркера до 32).
"""
import math, os, shutil, struct, sys, uuid, gzip
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca

FRAME = (304, 687, -400, -193)      # рамка чтения и снятия прежних зон (границы чанков)
M_LO, M_HI, WALL, DOWN = 5, 9, 3, 3  # над дном: накрыть не меньше / открытых блоков не выше; стены в стороны от щели; ниже дна
SKIP = {'air', 'cave_air', 'void_air'}
SOFT = {'vine', 'short_grass', 'tall_grass', 'fern', 'large_fern', 'water'}
NOT_GROUND = SOFT | {'jungle_leaves', 'oak_leaves', 'spruce_leaves', 'gameplaymod:dark_veil', 'jungle_log', 'oak_log', 'spruce_log', 'cocoa', 'bamboo'}
ZONE = {}   # зона только убивает: полей стрельбы больше нет

world = sys.argv[1]
backup = sys.argv[sys.argv.index('--write') + 1] if '--write' in sys.argv else None
D = os.path.join(world, 'dimensions/minecraft/overworld/'); REG = D + 'region/'; ENT = D + 'entities/'
X0, X1, Z0, Z1 = FRAME; Y0, NY = -64, 256
sx, sz = X1 - X0 + 1, Z1 - Z0 + 1
vol = np.zeros((sx, NY, sz), np.uint16); names = {'minecraft:air': 0}; lst = ['minecraft:air']
def gid(k):
    if k not in names: names[k] = len(lst); lst.append(k)
    return names[k]
regs = {}
for cx in range(X0 >> 4, (X1 >> 4) + 1):
    for cz in range(Z0 >> 4, (Z1 >> 4) + 1):
        p = REG + f'r.{cx >> 5}.{cz >> 5}.mca'
        regs.setdefault(p, mca.Region(p)); b = regs[p].raw(cx, cz)
        if b is None: continue
        for sec in mca.parse(b)[1][b'sections'][1][1]:
            y = int.from_bytes(sec[b'Y'][1], 'big', signed=True); bs = sec.get(b'block_states')
            if bs is None or not -4 <= y < 12: continue
            pal = np.array([gid(mca.key(q)) for q in bs[1][b'palette'][1][1]])
            ids = pal[mca.unpack(bs[1][b'data'][1], len(pal))].reshape(16, 16, 16) if b'data' in bs[1] else np.full((16, 16, 16), pal[0])
            vol[(cx - (X0 >> 4))*16:(cx - (X0 >> 4) + 1)*16, (y + 4)*16:(y + 5)*16, (cz - (Z0 >> 4))*16:(cz - (Z0 >> 4) + 1)*16] = ids.transpose(2, 0, 1)
bn = np.array([l.split('[')[0].replace('minecraft:', '') for l in lst]); B = bn[vol]
nonair = ~np.isin(B, list(SKIP))
exp = nonair.copy(); exp[:, :-1, :] &= ~nonair[:, 1:, :] | np.isin(B[:, 1:, :], list(SOFT))     # открытые сверху: на них можно встать
water = B == 'water'
ground = nonair & ~np.isin(B, list(NOT_GROUND - {'water'}))     # под водой дна нет: морское дно не берётся
top = NY - 1 - np.argmax(ground[:, ::-1, :], axis=1); tb = np.take_along_axis(B, top[:, None, :], 1)[:, 0, :]; top += Y0
# дно: гравий и мох щели (chasm_world.py кладёт их там, где верх не выше дна + 3); пропуски по x (ствол на дне) — по соседям
fl = np.isin(tb, ['gravel', 'moss_block']) & (top < 75)
fy, fz = {}, {}
for i in np.where(fl.any(1))[0]:
    js = np.where(fl[i])[0]; fy[i] = int(top[i, js].min()); fz[i] = (int(js.min()), int(js.max()))
I0, I1 = min(fy), max(fy)
for i in range(I0, I1 + 1):
    if i not in fy:
        a = max(k for k in fy if k < i); b = min(k for k in fy if k > i); fy[i] = min(fy[a], fy[b]); fz[i] = (min(fz[a][0], fz[b][0]), max(fz[a][1], fz[b][1]))

def box(a, b):
    r = range(a, b + 1)
    return a, b, min(fy[i] for i in r) - DOWN, max(fy[i] for i in r) + M_LO, min(fz[i][0] for i in r) - WALL, max(fz[i][1] for i in r) + WALL
def extra(bx):
    """Открытые блоки ящика вне дна: выше дна + M_HI, дальше 4 от щели, вода."""
    a, b, ylo, yhi, zlo, zhi = bx; n = 0
    for i in range(a, b + 1):
        sub = exp[i, ylo - Y0:yhi - Y0 + 1, zlo:zhi + 1] | water[i, ylo - Y0:yhi - Y0 + 1, zlo:zhi + 1]
        ys, js = np.where(sub); ys = ys + ylo; js = js + zlo
        n += int(((ys > fy[i] + M_HI) | (js < fz[i][0] - 4) | (js > fz[i][1] + 4) | water[i, ys - Y0, js]).sum())
    return n
boxes = []; a = I0
while a <= I1:
    b = a
    while b + 1 <= I1 and b + 1 - a < 64 and not extra(box(a, b + 1)): b += 1
    if extra(box(a, b)) or (b - a < 3 and b + 4 > I1): break     # к морю: последние столбцы порога (y 66 у воды) без зоны, обрубков не ставим
    boxes.append(box(a, b)); a = b + 1
uncovered = I1 - a + 1
zones = []
for a, b, ylo, yhi, zlo, zhi in boxes:
    x0, x1, z0, z1 = a + X0, b + X0, zlo + Z0, zhi + Z0
    c = ((x0 + x1 + 1)/2, (ylo + yhi + 1)/2, (z0 + z1 + 1)/2); h = ((x1 - x0 + 1)/2, (yhi - ylo + 1)/2, (z1 - z0 + 1)/2)
    zones.append((c, h))
    print(f'x {x0}..{x1}  y {ylo}..{yhi}  z {z0}..{z1}  (дно y {min(fy[i] for i in range(a, b + 1))}..{max(fy[i] for i in range(a, b + 1))})')
print(f'зон {len(zones)}, дно x {I0 + X0}..{I1 + X0}' + (f', у моря не накрыто столбцов: {uncovered}' if uncovered else ''))
if not backup:
    print('без --write мир не изменён'); sys.exit()

os.makedirs(os.path.join(backup, 'entities'), exist_ok=True)
dbl = lambda v: struct.pack('>d', float(v)); flt = lambda v: (5, struct.pack('>f', float(v)))
dv = mca.parse(gzip.open(os.path.join(world, 'level.dat')).read())[1][b'Data'][1][b'DataVersion']
touched = {}
def chunk_ents(cx, cz):
    if (cx, cz) not in touched:
        p = ENT + f'r.{cx >> 5}.{cz >> 5}.mca'
        if os.path.exists(p) and not os.path.exists(os.path.join(backup, 'entities', os.path.basename(p))): shutil.copy2(p, os.path.join(backup, 'entities'))
        b = mca.Region(p).raw(cx, cz) if os.path.exists(p) and os.path.getsize(p) >= 8192 else None
        if b is None: nm, c = b'', {b'Position': (11, struct.pack('>ii', cx, cz)), b'DataVersion': dv, b'Entities': (9, (10, []))}
        else: nm, c = mca.parse(b)
        if c[b'Entities'][1][0] == 0: c[b'Entities'] = (9, (10, c[b'Entities'][1][1]))
        touched[(cx, cz)] = (p, nm, c)
    return touched[(cx, cz)][2]
removed = 0
for cx in range(X0 >> 4, (X1 >> 4) + 1):
    for cz in range(Z0 >> 4, (Z1 >> 4) + 1):
        p = ENT + f'r.{cx >> 5}.{cz >> 5}.mca'
        if not os.path.exists(p) or os.path.getsize(p) < 8192 or mca.Region(p).raw(cx, cz) is None: continue
        c = chunk_ents(cx, cz); before = len(c[b'Entities'][1][1])
        c[b'Entities'] = (9, (10, [e for e in c[b'Entities'][1][1] if e[b'id'][1] != b'gameplaymod:deadzone']))
        removed += before - len(c[b'Entities'][1][1])
for (x, y, z), (hx, hy, hz) in zones:
    c = chunk_ents(math.floor(x) >> 4, math.floor(z) >> 4)
    e = {b'id': (8, b'gameplaymod:deadzone'), b'Pos': (9, (6, [dbl(x), dbl(y), dbl(z)])), b'Motion': (9, (6, [dbl(0)]*3)),
         b'Rotation': (9, (5, [struct.pack('>f', 0.0)]*2)), b'UUID': (11, uuid.uuid4().bytes), b'Group': (8, b''),
         b'HalfX': flt(hx), b'HalfY': flt(hy), b'HalfZ': flt(hz)}
    e.update({k: flt(v) for k, v in ZONE.items()})
    c[b'Entities'][1][1].append(e)
regions = {}
for (cx, cz), (p, nm, c) in touched.items():
    if p not in regions:
        if not os.path.exists(p) or os.path.getsize(p) < 8192:
            with open(p, 'wb') as f: f.write(b'\0'*8192)
        regions[p] = mca.Region(p)
    regions[p].put(cx, cz, mca.dump(nm, c))
for r in regions.values(): r.save()
print(f'записано зон {len(zones)}, прежних убрано {removed}; копия {backup}')
