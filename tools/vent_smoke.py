#!/usr/bin/env python3
"""Дым по всему жерлу вулкана на выходе из лавовой пещеры: столбы gameplaymod:smoke (GameplayMod, SmokeEntity) друг над другом вдоль шахты.

Запуск: vent_smoke.py <мир> [--write <новая папка копии>], игра закрыта. Без --write только печатает план. С --write все сущности
gameplaymod:smoke в рамке FRAME убираются и ставятся заново (прежние файлы сущностей — в папку копии), поэтому повторный запуск даёт то же.

Шахта берётся из мира, а не из lava_waves.py: от маркера выброса (gameplaymod:launcher) вверх по слоям воздух, связанный с прошлым слоем,
заливкой в пределах REACH от прошлой середины; его середина и радиус равной площади — ось и ширина шахты. Шахта кончается, где слой
перестаёт быть замкнутым (заливка упёрлась в REACH: вышли к небу). Слои собираются в столбы по SEG блоков: середина — средняя по
слоям, радиус — RADIUS_K от медианы слоёв (на пережимах дым немного уходит в стену). Ниша под шахтой открыта к озеру: в ней ось
ведётся от выброса к низу шахты, ширина — как у низа шахты. Число клубов в секунду — от размера столба.
"""
import math, os, shutil, struct, sys, uuid, gzip
from collections import deque
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca

FRAME = (64, 239, -400, -177)       # рамка чтения и снятия прежнего дыма (границы чанков)
Y_LO, Y_HI = -48, 208                 # границы секций
SEG = 6                             # высота столба дыма
REACH = 16                          # заливка слоя не дальше этого от середины прошлого слоя
RADIUS_K = 0.85
LIFE = 6.5                          # средняя жизнь клуба, с (client/SmokeParticle: 100–160 тиков)

def rate_for(r, h):
    """Клубов в секунду: клуб размером около радиуса, на столб держится около 5·h/r + 4 клубов (полупрозрачные, в два слоя)."""
    return round((5.0*h/max(1.0, r) + 4)/LIFE, 2)

world = sys.argv[1]
backup = sys.argv[sys.argv.index('--write') + 1] if '--write' in sys.argv else None
D = os.path.join(world, 'dimensions/minecraft/overworld/'); REG = D + 'region/'; ENT = D + 'entities/'
X0, X1, Z0, Z1 = FRAME
sx, sz, ny = X1 - X0 + 1, Z1 - Z0 + 1, Y_HI - Y_LO
solid = np.zeros((sx, ny, sz), bool)
regs = {}
for cx in range(X0 >> 4, (X1 >> 4) + 1):
    for cz in range(Z0 >> 4, (Z1 >> 4) + 1):
        p = REG + f'r.{cx >> 5}.{cz >> 5}.mca'
        if not os.path.exists(p): continue
        regs.setdefault(p, mca.Region(p)); b = regs[p].raw(cx, cz)
        if b is None: continue
        for sec in mca.parse(b)[1][b'sections'][1][1]:
            y = int.from_bytes(sec[b'Y'][1], 'big', signed=True); bs = sec.get(b'block_states')
            if bs is None or not Y_LO <= y*16 <= Y_HI - 16: continue
            pal = np.array(['air' not in mca.key(q) for q in bs[1][b'palette'][1][1]])
            ids = pal[mca.unpack(bs[1][b'data'][1], len(pal))].reshape(16, 16, 16) if b'data' in bs[1] else np.full((16, 16, 16), pal[0])
            solid[(cx - (X0 >> 4))*16:(cx - (X0 >> 4) + 1)*16, y*16 - Y_LO:y*16 - Y_LO + 16, (cz - (Z0 >> 4))*16:(cz - (Z0 >> 4) + 1)*16] = ids.transpose(2, 0, 1)

def entities():
    for cx in range(X0 >> 4, (X1 >> 4) + 1):
        for cz in range(Z0 >> 4, (Z1 >> 4) + 1):
            p = ENT + f'r.{cx >> 5}.{cz >> 5}.mca'
            if not os.path.exists(p) or os.path.getsize(p) < 8192: continue
            b = mca.Region(p).raw(cx, cz)
            if b is None: continue
            for e in mca.parse(b)[1][b'Entities'][1][1]: yield (cx, cz), e
d_ = lambda v: struct.unpack('>d', v)[0]
launchers = [tuple(d_(v) for v in e[b'Pos'][1][1]) for _, e in entities() if e[b'id'][1] == b'gameplaymod:launcher']
assert len(launchers) == 1, f'маркеров выброса в рамке {len(launchers)}, нужен один'
LX, LY, LZ = launchers[0]
print(f'выброс ({LX}, {LY}, {LZ})')

def slice_air(y, cx, cz):
    """Воздух слоя y, связанный с клеткой (cx, cz) (или ближайшей воздушной), в пределах REACH. -> клетки, упёрлась ли в REACH."""
    k = y - Y_LO; ci, cj = int(math.floor(cx)) - X0, int(math.floor(cz)) - Z0
    if solid[ci, k, cj]:
        best = None
        for a in range(-4, 5):
            for b in range(-4, 5):
                if not solid[ci + a, k, cj + b] and (best is None or a*a + b*b < best[0]): best = (a*a + b*b, ci + a, cj + b)
        if best is None: return [], False
        ci, cj = best[1], best[2]
    seen = {(ci, cj)}; q = deque([(ci, cj)]); open_ = False
    while q:
        i, j = q.popleft()
        for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            u, v = i + a, j + b
            if (u, v) in seen or solid[u, k, v]: continue
            if math.hypot(u + .5 - (cx - X0), v + .5 - (cz - Z0)) > REACH: open_ = True; continue
            seen.add((u, v)); q.append((u, v))
    return [(i + X0, j + Z0) for i, j in seen], open_

layers = []                         # (y, середина x, z, радиус)
cx, cz = LX, LZ; niche = []
for y in range(int(math.floor(LY)), Y_HI):
    cells, open_ = slice_air(y, cx, cz)
    if open_ and not layers and len(niche) < 40: niche.append(y); continue      # ниша под шахтой открыта к озеру
    if open_ or not cells: break
    xs = np.array([c[0] + .5 for c in cells]); zs = np.array([c[1] + .5 for c in cells])
    r = math.sqrt(len(cells)/math.pi)
    layers.append((y, float(xs.mean()), float(zs.mean()), r)); cx, cz = xs.mean(), zs.mean()
if layers and niche:                # в нише: от выброса к низу шахты, шириной низа шахты
    y1, x1, z1, r1 = layers[0]
    layers = [(y, LX + (x1 - LX)*(y - niche[0])/(y1 - niche[0]), LZ + (z1 - LZ)*(y - niche[0])/(y1 - niche[0]), r1) for y in niche] + layers
assert layers, 'шахта не найдена'
y_top = layers[-1][0]
print(f'шахта: y {layers[0][0]}..{y_top}, {len(layers)} слоёв; устье ({layers[-1][1]:.1f}, {layers[-1][2]:.1f}), радиус внизу {layers[0][3]:.1f}, у устья {layers[-1][3]:.1f}')
smokes = []
for s in range(0, len(layers), SEG):
    part = layers[s:s + SEG]; h = len(part)
    x = sum(p[1] for p in part)/h; z = sum(p[2] for p in part)/h
    r = round(RADIUS_K*float(np.median([p[3] for p in part])), 1)
    smokes.append((round(x, 2), float(part[0][0]), round(z, 2), r, float(h), rate_for(r, h)))
for sm in smokes: print(f'  дым ({sm[0]}, {sm[1]:.0f}, {sm[2]}) радиус {sm[3]}, высота {sm[4]:.0f}, клубов/с {sm[5]}')
print(f'столбов {len(smokes)}, всего клубов/с {sum(s[5] for s in smokes):.1f} (на экране около {sum(s[5] for s in smokes)*LIFE:.0f})')
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
for (cx, cz), e in list(entities()):
    if e[b'id'][1] == b'gameplaymod:smoke':
        c = chunk_ents(cx, cz); before = len(c[b'Entities'][1][1])
        c[b'Entities'] = (9, (10, [q for q in c[b'Entities'][1][1] if q[b'id'][1] != b'gameplaymod:smoke']))
        removed += before - len(c[b'Entities'][1][1])
for x, y, z, r, h, rate in smokes:
    c = chunk_ents(math.floor(x) >> 4, math.floor(z) >> 4)
    c[b'Entities'][1][1].append({b'id': (8, b'gameplaymod:smoke'), b'Pos': (9, (6, [dbl(x), dbl(y), dbl(z)])), b'Motion': (9, (6, [dbl(0)]*3)),
        b'Rotation': (9, (5, [struct.pack('>f', 0.0)]*2)), b'UUID': (11, uuid.uuid4().bytes), b'Group': (8, b''),
        b'Radius': flt(r), b'Height': flt(h), b'Rate': flt(rate)})
regions = {}
for (cx, cz), (p, nm, c) in touched.items():
    if p not in regions:
        if not os.path.exists(p) or os.path.getsize(p) < 8192:
            with open(p, 'wb') as f: f.write(b'\0'*8192)
        regions[p] = mca.Region(p)
    regions[p].put(cx, cz, mca.dump(nm, c))
for r_ in regions.values(): r_.save()
print(f'записано столбов дыма {len(smokes)}, прежних убрано {removed}; копия {backup}')
