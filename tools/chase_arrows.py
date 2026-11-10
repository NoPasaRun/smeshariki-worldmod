#!/usr/bin/env python3
"""Стрелы погони в джунглях: сетка маркеров обстрела (gameplaymod:barrage) от засады туземцев до расщелины, загоняет игроков к тарзанкам.

Запуск: chase_arrows.py <мир> [--write <папка копии>], игра закрыта. Без --write только печатает план. С --write прежние маркеры обстрела
группы GROUP в рамке убираются и ставятся заново (копия файлов сущностей — в папку копии), поэтому повторный запуск даёт то же.

Обстрел бьёт только сзади (DENSITY), урон DAMAGE, над рекой маркеров нет, а игрока обстреливает ближайший маркер в пределах Radius.
Сетка маркеров с курсом на начало тарзанок — поле, которое в любой точке гонит к воронке: бежать по курсу почти безопасно (−4 хп за 10 с),
стоять или бежать против — больно. У самой воронки курс прямо на север, к верёвкам. Маркер ставится на землю (карта высот без листвы).
Включает стрелы уровень (LevelMod): действие gameplaymod:arrows с группой GROUP.
"""
import math, os, shutil, struct, sys, uuid
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca
import log_routes_dat

GROUP = 'погоня'
TARGET = (500.0, -262.0)        # середина начала тарзанок на южном краю расщелины (первый ряд x 462..538, z −272..−276)
FUNNEL = 20.0                   # ближе к цели — курс на север, к верёвкам
X_FROM, X_TO, STEP = 395, 600, 22
Z_FROM, Z_TO = -95, -258        # от засады (x 398..594, z −142..−68) до края расщелины (z≈−268)
MIN_GROUND = 115                # ниже — обрыв к морю или в расщелину: туда маркер не нужен
RADIUS = 20.0                   # покрывает ячейку сетки (половина диагонали 15,6)
# 2026-11-04, по просьбе «стрелы слишком хардкорные»: только сзади (120°…240° от курса), около 2 стрел/с вместо 4,2 по умолчанию,
# урон вдвое меньше (половина сердечка), над рекой маркеров нет — там ждут бревно, а стоящего стрелы догоняют
DENSITY = [0, 0, 0, 0, 0.3, 0.45, 0.5, 0.45, 0.3, 0, 0, 0]
DAMAGE = 0.5
RIVER_CLEAR = 12.0              # без маркеров ближе этого к берегу реки (полуширина русла + это)
FRAME = (380, 615, -270, -80)   # где убирать прежние маркеры группы

world = sys.argv[1]
backup = sys.argv[sys.argv.index('--write') + 1] if '--write' in sys.argv else None
ENT = os.path.join(world, 'dimensions/minecraft/overworld/entities/')
REG = os.path.join(world, 'dimensions/minecraft/overworld/region/')

_ch = {}
def ground(x, z):
    cx, cz = x >> 4, z >> 4
    if (cx, cz) not in _ch:
        nm, c = mca.parse(mca.Region(REG + f'r.{cx >> 5}.{cz >> 5}.mca').raw(cx, cz))
        raw = np.frombuffer(c[b'Heightmaps'][1][b'MOTION_BLOCKING_NO_LEAVES'][1], '>u8')
        v = [(int(w) >> (9*k)) & 511 for w in raw for k in range(7)][:256]
        _ch[(cx, cz)] = np.array(v).reshape(16, 16) - 65
    return int(_ch[(cx, cz)][z & 15, x & 15])

river = log_routes_dat.read(world).get('река', [])
def near_river(x, z):
    return any(math.hypot(x - n[0], z - n[2]) < n[3] + RIVER_CLEAR for n in river)

markers = []
skipped_river = 0
for z in range(Z_FROM, Z_TO - 1, -STEP):
    for x in range(X_FROM, X_TO + 1, STEP):
        g = ground(x, z)
        if g < MIN_GROUND: continue
        if near_river(x + 0.5, z + 0.5):
            skipped_river += 1
            continue
        dx, dz = TARGET[0] - x, TARGET[1] - z
        yaw = 180.0 if math.hypot(dx, dz) < FUNNEL else math.degrees(math.atan2(-dx, dz))
        markers.append((x + 0.5, g + 1.0, z + 0.5, round(yaw, 1)))

arrows = {0: '↓', 45: '↙', 90: '←', 135: '↖', 180: '↑', -180: '↑', -135: '↗', -90: '→', -45: '↘'}
print(f'маркеров {len(markers)} (у реки не ставится {skipped_river}), группа «{GROUP}», цель {TARGET}, стрел/с {sum(DENSITY):.1f}, урон {DAMAGE}')
rows = {}
for x, y, z, yaw in markers: rows.setdefault(z, {})[x] = arrows[int(round(yaw / 45.0)) * 45 if round(yaw / 45.0) * 45 != 180 else 180]
for z in sorted(rows, reverse=True):
    print(f'{z:>7} ' + ' '.join(rows[z].get(x + 0.5, '·') for x in range(X_FROM, X_TO + 1, STEP)))
if not backup:
    print('без --write мир не изменён'); sys.exit()

os.makedirs(os.path.join(backup, 'entities'), exist_ok=True)
dbl = lambda v: struct.pack('>d', float(v)); flt = lambda v: (5, struct.pack('>f', float(v)))
touched = {}
def chunk_ents(cx, cz, dv):
    if (cx, cz) not in touched:
        p = ENT + f'r.{cx >> 5}.{cz >> 5}.mca'
        if os.path.exists(p) and not os.path.exists(os.path.join(backup, 'entities', os.path.basename(p))): shutil.copy2(p, os.path.join(backup, 'entities'))
        b = mca.Region(p).raw(cx, cz) if os.path.exists(p) else None
        if b is None: nm, c = b'', {b'Position': (11, struct.pack('>ii', cx, cz)), b'DataVersion': dv, b'Entities': (9, (10, []))}
        else: nm, c = mca.parse(b)
        if c[b'Entities'][1][0] == 0: c[b'Entities'] = (9, (10, c[b'Entities'][1][1]))
        touched[(cx, cz)] = (p, nm, c)
    return touched[(cx, cz)][2]
_, lvl = mca.parse(__import__('gzip').open(os.path.join(world, 'level.dat')).read())
dv = lvl[b'Data'][1][b'DataVersion']
removed = 0
for cx in range(FRAME[0] >> 4, (FRAME[1] >> 4) + 1):
    for cz in range(FRAME[2] >> 4, (FRAME[3] >> 4) + 1):
        p = ENT + f'r.{cx >> 5}.{cz >> 5}.mca'
        if not os.path.exists(p) or mca.Region(p).raw(cx, cz) is None: continue
        c = chunk_ents(cx, cz, dv); before = len(c[b'Entities'][1][1])
        c[b'Entities'] = (9, (10, [e for e in c[b'Entities'][1][1] if not (e[b'id'][1] == b'gameplaymod:barrage' and e.get(b'Group', (8, b''))[1] == GROUP.encode())]))
        removed += before - len(c[b'Entities'][1][1])
for x, y, z, yaw in markers:
    c = chunk_ents(math.floor(x) >> 4, math.floor(z) >> 4, dv)
    c[b'Entities'][1][1].append({b'id': (8, b'gameplaymod:barrage'), b'Pos': (9, (6, [dbl(x), dbl(y), dbl(z)])), b'Motion': (9, (6, [dbl(0)]*3)),
        b'Rotation': (9, (5, [struct.pack('>f', 0.0)]*2)), b'UUID': (11, uuid.uuid4().bytes), b'Yaw': flt(yaw), b'Radius': flt(RADIUS), b'Damage': flt(DAMAGE),
        b'Density': (9, (5, [struct.pack('>f', float(d)) for d in DENSITY])),
        b'Group': (8, GROUP.encode())})
regions = {}
for (cx, cz), (p, nm, c) in touched.items():
    r = regions.setdefault(p, mca.Region(p) if os.path.exists(p) else None)
    if r is None: continue
    r.put(cx, cz, mca.dump(nm, c))
for r in regions.values():
    if r is not None: r.save()
print(f'записано маркеров {len(markers)}, прежних убрано {removed}; копия {backup}')
