#!/usr/bin/env python3
"""Маршрут брёвен по реке для GameplayMod: читает мир и считает узлы маршрута.

Запуск: log_route.py <папка мира> [--name река] [--write <папка копии>], игра закрыта. Без --write только печатает узлы; с --write
записывает маршрут с этим именем в data/gameplaymod/log_routes.dat мира (прежний файл копируется в папку копии), GameplayMod читает его
при запуске мира. Перезапускать, если русло реки поменялось. Другая река — другие START, GOAL, рамка и CASCADE и своё --name.
Средняя линия — кратчайший путь по воде со штрафом за близость к берегу, затем сдвинутая к середине между берегами:
кратчайший путь прижимается к внешней стороне поворотов на 2–3 блока. Каскад к морю заменён дугой в стороне от скалы:
вода там стекает по отвесной стене, и прямая линия прошла бы сквозь камень.
"""
import heapq, math, os, struct, sys
from collections import deque
import numpy as np
from scipy import ndimage as ni
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca

X0, X1, Z0, Z1, Y0, Y1 = 340, 660, -260, -110, 50, 180
START, GOAL = (357, 140, -187), (621, 63, -190)
STEP = 3.0                  # шаг узлов, блоков
DRAFT_TOP = 8/9 - 0.3       # поверхность воды минус осадка плота, от верхней клетки воды
CASCADE_X = 606             # восточнее вместо средней линии идёт дуга
CENTER_PASSES = 80          # сдвигов к середине; между ними сглаживание, чтобы бревно не дёргалось
CASCADE = [(610, 115, -196.5), (615, 111, -194), (619, 103, -192), (620.5, 90, -191), (621, 75, -190.5), (621, 63 + DRAFT_TOP, -190)]
CASCADE_HW = 2.3            # полуширина на каскаде: брёвна сходятся к середине

world = sys.argv[1]
region = os.path.join(world, 'dimensions/minecraft/overworld/region/')
is_water = np.zeros((X1 - X0, Y1 - Y0, Z1 - Z0), bool)
regions = {}
for cx in range(X0 >> 4, (X1 >> 4) + 1):
    for cz in range(Z0 >> 4, (Z1 >> 4) + 1):
        path = f'{region}r.{cx >> 5}.{cz >> 5}.mca'
        r = regions.setdefault(path, mca.Region(path))
        if not r.present(cx, cz):
            continue
        for s in mca.parse(r.raw(cx, cz))[1][b'sections'][1][1]:
            sy = struct.unpack('>b', s[b'Y'][1])[0]
            bs = s.get(b'block_states')
            if not bs or sy*16 + 16 <= Y0 or sy*16 >= Y1:
                continue
            pal = np.array([mca.key(q).startswith('minecraft:water') for q in bs[1][b'palette'][1][1]])
            if not pal.any():
                continue
            ids = pal[mca.unpack(bs[1][b'data'][1], len(pal))].reshape(16, 16, 16) if b'data' in bs[1] else np.full((16, 16, 16), pal[0])
            for y, z, x in zip(*np.where(ids)):
                wx, wy, wz = cx*16 + x, sy*16 + y, cz*16 + z
                if X0 <= wx < X1 and Y0 <= wy < Y1 and Z0 <= wz < Z1:
                    is_water[wx - X0, wy - Y0, wz - Z0] = True

# связная вода от начала и её верхняя поверхность
s0 = (START[0] - X0, START[1] - Y0, START[2] - Z0)
assert is_water[s0], 'начало маршрута не в воде'
seen = np.zeros(is_water.shape, bool); seen[s0] = True; q = deque([s0])
while q:
    p = q.popleft()
    for a in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        n = (p[0] + a[0], p[1] + a[1], p[2] + a[2])
        if all(0 <= n[i] < is_water.shape[i] for i in range(3)) and is_water[n] and not seen[n]:
            seen[n] = True; q.append(n)
top = np.where(seen.any(1), seen.shape[1] - 1 - np.argmax(seen[:, ::-1, :], axis=1), -1)

# русло — всё выше моря, плюс море рядом с концом маршрута
river = (top >= 0) & (top + Y0 >= 64)
gx, gz = np.meshgrid(np.arange(river.shape[0]) + X0 - GOAL[0], np.arange(river.shape[1]) + Z0 - GOAL[2], indexing='ij')
river |= (top >= 0) & (top + Y0 < 64) & (np.hypot(gx, gz) < 16)
dt = ni.distance_transform_edt(river)

start, goal = (START[0] - X0, START[2] - Z0), (GOAL[0] - X0, GOAL[2] - Z0)
best = {start: 0.0}; prev = {}; heap = [(0.0, start)]
while heap:
    c, p = heapq.heappop(heap)
    if p == goal:
        break
    if c > best[p]:
        continue
    for a in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        n = (p[0] + a[0], p[1] + a[1])
        if not (0 <= n[0] < river.shape[0] and 0 <= n[1] < river.shape[1]) or not river[n]:
            continue
        cost = c + math.hypot(*a)*(1 + 40/dt[n]**2)
        if cost < best.get(n, 1e18):
            best[n] = cost; prev[n] = p; heapq.heappush(heap, (cost, n))
line = [goal]
while line[-1] != start:
    line.append(prev[line[-1]])
line = np.array(line[::-1], float)
for _ in range(60):
    line[1:-1] = 0.25*line[:-2] + 0.5*line[1:-1] + 0.25*line[2:]

def resample(p):
    a = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    t = np.r_[np.arange(0, a[-1] - 0.5, 1.0), a[-1]]   # конец сохраняется, иначе линия укорачивается с каждым проходом
    return np.c_[np.interp(t, a, p[:, 0]), np.interp(t, a, p[:, 1])]
def reach(x, z, dx, dz):
    """Сколько воды от точки в сторону (dx, dz)."""
    d = 0.0
    while d < 30:
        i, j = int(round(x + dx*d)), int(round(z + dz*d))
        if not (0 <= i < river.shape[0] and 0 <= j < river.shape[1]) or not river[i, j]:
            break
        d += 0.25
    return d
for _ in range(CENTER_PASSES):
    line = resample(line)
    tan = np.gradient(line, axis=0); tan /= np.hypot(*tan.T)[:, None]
    move = line[:, 0] + X0 < CASCADE_X
    move[0] = False
    shift = np.zeros(len(line))
    for k in np.nonzero(move)[0]:
        x, z, tx, tz = line[k, 0], line[k, 1], tan[k, 0], tan[k, 1]
        shift[k] = (reach(x, z, tz, -tx) - reach(x, z, -tz, tx))/2
    line += 0.3*shift[:, None]*np.c_[tan[:, 1], -tan[:, 0]]
    for _ in range(3):
        fixed = line.copy()
        line[1:-1] = 0.25*line[:-2] + 0.5*line[1:-1] + 0.25*line[2:]
        line[~move] = fixed[~move]

arc = np.r_[0, np.cumsum(np.hypot(*np.diff(line, axis=0).T))]
ss = np.r_[np.arange(0, arc[-1], STEP), arc[-1]]
xs, zs = np.interp(ss, arc, line[:, 0]), np.interp(ss, arc, line[:, 1])
ys, hw = [], []
txs, tzs = np.gradient(xs), np.gradient(zs); tl = np.hypot(txs, tzs); txs, tzs = txs/tl, tzs/tl
for x, z, tx, tz in zip(xs, zs, txs, tzs):
    i, j = int(round(x)), int(round(z))
    ys.append(top[i, j] + Y0 + DRAFT_TOP if top[i, j] >= 0 else np.nan)
    # полуширина поперёк линии: по ней бревно и качается, зазор до берега выходит одинаковым
    hw.append((reach(x, z, tz, -tx) + reach(x, z, -tz, tx))/2)
ys = np.minimum.accumulate(np.where(np.isnan(ys), START[1] + DRAFT_TOP, ys))
hw = ni.uniform_filter1d(ni.minimum_filter1d(np.array(hw), 3), 3)

nodes = [(xs[k] + X0, ys[k], zs[k] + Z0, hw[k]) for k in range(len(xs)) if xs[k] + X0 < CASCADE_X]
nodes += [(x, y, z, CASCADE_HW) for x, y, z in CASCADE]
nodes = [[round(float(v), 2) for v in n] for n in nodes]
for k in range(0, len(nodes), 6):
    print(' '.join(f'{v:.2f}' for n in nodes[k:k + 6] for v in n))
if '--write' in sys.argv:
    import log_routes_dat
    name = sys.argv[sys.argv.index('--name') + 1] if '--name' in sys.argv else 'река'
    log_routes_dat.write(world, name, nodes, sys.argv[sys.argv.index('--write') + 1])
    print(f'маршрут «{name}» ({len(nodes)} узлов) записан в {log_routes_dat.path(world)}')
