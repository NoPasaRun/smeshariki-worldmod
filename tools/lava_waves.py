#!/usr/bin/env python3
"""Озеро с волнами лавы в пещере конца тоннеля (вместо паркура lava_parkour.py).

Запуск: lava_waves.py <папка мира> --restore <копия до паркура> [--preview файл.png] [--write <новая папка копии>]. Без --write мир не меняется.
Игра должна быть закрыта. Сначала рамка пещеры (блоки и сущности) возвращается из копии (там пещера final_cave.py без паркура), потом строится озеро,
поэтому повторный запуск даёт тот же результат.

Вся пещера — озеро лавы (lon 0,3..72, |lat| ≤ 11), свод куполом; плоты (4 ряда поперёк, gameplaymod:raft) лежат на лаве у выхода из тоннеля под первыми
(прежними) тарзанками, те удлиняются до плотов.
Волны идут вдоль, по очереди от концов A (у тоннеля) и B, WAVES = 4 волны, каждое обрушение поднимает лаву на RISE = 3 слоя (−38 → −29), скорость растёт 2 → 5 блоков/с.
Волна k оставляет плоты у P_k, там уступ L_k (нечётные юг, чётные север); с него игрок садится на плот волны k+1 и перебегает по плотам к L_{k+1} на другой стене.
Верх плота на гребне = верх лавы + 4,48 (замер в gametest волн), поэтому верх уступа L_k = лава волны k + 4: с плота на уступ +0,52, с уступа на плот −0,41.
Маркеры волн: группы lava1..lava4 (нечётные у A, чётные у B); уровень LevelMod пускает их цепочкой (начало → lava1, конец lava_k → lava_{k+1}).
Выход: ниша за последним уступом и жерло (сужается кверху, идёт за траекторией полёта) с выбросом в море.
"""
import math, os, shutil, struct, sys, uuid
import numpy as np
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca, tunnel_field as TF

SEED = 20261104
Y0, Y1 = -64, 320
WAVES, RISE = 4, 3                         # 4 волны, по 2 в каждую сторону (чётное число: последний уступ на севере, там ниша); обрушение +3 слоя (кроме последней)
PEAK = -29                                 # лава последней волны (её обрушение уже не поднимает): последний уступ −25, ниша и жерло на той же высоте, что при 12 волнах
LAVA_BOTTOM, BASE = -63, PEAK - RISE*(WAVES - 1)   # лава: низ, верхняя клетка в начале (−38)
SPEEDS = [round(2.0 + 3.0*k/(WAVES - 1), 2) for k in range(WAVES)]   # волна k+1: блоков/с, от 2 до 5
HEIGHT = 3.0                              # высота гребня
RAFT_RISE = 4.48                          # верх плота на гребне над верхней клеткой лавы (gametest: низ плота base+3,48)
CALM_TOP = 8/9 - 0.3 + 1                  # верх плота в покое над верхней клеткой лавы (1,59)
LON0, LON1 = 0.3, 72.0                    # лава от обрыва под тоннелем до конца пещеры
HW = 11.0                                 # стены озера |lat| = 11
LEDGE_IN = (7.5, 8.3)                     # полки нависают от стены до |lat| 7,5–8,3 (своя у каждой)
LIFT = PEAK + 41                          # свод выше первого варианта (последняя лава −41) на столько же, на сколько выше последний уступ (12)
DOME = (-22.0 + LIFT, -15.0 + LIFT)       # свод над озером: у стен и посередине (−10 / −3)
RAFT_LEAD = 1.7                           # плот едет впереди гребня (gametest)
# ряды плотов: (поперёк, размер). Зазоры: стена 1,75 | 2×2 | 2,2 | 1×1 | 2,6 | 1×1 | 2,2 | 2×2 | 1,75 стена
RAFTS = [(-5.6, 2), (-1.8, 1), (1.8, 1), (5.6, 2)]
P0 = 6.5                                  # плоты в начале: под первыми тарзанками у выхода из тоннеля (прежние, lon 6–6,8, крепления на −14)
DESCENT_FEET = 2.8                        # тарзанки спуска: ноги висящего над плотом в покое
HAND = 1.08                               # рука висящего над его ногами
ROPE_MAX = 46.0                           # calm_rope до 48 (RopeEntity.LONGEST): со свода без наплывов в путь плотов
# крест: волна k оставляет плоты у P_k = C ± D_k (нечётные к B, чётные к A), там уступ L_k на стене (нечётные юг, чётные север);
# D растёт — крест расходится. Уступ длиной LEDGE_LEN вдоль, из него видно подходящий плот.
# где волна k оставляет плоты (P_k) и стоит уступ L_k (нечётные — юг, к B; чётные — север, к A). Подбор случайный с фиксированным seed:
# в пути не меньше 4 с (+1 блок; с ростом скорости путь длиннее), на одной стене уступы не ближе MIN_SAME вдоль и вразнобой по высоте.
LEDGE_LEN, MIN_SAME, P_RANGE = 5.0, 8.0, (6.0, 66.0)
NEAR_START = 21.0                         # уступ ближе к выходу из тоннеля: край не ближе 8,3, с тарзанки спуска (размах 2) не дотянуться
T_MAX = 14.0                              # в пути не дольше 14 с
KGAP = 6                                  # на одной стене уступы ближе MIN_SAME вдоль — только через 6 волн (18 блоков по высоте)
def pick_route(seed):
    """Случайные пробы (P_k на сетке 0,5 среди допустимых); берётся маршрут с наибольшим наименьшим зазором вдоль у соседних по высоте уступов одной стены."""
    g = np.random.default_rng(seed)
    mono = lambda q: len(q) > 2 and (all(y > x for x, y in zip(q, q[1:])) or all(y < x for x, y in zip(q, q[1:])))   # из двух порядок не уйдёт
    grid = np.arange(P_RANGE[0], P_RANGE[1] + 0.01, 0.5)
    best = None
    for _ in range(40000):
        P_ = {0: P0}; ok = True; same = 1e9
        for k in range(1, WAVES + 1):
            d = 1 if k % 2 else -1; need = 4*SPEEDS[k - 1] + 1
            c = [v for v in grid[((grid - P_[k - 1])*d >= need) & ((grid - P_[k - 1])*d <= T_MAX*SPEEDS[k - 1])] if all(abs(v - P_[q]) >= MIN_SAME or k - q >= KGAP for q in range(k - 2, 0, -2))]
            if k == WAVES: c = [v for v in c if v - 1.5 >= NEAR_START + 1]   # уступ с выходом подальше от тоннеля: с вылета до него не допрыгнуть
            if not c: ok = False; break
            P_[k] = float(g.choice(c))
            same = min([same] + [abs(P_[k] - P_[q]) for q in range(k - 2, 0, -2) if k - q < KGAP])
        if not ok: continue
        # вразнобой: на каждой стене уступы не идут вдоль по порядку высоты
        if mono([P_[k] for k in range(1, WAVES + 1, 2)]) or mono([P_[k] for k in range(2, WAVES + 1, 2)]): continue
        if best is None or same > best[0]: best = (same, P_)
    return best[1]
P = pick_route(SEED)
print('плоты останавливаются (P_k):', P)
LEDGES = []                               # (номер, сторона: +1 север, -1 юг, от, до вдоль, край полки |lat|)
_lg = np.random.default_rng(SEED + 7)
for k in range(1, WAVES + 1):
    d = 1 if k % 2 else -1                # куда шла волна k
    lo0, lo1 = sorted((P[k] + d*1.5, P[k] - d*(LEDGE_LEN - 1.5)))
    edge = round(float(_lg.uniform(*LEDGE_IN)), 2)
    if lo0 < NEAR_START: edge = LEDGE_IN[1]
    LEDGES.append((k, -1 if k % 2 else +1, lo0, lo1, edge))
ALCOVE_LEN, ALCOVE_TO = 3.0, 15.0         # ниша за L8: длина вдоль, до |lat|
def lava_top(k): return BASE + RISE*(k - 1)          # лава во время волны k
def ledge_top(k): return lava_top(k) + 4

world = sys.argv[1]
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
backup, preview, restore = arg('--write'), arg('--preview'), arg('--restore')
if not restore: raise SystemExit('нужен --restore <копия до паркура>: скрипт строит на пещере без паркура')
D = os.path.join(world, 'dimensions/minecraft/overworld/'); REG = D + 'region/'; ENT = D + 'entities/'
X0, X1, Z0, Z1 = 96, 207, -304, -209
cx0, cx1, cz0, cz1 = X0 >> 4, X1 >> 4, Z0 >> 4, Z1 >> 4
ox, oz = cx0*16, cz0*16; sx, sz = (cx1 - cx0 + 1)*16, (cz1 - cz0 + 1)*16; NY = Y1 - Y0

# ---------- чтение: рамка из копии, остальное из мира
regs, eregs = {}, {}
def region(store, p):
    if p not in store: store[p] = mca.Region(p)
    return store[p]
src_r = {}; src_e = {}
restored = 0
for cx in range(cx0, cx1 + 1):
    for cz in range(cz0, cz1 + 1):
        name = f'r.{cx >> 5}.{cz >> 5}.mca'
        b = region(src_r, os.path.join(restore, 'region', name)).raw(cx, cz)
        if b is None: raise SystemExit(f'в копии нет чанка {cx} {cz}')
        r = region(regs, REG + name)
        if r.raw(cx, cz) != b: r.put(cx, cz, b); restored += 1
print('чанков блоков возвращено из копии', restored)
vol = np.zeros((sx, NY, sz), np.uint16); names = {'minecraft:air': 0}; lst = ['minecraft:air']; chunks = {}
def gid(k):
    if k not in names: names[k] = len(lst); lst.append(k)
    return names[k]
for cx in range(cx0, cx1 + 1):
    for cz in range(cz0, cz1 + 1):
        p = REG + f'r.{cx >> 5}.{cz >> 5}.mca'
        nm, c = mca.parse(regs[p].raw(cx, cz)); chunks[(cx, cz)] = (p, nm, c)
        for sec in c[b'sections'][1][1]:
            y = int.from_bytes(sec[b'Y'][1], 'big', signed=True); bs = sec.get(b'block_states')
            if bs is None or not Y0 <= y*16 < Y1: continue
            pal = np.array([gid(mca.key(q)) for q in bs[1][b'palette'][1][1]])
            ids = pal[mca.unpack(bs[1][b'data'][1], len(pal))].reshape(16, 16, 16) if b'data' in bs[1] else np.full((16, 16, 16), pal[0])
            vol[(cx - cx0)*16:(cx - cx0 + 1)*16, y*16 - Y0:y*16 - Y0 + 16, (cz - cz0)*16:(cz - cz0 + 1)*16] = ids.transpose(2, 0, 1)
orig = vol.copy()
base = lambda i: lst[i].split('[')[0].split(':')[1]
air = 0; lava_id = gid('minecraft:lava[level=0]')
yi = lambda y: y - Y0
lava_cells = int((vol[:, yi(BASE), :] == lava_id).sum())
print('лавы на y', BASE, 'в копии:', lava_cells)

# ---------- оси пещеры (как в final_cave.py)
k_ = np.searchsorted(TF.ts, TF.LIP_S); lip = TF.tc[k_]; ax = (TF.tc[-1] - lip)/np.hypot(*(TF.tc[-1] - lip))
gx, gz = np.meshgrid(np.arange(sx) + ox + .5, np.arange(sz) + oz + .5, indexing='ij')
lon = (gx - lip[0])*ax[0] + (gz - lip[1])*ax[1]; lat = -(gx - lip[0])*ax[1] + (gz - lip[1])*ax[0]
def world_xz(lo, la): return lip[0] + ax[0]*lo - ax[1]*la, lip[1] + ax[1]*lo + ax[0]*la
def lonlat(x, z): return (x - lip[0])*ax[0] + (z - lip[1])*ax[1], -(x - lip[0])*ax[1] + (z - lip[1])*ax[0]
YAW_A = math.degrees(math.atan2(-ax[0], ax[1]))          # курс (−sin, cos) = ax
YAW_B = YAW_A - 180.0
rng = np.random.default_rng(SEED)
black = [gid('minecraft:blackstone'), gid('minecraft:basalt[axis=y]'), gid('minecraft:tuff'), gid('minecraft:deepslate')]
ice_p, ice_b = gid('minecraft:packed_ice'), gid('minecraft:blue_ice')
top_id = gid('minecraft:polished_blackstone')
lamp = gid('minecraft:sea_lantern')
def rock(): return black[int(rng.random()*4)]
cave_air = (vol == air) | (vol == lava_id)
def column(i, j, top, cap=None):
    """Столбец камня от дна до top, верх полированный; над ним воздух до cap (или пещерный воздух остаётся)."""
    for y in range(LAVA_BOTTOM - 1, top + 1):
        if base(vol[i, yi(y), j]) != 'bedrock': vol[i, yi(y), j] = rock()
    vol[i, yi(top), j] = top_id
    for y in range(top + 1, (cap if cap is not None else top + 1)):
        vol[i, yi(y), j] = air

# ---------- зоны
cave = (lon >= LON0) & (lon < 90) & (np.abs(lat) < 34)    # перестраиваемая часть: всё озеро и стены вокруг
inlake = (lon >= LON0) & (lon < LON1) & (np.abs(lat) <= HW)
ledge_mask = {}                           # уступы одной стены могут стоять один над другим (через KGAP волн), поэтому маска у каждого своя
for k, side, lo0, lo1, edge in LEDGES:
    ledge_mask[k] = (lon >= lo0) & (lon < lo1) & (side*lat >= edge) & (side*lat <= HW)
L8C = (LEDGES[-1][2] + LEDGES[-1][3])/2
LSIDE = LEDGES[-1][1]
alc = (lon >= L8C - ALCOVE_LEN/2) & (lon < L8C + ALCOVE_LEN/2) & (LSIDE*lat > HW) & (LSIDE*lat <= ALCOVE_TO)
for i, j in zip(*np.nonzero(cave)):
    for y in range(LAVA_BOTTOM - 4, 16):
        n_ = base(vol[i, yi(y), j])
        if 'spike' in n_ or 'column' in n_: vol[i, yi(y), j] = air
# стены: всё вне озера в этой части на всю высоту пещеры, конец пещеры за LON1 тоже
for i, j in zip(*np.nonzero(cave & ~inlake & ~alc)):
    col = vol[i, :, j]
    for y in range(LAVA_BOTTOM - 1, 16):
        if col[yi(y)] in (air, lava_id): col[yi(y)] = rock()
# дно, лава, воздух над ней до свода-купола (выше старого свода — пробивается)
dome_n = gaussian_filter(rng.normal(size=(sx, sz)), 3, mode='wrap'); dome_n /= dome_n.std()
for i, j in zip(*np.nonzero(inlake)):
    col = vol[i, :, j]
    for y in range(LAVA_BOTTOM - 1, LAVA_BOTTOM):
        if base(col[yi(y)]) != 'bedrock': col[yi(y)] = rock()
    for y in range(LAVA_BOTTOM, BASE + 1): col[yi(y)] = lava_id
    u = min(1.0, abs(lat[i, j])/HW)
    top = int(round(DOME[0] + (DOME[1] - DOME[0])*math.sqrt(1 - u*u) + 1.2*dome_n[i, j]))
    for y in range(BASE + 1, top + 1): col[yi(y)] = air
# прежние тарзанки у выхода из тоннеля (lon 0..13) крепились к старому своду; он поднят — от крепления вверх до нового свода ледяной наплыв
old_anchors = []
for cx in range(cx0, cx1 + 1):
    for cz in range(cz0, cz1 + 1):
        b = region(src_e, os.path.join(restore, 'entities', f'r.{cx >> 5}.{cz >> 5}.mca')).raw(cx, cz)
        if b is None: continue
        for e in mca.parse(b)[1][b'Entities'][1][1]:
            if e[b'id'][1] not in (b'gameplaymod:tarzan_rope', b'gameplaymod:calm_rope'): continue
            lo_, la_e = lonlat(struct.unpack('>d', e[b'AX'][1])[0], struct.unpack('>d', e[b'AZ'][1])[0])
            if 0 <= lo_ < 13 and abs(la_e) < 22:
                old_anchors.append(tuple(int.from_bytes(e[k_][1], 'big', signed=True) for k_ in (b'BX', b'BY', b'BZ')))
for bx, by, bz in old_anchors:
    i, j = bx - ox, bz - oz
    top = by
    while vol[i, yi(top), j] in (air, lava_id): top += 1
    for ry in range(by, top):
        for a in range(-3, 4):
            for b in range(-3, 4):
                if a*a + b*b <= 1 + (top - ry)*0.35 and vol[i + a, yi(ry), j + b] == air: vol[i + a, yi(ry), j + b] = ice_p
print('прежние тарзанки у тоннеля, крепления (наплыв до нового свода):', old_anchors)
# уступы — столбы от дна
ledge_cells = {k: [] for k, *_ in LEDGES}
for k, i, j in ((k, i, j) for k in ledge_mask for i, j in zip(*np.nonzero(ledge_mask[k]))):
    t = ledge_top(k)
    thick = 2 + int(round(1.3*max(0.0, abs(lat[i, j]) - LEDGES[k - 1][4])))   # полка: у края 2 блока, к стене толще (скос снизу)
    for y in range(t - thick, t + 1): vol[i, yi(y), j] = rock()
    vol[i, yi(t), j] = top_id; ledge_cells[k].append((i, j))
# ниша за последним уступом: пол вровень с ним, высота 3
for i, j in zip(*np.nonzero(alc)):
    column(i, j, ledge_top(WAVES), ledge_top(WAVES) + 4)

# ---------- облицовка: открытые грани новых стен выше −40 — лёд, ниже — тёмный камень; фонари
changed = vol != orig
op = (vol == air) | (vol == lava_id)
nb = np.zeros_like(op)
for a_, s_ in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)): nb |= np.roll(op, s_, a_)
bed = gid('minecraft:bedrock')
opl = op & (inlake | alc)[:, None, :]                   # воздух и лава озера и ниши (не закрытые полости прежней пещеры за стенами)
nbl = np.zeros_like(opl)
for a_, s_ in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)): nbl |= np.roll(opl, s_, a_)
# новые тёмные грани и все грани, смотрящие в озеро (свод поднят выше прежнего — там нетронутый камень), кроме верха уступов и наплывов
face = (changed & ~op & nb & np.isin(vol, black)) | (~op & nbl & cave[:, None, :] & ~np.isin(vol, [top_id, bed, ice_p, ice_b, lamp]))
yy = (np.arange(NY) + Y0)[None, :, None]
HEAT_Y = PEAK + 2                         # ниже — тёмный камень (лава до PEAK у него), выше — лёд
heat = -(yy - HEAT_Y)/5.0 + 0.6*gaussian_filter(rng.normal(size=vol.shape), 2.5, mode='wrap')
cold = face & (heat < 0)
r_ = rng.random(vol.shape)
vol[cold] = np.where(r_[cold] < 0.25, ice_b, ice_p)
pts = np.argwhere(face & (yy >= -36) & (yy <= -18 + LIFT))
rng.shuffle(pts); lamps = []
for p in pts:
    if len(lamps) >= 40 + 2*LIFT: break
    if all(((p - q)**2).sum() > 64 for q in lamps): vol[tuple(p)] = lamp; lamps.append(p)

# ---------- плоты: под тарзанками у выхода из тоннеля
raft_ents = []
for la, size in RAFTS:
    best = None
    for lo in np.arange(P0 - 0.6, P0 + 0.6, 0.02):
        x, z = world_xz(lo, la)
        mx = round(x - size/2); mz = round(z - size/2)
        cxw, czw = mx + size/2, mz + size/2
        lo2, la2 = lonlat(cxw, czw)
        e = abs(la2 - la) + 0.2*abs(lo2 - P0)
        if best is None or e < best[0]: best = (e, mx, mz, lo2, la2)
    _, mx, mz, lo2, la2 = best
    raft_ents.append((mx, mz, size, lo2, la2))
print('плоты (угол x, z, размер, вдоль, поперёк):', [(m[0], m[1], m[2], round(m[3], 2), round(m[4], 2)) for m in raft_ents])
for mx, mz, size, *_ in raft_ents:
    for a in range(size):
        for b in range(size):
            assert vol[mx + a - ox, yi(BASE), mz + b - oz] == lava_id and vol[mx + a - ox, yi(BASE + 1), mz + b - oz] == air, 'плот не на лаве'
# зазоры поперёк (поворот на угол оси: квадрат s шириной s·(cos+sin))
th = math.atan2(abs(ax[1]), abs(ax[0])); ext = lambda s: s*(math.cos(th) + math.sin(th))/2
edges = [(-LEDGE_IN[0], -LEDGE_IN[0])] + [(m[4] - ext(m[2]), m[4] + ext(m[2])) for m in raft_ents] + [(LEDGE_IN[0], LEDGE_IN[0])]
print('зазоры поперёк от уступа до уступа:', [round(b[0] - a[1], 2) for a, b in zip(edges, edges[1:])])

# ---------- маркеры волн: у концов озера, длина пути своя у каждой волны (плоты останавливаются у P_k)
def marker_at(lo):
    x, z = world_xz(lo, 0.0); i, j = int(math.floor(x)), int(math.floor(z))
    return i + .5, j + .5
ma = marker_at(LON0 + 1.2); mb = marker_at(LON1 - 0.5)
la_, _ = lonlat(*ma); lb_, _ = lonlat(*mb)
LENS = {k: round((P[k] - RAFT_LEAD) - la_ if k % 2 else lb_ - (P[k] + RAFT_LEAD), 2) for k in range(1, WAVES + 1)}
print(f'маркеры: A ({ma[0]}, {ma[1]}) вдоль {la_:.2f} курс {YAW_A:.1f}; B ({mb[0]}, {mb[1]}) вдоль {lb_:.2f} курс {YAW_B:.1f}; пути', LENS)
for m in (ma, mb):
    i, j = math.floor(m[0]) - ox, math.floor(m[1]) - oz
    assert vol[i, yi(BASE), j] == lava_id and all(vol[i, yi(y), j] == air for y in range(BASE + 1, PEAK + 4)), 'столбец маркера занят'

# ---------- расписание
print('волна | скорость | лава | плоты от → до | уступ (вдоль, верх) | в пути, с | на уступ с гребня / с уступа на плот в покое после подъёма')
for k in range(1, WAVES + 1):
    T = lava_top(k); v = SPEEDS[k - 1]
    t = abs(P[k] - P[k - 1])/v
    on = ledge_top(k) + 1 - (T + RAFT_RISE)
    off = (T + RISE + CALM_TOP) - (ledge_top(k) + 1) if k < WAVES else float('nan')
    _, side, lo0, lo1, edge = LEDGES[k - 1]
    print(f'  {k} | {v} | {T} | {P[k-1]:.1f} → {P[k]:.1f} | L{k} {"юг" if side < 0 else "север"} {lo0:.1f}..{lo1:.1f}, {ledge_top(k)} | {t:.1f} | {on:+.2f} / {off:+.2f}')
    assert on <= 1.0 and (k == WAVES or -3 < off <= 1.0), 'прыжок выше блока'
    assert k == 1 or t >= 4.0, 'мало времени на перебежку'
    assert ledge_top(k) > T + RISE or k == WAVES, 'уступ уходит под лаву раньше времени'
    assert LENS[k] > 3, 'короткая волна'

# ---------- тарзанки: спуск на плоты у выхода из тоннеля и страховочные (по две на волну)
ropes = []
def hang(lo, la, end):
    x, z = world_xz(lo, la); i, j = int(math.floor(x)) - ox, int(math.floor(z)) - oz
    y = int(math.floor(end)) + 1
    while vol[i, yi(y), j] in (air, lava_id): y += 1
    ry = y
    while ry - 0.05 - end > ROPE_MAX - 0.5:                 # наплыв льда со свода
        ry -= 1
        for a in range(-3, 4):
            for b in range(-3, 4):
                if a*a + b*b <= 1 + (y - ry)*0.35 and vol[i + a, yi(ry), j + b] == air: vol[i + a, yi(ry), j + b] = ice_p
    ropes.append((i + ox, ry, j + oz, round(ry - 0.05 - end, 1)))
    return ropes[-1]
for k in range(1, WAVES):                                     # сорвавшийся с волны k висит, пока под ним не пройдут плоты волны k+1
    end = lava_top(k + 1) + RAFT_RISE + 2.5 + HAND
    for f, la in ((0.3, -3.4), (0.7, 3.4)):
        lo = P[k - 1] + (P[k] - P[k - 1])*f
        if any(abs(lonlat(r[0] + .5, r[2] + .5)[0] - lo) < 2.5 and abs(lonlat(r[0] + .5, r[2] + .5)[1] - la) < 2 for r in ropes): la = 0.0
        hang(lo, la, end)
# спуск — по прежним тарзанкам у выхода из тоннеля: удлиняются так, что висящий на конце над плотом, как у прежних тарзанок спуска
DESCENT_END = BASE + CALM_TOP + DESCENT_FEET + HAND
print('тарзанки (x, y крепления, z, длина):', ropes, '; спуск по прежним, конец на y', round(DESCENT_END, 2))
for rx, ry, rz in old_anchors:
    rlo, rla = lonlat(rx + .5, rz + .5)
    for k, side, lo0, lo1, edge in LEDGES:
        if lo0 - 2.3 < rlo < lo1 + 2.3 and side*rla > 0 and edge - abs(rla) < 2.3 + 0.3:
            raise SystemExit(f'с тарзанки спуска ({rlo:.1f}, {rla:.1f}) достать до L{k}')
tips = min(r[1] for r in ropes)
print(f'нижнее крепление y={tips}; голова на плоту последней волны y={lava_top(WAVES) + RAFT_RISE + 1.8:.1f}')
assert tips > lava_top(WAVES) + RAFT_RISE + 2.5, 'наплыв льда в пути плотов'

# ---------- шахта с жерлом из ниши (как в lava_parkour.py)
pad_x, pad_z = world_xz(L8C, (HW + ALCOVE_TO)/2 + 0.5)
fx, fz = math.floor(pad_x) + .5, math.floor(pad_z) + .5
STAND = ledge_top(WAVES) + 1
assert alc[math.floor(fx) - ox, math.floor(fz) - oz], 'жерло не в нише'
def surface_at(i, j):
    k2 = np.nonzero(vol[i, :, j] != air)[0]; return int(k2.max()) + Y0
pi_, pj_ = math.floor(fx) - ox, math.floor(fz) - oz
ground = max(surface_at(pi_ + a, pj_ + b) for a in range(-14, 15) for b in range(-22, 7))
G, DY = 0.08, 0.98
def apex(v_):
    y = 0
    while v_ > 0: y += v_; v_ = (v_ - G)*DY
    return y
def speed_for(h):
    lo_, hi_ = 0.0, 16.0
    for _ in range(40):
        m_ = (lo_ + hi_)/2
        if apex(m_) < h: lo_ = m_
        else: hi_ = m_
    return hi_
def arc(h):
    v_ = speed_for(h); ys = [0.0]; v0 = v_
    for t_ in range(1, 600):
        ys.append(ys[-1] + v_); v_ = (v_ - G)*DY
        if ys[-1] <= 0 and t_ > 1: break
    return v0, ys
SEA_Y = 62
# рельеф по курсу за рамкой (только чтение, мир там не меняется): верх непустого блока (листва и лианы считаются), вода
_ext = {}
def ext_col(x, z):
    cx, cz = x >> 4, z >> 4
    if (cx, cz) not in _ext:
        p = REG + f'r.{cx >> 5}.{cz >> 5}.mca'
        nm, c = mca.parse(region(regs, p).raw(cx, cz)); cols = {}
        for sec in c[b'sections'][1][1]:
            y = int.from_bytes(sec[b'Y'][1], 'big', signed=True); bs = sec.get(b'block_states')
            if bs is None or not Y0 <= y*16 < Y1: continue
            pal = [mca.key(q) for q in bs[1][b'palette'][1][1]]
            ids = mca.unpack(bs[1][b'data'][1], len(pal)).reshape(16, 16, 16) if b'data' in bs[1] else np.zeros((16, 16, 16), int)
            cols[y] = (pal, ids)
        _ext[(cx, cz)] = cols
    best = (Y0, '')
    for y, (pal, ids) in _ext[(cx, cz)].items():
        for ly in range(15, -1, -1):
            k = pal[ids[ly, z & 15, x & 15]]
            if 'air' not in k:
                if y*16 + ly > best[0]: best = (y*16 + ly, k)
                break
    return best
def surf(x, z):
    i, j = x - ox, z - oz
    if 0 <= i < sx and 0 <= j < sz: return surface_at(i, j)
    return ext_col(x, z)[0]
def cells_at(course, xp):
    """Клетки на траектории и в 2 блоках по бокам от неё."""
    x, z = fx + course[0]*xp, fz + course[1]*xp
    return [(math.floor(x - course[1]*a), math.floor(z + course[0]*a)) for a in (-2, 0, 2)]
sh = ground + 1 - STAND; water = SEA_Y + 1 - STAND
choice = None
for h in range(int(sh + 12), 300, 2):
    vy0, ys = arc(h)
    if vy0 > 9.5: break
    T_ = len(ys) - 1
    up = next(n for n, y_ in enumerate(ys) if y_ >= sh); down = next(n for n, y_ in enumerate(ys) if n > up and y_ < water)
    for dyaw in sorted(range(-70, 71, 5), key=abs):
        yaw = 180.0 + dyaw; course = (-math.sin(math.radians(yaw)), math.cos(math.radians(yaw)))
        wet = [d for d in range(20, 90) if all(ext_col(x, z)[1].startswith('minecraft:water') and ext_col(x, z)[0] == SEA_Y for x, z in cells_at(course, d))]
        d_water = next((d for d in wet if all(d + q in wet for q in range(12))), None)   # вода не меньше 12 по курсу
        if d_water is None: continue
        need = (d_water + 5)*T_/down
        if need > 64: continue
        cl = min(STAND + ys[t_] - max(surf(x, z) for x, z in cells_at(course, need*t_/T_)) for t_ in range(up, down))
        if cl < 3: continue
        choice = (h, vy0, T_, up, down, need, cl, yaw, course, d_water); break
    if choice: break
assert choice, 'до воды не долететь'
H, vy0, T_, up, down, DIST, cl, YAW, COURSE, d_water = choice
xe = DIST*up/T_
print(f'курс {YAW}: вода через {d_water} блоков, посадка на {d_water + 5}, над рельефом не меньше {cl:.1f}')
DIST = float(round(DIST, 1)); xl = DIST*down/T_
print(f'жерло ({fx}, {STAND}, {fz}): Yaw {YAW}, Height {H}, скорость вверх {vy0:.2f}, Distance {DIST}, выход на {xe:.1f}, в воду на {xl:.1f}, земля y={ground}')
# шахта-жерло: ось идёт за траекторией подъёма, радиус от R_BOT внизу до R_TOP у поверхности, сечение — вытянутое и бугристое,
# поворачивается с высотой, через каждые 7–11 блоков пережимы; стенки — базальт, чернокамень, магма, светятся охряные огни
R_BOT, R_TOP, KEEP = 8.0, 2.8, 2.3                        # KEEP: от траектории до стенки не меньше
ta = int(np.argmax(ys)); asc_y = np.array(ys[:ta + 1]); asc_x = DIST*np.arange(ta + 1)/T_
def path_x(h): return float(np.interp(h, asc_y, asc_x))
y_from = STAND + 2; y_top = ground + 3
noise = gaussian_filter(rng.normal(size=(NY, 96)), (2.5, 3), mode='wrap'); noise /= noise.std()
rings = set(); yr = y_from + 6
while yr < y_top - 4: rings.add(yr); yr += int(rng.integers(7, 12))
lining = [gid('minecraft:basalt[axis=y]'), gid('minecraft:blackstone'), gid('minecraft:magma_block'), gid('minecraft:smooth_basalt'), gid('minecraft:tuff')]
froglight = gid('minecraft:ochre_froglight[axis=y]')
carved = lined = lights = 0
shaft_axis = {}
for y in range(y_from, y_top + 1):
    u = (y - y_from)/(y_top - y_from)
    r = R_TOP + (R_BOT - R_TOP)*(1 - u)**1.6
    xp = path_x(y + 0.5 - STAND)
    room = max(0.0, r - KEEP)
    ca = xp + room*0.55*math.sin(0.19*y + 1.3)               # смещение оси вдоль курса и поперёк, траектория остаётся внутри
    cb = room*0.45*math.sin(0.13*y + 4.0)
    ccx, ccz = fx + COURSE[0]*ca - COURSE[1]*cb, fz + COURSE[1]*ca + COURSE[0]*cb
    px, pz = fx + COURSE[0]*xp, fz + COURSE[1]*xp
    shaft_axis[y] = (ccx, ccz, r)
    phi = 0.08*y; ring = 1.3 if y in rings or y - 1 in rings else 0.0
    rr = int(r*1.5) + 4
    for i in range(int(ccx) - rr - ox, int(ccx) + rr - ox + 1):
        for j in range(int(ccz) - rr - oz, int(ccz) + rr - oz + 1):
            dx, dz = i + ox + .5 - ccx, j + oz + .5 - ccz
            d = math.hypot(dx, dz); th = math.atan2(dz, dx); a2 = int((th + math.pi)/(2*math.pi)*96) % 96
            lim = r*(1 + 0.22*math.cos(2*(th - phi)) + 0.13*noise[yi(y), a2]) - ring
            near = math.hypot(i + ox + .5 - px, j + oz + .5 - pz) <= KEEP
            if lon[i, j] < 1.5 and abs(lat[i, j]) < 14: continue          # тоннель и обрыв под ним не трогать
            cur = vol[i, yi(y), j]
            if d <= lim or near:
                if cur != air: vol[i, yi(y), j] = air; carved += 1
            elif d <= lim + 1.7 and cur != air and base(cur) not in ('bedrock',):
                q = rng.random(); mag = 0.25*(1 - u) + 0.05
                vol[i, yi(y), j] = lining[2] if q < mag else lining[0] if q < 0.55 else lining[1] if q < 0.8 else lining[3] if q < 0.92 else lining[4]
                lined += 1
                if y in rings and d <= lim + 0.9 and a2 % 19 == 0: vol[i, yi(y), j] = froglight; lights += 1
# кромка жерла: невысокий вал из базальта вокруг устья на поверхности
ccx, ccz, r = shaft_axis[y_top]
rim = 0
for i in range(int(ccx) - 9 - ox, int(ccx) + 10 - ox):
    for j in range(int(ccz) - 9 - oz, int(ccz) + 10 - oz):
        d = math.hypot(i + ox + .5 - ccx, j + oz + .5 - ccz)
        if not r + 0.8 < d <= r + 5: continue
        rx_, rz_ = i + ox + .5 - fx, j + oz + .5 - fz
        if rx_*COURSE[0] + rz_*COURSE[1] > 0 and abs(-rx_*COURSE[1] + rz_*COURSE[0]) < 3.5: continue   # по курсу полёта вала нет
        g = surface_at(i, j)
        if g < ground - 8: continue
        h = int(round(2.4*(1 - (d - r - 0.8)/4.2) + 0.6*rng.normal()))
        for y in range(g + 1, g + 1 + max(0, h)):
            if vol[i, yi(y), j] == air: vol[i, yi(y), j] = lining[0] if rng.random() < 0.6 else lining[1]; rim += 1
# над нишей до шахты пусто
for i, j in zip(*np.nonzero(alc)):
    for y in range(STAND, y_from + 1): vol[i, yi(y), j] = air
print(f'жерло: вынуто {carved}, облицовка {lined}, огней {lights}, вал {rim}; радиус {R_BOT} внизу → {R_TOP} у поверхности, пережимов {len(rings)}')
# траектория: в шахте и над скалой
clear_min, inside_min = 1e9, 1e9
for t_ in range(0, down):
    xp = DIST*t_/T_; y_ = STAND + ys[t_]; cs = cells_at(COURSE, xp)
    if t_ >= up: clear_min = min(clear_min, y_ - max(surf(x, z) for x, z in cs))
    else:
        x, z = cs[1]
        for yy in (math.floor(y_), math.floor(y_ + 1.8)):
            if Y0 <= yy < Y1 and vol[x - ox, yi(yy), z - oz] != air: inside_min = -1
print(f'над скалой на пути в воду запас не меньше {clear_min:.1f}; траектория в шахте свободна: {"да" if inside_min > 0 else "НЕТ"}')

# ---------- проверки
leak = False
start = (math.floor(ma[0]) - ox, math.floor(ma[1]) - oz)
def flood(y):
    seen = {start}; st = [start]
    while st:
        i, j = st.pop()
        for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (i + a, j + b)
            if q in seen: continue
            v_ = vol[q[0], yi(y), q[1]]
            if v_ == air or v_ == lava_id:
                seen.add(q); st.append(q)
                if len(seen) > 6000: return None
    return seen
for y in range(BASE + 1, PEAK + 1):
    cells = flood(y)
    if cells is None: print('слой', y, 'больше 6000'); leak = True; continue
    out = [(i, j) for i, j in cells if not (LON0 - 1.5 <= lon[i, j] < LON1 and abs(lat[i, j]) <= HW + 0.5)]
    hit = [k for k in ledge_cells if ledge_top(k) == y and any((i, j) in cells for i, j in ledge_cells[k])]   # верх полки цел (под полкой лава — так задумано)
    print('слой', y, 'клеток', len(cells), 'вне озера', len(out), ('уступы под лавой: ' + str([k for k in ledge_cells if ledge_top(k) < y])) if y > BASE + 1 else '')
    if out or hit: leak = True; print('   утечка', out[:5], hit)
    if (math.floor(mb[0]) - ox, math.floor(mb[1]) - oz) not in cells: print('   маркер B в другом озере'); leak = True
# вылет из тоннеля: прежняя дуга (падение 30 на 20,8 блока) свободна, у тоннеля (lon < 0) ничего не менялось
hits = []
for lo in np.arange(0.3, 22, 0.25):
    y_ = -24 - 30*(lo/20.8)**2
    for la in (-0.4, 0.4):
        x, z = world_xz(lo, la); i, j = math.floor(x) - ox, math.floor(z) - oz
        for yy in range(math.floor(y_), math.floor(y_ + 1.8) + 1):
            if vol[i, yi(yy), j] not in (air, lava_id): hits.append((round(lo, 1), yy, base(vol[i, yi(yy), j])))
print('дуга вылета из тоннеля:', 'свободна' if not hits else hits[:6])
tun = int(((vol != orig) & (lon < 0)[:, None, :] & (np.abs(lat) < 12)[:, None, :] & ((np.arange(NY) + Y0) > -40)[None, :, None]).sum())
print('изменено у тоннеля (lon < 0, выше −40):', tun)
if hits or tun: leak = True
for k, cells in ledge_cells.items():
    if not cells: print('уступ без клеток', k); leak = True
    hd = min(next((y for y in range(ledge_top(k) + 1, 16) if vol[i, yi(y), j] != air), 16) - ledge_top(k) - 1 for i, j in cells)
    if hd < 3: print('низко над уступом', k, hd); leak = True
changed = vol != orig
print('изменено блоков', int(changed.sum()), 'фонарей', len(lamps))
print('скорости волн', SPEEDS)

if preview:
    from PIL import Image
    S_ = 4
    def colour(vv, y):
        if vv == air: return (20, 20, 30)
        if vv == lava_id: return (255, 110, 20)
        if vv == top_id: return (230, 230, 120) if y % 2 else (200, 120, 230)
        if vv == lamp: return (190, 240, 255)
        if vv in (ice_p, ice_b): return (150, 190, 230)
        return (70, 66, 75)
    im = np.zeros((sz*S_, sx*S_, 3), np.uint8)
    for i in range(sx):
        for j in range(sz):
            y = next((y for y in range(-30 + LIFT, -64, -1) if vol[i, yi(y), j] != air), -64)
            c = colour(vol[i, yi(y), j], y)
            if vol[i, yi(y), j] == top_id: c = tuple(int(q) for q in np.array(c)*min(1.0, 0.5 + 0.02*(y + 62)))
            im[j*S_:(j + 1)*S_, i*S_:(i + 1)*S_] = c
    for mx, mz, size, *_ in raft_ents:
        im[(mz - oz)*S_:(mz - oz + size)*S_, (mx - ox)*S_:(mx - ox + size)*S_] = (90, 240, 120)
    for rx, ry, rz, _ in ropes: im[(rz - oz)*S_ + 1:(rz - oz)*S_ + 3, (rx - ox)*S_ + 1:(rx - ox)*S_ + 3] = (255, 255, 255)
    for m in (ma, mb): im[(math.floor(m[1]) - oz)*S_:(math.floor(m[1]) - oz + 1)*S_, (math.floor(m[0]) - ox)*S_:(math.floor(m[0]) - ox + 1)*S_] = (255, 40, 40)
    im[(math.floor(fz) - oz)*S_:(math.floor(fz) - oz + 1)*S_, (math.floor(fx) - ox)*S_:(math.floor(fx) - ox + 1)*S_] = (40, 255, 255)
    top = Image.fromarray(im).crop(((100 - ox)*S_, (-292 - oz)*S_, (200 - ox)*S_, (-226 - oz)*S_))
    # стены озера сбоку (lat = ±10, вид от середины озера: север сверху, юг снизу, вдоль lon 20..70) и разрез жерла по курсу
    secs = []
    for la in (10.0, -10.0):
        a2 = np.zeros(((50 + LIFT)*S_, 74*S_, 3), np.uint8)
        for n, lo in enumerate(np.arange(0, 74, 1.0)):
            x, z = world_xz(lo, la); i, j = int(math.floor(x)) - ox, int(math.floor(z)) - oz
            for m, y in enumerate(range(-14 + LIFT, -64, -1)): a2[m*S_:(m + 1)*S_, n*S_:(n + 1)*S_] = colour(vol[i, yi(y), j], y)
        secs.append(a2)
    walls = np.vstack([np.pad(s_, ((0, 4), (0, 0), (0, 0))) for s_ in secs])
    S2 = 2; ytop = min(Y1, y_top + 12)
    vent = np.zeros(((ytop - (STAND - 4))*S2, 60*S2, 3), np.uint8)
    for n, dz in enumerate(range(-50, 10)):
        i, j = cells_at(COURSE, -dz)[1]; i -= ox; j -= oz
        if not (0 <= i < sx and 0 <= j < sz): continue
        for m, y in enumerate(range(ytop - 1, STAND - 5, -1)):
            vv = vol[i, yi(y), j]
            c = (60, 30, 20) if vv in lining[:3] or vv == lining[3] else (255, 200, 80) if vv == froglight else colour(vv, y)
            if vv == air and y > ground: c = (40, 60, 90)
            vent[m*S2:(m + 1)*S2, n*S2:(n + 1)*S2] = c
    for t_ in range(0, down + 1):
        n = int(round(-DIST*t_/T_)) + 50; m = ytop - 1 - int(STAND + ys[t_])
        if 0 <= n < 60 and 0 <= m*S2 < vent.shape[0]: vent[m*S2:(m + 1)*S2, n*S2:(n + 1)*S2] = (255, 255, 255)
    vh = max(walls.shape[0], vent.shape[0])
    sec = np.hstack([np.pad(walls, ((0, vh - walls.shape[0]), (0, 8), (0, 0))), np.pad(vent, ((0, vh - vent.shape[0]), (0, 0), (0, 0)))])
    Wd = max(top.size[0], sec.shape[1])
    t_ = np.array(top); out_ = np.zeros((t_.shape[0] + sec.shape[0] + 8, Wd, 3), np.uint8)
    out_[:t_.shape[0], :t_.shape[1]] = t_; out_[t_.shape[0] + 8:, :sec.shape[1]] = sec
    Image.fromarray(out_).save(preview); print('предпросмотр', preview)
if clear_min < 2.5 or inside_min < 0: leak = True
if leak: raise SystemExit('проверки не прошли: в мир не пишу')
if not backup:
    print('без --write мир не изменён'); sys.exit()

# ---------- запись
os.makedirs(os.path.join(backup, 'region')); os.makedirs(os.path.join(backup, 'entities'))
for p in {v_[0] for v_ in chunks.values()}: shutil.copy2(p, os.path.join(backup, 'region'))
touched = 0
for (cx, cz), (p, nm, c) in chunks.items():
    a, b = (cx - cx0)*16, (cz - cz0)*16
    if not changed[a:a + 16, :, b:b + 16].any(): continue
    touched += 1
    for sec in c[b'sections'][1][1]:
        y = int.from_bytes(sec[b'Y'][1], 'big', signed=True); bs = sec.get(b'block_states')
        sec.pop(b'SkyLight', None); sec.pop(b'BlockLight', None)
        if bs is None or not Y0 <= y*16 < Y1: continue
        oy = y*16 - Y0
        if not changed[a:a + 16, oy:oy + 16, b:b + 16].any(): continue
        arr = vol[a:a + 16, oy:oy + 16, b:b + 16].transpose(1, 2, 0).reshape(4096)
        u, inv = np.unique(arr, return_inverse=True)
        bs[1][b'palette'] = (9, (10, [mca.state(lst[i]) for i in u]))
        if len(u) == 1: bs[1].pop(b'data', None)
        else: bs[1][b'data'] = (12, mca.pack(inv, len(u)))
    c[b'isLightOn'] = (1, b'\0')
    leaf = [i for i in range(len(lst)) if base(i).endswith('_leaves')]
    noblock = [i for i in range(len(lst)) if base(i) in ('air', 'vine', 'tall_grass', 'short_grass', 'fern', 'large_fern', 'cocoa', 'snow', 'dead_bush') or base(i).endswith('_spike')]
    fluid = [i for i in range(len(lst)) if base(i) in ('water', 'lava')]
    w_ = {t: np.zeros(math.ceil(256/7), np.uint64) for t in (0, 1, 2, 3)}
    for lx in range(16):
        for lz in range(16):
            col = vol[a + lx, :, b + lz]; n = col != 0; mb_ = ~np.isin(col, noblock); fl = np.isin(col, fluid)
            hs = []
            for m in (n, mb_ | fl, (mb_ & ~np.isin(col, leaf)) | fl, mb_ & ~fl):
                idx = np.nonzero(m)[0]; hs.append(int(idx[-1]) + 1 + (Y0 - (-64)) if len(idx) else 0)
            q_ = lz*16 + lx
            for t in (0, 1, 2, 3): w_[t][q_//7] |= np.uint64(hs[t] << ((q_ % 7)*9))
    for nmk, t in {b'WORLD_SURFACE': 0, b'MOTION_BLOCKING': 1, b'MOTION_BLOCKING_NO_LEAVES': 2, b'OCEAN_FLOOR': 3}.items():
        c[b'Heightmaps'][1][nmk] = (12, w_[t].astype('>u8').tobytes())
    regs[p].put(cx, cz, mca.dump(nm, c))
for r in regs.values(): r.save()
# сущности: рамка из копии, без жерл, приливов, волн и плотов; новые
dv = next(iter(chunks.values()))[2][b'DataVersion']
dbl = lambda x: struct.pack('>d', x); flt = lambda x: (5, struct.pack('>f', float(x))); int_ = lambda x: (3, struct.pack('>i', int(x)))
DROP = {b'gameplaymod:launcher', b'gameplaymod:lava_tide', b'gameplaymod:lava_wave', b'gameplaymod:raft'}
ents = {}
for cx in range(cx0, cx1 + 1):
    for cz in range(cz0, cz1 + 1):
        p = ENT + f'r.{cx >> 5}.{cz >> 5}.mca'
        if not os.path.exists(p): continue
        er = region(eregs, p)
        if not os.path.exists(os.path.join(backup, 'entities', os.path.basename(p))): shutil.copy2(p, os.path.join(backup, 'entities'))
        b = region(src_e, os.path.join(restore, 'entities', os.path.basename(p))).raw(cx, cz)
        if b is None: b = er.raw(cx, cz)
        if b is None: continue
        nm, c = mca.parse(b)
        if c[b'Entities'][1][0] == 0: c[b'Entities'] = (9, (10, c[b'Entities'][1][1]))
        before = len(c[b'Entities'][1][1])
        def keep(e):
            if e[b'id'][1] in DROP: return False
            if e[b'id'][1] in (b'gameplaymod:tarzan_rope', b'gameplaymod:calm_rope'):                 # прежние верёвки пещеры: в озере висят только новые
                lo_, la_e = lonlat(struct.unpack('>d', e[b'AX'][1])[0], struct.unpack('>d', e[b'AZ'][1])[0])
                if 13 <= lo_ < 75 and abs(la_e) < 22: return False
                if 0 <= lo_ < 13 and abs(la_e) < 22:                         # прежние у выхода из тоннеля: та же верёвка, но без размаха и до плотов
                    e[b'id'] = (8, b'gameplaymod:calm_rope')
                    ay = struct.unpack('>d', e[b'AY'][1])[0]; L_ = round(ay - DESCENT_END, 1)
                    e[b'Length'] = (5, struct.pack('>f', L_))
                    pos = e[b'Pos'][1][1]
                    e[b'Pos'] = (9, (6, [pos[0], dbl(ay - L_), pos[2]])); e[b'Motion'] = (9, (6, [dbl(0.0)]*3))
                    print('прежняя тарзанка', round(lo_, 1), round(la_e, 1), 'длина', L_)
            return True
        c[b'Entities'] = (9, (10, [e for e in c[b'Entities'][1][1] if keep(e)]))
        if before != len(c[b'Entities'][1][1]): print('чанк', cx, cz, 'убрано сущностей', before - len(c[b'Entities'][1][1]))
        ents[(cx, cz)] = (p, nm, c)
def add(entity, x, y, z):
    cx, cz = int(math.floor(x)) >> 4, int(math.floor(z)) >> 4; p = ENT + f'r.{cx >> 5}.{cz >> 5}.mca'
    if (cx, cz) not in ents:
        if not os.path.exists(os.path.join(backup, 'entities', os.path.basename(p))) and os.path.exists(p): shutil.copy2(p, os.path.join(backup, 'entities'))
        ents[(cx, cz)] = (p, b'', {b'Position': (11, struct.pack('>ii', cx, cz)), b'DataVersion': dv, b'Entities': (9, (10, []))})
    e = {b'Pos': (9, (6, [dbl(x), dbl(y), dbl(z)])), b'Motion': (9, (6, [dbl(0.0)]*3)), b'Rotation': (9, (5, [struct.pack('>f', 0.0)]*2)), b'UUID': (11, uuid.uuid4().bytes)}
    e.update(entity); ents[(cx, cz)][2][b'Entities'][1][1].append(e)
for w in range(1, WAVES + 1):
    m, yaw, ln = (ma, YAW_A, LENS[w]) if w % 2 else (mb, YAW_B, LENS[w])
    add({b'id': (8, b'gameplaymod:lava_wave'), b'Group': (8, f'lava{w}'.encode()), b'Yaw': flt(yaw), b'Length': flt(ln), b'Height': flt(HEIGHT), b'Speed': flt(SPEEDS[w - 1]),
         b'HalfWidth': flt(HW + 1), b'PeakY': int_(PEAK), b'Rise': int_(RISE), b'Interval': int_(200), b'Delay': int_(0), b'Radius': flt(64), b'Launch': (1, b'\0')}, m[0], BASE + 1, m[1])
blocks = {1: 'minecraft:polished_basalt[axis=y]', 2: 'minecraft:basalt[axis=y]'}
for mx, mz, size, *_ in raft_ents:
    add({b'id': (8, b'gameplaymod:raft'), b'SX': int_(size), b'SY': int_(1), b'SZ': int_(size), b'Palette': (9, (10, [mca.state(blocks[size])])),
         b'Cells': (11, struct.pack(f'>{size*size}i', *([0]*(size*size))))}, mx + size/2, BASE + 8/9 - 0.3, mz + size/2)
for rx, ry, rz, L_ in ropes:
    ax_, ay_, az_ = rx + 0.5, ry - 0.05, rz + 0.5
    add({b'id': (8, b'gameplaymod:calm_rope'), b'AX': (6, dbl(ax_)), b'AY': (6, dbl(ay_)), b'AZ': (6, dbl(az_)), b'Length': (5, struct.pack('>f', float(L_))),
         b'BX': (3, struct.pack('>i', rx)), b'BY': (3, struct.pack('>i', ry)), b'BZ': (3, struct.pack('>i', rz)), b'Attached': (3, struct.pack('>i', 0))}, ax_, ay_ - L_, az_)
add({b'id': (8, b'gameplaymod:launcher'), b'Yaw': flt(YAW), b'Distance': flt(DIST), b'Height': flt(H)}, fx, STAND, fz)
for (cx, cz), (p, nm, c) in ents.items(): region(eregs, p).put(cx, cz, mca.dump(nm, c))
for r in eregs.values(): r.save()
print('записано чанков', touched, 'копия', backup)
