#!/usr/bin/env python3
"""Область LevelMod «редактирование» (западный склон главной горы к реке): положе склон, снег выше 117, скала и мох, берег озерца с ивами.

Запуск: edit_area.py <папка мира> [--write <новая папка копии>] [--preview файл.png], игра закрыта. Без --write мир не меняется, только отчёт.
С --write вся папка region копируется в новую папку (сущности не трогаются), потом пишутся блоки. Область читается из data/levelmod/levels.dat
по имени AREA (многоугольник из LevelMod). Повторный запуск поверх готового результата не нужен: сначала вернуть копию.

1. Склон. Крутизна поверхности внутри области ограничена STEP блоков на блок (по осям). Срез и досыпка делаются поровну: поверхность =
   среднее нижней и верхней огибающей (нижняя срезает, верхняя досыпает), чтобы пик менялся меньше. Столбцы с открытой водой (родничок вверху,
   озерцо и река внизу) и коробка водопада остаются как есть; изменения плавно сходят на нет за BLEND блоков от них и от границы области.
   Над каналом воды внутри горы (он идёт к водопаду) поверхность не опускается ниже его свода + 3. Деревья на изменённых столбцах снимаются.
2. Скала (кадр 1 референсов). Новая и открытая порода внутри области — бежево-серая смесь tuff / andesite / stone / sandstone / calcite по
   гладкому шуму; на горизонтальных гранях ниже снега пятна moss_block с травой.
3. Снег. От y≈SNOW_FROM (граница волнистая) каждый блок породы, земли и травы, который касается воздуха (включая отвесные стены и потолки
   навесов), становится snow_block; растения на нём убираются.
4. Берег озерца (кадры 2–3). Вдоль открытой воды внизу узкая песчаная отмель на уровне воды, камыш (sugar_cane) на кромке, кувшинки на воде,
   несколько ив: ствол oak_log, плоская крона oak_leaves (persistent) и свисающие лианы.
Свет в затронутых чанках сбрасывается (игра посчитает при загрузке), карты высот пересчитаны.
"""
import gzip, math, os, shutil, struct, sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca
import snow_and_clearing as S

NY, YMIN = S.NY, S.Y_MIN
AREA = 'редактирование'
STEP = 2.0                  # блоков высоты на блок, по осям
BLEND = 10.0                # блоков, на которых правка сходит на нет у границы и у воды
WATER_BOX = (266, 279, -126, -107)    # водопад и его брызги: x0, x1, z0, z1 (включительно у x0, z0; до x1, z1)
PROT_WATER = 4              # защитный пояс вокруг открытой воды, блоков
SNOW_FROM, SNOW_WAVE = 117, 3.5
MARGIN = 16
SEED = 2611081
STONE_FAM = ('stone', 'andesite', 'tuff', 'calcite', 'granite', 'diorite', 'sandstone')
SNOWABLE = STONE_FAM + ('dirt', 'grass_block', 'coarse_dirt', 'podzol', 'gravel', 'moss_block', 'sand', 'packed_ice', 'blue_ice', 'ice')
VEG = S.PLANTS + ('sugar_cane', 'lily_pad')

world = sys.argv[1]
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
backup, preview = arg('--write'), arg('--preview')
REG = os.path.join(world, 'dimensions/minecraft/overworld/region')

# ---------------------------------------------------------------- область
_, lroot = mca.parse(gzip.open(os.path.join(world, 'data/levelmod/levels.dat')).read())
A = lroot[b'data'][1][b'location_areas'][1][AREA.encode()][1]
poly = [tuple(struct.unpack('>d', c)[0] for c in p[1]) for p in A[b'loops'][1][1][0][1]]
px, pz = [p[0] for p in poly], [p[1] for p in poly]


def inside(x, z):
    c = False
    for (x1, z1), (x2, z2) in zip(poly, poly[1:] + poly[:1]):
        if (z1 > z) != (z2 > z) and x < (x2 - x1)*(z - z1)/(z2 - z1) + x1: c = not c
    return c


bx0, bx1 = (int(min(px)) - MARGIN) >> 4, (int(max(px)) + MARGIN) >> 4
bz0, bz1 = (int(min(pz)) - MARGIN) >> 4, (int(max(pz)) + MARGIN) >> 4
OX, OZ = bx0*16, bz0*16
sx, sz = (bx1 - bx0 + 1)*16, (bz1 - bz0 + 1)*16
print(f'область «{AREA}»: x {min(px):.0f}..{max(px):.0f}, z {min(pz):.0f}..{max(pz):.0f}; рамка чтения x {OX}..{OX + sx}, z {OZ}..{OZ + sz}')

# ---------------------------------------------------------------- чтение
idx = {'minecraft:air': 0}
NEW = ['minecraft:snow_block', 'minecraft:moss_block', 'minecraft:sand', 'minecraft:stone', 'minecraft:andesite', 'minecraft:tuff', 'minecraft:sandstone',
       'minecraft:calcite', 'minecraft:oak_log[axis=y]', 'minecraft:oak_log[axis=x]', 'minecraft:oak_log[axis=z]',
       'minecraft:oak_leaves[distance=1,persistent=true,waterlogged=false]', 'minecraft:sugar_cane[age=0]', 'minecraft:lily_pad',
       'minecraft:short_grass', 'minecraft:fern']
VINE = {}
for d in ('east', 'north', 'south', 'west'):
    VINE[d] = 'minecraft:vine[east=%s,north=%s,south=%s,up=false,west=%s]' % tuple('true' if d == k else 'false' for k in ('east', 'north', 'south', 'west'))
vol = np.zeros((sx, NY, sz), np.uint16); chunks = {}; REGS = {}
for cx in range(bx0, bx1 + 1):
    for cz in range(bz0, bz1 + 1):
        p = os.path.join(REG, f'r.{cx >> 5}.{cz >> 5}.mca')
        if p not in REGS: REGS[p] = mca.Region(p)
        raw = REGS[p].raw(cx, cz)
        if raw is None: continue
        nm, c = mca.parse(raw); chunks[(cx, cz)] = (p, nm, c)
        names, v = S._volume(c)
        mp = np.array([idx.setdefault(k, len(idx)) for k in names])
        vol[(cx - bx0)*16:(cx - bx0 + 1)*16, :, (cz - bz0)*16:(cz - bz0 + 1)*16] = mp[v]
for k in NEW + list(VINE.values()): idx.setdefault(k, len(idx))
nm_list = [None]*len(idx)
for k, i in idx.items(): nm_list[i] = k
ID = lambda k: idx[k]
orig = vol.copy()
bn = np.array([S.bname(k) for k in nm_list])


def cat(names): return np.isin(bn, names)[vol]


AIR, WATER = cat(['air', 'cave_air', 'void_air']), cat(['water'])
LEAF = np.array([b.endswith('leaves') for b in bn])[vol]; LOG = np.array([b.endswith('_log') for b in bn])[vol]
PLANT = cat(list(VEG)); VINES = cat(['vine'])
GROUND = ~AIR & ~WATER & ~LEAF & ~LOG & ~PLANT
G = np.where(GROUND.any(1), NY - 1 - np.argmax(GROUND[:, ::-1, :], axis=1), 0)              # индекс верхнего блока земли [x, z]
yi = np.arange(NY)[None, :, None]
below = yi < (G[:, None, :] - 1)
cav = (AIR | WATER) & below & GROUND.any(1)[:, None, :]
CAV_TOP = np.where(cav.any(1), NY - 1 - np.argmax(cav[:, ::-1, :], axis=1), -1)            # свод пустоты под поверхностью (канал воды)

gx, gz = np.meshgrid(np.arange(sx), np.arange(sz), indexing='ij')
IN = np.array([[inside(OX + i + 0.5, OZ + j + 0.5) for j in range(sz)] for i in range(sx)])
# открытая вода: верх воды не ниже земли
Wtop = np.where(WATER.any(1), NY - 1 - np.argmax(WATER[:, ::-1, :], axis=1), -1)
OPEN_W = (Wtop >= G - 0) & (Wtop >= 0)
box = np.zeros((sx, sz), bool)
box[max(WATER_BOX[0] - OX, 0):WATER_BOX[1] - OX, max(WATER_BOX[2] - OZ, 0):WATER_BOX[3] - OZ] = True
prot = ndimage.binary_dilation(OPEN_W, iterations=PROT_WATER) | box
interior = IN & ~prot
print('столбцов в области', int(IN.sum()), ', правится', int(interior.sum()), ', открытая вода', int((OPEN_W & IN).sum()), ', защищено вокруг воды и водопада', int((IN & prot).sum()))

# ---------------------------------------------------------------- 1. склон: сглаживание только там, где перепад больше STEP
def steps(h):
    out = np.zeros(h.shape)
    dx = np.abs(np.diff(h, axis=0)); dz = np.abs(np.diff(h, axis=1))
    out[1:, :] = np.maximum(out[1:, :], dx); out[:-1, :] = np.maximum(out[:-1, :], dx)
    out[:, 1:] = np.maximum(out[:, 1:], dz); out[:, :-1] = np.maximum(out[:, :-1], dz)
    return out


def envelope_raw(h, sources, s):
    """min по опорам (h_j + s * расстояние L1) для всех столбцов; без опор значение огромное. Два прохода пачкой по трём кругам."""
    a = np.where(sources, h, 1e9)
    n, m = a.shape
    for _ in range(3):
        for i in range(n):
            if i > 0: np.minimum(a[i], a[i - 1] + s, out=a[i])
            ai = a[i]
            for j in range(1, m):
                if ai[j - 1] + s < ai[j]: ai[j] = ai[j - 1] + s
            for j in range(m - 2, -1, -1):
                if ai[j + 1] + s < ai[j]: ai[j] = ai[j + 1] + s
        for i in range(n - 2, -1, -1):
            np.minimum(a[i], a[i + 1] + s, out=a[i])
            ai = a[i]
            for j in range(1, m):
                if ai[j - 1] + s < ai[j]: ai[j] = ai[j - 1] + s
            for j in range(m - 2, -1, -1):
                if ai[j + 1] + s < ai[j]: ai[j] = ai[j + 1] + s
    return a


# Срез до откоса STEP:1. Нижняя граница среза не даёт срезать глубже, чем на откос STEP:1 ниже соседних нетронутых столбцов
# (граница области, уступ водопада): так нет ни зубьев, ни стенок. Подсыпки нет: ровный пол и луг внизу остаются как есть.
KAIMA = 20.0                                                                       # срез выходит за границу области на столько блоков и сходит на нет к нетронутой горе
d_out = ndimage.distance_transform_edt(~IN)
free = (d_out <= KAIMA) & ~prot
low_prot = prot & (G < 80 - YMIN)                                                  # дно озерца и реки: не опора для среза (иначе обрыв срежется до воды)
h0 = G.astype(float)
lo = np.minimum(envelope_raw(h0, ~low_prot, STEP), h0)
anchors = ~free & ~low_prot                                                        # нетронутые столбцы: за границей области и защищённые, кроме дна воды
L = -envelope_raw(-h0, anchors, STEP)
Gf = np.maximum(lo, np.minimum(L, h0))
floor = np.where(CAV_TOP >= 0, CAV_TOP + 3, 0).astype(float)                       # не вскрыть канал воды сверху
its = 0
G2 = np.where(free, np.rint(np.maximum(Gf, floor)), G).astype(int)
G2 = np.minimum(G2, G)                                                             # только срез
for _ in range(8):                                                                 # одиночные шипы
    pad = np.pad(G2, 1, mode='edge')
    nb_max = np.maximum.reduce([pad[2:, 1:-1], pad[:-2, 1:-1], pad[1:-1, 2:], pad[1:-1, :-2]])
    spike = free & (G2 > nb_max + STEP)
    if not spike.any(): break
    G2 = np.where(spike, nb_max + 1, G2)
interior = free
delta = G2 - G
if os.environ.get('EDIT_DEBUG'): np.savez(os.environ['EDIT_DEBUG'], G=G, G2=G2, prot=prot, interior=interior, IN=IN, OPEN_W=OPEN_W, OX=OX, OZ=OZ, CAV=CAV_TOP)
sb, sa = steps(G.astype(float)), steps(G2.astype(float))
print(f'склон: итераций {its}; изменено столбцов {int((delta != 0).sum())}, срезано {int(-delta[delta < 0].sum())} блоков, досыпано {int(delta[delta > 0].sum())}, '
      f'сдвиг в среднем {np.abs(delta[free]).mean():.1f}, максимум {int(np.abs(delta).max())}, пик {int(G2[IN].max()) + YMIN} (был {int(G[IN].max()) + YMIN}); '
      f'доля шагов > {STEP:g} / > {STEP + 1:g}: в области было {float((sb[IN] > STEP).mean()):.2f} / {float((sb[IN] > STEP + 1).mean()):.2f}, стало {float((sa[IN] > STEP).mean()):.2f} / {float((sa[IN] > STEP + 1).mean()):.2f}; среди правимых было {float((sb[free] > STEP).mean()):.2f} / {float((sb[free] > STEP + 1).mean()):.2f}, стало {float((sa[free] > STEP).mean()):.2f} / {float((sa[free] > STEP + 1).mean()):.2f}')


# ---------------------------------------------------------------- породы
def vn(x, z): return S.vnoise(np.asarray(x, float), np.asarray(z, float))


def rock_id(x, y, z):
    """Бежево-серая скала: гладкий шум по x, y, z выбирает от тёмного туфа до светлых sandstone и calcite."""
    n = 0.7*vn(x/11.0 + y/17.0, z/11.0 - y/15.0) + 0.3*vn(x/5.5 + 11.0 + y/9.0, z/5.5 + 5.0)
    n = np.asarray(n)
    out = np.full(n.shape, ID('minecraft:stone'))
    out = np.where(n < -0.4, ID('minecraft:tuff'), out)
    out = np.where((n >= -0.4) & (n < -0.1), ID('minecraft:andesite'), out)
    out = np.where((n >= 0.28) & (n < 0.55), ID('minecraft:sandstone'), out)
    out = np.where(n >= 0.55, ID('minecraft:calcite'), out)
    return out


# убрать растения, деревья на изменённых столбцах
changed_cols = delta != 0
vegmask = (PLANT | LEAF | LOG)
rm_logs = 0
for i, j in zip(*np.nonzero(changed_cols)):
    col = vol[i, :, j]; m = vegmask[i, :, j] & (np.arange(NY) > G[i, j])
    rm_logs += int((LOG[i, :, j] & m).sum())
    col[m] = 0
for i, j in zip(*np.nonzero(changed_cols)):
    g, g2 = int(G[i, j]), int(G2[i, j])
    if g2 > g:
        ys = np.arange(g + 1, g2 + 1)
        vol[i, ys, j] = rock_id(OX + i + 0.5, ys + YMIN, OZ + j + 0.5)
    else:
        vol[i, g2 + 1:g + 1, j] = 0
    vol[i, g2, j] = rock_id(OX + i + 0.5, g2 + YMIN, OZ + j + 0.5)
print('убрано стволов (блоков) на изменённых столбцах:', rm_logs)

# осиротевшая листва рядом со снятыми деревьями: без ствола дальше 6 блоков по листве
if rm_logs:
    LEAF2 = np.array([b.endswith('leaves') for b in bn])[vol]; LOG2 = np.array([b.endswith('_log') for b in bn])[vol]
    st6 = ndimage.generate_binary_structure(3, 1)
    seen = LOG2.copy(); front = LOG2
    for _ in range(6):
        front = ndimage.binary_dilation(front, structure=st6) & LEAF2 & ~seen
        seen |= front
    orphan = LEAF2 & ~seen & ~np.isin(vol, [ID('minecraft:oak_leaves[distance=1,persistent=true,waterlogged=false]')])
    # только листва, чья персистентность не задана (естественная)
    nat = np.array(['persistent=true' not in k and b.endswith('leaves') for k, b in zip(nm_list, bn)])[vol]
    orphan &= nat
    print('осиротевшей листвы убрано:', int(orphan.sum())); vol[orphan] = 0

# пересчёт категорий
AIR = cat(['air', 'cave_air', 'void_air']); WATER = cat(['water'])
LEAF = np.array([b.endswith('leaves') for b in bn])[vol]; LOG = np.array([b.endswith('_log') for b in bn])[vol]
PLANT = cat(list(VEG)); GROUND = ~AIR & ~WATER & ~LEAF & ~LOG & ~PLANT

# ---------------------------------------------------------------- 2. скала и мох на открытой породе области
# граница снега: волнистая
thr = SNOW_FROM + SNOW_WAVE*vn((OX + gx)/12.0 + 0.37, (OZ + gz)/12.0 + 0.11)
Y3 = (np.arange(NY) + YMIN)[None, :, None]
air_like = AIR | PLANT
touch = np.zeros(vol.shape, bool)
for ax in range(3):
    for sgn in (1, -1):
        sh = np.roll(air_like, sgn, axis=ax)
        # края объёма считаем «не воздухом», чтобы не красить рамку
        if ax == 0: sh[0 if sgn == 1 else -1, :, :] = False
        if ax == 1: sh[:, 0 if sgn == 1 else -1, :] = False
        if ax == 2: sh[:, :, 0 if sgn == 1 else -1] = False
        touch |= sh
inarea = np.broadcast_to(IN[:, None, :], vol.shape)
stonef = np.isin(bn, list(STONE_FAM))[vol]
re_rock = stonef & touch & inarea & ~np.broadcast_to(prot[:, None, :], vol.shape)
ri = np.nonzero(re_rock)
vol[ri] = rock_id(OX + ri[0] + 0.5, ri[1] + YMIN + 0.0, OZ + ri[2] + 0.5)
print('открытой породы перекрашено в бежево-серую:', int(re_rock.sum()))
# мох: горизонтальные грани (воздух над породой), ниже снега, пятнами
top_open = stonef & np.roll(air_like, -1, axis=1) & inarea & (Y3 < thr[:, None, :] - 2) & ~np.broadcast_to(prot[:, None, :], vol.shape)
mossn = vn((OX + gx)/5.0 + 40.0, (OZ + gz)/5.0 - 17.0)
moss = top_open & (mossn[:, None, :] > 0.55)
vol[moss] = ID('minecraft:moss_block')
mi = np.argwhere(moss)
rng = np.random.default_rng(SEED)
for i, y, j in mi:
    if y + 1 < NY and AIR[i, y + 1, j] and rng.random() < 0.55:
        vol[i, y + 1, j] = ID('minecraft:short_grass') if rng.random() < 0.8 else ID('minecraft:fern')
print('мха: блоков', int(moss.sum()))

# ---------------------------------------------------------------- 3. снег от SNOW_FROM на всё, что касается воздуха
AIR = cat(['air', 'cave_air', 'void_air']); PLANT = cat(list(VEG)); WATER = cat(['water'])
LEAF = np.array([b.endswith('leaves') for b in bn])[vol]; LOG = np.array([b.endswith('_log') for b in bn])[vol]
air_like = AIR | PLANT
touch = np.zeros(vol.shape, bool)
for ax in range(3):
    for sgn in (1, -1):
        sh = np.roll(air_like, sgn, axis=ax)
        if ax == 0: sh[0 if sgn == 1 else -1, :, :] = False
        if ax == 1: sh[:, 0 if sgn == 1 else -1, :] = False
        if ax == 2: sh[:, :, 0 if sgn == 1 else -1] = False
        touch |= sh
snowable = np.isin(bn, list(SNOWABLE))[vol]
snow = snowable & touch & inarea & (Y3 >= thr[:, None, :])
vol[snow] = ID('minecraft:snow_block')
above = np.roll(snow, 1, axis=1); above[:, 0, :] = False
rmpl = PLANT & above
vol[rmpl] = 0
print('снега: блоков', int(snow.sum()), ', растений на них убрано', int(rmpl.sum()))

# ---------------------------------------------------------------- 4. берег озерца
GROUND = ~cat(['air', 'cave_air', 'void_air']) & ~cat(['water']) & ~(np.array([b.endswith('leaves') for b in bn])[vol]) \
    & ~(np.array([b.endswith('_log') for b in bn])[vol]) & ~cat(list(VEG))
G3 = np.where(GROUND.any(1), NY - 1 - np.argmax(GROUND[:, ::-1, :], axis=1), 0)
WAT = cat(['water'])
Wt = np.where(WAT.any(1), NY - 1 - np.argmax(WAT[:, ::-1, :], axis=1), -1)
pond = (Wt >= 0) & (Wt >= G3) & IN & (Wt + YMIN < 100) & ~box       # открытая вода внизу
dwater = ndimage.distance_transform_edt(~pond)
rngs = np.random.default_rng(SEED + 1)
sand_n = reeds = pads = 0
for i, j in zip(*np.nonzero(IN & ~pond & (dwater > 0) & (dwater <= 3.0) & ~box)):
    # уровень воды рядом
    d = int(math.ceil(dwater[i, j] - 1e-6))
    nb = [(i + a, j + b) for a in (-4, -3, -2, -1, 0, 1, 2, 3, 4) for b in (-4, -3, -2, -1, 0, 1, 2, 3, 4) if 0 <= i + a < sx and 0 <= j + b < sz and pond[i + a, j + b]]
    if not nb: continue
    wl = int(np.median([Wt[a, b] for a, b in nb]))
    g = int(G3[i, j])
    if g < wl - 1 or g > wl + 4: continue
    if vn((OX + i)/6.0, (OZ + j)/6.0) < -0.15: continue                              # часть берега остаётся травяной
    target = wl + (d - 1)                                                            # отмель: вплотную к воде на её уровне, дальше плавно вверх
    if target >= g + 1 and d > 1: continue
    tgt = min(g, target) if g > target else target
    for y in range(min(g, tgt) + 1, max(g, tgt) + 1): vol[i, y, j] = 0 if tgt < g else ID('minecraft:sand')
    for y in range(tgt - 1, tgt + 1): vol[i, y, j] = ID('minecraft:sand')
    vol[i, tgt, j] = ID('minecraft:sand')
    sand_n += 1
    # плоскость выше: убрать растения
    for y in range(tgt + 1, tgt + 4):
        if PLANT[i, y, j] or LEAF[i, y, j]: vol[i, y, j] = 0
    if d == 1 and rngs.random() < 0.55:
        h = int(rngs.integers(2, 4))
        for k in range(1, h + 1):
            if vol[i, tgt + k, j] == 0: vol[i, tgt + k, j] = ID('minecraft:sugar_cane[age=0]'); reeds += 1
for i, j in zip(*np.nonzero(pond)):
    if dwater[i, j] == 0 and vn((OX + i)/3.0 + 7.0, (OZ + j)/3.0) > 0.45 and rngs.random() < 0.35 and vol[i, Wt[i, j] + 1, j] == 0:
        # кувшинки не у самого водопада
        if not (WATER_BOX[0] - 4 <= OX + i <= WATER_BOX[1] + 4 and WATER_BOX[2] - 4 <= OZ + j <= WATER_BOX[3] + 4):
            vol[i, Wt[i, j] + 1, j] = ID('minecraft:lily_pad'); pads += 1


def willow(i, j, g, rng):
    """Ива: слегка наклонный ствол из oak_log, плоская крона и свисающие лианы."""
    h = int(rng.integers(6, 9)); lx, lz = rng.integers(-1, 2), rng.integers(-1, 2)
    x, z = i, j; y = g
    for k in range(1, h + 1):
        if k > 2 and k % 2 == 1:
            x += int(np.sign(lx)); z += int(np.sign(lz))
        if 0 <= x < sx and 0 <= z < sz and 0 <= y + k < NY and (vol[x, y + k, z] == 0 or PLANT[x, y + k, z]):
            vol[x, y + k, z] = ID('minecraft:oak_log[axis=y]')
    cy = y + h; cx_, cz_ = x, z
    leaf = ID('minecraft:oak_leaves[distance=1,persistent=true,waterlogged=false]')
    for dy, rad in ((0, 2.2), (-1, 3.6), (-2, 3.8), (-3, 3.0)):
        r = int(math.ceil(rad))
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                if a*a + b*b <= rad*rad + 0.4 and rng.random() < 0.9:
                    xx, yy, zz = cx_ + a, cy + dy, cz_ + b
                    if 0 <= xx < sx and 0 <= zz < sz and vol[xx, yy, zz] == 0: vol[xx, yy, zz] = leaf
    # лианы по краю нижнего слоя крон
    dirs = (('east', 1, 0), ('west', -1, 0), ('south', 0, 1), ('north', 0, -1))
    for a in range(-4, 5):
        for b in range(-4, 5):
            xx, yy, zz = cx_ + a, cy - 3, cz_ + b
            if not (0 <= xx < sx and 0 <= zz < sz) or vol[xx, yy, zz] != leaf: continue
            for name, dx, dz in dirs:
                ox_, oz_ = xx - dx, zz - dz        # клетка снаружи (лиана на грани, обращённой к листве)
                if 0 <= ox_ < sx and 0 <= oz_ < sz and vol[ox_, yy, oz_] == 0 and rng.random() < 0.55:
                    # грань смотрит на листву: к блоку со стороны (dx, dz)
                    face = {'east': 'east', 'west': 'west', 'south': 'south', 'north': 'north'}[name]
                    # лиана в клетке (ox_, oz_) прилегает гранью к листве, лежащей на +(dx, dz): направление грани = name
                    ln = int(rng.integers(3, 8))
                    for k in range(ln):
                        if vol[ox_, yy - k, oz_] == 0: vol[ox_, yy - k, oz_] = ID(VINE[face])
                        else: break
    return h


# ивы: на суше 4..9 блоков от воды, ровно, поодаль друг от друга и от границы
cand = [(i, j) for i, j in zip(*np.nonzero(IN & ~pond & ~prot & (dwater >= 4) & (dwater <= 9) & (G3 <= Wt.max() + 6)))]
rngs.shuffle(cand)
placed = []
for i, j in cand:
    if len(placed) >= 5: break
    g = int(G3[i, j])
    if g + YMIN > 85: continue
    if any((i - a)**2 + (j - b)**2 < 10**2 for a, b in placed): continue
    if inside(OX + i + 0.5 - 6, OZ + j + 0.5) is False or inside(OX + i + 6.5, OZ + j + 0.5) is False: continue
    if vol[i, g + 1, j] not in (0,) and not PLANT[i, g + 1, j]: continue
    willow(i, j, g, rngs); placed.append((i, j))
print(f'берег: песка {sand_n} столбцов, камыша {reeds}, кувшинок {pads}, ив {len(placed)}', [(OX + a, OZ + b) for a, b in placed])

# ---------------------------------------------------------------- отчёт и запись
chg = vol != orig
print('всего изменено блоков:', int(chg.sum()))
if preview:
    from PIL import Image, ImageDraw
    ys0, ys1 = 56, 206; Sx = 4
    xs_ = (262, 282, 292, 300)

    def side(v, x):
        sl = v[x - OX, :, :]; h = sl.shape[1]
        img = np.zeros((ys1 - ys0, h, 3), np.uint8); img[:] = (135, 185, 235)
        col = {}
        for k, b in enumerate(nm_list): col[k] = (130, 125, 120)
        for y in range(ys0, ys1):
            row = ys1 - 1 - y
            ids = sl[y - YMIN]
            for z in range(h):
                b = bn[ids[z]]
                if b in ('air', 'cave_air', 'void_air'): continue
                c = (130, 125, 120)
                if b == 'water': c = (30, 90, 230)
                elif b == 'snow_block': c = (245, 245, 250)
                elif b.endswith('leaves') or b.endswith('_log') or b == 'vine': c = (40, 120, 40)
                elif b in ('grass_block', 'moss_block'): c = (80, 150, 60)
                elif b == 'sand': c = (225, 210, 150)
                elif b in ('sandstone', 'calcite'): c = (200, 190, 160)
                elif b == 'tuff': c = (100, 105, 95)
                elif b == 'sugar_cane': c = (130, 190, 90)
                img[row, z] = c
        return Image.fromarray(img).resize((h*Sx, (ys1 - ys0)*Sx), Image.NEAREST)
    rows = []
    for x in xs_:
        a_, b_ = side(orig, x), side(vol, x)
        im = Image.new('RGB', (a_.width*2 + 8, a_.height), (20, 20, 20)); im.paste(a_, (0, 0)); im.paste(b_, (a_.width + 8, 0)); rows.append(im)
    out = Image.new('RGB', (rows[0].width, sum(r.height + 14 for r in rows)), (20, 20, 20)); d = ImageDraw.Draw(out); y = 0
    for x, r in zip(xs_, rows):
        d.text((4, y), f'x={x}: слева было, справа стало (z {OZ}..{OZ + sz}, y {ys0}..{ys1})', fill=(255, 255, 0)); out.paste(r, (0, y + 12)); y += r.height + 14
    out.save(preview); print('предпросмотр:', preview)
if os.environ.get('EDIT_DUMP'): np.savez_compressed(os.environ['EDIT_DUMP'], vol=vol, orig=orig, names=np.array(nm_list, dtype=object), OX=OX, OZ=OZ)
if not backup:
    print('без --write мир не изменён'); sys.exit()

os.makedirs(backup, exist_ok=False)
for f in os.listdir(REG):
    if f.endswith('.mca'): shutil.copy2(os.path.join(REG, f), backup)
print('копия региона:', backup)
touched = 0; lost = []
for (cx, cz), (p, nm, c) in chunks.items():
    have = {int.from_bytes(sec[b'Y'][1], 'big', signed=True) + 4 for sec in c[b'sections'][1][1] if sec.get(b'block_states') is not None}
    a, b = (cx - bx0)*16, (cz - bz0)*16
    miss = {y >> 4 for y in np.nonzero(chg[a:a + 16, :, b:b + 16].any(axis=(0, 2)))[0]} - have
    if miss: lost.append((cx, cz, sorted(miss)))
if lost: print('ВНИМАНИЕ: нет секций для изменений:', lost); sys.exit(1)
for (cx, cz), (p, nm, c) in chunks.items():
    a, b = (cx - bx0)*16, (cz - bz0)*16
    sub = chg[a:a + 16, :, b:b + 16]
    if not sub.any(): continue
    v = vol[a:a + 16, :, b:b + 16]
    changed = {y >> 4 for y in np.nonzero(sub.any(axis=(0, 2)))[0]}
    S.write_sections(c, v, nm_list, changed, True)
    c[b'isLightOn'] = (1, b'\0')
    S.set_heightmaps(c, v, nm_list)
    REGS[p].put(cx, cz, mca.dump(nm, c)); touched += 1
for r in REGS.values(): r.save()
print('чанков записано:', touched)
