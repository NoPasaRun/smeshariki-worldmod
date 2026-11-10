#!/usr/bin/env python3
"""Стена с водопадом, большое озеро и берег по референсам (кадры 1–3) в области «редактирование» (после edit_area.py).

Запуск: refs_wall_lake.py <папка мира> [--write <новая папка копии>] [--dump файл.npz], игра закрыта. Без --write только отчёт. С --write все
файлы region и entities копируются в новую папку. Не идемпотентен: повторный запуск поверх готового сначала вернуть копию.

Что делает (координаты мира, плоскость стены z=WALL_Z, вода озерца y=WL):
1. Стена x 262..286: порода по вертикальным разводам (sandstone / andesite / stone / calcite / tuff, как бежево-серая скала на кадре 1),
   выступы-«валуны» на лицевой стороне, мох на уступах.
2. Свод-пещерка: канал воды, идущий к водопаду, у выхода расширен до 7 блоков и открыт аркой радиуса 5 над водой; водопад на 9 блоков
   шире и в 3 слоя, доходит до воды озера.
3. Озеро расширено на площадку перед стеной (эллипс с неровным краем, глубже к середине), песчаный пляж по кромке, камыш, кувшинки, три
   больших валуна в воде.
4. Ивы (прежние маленькие заменены): толстый наклонный ствол, широкая плоская крона, длинные лианы до воды.
5. Лодка (oak_boat) на озере; снег от y≈117 на новых открытых блоках стены.
Свет в затронутых чанках сбрасывается, карты высот пересчитываются.
"""
import math, os, shutil, struct, sys, uuid
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca
import snow_and_clearing as S

NY, YMIN = S.NY, S.Y_MIN
OX, OZ, SX, SZ = 240, -144, 64, 80                     # рамка чтения: x 240..303, z -144..-65 (границы чанков)
WALL_Z = -116                                           # первый блок породы лицом на юг; ниже по z перед ним воздух и падающая вода
WL = 67                                                 # верх воды озера
PAD_Y = 69
ARCH_X, ARCH_Y0, ARCH_R = 272, 88, 5                    # центр свода, первый воздух над водой, радиус
SNOW_FROM, SNOW_WAVE = 117, 3.5
SEED = 26110803
STONE_FAM = ('stone', 'andesite', 'tuff', 'calcite', 'granite', 'diorite', 'sandstone')

world = sys.argv[1]
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
backup, dump = arg('--write'), arg('--dump')
REG = os.path.join(world, 'dimensions/minecraft/overworld/region')
ENT = os.path.join(world, 'dimensions/minecraft/overworld/entities')

# ---------------------------------------------------------------- чтение
bx0, bx1, bz0, bz1 = OX >> 4, (OX + SX - 1) >> 4, OZ >> 4, (OZ + SZ - 1) >> 4
idx = {'minecraft:air': 0}
vol = np.zeros((SX, NY, SZ), np.uint16); chunks = {}; REGS = {}
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
VINE = {}
for d in ('east', 'north', 'south', 'west'):
    VINE[d] = 'minecraft:vine[east=%s,north=%s,south=%s,up=false,west=%s]' % tuple('true' if d == k else 'false' for k in ('east', 'north', 'south', 'west'))
LEAF_P = 'minecraft:oak_leaves[distance=1,persistent=true,waterlogged=false]'
NEW = ['minecraft:snow_block', 'minecraft:moss_block', 'minecraft:sand', 'minecraft:gravel', 'minecraft:stone', 'minecraft:andesite', 'minecraft:tuff',
       'minecraft:sandstone', 'minecraft:calcite', 'minecraft:water[level=0]', 'minecraft:oak_log[axis=y]', LEAF_P, 'minecraft:sugar_cane[age=0]',
       'minecraft:lily_pad', 'minecraft:short_grass', 'minecraft:fern', 'minecraft:grass_block[snowy=false]'] + list(VINE.values())
for k in NEW: idx.setdefault(k, len(idx))
nm_list = [None]*len(idx)
for k, i in idx.items(): nm_list[i] = k
ID = lambda k: idx[k]
orig = vol.copy()
bn = np.array([S.bname(k) for k in nm_list])
rng = np.random.default_rng(SEED)
vn = lambda x, z: S.vnoise(np.asarray(x, float), np.asarray(z, float))
XI = lambda x: int(x) - OX
ZI = lambda z: int(z) - OZ
YI = lambda y: int(y) - YMIN


def inb(x, z): return 0 <= x - OX < SX and 0 <= z - OZ < SZ


def cats():
    air = np.isin(bn, ['air', 'cave_air', 'void_air'])[vol]; water = (bn == 'water')[vol]
    leaf = np.array([b.endswith('leaves') for b in bn])[vol]; log = np.array([b.endswith('_log') for b in bn])[vol]
    plant = np.isin(bn, list(S.PLANTS) + ['sugar_cane', 'lily_pad', 'vine'])[vol]
    return air, water, leaf, log, plant


def ground_top():
    air, water, leaf, log, plant = cats()
    gm = ~air & ~water & ~leaf & ~log & ~plant
    return np.where(gm.any(1), NY - 1 - np.argmax(gm[:, ::-1, :], axis=1), 0) + YMIN


def touch_air(extra=None):
    air, water, leaf, log, plant = cats()
    al = air | plant
    t = np.zeros(vol.shape, bool)
    for ax in range(3):
        for sg in (1, -1):
            sh = np.roll(al, sg, axis=ax)
            if ax == 0: sh[0 if sg == 1 else -1, :, :] = False
            if ax == 1: sh[:, 0 if sg == 1 else -1, :] = False
            if ax == 2: sh[:, :, 0 if sg == 1 else -1] = False
            t |= sh
    return t


def rock_streak(x, y, z):
    """Вертикальные разводы бежево-серой скалы (кадр 1): шум растянут по y."""
    n = 0.78*np.asarray(vn(x/2.3 + 3.0 + 0.0*y, y/18.0 + 0.0*x)) + 0.22*np.asarray(vn(x/6.5 + y/13.0, z/6.5 - y/15.0))
    out = np.full(n.shape, ID('minecraft:stone'))
    out = np.where(n < -0.30, ID('minecraft:tuff'), out)
    out = np.where((n >= -0.30) & (n < -0.08), ID('minecraft:andesite'), out)
    out = np.where((n >= -0.08) & (n < 0.30), ID('minecraft:sandstone'), out)
    out = np.where((n >= 0.30) & (n < 0.5), ID('minecraft:calcite'), out)
    return out


G0 = ground_top()
print('рамка', (OX, OX + SX), (OZ, OZ + SZ), '; стена z=', WALL_Z, '; вода озера y=', WL)

# ---------------------------------------------------------------- 0. убрать прежние маленькие ивы (persistent oak_leaves и их стволы и лианы)
air, water, leaf, log, plant = cats()
leafp = vol == ID(LEAF_P)
if leafp.any():
    near = ndimage.binary_dilation(leafp, structure=np.ones((3, 3, 3)), iterations=2)
    oak = np.isin(bn, ['oak_log'])[vol]
    vines = (bn == 'vine')[vol]
    seed = ndimage.binary_dilation(leafp, structure=ndimage.generate_binary_structure(3, 1)) & vines
    grown = seed.copy()
    for _ in range(14):
        n2 = ndimage.binary_dilation(grown, structure=ndimage.generate_binary_structure(3, 1)) & vines
        if (n2 == grown).all(): break
        grown = n2
    rm = leafp | (near & oak) | grown
    print('прежние ивы убраны блоков:', int(rm.sum())); vol[rm] = 0

# ---------------------------------------------------------------- 1. стена
G0 = ground_top()
bodies = []
# выступы на лицевой стороне (юг): центр на плоскости, y, радиусы по x, y, z; не заходят в падение воды и арку
BULGES = [(264, 78, 2.6, 4.5, 2.4), (266, 100, 3.2, 5.0, 2.8), (279, 82, 3.0, 5.5, 2.6), (283, 99, 3.6, 5.5, 3.2),
          (277, 108, 4.0, 4.5, 3.0), (269, 112, 3.4, 4.0, 2.6), (284, 72, 2.4, 3.5, 2.2), (262, 90, 2.4, 4.0, 2.2)]
added = 0
for cx_, cy_, rx, ry, rz in BULGES:
    for x in range(int(cx_ - rx) - 1, int(cx_ + rx) + 2):
        for y in range(int(cy_ - ry) - 1, int(cy_ + ry) + 2):
            for z in range(int(WALL_Z - rz) - 1, int(WALL_Z + rz) + 2):
                if not inb(x, z): continue
                if ((x - cx_)/rx)**2 + ((y - cy_)/ry)**2 + ((z - WALL_Z)/rz)**2 > 1.0: continue
                if 267 <= x <= 277 and y <= 95: continue                  # место падения воды и арки
                if z < WALL_Z - 1: continue                                  # только выступает наружу, внутрь горы не лезет
                if vol[XI(x), YI(y), ZI(z)] == 0:
                    vol[XI(x), YI(y), ZI(z)] = ID('minecraft:stone'); added += 1
print('выступов стены, блоков:', added)

# арка и канал
carved = 0
for z in range(WALL_Z - 9, WALL_Z + 1):
    for dx in range(-ARCH_R + 1, ARCH_R):
        x = ARCH_X + dx
        top = ARCH_Y0 + int(round(math.sqrt(ARCH_R**2 - dx**2))) - 1
        for y in range(ARCH_Y0, top + 1):
            if vol[XI(x), YI(y), ZI(z)] != 0: carved += 1
            vol[XI(x), YI(y), ZI(z)] = 0
        if abs(dx) <= 3:                                                  # вода канала 7 блоков шириной
            for y in (85, 86, 87): vol[XI(x), YI(y), ZI(z)] = ID('minecraft:water[level=0]')
            if vol[XI(x), YI(84), ZI(z)] in (0,): vol[XI(x), YI(84), ZI(z)] = ID('minecraft:stone')
print('свод: вынуто блоков породы', carved)
# падение воды: x 268..276, три слоя z -115..-113, от кромки (y 87) до зеркала озера
for x in range(ARCH_X - 4, ARCH_X + 5):
    for z in (WALL_Z + 1, WALL_Z + 2, WALL_Z + 3):
        y_low = 70 if z == WALL_Z + 1 else WL + 1
        for y in range(y_low, 88):
            if vol[XI(x), YI(y), ZI(z)] in (0,) or bn[vol[XI(x), YI(y), ZI(z)]] in ('water', 'short_grass', 'fern'):
                vol[XI(x), YI(y), ZI(z)] = ID('minecraft:water[level=0]')
# выступающий козырёк над падением: порода справа и слева от воды на уровне кромки (чтобы вода текла между двумя «плечами»)
for x in (ARCH_X - 5, ARCH_X + 5):
    for z in (WALL_Z + 1, WALL_Z + 2):
        for y in range(84, 90):
            if vol[XI(x), YI(y), ZI(z)] == 0: vol[XI(x), YI(y), ZI(z)] = ID('minecraft:andesite')

# перекраска лицевой породы стены
touch = touch_air()
wz = np.zeros(vol.shape, bool)
wz[XI(258):XI(291), YI(64):YI(140), ZI(-127):ZI(-111)] = True
stonef = np.isin(bn, list(STONE_FAM))[vol]
rr = stonef & touch & wz
ri = np.nonzero(rr)
vol[ri] = rock_streak(OX + ri[0] + 0.5, ri[1] + YMIN + 0.0, OZ + ri[2] + 0.5)
print('лицевой породы стены перекрашено:', int(rr.sum()))
# мох на уступах стены ниже снега
air, water, leaf, log, plant = cats()
top_open = np.isin(bn, list(STONE_FAM))[vol] & np.roll(air, -1, axis=1) & wz
ym = (np.arange(NY) + YMIN)[None, :, None]
moss = top_open & (vn((OX + np.arange(SX))[:, None, None]/4.0 + 5.0 + 0*ym, (OZ + np.arange(SZ))[None, None, :]/4.0 - 3.0 + 0*ym) > 0.35) & (ym < 112)
vol[moss] = ID('minecraft:moss_block')
for i, y, j in np.argwhere(moss):
    if y + 1 < NY and vol[i, y + 1, j] == 0 and rng.random() < 0.6: vol[i, y + 1, j] = ID('minecraft:short_grass') if rng.random() < 0.7 else ID('minecraft:fern')
print('мха на стене:', int(moss.sum()))

# ---------------------------------------------------------------- 2. озеро
G0 = ground_top()
air, water, leaf, log, plant = cats()
Wt = np.where(water.any(1), NY - 1 - np.argmax(water[:, ::-1, :], axis=1) + YMIN, -999)
xs = np.arange(SX) + OX; zs = np.arange(SZ) + OZ
XX, ZZ = np.meshgrid(xs, zs, indexing='ij')
openw = (Wt >= G0) & (Wt <= 70) & (Wt > 0)
pad = (G0 >= 68) & (G0 <= 71)
noise = np.asarray(vn(XX/6.0 + 0.3, ZZ/6.0 - 0.7))
ell = ((XX - 270)/14.0)**2 + ((ZZ + 100)/13.5)**2 + 0.30*noise
lake_new = (ell < 1.0) & pad & (ZZ >= -114) & (ZZ <= -84) & ~openw
allw = openw | lake_new
dist_in = ndimage.distance_transform_edt(allw)
n_lake = 0
for i, j in zip(*np.nonzero(lake_new)):
    bed = WL - 1 - min(3, int(dist_in[i, j]/1.8))
    g = int(G0[i, j])
    for y in range(bed + 1, WL + 1): vol[i, y - YMIN, j] = ID('minecraft:water[level=0]')
    for y in range(WL + 1, g + 1): vol[i, y - YMIN, j] = 0
    vol[i, bed - YMIN, j] = ID('minecraft:sand') if dist_in[i, j] < 3 else ID('minecraft:gravel')
    for y in range(WL + 1, g + 8):                                       # растения и деревья над озером
        v_ = vol[i, y - YMIN, j]
        if v_ != 0 and bn[v_] != 'water': vol[i, y - YMIN, j] = 0
    n_lake += 1
print('озеро: новых столбцов воды', n_lake, ', всего поверхности', int(allw.sum()))
# пляж: сухие столбцы вокруг озера
dist_out = ndimage.distance_transform_edt(~allw)
beach = 0; reeds = 0
for i, j in zip(*np.nonzero((~allw) & (dist_out <= 2.5) & pad & (ZZ > WALL_Z + 1) & (XX > 255) & (XX < 290))):
    d = 1 if dist_out[i, j] <= 1.5 else 2
    top = WL if d == 1 else WL + 1
    g = int(G0[i, j])
    if g < top - 1 or g > top + 3: continue
    if vn((OX + i)/7.0, (OZ + j)/7.0) < -0.2: continue                   # часть берега остаётся травяной
    for y in range(top + 1, g + 8):
        v_ = vol[i, y - YMIN, j]
        if v_ != 0 and bn[v_] != 'water': vol[i, y - YMIN, j] = 0
    for y in range(max(top - 2, YMIN), top + 1): vol[i, y - YMIN, j] = ID('minecraft:sand')
    beach += 1
    if d == 1 and rng.random() < 0.55:
        h = int(rng.integers(2, 4))
        for k in range(1, h + 1): vol[i, top + k - YMIN, j] = ID('minecraft:sugar_cane[age=0]'); reeds += 1
print('пляж: столбцов', beach, ', камыша', reeds)

# кувшинки на воде: пятнами
pads = 0
for i, j in zip(*np.nonzero(allw)):
    if vol[i, WL + 1 - YMIN, j] != 0 or bn[vol[i, WL - YMIN, j]] != 'water': continue
    x, z = OX + i, OZ + j
    if 264 <= x <= 280 and z <= -108: continue                          # у падения воды
    if vn(x/3.5 + 7.0, z/3.5) > 0.35 and rng.random() < 0.5:
        vol[i, WL + 1 - YMIN, j] = ID('minecraft:lily_pad'); pads += 1
print('кувшинок:', pads)


def ellipsoid(cx_, cy_, cz_, rx, ry, rz, mat):
    n = 0
    for x in range(int(cx_ - rx) - 1, int(cx_ + rx) + 2):
        for y in range(int(cy_ - ry) - 1, int(cy_ + ry) + 2):
            for z in range(int(cz_ - rz) - 1, int(cz_ + rz) + 2):
                if inb(x, z) and ((x - cx_)/rx)**2 + ((y - cy_)/ry)**2 + ((z - cz_)/rz)**2 <= 1.0:
                    vol[XI(x), YI(y), ZI(z)] = mat(x, y, z); n += 1
    return n


rockmat = lambda x, y, z: int(rock_streak(np.array([x + 0.5]), np.array([y + 0.0]), np.array([z + 0.5]))[0])
for (bx, by, bz, rx, ry, rz) in ((264.5, 65, -110.5, 3.2, 6.0, 3.0), (283.0, 65, -109.0, 2.8, 5.5, 2.6), (258.5, 66, -100.5, 2.6, 5.0, 2.4), (279.0, 65, -92.0, 2.3, 4.6, 2.2)):
    ellipsoid(bx, by, bz, rx, ry, rz, rockmat)
    # мох на макушке
    for x in range(int(bx - rx), int(bx + rx) + 1):
        for z in range(int(bz - rz), int(bz + rz) + 1):
            col = vol[XI(x), :, ZI(z)]
            ys = np.nonzero(col != 0)[0]
            if len(ys) and ys[-1] + YMIN > WL + 2 and rng.random() < 0.5:
                yy = ys[-1]; vol[XI(x), yy, ZI(z)] = ID('minecraft:moss_block')
                if rng.random() < 0.6: vol[XI(x), yy + 1, ZI(z)] = ID('minecraft:short_grass')

# ---------------------------------------------------------------- 3. ивы
LEAF_ID, LOG_ID = ID(LEAF_P), ID('minecraft:oak_log[axis=y]')


def willow(x0, z0, lean, rg):
    g = int(ground_top()[XI(x0), ZI(z0)])
    h = int(rg.integers(8, 11)); x, z = x0, z0
    for k in range(1, h + 1):
        if k > 2 and k % 2 == 1: x += lean[0]; z += lean[1]
        for dx, dz in ((0, 0), (1, 0), (0, 1), (1, 1)) if k <= 3 else ((0, 0),):
            if inb(x + dx, z + dz) and vol[XI(x + dx), YI(g + k), ZI(z + dz)] in (0,) or (inb(x + dx, z + dz) and bn[vol[XI(x + dx), YI(g + k), ZI(z + dz)]] in ('short_grass', 'fern')):
                vol[XI(x + dx), YI(g + k), ZI(z + dz)] = LOG_ID
    cy = g + h
    for dy, rad in ((1, 2.2), (0, 3.8), (-1, 5.2), (-2, 5.6), (-3, 4.8), (-4, 3.6)):
        r = int(math.ceil(rad))
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                if a*a + b*b <= rad*rad + 0.5 and rg.random() < 0.92 and inb(x + a, z + b):
                    if vol[XI(x + a), YI(cy + dy), ZI(z + b)] == 0: vol[XI(x + a), YI(cy + dy), ZI(z + b)] = LEAF_ID
    # лианы до воды или земли
    hung = 0
    for a in range(-7, 8):
        for b in range(-7, 8):
            xx, zz = x + a, z + b
            if not inb(xx, zz): continue
            for dy in (-2, -3, -4):
                if vol[XI(xx), YI(cy + dy), ZI(zz)] != LEAF_ID: continue
                for name, dx, dz in (('west', 1, 0), ('east', -1, 0), ('north', 0, 1), ('south', 0, -1)):   # лиана справа (dx,dz) от листа смотрит гранью на лист
                    px, pz = xx + dx, zz + dz
                    if inb(px, pz) and vol[XI(px), YI(cy + dy), ZI(pz)] == 0 and rg.random() < 0.7:
                        ln = int(rg.integers(5, 10))
                        for k in range(ln):
                            if vol[XI(px), YI(cy + dy - k), ZI(pz)] == 0: vol[XI(px), YI(cy + dy - k), ZI(pz)] = ID(VINE[name]); hung += 1
                            else: break
                break
    return h, hung


# места ив: ближайшая к цели суха́я клетка площадки в 3..8 блоках от воды; наклон к середине озера
G0 = ground_top()
air, water, leaf, log, plant = cats()
wat_now = water.any(1)
dwat = ndimage.distance_transform_edt(~wat_now)
used = []
for (tx, tz) in ((260, -91), (283, -94), (265, -75)):
    best = None
    for i in range(SX):
        for j in range(SZ):
            x, z = OX + i, OZ + j
            if not (3 <= dwat[i, j] <= 8) or not (68 <= G0[i, j] <= 70) or z < -112 or z > -72: continue
            if any((x - a)**2 + (z - b)**2 < 14**2 for a, b in used): continue
            dd = (x - tx)**2 + (z - tz)**2
            if best is None or dd < best[0]: best = (dd, x, z)
    if best is None: continue
    _, wx, wz_ = best; used.append((wx, wz_))
    ddx, ddz = 271 - wx, -100 - wz_
    lean = (int(np.sign(ddx)), 0) if abs(ddx) >= abs(ddz) else (0, int(np.sign(ddz)))
    h, hung = willow(wx, wz_, lean, rng)
    print(f'ива ({wx}, {wz_}): высота {h}, наклон {lean}, лиан {hung}')

# ---------------------------------------------------------------- 4. снег на новых открытых блоках стены выше SNOW_FROM
touch = touch_air()
thr = SNOW_FROM + SNOW_WAVE*np.asarray(vn((OX + np.arange(SX))[:, None]/12.0 + 0.37, (OZ + np.arange(SZ))[None, :]/12.0 + 0.11))
snowable = np.isin(bn, list(STONE_FAM) + ['dirt', 'grass_block', 'gravel', 'moss_block'])[vol]
snow = snowable & touch & (ym >= thr[:, None, :])
snow &= (vol != orig) | np.isin(bn, ['moss_block'])[vol]
vol[snow] = ID('minecraft:snow_block')
print('снега на новых блоках:', int(snow.sum()))

chg = vol != orig
print('всего изменено блоков:', int(chg.sum()))
if dump: np.savez_compressed(dump, vol=vol, orig=orig, names=np.array(nm_list, dtype=object), OX=OX, OZ=OZ)
if not backup:
    print('без --write мир не изменён'); sys.exit()

# ---------------------------------------------------------------- запись
os.makedirs(backup, exist_ok=False)
for sub, d in (('region', REG), ('entities', ENT)):
    os.makedirs(os.path.join(backup, sub))
    for f in os.listdir(d):
        if f.endswith('.mca'): shutil.copy2(os.path.join(d, f), os.path.join(backup, sub, f))
print('копия региона и сущностей:', backup)
lost = []
for (cx, cz), (p, nm, c) in chunks.items():
    have = {int.from_bytes(sec[b'Y'][1], 'big', signed=True) + 4 for sec in c[b'sections'][1][1] if sec.get(b'block_states') is not None}
    a, b = (cx - bx0)*16, (cz - bz0)*16
    miss = {y >> 4 for y in np.nonzero(chg[a:a + 16, :, b:b + 16].any(axis=(0, 2)))[0]} - have
    if miss: lost.append((cx, cz, sorted(miss)))
if lost: print('ВНИМАНИЕ: нет секций для изменений:', lost); sys.exit(1)
touched = 0
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

# лодка на озере
dbl = lambda v: struct.pack('>d', v); flt = lambda v: struct.pack('>f', v)
bx_, bz_ = 271.5, -100.5
cx, cz = int(bx_) >> 4, int(bz_) >> 4
pth = os.path.join(ENT, f'r.{cx >> 5}.{cz >> 5}.mca'); er = mca.Region(pth)
raw = er.raw(cx, cz)
dv = next(iter(chunks.values()))[2][b'DataVersion']
if raw is None: nm, c = b'', {b'Position': (11, struct.pack('>ii', cx, cz)), b'DataVersion': dv, b'Entities': (9, (10, []))}
else: nm, c = mca.parse(raw)
if c[b'Entities'][1][0] == 0: c[b'Entities'] = (9, (10, c[b'Entities'][1][1]))
c[b'Entities'][1][1].append({b'id': (8, b'minecraft:oak_boat'), b'Pos': (9, (6, [dbl(bx_), dbl(float(WL) + 0.05), dbl(bz_)])),
    b'Motion': (9, (6, [dbl(0.0)]*3)), b'Rotation': (9, (5, [flt(65.0), flt(0.0)])), b'UUID': (11, uuid.uuid4().bytes)})
er.put(cx, cz, mca.dump(nm, c)); er.save()
print('лодка поставлена:', (bx_, WL, bz_))
