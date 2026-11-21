#!/usr/bin/env python3
"""Расширение острова на юг в сохранённом мире «Ромашковая Долина» по локации LevelMod «расширение_острова».

Запуск: island_extension.py <папка мира> [--places places.json] [--preview файл.png] [--write <новая папка копии>], игра закрыта.
Без --write мир и places.json не меняются, только отчёт и предпросмотр. С --write в папку копии кладутся region/ целиком и
places.json, потом пишутся блоки и новые границы локаций. Не идемпотентен: повторный запуск — только с копии.

Решения пользователя (2026-11-16):
- пролив между южным пляжем и локацией засыпается, прежний пляж там, где он стал внутри суши, — трава и лес (шва нет);
- суша доходит до границы локации, кромка плавная (маска суши сглажена), подводный склон уходит наружу;
- рельеф низкий: +2…+6 над морем (y 64–68), к острову плавно подводится под его высоты;
- у новой кромки сразу песчаный пляж (BEACH ± 3 блока), к воде y=62;
- на западе густой хвойный лес, на юге и юго-востоке сначала дубовое редколесье у острова, дальше густой хвойный;
- ели — постройки BuildMod «Хвойное дерево 1…8» (data/buildmod/buildings/911–918.bin), дубы — образцы с самого острова;
- границы «остров» и «расширение_острова» переписываются по новой суше, плавные.
"""
import gzip, io, json, os, shutil, struct, sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca
import snow_and_clearing as S

NY, YMIN = S.NY, S.Y_MIN
SEA = 62                       # верхний блок воды
X0, X1, Z0, Z1 = -416, 208, 192, 512    # рамка (кратна 16)
ZONE = dict(x0=-380, x1=175, z0=250)    # где разрешено добавлять сушу
STRAIT_X = (-362, 124)         # пролив засыпается только в этих x (восточная бухта и западный берег не трогаются)
SMOOTH = 8.0                   # сглаживание маски суши, блоков
BEACH, BEACH_VAR = 11.0, 3.0   # ширина пляжа
BIG_AT, BIG_HALF, BIG_DEPTH = (95, 480), 75, 45     # большой пляж: ближайшая к BIG_AT точка берега, полудлина, глубина
PALM_SAND = 8                  # пальмы на песке: на юге и юго-востоке пляж шире на эту полосу, свободный песок у воды прежний
H_LOW, H_HIGH = 64, 68         # низина
BLEND_OLD = 24.0               # на скольких блоках высоты подводятся к старому острову
SHELF = 45                     # подводный склон, блоков от кромки
SPRUCE_GAP, OAK_GAP = 5.0, 13.0
SEED = 2611161
AREA_EXT, AREA_ISLAND = 'расширение_острова', 'остров'

world = sys.argv[1]
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
backup, preview = arg('--write'), arg('--preview')
places = arg('--places') or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(world))), 'data/levelmod/worlds',
                                          os.path.basename(os.path.abspath(world)), 'places.json')
REG = os.path.join(world, 'dimensions/minecraft/overworld/region')
rng = np.random.default_rng(SEED)
P = json.load(open(places))
sx, sz = X1 - X0, Z1 - Z0
XX, ZZ = np.meshgrid(np.arange(sx) + X0, np.arange(sz) + Z0, indexing='ij')


def poly_mask(loop, ox, oz, w, h, scale=1):
    from matplotlib.path import Path
    gx, gz = np.meshgrid((np.arange(w) + 0.5)/scale + ox, (np.arange(h) + 0.5)/scale + oz, indexing='ij')
    return Path(loop).contains_points(np.c_[gx.ravel(), gz.ravel()]).reshape(w, h)


def smooth_noise(sigma, shape=(sx, sz)):
    n = ndimage.gaussian_filter(rng.standard_normal(shape), sigma, mode='wrap')
    return (n - n.min())/(n.max() - n.min())


def sstep(a, b, x):
    t = np.clip((x - a)/(b - a), 0, 1); return t*t*(3 - 2*t)


# ---------------------------------------------------------------- чтение
idx = {'minecraft:air': 0}
vol = np.zeros((sx, NY, sz), np.uint16); chunks = {}; REGS = {}
for cx in range(X0 >> 4, X1 >> 4):
    for cz in range(Z0 >> 4, Z1 >> 4):
        p = os.path.join(REG, f'r.{cx >> 5}.{cz >> 5}.mca')
        if p not in REGS: REGS[p] = mca.Region(p)
        raw = REGS[p].raw(cx, cz)
        if raw is None: sys.exit(f'нет чанка {cx},{cz}')
        nm, c = mca.parse(raw); chunks[(cx, cz)] = (p, nm, c)
        names, v = S._volume(c)
        mp = np.array([idx.setdefault(k, len(idx)) for k in names])
        vol[(cx - (X0 >> 4))*16:(cx - (X0 >> 4) + 1)*16, :, (cz - (Z0 >> 4))*16:(cz - (Z0 >> 4) + 1)*16] = mp[v]


def ID(k): return idx.setdefault(k, len(idx))


for k in ('minecraft:grass_block[snowy=false]', 'minecraft:dirt', 'minecraft:stone', 'minecraft:sand', 'minecraft:sandstone',
          'minecraft:gravel', 'minecraft:water[level=0]', 'minecraft:short_grass', 'minecraft:oxeye_daisy', 'minecraft:azure_bluet'):
    ID(k)
orig = vol.copy()


def names_arr():
    a = [None]*len(idx)
    for k, i in idx.items(): a[i] = k
    return a


bn = np.array([S.bname(k) for k in names_arr()])
SOFTN = set(S.PLANTS) | {'air', 'cave_air', 'void_air', 'water', 'vine', 'sugar_cane', 'lily_pad', 'kelp', 'kelp_plant', 'seagrass',
                          'tall_seagrass', 'bubble_column', 'cocoa', 'bamboo'}
soft = np.array([b in SOFTN or b.endswith('leaves') or b.endswith('_log') for b in bn])
isw = np.array([b in ('water', 'kelp', 'kelp_plant', 'seagrass', 'tall_seagrass', 'bubble_column') for b in bn])
G = np.zeros((sx, sz), np.int32); WAT = np.zeros((sx, sz), bool); SURF = np.empty((sx, sz), object)
for i in range(sx):
    sl = vol[i]                                  # [y, z]
    gm = ~soft[sl]
    top = NY - 1 - np.argmax(gm[::-1, :], axis=0)
    G[i] = top
    SURF[i] = bn[sl[top, np.arange(sz)]]
    WAT[i] = isw[sl[np.minimum(top + 1, NY - 1), np.arange(sz)]]
GY = G + YMIN
old_land = ~WAT & (GY >= SEA - 1)

# ---------------------------------------------------------------- новая суша
E = poly_mask(P['areas'][AREA_EXT]['loops'][0], X0, Z0, sx, sz)
I = poly_mask(P['areas'][AREA_ISLAND]['loops'][0], X0, Z0, sx, sz)
zone = (XX >= ZONE['x0']) & (XX <= ZONE['x1']) & (ZZ >= ZONE['z0'])
lab, _ = ndimage.label(~old_land)
ocean = lab == lab[-1, -1]
strait = ocean & I & zone & (XX >= STRAIT_X[0]) & (XX <= STRAIT_X[1])
raw = old_land | E | strait
f = ndimage.gaussian_filter(raw.astype(float), SMOOTH)
land = old_land | (zone & (f > 0.5))
holes = ndimage.binary_fill_holes(land) & ~land
hl, hn = ndimage.label(holes)
for k in range(1, hn + 1):
    if (strait & (hl == k)).any() or (E & (hl == k)).any(): land |= hl == k
d_in = ndimage.distance_transform_edt(land)
d_old = ndimage.distance_transform_edt(old_land)
conv = old_land & zone & (SURF == 'sand') & (d_in - d_old > 4)          # прежний пляж, ставший внутренним
new = land & ~old_land
keep = old_land & ~conv
e_old, near = ndimage.distance_transform_edt(~keep, return_indices=True)
hN = GY[near[0], near[1]].astype(float)

bw = BEACH + BEACH_VAR*(2*smooth_noise(18) - 1)
# большой пляж на юго-востоке: ≈150 блоков вдоль берега, 40–50 вглубь, края плавные
_, nc = ndimage.distance_transform_edt(land, return_indices=True)        # ближайшая вода к каждой клетке суши
coast = land & ndimage.binary_dilation(~land)
ci = np.argwhere(coast & zone)
k_ = np.argmin((ci[:, 0] + X0 - BIG_AT[0])**2 + (ci[:, 1] + Z0 - BIG_AT[1])**2)
BIG_C = (ci[k_, 0] + X0, ci[k_, 1] + Z0)
dc = np.hypot(nc[0] + X0 - BIG_C[0], nc[1] + Z0 - BIG_C[1])
wbig = 1 - sstep(BIG_HALF - 15, BIG_HALF + 20, dc + 10*(2*smooth_noise(8) - 1))
bw = bw*(1 - wbig) + (BIG_DEPTH + 5*(2*smooth_noise(10) - 1))*wbig
# хвоя только на западе; у южного берега её край вытянут на юго-восток (≈35 + 50 блоков к z≈455)
west = sstep(-180, -250, XX - 0.3*np.clip(ZZ - 340, 0, None) - 1.1*np.clip(ZZ - 385, 0, None)
             + 45*(2*smooth_noise(20) - 1) + 12*(2*smooth_noise(5) - 1))
pband = (1 - sstep(0.2, 0.5, west))*(1 - sstep(0.15, 0.35, wbig))       # где по пляжу стоят пальмы
bw0 = bw.copy()
bw = bw + PALM_SAND*pband
print(f'большой пляж: середина у {BIG_C}')
hn_ = smooth_noise(22)
h_t = H_LOW + (H_HIGH - H_LOW)*(0.55*hn_ + 0.45*smooth_noise(9))
h_beach = SEA + 1.6*np.clip(d_in/np.maximum(bw, 1), 0, 1)
h_in = h_t*sstep(0, BLEND_OLD, e_old) + hN*(1 - sstep(0, BLEND_OLD, e_old))
H = h_beach*(1 - sstep(bw, bw + 14, d_in)) + h_in*sstep(bw, bw + 14, d_in)
H = np.rint(H).astype(int)
work = new | conv
sandy = work & (d_in < bw)

# подводный склон
d_out, nearL = ndimage.distance_transform_edt(~land, return_indices=True)
from_new = work[nearL[0], nearL[1]]
bed = np.rint(SEA - 1 - 0.5*(d_out - 1) - 2*smooth_noise(6)).astype(int)
shelf = ~land & zone & from_new & (d_out <= SHELF) & (bed > GY)
print(f'новая суша {int(new.sum())} столбцов, прежний пляж в траву {int(conv.sum())}, пляж {int(sandy.sum())}, подводный склон {int(shelf.sum())}')

# ---------------------------------------------------------------- блоки
GRASS, DIRT, STONE, SAND, SST, GRAV, WATER = (ID(k) for k in ('minecraft:grass_block[snowy=false]', 'minecraft:dirt', 'minecraft:stone',
                                                               'minecraft:sand', 'minecraft:sandstone', 'minecraft:gravel', 'minecraft:water[level=0]'))
AIR = 0
bn = np.array([S.bname(k) for k in names_arr()])
for i, j in zip(*np.nonzero(work)):
    h = H[i, j] - YMIN; g = G[i, j]; col = vol[i, :, j]
    # всё рыхлое и вода над новой поверхностью — воздух (кроны деревьев сверху остаются)
    top = max(g, h) + 3
    for y in range(h + 1, min(top + 40, NY)):
        b = bn[col[y]]
        if b.endswith('leaves') or b.endswith('_log'): continue
        if b != 'air': col[y] = AIR
    lo = min(g, h - 6)
    for y in range(lo + 1, h + 1):
        col[y] = STONE
    if sandy[i, j]:
        for k in range(3): col[h - k] = SAND
        for k in (3, 4): col[h - k] = SST
    else:
        col[h] = GRASS
        for k in (1, 2, 3): col[h - k] = DIRT
for i, j in zip(*np.nonzero(shelf)):
    b_ = bed[i, j] - YMIN; g = G[i, j]; col = vol[i, :, j]
    for y in range(g + 1, b_ - 1): col[y] = STONE
    top = SAND if d_out[i, j] < 12 else GRAV
    for y in (b_ - 1, b_): col[y] = top
    for y in range(b_ + 1, SEA - YMIN + 1):
        if bn[col[y]] != 'water': col[y] = WATER

# ---------------------------------------------------------------- деревья
def load_building(path):
    s = io.BytesIO(gzip.open(path).read())
    I_ = lambda: struct.unpack('>i', s.read(4))[0]
    def U(): n = struct.unpack('>H', s.read(2))[0]; return s.read(n).decode()
    I_(); U(); I_(); I_(); I_()
    pal = [U() for _ in range(I_())]
    out = [(I_(), I_(), I_(), pal[I_()]) for _ in range(I_())]
    logs = [(x, y, z) for x, y, z, k in out if '_log' in k]
    y0 = min(y for _, y, _ in logs)
    tx, tz = [(x, z) for x, y, z in logs if y == y0][0]
    return [(x - tx, y - y0, z - tz, k) for x, y, z, k in out]


BLD = os.path.join(world, 'data/buildmod/buildings')
SPRUCES = [load_building(os.path.join(BLD, f'{n}.bin')) for n in range(911, 919)]
SPRUCE_W = np.array([0.5, 0.5, 1, 1, 1.4, 1.4, 1.2, 1.2]); SPRUCE_W /= SPRUCE_W.sum()


def persistent(k):
    return k.replace('persistent=false', 'persistent=true') if 'leaves' in k else k


def extract_oaks():
    """Образцы дубов из центральной части острова (дубовое редколесье)."""
    nl = names_arr(); out = []
    isoak = np.array([S.bname(k) in ('oak_log', 'oak_leaves') for k in nl])
    gnd = np.array([S.bname(k) in ('grass_block', 'dirt') for k in nl])
    logb = np.array([S.bname(k) == 'oak_log' for k in nl])
    bases = []
    for i in range(int(-208 - X0), int(-16 - X0)):
        for j in range(int(208 - Z0), int(290 - Z0)):
            y = G[i, j] + 1
            if logb[orig[i, y, j]] and gnd[orig[i, y - 1, j]]: bases.append((i, y, j))
    for (i, y, j) in bases:
        if any(abs(a - i) <= 7 and abs(c - j) <= 7 and (a, c) != (i, j) for a, _, c in bases): continue
        seen = {(i, y, j)}; st = [(i, y, j)]
        while st:
            a, b, c = st.pop()
            for da, db, dc in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                q = (a + da, b + db, c + dc)
                if q in seen or abs(q[0] - i) > 5 or abs(q[2] - j) > 5 or not 0 <= q[1] - y < 16: continue
                if 0 <= q[0] < sx and 0 <= q[2] < sz and isoak[orig[q]]: seen.add(q); st.append(q)
        if len(seen) < 25 or any(abs(a - i) == 5 or abs(c - j) == 5 for a, _, c in seen): continue
        out.append([(a - i, b - y, c - j, persistent(nl[orig[a, b, c]])) for a, b, c in seen])
    out.sort(key=len)
    if len(out) > 10: out = [out[k] for k in np.linspace(0, len(out) - 1, 10).astype(int)]
    return out


OAKS = extract_oaks()
PALM_LOG, PALM_LEAF = 'minecraft:jungle_log[axis=y]', 'minecraft:jungle_leaves[distance=1,persistent=true,waterlogged=false]'
ID(PALM_LOG); ID(PALM_LEAF)


def palm():
    """Пальма как в оазисе острова (IslandChunkGenerator.palm): наклонный ствол 7–10, восемь перьев листвы."""
    out = []; trunk = 7 + int(rng.integers(4)); lx, lz = ((1, 0), (-1, 0), (0, 1), (0, -1))[rng.integers(4)]
    tx = tz = 0
    for i in range(trunk):
        off = int(np.floor(2.4*(i/trunk)**2)); tx, tz = lx*off, lz*off
        out.append((tx, i, tz, PALM_LOG))
    ty = trunk - 1
    out.append((tx, ty + 1, tz, PALM_LEAF))
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            if dx == 0 and dz == 0: continue
            for k in range(1, (3 if dx and dz else 4) + 1):
                drop = (1 if k == 3 else 2) if k >= 3 else 0
                out.append((tx + dx*k, ty + (1 if k == 1 else 0) - drop, tz + dz*k, PALM_LEAF))
    return out


print('образцов дубов', len(OAKS), 'размеры', [len(t) for t in OAKS])
if not OAKS: sys.exit('не нашлось ни одного отдельного дуба')
for t in SPRUCES + OAKS:
    for *_, k in t: ID(k); ID(k.replace('axis=x', 'axis=z')); ID(k.replace('axis=z', 'axis=x'))
bn = np.array([S.bname(k) for k in names_arr()])

# где какой лес (третья попытка 2026-11-16): хвоя только на западе; юг и юго-восток — редколесье, у берега полоса пальм
plant_ok = work & ~sandy & (d_in >= bw + 2)
palm_zone = work & sandy & (pband > 0.5) & (d_in >= bw0 - 1) & (d_in < bw - 1)
pine = plant_ok & (west > 0.5 + 0.45*(smooth_noise(3) - 0.5))
trunks = [(i + X0, j + Z0) for i, j in zip(*np.nonzero(np.vectorize(lambda b: b.endswith('_log'))(bn)[vol[np.arange(sx)[:, None], np.minimum(G + 1, NY - 1), np.arange(sz)[None, :]]]))]
from scipy.spatial import cKDTree
placed = []; pts = list(trunks)


def free(x, z, gap):
    return not any((x - a)**2 + (z - b)**2 < gap*gap for a, b in pts[-400:]) and not base_tree.query_ball_point((x, z), gap)


def add(i, j, kind):
    placed.append((i, j, kind)); pts.append((i + X0, j + Z0))


# 1) густой ельник на западе
base_tree = cKDTree(pts)
cand = np.argwhere(pine); rng.shuffle(cand)
for n_, (i, j) in enumerate(cand):
    if n_ % 3000 == 0: base_tree = cKDTree(pts)
    if free(i + X0, j + Z0, SPRUCE_GAP): add(i, j, 'spruce')
# 2) пальмы: полоса за пляжем, вразнобой, кое-где по две-три рядом
base_tree = cKDTree(pts)
cand = np.argwhere(palm_zone); rng.shuffle(cand); npalm = 0
for n_, (i, j) in enumerate(cand):
    if n_ % 3000 == 0: base_tree = cKDTree(pts)
    if rng.random() < 0.45: continue
    if free(i + X0, j + Z0, rng.uniform(5.5, 12)): add(i, j, "palm"); npalm += 1
# 3) редколесье: группы по 2–6 деревьев, одиночки, поляны — по пятнистому шуму
base_tree = cKDTree(pts)
glade = smooth_noise(7)
open_ = plant_ok & ~pine & ~palm_zone & (glade < 0.7)
cand = np.argwhere(open_); rng.shuffle(cand)
centers = []
for i, j in cand:
    x, z = i + X0, j + Z0
    if any((x - a)**2 + (z - b)**2 < 15**2 for a, b in centers): continue
    centers.append((x, z))
ng = ns = 0
for x, z in centers:
    r = rng.random()
    if r < 0.55:
        k = rng.integers(2, 7); spread = rng.uniform(4, 9)
        for _ in range(k*4):
            if k == 0: break
            ang = rng.uniform(0, 2*np.pi); rr = spread*np.sqrt(rng.random())
            xi, zj = int(round(x + rr*np.cos(ang))) - X0, int(round(z + rr*np.sin(ang))) - Z0
            if not (0 <= xi < sx and 0 <= zj < sz and open_[xi, zj]): continue
            if not free(xi + X0, zj + Z0, rng.uniform(3.5, 5.5)): continue
            u = rng.random(); c = west[xi, zj]
            add(xi, zj, 'spruce' if u < 0.6*c else 'spruce_s' if u < 0.6*c + 0.06 else 'oak'); k -= 1
        ng += 1
    elif r < 0.9:
        xi, zj = int(x + rng.uniform(-10, 10)) - X0, int(z + rng.uniform(-10, 10)) - Z0
        if 0 <= xi < sx and 0 <= zj < sz and open_[xi, zj] and free(xi + X0, zj + Z0, 6):
            add(xi, zj, 'oak'); ns += 1
print(f'редколесье: групп {ng}, одиночек {ns}; пальм {npalm}')
print('деревьев:', {k: sum(1 for p in placed if p[2] == k) for k in ('spruce', 'spruce_s', 'oak', 'palm')})
for i, j, kind in placed:
    t = (SPRUCES[rng.choice(8, p=SPRUCE_W)] if kind == 'spruce' else SPRUCES[rng.integers(4)] if kind == 'spruce_s'
         else palm() if kind == 'palm' else OAKS[rng.integers(len(OAKS))])
    y0 = H[i, j] - YMIN + 1
    rot = rng.integers(4)
    for dx, dy, dz, k in t:
        for _ in range(rot): dx, dz = -dz, dx
        a, c = i + dx, j + dz; b = y0 + dy
        if not (0 <= a < sx and 0 <= c < sz and b < NY): continue
        isl = '_log' in k
        if rot % 2 and isl: k = k.replace('axis=x', 'axis=Q').replace('axis=z', 'axis=x').replace('axis=Q', 'axis=z')
        cur = bn[vol[a, b, c]]
        if cur == 'air' or (isl and (cur.endswith('leaves') or cur in SOFTN)):
            vol[a, b, c] = ID(k)
    if vol[i, y0 - 1, j] == GRASS: vol[i, y0 - 1, j] = DIRT
bn = np.array([S.bname(k) for k in names_arr()])

# трава и цветы как на острове
SG, OX_, AZ = ID('minecraft:short_grass'), ID('minecraft:oxeye_daisy'), ID('minecraft:azure_bluet')
for i, j in zip(*np.nonzero(work & ~sandy)):
    y = H[i, j] - YMIN + 1
    if vol[i, y, j] != AIR or vol[i, y - 1, j] != GRASS: continue
    r = rng.random(); fl = 0.012*(1 - west[i, j]) + 0.003
    if r < 0.05: vol[i, y, j] = SG
    elif r < 0.05 + fl: vol[i, y, j] = OX_ if rng.random() < 0.5 else AZ

# ---------------------------------------------------------------- границы локаций
def contour_loops(mask, ox, oz, scale, sigma):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fm = ndimage.gaussian_filter(mask.astype(float), sigma*scale)
    cs = plt.contour((np.arange(mask.shape[0]) + 0.5)/scale + ox, (np.arange(mask.shape[1]) + 0.5)/scale + oz, fm.T, levels=[0.5])
    segs = cs.allsegs[0]; plt.close('all')
    seg = max(segs, key=len)
    return seg


def rdp(pts, eps):
    if len(pts) < 3: return pts
    a, b = pts[0], pts[-1]; ab = b - a; n = np.hypot(*ab) or 1
    dd = np.abs(np.cross(ab, pts - a))/n
    k = int(np.argmax(dd))
    if dd[k] > eps: return np.vstack([rdp(pts[:k + 1], eps)[:-1], rdp(pts[k:], eps)])
    return np.vstack([a, b])


def area_from(seg, ground):
    seg = np.asarray(seg)
    if np.allclose(seg[0], seg[-1]): seg = seg[:-1]
    m = int(np.argmax(np.hypot(*(seg - seg[0]).T)))          # замкнутую линию упрощаем двумя половинами
    s = np.vstack([rdp(seg[:m + 1], 0.6)[:-1], rdp(np.vstack([seg[m:], seg[:1]]), 0.6)[:-1]])
    pts = [[int(np.floor(x)), int(ground(x, z)), int(np.floor(z))] for x, z in s]
    ded = [p for k, p in enumerate(pts) if k == 0 or p[0::2] != pts[k - 1][0::2]]
    if ded[0][0::2] == ded[-1][0::2]: ded.pop()
    n = len(ded)
    return {'dimension': 'minecraft:overworld', 'loops': [[[p[0] + 0.5, p[2] + 0.5] for p in ded]], 'points': ded,
            'edges': [[k, (k + 1) % n, 0, 0] for k in range(n)]}


OLDPTS = np.array(P['areas'][AREA_ISLAND]['points'])


def ground_at(x, z):
    i, j = int(np.floor(x)) - X0, int(np.floor(z)) - Z0
    if 0 <= i < sx and 0 <= j < sz: return max(SEA, H[i, j] if work[i, j] else GY[i, j])
    return OLDPTS[np.argmin((OLDPTS[:, 0] - x)**2 + (OLDPTS[:, 2] - z)**2), 1]


IL = np.array(P['areas'][AREA_ISLAND]['loops'][0])
bx0, bx1 = int(min(IL[:, 0].min(), X0)) - 8, int(max(IL[:, 0].max(), X1)) + 8
bz0, bz1 = int(min(IL[:, 1].min(), Z0)) - 8, int(max(IL[:, 1].max(), Z1)) + 8
SC = 2
big = poly_mask(IL, bx0, bz0, (bx1 - bx0)*SC, (bz1 - bz0)*SC, SC)
lw = np.repeat(np.repeat(land & zone, SC, 0), SC, 1)
a0, c0 = (X0 - bx0)*SC, (Z0 - bz0)*SC
big[a0:a0 + sx*SC, c0:c0 + sz*SC] |= lw
new_island = area_from(contour_loops(big, bx0, bz0, SC, 1.5), ground_at)
emask = (work | (land & E)) & zone
emask = ndimage.binary_opening(emask, iterations=2)
new_ext = area_from(contour_loops(np.repeat(np.repeat(emask, SC, 0), SC, 1), X0, Z0, SC, 1.5), ground_at)
print(f'граница «{AREA_ISLAND}»: {len(IL)} → {len(new_island["points"])} точек; «{AREA_EXT}»: '
      f'{len(P["areas"][AREA_EXT]["points"])} → {len(new_ext["points"])} точек')

# ---------------------------------------------------------------- предпросмотр
chg = vol != orig
print('изменено блоков:', int(chg.sum()))
print('высоты новой суши: пляж', np.percentile(H[sandy], [0, 50, 100]), 'внутри', np.percentile(H[work & ~sandy], [0, 5, 50, 95, 100]))
for xx in (-300, -100, 60):
    i = xx - X0
    print(f'разрез x={xx}:', ' '.join(f'{z}:{H[i, z - Z0] if work[i, z - Z0] else ("~" if not land[i, z - Z0] else GY[i, z - Z0])}' for z in range(270, 480, 10)))
if preview:
    from PIL import Image, ImageDraw
    COL = {'grass_block': (90, 150, 60), 'sand': (220, 205, 150), 'dirt': (130, 100, 60), 'stone': (128, 128, 128), 'gravel': (140, 135, 130),
           'clay': (160, 160, 175), 'sandstone': (210, 195, 140), 'dirt_path': (200, 0, 200)}
    bn2 = np.array([S.bname(k) for k in names_arr()])

    def top_view(v):
        img = np.zeros((sz, sx, 3), np.uint8)
        nonair = ~np.isin(bn2, ['air', 'cave_air', 'void_air'])
        for i in range(sx):
            sl = v[i]; nz = nonair[sl]
            t = NY - 1 - np.argmax(nz[::-1, :], axis=0)
            names_ = bn2[sl[t, np.arange(sz)]]
            for j in range(sz):
                b = names_[j]
                if b == 'water': c = np.array((40, 80, 180))*(0.55 + 0.45*np.clip((GY[i, j] if not shelf[i, j] else bed[i, j]) - 30, 0, 32)/32)
                elif b == 'spruce_leaves': c = (25, 80, 30)
                elif b.endswith('leaves'): c = (60, 130, 40)
                else: c = COL.get(b, (90, 150, 60))
                sh = 1.0 + (0.06*(int(t[j]) - int(t[j - 1])) if j else 0)
                img[j, i] = np.clip(np.array(c)*np.clip(sh, 0.7, 1.3), 0, 255)
        return Image.fromarray(img)
    a_, b_ = top_view(orig), top_view(vol)
    d = ImageDraw.Draw(b_)
    for ar, cl in ((new_island, (255, 0, 0)), (new_ext, (255, 255, 0))):
        L = ar['loops'][0]; d.line([(x - X0, z - Z0) for x, z in L + [L[0]]], fill=cl, width=1)
    out = Image.new('RGB', (sx, sz*2 + 6), (0, 0, 0)); out.paste(a_, (0, 0)); out.paste(b_, (0, sz + 6))
    out.resize((sx*2, (sz*2 + 6)*2), Image.NEAREST).save(preview); print('предпросмотр:', preview)

if not backup:
    print('без --write мир не изменён'); sys.exit()

# ---------------------------------------------------------------- запись
os.makedirs(os.path.join(backup, 'region'), exist_ok=False)
for f_ in os.listdir(REG):
    if f_.endswith('.mca'): shutil.copy2(os.path.join(REG, f_), os.path.join(backup, 'region'))
shutil.copy2(places, os.path.join(backup, 'places.json'))
print('копия:', backup)
nl = names_arr(); touched = 0
for (cx, cz), (p, nm, c) in chunks.items():
    a, b = (cx - (X0 >> 4))*16, (cz - (Z0 >> 4))*16
    sub = chg[a:a + 16, :, b:b + 16]
    if not sub.any(): continue
    have = {int.from_bytes(sec[b'Y'][1], 'big', signed=True) + 4 for sec in c[b'sections'][1][1] if sec.get(b'block_states') is not None}
    changed = {y >> 4 for y in np.nonzero(sub.any(axis=(0, 2)))[0]}
    if changed - have: sys.exit(f'нет секций {sorted(changed - have)} в чанке {cx},{cz}; ничего не записано не было до этого чанка: вернуть копию')
    v = vol[a:a + 16, :, b:b + 16]
    S.write_sections(c, v, nl, changed, True)
    S.set_heightmaps(c, v, nl)
    REGS[p].put(cx, cz, mca.dump(nm, c)); touched += 1
for r in REGS.values(): r.save()
P['areas'][AREA_ISLAND] = new_island
P['areas'][AREA_EXT] = new_ext
tmp = places + '.tmp'
with open(tmp, 'w') as fo: json.dump(P, fo, ensure_ascii=False)
os.replace(tmp, places)
print(f'записано чанков: {touched}; границы локаций обновлены в {places}')
