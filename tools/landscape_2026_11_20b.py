#!/usr/bin/env python3
"""Вторая правка ландшафта «Ромашковой Долины» по рисунку пользователя (2026-11-20, ~/Изображения/update.png поверх снимка с областями отката).

Запуск: landscape_2026_11_20b.py <мир> --part <часть> [--preview файл.png] [--write], игра закрыта (pgrep -f "[K]notClient").
Части по порядку, каждая читает мир заново: backup <папка> → north → se → hills → blend → jcliff → coast → spruce.
  north   северный красный участок: горы как главный массив (пики ~185, снег выше ~140), на севере альпийский луг (чаша y ~136, озерцо,
          цветы, валуны, редкие ели), к морю склоном с каменистым берегом, к джунглям плавно (60 блоков), ели низкой-средней плотности;
  se      юго-восточный красный участок: горы до ~160 не ближе 45 блоков к «засаде_туземцев» (поднимаются от 45 до 85),
          коридор шара (склон не выше линии полёта − 15 в 15 блоках от неё), к морю склоном, к джунглям плавно, ели;
  hills   зелёная трапеция: пролив засыпан мягкими холмами 70–100, на западе сходят к холму с руинами, на востоке к подножию гор,
          трава, цветы, редкие дубы и берёзы, подводный склон у новой кромки;
  blend   прежние горы к западу от джунглей: стенка гор к джунглям сглажена (предгорья с травой, горы у стыка ниже);
  jcliff  восточный обрыв джунглей: скальные уступы (пласты, мох, лианы, кусты на полках), у воды каменистая полоска;
  coast   западный берег леса: пляж заменён травой на SEA+3…4, лес обрывается к воде земляным уступом; пальмы у леса и у южного края пустыни сняты.
Запрет везде: жерло вулкана и устье (26 блоков), расщелина с тарзанками и вход в тоннель, река с брёвнами (14 блоков от узлов),
коридор шара (высота). Без --write мир не меняется (отчёт и предпросмотр «было | стало»). Маски — landscape_2026_11_20b_masks.png
(пиксель = мир + (700, 590); (150,150,150) север, (120,120,120) юго-восток, (60,160,60) холмы). Не идемпотентно: повторять после возврата копии.
Решения пользователя (2026-11-20): горы как главный массив (юго-восток ниже, до ~160), низменность — альпийский луг, горы к морю склоном,
холмы 70–100 до руин, обрыв джунглей — скальные уступы, обрыв 3–4 блока и у стыка пустыни с лесом, и на западном берегу; ели в горах
низкой-средней плотности; плавные переходы гор к джунглям; коридор для шара.
"""
import gzip, io, json, os, shutil, struct, sys, glob
import numpy as np
from scipy import ndimage
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import landscape as L
import mca
from landscape import sstep, NY, YMIN, SEA

world = sys.argv[1]
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
part, preview, WRITE = arg('--part'), arg('--preview'), '--write' in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
PLACES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(world))), 'data/levelmod/worlds', os.path.basename(os.path.abspath(world)), 'places.json')
MASKS = np.array(Image.open(os.path.join(HERE, 'landscape_2026_11_20b_masks.png')).convert('RGB'))
MOX, MOZ = 700, 590
MASK_ID = np.zeros(MASKS.shape[:2], np.uint8)
for k, c in ((1, (150, 150, 150)), (2, (120, 120, 120)), (3, (60, 160, 60))): MASK_ID[(MASKS == c).all(2)] = k
rng = np.random.default_rng(2611202)
VENT, VENT_R = (137, -273), 26
CHASM = (338, 628, -338, -250)               # расщелина и вход в тоннель: x0, x1, z0, z1
ROCKS = ('stone', 'andesite', 'tuff', 'calcite', 'granite', 'diorite', 'gravel', 'snow_block', 'packed_ice', 'ice', 'cobblestone', 'deepslate')
SOILS = ('grass_block', 'podzol', 'dirt', 'coarse_dirt', 'mud', 'moss_block', 'rooted_dirt', 'mycelium')


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]; return x*x + y*y <= r*r


def BACKUP_HINT():
    p = os.path.join(world, '.landscape_backup_b'); return open(p).read().strip() if os.path.exists(p) else ''


def finish(f, title):
    chg = f.changed()
    print(f'{title}: изменено блоков {int(chg.sum())}, столбцов {int(chg.any(1).sum())}')
    if preview:
        a, b = f.top_view(f.orig), f.top_view()
        out = Image.new('RGB', (f.sx*2 + 6, f.sz), (0, 0, 0)); out.paste(a, (0, 0)); out.paste(b, (f.sx + 6, 0))
        out.save(preview); print('предпросмотр:', preview)
    if WRITE:
        if not os.path.isdir(BACKUP_HINT()): sys.exit('нет копии: сначала --part backup <папка>')
        print('записано чанков:', f.write())
    else:
        print('без --write мир не изменён')


def mask(f, k):
    return f.mask(MASK_ID, k, -MOX, -MOZ)


# ---------------------------------------------------------------- запреты и коридор шара
def river_nodes():
    import log_routes_dat as R
    return np.array([[n[0], n[2]] for route in R.read(world).values() for n in route])


def wind_markers():
    E = os.path.join(world, 'dimensions/minecraft/overworld/entities')
    out = []
    for p in glob.glob(os.path.join(E, '*.mca')):
        if os.path.getsize(p) < 8192: continue
        r = mca.Region(p); rx, rz = map(int, os.path.basename(p)[2:-4].split('.'))
        for cx in range(rx*32, rx*32 + 32):
            for cz in range(rz*32, rz*32 + 32):
                b = r.raw(cx, cz)
                if b is None: continue
                for e in mca.parse(b)[1][b'Entities'][1][1]:
                    if e[b'id'][1] == b'gameplaymod:wind_marker':
                        out.append([struct.unpack('>d', v)[0] for v in e[b'Pos'][1][1]])
    return np.array(out)


def polygon_mask(f, name):
    from matplotlib.path import Path
    P = json.load(open(PLACES))['areas'][name]
    m = np.zeros((f.sx, f.sz), bool)
    pts = np.c_[f.XX.ravel() + 0.5, f.ZZ.ravel() + 0.5]
    for loop in P['loops']:
        m ^= Path(np.array(loop)).contains_points(pts).reshape(f.sx, f.sz)
    return m


def sea(f):
    """Море: вода на уровне моря (не река и не озеро выше него), связанная с краем рамки."""
    bn = f.bn(); isw = np.array([b in L.WATERS for b in bn])
    lvl = np.zeros((f.sx, f.sz), bool)
    s0 = SEA - YMIN
    for i in range(f.sx): lvl[i] = isw[f.vol[i, s0, :]] & ~isw[f.vol[i, s0 + 1, :]]
    lab, n = ndimage.label(lvl)
    edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    return np.isin(lab, list(edge))


def bans(f):
    """Где рельеф не трогаем: жерло и устье, расщелина и вход в тоннель, река с брёвнами."""
    b = np.hypot(f.XX - VENT[0], f.ZZ - VENT[1]) < VENT_R
    x0, x1, z0, z1 = CHASM
    b |= (f.XX >= x0) & (f.XX <= x1) & (f.ZZ >= z0) & (f.ZZ <= z1)
    rn = river_nodes()
    rv = np.zeros_like(b)
    for x, z in rn:
        i, j = int(round(x)) - f.X0, int(round(z)) - f.Z0
        if 0 <= i < f.sx and 0 <= j < f.sz: rv[i, j] = True
    if rv.any(): b |= ndimage.distance_transform_edt(~rv) < 14
    return b


def balloon_cap(f):
    """Предел высоты рельефа у маршрута шара: в 15 блоках от линии маркеров ветра — высота линии − 15, дальше на блок выше за блок."""
    W = wind_markers()
    if len(W) < 2: return np.full((f.sx, f.sz), 1e9)
    W = W[np.argsort(-W[:, 2])]                      # маршрут идёт с юга на север
    cap = np.full((f.sx, f.sz), 1e9)
    X, Z = f.XX + 0.5, f.ZZ + 0.5
    for a, b in zip(W[:-1], W[1:]):
        if np.hypot(b[0] - a[0], b[2] - a[2]) > 60: continue   # разрыв между площадками ветра — шар идёт по инерции, но прямо
        ax, az, bx, bz = a[0], a[2], b[0], b[2]
        dx, dz = bx - ax, bz - az; L2 = dx*dx + dz*dz or 1
        t = np.clip(((X - ax)*dx + (Z - az)*dz)/L2, 0, 1)
        d = np.hypot(X - ax - t*dx, Z - az - t*dz)
        y = a[1] + t*(b[1] - a[1])
        cap = np.minimum(cap, y - 15 + np.maximum(0, d - 15))
    return cap


# ---------------------------------------------------------------- деревья и мелочи
def kill_trees(f, doomed, above):
    bn = f.bn()
    islog = np.array([b.endswith('_log') or b.endswith('_wood') for b in bn]); isleaf = np.array([b.endswith('leaves') for b in bn])
    hang = np.array([b in ('vine', 'cocoa') for b in bn])
    logs = islog[f.vol]
    lab, n = ndimage.label(logs, structure=np.ones((3, 3, 3)))
    yy = np.arange(NY)[None, :, None]
    hit = logs & doomed[:, None, :] & (yy > above[:, None, :])
    bad = np.unique(lab[hit]); bad = bad[bad > 0]
    kill = np.isin(lab, bad)
    f.vol[kill] = 0
    leaves = isleaf[f.vol]; reach = islog[f.vol]
    for _ in range(6):
        reach = reach | (ndimage.binary_dilation(reach) & leaves)
    zone = ndimage.binary_dilation(kill, structure=np.ones((3, 3, 3)), iterations=7) if kill.any() else np.zeros_like(kill)
    orphan = leaves & ~reach & zone
    f.vol[orphan] = 0
    solid = (f.vol != 0) & ~hang[f.vol]
    near = ndimage.binary_dilation(solid, structure=ndimage.generate_binary_structure(3, 1))
    h = hang[f.vol]
    held = h & near
    for y in range(NY - 2, -1, -1):
        held[:, y, :] |= h[:, y, :] & held[:, y + 1, :]
    f.vol[h & ~held] = 0
    print(f'  деревьев снято {len(bad)}, брёвен {int(kill.sum())}, листвы без опоры {int(orphan.sum())}')
    return len(bad)


def rock_ids(f):
    ids = np.array([f.ID('minecraft:' + b) for b in ('stone', 'andesite', 'tuff', 'stone')])
    off = 4*L.S.vnoise(f.XX/23.0, f.ZZ/23.0)
    return lambda i, j, y0, y1: ids[(np.floor((np.arange(y0, y1 + 1) + off[i, j])/3).astype(int)) % 4]


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


def place(f, i, j, y0, t, rot=0):
    bn = f.bn()
    for dx, dy, dz, k in t:
        for _ in range(rot): dx, dz = -dz, dx
        a, c, b = i + dx, j + dz, y0 + dy
        if not (0 <= a < f.sx and 0 <= c < f.sz and 0 <= b < NY): continue
        if rot % 2 and '_log' in k: k = k.replace('axis=x', 'axis=Q').replace('axis=z', 'axis=x').replace('axis=Q', 'axis=z')
        cur = f.vol[a, b, c]
        if cur == 0 or ('_log' in k and cur < len(bn) and (bn[cur].endswith('leaves') or bn[cur] in L.SOFTN)):
            f.vol[a, b, c] = f.ID(k)


def scatter(cand, gap, prob=1.0):
    from scipy.spatial import cKDTree
    out = []; c = cand.copy(); rng.shuffle(c); pts = []; tree = None
    for i, j in c:
        if rng.random() > prob: continue
        g = gap() if callable(gap) else gap
        if tree is not None and tree.query_ball_point((i, j), g): continue
        if any((i - a)**2 + (j - b)**2 < g*g for a, b in out[len(pts):]): continue
        out.append((i, j))
        if len(out) - len(pts) > 400: pts = list(out); tree = cKDTree(pts)
    return out


SPRUCES = None


def spruces(f):
    global SPRUCES
    if SPRUCES is None:
        BLD = os.path.join(world, 'data/buildmod/buildings')
        SPRUCES = [load_building(os.path.join(BLD, f'{n}.bin')) for n in range(911, 919)]
    return SPRUCES


LEAF = lambda w: f'minecraft:{w}_leaves[distance=1,persistent=true,waterlogged=false]'


def bush(kind='jungle'):
    out = [(0, 0, 0, f'minecraft:{kind}_log[axis=y]')]
    r = 1 + int(rng.random() < 0.4)
    for dx in range(-r, r + 1):
        for dz in range(-r, r + 1):
            for dy in (0, 1):
                if dx == dz == 0 and dy == 0: continue
                if dx*dx + dz*dz + dy*dy*2 <= r*r + 1 and rng.random() < 0.9: out.append((dx, dy, dz, LEAF(kind)))
    return out


def jungle_tree():
    """Джунглевое дерево: ствол 6–10 с лианами нет, крона двумя слоями."""
    h = 6 + int(rng.integers(5))
    out = [(0, y, 0, 'minecraft:jungle_log[axis=y]') for y in range(h)]
    for dy, r in ((-1, 3), (0, 3), (1, 2)):
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if dx*dx + dz*dz > r*r + 1 or (dx*dx + dz*dz > r*r - 1 and rng.random() < 0.4): continue
                out.append((dx, h + dy, dz, LEAF('jungle')))
    return out


def broadleaf(kind):
    """Простой дуб или берёза: ствол 4–6, крона шаром."""
    h = 4 + int(rng.integers(3)) + (1 if kind == 'birch' else 0)
    out = [(0, y, 0, f'minecraft:{kind}_log[axis=y]') for y in range(h)]
    for dy in range(-2, 2):
        r = 2 if dy < 1 else 1
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if abs(dx) == abs(dz) == r and rng.random() < 0.6: continue
                out.append((dx, h + dy, dz, LEAF(kind)))
    out.append((0, h + 1, 0, LEAF(kind)))
    return out


def plants(f, cells, H, rate_grass=0.18, rate_flower=0.03, flowers=('dandelion', 'poppy', 'azure_bluet', 'oxeye_daisy', 'cornflower')):
    GR = f.ID('minecraft:grass_block[snowy=false]'); SG = f.ID('minecraft:short_grass')
    FL = [f.ID('minecraft:' + n) for n in flowers]
    for i, j in zip(*np.nonzero(cells)):
        y = H[i, j] - YMIN + 1
        if y >= NY or f.vol[i, y, j] != 0 or f.vol[i, y - 1, j] != GR: continue
        r = rng.random()
        if r < rate_flower: f.vol[i, y, j] = FL[int(rng.integers(len(FL)))]
        elif r < rate_flower + rate_grass: f.vol[i, y, j] = SG


def smooth_outside(hv, out):
    hN = hv.copy(); prev = None
    for sig in (6, 12, 24, 48):
        num, den = ndimage.gaussian_filter(hv*out, sig), ndimage.gaussian_filter(out.astype(float), sig)
        cand = num/np.maximum(den, 1e-9)
        hN = np.where(den > 1e-3, cand, hN) if prev is None else np.where(prev < 1e-3, cand, hN)
        prev = den
    return hN


def build(f, cells, Hn, top):
    """Столбцы cells к высоте Hn: выше — воздух (деревья сняты заранее), ниже — порода, верх по top[i,j]: 'rock','snow','snow2','grass','gravel','sand','moss'."""
    rock = rock_ids(f)
    GR, DIRT, SNOW, GRAV, SAND, STONE, MOSS = (f.ID(k) for k in ('minecraft:grass_block[snowy=false]', 'minecraft:dirt', 'minecraft:snow_block',
                                                              'minecraft:gravel', 'minecraft:sand', 'minecraft:stone', 'minecraft:moss_block'))
    WATERS = np.array([b in L.WATERS for b in f.bn()])
    for i, j in zip(*np.nonzero(cells)):
        g, h = f.G[i, j], int(Hn[i, j]) - YMIN; col = f.vol[i, :, j]
        col[min(g, h) + 1:min(max(g, h) + 60, NY)] = 0
        if h > g:
            lo = g
            while lo > 0 and WATERS[col[lo]]: lo -= 1          # под водой — заполнить до дна
            col[lo:h + 1] = rock(i, j, lo + YMIN, h + YMIN)
        t = top[i, j]
        if t == 'snow': col[h] = SNOW
        elif t == 'snow2': col[h] = SNOW; col[h - 1] = SNOW
        elif t == 'grass': col[h - 3:h] = DIRT; col[h] = GR
        elif t == 'snowgrass':
            col[h - 3:h] = DIRT; col[h] = f.ID('minecraft:grass_block[snowy=true]'); col[h + 1] = f.ID(f'minecraft:snow[layers={1 + int(rng.random() < 0.35)}]')
        elif t == 'moss': col[h] = MOSS
        elif t == 'gravel': col[h - 1:h + 1] = GRAV
        elif t == 'sand': col[h - 2:h + 1] = SAND
        else: col[h] = rock(i, j, h + YMIN, h + YMIN)[0] if col[h] == 0 else col[h]
        if h < SEA - YMIN:                                          # ниже моря: вода над верхом (у берега)
            col[h + 1:SEA - YMIN + 1] = f.ID('minecraft:water[level=0]')


# ---------------------------------------------------------------- горы
def mountains(f, M, peak, base_low, R_edge, extra_w=None, cap=None, lowland=None, title='горы', snow_y=140, ring=None, inner=None):
    """lowland: None | 'north' | 'south' — где на участке луг (снежный); ring — суша вокруг участка, где джунглей быть не должно (склоны гор
    до моря, ели); inner — джунгли «за горами»: там у подножия джунгли, снаружи — ели."""
    banned = bans(f)
    M0 = M
    if ring is not None: M = M | ring
    if inner is None: inner = np.zeros_like(M)
    free = M & ~banned
    land_in = free & ~f.WAT & (f.GY >= SEA)
    hv = np.where(f.WAT | (f.GY < SEA), SEA - 3, f.GY).astype(float)
    hN = smooth_outside(hv, ~land_in)
    e = ndimage.distance_transform_edt(land_in)
    s = sstep(0, R_edge, e)
    if extra_w is not None: s = s*extra_w
    n1, n2, n3 = f.noise(16, rng), f.noise(40, rng), f.noise(6, rng)
    P = 0.55*(1 - np.abs(2*n1 - 1)) + 0.3*n2 + 0.15*(1 - np.abs(2*n3 - 1))
    core = land_in & (s > 0.9)
    Pk = base_low + (peak - base_low)*P/(P[core].max() if core.any() else P.max())
    H = hN*(1 - s) + Pk*s
    low = np.zeros_like(land_in); lowering = np.zeros_like(land_in)
    pond = None
    if lowland:
        zs = f.ZZ[M0 & land_in]; zmin, zmax = zs.min(), zs.max()
        side = (f.ZZ < zmin + 0.42*(zmax - zmin)) if lowland == 'north' else (f.ZZ > zmax - 0.42*(zmax - zmin))
        zone = land_in & M0 & side & (e > 22)
        lab, n = ndimage.label(zone)
        if n:
            sz = np.bincount(lab.ravel()); sz[0] = 0; zone = lab == sz.argmax()
            w = sstep(0.2, 0.7, ndimage.gaussian_filter(zone.astype(float), 7))*land_in
            target = 136 + 3*(f.noise(10, rng) - 0.5)
            H = H*(1 - w) + target*w
            low = w > 0.5
            lowering = w > 0.03                                # чаша опускает горы плавно по всему переходу, без уступа
            ce = ndimage.distance_transform_edt(zone); ci, cj = np.unravel_index(np.argmax(ce), ce.shape)
            pond = (ci, cj)
            print(f'  луг: {int(low.sum())} столбцов, середина x {f.XX[ci, cj]} z {f.ZZ[ci, cj]}')
    H = np.where(land_in, ndimage.gaussian_filter(H, 1.5), H)
    if cap is not None: H = np.minimum(H, cap)
    Hn = np.rint(np.where(lowering, H, np.maximum(H, f.GY))).astype(int)
    # сдвиг меньше 2 блоков — это шум сглаживания: такие столбцы (и деревья на них) не трогаем
    Hn = np.where(np.abs(Hn - f.GY) >= 2, Hn, f.GY)
    cells = land_in & (Hn != f.GY)
    print(f'{title}: в маске {int(M.sum())}, суши {int(land_in.sum())}, меняется {int(cells.sum())}, верх {Hn[cells].max() if cells.any() else "-"}, '
          f'медиана {np.median(Hn[cells]) if cells.any() else "-"}')
    kill_trees(f, cells, np.minimum(f.G, Hn - YMIN))
    # джунгли снаружи гор (полосы по краям) снимаются целиком, и там, где высота не менялась
    outside = land_in & ~inner & ~cells
    if outside.any(): kill_trees(f, outside, f.G)
    gx, gz = np.gradient(Hn.astype(float)); slope = np.hypot(gx, gz)
    dw = ndimage.distance_transform_edt(~(f.WAT | (f.GY < SEA)))
    sn = L.S.vnoise(f.XX/11.0, f.ZZ/11.0)
    top = np.full(Hn.shape, 'rock', object)
    foot = (s < 0.35) & (Hn < hN + 14) & (slope < 1.3)
    top[foot] = 'grass'
    top[low & (slope < 1.6)] = 'snowgrass'                      # луг заснеженный
    snowy = (Hn >= snow_y + 4*sn) & ((slope < 2.3) | (Hn >= 168)) & ~low
    top[snowy] = 'snow'; top[snowy & (Hn >= 152)] = 'snow2'
    top[land_in & (dw <= 2.5) & (Hn <= SEA + 4)] = 'gravel'
    build(f, cells, Hn, top)
    # озерцо на лугу
    if pond is not None:
        WATER, SAND = f.ID('minecraft:water[level=0]'), f.ID('minecraft:sand')
        ci, cj = pond; lvl = int(Hn[ci, cj]) - YMIN
        rr = 6 + 2*f.noise(5, rng)
        pd = np.hypot(f.XX - f.XX[ci, cj], (f.ZZ - f.ZZ[ci, cj])*1.3) < rr
        for i, j in zip(*np.nonzero(pd & low)):
            col = f.vol[i, :, j]; depth = 1 + int(2*(1 - np.hypot(f.XX[i, j] - f.XX[ci, cj], f.ZZ[i, j] - f.ZZ[ci, cj])/(rr[i, j] + 1)))
            col[lvl + 1:lvl + 60] = 0
            col[lvl - depth:lvl - depth + 1] = SAND
            col[lvl - depth + 1:lvl + 1] = WATER
            col[lvl] = f.ID('minecraft:ice')                       # снежный луг: озерцо подо льдом
            Hn[i, j] = lvl - depth + YMIN
        top[pd & low] = 'pond'
    # валуны на лугу
    if low.any():
        BOU = [f.ID('minecraft:' + k) for k in ('stone', 'andesite', 'mossy_cobblestone', 'cobblestone')]
        for i, j in scatter(np.argwhere(low & (top == 'snowgrass')), 18, 0.25):
            r = 1 + int(rng.random() < 0.5); y0 = Hn[i, j] - YMIN + 1
            for dx in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    for dy in range(0, r + 1):
                        if dx*dx + dz*dz + (dy*1.4)**2 <= r*r + 0.5 and 0 <= i + dx < f.sx and 0 <= j + dz < f.sz:
                            f.vol[i + dx, y0 + dy - 1, j + dz] = BOU[int(rng.integers(4))]
    # ели низкой-средней плотности на склонах и редкие на лугу, кусты на стыке с джунглями
    SPR = spruces(f)
    for t in SPR:
        for *_, k in t: f.ID(k)
    # и на прежних склонах участка (не только там, где рельеф менялся), на снегу тоже — заснеженные ели
    # склоны и снаружи у подножия — ели (снаружи гуще: вместо снятых джунглей), на лугу редкие
    ok = land_in & (top != 'gravel') & (top != 'pond') & (slope < 1.6) & (Hn >= 70) & (Hn <= 165) & ~(foot & inner)
    spots = scatter(np.argwhere(ok & ~low & ~(foot & ~inner)), lambda: rng.uniform(7, 12), 0.7) \
        + scatter(np.argwhere(ok & foot & ~inner), lambda: rng.uniform(5, 9), 0.75) + scatter(np.argwhere(ok & low), lambda: rng.uniform(13, 18), 0.5)
    for i, j in spots: place(f, i, j, Hn[i, j] - YMIN + 1, SPR[int(rng.integers(8))], int(rng.integers(4)))
    # у подножия со стороны джунглей (за горами), где деревья сняты, — снова джунгли
    jt = scatter(np.argwhere(cells & foot & inner & (slope < 1.2)), lambda: rng.uniform(5, 8), 0.8)
    for i, j in jt: place(f, i, j, Hn[i, j] - YMIN + 1, jungle_tree())
    bs = scatter(np.argwhere(cells & foot & inner & (slope < 1.0)), lambda: rng.uniform(3, 6), 0.6)
    for i, j in bs: place(f, i, j, Hn[i, j] - YMIN + 1, bush('jungle'))
    plants(f, land_in & foot & ~inner & (top == 'grass'), Hn, 0.2, 0.01, ('fern',))
    plants(f, cells & foot & inner, Hn, 0.2, 0.01, ('fern',))
    print(f'  елей {len(spots)}, кустов {len(bs)}')
    return Hn, cells


def north():
    f = L.Frame(world, 96, -512, 448, -224).surface()
    M = mask(f, 1)
    inner = ndimage.binary_dilation(polygon_mask(f, 'джунгли') & ~M, structure=disk(6))
    land = ~f.WAT & (f.GY >= SEA)
    # полосы джунглей вокруг гор (у моря, на западе, на юге) — склоны гор; джунгли остаются только внутри, за горами
    ring = land & ndimage.binary_dilation(M, structure=disk(45)) & ~M & ~inner & ~mask(f, 3)
    mountains(f, M, 186, 125, 60, lowland='south', title='северные горы', ring=ring, inner=inner)
    finish(f, 'северные горы')


def se():
    f = L.Frame(world, 352, -176, 640, 128).surface()
    zas = polygon_mask(f, 'засада_туземцев')
    dz = ndimage.distance_transform_edt(~zas)
    dz = dz + 14*(f.noise(9, rng) - 0.5)                     # край подъёма у засады волнистый, а не по линейке
    inner = ndimage.binary_dilation(polygon_mask(f, 'джунгли'), structure=disk(6))   # участок внутри джунглей: у подножия джунгли, их не снимать
    mountains(f, mask(f, 2), 160, 124, 25, extra_w=sstep(45, 75, dz), cap=balloon_cap(f), title='юго-восточные горы', snow_y=152, inner=inner)
    finish(f, 'юго-восточные горы')


# ---------------------------------------------------------------- холмы
def hills():
    f = L.Frame(world, 16, -448, 224, -240).surface()
    M = mask(f, 3)
    # узкие протоки между маской и соседней сушей тоже засыпать (иначе у гор остаётся полоска воды)
    landnow = ~f.WAT & (f.GY >= SEA)
    M = M | (ndimage.binary_closing(M | landnow, structure=disk(9)) & ~landnow & ndimage.binary_dilation(M, structure=disk(14)))
    free = M & ~bans(f)
    hv = np.where(f.WAT | (f.GY < SEA), SEA - 3, f.GY).astype(float)
    hN = smooth_outside(hv, ~free)
    e = ndimage.distance_transform_edt(free)
    n = 0.6*f.noise(14, rng) + 0.4*f.noise(30, rng)
    Ht = 70 + 30*(n - n[free].min())/max(1e-6, n[free].max() - n[free].min())
    s = sstep(0, 30, e)
    H = np.where(free, ndimage.gaussian_filter(hN*(1 - s) + Ht*s, 2.0), f.GY)
    land = free & (H >= SEA + 0.5)
    Hn = np.rint(np.where(land, np.maximum(H, np.where(f.WAT, 0, f.GY)), f.GY)).astype(int)
    cells = land & ((Hn != f.GY) | f.WAT)
    print(f'холмы: в маске {int(M.sum())}, суша {int(land.sum())}, меняется {int(cells.sum())}, высоты {np.percentile(Hn[land], [5, 50, 95])}')
    kill_trees(f, cells & ~f.WAT, np.minimum(f.G, Hn - YMIN))
    top = np.full(Hn.shape, 'grass', object)
    top[land & (Hn <= SEA + 2)] = 'sand'
    build(f, cells, Hn, top)
    # подводный склон у новой кромки
    newland = land & f.WAT
    d_out, nearL = ndimage.distance_transform_edt(~land, return_indices=True)
    from_new = newland[nearL[0], nearL[1]]
    bed = np.rint(SEA - 1 - 0.5*(d_out - 1) - 2*f.noise(6, rng)).astype(int)
    shelf = ~land & from_new & (d_out <= 30) & (bed > f.GY) & (f.WAT | (f.GY < SEA))
    GRAV, SAND, STONE = f.ID('minecraft:gravel'), f.ID('minecraft:sand'), f.ID('minecraft:stone')
    for i, j in zip(*np.nonzero(shelf)):
        b_ = bed[i, j] - YMIN; col = f.vol[i, :, j]
        col[f.G[i, j] + 1:b_ - 1] = STONE
        col[b_ - 1:b_ + 1] = SAND if d_out[i, j] < 8 else GRAV
    print(f'  подводный склон {int(shelf.sum())}')
    grass = cells & (top == 'grass')
    trees = scatter(np.argwhere(grass & (Hn >= 68)), lambda: rng.uniform(9, 16), 0.5)
    for i, j in trees: place(f, i, j, Hn[i, j] - YMIN + 1, broadleaf('birch' if rng.random() < 0.4 else 'oak'))
    plants(f, grass, Hn, 0.22, 0.05)
    print(f'  деревьев {len(trees)}')
    finish(f, 'холмы')


# ---------------------------------------------------------------- стык прежних гор с джунглями
def blend():
    f = L.Frame(world, 320, -480, 448, 112).surface()
    banned = bans(f) | ndimage.binary_dilation(polygon_mask(f, 'засада_туземцев'), structure=disk(10))
    water = f.WAT | (f.GY < SEA)
    rocky = np.isin(f.SURF, ROCKS) & ~water
    soil = (np.isin(f.SURF, SOILS) | ~np.isin(f.SURF, ROCKS)) & ~water & ~rocky
    d_r = ndimage.distance_transform_edt(~rocky); d_s = ndimage.distance_transform_edt(~soil)
    # стенка: гора выше джунглей рядом больше чем на 8
    jl = ndimage.median_filter(np.where(soil, f.GY, 0), size=9)
    hv = np.where(water, SEA, f.GY).astype(float)
    T = ndimage.gaussian_filter(hv, 10)
    w_j = (1 - sstep(0, 30, d_r))*soil
    w_r = (1 - sstep(0, 12, d_s))*rocky
    H = f.GY + (T - f.GY)*np.where(soil, w_j, w_r)
    H = np.where(soil, np.maximum(H, f.GY), H)            # джунгли только поднимаем (предгорья), горы у стыка только опускаем
    H = np.where(rocky, np.minimum(H, f.GY), H)
    H = np.minimum(H, balloon_cap(f).clip(min=f.GY))
    Hn = np.rint(H).astype(int)
    cells = ~banned & ~water & (np.abs(Hn - f.GY) >= 1) & ((w_j > 0.02) | (w_r > 0.02))
    print(f'стык горы–джунгли: меняется {int(cells.sum())} (поднято {int((cells & (Hn > f.GY)).sum())}, опущено {int((cells & (Hn < f.GY)).sum())}), '
          f'наибольший сдвиг {int(np.abs(Hn - f.GY)[cells].max()) if cells.any() else 0}')
    kill_trees(f, cells, np.minimum(f.G, Hn - YMIN))
    gx, gz = np.gradient(Hn.astype(float)); slope = np.hypot(gx, gz)
    top = np.full(Hn.shape, 'rock', object)
    top[soil | (Hn < T + 6) & (slope < 1.4)] = 'grass'
    sn = L.S.vnoise(f.XX/11.0, f.ZZ/11.0)
    snowy = rocky & (Hn >= 140 + 4*sn) & (slope < 2.3)
    top[snowy] = 'snow'
    build(f, cells, Hn, top)
    bs = scatter(np.argwhere(cells & (top == 'grass') & (slope < 1.0)), lambda: rng.uniform(5, 9), 0.5)
    for i, j in bs: place(f, i, j, Hn[i, j] - YMIN + 1, bush('jungle'))
    plants(f, cells & (top == 'grass'), Hn, 0.2, 0.01, ('fern',))
    print(f'  кустов {len(bs)}')
    finish(f, 'стык горы–джунгли')


# ---------------------------------------------------------------- восточный обрыв джунглей
def jcliff():
    f = L.Frame(world, 432, -480, 672, 112).surface()
    banned = bans(f) | mask(f, 1) | mask(f, 2)
    water = sea(f)                                        # только море: берега реки — не обрыв к морю
    land = ~water
    d_l = ndimage.distance_transform_edt(land)
    W = 16
    inner = land & (d_l > W) & (d_l <= W + 8)
    top = smooth_outside(np.where(inner, f.GY, 0).astype(float), inner)       # высота плато за краем
    zone = land & (d_l <= W) & (top - SEA > 25) & ~banned
    lab, n = ndimage.label(zone); sz = np.bincount(lab.ravel()); sz[0] = 0
    zone &= np.isin(lab, np.nonzero(sz > 200)[0])
    u = np.clip(d_l/W, 0, 1)
    K = np.maximum(3, np.rint((top - SEA)/9))
    wob = 0.6*(f.noise(5, rng) - 0.5) + 0.25*(f.noise(2, rng) - 0.5)
    q = u*K + wob
    g = (np.floor(q) + sstep(0.72, 1.0, q - np.floor(q)))/K
    H = SEA + 2 + (top - SEA - 2)*np.clip(g, 0, 1)
    Hn = np.rint(np.minimum(H, f.GY)).astype(int)
    cells = zone & (Hn < f.GY)
    print(f'обрыв джунглей: меняется {int(cells.sum())}, полки {int(K[zone].min()) if zone.any() else 0}…{int(K[zone].max()) if zone.any() else 0}')
    kill_trees(f, cells, Hn - YMIN)
    gx, gz = np.gradient(Hn.astype(float)); slope = np.hypot(gx, gz)
    topm = np.full(Hn.shape, 'rock', object)
    ledge = cells & (slope < 1.2)
    topm[ledge] = 'grass'; topm[ledge & (f.noise(4, rng) > 0.62)] = 'moss'
    topm[cells & (Hn <= SEA + 3)] = 'gravel'
    build(f, cells, Hn, topm)
    # полоска у воды: мелководье у обрыва засыпать до уровня моря гравием, кое-где валуны
    d_w = ndimage.distance_transform_edt(water)
    strip = water & (d_w <= 3) & ndimage.binary_dilation(cells, structure=disk(4)) & ~banned
    GRAV, STONE, MCOB = f.ID('minecraft:gravel'), f.ID('minecraft:stone'), f.ID('minecraft:mossy_cobblestone')
    for i, j in zip(*np.nonzero(strip)):
        col = f.vol[i, :, j]; s0 = SEA - YMIN
        col[f.G[i, j] + 1:s0] = STONE; col[s0 - 1:s0 + 1] = GRAV; col[s0 + 1:s0 + 4] = 0
        if rng.random() < 0.08: col[s0 + 1] = STONE if rng.random() < 0.6 else MCOB
    # мох пятнами на гранях, лианы с полок вниз
    bn = f.bn()
    VINE = {d: f.ID('minecraft:vine[east={},north={},south={},up=false,west={}]'.format(*(str(d == k).lower() for k in ('east', 'north', 'south', 'west'))))
            for d in ('east', 'north', 'south', 'west')}
    solid = np.array([b not in L.SOFTN and not b.endswith('leaves') for b in bn])
    nv = 0
    zl = ndimage.binary_dilation(cells, structure=disk(1))
    for i, j in zip(*np.nonzero(zl)):
        if rng.random() > 0.22: continue
        for d, (di, dj) in (('west', (-1, 0)), ('east', (1, 0)), ('north', (0, -1)), ('south', (0, 1))):
            a, b = i + di, j + dj
            if not (0 <= a < f.sx and 0 <= b < f.sz) or not water[a, b] and not cells[a, b]: continue
            # лиана в столбце (a,b), держится за грань столбца (i,j) со стороны, противоположной d: сторона к (i,j)
            side = {'west': 'east', 'east': 'west', 'north': 'south', 'south': 'north'}[d]
            ytop = int(Hn[i, j]) - YMIN; ybot = max(int(Hn[a, b]) - YMIN + 1, SEA - YMIN + 1)
            y = ytop - int(rng.integers(0, 3)); ln = 2 + int(rng.integers(6))
            while ln > 0 and y >= ybot:
                if f.vol[a, y, b] != 0 or not solid[f.vol[i, y, j]]: break
                f.vol[a, y, b] = VINE[side]; y -= 1; ln -= 1; nv += 1
            break
    bs = scatter(np.argwhere(ledge & (topm == 'grass')), lambda: rng.uniform(4, 8), 0.45)
    for i, j in bs: place(f, i, j, Hn[i, j] - YMIN + 1, bush('jungle'))
    print(f'  полоска у воды {int(strip.sum())}, лиан {nv}, кустов на полках {len(bs)}')
    finish(f, 'восточный обрыв джунглей')


# ---------------------------------------------------------------- западный берег леса и край пустыни
def coast():
    f = L.Frame(world, -704, -272, -336, 400).surface()
    water = f.WAT | (f.GY < SEA)
    soil = np.isin(f.SURF, SOILS) & ~water
    beach = np.isin(f.SURF, ('sand', 'sandstone')) & ~water & (f.ZZ >= -168)
    d_g = ndimage.distance_transform_edt(~soil)
    conv = beach & (d_g <= 24)
    # пальмы (стволы jungle_log) у леса и у южного края пустыни
    bn = f.bn(); jl = np.array([b.startswith('minecraft:jungle_log') for b in f.names()])
    base = np.zeros((f.sx, f.sz), bool)
    for i in range(f.sx):
        base[i] = jl[f.vol[i, np.minimum(f.G[i] + 1, NY - 1), np.arange(f.sz)]]
    river = water & (f.ZZ > -230) & (f.ZZ < -150)
    d_riv = ndimage.distance_transform_edt(~river)
    doomed_p = base & ((d_g <= 30) | (d_riv <= 35))
    kill_trees(f, doomed_p, f.G)
    # высота берега: у воды SEA+3…4, к лесу — как лес
    hv = np.where(soil, f.GY, 0).astype(float)
    hS = smooth_outside(hv, soil)
    d_w = ndimage.distance_transform_edt(~water)
    # кромка хаотичнее: высота уступа 3–5 с мелким шумом, кое-где суша выступает в воду на 1–2 блока
    lip = SEA + np.clip(np.rint(3 + 1.6*(f.noise(3, rng) - 0.5) + 1.2*(f.noise(9, rng) - 0.5)), 3, 5)
    jag = water & (ndimage.distance_transform_edt(~conv) <= 2) & (f.noise(2, rng) > 0.6) & (f.GY >= SEA - 6)
    cells = conv | jag
    d_w = ndimage.distance_transform_edt(~(water & ~jag))
    H = np.maximum(lip, np.minimum(hS, lip + 0.25*np.maximum(0, d_w - 2)))
    Hn = np.rint(H).astype(int)
    print(f'западный берег: пляж → трава {int(cells.sum())} столбцов, пальм снято (см. выше)')
    top = np.full(Hn.shape, 'grass', object)
    build(f, cells, Hn, top)
    # уступ к воде — землёй: в крайних столбцах под травой земля до воды, грубая земля пятнами
    DIRT, CD = f.ID('minecraft:dirt'), f.ID('minecraft:coarse_dirt')
    edge = cells & (d_w <= 1.5)
    for i, j in zip(*np.nonzero(edge)):
        col = f.vol[i, :, j]; h = Hn[i, j] - YMIN
        for y in range(SEA - YMIN - 1, h):
            col[y] = CD if rng.random() < 0.3 else DIRT
    plants(f, cells, Hn, 0.25, 0.01, ('fern',))
    # ели заступают прямо на кромку и на бывший пляж
    SPR = spruces(f)
    for t in SPR:
        for *_, k in t: f.ID(k)
    sp = scatter(np.argwhere(cells & (d_w <= 3)), lambda: rng.uniform(5, 10), 0.55) + scatter(np.argwhere(cells & (d_w > 3)), lambda: rng.uniform(6, 10), 0.6)
    for i, j in sp: place(f, i, j, Hn[i, j] - YMIN + 1, SPR[int(rng.integers(8))], int(rng.integers(4)))
    print(f'  выступов в воду {int(jag.sum())}, елей на берегу {len(sp)}')
    finish(f, 'западный берег и край пустыни')


# ---------------------------------------------------------------- ели на прежних горах острова
def spruce():
    """Плитками 256×256 (с полями 16: дерево у края плитки не режется) по всему острову: ели низкой-средней плотности на склонах гор —
    порода или снег, либо трава выше 110; не в джунглях, не у жерла, расщелины и реки, не в «редактировании»; участки north/se/hills уже засажены."""
    total = 0; best = None
    for tx in range(-704, 704, 256):
        for tz in range(-592, 512, 256):
            f = L.Frame(world, tx - 16, tz - 16, tx + 256 + 16, tz + 256 + 16).surface()
            water = f.WAT | (f.GY < SEA)
            gx, gz = np.gradient(f.GY.astype(float)); slope = np.hypot(gx, gz)
            rocky = np.isin(f.SURF, ('stone', 'andesite', 'tuff', 'calcite', 'granite', 'diorite', 'snow_block', 'gravel', 'cobblestone'))
            hill = np.isin(f.SURF, ('grass_block', 'dirt', 'coarse_dirt', 'podzol')) & (f.GY >= 110)
            avoid = bans(f) | polygon_mask(f, 'джунгли') | polygon_mask(f, 'редактирование') | mask(f, 1) | mask(f, 2) | mask(f, 3)
            inner = np.zeros_like(water); inner[16:-16, 16:-16] = True
            ok = inner & ~water & ~avoid & (rocky | hill) & (f.GY >= 96) & (f.GY <= 170) & (slope < 1.6)
            if not ok.any(): continue
            # рядом с уже стоящими деревьями не сажать
            bn = f.bn(); tree = np.array([b.endswith('_log') or b.endswith('leaves') for b in bn])
            near = np.zeros_like(ok)
            for i in range(f.sx): near[i] = tree[f.vol[i, np.minimum(f.G[i] + 1, NY - 1), np.arange(f.sz)]]
            ok &= ~ndimage.binary_dilation(near, structure=disk(5))
            SPR = spruces(f)
            for t in SPR:
                for *_, k in t: f.ID(k)
            sp = scatter(np.argwhere(ok), lambda: rng.uniform(7, 12), 0.7)
            for i, j in sp: place(f, i, j, f.G[i, j] + 1, SPR[int(rng.integers(8))], int(rng.integers(4)))
            if not sp: continue
            total += len(sp)
            print(f'плитка x {tx} z {tz}: елей {len(sp)}')
            if best is None or len(sp) > best[0]: best = (len(sp), f.top_view(f.orig), f.top_view())
            if WRITE:
                if not os.path.isdir(BACKUP_HINT()): sys.exit('нет копии')
                f.write()
    print('елей всего', total)
    if preview and best:
        _, a, b = best; out = Image.new('RGB', (a.width*2 + 6, a.height)); out.paste(a, (0, 0)); out.paste(b, (a.width + 6, 0)); out.save(preview)
    if not WRITE: print('без --write мир не изменён')


def backup():
    dst = sys.argv[sys.argv.index('--part') + 2]
    os.makedirs(dst, exist_ok=False)
    shutil.copytree(os.path.join(world, 'dimensions/minecraft/overworld/region'), os.path.join(dst, 'region'))
    shutil.copy2(PLACES, os.path.join(dst, 'places.json'))
    open(os.path.join(world, '.landscape_backup_b'), 'w').write(dst)
    print('копия:', dst)


if __name__ == '__main__':
    {'backup': backup, 'north': north, 'se': se, 'hills': hills, 'blend': blend, 'jcliff': jcliff, 'coast': coast, 'spruce': spruce}[part]()
