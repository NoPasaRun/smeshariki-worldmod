#!/usr/bin/env python3
"""Правка ландшафта «Ромашковой Долины» по рисунку пользователя (2026-11-20, ~/Изображения/update.png поверх снимка top_map.py).

Запуск: landscape_2026_11_20.py <мир> --part <часть> [--preview файл.png] [--write], игра закрыта (pgrep -f "[K]notClient").
Части запускать по порядку, каждая читает мир заново: backup → gorge → west → desert → mountains → places.
  backup <папка>  копия region/ и places.json перед всем (откат: вернуть оба);
  gorge      северо-запад: ущелье реки вдоль южного края пустыни шире (5 → 8 по воде), каменная гряда южнее реки выше (до ~125), снег выше ~112;
  west       лес на западе: суша по маске, пролив и прежний пляж засыпаны (шва нет), низина +2…+6, пляж у новой кромки, густой ельник;
  desert     песчаная дуга на севере от пустыни к островку с руинами: дюны как у старой пустыни (~78–92), пляж, внутри бухта остаётся водой;
  mountains  западная часть джунглей — горы как главный массив (пики ~185), снег; жерло вулкана (137, −271) и линия вылета к бухте не трогаются;
  places     границы LevelMod: «остров» += новая суша, «джунгли» −= горы.
Без --write мир не меняется (отчёт и предпросмотр). Маски — landscape_2026_11_20_masks.png (пиксель = мир + (1008, 848); 1 лес, 2 пустыня, 3 горы),
сняты с разницы рисунка пользователя и снимка. Не идемпотентно: повторять — только после возврата копии.
Решения пользователя (2026-11-20): лес — засыпать без шва; пустыня — дюны как старая пустыня; горы — как главный массив;
северо-запад — гряда до ~125, ущелье 8.
"""
import gzip, io, json, os, shutil, struct, sys
import numpy as np
from scipy import ndimage
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import landscape as L
from landscape import sstep, NY, YMIN, SEA

world = sys.argv[1]
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
part, preview, WRITE = arg('--part'), arg('--preview'), '--write' in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
PLACES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(world))), 'data/levelmod/worlds', os.path.basename(os.path.abspath(world)), 'places.json')
MASKS = np.array(Image.open(os.path.join(HERE, 'landscape_2026_11_20_masks.png')).convert('RGB'))
MASK_ID = np.zeros(MASKS.shape[:2], np.uint8)
for k, c in ((1, (40, 90, 40)), (2, (230, 210, 150)), (3, (150, 150, 150))): MASK_ID[(MASKS == c).all(2)] = k
rng = np.random.default_rng(2611201)
ROCK = ('stone', 'andesite', 'tuff', 'calcite')
VENT, VENT_R, LAUNCH_TO = (137, -271), 26, (137, -330)     # жерло, запретный радиус, куда летит вылет (бухта)


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]; return x*x + y*y <= r*r


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


def BACKUP_HINT():
    p = os.path.join(world, '.landscape_backup'); return open(p).read().strip() if os.path.exists(p) else ''


# ---------------------------------------------------------------- деревья
def kill_trees(f, doomed, above):
    """Убрать деревья, чьи брёвна стоят в столбцах doomed выше above[x,z] (компоненты связности брёвен целиком),
    потом листву, оставшуюся без бревна ближе 6 шагов (как в игре: иначе висит или осыплется), и лианы/какао без опоры."""
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
    # только около снятых деревьев: у края рамки и у построенных деревьев (постоянная листва дальше 6) брёвна могут быть не видны
    zone = ndimage.binary_dilation(kill, structure=np.ones((3, 3, 3)), iterations=7) if kill.any() else np.zeros_like(kill)
    orphan = leaves & ~reach & zone
    f.vol[orphan] = 0
    # лианы и какао без опоры по соседству
    solid = (f.vol != 0) & ~hang[f.vol]
    near = ndimage.binary_dilation(solid, structure=ndimage.generate_binary_structure(3, 1))
    h = hang[f.vol]
    held = h & near
    for y in range(NY - 2, -1, -1):           # лиана держится и на лиане сверху
        held[:, y, :] |= h[:, y, :] & held[:, y + 1, :]
    f.vol[h & ~held] = 0
    print(f'  деревьев снято {len(bad)}, брёвен {int(kill.sum())}, листвы без опоры {int(orphan.sum())}')


def rock_ids(f):
    """Пласты породы по 3 блока (камень/андезит/туф), граница волнистая: rock(i, j, y0, y1) -> ids для мировых y0..y1."""
    ids = np.array([f.ID('minecraft:' + b) for b in ('stone', 'andesite', 'tuff', 'stone')])
    off = 4*L.S.vnoise(f.XX/23.0, f.ZZ/23.0)
    return lambda i, j, y0, y1: ids[(np.floor((np.arange(y0, y1 + 1) + off[i, j])/3).astype(int)) % 4]


# ---------------------------------------------------------------- северо-запад: ущелье и гряда
def gorge():
    f = L.Frame(world, -352, -272, -128, -128).surface()
    WAT = f.WAT
    wide = ndimage.binary_opening(WAT, structure=disk(5))
    river = WAT & ~ndimage.binary_dilation(wide, structure=disk(6))
    hi = ndimage.maximum_filter(np.where(WAT, 0, f.GY), size=17)
    river &= hi >= 78                                            # только там, где берега высокие — это ущелье
    lab, n = ndimage.label(river); sz = np.bincount(lab.ravel()); sz[0] = 0
    river = lab == sz.argmax()
    print(f'русло в ущелье: {int(river.sum())} столбцов, x {f.XX[river].min()}..{f.XX[river].max()}')
    d = ndimage.distance_transform_edt(~river)
    chan = (d <= 1.6) & ~WAT                                     # 5 → 8 по воде
    # концы плавно: у широкой воды не расширять
    chan &= ndimage.distance_transform_edt(~wide) > 4
    bridge = []
    bn = f.bn(); J = np.arange(f.sz)
    for i, j in zip(*np.nonzero(river)):
        col = bn[f.vol[i, f.G[i, j] + 1:f.G[i, j] + 30, j]]
        if any(b not in L.SOFTN and not b.endswith('leaves') for b in col): bridge.append((f.XX[i, j], f.ZZ[i, j]))
    if bridge: print('  над руслом есть твёрдые блоки (мост?):', bridge[:10], len(bridge))
    kill_trees(f, chan, f.G)
    WATER, SAND, STONE = f.ID('minecraft:water[level=0]'), f.ID('minecraft:sand'), f.ID('minecraft:stone')
    for i, j in zip(*np.nonzero(chan)):
        bed = 59 + int(rng.random() < 0.5) - YMIN; col = f.vol[i, :, j]
        for y in range(bed - 3, bed - 1):
            if col[y] == 0: col[y] = STONE
        col[bed - 2:bed + 1] = SAND
        col[bed + 1:SEA - YMIN + 1] = WATER
        col[SEA - YMIN + 1:f.G[i, j] + 2] = 0                    # берег до прежнего верха и растения на нём; нависающие кроны остаются
    print(f'  ущелье: {int(chan.sum())} столбцов срезано')

    # гряда: самый большой каменный участок южнее реки
    rocky = np.isin(f.SURF, ['stone', 'andesite', 'tuff', 'calcite', 'gravel']) & ~WAT
    lab, n = ndimage.label(rocky); sz = np.bincount(lab.ravel()); sz[0] = 0
    core = lab == sz.argmax()
    print(f'гряда: x {f.XX[core].min()}..{f.XX[core].max()}, z {f.ZZ[core].min()}..{f.ZZ[core].max()}, верх {f.GY[core].max()}')
    zone = ndimage.binary_dilation(core, structure=disk(12)) & ~WAT & (ndimage.distance_transform_edt(~(chan | river)) > 3)
    ring = ndimage.binary_dilation(zone, structure=disk(3)) & ~zone & ~WAT
    base = float(np.median(f.GY[ring]))
    rel = np.clip(f.GY - base, 0, None)*zone
    rel = ndimage.gaussian_filter(rel.astype(float), 1.2)*zone
    H = np.rint(base + (rel/rel.max())**0.6*(125 - base)).astype(int)    # степень < 1: склоны полнее, снега больше
    up = zone & (H > f.GY)
    print(f'  основание {base:.0f}, верх {H[up].max()}, поднято столбцов {int(up.sum())}, выше 108: {int((H[up] >= 108).sum())}')
    kill_trees(f, up, f.G)
    SNOW = f.ID('minecraft:snow_block'); rock = rock_ids(f)
    GRASS, DIRT = f.ID('minecraft:grass_block[snowy=false]'), f.ID('minecraft:dirt')
    sn = L.S.vnoise(f.XX/9.0, f.ZZ/9.0)
    for i, j in zip(*np.nonzero(up)):
        g, h = f.G[i, j], H[i, j] - YMIN; col = f.vol[i, :, j]
        col[g + 1:min(h + 40, NY)] = 0
        if f.SURF[i, j] in ('grass_block', 'dirt') and h - g <= 3:      # подножие: та же трава, чуть выше
            col[g:h] = DIRT; col[h] = GRASS; continue
        col[g - 1:h + 1] = rock(i, j, g - 1 + YMIN, h + YMIN)
        snowline = 108 + 3*sn[i, j]
        if H[i, j] >= snowline:
            col[h] = SNOW
            if H[i, j] >= snowline + 6: col[h - 1] = SNOW
    finish(f, 'северо-запад')


# ---------------------------------------------------------------- суша по маске (лес, пустыня)
def add_land(f, M, kind):
    """Новая суша по маске M (засыпать проливы и узкие протоки к старой суше). Возвращает словарь массивов."""
    old_land = ~f.WAT & (f.GY >= SEA - 1)
    zone = ndimage.binary_dilation(M, structure=disk(14))
    raw = old_land | M
    closed = ndimage.binary_closing(raw, structure=disk(7)) & zone
    sm = ndimage.gaussian_filter((raw | closed).astype(float), 5) > 0.5
    land = old_land | (sm & zone)
    holes = ndimage.binary_fill_holes(land) & ~land
    hl, hn = ndimage.label(holes); hs = np.bincount(hl.ravel())
    for k in range(1, hn + 1):
        if hs[k] < 3000: land |= hl == k            # мелкие ямы засыпать, бухту (большую) оставить
    d_in = ndimage.distance_transform_edt(land)
    d_old = ndimage.distance_transform_edt(old_land)
    if kind == 'west':                     # прежний пляж и поляны стыка (без деревьев) — в лес, шва нет
        bn = f.bn(); tree = np.array([x.endswith('leaves') or x.endswith('_log') for x in bn])
        covered = np.zeros_like(M)
        for i in range(f.sx): covered[i] = tree[f.vol[i]].any(0)
        conv = old_land & ndimage.binary_dilation(M, structure=disk(40)) & (d_in - d_old > 4) & ~covered \
            & np.isin(f.SURF, ['sand', 'grass_block', 'sandstone', 'gravel'])
    else:                                  # пустыня: стык со старой пустыней переделывается шире, чтобы не было уступа
        conv = old_land & ndimage.binary_dilation(M, structure=disk(30)) & (d_in - d_old > 4) & np.isin(f.SURF, ['sand', 'sandstone'])
    new = land & ~old_land
    keep = old_land & ~conv
    e_old, near = ndimage.distance_transform_edt(~keep, return_indices=True)
    hN = f.GY[near[0], near[1]].astype(float)
    # океан — вода, связанная с северным, западным или восточным краем рамки (на юге рамку режет бухта); бухта — остальная вода
    lab, _ = ndimage.label(~land)
    edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0]])) - {0}
    ocean = np.isin(lab, list(edge))
    return dict(old_land=old_land, zone=zone, land=land, d_in=d_in, conv=conv, new=new, keep=keep, e_old=e_old, hN=hN, ocean=ocean)


def build_columns(f, A, H, sandy, top_soil):
    """Столбцы новой суши: всё рыхлое и вода выше — воздух (кроны остаются), ниже — камень, сверху песок или трава.
    Высоты сначала сглаживаются вместе с соседней старой сушей (вода = уровень моря), чтобы на стыке не было ступенек."""
    work = A['new'] | A['conv']
    base = np.where(work, H, np.where(A['land'], f.GY, SEA)).astype(float)
    H[:] = np.where(work, np.rint(ndimage.gaussian_filter(base, 2.0)), H).astype(int)
    H[:] = np.maximum(H, SEA)
    bn = f.bn()
    GRASS, DIRT, STONE, SAND, SST = (f.ID(k) for k in ('minecraft:grass_block[snowy=false]', 'minecraft:dirt', 'minecraft:stone',
                                                       'minecraft:sand', 'minecraft:sandstone'))
    keep_up = np.array([b.endswith('leaves') or b.endswith('_log') for b in bn])
    for i, j in zip(*np.nonzero(work)):
        h = H[i, j] - YMIN; g = f.G[i, j]; col = f.vol[i, :, j]
        top = min(max(g, h) + 45, NY)
        seg = col[h + 1:top]
        col[h + 1:top] = np.where(keep_up[seg], seg, 0)
        lo = min(g, h - 6)
        col[lo + 1:h + 1] = STONE
        if sandy[i, j]:
            col[h - 2:h + 1] = SAND; col[h - 5:h - 2] = SST
        else:
            col[h] = GRASS if top_soil == 'grass' else SAND; col[h - 3:h] = DIRT if top_soil == 'grass' else SAND
    # подводный склон от новой кромки
    land = A['land']
    d_out, nearL = ndimage.distance_transform_edt(~land, return_indices=True)
    from_new = work[nearL[0], nearL[1]]
    bed = np.rint(SEA - 1 - 0.5*(d_out - 1) - 2*f.noise(6, rng)).astype(int)
    shelf = ~land & from_new & (d_out <= 45) & (bed > f.GY)
    GRAV, WATER = f.ID('minecraft:gravel'), f.ID('minecraft:water[level=0]')
    for i, j in zip(*np.nonzero(shelf)):
        b_ = bed[i, j] - YMIN; col = f.vol[i, :, j]
        col[f.G[i, j] + 1:b_ - 1] = STONE
        col[b_ - 1:b_ + 1] = SAND if d_out[i, j] < 12 else GRAV
        col[b_ + 1:SEA - YMIN + 1] = WATER
    print(f'  новая суша {int(A["new"].sum())}, переделано прежней {int(A["conv"].sum())}, подводный склон {int(shelf.sum())}')
    return work


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


PALM_LOG, PALM_LEAF = 'minecraft:jungle_log[axis=y]', 'minecraft:jungle_leaves[distance=1,persistent=true,waterlogged=false]'


def palm():
    """Пальма как в оазисе острова (IslandChunkGenerator.palm)."""
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


def place(f, i, j, y0, t, rot):
    bn = f.bn()
    for dx, dy, dz, k in t:
        for _ in range(rot): dx, dz = -dz, dx
        a, c, b = i + dx, j + dz, y0 + dy
        if not (0 <= a < f.sx and 0 <= c < f.sz and b < NY): continue
        if rot % 2 and '_log' in k: k = k.replace('axis=x', 'axis=Q').replace('axis=z', 'axis=x').replace('axis=Q', 'axis=z')
        cur = f.vol[a, b, c]
        if cur == 0 or ('_log' in k and cur < len(bn) and (bn[cur].endswith('leaves') or bn[cur] in L.SOFTN)):
            f.vol[a, b, c] = f.ID(k)


def scatter(cand, gap, prob=1.0, pts=None):
    from scipy.spatial import cKDTree
    pts = list(pts or []); out = []; tree = cKDTree(pts) if pts else None; fresh = []
    c = cand.copy(); rng.shuffle(c)
    for n_, (i, j) in enumerate(c):
        if rng.random() > prob: continue
        g = gap() if callable(gap) else gap
        if tree is not None and tree.query_ball_point((i, j), g): continue
        if any((i - a)**2 + (j - b)**2 < g*g for a, b in fresh[-300:]): continue
        out.append((i, j)); fresh.append((i, j))
        if len(fresh) > 2000: pts += fresh; tree = cKDTree(pts); fresh = []
    return out


def west():
    f = L.Frame(world, -688, -224, -272, 400).surface()
    M = f.mask(MASK_ID, 1)
    A = add_land(f, M, 'west')
    d_in = A['d_in']
    bw = 9 + 3*(2*f.noise(18, rng) - 1)
    h_t = 64 + 4*(0.55*f.noise(22, rng) + 0.45*f.noise(9, rng))
    s_old = sstep(0, 24, A['e_old'])
    h_in = h_t*s_old + A['hN']*(1 - s_old)
    H = np.rint((SEA + 1.6*np.clip(d_in/np.maximum(bw, 1), 0, 1))*(1 - sstep(bw, bw + 14, d_in)) + h_in*sstep(bw, bw + 14, d_in)).astype(int)
    sandy = d_in < bw
    work = build_columns(f, A, H, sandy, 'grass')
    BLD = os.path.join(world, 'data/buildmod/buildings')
    SPR = [load_building(os.path.join(BLD, f'{n}.bin')) for n in range(911, 919)]
    W_ = np.array([0.5, 0.5, 1, 1, 1.4, 1.4, 1.2, 1.2]); W_ /= W_.sum()
    for t in SPR:
        for *_, k in t: f.ID(k); f.ID(k.replace('axis=x', 'axis=z')); f.ID(k.replace('axis=z', 'axis=x'))
    ok = work & ~sandy & (d_in >= bw + 2)
    bn = f.bn(); islog = np.array([b.endswith('_log') for b in bn])
    J = np.arange(f.sz)
    trunks = [(i, j) for i in range(f.sx) for j in np.nonzero(islog[f.vol[i, np.minimum(f.G[i] + 1, NY - 1), J]])[0]]
    spots = scatter(np.argwhere(ok), 5.0, 1.0, trunks)
    for i, j in spots: place(f, i, j, H[i, j] - YMIN + 1, SPR[rng.choice(8, p=W_)], int(rng.integers(4)))
    print(f'  елей {len(spots)}')
    GRASS, DIRT = f.ID('minecraft:grass_block[snowy=false]'), f.ID('minecraft:dirt')
    SG, FERN = f.ID('minecraft:short_grass'), f.ID('minecraft:fern')
    for i, j in zip(*np.nonzero(work & ~sandy)):
        y = H[i, j] - YMIN + 1
        if f.vol[i, y, j] == 0 and f.vol[i, y - 1, j] == GRASS:
            r = rng.random(); f.vol[i, y, j] = SG if r < 0.06 else FERN if r < 0.09 else 0
        elif f.vol[i, y, j] != 0 and f.vol[i, y - 1, j] == GRASS: f.vol[i, y - 1, j] = DIRT
    finish(f, 'лес на западе')


def desert():
    f = L.Frame(world, -384, -576, 128, -304).surface()
    M = f.mask(MASK_ID, 2)
    A = add_land(f, M, 'desert')
    d_in = A['d_in']
    bw = 8 + 3*(2*f.noise(15, rng) - 1)
    n1 = f.noise(12, rng); n2 = f.noise(30, rng)
    dunes = 0.6*(1 - np.abs(2*n1 - 1)) + 0.4*n2                  # гребни дюн + крупная волна
    # к островку с руинами (восток) ниже: старая пустыня ~86, у островка ~74
    east = sstep(-120, 40, f.XX.astype(float))
    h_t = (84 - 10*east) + 8*(dunes - 0.5)
    rise = sstep(bw, bw + 26, d_in)
    s_old = sstep(0, 40, A['e_old'])
    h_in = h_t*s_old + A['hN']*(1 - s_old)
    H = np.rint((SEA + 1.6*np.clip(d_in/np.maximum(bw, 1), 0, 1))*(1 - rise) + h_in*rise).astype(int)
    sandy = np.ones_like(d_in, bool)
    work = build_columns(f, A, H, sandy, 'sand')
    # кактусы, сухие кусты, пальмы по внешнему пляжу
    CAC, DB = f.ID('minecraft:cactus[age=0]'), f.ID('minecraft:dead_bush')
    for t in (palm(),):
        for *_, k in t: f.ID(k)
    inner = work & (d_in > bw + 6)
    nc = nd = 0
    for i, j in zip(*np.nonzero(inner)):
        r = rng.random(); y = H[i, j] - YMIN + 1
        if f.vol[i, y, j] != 0: continue
        if r < 0.0025:
            for k in range(1 + int(rng.integers(3))): f.vol[i, y + k, j] = CAC
            nc += 1
        elif r < 0.0065: f.vol[i, y, j] = DB; nd += 1
    _, nw = ndimage.distance_transform_edt(A['land'], return_indices=True)
    outer = A['ocean'][nw[0], nw[1]]
    pz = work & outer & (d_in >= bw - 2) & (d_in <= bw + 5)
    spots = scatter(np.argwhere(pz), lambda: rng.uniform(6, 14), 0.5)
    for i, j in spots: place(f, i, j, H[i, j] - YMIN + 1, palm(), 0)
    print(f'  кактусов {nc}, сухих кустов {nd}, пальм {len(spots)}; дюны y {np.percentile(H[inner], [5, 50, 95]) if inner.any() else "-"}')
    finish(f, 'пустыня на севере')


def mountains():
    f = L.Frame(world, 112, -512, 432, -240).surface()
    M = f.mask(MASK_ID, 3)
    M = ndimage.binary_opening(M, structure=disk(2))
    # где поднимать нельзя: жерло и линия вылета к бухте, расщелина
    t = np.clip((f.ZZ - VENT[1])/(LAUNCH_TO[1] - VENT[1]), 0, 1)
    dl = np.hypot(f.XX - VENT[0], f.ZZ - (VENT[1] + t*(LAUNCH_TO[1] - VENT[1])))
    banned = (dl < VENT_R) | ((f.XX >= 340) & (f.ZZ > -330))
    free = M & ~banned
    WATER_COL = f.WAT
    e = ndimage.distance_transform_edt(free & ~WATER_COL)
    # высота окрестности снаружи: сглаженная (нормированная свёртка), иначе «ближайшая точка» даёт лучи; вода — уровень воды
    out = ~(free & ~WATER_COL); hv = np.where(f.WAT, SEA, f.GY).astype(float)
    hN = hv.copy()
    for sig in (6, 12, 24, 48):
        num, den = ndimage.gaussian_filter(hv*out, sig), ndimage.gaussian_filter(out.astype(float), sig)
        hN = np.where(den > 1e-3, num/np.maximum(den, 1e-9), hN) if sig == 6 else np.where(den_prev < 1e-3, num/np.maximum(den, 1e-9), hN)
        den_prev = den
    n1, n2, n3 = f.noise(16, rng), f.noise(40, rng), f.noise(6, rng)
    P = 0.55*(1 - np.abs(2*n1 - 1)) + 0.3*n2 + 0.15*(1 - np.abs(2*n3 - 1))
    s = sstep(0, 45, e)
    core = s > 0.95
    A_ = (186 - 110)/P[core].max()
    Pk = 110 + A_*P
    H = np.rint(np.maximum(hN*(1 - s) + Pk*s, f.GY)).astype(int)
    up = free & (H > f.GY)
    print(f'горы: столбцов в маске {int(M.sum())}, поднято {int(up.sum())}, верх {H[up].max()}, медиана {np.median(H[up]):.0f}')
    kill_trees(f, up, f.G)
    # слоны: крутизна по соседям
    gx, gz = np.gradient(H.astype(float))
    steep = np.hypot(gx, gz)
    SNOW = f.ID('minecraft:snow_block'); rock = rock_ids(f)
    sn = L.S.vnoise(f.XX/11.0, f.ZZ/11.0)
    for i, j in zip(*np.nonzero(up)):
        g, h = f.G[i, j], H[i, j] - YMIN; col = f.vol[i, :, j]
        col[g + 1:min(h + 60, NY)] = 0
        col[g:h + 1] = rock(i, j, g + YMIN, h + YMIN)
        line = 104 + 4*sn[i, j]
        if H[i, j] >= line and (steep[i, j] < 2.3 or H[i, j] >= 165):
            col[h] = SNOW
            if H[i, j] >= 140: col[h - 1] = SNOW
    finish(f, 'горы')


def places():
    P = json.load(open(PLACES))
    sys.path.insert(0, HERE)
    import matplotlib; matplotlib.use('Agg')
    from matplotlib.path import Path
    import matplotlib.pyplot as plt

    def poly(loop, ox, oz, w, h):
        gx, gz = np.meshgrid(np.arange(w) + 0.5 + ox, np.arange(h) + 0.5 + oz, indexing='ij')
        return Path(loop).contains_points(np.c_[gx.ravel(), gz.ravel()]).reshape(w, h)

    def rdp(pts, eps):
        if len(pts) < 3: return pts
        a, b = pts[0], pts[-1]; ab = b - a; n = np.hypot(*ab) or 1
        dd = np.abs(np.cross(ab, pts - a))/n; k = int(np.argmax(dd))
        if dd[k] > eps: return np.vstack([rdp(pts[:k + 1], eps)[:-1], rdp(pts[k:], eps)])
        return np.vstack([a, b])

    def area(mask, ox, oz, ground):
        fm = ndimage.gaussian_filter(mask.astype(float), 1.5)
        cs = plt.contour(np.arange(mask.shape[0]) + 0.5 + ox, np.arange(mask.shape[1]) + 0.5 + oz, fm.T, levels=[0.5])
        seg = max(cs.allsegs[0], key=len); plt.close('all')
        if np.allclose(seg[0], seg[-1]): seg = seg[:-1]
        m = int(np.argmax(np.hypot(*(seg - seg[0]).T)))
        s = np.vstack([rdp(seg[:m + 1], 0.8)[:-1], rdp(np.vstack([seg[m:], seg[:1]]), 0.8)[:-1]])
        pts = [[int(np.floor(x)), int(ground(x, z)), int(np.floor(z))] for x, z in s]
        ded = [p for k, p in enumerate(pts) if k == 0 or p[0::2] != pts[k - 1][0::2]]
        if ded[0][0::2] == ded[-1][0::2]: ded.pop()
        n = len(ded)
        return {'dimension': 'minecraft:overworld', 'loops': [[[p[0] + 0.5, p[2] + 0.5] for p in ded]], 'points': ded,
                'edges': [[k, (k + 1) % n, 0, 0] for k in range(n)]}

    # суша мира крупно: весь остров одной рамкой по картам высот чанков
    X0, Z0, X1, Z1 = -704, -592, 704, 512
    land = np.zeros((X1 - X0, Z1 - Z0), bool); GY = np.zeros_like(land, np.int32)
    import mca
    reg = os.path.join(world, 'dimensions/minecraft/overworld/region'); R = {}
    for cx in range(X0 >> 4, X1 >> 4):
        for cz in range(Z0 >> 4, Z1 >> 4):
            p = os.path.join(reg, f'r.{cx >> 5}.{cz >> 5}.mca')
            if p not in R: R[p] = mca.Region(p)
            raw = R[p].raw(cx, cz)
            if raw is None: continue
            _, c = mca.parse(raw)
            hm = c[b'Heightmaps'][1]
            def unpackhm(b_):
                w = np.frombuffer(b_, '>u8').astype(np.uint64)
                q = np.arange(256)
                return ((w[q//7] >> ((q % 7)*9).astype(np.uint64)) & np.uint64(511)).astype(np.int32).reshape(16, 16)  # [z, x]
            floor = unpackhm(hm[b'OCEAN_FLOOR'][1]) + YMIN; surf = unpackhm(hm[b'MOTION_BLOCKING_NO_LEAVES'][1]) + YMIN
            a, b = (cx << 4) - X0, (cz << 4) - Z0
            land[a:a + 16, b:b + 16] = (floor >= surf).T & (floor.T >= SEA)
            GY[a:a + 16, b:b + 16] = surf.T
    old = P['areas']['остров']; IL = np.array(old['loops'][0])
    isl = poly(IL, X0, Z0, X1 - X0, Z1 - Z0)
    lab, _ = ndimage.label(land | isl)
    ci = lab[int(np.mean(IL[:, 0])) - X0, int(np.mean(IL[:, 1])) - Z0]
    lw = ndimage.binary_dilation(ndimage.binary_closing(land, structure=disk(3)), structure=disk(1))
    newisl = isl | (lw & (lab == ci))
    newisl = ndimage.binary_fill_holes(ndimage.binary_opening(newisl, structure=disk(3)))
    ground = lambda x, z: max(SEA, GY[int(np.floor(x)) - X0, int(np.floor(z)) - Z0])
    a_isl = area(newisl, X0, Z0, ground)
    jl = P['areas']['джунгли']
    jm = poly(np.array(jl['loops'][0]), X0, Z0, X1 - X0, Z1 - Z0)
    xx, zz = np.meshgrid(np.arange(X1 - X0) + X0, np.arange(Z1 - Z0) + Z0, indexing='ij')
    px, pz = xx + 1008, zz + 848
    Mm = np.zeros_like(jm); ok = (px >= 0) & (px < MASK_ID.shape[1]) & (pz >= 0) & (pz < MASK_ID.shape[0])
    Mm[ok] = MASK_ID[pz[ok], px[ok]] == 3
    jnew = ndimage.binary_opening(jm & ~ndimage.binary_dilation(Mm, structure=disk(2)), structure=disk(3))
    a_jun = area(jnew, X0, Z0, ground)
    print(f'«остров»: {len(old["points"])} → {len(a_isl["points"])} точек, x {min(p[0] for p in a_isl["points"])}..{max(p[0] for p in a_isl["points"])}, '
          f'z {min(p[2] for p in a_isl["points"])}..{max(p[2] for p in a_isl["points"])}')
    print(f'«джунгли»: {len(jl["points"])} → {len(a_jun["points"])} точек, площадь {int(jm.sum())} → {int(jnew.sum())}')
    if preview:
        img = np.zeros((Z1 - Z0, X1 - X0, 3), np.uint8)
        img[land.T] = (80, 140, 60); img[isl.T & ~newisl.T] = (200, 0, 0)
        from PIL import ImageDraw
        im = Image.fromarray(img); d = ImageDraw.Draw(im)
        for ar, cl in ((old, (255, 255, 0)), (a_isl, (255, 0, 0)), (jl, (0, 255, 255)), (a_jun, (255, 0, 255))):
            Lp = ar['loops'][0]; d.line([(x - X0, z - Z0) for x, z in Lp + [Lp[0]]], fill=cl, width=2)
        im.save(preview); print('предпросмотр:', preview)
    if WRITE:
        if not os.path.isdir(BACKUP_HINT()): sys.exit('нет копии')
        P['areas']['остров'] = a_isl; P['areas']['джунгли'] = a_jun
        tmp = PLACES + '.tmp'
        with open(tmp, 'w') as fo: json.dump(P, fo, ensure_ascii=False)
        os.replace(tmp, PLACES); print('записано', PLACES)


def backup():
    dst = sys.argv[sys.argv.index('--part') + 2]
    os.makedirs(dst, exist_ok=False)
    shutil.copytree(os.path.join(world, 'dimensions/minecraft/overworld/region'), os.path.join(dst, 'region'))
    shutil.copy2(PLACES, os.path.join(dst, 'places.json'))
    open(os.path.join(world, '.landscape_backup'), 'w').write(dst)
    print('копия:', dst)


{'backup': backup, 'gorge': gorge, 'west': west, 'desert': desert, 'mountains': mountains, 'places': places}[part]()
