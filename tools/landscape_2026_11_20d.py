#!/usr/bin/env python3
"""Четвёртая правка ландшафта «Ромашковой Долины» (2026-11-20, по замечаниям к третьей). Пользуется функциями landscape_2026_11_20b.py.

Запуск: landscape_2026_11_20d.py <мир> --part <часть> [--preview файл.png] [--write], игра закрыта. Части: backup <папка> → plateau → valley → terraces → slopes.
  plateau   плато северных гор ниже и холмистее: y ~126 ±6 (было 137 и ровно), озерцо подо льдом заново в низине; вокруг хребты ~146 (на 30–40 ниже
            главных вершин), подъём к ним за 22 блока, дальше за 30 блоков плавно к тому, что за ними (джунгли на востоке, вершины на севере);
            ущелье с запада от холмов (≈126, −336) к краю плато: дно 10 блоков, плавный подъём, стенки не круче 1,4;
  valley    долина на месте пролива плавно переходит в горы и в джунгли (сглаживание в полосе 25 блоков вокруг дна, под шаром не выше коридора);
  terraces  западный склон большого массива (x ~200–400, z −260…−20) — три подъёма по ~13 с полками ~18, сверху ±ровно, дальше к вершинам;
            область «редактирование» (+5, переход до 25) не трогается;
  slopes    по всем горам нет отвесных стен: подъём не больше 3 блоков на блок (только срезая верх), плитками; берега реки в горах сглаживаются,
            русло (3 блока от воды) не трогается. Не трогаются: джунгли (кроме берегов реки у гор), расщелина и тоннель, кратер (26), «редактирование», «руины».
Решения пользователя (2026-11-20): плато ~126 с холмами ±6; ущелье от холмов на западе; ярусы — три и верх, «редактирование» обойти; у реки только берега.
"""
import os, sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import landscape_2026_11_20b as B
from landscape_2026_11_20b import (L, world, preview, WRITE, rng, disk, sstep, NY, YMIN, SEA, mask, bans, sea, polygon_mask, balloon_cap,
                                  kill_trees, build, spruces, place, scatter, plants, smooth_outside, jungle_tree, bush, finish, river_nodes)
from PIL import Image

part = sys.argv[sys.argv.index('--part') + 1]
ROCKISH = ('stone', 'andesite', 'tuff', 'calcite', 'granite', 'diorite', 'gravel', 'snow_block', 'cobblestone', 'packed_ice', 'ice', 'deepslate')


def tops_like_before(f, Hn, cells, snow_y=140):
    """Верх после правки: снег выше снеговой линии на некрутом, трава где была трава или ниже ~118 полого, иначе порода; снежная трава на плато."""
    gx, gz = np.gradient(Hn.astype(float)); slope = np.hypot(gx, gz)
    sn = L.S.vnoise(f.XX/11.0, f.ZZ/11.0)
    top = np.full(Hn.shape, 'rock', object)
    was_grass = np.isin(f.SURF, ('grass_block', 'dirt', 'podzol', 'coarse_dirt', 'moss_block'))
    top[(was_grass | (Hn < 118)) & (slope < 1.3)] = 'grass'
    snowy = (Hn >= snow_y + 4*sn) & ((slope < 2.3) | (Hn >= 168))
    top[snowy] = 'snow'; top[snowy & (Hn >= 152)] = 'snow2'
    return top, slope


# ---------------------------------------------------------------- плато
def plateau():
    f = L.Frame(world, 32, -528, 464, -208).surface()
    banned = bans(f)
    # запрет у расщелины здесь с x 352 (её западный конец), а не 338: иначе у плато оставалась полоса старого хребта и не было перехода в джунгли
    banned &= ~((f.XX >= B.CHASM[0]) & (f.XX < 352) & (f.ZZ >= B.CHASM[2]) & (f.ZZ <= B.CHASM[3]) & (np.hypot(f.XX - B.VENT[0], f.ZZ - B.VENT[1]) >= B.VENT_R))
    seaw = sea(f)
    nm = f.names(); snowyg = np.array(['grass_block[snowy=true]' in k for k in nm])
    topb = np.zeros((f.sx, f.sz), bool)
    for i in range(f.sx): topb[i] = snowyg[f.vol[i, f.G[i], np.arange(f.sz)]]
    box = (f.XX >= 160) & (f.XX <= 340) & (f.ZZ >= -370) & (f.ZZ <= -240)
    P = box & ((topb & (f.GY >= 128) & (f.GY <= 146)) | (f.WAT & (f.GY >= 128)))
    P = ndimage.binary_fill_holes(ndimage.binary_closing(P, structure=disk(5)))
    lab, n = ndimage.label(P); sz = np.bincount(lab.ravel()); sz[0] = 0; P = ndimage.binary_fill_holes(lab == sz.argmax())
    print(f'плато: {int(P.sum())} столбцов, x {f.XX[P].min()}..{f.XX[P].max()}, z {f.ZZ[P].min()}..{f.ZZ[P].max()}')
    hn = 0.6*f.noise(10, rng) + 0.4*f.noise(22, rng); hn = (hn - hn[P].mean())/max(1e-6, np.abs(hn[P] - hn[P].mean()).max())
    Hp = 126 + 6*hn
    dP = ndimage.distance_transform_edt(~P)
    E = f.GY.astype(float)                                           # то, что за хребтами, как есть
    crest = 146 + 5*(f.noise(12, rng) - 0.5)*2
    edge = smooth_outside(np.where(P, Hp, 0.0), P)                   # высота плато у его края, продолженная наружу
    up = edge + (crest - edge)*sstep(0, 22, dP)
    t = sstep(22, 52, dP)
    H = np.where(P, Hp, np.where(dP <= 22, up, crest*(1 - t) + E*t))
    zone = (dP <= 52) & ~seaw
    # ущелье с запада: путь от холмов к краю плато
    path = np.array([[126, -336], [150, -341], [172, -334], [200, -329]], float)   # от холмов (~92) сквозь западный хребет к плато
    gx0, gz0 = f.XX + 0.5, f.ZZ + 0.5
    dmin = np.full(f.XX.shape, 1e9); tpos = np.zeros(f.XX.shape); acc = 0.0
    segs = list(zip(path[:-1], path[1:])); total = sum(np.hypot(*(b - a)) for a, b in segs)
    for a, b in segs:
        dx, dz = b - a; L2 = dx*dx + dz*dz
        tt = np.clip(((gx0 - a[0])*dx + (gz0 - a[1])*dz)/L2, 0, 1)
        d = np.hypot(gx0 - a[0] - tt*dx, gz0 - a[1] - tt*dz)
        better = d < dmin
        dmin = np.where(better, d, dmin); tpos = np.where(better, (acc + tt*np.sqrt(L2))/total, tpos)
        acc += np.sqrt(L2)
    i0, j0 = int(path[0, 0]) - f.X0, int(path[0, 1]) - f.Z0
    start_h = float(np.median(f.GY[i0 - 3:i0 + 4, j0 - 3:j0 + 4]))
    floor = start_h + (126 - start_h)*sstep(0, 1, tpos) + 1.5*(f.noise(4, rng) - 0.5)
    gorge = dmin <= 34
    Hg = floor + 1.4*np.maximum(0, dmin - 5)
    H = np.where(gorge, np.minimum(np.where(zone, H, E), Hg), H)
    zone |= gorge & ~seaw
    print(f'ущелье: начало y {start_h:.0f} → плато 126, длина {total:.0f}')
    # к запретам (кратер, расщелина, река) — плавно к тому, что есть
    db = ndimage.distance_transform_edt(~banned)
    H = E + (H - E)*sstep(0, 20, db)
    H = np.where(zone, ndimage.gaussian_filter(H, 1.2), E)
    H = np.where(P, Hp, H)
    Hn = np.rint(H).astype(int)
    cells = zone & ~banned & (np.abs(Hn - f.GY) >= 1)
    cells |= P & f.WAT                                                 # старое озерцо: убрать (вода над новым дном)
    print(f'плато и хребты: меняется {int(cells.sum())}, хребет {np.percentile(Hn[(dP > 15) & (dP < 26) & zone], [10, 50, 90])}')
    kill_trees(f, cells, np.minimum(f.G, Hn - YMIN))
    # старая вода озерца: столбцы плато с водой — снять воду и лёд до нового дна
    WATERS = np.array([b in L.WATERS or b == 'ice' for b in f.bn()])
    for i, j in zip(*np.nonzero(cells & P)):
        col = f.vol[i, :, j]; w = WATERS[col]; col[w] = 0
    f.G = np.where(cells & P & f.WAT, np.minimum(f.G, Hn - YMIN - 1), f.G)
    top, slope = tops_like_before(f, Hn, cells)
    top[P & (slope < 1.8)] = 'snowgrass'
    top[gorge & (dmin <= 5)] = np.where(Hn[gorge & (dmin <= 5)] >= 120, 'snowgrass', 'grass')
    build(f, cells, Hn, top)
    # новое озерцо подо льдом в самой низкой части плато
    low = np.where(P & (ndimage.distance_transform_edt(P) > 10), Hn, 999)
    ci, cj = np.unravel_index(np.argmin(low), low.shape); lvl = int(Hn[ci, cj]) - YMIN
    WATER, SAND, ICE = f.ID('minecraft:water[level=0]'), f.ID('minecraft:sand'), f.ID('minecraft:ice')
    pd = (np.hypot(f.XX - f.XX[ci, cj], (f.ZZ - f.ZZ[ci, cj])*1.3) < 6 + 2*f.noise(5, rng)) & P
    for i, j in zip(*np.nonzero(pd)):
        col = f.vol[i, :, j]; depth = 1 + int(2*(1 - np.hypot(f.XX[i, j] - f.XX[ci, cj], f.ZZ[i, j] - f.ZZ[ci, cj])/8))
        col[lvl + 1:lvl + 40] = 0; col[lvl - depth] = SAND; col[lvl - depth + 1:lvl + 1] = WATER; col[lvl] = ICE
    print(f'  озерцо у x {f.XX[ci, cj]} z {f.ZZ[ci, cj]} y {lvl + YMIN}')
    SPR = spruces(f)
    for tt_ in SPR:
        for *_, k in tt_: f.ID(k)
    sp = scatter(np.argwhere(cells & P & ~pd & (top == 'snowgrass')), lambda: rng.uniform(13, 19), 0.5) + \
        scatter(np.argwhere(cells & ~P & np.isin(top, ['rock', 'snow', 'grass']) & (slope < 1.6) & (Hn <= 160)), lambda: rng.uniform(8, 13), 0.6)
    for i, j in sp: place(f, i, j, Hn[i, j] - YMIN + 1, SPR[int(rng.integers(8))], int(rng.integers(4)))
    print(f'  елей {len(sp)}')
    finish(f, 'плато, хребты, ущелье')


# ---------------------------------------------------------------- долина
def valley():
    f = L.Frame(world, 288, -160, 448, 176).surface()
    banned = bans(f)
    water = f.WAT | (f.GY < SEA)
    nm = f.names()
    # дно долины прошлой правки: суша 66–86 в полосе бывшего пролива
    V = ~water & (f.XX >= 355) & (f.XX <= 405) & (f.ZZ >= -90) & (f.ZZ <= 100) & (f.GY <= 86)
    lab, n = ndimage.label(V); sz = np.bincount(lab.ravel()); sz[0] = 0; V = lab == sz.argmax()
    dV = ndimage.distance_transform_edt(~V)
    band = (dV <= 26) & ~water & ~banned
    hv = np.where(water, SEA, f.GY).astype(float)
    T = ndimage.gaussian_filter(hv, 9)
    w = (1 - sstep(4, 26, dV))*band
    H = f.GY + (T - f.GY)*w
    H = np.where(V, np.maximum(f.GY, np.minimum(T, f.GY + 6)), H)    # дно чуть поднимается к краям, не ниже прежнего
    H = np.minimum(H, np.maximum(balloon_cap(f), np.minimum(f.GY, H)))
    Hn = np.rint(H).astype(int)
    cells = band & (np.abs(Hn - f.GY) >= 1)
    print(f'долина: полоса {int(band.sum())}, меняется {int(cells.sum())}, наибольший сдвиг {int(np.abs(Hn - f.GY)[cells].max()) if cells.any() else 0}')
    kill_trees(f, cells, np.minimum(f.G, Hn - YMIN))
    top, slope = tops_like_before(f, Hn, cells, 160)
    inner = ndimage.binary_dilation(polygon_mask(f, 'джунгли'), structure=disk(4))
    top[cells & (Hn < 125) & (slope < 1.5)] = 'grass'
    build(f, cells, Hn, top)
    jt = scatter(np.argwhere(cells & inner & (top == 'grass') & (slope < 1.2)), lambda: rng.uniform(5, 8), 0.7)
    for i, j in jt: place(f, i, j, Hn[i, j] - YMIN + 1, jungle_tree())
    plants(f, cells & (top == 'grass'), Hn, 0.2, 0.03)
    print(f'  джунглевых деревьев на переходе {len(jt)}')
    finish(f, 'долина: переходы')


# ---------------------------------------------------------------- ярусы
def terraces():
    f = L.Frame(world, 176, -272, 416, -16).surface()
    banned = bans(f)
    red = polygon_mask(f, 'редактирование')
    dred = ndimage.distance_transform_edt(~red)
    water = f.WAT | (f.GY < SEA)
    foot = ~water & (f.GY < 95) & np.isin(f.SURF, ('grass_block', 'dirt', 'podzol', 'coarse_dirt', 'sand'))
    lab, n = ndimage.label(foot); sz = np.bincount(lab.ravel()); sz[0] = 0
    foot = np.isin(lab, np.nonzero(sz > 2000)[0])                     # луга, а не карманы в горах
    massif = ~water & ~foot & (f.GY >= 95)
    lab, n = ndimage.label(massif); sz = np.bincount(lab.ravel()); sz[0] = 0
    massif = lab == sz.argmax()
    df = ndimage.distance_transform_edt(~foot)
    footH = smooth_outside(np.where(foot, f.GY, 0).astype(float), foot)
    # три подъёма по 13 (каждый на 6 блоках), полки по 18, сверху ровно до 100 блоков вглубь, дальше к прежним вершинам
    D, RISE, STEP = 18 + 3*(f.noise(14, rng) - 0.5), 13, 6
    d = df + 3*(f.noise(6, rng) - 0.5)
    prof = np.zeros_like(d)
    for k in range(3):
        prof += RISE*sstep(k*(D + STEP) + D*0.4, k*(D + STEP) + D*0.4 + STEP, d)
    Ht = footH + prof + 1.5*(f.noise(8, rng) - 0.5)
    t = sstep(85, 115, df)
    H = Ht*(1 - t) + f.GY*t
    w = sstep(5, 25, dred)*sstep(0, 15, ndimage.distance_transform_edt(~banned))
    zone = massif & (f.XX <= 400) & (df <= 115) & (w > 0)
    H = f.GY + (H - f.GY)*w
    H = np.where(zone, ndimage.gaussian_filter(H, 0.8), f.GY)
    Hn = np.rint(H).astype(int)
    cells = zone & (np.abs(Hn - f.GY) >= 1)
    print(f'ярусы: склон {int(zone.sum())}, меняется {int(cells.sum())}, полки {np.percentile(prof[zone], [10, 50, 90])}')
    kill_trees(f, cells, np.minimum(f.G, Hn - YMIN))
    top, slope = tops_like_before(f, Hn, cells)
    top[cells & (slope < 1.0) & (Hn < 135)] = 'grass'                  # полки — трава
    build(f, cells, Hn, top)
    SPR = spruces(f)
    for tt_ in SPR:
        for *_, k in tt_: f.ID(k)
    sp = scatter(np.argwhere(cells & (slope < 1.0) & np.isin(top, ['grass', 'snow'])), lambda: rng.uniform(8, 13), 0.6)
    for i, j in sp: place(f, i, j, Hn[i, j] - YMIN + 1, SPR[int(rng.integers(8))], int(rng.integers(4)))
    plants(f, cells & (top == 'grass'), Hn, 0.2, 0.03)
    print(f'  елей на полках {len(sp)}')
    finish(f, 'ярусы на западном склоне')


def terraces2():
    """Ярусы по существующему склону (вторая версия: первая срезала весь массив до ~110, и «редактирование» осталось башней со стенками до 82;
    участок x 176–416, z −208…−16 возвращён из копии): высота над лугами у подножия квантуется ступенями по 13 — полка, крутой подъём, полка —
    три ступени; выше склон положе (превышение над третьей ступенью × 0,6). Сдвиг у ступеней не больше 13, у вершин до ~30;
    к «редактированию» и запретам сходит на нет за 25 блоков."""
    f = L.Frame(world, 176, -224, 416, 0).surface()
    banned = bans(f)
    red = polygon_mask(f, 'редактирование')
    dred = ndimage.distance_transform_edt(~red)
    water = f.WAT | (f.GY < SEA)
    foot = ~water & (f.GY < 95) & np.isin(f.SURF, ('grass_block', 'dirt', 'podzol', 'coarse_dirt', 'sand'))
    lab, n = ndimage.label(foot); sz = np.bincount(lab.ravel()); sz[0] = 0
    foot = np.isin(lab, np.nonzero(sz > 2000)[0])
    footH = smooth_outside(np.where(foot, f.GY, 0).astype(float), foot)
    Gs = ndimage.gaussian_filter(f.GY.astype(float), 2.0)
    rel = np.maximum(0, Gs - footH)
    k = np.floor(np.minimum(rel, 38.999)/13); fr = np.minimum(rel, 38.999)/13 - k
    q = np.where(rel < 39, 13*(k + sstep(0.62, 1.0, fr)), 39 + 0.6*(rel - 39))
    Ht = footH + q
    w = sstep(5, 25, dred)*sstep(0, 15, ndimage.distance_transform_edt(~banned))
    zone = ~water & ~foot & (f.GY >= 80) & (f.XX <= 400) & (w > 0) & ~((f.XX >= 350) & (f.ZZ >= -100))
    H = np.where(zone, f.GY + (Ht - f.GY)*w, f.GY)
    Hn = np.rint(H).astype(int)
    cells = zone & (np.abs(Hn - f.GY) >= 1)
    print(f'ярусы (2): склон {int(zone.sum())}, меняется {int(cells.sum())}, наибольший сдвиг {int(np.abs(Hn - f.GY)[cells].max()) if cells.any() else 0}')
    kill_trees(f, cells, np.minimum(f.G, Hn - YMIN))
    top, slope = tops_like_before(f, Hn, cells)
    top[cells & (slope < 1.0) & (Hn < 135)] = 'grass'
    build(f, cells, Hn, top)
    SPR = spruces(f)
    for tt_ in SPR:
        for *_, kk in tt_: f.ID(kk)
    sp = scatter(np.argwhere(cells & (slope < 1.0) & np.isin(top, ['grass', 'snow'])), lambda: rng.uniform(8, 13), 0.6)
    for i, j in sp: place(f, i, j, Hn[i, j] - YMIN + 1, SPR[int(rng.integers(8))], int(rng.integers(4)))
    plants(f, cells & (top == 'grass'), Hn, 0.2, 0.03)
    print(f'  елей на полках {len(sp)}')
    finish(f, 'ярусы (2)')


# ---------------------------------------------------------------- без отвесных стен
def slopes():
    MAXD = 3
    total = 0; best = None
    rn = river_nodes()
    only = None
    if '--tiles' in sys.argv:
        only = {tuple(int(v) for v in t.split(':')) for t in sys.argv[sys.argv.index('--tiles') + 1].split(',')}
    for tx in range(-448, 704, 192):
        for tz in range(-576, 256, 192):
            if only is not None and (tx, tz) not in only: continue
            f = L.Frame(world, tx - 32, tz - 32, tx + 192 + 32, tz + 192 + 32).surface()
            water = f.WAT | (f.GY < SEA)
            rockish = np.isin(f.SURF, ROCKISH)
            mount = (rockish & (f.GY >= 70)) | (f.GY >= 115)
            # джунгли и их обрыв к морю не трогаем; юго-восточный выступ (z ≥ −95, x ≥ 398) — уже гора, его можно
            jungle = polygon_mask(f, 'джунгли') & ~((f.ZZ >= -95) & (f.XX >= 398))
            jungle |= (f.XX >= 440) & (f.ZZ < -95)
            near_river = np.zeros_like(water)
            for x, z in rn:
                if x < 430:
                    i, j = int(x) - f.X0, int(z) - f.Z0
                    if 0 <= i < f.sx and 0 <= j < f.sz: near_river[i, j] = True
            if near_river.any(): near_river = ndimage.distance_transform_edt(~near_river) <= 30
            chan = ndimage.distance_transform_edt(~water) <= 3
            keep = chan | B.bans(f) & ~near_river | polygon_mask(f, 'редактирование') | polygon_mask(f, 'руины') | (jungle & ~near_river)
            # кратер, расщелина и тоннель — всегда
            keep |= np.hypot(f.XX - B.VENT[0], f.ZZ - B.VENT[1]) < B.VENT_R
            x0, x1, z0, z1 = B.CHASM
            keep |= (f.XX >= x0) & (f.XX <= x1) & (f.ZZ >= z0) & (f.ZZ <= z1)
            edit = (mount | near_river) & ~keep & ~water
            if not edit.any(): continue
            H = f.GY.astype(float).copy()
            fixed = ~edit
            for _ in range(200):
                m = H.copy()
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = np.roll(np.roll(H, di, 0), dj, 1)
                    m = np.minimum(m, nb + MAXD)
                new = np.where(fixed, H, m)
                if np.array_equal(new, H): break
                H = new
            inner = np.zeros_like(edit); inner[32:-32, 32:-32] = True
            Hn = np.rint(H).astype(int)
            cells = edit & inner & (Hn < f.GY)
            if not cells.any(): continue
            kill_trees(f, cells, Hn - YMIN)
            top, slope = tops_like_before(f, Hn, cells)
            build(f, cells, Hn, top)
            n = int(cells.sum()); total += n
            print(f'плитка x {tx} z {tz}: срезано столбцов {n}, наибольший срез {int((f.GY - Hn)[cells].max())}')
            if best is None or n > best[0]: best = (n, f.top_view(f.orig), f.top_view())
            if WRITE:
                if not os.path.isdir(B.BACKUP_HINT()): sys.exit('нет копии')
                f.write()
    print('столбцов срезано всего', total)
    if preview and best:
        _, a, b = best; out = Image.new('RGB', (a.width*2 + 6, a.height)); out.paste(a, (0, 0)); out.paste(b, (a.width + 6, 0)); out.save(preview)
    if not WRITE: print('без --write мир не изменён')


def mend():
    """Досыпка вместо среза у мест, которые трогать нельзя: снаружи «редактирования» (его кромку срезать нельзя — стенка до 22 после ярусов)
    и долина у западной кромки засады (засаду, где садится шар, не трогаем): поднять доступные столбцы до соседа − 3, пологий подъём к кромке.
    И вернуть из копии (--bk <мир-копия>) столбцы, где пропала вода: вершина родника над «редактированием», исток реки."""
    bk = sys.argv[sys.argv.index('--bk') + 1]
    for x0, z0, x1, z1, keepname in ((208, -240, 336, -32, 'редактирование'), (336, -160, 448, -48, 'засада_туземцев')):
        f = L.Frame(world, x0, z0, x1, z1).surface()
        keep = polygon_mask(f, keepname)
        water = f.WAT | (f.GY < SEA)
        near = ndimage.distance_transform_edt(~keep) <= 30
        edit = near & ~keep & ~water & ~bans(f)
        H = f.GY.astype(float)
        for _ in range(200):
            m = H.copy()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                m = np.maximum(m, np.roll(np.roll(H, di, 0), dj, 1) - 3)
            new = np.where(edit, m, H)
            if np.array_equal(new, H): break
            H = new
        if keepname == 'засада_туземцев': H = np.minimum(H, np.maximum(balloon_cap(f), f.GY))
        Hn = np.rint(H).astype(int)
        cells = edit & (Hn > f.GY)
        cells[:2] = cells[-2:] = False; cells[:, :2] = cells[:, -2:] = False
        if not cells.any():
            print(keepname, ': досыпать нечего'); continue
        print(f'у «{keepname}»: досыпано столбцов {int(cells.sum())}, наибольшая досыпка {int((Hn - f.GY)[cells].max())}')
        kill_trees(f, cells, f.G)
        top, slope = tops_like_before(f, Hn, cells)
        build(f, cells, Hn, top)
        finish(f, f'досыпка у «{keepname}»')
    # вода из копии
    for x0, z0, x1, z1 in ((288, -192, 320, -160), (352, -208, 384, -176)):
        a = L.Frame(bk, x0, z0, x1, z1); f = L.Frame(world, x0, z0, x1, z1)
        isw = lambda fr: np.array([L.S.bname(k) in L.WATERS or L.S.bname(k) == 'ice' for k in fr.names()])
        lost = (isw(a)[a.vol] & ~isw(f)[f.vol]).any(1)
        if not lost.any(): continue
        cols = ndimage.binary_dilation(lost, structure=np.ones((3, 3), bool))
        mp = np.array([f.ID(k) for k in a.names()])
        f.vol = np.where(cols[:, None, :], mp[a.vol], f.vol).astype(f.vol.dtype)
        print(f'вода: возвращено столбцов {int(cols.sum())} у x {f.XX[lost].min()}..{f.XX[lost].max()}')
        if WRITE: print('  записано чанков', f.write())


if __name__ == '__main__':
    {'backup': B.backup, 'plateau': plateau, 'valley': valley, 'terraces': terraces, 'terraces2': terraces2, 'slopes': slopes, 'mend': mend}[part]()
