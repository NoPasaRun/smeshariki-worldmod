#!/usr/bin/env python3
"""Пятая правка ландшафта «Ромашковой Долины» (2026-11-20, по замечаниям к четвёртой). Пользуется функциями landscape_2026_11_20b/d.

Запуск: landscape_2026_11_20e.py <мир> --part <часть> [--preview файл.png] [--write], игра закрыта. Части: backup <папка> → kusok → ridge →
terraces3 → caps → spikes (--orig <копия-мир до правки 2>).
  kusok      локация «кусок_джунглей» (нарисована пользователем): джунгли сняты, подножие горы — плавный подъём от долины к юго-восточной горе,
             трава и камень, ели обычной редкости; под шаром — коридор;
  ridge      плато скруглено (без углов), хребет вокруг заново: изнутри подъём за 22 блока, снаружи плавный спуск за ~60 к тому, что за ним;
             ущелье с запада не засыпается;
  terraces3  чёткие ярусы на западном склоне большого массива: полки по горизонталям склона (+12 на ярус, три яруса), подъёмы ровно 2,5 блока
             на блок (ограничение крутизны их не срезает), на полках трава, на подъёмах камень; выше третьего яруса склон положе (×0,6);
             «редактирование» не трогается (сходит на нет за 25 блоков);
  caps       гладкие снежные шапки: вершины выше 135 чуть сглажены (σ 2), снеговая линия плавная (~138 ± 8, крупный шум), снег только на некрутом;
             во всех горах, кроме «редактирования», «руин», кратера и расщелины;
  spikes     «колья» — узкие столбы (уже 3 блоков) выше окружения на 3+ — срезаются до окружения, только в столбцах, изменённых правками 2–5
             (сравнение с копией до второй правки), не у построек.
"""
import os, sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import landscape_2026_11_20b as B
import landscape_2026_11_20d as D
from landscape_2026_11_20b import (L, world, preview, WRITE, rng, disk, sstep, NY, YMIN, SEA, mask, bans, sea, polygon_mask, balloon_cap,
                                  kill_trees, build, spruces, place, scatter, plants, smooth_outside, finish)
from landscape_2026_11_20d import tops_like_before
from PIL import Image

part = sys.argv[sys.argv.index('--part') + 1]


def plant_spruce(f, cells, Hn, top, slope, gap=(7, 11), p=0.65):
    SPR = spruces(f)
    for tt in SPR:
        for *_, k in tt: f.ID(k)
    sp = scatter(np.argwhere(cells & np.isin(top, ['grass', 'rock', 'snow', 'snowgrass']) & (slope < 1.5)), lambda: rng.uniform(*gap), p)
    for i, j in sp: place(f, i, j, Hn[i, j] - YMIN + 1, SPR[int(rng.integers(8))], int(rng.integers(4)))
    return len(sp)


# ---------------------------------------------------------------- кусок джунглей
def kusok():
    f = L.Frame(world, 352, -80, 464, 112).surface()
    K = polygon_mask(f, 'кусок_джунглей')
    banned = bans(f)
    water = f.WAT | (f.GY < SEA)
    zone = ndimage.binary_dilation(K, structure=disk(8)) & ~water & ~banned
    anchor = ~zone & ~water
    base = smooth_outside(f.GY.astype(float), anchor)
    H = np.maximum(f.GY, base + (4 + 8*f.noise(10, rng))*sstep(0, 10, ndimage.distance_transform_edt(zone)))
    H = np.where(zone, ndimage.gaussian_filter(H, 2.0), f.GY)
    H = np.minimum(H, np.maximum(balloon_cap(f), f.GY))
    Hn = np.rint(H).astype(int)
    cells = zone & (Hn != f.GY)
    print(f'кусок джунглей: {int(K.sum())} столбцов, меняется {int(cells.sum())}')
    kill_trees(f, zone, f.G)                                       # джунгли сняты целиком (и где высота не менялась)
    top, slope = tops_like_before(f, Hn, zone)
    top[zone & (slope < 1.3)] = 'grass'
    build(f, zone, Hn, top)
    n = plant_spruce(f, zone, Hn, top, slope, (6, 10), 0.75)
    plants(f, zone & (top == 'grass'), Hn, 0.2, 0.01, ('fern',))
    print(f'  елей {n}')
    finish(f, 'кусок джунглей → подножие с елями')


# ---------------------------------------------------------------- плато и хребет
GORGE = np.array([[126, -336], [150, -341], [172, -334], [200, -329]], float)


def ridge():
    f = L.Frame(world, 32, -528, 464, -192).surface()
    banned = bans(f)
    banned &= ~((f.XX >= B.CHASM[0]) & (f.XX < 352) & (f.ZZ >= B.CHASM[2]) & (f.ZZ <= B.CHASM[3]) & (np.hypot(f.XX - B.VENT[0], f.ZZ - B.VENT[1]) >= B.VENT_R))
    seaw = sea(f)
    nm = f.names(); snowyg = np.array(['grass_block[snowy=true]' in k for k in nm])
    topb = np.zeros((f.sx, f.sz), bool)
    for i in range(f.sx): topb[i] = snowyg[f.vol[i, f.G[i], np.arange(f.sz)]]
    box = (f.XX >= 160) & (f.XX <= 340) & (f.ZZ >= -372) & (f.ZZ <= -240)
    P0 = box & ((topb & (f.GY >= 116) & (f.GY <= 136)) | (f.WAT & (f.GY >= 115) & (f.GY <= 130)))
    P0 = ndimage.binary_fill_holes(ndimage.binary_closing(P0, structure=disk(5)))
    lab, n = ndimage.label(P0); sz = np.bincount(lab.ravel()); sz[0] = 0; P0 = ndimage.binary_fill_holes(lab == sz.argmax())
    # скругление: размытие большим ядром и порог — углы уходят, край плавный
    P = ndimage.gaussian_filter(P0.astype(float), 26) > 0.5 + 0.22*(f.noise(14, rng) - 0.5)*2
    P = ndimage.binary_fill_holes(P)
    pond = f.WAT & P0
    print(f'плато: было {int(P0.sum())}, после скругления {int(P.sum())}')
    Hp = np.where(P0, f.GY, 0).astype(float)
    Hp = smooth_outside(Hp, P0)                                       # высота плато, продолженная на новые края
    dP = ndimage.distance_transform_edt(~P)
    crest = 146 + 5*(f.noise(22, rng) - 0.5)*2                       # крупный шум: гребень без угловатости
    edgeH = smooth_outside(np.where(P, Hp, 0.0), P)
    up = edgeH + (crest - edgeH)*sstep(0, 22, dP)
    far = dP > 85
    E = smooth_outside(f.GY.astype(float), far | seaw)               # то, что за хребтом (и море)
    E = np.where(seaw, SEA - 2, E)
    down = crest + (E - crest)*sstep(24, 85, dP)
    H = np.where(P, Hp, np.where(dP <= 24, np.maximum(up, crest*sstep(22, 24, dP) + up*(1 - sstep(22, 24, dP))), down))
    zone = (dP <= 85) & ~seaw
    # ущелье остаётся: в 34 блоках от пути — не выше прежнего
    gx0, gz0 = f.XX + 0.5, f.ZZ + 0.5
    dg = np.full(f.XX.shape, 1e9)
    for a, b in zip(GORGE[:-1], GORGE[1:]):
        dx, dz = b - a; L2 = dx*dx + dz*dz
        t = np.clip(((gx0 - a[0])*dx + (gz0 - a[1])*dz)/L2, 0, 1)
        dg = np.minimum(dg, np.hypot(gx0 - a[0] - t*dx, gz0 - a[1] - t*dz))
    H = np.where(dg <= 34, np.minimum(H, np.maximum(f.GY, f.GY + (H - f.GY)*sstep(10, 34, dg))), H)
    db = ndimage.distance_transform_edt(~banned)
    H = f.GY + (H - f.GY)*sstep(0, 20, db)
    H = np.where(zone, ndimage.gaussian_filter(H, 1.5), f.GY)
    Hn = np.rint(H).astype(int)
    cells = zone & ~banned & ~pond & (np.abs(Hn - f.GY) >= 1)
    print(f'плато и хребет: меняется {int(cells.sum())}, гребень {np.percentile(Hn[(dP > 18) & (dP < 28) & zone], [10, 50, 90])}')
    kill_trees(f, cells, np.minimum(f.G, Hn - YMIN))
    top, slope = tops_like_before(f, Hn, cells)
    top[P & (slope < 1.8)] = 'snowgrass'
    build(f, cells, Hn, top)
    jun = polygon_mask(f, 'джунгли')
    n = plant_spruce(f, cells & ~P & ~jun & (Hn <= 160), Hn, top, slope, (11, 16), 0.5) + plant_spruce(f, cells & P, Hn, top, slope, (18, 26), 0.3)
    print(f'  елей {n}')
    finish(f, 'плато скруглено, хребет: плавно снаружи')


# ---------------------------------------------------------------- ярусы
def terraces3():
    f = L.Frame(world, 176, -272, 416, 0).surface()
    banned = bans(f)
    red = polygon_mask(f, 'редактирование')
    dred = ndimage.distance_transform_edt(~red)
    water = f.WAT | (f.GY < SEA)
    foot = ~water & (f.GY < 95) & np.isin(f.SURF, ('grass_block', 'dirt', 'podzol', 'coarse_dirt', 'sand'))
    lab, n = ndimage.label(foot); sz = np.bincount(lab.ravel()); sz[0] = 0
    foot = np.isin(lab, np.nonzero(sz > 2000)[0])
    footH = smooth_outside(np.where(foot, f.GY, 0).astype(float), foot)
    Gs = ndimage.gaussian_filter(f.GY.astype(float), 5.0)
    rel = Gs - footH
    RISE, SLOPE = 12, 2.5
    H = footH.copy()
    for k in range(3):
        R = rel >= RISE*k + RISE*0.5 + 2                              # выше середины этого подъёма — ярус k уже взят
        dR = ndimage.distance_transform_edt(R)
        H = H + RISE*np.clip(dR*SLOPE/RISE, 0, 1)
    above = np.maximum(0, rel - 3*RISE - 4)
    H = H + 0.6*above
    w = sstep(5, 25, dred)*sstep(0, 15, ndimage.distance_transform_edt(~banned))*(1 - sstep(55, 95, rel))
    zone = ~water & ~foot & (f.GY >= 72) & (w > 0) & ~((f.XX >= 350) & (f.ZZ >= -100))
    w = w*(1 - sstep(385, 405, f.XX))*sstep(-250, -222, f.ZZ)   # (добавлено в правке 6: прежде задевало юг плато)
    H = np.where(zone, f.GY + (H - f.GY)*w, f.GY)
    Hn = np.rint(H).astype(int)
    cells = zone & (np.abs(Hn - f.GY) >= 1)
    print(f'ярусы (3): склон {int(zone.sum())}, меняется {int(cells.sum())}, сдвиг до {int(np.abs(Hn - f.GY)[cells].max()) if cells.any() else 0}')
    kill_trees(f, cells, np.minimum(f.G, Hn - YMIN))
    top, slope = tops_like_before(f, Hn, cells)
    top[cells & (slope < 0.8)] = 'grass'                              # полки трава
    top[cells & (slope >= 1.6)] = 'rock'                              # подъёмы камень
    build(f, cells, Hn, top)
    n = plant_spruce(f, cells & (slope < 0.6), Hn, top, slope, (13, 19), 0.35)
    plants(f, cells & (top == 'grass'), Hn, 0.2, 0.03)
    print(f'  елей на полках {n}')
    finish(f, 'ярусы (3)')


# ---------------------------------------------------------------- шапки
def caps():
    total = 0
    for tx in range(-448, 704, 192):
        for tz in range(-576, 256, 192):
            f = L.Frame(world, tx - 16, tz - 16, tx + 192 + 16, tz + 192 + 16).surface()
            rockish = np.isin(f.SURF, D.ROCKISH)
            keep = polygon_mask(f, 'редактирование') | polygon_mask(f, 'руины') | (np.hypot(f.XX - B.VENT[0], f.ZZ - B.VENT[1]) < B.VENT_R)
            x0, x1, z0, z1 = B.CHASM
            keep |= (f.XX >= x0) & (f.XX <= x1) & (f.ZZ >= z0) & (f.ZZ <= z1)
            keep |= ndimage.binary_dilation(f.WAT | (f.GY < SEA), structure=disk(3))
            mount = (rockish | (f.GY >= 130)) & (f.GY >= 125) & ~keep
            if not mount.any(): continue
            Gs = ndimage.gaussian_filter(f.GY.astype(float), 2.0)
            H = np.where(mount & (f.GY >= 135), Gs, f.GY)
            Hn = np.rint(H).astype(int)
            inner = np.zeros_like(mount); inner[16:-16, 16:-16] = True
            gx, gz = np.gradient(Hn.astype(float)); slope = np.hypot(gx, gz)
            line = 138 + 8*(0.55*np.sin(f.XX/37 + 1.3)*np.cos(f.ZZ/29 + 0.7) + 0.45*np.sin((f.XX + f.ZZ)/53 + 2.1))   # от мировых координат: без швов на стыках плиток
            snowy = (Hn >= line) & ((slope < 2.2) | (Hn >= 170))
            nm = f.names(); snowblk = np.array([k.startswith('minecraft:snow_block') for k in nm])
            was_snow = np.zeros_like(mount)
            for i in range(f.sx): was_snow[i] = snowblk[f.vol[i, f.G[i], np.arange(f.sz)]]
            change_h = mount & inner & (Hn != f.GY)
            change_s = mount & inner & (snowy != was_snow) & np.isin(f.SURF, ('stone', 'andesite', 'tuff', 'snow_block', 'gravel', 'calcite'))
            cells = change_h | change_s
            if not cells.any(): continue
            kill_trees(f, change_h & (Hn < f.GY), Hn - YMIN)
            top = np.where(snowy, 'snow', 'rock').astype(object)
            top[snowy & (Hn >= 152)] = 'snow2'
            # снятый снег: верх — порода пласта (а не прежний снег)
            ROCKID = f.ID('minecraft:stone')
            for i, j in zip(*np.nonzero(change_s & ~snowy & was_snow)):
                f.vol[i, f.G[i, j], j] = ROCKID
                if f.vol[i, f.G[i, j] - 1, j] == f.ID('minecraft:snow_block'): f.vol[i, f.G[i, j] - 1, j] = ROCKID
            build(f, cells, Hn, top)
            n = int(cells.sum()); total += n
            print(f'плитка x {tx} z {tz}: шапки — столбцов {n} (высота {int(change_h.sum())}, снег {int(change_s.sum())})')
            if WRITE:
                if not os.path.isdir(B.BACKUP_HINT()): sys.exit('нет копии')
                f.write()
    print('шапки: столбцов всего', total)
    if not WRITE: print('без --write мир не изменён')


# ---------------------------------------------------------------- колья
def spikes():
    orig = sys.argv[sys.argv.index('--orig') + 1]
    import json
    PL = json.load(open(B.PLACES))
    homes = np.array(list(PL.get('points', {}).values()), float) if PL.get('points') else np.zeros((0, 3))
    total = 0
    for tx in range(-704, 704, 192):
        for tz in range(-576, 512, 192):
            f = L.Frame(world, tx - 16, tz - 16, tx + 192 + 16, tz + 192 + 16).surface()
            o = L.Frame(orig, tx - 16, tz - 16, tx + 192 + 16, tz + 192 + 16).surface()
            changed = f.GY != o.GY
            if not changed.any(): continue
            g = f.GY.astype(float)
            op = ndimage.grey_opening(g, size=(3, 3))
            sp = (g - op >= 3) & changed
            keep = polygon_mask(f, 'руины') | polygon_mask(f, 'редактирование') | (f.WAT | (f.GY < SEA))
            for hx, hy, hz in homes: keep |= np.hypot(f.XX - hx, f.ZZ - hz) < 20
            sp &= ~keep
            sp[:16] = sp[-16:] = False; sp[:, :16] = sp[:, -16:] = False
            if not sp.any(): continue
            Hn = np.rint(op).astype(int)
            kill_trees(f, sp, Hn - YMIN)
            top, slope = tops_like_before(f, Hn, sp)
            build(f, sp, Hn, top)
            total += int(sp.sum())
            print(f'плитка x {tx} z {tz}: кольев срезано {int(sp.sum())}, наибольший {int((g - op)[sp].max())}')
            if WRITE:
                if not os.path.isdir(B.BACKUP_HINT()): sys.exit('нет копии')
                f.write()
    print('кольев всего', total)
    if not WRITE: print('без --write мир не изменён')


import json as _json
HOMES = list(_json.load(open(B.PLACES)).get('points', {}).values())


def nowalls():
    """Без стенок там, где правки 5 что-то меняли (сравнение с копией --bk <мир-копия>, полоса 12 вокруг): сначала срез выступов до соседа + 3,
    потом досыпка снизу до соседа − 3 у того, что трогать нельзя («редактирование», засада, руины, жерло, расщелина, вода и 2 блока у неё).
    Плитки 192, поле 32 — только контекст."""
    bk = sys.argv[sys.argv.index('--bk') + 1]
    total = 0
    for rnd in range(2):
        for tx in range(-704, 704, 192):
            for tz in range(-576, 512, 192):
                f = L.Frame(world, tx - 32, tz - 32, tx + 224, tz + 224).surface()
                o = L.Frame(bk, tx - 32, tz - 32, tx + 224, tz + 224).surface()
                chg = f.GY != o.GY
                del o
                if not chg.any(): continue
                icy = np.isin(f.SURF, ('ice', 'packed_ice', 'blue_ice'))      # лёд над водой (родник над «редактированием») — тоже вода
                water = ndimage.binary_dilation(f.WAT | icy | (f.GY < SEA), structure=disk(2))
                prot = polygon_mask(f, 'редактирование') | polygon_mask(f, 'засада_туземцев') | polygon_mask(f, 'руины')
                prot |= np.hypot(f.XX - B.VENT[0], f.ZZ - B.VENT[1]) < B.VENT_R
                for hx, hy, hz in HOMES: prot |= np.hypot(f.XX - hx, f.ZZ - hz) < 30   # дома (правка 6 срезала дом Совуньи)
                x0, x1, z0, z1 = B.CHASM
                prot |= (f.XX >= 352) & (f.XX <= x1) & (f.ZZ >= z0) & (f.ZZ <= z1)   # западнее 352 хребет этой правки
                inner = np.zeros_like(chg); inner[32:-32, 32:-32] = True
                el = ndimage.binary_dilation(chg, structure=disk(12)) & inner & ~prot & ~water
                if not el.any(): continue
                H = f.GY.astype(int).copy()
                NB = ((1, 0), (-1, 0), (0, 1), (0, -1))
                for _ in range(300):
                    m = H.copy()
                    for di, dj in NB: m = np.minimum(m, np.roll(np.roll(H, di, 0), dj, 1) + 3)
                    new = np.where(el, m, H)
                    if np.array_equal(new, H): break
                    H = new
                for _ in range(300):
                    m = H.copy()
                    for di, dj in NB: m = np.maximum(m, np.roll(np.roll(H, di, 0), dj, 1) - 3)
                    new = np.where(el, m, H)
                    if np.array_equal(new, H): break
                    H = new
                H = np.where(H > f.GY, np.minimum(H, np.maximum(balloon_cap(f), f.GY)), H).astype(int)
                cells = el & (H != f.GY)
                if not cells.any(): continue
                kill_trees(f, cells & (H < f.GY), H - YMIN)
                top, slope = tops_like_before(f, H, cells)
                build(f, cells, H, top)
                n = int(cells.sum()); total += n
                print(f'круг {rnd} плитка x {tx} z {tz}: срез {int((cells & (H < f.GY)).sum())} (до {int((f.GY - H)[cells].max())}), '
                      f'досыпка {int((cells & (H > f.GY)).sum())} (до {int((H - f.GY)[cells].max())})')
                if WRITE: f.write()
        if not WRITE: break
    print('без стенок: столбцов', total)


def raise_ring():
    """Правка 6: горы вокруг плато выше на ~35 (30–40, крупный шум): прибавка растёт от края плато за 32 блока и сходит на нет снаружи
    к 110; у моря, джунглей, ущелья, кратера, расщелины и реки — на нет; плато не меняется."""
    f = L.Frame(world, 32, -560, 480, -160).surface()
    banned = bans(f)
    banned &= ~((f.XX >= B.CHASM[0]) & (f.XX < 352) & (f.ZZ >= B.CHASM[2]) & (f.ZZ <= B.CHASM[3]) & (np.hypot(f.XX - B.VENT[0], f.ZZ - B.VENT[1]) >= B.VENT_R))
    seaw = sea(f)
    nm = f.names(); snowyg = np.array(['grass_block[snowy=true]' in k for k in nm])
    topb = np.zeros((f.sx, f.sz), bool)
    for i in range(f.sx): topb[i] = snowyg[f.vol[i, f.G[i], np.arange(f.sz)]]
    box = (f.XX >= 160) & (f.XX <= 340) & (f.ZZ >= -372) & (f.ZZ <= -240)
    P = box & ((topb & (f.GY >= 116) & (f.GY <= 136)) | (f.WAT & (f.GY >= 115) & (f.GY <= 130)))
    P = ndimage.binary_fill_holes(ndimage.binary_closing(P, structure=disk(5)))
    lab, n = ndimage.label(P); sz = np.bincount(lab.ravel()); sz[0] = 0; Pd = ndimage.binary_fill_holes(lab == sz.argmax())
    # юг плато срезали ярусы (terraces3, рамка с z −272): прямой край — достроить скруглённую кромку на 14 блоков
    zcut = f.ZZ[Pd].max()
    P = Pd | (ndimage.binary_dilation(Pd, structure=disk(14)) & (f.ZZ >= zcut))
    P = ndimage.gaussian_filter(P.astype(float), 3) > 0.5
    cap = P & ~Pd
    Hp = smooth_outside(np.where(Pd, f.GY, 0).astype(float), Pd)
    print(f'плато: найдено {int(Pd.sum())}, южная кромка до z {zcut}, достроено {int(cap.sum())}')
    dP = ndimage.distance_transform_edt(~P)
    A = 35 + 5*(f.noise(30, rng) - 0.5)*2
    bump = A*sstep(0, 32, dP)*(1 - sstep(36, 110, dP))
    gx0, gz0 = f.XX + 0.5, f.ZZ + 0.5
    dg = np.full(f.XX.shape, 1e9)
    for a, b in zip(GORGE[:-1], GORGE[1:]):
        dx, dz = b - a; L2 = dx*dx + dz*dz
        t = np.clip(((gx0 - a[0])*dx + (gz0 - a[1])*dz)/L2, 0, 1)
        dg = np.minimum(dg, np.hypot(gx0 - a[0] - t*dx, gz0 - a[1] - t*dz))
    jun = polygon_mask(f, 'джунгли')
    bump = bump*sstep(12, 36, dg)*sstep(0, 25, ndimage.distance_transform_edt(~banned))*sstep(0, 30, ndimage.distance_transform_edt(~seaw))
    bump = bump*sstep(0, 25, ndimage.distance_transform_edt(~jun))
    bump = ndimage.gaussian_filter(bump, 2.0)
    H = f.GY + bump
    H = np.minimum(H, np.maximum(balloon_cap(f), f.GY))
    H = np.where(cap, Hp, H)
    Hn = np.rint(H).astype(int)
    cells = ((~P & (Hn > f.GY)) | (cap & (Hn != f.GY))) & ~banned & ~seaw & ~jun & ~f.WAT
    cells[:2] = cells[-2:] = False; cells[:, :2] = cells[:, -2:] = False
    ring = (dP > 20) & (dP < 40) & cells
    print(f'кольцо гор: столбцов {int(cells.sum())}, прибавка до {int((Hn - f.GY)[cells].max())}, гребень {np.percentile(Hn[ring], [10, 50, 90])}')
    kill_trees(f, cells, f.G)
    top, slope = tops_like_before(f, Hn, cells, snow_y=116)
    top[cap] = 'snowgrass'
    build(f, cells, Hn, top)
    n = plant_spruce(f, cells & (Hn <= 175), Hn, top, slope, (11, 16), 0.5)
    print(f'  елей {n}')
    finish(f, 'кольцо гор выше')


SNOW_Y = 116


def snow116():
    """Правка 6: снег от y 116 (±2, гладко от мировых координат) на всех горах: порода/гравий → снежный блок, земля/трава → снежная трава
    со слоем снега (если над ней пусто или трава/цветы; под деревьями только снежная трава). Не в джунглях, не у воды, не в кратере."""
    PLANTS = ('short_grass', 'tall_grass', 'fern', 'large_fern', 'dandelion', 'poppy', 'azure_bluet', 'oxeye_daisy', 'cornflower', 'bush',
              'short_dry_grass', 'tall_dry_grass', 'dead_bush')
    ROCKS = ('stone', 'andesite', 'tuff', 'calcite', 'granite', 'diorite', 'gravel', 'cobblestone', 'deepslate', 'coarse_dirt', 'sandstone')
    GRASSY = ('grass_block', 'dirt', 'podzol', 'moss_block', 'rooted_dirt')
    total = 0
    for tx in range(-704, 704, 192):
        for tz in range(-576, 512, 192):
            f = L.Frame(world, tx, tz, tx + 192, tz + 192).surface()
            line = SNOW_Y + 2*np.sin(f.XX/23 + 0.4)*np.cos(f.ZZ/19 + 1.1)
            keep = polygon_mask(f, 'джунгли') | (np.hypot(f.XX - B.VENT[0], f.ZZ - B.VENT[1]) < B.VENT_R)
            keep |= ndimage.binary_dilation(f.WAT, structure=disk(1))
            cand = (f.GY >= line) & ~keep
            if not cand.any(): continue
            bn = np.array([L.S.bname(k) for k in f.names()])
            SN, SG = f.ID('minecraft:snow_block'), f.ID('minecraft:grass_block[snowy=true]')
            AIR = set(np.nonzero(bn == 'air')[0].tolist())
            PL = set(np.nonzero(np.isin(bn, PLANTS))[0].tolist())
            UPPER = set(i for i, k in enumerate(f.names()) if 'half=upper' in k)
            n = 0
            for i, j in zip(*np.nonzero(cand)):
                g = f.G[i, j]; col = f.vol[i, :, j]; b = bn[col[g]]
                if b in ROCKS: col[g] = SN; n += 1
                elif b in GRASSY:
                    col[g] = SG
                    up = int(col[g + 1])
                    if up in AIR or up in PL:
                        if int(col[g + 2]) in UPPER: col[g + 2] = 0
                        col[g + 1] = f.ID(f'minecraft:snow[layers={1 + int(rng.random() < 0.35)}]')
                    n += 1
            if n:
                total += n
                print(f'плитка x {tx} z {tz}: снег {n}')
                if WRITE: f.write()
    print('снег от 116: столбцов', total)
    if not WRITE: print('без --write мир не изменён')


if __name__ == '__main__':
    {'backup': B.backup, 'kusok': kusok, 'ridge': ridge, 'terraces3': terraces3, 'caps': caps, 'spikes': spikes, 'nowalls': nowalls, 'raise': raise_ring, 'snow116': snow116}[part]()
