#!/usr/bin/env python3
"""Третья правка ландшафта «Ромашковой Долины» (2026-11-20, по замечаниям к второй, landscape_2026_11_20b.py). Пользуется его функциями.

Запуск: landscape_2026_11_20c.py <мир> --part <часть> [--preview файл.png] [--write], игра закрыта. Копия перед записью — та же, что у второй правки
(`<мир>/.landscape_backup_b`), либо backup своей части. Части: backup <папка> → unspruce → north2 → se2 → strait.
  unspruce  ели прочь с маленького каменного хребта на северо-западе (у ущелья реки под пустыней) и с руин (локация «руины» + 20 блоков);
  north2    северные горы целиком: локация «ущелье» (её нарисовал пользователь) поднята до уровня плато (y ~137) и стала его частью, под снегом;
            плато со всех сторон окружено хребтами (~170, профиль: от края плато вверх за 20 блоков, потом вниз за 30); полосы вокруг гор —
            тело гор со склонами до воды на севере и северо-востоке, к джунглям (восток, юг) плавный переход, на западе хребты сходят к холмам;
  se2       юго-восточная гора растёт склонами до моря на юге и востоке, к «засаде_туземцев» плавный переход (подъём с 45 до 75 блоков), вершина — гребни;
  strait    пролив между юго-восточной горой и хребтом западнее засыпан земляной долиной y ~72–85, луга у (375, 120) подведены к ней холмами.
Везде: жерло вулкана (26), расщелина и вход в тоннель, река с брёвнами — не трогаются; у маршрута шара рельеф ниже линии полёта на 15.
"""
import os, sys
import numpy as np
from scipy import ndimage
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import landscape_2026_11_20b as B
from landscape_2026_11_20b import (L, world, preview, WRITE, rng, disk, sstep, NY, YMIN, SEA, mask, bans, sea, polygon_mask, balloon_cap,
                                  kill_trees, build, spruces, place, scatter, plants, smooth_outside, jungle_tree, broadleaf, bush, finish)

part = sys.argv[sys.argv.index('--part') + 1]


def ridged(f, sig):
    n = f.noise(sig, rng); return 1 - np.abs(2*n - 1)


def plant_mountain(f, cells, Hn, top, slope, inner_foot=None, density=(7, 12)):
    SPR = spruces(f)
    for t in SPR:
        for *_, k in t: f.ID(k)
    ok = cells & np.isin(top, ['rock', 'snow', 'snow2', 'grass']) & (slope < 1.6) & (Hn >= 90) & (Hn <= 165)
    if inner_foot is not None: ok &= ~inner_foot
    sp = scatter(np.argwhere(ok), lambda: rng.uniform(*density), 0.7)
    for i, j in sp: place(f, i, j, Hn[i, j] - YMIN + 1, SPR[int(rng.integers(8))], int(rng.integers(4)))
    jt = []
    if inner_foot is not None:
        jt = scatter(np.argwhere(cells & inner_foot & (slope < 1.2)), lambda: rng.uniform(5, 8), 0.8)
        for i, j in jt: place(f, i, j, Hn[i, j] - YMIN + 1, jungle_tree())
        for i, j in scatter(np.argwhere(cells & inner_foot & (slope < 1.0)), lambda: rng.uniform(3, 6), 0.6): place(f, i, j, Hn[i, j] - YMIN + 1, bush('jungle'))
    print(f'  елей {len(sp)}, джунглевых деревьев у подножия {len(jt)}')


def materials(f, Hn, cells, snow_y=140, grass_below=None, low=None):
    gx, gz = np.gradient(Hn.astype(float)); slope = np.hypot(gx, gz)
    sn = L.S.vnoise(f.XX/11.0, f.ZZ/11.0)
    top = np.full(Hn.shape, 'rock', object)
    if grass_below is not None: top[(Hn < grass_below) & (slope < 1.3)] = 'grass'
    snowy = (Hn >= snow_y + 4*sn) & ((slope < 2.3) | (Hn >= 168))
    top[snowy] = 'snow'; top[snowy & (Hn >= 152)] = 'snow2'
    if low is not None: top[low & (slope < 1.8)] = 'snowgrass'
    seaw = sea(f)
    dw = ndimage.distance_transform_edt(~seaw)
    top[cells & (dw <= 2.5) & (Hn <= SEA + 4)] = 'gravel'
    return top, slope


# ---------------------------------------------------------------- ели прочь
def unspruce():
    for x0, z0, x1, z1, what in ((-320, -224, -128, -128, 'хребет'), (-16, -448, 128, -320, 'руины')):
        f = L.Frame(world, x0, z0, x1, z1).surface()
        nm = f.names()
        slog = np.array([k.startswith('minecraft:spruce_log') for k in nm])
        base = np.zeros((f.sx, f.sz), bool)
        for i in range(f.sx): base[i] = slog[f.vol[i, np.minimum(f.G[i] + 1, NY - 1), np.arange(f.sz)]]
        if what == 'хребет':
            region = np.isin(f.SURF, ('stone', 'andesite', 'tuff', 'calcite', 'snow_block', 'gravel', 'cobblestone')) & (f.GY >= 80)
        else:
            region = ndimage.binary_dilation(polygon_mask(f, 'руины'), structure=disk(20))
        doomed = base & region
        print(f'{what}: елей {int(doomed.sum())}')
        if doomed.any():
            kill_trees(f, doomed, f.G)
        finish(f, f'ели прочь: {what}')


# ---------------------------------------------------------------- северные горы
def north2():
    f = L.Frame(world, 32, -528, 464, -208).surface()
    banned = bans(f)
    M1, H3 = mask(f, 1), mask(f, 3)
    Z = polygon_mask(f, 'ущелье')
    seaw = sea(f)
    nonsea = ~seaw
    inner = ndimage.binary_dilation(polygon_mask(f, 'джунгли') & ~M1 & ~Z, structure=disk(6))
    # плато: снежный луг прошлой правки (трава на 128–144 в маске) и «ущелье»
    nm = f.names(); snowy = np.array(['grass_block[snowy=true]' in k for k in nm])
    topb = np.zeros((f.sx, f.sz), bool)
    for i in range(f.sx): topb[i] = snowy[f.vol[i, f.G[i], np.arange(f.sz)]]
    meadow = M1 & topb & (f.GY >= 128) & (f.GY <= 144)
    P = ndimage.binary_fill_holes(ndimage.binary_closing(meadow | Z, structure=disk(6)))
    lab, n = ndimage.label(P); sz = np.bincount(lab.ravel()); sz[0] = 0; P = lab == sz.argmax()
    # скруглить: углы нарисованной локации не должны читаться, край чуть неровный
    P = ndimage.gaussian_filter(P.astype(float), 9) + 0.42*(f.noise(8, rng) - 0.5) + 0.2*(f.noise(20, rng) - 0.5) > 0.5
    lab, n = ndimage.label(P); sz = np.bincount(lab.ravel()); sz[0] = 0; P = ndimage.binary_fill_holes(lab == sz.argmax())
    P &= ~banned
    pondw = f.WAT & P & (f.GY >= 128)                       # озерцо на плато (подо льдом) не трогаем
    dP = ndimage.distance_transform_edt(~P)
    print(f'плато: {int(P.sum())} столбцов, x {f.XX[P].min()}..{f.XX[P].max()}, z {f.ZZ[P].min()}..{f.ZZ[P].max()}')
    ring = nonsea & ndimage.binary_dilation(M1 | Z, structure=disk(45)) & ~inner & ~H3
    body = (M1 | Z | ring | (H3 & (dP <= 50))) & nonsea & ~banned & ~pondw
    # общее поле гор: опоры снаружи — море (ниже уровня), джунгли и холмы (как есть)
    # опоры снаружи — только джунгли и холмы: к морю тело гор не сходит на нет, а режется склоном от воды (ниже)
    anchor = ~body & ~seaw
    hN = smooth_outside(f.GY.astype(float), anchor)
    e = ndimage.distance_transform_edt(~anchor)
    s = sstep(0, 45, e)
    Pk = 125 + 61*(0.55*ridged(f, 16) + 0.3*f.noise(40, rng) + 0.15*ridged(f, 6))
    Hm = hN*(1 - s) + Pk*s
    dws = ndimage.distance_transform_edt(~seaw)
    coast = SEA + 1 + 1.6*dws + 3*(f.noise(5, rng) - 0.5)
    # хребты вокруг плато: от края вверх за 20 блоков до ~170–178, потом вниз к окружению за 30
    crest = 166 + 14*f.noise(12, rng)
    wid = 16 + 8*f.noise(15, rng)                               # ширина подъёма к гребню разная по кругу
    up = 137 + (crest - 137)*sstep(2, wid, dP)
    down = crest - (crest - hN)*sstep(wid + 2, wid + 32, dP)
    ridge = np.where(dP <= wid + 1, up, down)
    H = np.where(dP <= wid + 34, np.maximum(Hm, ridge), Hm)
    H = np.where(body, np.minimum(H, coast), H)
    H = np.where(P, 137 + 2*(f.noise(10, rng) - 0.5), H)
    H = np.where(body, ndimage.gaussian_filter(H, 1.5), H)
    H = np.where(P, 137 + 2*(f.noise(10, rng) - 0.5), H)
    near_sea = body & (dws <= 40)                                  # у моря можно и понижать (склон вместо уступа)
    Hn = np.rint(np.where(P | near_sea, H, np.maximum(H, f.GY))).astype(int)
    Hn = np.where(body & (np.abs(Hn - f.GY) >= 2), Hn, f.GY)
    cells = body & (Hn != f.GY)
    print(f'северные горы: тело {int(body.sum())}, меняется {int(cells.sum())}, верх {Hn[cells].max()}, плато {int((cells & P).sum())}')
    kill_trees(f, cells, np.minimum(f.G, Hn - YMIN))
    foot_in = body & inner.__or__(ndimage.binary_dilation(inner, structure=disk(20))) & (Hn < 135)
    top, slope = materials(f, Hn, cells, 140, grass_below=None, low=P)
    top[foot_in & (slope < 1.3)] = 'grass'
    top[body & H3 & (Hn < 110) & (slope < 1.4)] = 'grass'         # холмы снизу у хребтов — трава
    build(f, cells, Hn, top)
    plant_mountain(f, cells & ~P, Hn, top, slope, inner_foot=foot_in & (top == 'grass'))
    sp = scatter(np.argwhere(cells & P & (top == 'snowgrass')), lambda: rng.uniform(14, 20), 0.5)
    SPR = spruces(f)
    for i, j in sp: place(f, i, j, Hn[i, j] - YMIN + 1, SPR[int(rng.integers(8))], int(rng.integers(4)))
    plants(f, cells & H3 & (top == 'grass'), Hn, 0.2, 0.03)
    print(f'  елей на плато {len(sp)}')
    finish(f, 'северные горы (2)')


# ---------------------------------------------------------------- юго-восток
def se2():
    f = L.Frame(world, 368, -176, 656, 128).surface()
    banned = bans(f)
    seaw = sea(f)
    zas = polygon_mask(f, 'засада_туземцев')
    dz = ndimage.distance_transform_edt(~zas) + 14*(f.noise(9, rng) - 0.5)
    # только южнее засады (z ≥ −95): севернее — джунгли у реки (в первой записи без этого гора залезла туда, прямоугольник z < −96 возвращён из копии)
    body = ~seaw & (dz >= 45) & (f.XX >= 398) & (f.ZZ >= -95) & ~banned
    lab, n = ndimage.label(body); sz = np.bincount(lab.ravel()); sz[0] = 0
    body = np.isin(lab, np.nonzero(sz > 400)[0])
    hv = np.where(seaw, SEA - 3, f.GY).astype(float)
    hN = smooth_outside(hv, ~body)
    e = ndimage.distance_transform_edt(body)
    s = sstep(0, 35, e)*sstep(45, 75, dz)
    Pk = 128 + 34*(0.6*ridged(f, 12) + 0.25*ridged(f, 5) + 0.15*f.noise(30, rng))
    w = sstep(45, 75, dz)
    H = np.maximum(f.GY, f.GY*(1 - w) + Pk*w)                 # к засаде джунгли как есть, дальше гребни
    # к морю склоном, а не обрывом: не выше SEA+1 + 1,6 на блок от воды (здесь можно и понижать)
    dws = ndimage.distance_transform_edt(~seaw)
    coast = SEA + 1 + 1.6*dws + 3*(f.noise(5, rng) - 0.5)
    H = np.where(body, np.minimum(H, coast), H)
    H = np.where(body, ndimage.gaussian_filter(H, 1.5), H)
    H = np.minimum(H, np.maximum(balloon_cap(f), np.minimum(f.GY, coast)))
    Hn = np.rint(H).astype(int)
    Hn = np.where(body & (np.abs(Hn - f.GY) >= 2), Hn, f.GY)
    cells = body & (Hn != f.GY)
    print(f'юго-восток: тело {int(body.sum())}, меняется {int(cells.sum())}, верх {Hn[cells].max() if cells.any() else "-"}')
    kill_trees(f, cells, np.minimum(f.G, Hn - YMIN))
    inner = ndimage.binary_dilation(polygon_mask(f, 'джунгли'), structure=disk(6))
    foot = (w < 0.3) & inner
    top, slope = materials(f, Hn, cells, 150)
    top[cells & foot & (slope < 1.3)] = 'grass'
    build(f, cells, Hn, top)
    plant_mountain(f, cells, Hn, top, slope, inner_foot=cells & foot & (top == 'grass'))
    finish(f, 'юго-восток (2)')


# ---------------------------------------------------------------- пролив → долина, луга → холмы
def strait():
    f = L.Frame(world, 288, -112, 432, 176).surface()
    banned = bans(f)
    water = f.WAT | (f.GY < SEA)
    box = (f.XX >= 352) & (f.XX <= 404) & (f.ZZ >= -85) & (f.ZZ <= 95)
    lab, n = ndimage.label(water & box)
    k = lab[380 - f.X0, 0 - f.Z0]
    V = lab == k if k else np.zeros_like(water)
    print(f'пролив: {int(V.sum())} столбцов воды, x {f.XX[V].min()}..{f.XX[V].max()}, z {f.ZZ[V].min()}..{f.ZZ[V].max()}')
    free = (V | (ndimage.binary_dilation(V, structure=disk(10)) & ~water)) & ~banned
    hv = np.where(water, SEA - 3, f.GY).astype(float)
    hN = smooth_outside(hv, ~free)
    e = ndimage.distance_transform_edt(free)
    floor = 73 + 9*f.noise(14, rng)
    H = hN*(1 - sstep(0, 14, e)) + floor*sstep(0, 14, e)
    # луга к югу (у 375, 120): холмы, подведённые к долине
    box = ((f.XX >= 328) & (f.XX <= 410) & (f.ZZ >= 88) & (f.ZZ <= 150)).astype(float)
    wm = np.clip(ndimage.gaussian_filter(box, 8)*1.6, 0, 1)        # без прямых краёв
    mead = ~water & (wm > 0.02) & ~banned
    dws = ndimage.distance_transform_edt(~water)
    hills = f.GY + (6 + 10*f.noise(11, rng))*sstep(4, 14, dws)*wm
    H = np.where(mead & ~free, np.maximum(f.GY, hills), H)
    zone = free | mead
    H = np.where(zone, ndimage.gaussian_filter(H, 2.0), H)
    H = np.minimum(H, balloon_cap(f))
    Hn = np.rint(np.where(V, H, np.maximum(H, f.GY))).astype(int)
    land = zone & (Hn >= SEA + 1)
    Hn = np.where(land & (np.abs(Hn - f.GY) >= 1) | (V & land), Hn, f.GY)
    cells = land & ((Hn != f.GY) | V)
    print(f'долина и холмы: меняется {int(cells.sum())}, дно долины {np.percentile(Hn[V & land], [5, 50, 95]) if (V & land).any() else "-"}')
    kill_trees(f, cells & ~V, np.minimum(f.G, Hn - YMIN))
    top, slope = materials(f, Hn, cells, 160, grass_below=200)
    top[cells & (Hn <= SEA + 2)] = 'sand'
    build(f, cells, Hn, top)
    CD = f.ID('minecraft:coarse_dirt')
    spots = (f.noise(4, rng) > 0.7) & cells & (top == 'grass') & V
    for i, j in zip(*np.nonzero(spots)): f.vol[i, Hn[i, j] - YMIN, j] = CD
    trees = scatter(np.argwhere(cells & (top == 'grass') & (slope < 1.2)), lambda: rng.uniform(10, 18), 0.5)
    for i, j in trees: place(f, i, j, Hn[i, j] - YMIN + 1, broadleaf('oak'))
    plants(f, cells & (top == 'grass'), Hn, 0.22, 0.04)
    print(f'  дубов {len(trees)}')
    finish(f, 'долина на месте пролива и холмы')


def backup():
    B.backup()


{'backup': backup, 'unspruce': unspruce, 'north2': north2, 'se2': se2, 'strait': strait}[part]()
