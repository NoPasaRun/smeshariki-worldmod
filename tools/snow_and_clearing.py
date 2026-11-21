#!/usr/bin/env python3
"""Две правки сохранённого мира: снеговые шапки на горах и расчищенная площадка перед расщелиной.

Запуск: snow_and_clearing.py <папка мира> [--write <новая папка копии>] [--restore-clearing <папка копии региона>] [--preview файл.png],
игра закрыта. Без --write мир не меняется, только отчёт (и предпросмотр). С --write вся папка region копируется в новую папку (сущности
не трогаются), потом пишутся блоки. Повторный запуск ничего не находит, если уже сделано.

Снег: верхний блок земли горного столбца становится snow_block, если он выше SNOW_FROM. Порода (stone, andesite, tuff, calcite, granite,
diorite) — везде; земля (dirt, grass_block, coarse_dirt, podzol, gravel) и лёд (ice, packed_ice, blue_ice) — только там, где карта
материалов острова (src/main/resources/island/materials.png, пиксель = мир + (550, 575)) говорит «горы» (индекс 0): на плато джунглей и на
берегах расщелины (трава и земля на y≈122) снега нет. Граница волнистая (гладкий шум ±SNOW_WAVE), выше DEEP_FROM шапка в два блока.
Деревья, вода, трава, скаты и всё прочее не трогаются; столбцы под кроной (верх — листва или ствол) пропускаются. Свет и карты высот не
меняются (оба блока непрозрачные).

Площадка: южный берег перед расщелиной (край z=-268, игрок идёт на север), полоса CLEAR_Z шириной CLEAR_X, где стоит игрок перед прыжком на
тарзанку. Убирается то, что мешает бежать и видеть по земле: стволы деревьев, чьи стволы стоят в полосе (целиком, с какао и лианами),
листва ниже LOW блоков над землёй, высокие растения и лианы ниже LOW. Кроны выше LOW остаются: у срезанных деревьев они остаются без
ствола и получают persistent=true, иначе листва осыплется. Стволы, начинающиеся ниже CHASM_Y (деревья расщелины с верёвками), не
трогаются. С --restore-clearing сначала возвращается прежняя площадка (чанки рамки берутся из копии до первой правки), потом применяется
нынешнее правило. Свет и карты высот в затронутых чанках пересчитываются: свет сбрасывается, игра посчитает при загрузке.
"""
import math, os, shutil, sys
import numpy as np
from multiprocessing import Pool
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca

Y_MIN, NY = -64, 384
SNOW_FROM, SNOW_WAVE, DEEP_FROM = 117, 3.5, 140
STONE = ('stone', 'andesite', 'tuff', 'calcite', 'granite', 'diorite')
EARTH_ICE = ('dirt', 'grass_block', 'coarse_dirt', 'podzol', 'gravel', 'ice', 'packed_ice', 'blue_ice')   # только на горах по карте материалов
CLEAR_X, CLEAR_Z = (452, 548), (-268, -257)     # [от, до): 96 × 11 блоков; z = -268 это первый ряд над краем, игрок стоит южнее
CHASM_Y = 118                                   # ствол, чей низ ниже, — дерево расщелины
LOW = 4                                         # блоков над землёй: ниже убирается, выше остаётся (головы и прыжок)
TALL = ('tall_grass', 'large_fern', 'fern', 'bush', 'firefly_bush', 'vine', 'bamboo', 'bamboo_sapling', 'sapling', 'jungle_sapling', 'melon', 'cocoa')
PLANTS = TALL + ('short_grass', 'azure_bluet', 'oxeye_daisy', 'dead_bush', 'poppy', 'dandelion', 'snow', 'leaf_litter', 'wildflowers',
                 'short_dry_grass', 'tall_dry_grass', 'seagrass', 'tall_seagrass', 'lily_pad')
NOBLOCK = ('air', 'cave_air', 'void_air', 'vine', 'tall_grass', 'short_grass', 'fern', 'large_fern', 'cocoa', 'snow', 'azure_bluet', 'oxeye_daisy',
           'dead_bush', 'bush', 'firefly_bush', 'poppy', 'dandelion', 'leaf_litter', 'wildflowers', 'short_dry_grass', 'tall_dry_grass')
HM_TYPES = {b'WORLD_SURFACE': 0, b'MOTION_BLOCKING': 1, b'MOTION_BLOCKING_NO_LEAVES': 2, b'OCEAN_FLOOR': 3}


_MAT = None
def mountain_mask(wx, wz):
    """Булева маска «горы» по карте материалов острова для массивов мировых координат."""
    global _MAT
    if _MAT is None:
        from PIL import Image
        _MAT = np.array(Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src/main/resources/island/materials.png')))
    px, pz = wx + 550, wz + 575
    ok = (px >= 0) & (px < _MAT.shape[1]) & (pz >= 0) & (pz < _MAT.shape[0])
    out = np.zeros(wx.shape, bool)
    out[ok] = _MAT[pz[ok], px[ok]] == 0
    return out


def bname(k): return k.split('[')[0].replace('minecraft:', '')


def read_chunk(reg, cx, cz):
    """Чанк -> (имя NBT, дерево NBT, объём [x, y, z] индексов, список имён блоков) или None."""
    b = reg.raw(cx, cz)
    if b is None: return None
    nm, c = mca.parse(b)
    names, idx = ['minecraft:air'], {'minecraft:air': 0}
    vol = np.zeros((16, NY, 16), np.uint16)
    for sec in c[b'sections'][1][1]:
        y = int.from_bytes(sec[b'Y'][1], 'big', signed=True); bs = sec.get(b'block_states')
        if bs is None or not -4 <= y < 20: continue
        pal = []
        for q in bs[1][b'palette'][1][1]:
            k = mca.key(q)
            if k not in idx: idx[k] = len(names); names.append(k)
            pal.append(idx[k])
        pal = np.array(pal)
        ids = pal[mca.unpack(bs[1][b'data'][1], len(pal))].reshape(16, 16, 16) if b'data' in bs[1] else np.full((16, 16, 16), pal[0])
        vol[:, (y + 4)*16:(y + 5)*16, :] = ids.transpose(2, 0, 1)
    return nm, c, vol, names


def write_sections(c, vol, names, changed_y, drop_light):
    """Перезаписать секции, где менялись блоки (changed_y — индексы секций 0..23). drop_light: свет удаляется и isLightOn=0,
    иначе игра считает свет готовым и чанк остаётся тёмным (2026-11-16, расширение острова)."""
    if drop_light: c[b'isLightOn'] = (1, b'\x00')
    for sec in c[b'sections'][1][1]:
        y = int.from_bytes(sec[b'Y'][1], 'big', signed=True); bs = sec.get(b'block_states')
        if drop_light:
            sec.pop(b'SkyLight', None); sec.pop(b'BlockLight', None)
        if bs is None or (y + 4) not in changed_y: continue
        arr = vol[:, (y + 4)*16:(y + 5)*16, :].transpose(1, 2, 0).reshape(4096)
        u, inv = np.unique(arr, return_inverse=True)
        bs[1][b'palette'] = (9, (10, [mca.state(names[i]) for i in u]))
        if len(u) == 1: bs[1].pop(b'data', None)
        else: bs[1][b'data'] = (12, mca.pack(inv, len(u)))


def set_heightmaps(c, vol, names):
    bn = [bname(k) for k in names]
    noblock = np.array([b in NOBLOCK for b in bn]); leaf = np.array([b.endswith('leaves') for b in bn]); water = np.array([b == 'water' for b in bn])
    air = np.array([b in ('air', 'cave_air', 'void_air') for b in bn])
    masks = (~air[vol], ~noblock[vol] | water[vol], (~noblock[vol] & ~leaf[vol]) | water[vol], ~noblock[vol] & ~water[vol])
    for (nmk, t), m in zip(HM_TYPES.items(), masks):
        top = np.where(m.any(1), NY - np.argmax(m[:, ::-1, :], axis=1), 0)   # [x, z], высота над низом мира
        w = np.zeros(math.ceil(256/7), np.uint64)
        for q_ in range(256):
            w[q_//7] |= np.uint64(int(top[q_ % 16, q_//16]) << ((q_ % 7)*9))
        c[b'Heightmaps'][1][nmk] = (12, w.astype('>u8').tobytes())


# ---------------------------------------------------------------- снег
def vnoise(x, z):
    """Гладкий шум в [-1, 1] от мировых координат (значения в узлах решётки по хэшу)."""
    def h(i, j):
        n = (i*374761393 + j*668265263) & 0xffffffff
        n = ((n ^ (n >> 13))*1274126177) & 0xffffffff
        return ((n ^ (n >> 16)) & 0xffff)/65535.0
    xi, zi = np.floor(x).astype(np.int64), np.floor(z).astype(np.int64); fx, fz = x - xi, z - zi
    fx, fz = fx*fx*(3 - 2*fx), fz*fz*(3 - 2*fz)
    a = h(xi, zi)*(1 - fx) + h(xi + 1, zi)*fx; b = h(xi, zi + 1)*(1 - fx) + h(xi + 1, zi + 1)*fx
    return (a*(1 - fz) + b*fz)*2 - 1


def snow_region(args):
    path, write = args
    reg = mca.Region(path); mask = np.zeros((512, 512), bool); cols = 0; chunks = 0
    parts = os.path.basename(path).split('.'); rx, rz = int(parts[1]), int(parts[2])
    for cz in range(32):
        for cx in range(32):
            if not reg.present(cx, cz): continue
            r = read_chunk(reg, cx, cz)
            if r is None: continue
            nm, c, vol, names = r
            bn = [bname(k) for k in names]
            cat = np.array([0 if b in PLANTS or b in ('air', 'cave_air', 'void_air') else (1 if b in STONE else (4 if b in EARTH_ICE else (3 if b.endswith('leaves') or b.endswith('_log') or b == 'water' else 2))) for b in bn], np.int8)
            cv = cat[vol]
            solid = cv != 0
            ok = solid.any(1)
            top = NY - 1 - np.argmax(solid[:, ::-1, :], axis=1)                       # [x, z]
            xs, zs = np.meshgrid(np.arange(16), np.arange(16), indexing='ij')
            topcat = cv[xs, top, zs]
            wx, wz = (rx*32 + cx)*16 + xs, (rz*32 + cz)*16 + zs
            thr = SNOW_FROM + SNOW_WAVE*vnoise(wx/12.0 + 0.37, wz/12.0 + 0.11)
            ytop = top + Y_MIN
            sel = ok & ((topcat == 1) | ((topcat == 4) & mountain_mask(wx, wz))) & (ytop >= thr)
            if not sel.any(): continue
            snow_id = len(names); names = names + ['minecraft:snow_block']
            changed = set()
            for x, z in zip(*np.nonzero(sel)):
                y = top[x, z]
                vol[x, y, z] = snow_id; changed.add(y >> 4)
                if ytop[x, z] >= DEEP_FROM and y > 0 and cat[vol[x, y - 1, z]] in (1, 4):
                    vol[x, y - 1, z] = snow_id; changed.add((y - 1) >> 4)
                mask[((cz*16) + z), ((cx*16) + x)] = True
            cols += int(sel.sum()); chunks += 1
            if write:
                write_sections(c, vol, names, changed, False)
                reg.put(cx, cz, mca.dump(nm, c))
    if write and chunks: reg.save()
    return path, cols, chunks, mask


# ---------------------------------------------------------------- площадка
def clearing(regs, report_only, preview_rows, restore_from=None):
    X0, X1 = CLEAR_X; Z0, Z1 = CLEAR_Z
    mx = 14                                              # запас по краям для кроны и стволов
    bx0, bx1, bz0, bz1 = (X0 - mx) >> 4, (X1 + mx - 1) >> 4, (Z0 - mx) >> 4, (Z1 + mx - 1) >> 4
    ox, oz = bx0*16, bz0*16; sx, sz = (bx1 - bx0 + 1)*16, (bz1 - bz0 + 1)*16
    idx = {'minecraft:air': 0}
    vol = np.zeros((sx, NY, sz), np.uint16); chunks = {}; srcs = {}
    for cx in range(bx0, bx1 + 1):
        for cz in range(bz0, bz1 + 1):
            f = f'r.{cx >> 5}.{cz >> 5}.mca'; p = os.path.join(regs, f)
            if p not in REGS: REGS[p] = mca.Region(p)
            src = os.path.join(restore_from, f) if restore_from else p      # откуда читать исходное состояние чанка
            if src not in srcs: srcs[src] = mca.Region(src)
            raw = srcs[src].raw(cx, cz)
            if raw is None: continue
            nm, c = mca.parse(raw); chunks[(cx, cz)] = (p, nm, c, raw)
            names, v = _volume(c)
            mp = np.array([idx.setdefault(k, len(idx)) for k in names])
            vol[(cx - bx0)*16:(cx - bx0 + 1)*16, :, (cz - bz0)*16:(cz - bz0 + 1)*16] = mp[v]
    nm_list = [None]*len(idx)
    for k, i in idx.items(): nm_list[i] = k
    bn = np.array([bname(k) for k in nm_list])
    is_log = np.array([b.endswith('_log') or b.endswith('_wood') for b in bn]); is_leaf = np.array([b.endswith('leaves') for b in bn])
    tall = np.isin(bn, TALL); vine = bn == 'vine'
    L = is_log[vol]; LF = is_leaf[vol]
    band = np.zeros(vol.shape, bool)
    band[X0 - ox:X1 - ox, :, Z0 - oz:Z1 - oz] = True
    # земля под ногами: верхний блок, не ствол, не листва, не растение
    ground_mask = (vol != 0) & ~L & ~LF & ~np.isin(bn, PLANTS)[vol]
    g1 = np.where(ground_mask.any(1), NY - np.argmax(ground_mask[:, ::-1, :], axis=1) + Y_MIN, Y_MIN)     # первая клетка над землёй, [x, z]
    yy = (np.arange(NY) + Y_MIN)[None, :, None]
    low = yy < (g1[:, None, :] + LOW)
    # деревья: связные компоненты стволов (26 соседей), задевающие полосу
    lab, n = ndimage.label(L, structure=np.ones((3, 3, 3)))
    hit = np.unique(lab[band & L]); hit = hit[hit > 0]
    objs = ndimage.find_objects(lab)
    cut = np.zeros(vol.shape, bool); skipped = 0; trees = 0
    for k in hit:
        if objs[k - 1][1].start + Y_MIN < CHASM_Y: skipped += 1; continue
        cut |= lab == k; trees += 1
    near = ndimage.binary_dilation(cut, structure=ndimage.generate_binary_structure(3, 1), iterations=1)
    cut |= near & (tall[vol] | vine[vol])                                  # лианы и какао у срезанных стволов
    cut |= band & low & (tall[vol] | vine[vol])                             # высокие растения и лианы на уровне головы
    cut |= band & low & LF                                                  # листва на уровне головы
    # кроны срезанных деревьев: остаются в воздухе, поэтому без опоры получают persistent=true
    st = ndimage.generate_binary_structure(3, 1)
    def reach(src, steps):
        seen = src.copy(); front = src
        for _ in range(steps):
            front = ndimage.binary_dilation(front, structure=st) & LF & ~seen
            seen |= front
        return seen & LF
    removed_logs = cut & L
    cand = reach(removed_logs, 7)
    keep = reach(L & ~removed_logs, 6)
    persist = LF & cand & ~keep & ~cut
    cut &= vol != 0
    print(f'площадка: деревьев срезано {trees} (стволы), пропущено (деревья расщелины) {skipped}, блоков убрано {int(cut.sum())}: '
          f'брёвна {int((cut & L).sum())}, листва {int((cut & LF).sum())}, растения и лианы {int((cut & (tall[vol] | vine[vol])).sum())}; '
          f'крон остаётся без ствола (persistent) {int(persist.sum())} блоков листвы')
    if preview_rows is not None:
        def top_of(v):
            m = v != 0
            return np.where(m.any(1), NY - 1 - np.argmax(m[:, ::-1, :], axis=1) + Y_MIN, -999)
        preview_rows.append((ox, oz, top_of(vol), top_of(np.where(cut, 0, vol))))
    if report_only: return 0
    # persistent=true: новые состояния листвы рядом с прежними
    twin = np.arange(len(nm_list))
    for i, k in enumerate(list(nm_list)):
        if is_leaf[i] and 'persistent=false' in k:
            nk = k.replace('persistent=false', 'persistent=true')
            if nk not in idx: idx[nk] = len(nm_list); nm_list.append(nk)
            twin[i] = idx[nk]
    vol[persist] = twin[vol[persist]]
    vol[cut] = 0
    changed_vox = cut | persist
    touched = 0
    for (cx, cz), (p, nm, c, raw) in chunks.items():
        a, b = (cx - bx0)*16, (cz - bz0)*16
        sub = changed_vox[a:a + 16, :, b:b + 16]
        if not sub.any():
            if restore_from: REGS[p].put(cx, cz, raw)                      # вернуть прежнее состояние чанка из копии
            continue
        v = vol[a:a + 16, :, b:b + 16]
        changed = {y >> 4 for y in np.nonzero(sub.any(axis=(0, 2)))[0]}
        write_sections(c, v, nm_list, changed, True)
        c[b'isLightOn'] = (1, b'\0')
        set_heightmaps(c, v, nm_list)
        REGS[p].put(cx, cz, mca.dump(nm, c)); touched += 1
    for r in REGS.values(): r.save()
    print('чанков записано (площадка):', touched, '(остальные в рамке возвращены из копии)' if restore_from else '')
    return touched


def _volume(c):
    """Дерево NBT чанка -> (список имён, объём [x, y, z] индексов)."""
    names, idx = ['minecraft:air'], {'minecraft:air': 0}
    vol = np.zeros((16, NY, 16), np.uint16)
    for sec in c[b'sections'][1][1]:
        y = int.from_bytes(sec[b'Y'][1], 'big', signed=True); bs = sec.get(b'block_states')
        if bs is None or not -4 <= y < 20: continue
        pal = []
        for q in bs[1][b'palette'][1][1]:
            k = mca.key(q)
            if k not in idx: idx[k] = len(names); names.append(k)
            pal.append(idx[k])
        pal = np.array(pal)
        ids = pal[mca.unpack(bs[1][b'data'][1], len(pal))].reshape(16, 16, 16) if b'data' in bs[1] else np.full((16, 16, 16), pal[0])
        vol[:, (y + 4)*16:(y + 5)*16, :] = ids.transpose(2, 0, 1)
    return names, vol


REGS = {}

if __name__ == '__main__':
    world = sys.argv[1]
    arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
    backup, preview, restore = arg('--write'), arg('--preview'), arg('--restore-clearing')
    reg_dir = os.path.join(world, 'dimensions/minecraft/overworld/region')
    files = sorted(os.path.join(reg_dir, f) for f in os.listdir(reg_dir) if f.endswith('.mca'))
    if backup:
        os.makedirs(backup, exist_ok=False)
        for f in files: shutil.copy2(f, backup)
        print('копия региона:', backup)
    rows = [] if preview else None
    clearing(reg_dir, not backup, rows, restore)
    with Pool(min(8, os.cpu_count() or 1)) as pool:
        res = pool.map(snow_region, [(f, bool(backup)) for f in files])
    total = sum(r[1] for r in res); print('снег: столбцов', total, 'чанков', sum(r[2] for r in res))
    if preview:
        from PIL import Image
        xs = [int(os.path.basename(p).split('.')[1]) for p, *_ in res]; zs = [int(os.path.basename(p).split('.')[2]) for p, *_ in res]
        gx0, gz0 = min(xs), min(zs)
        img = np.zeros(((max(zs) - gz0 + 1)*512, (max(xs) - gx0 + 1)*512), bool)
        for p, cols, ch, m in res:
            a, b = int(os.path.basename(p).split('.')[1]) - gx0, int(os.path.basename(p).split('.')[2]) - gz0
            img[b*512:(b + 1)*512, a*512:(a + 1)*512] = m
        Image.fromarray((img*255).astype(np.uint8)).save(preview)
        ox, oz, before, after = rows[0]
        h = np.concatenate([before, np.full((before.shape[0], 4), 0), after], axis=1)
        h = np.clip((h - 100)*5, 0, 255).astype(np.uint8)
        Image.fromarray(h.T).save(preview.replace('.png', '_площадка.png'))
        print('предпросмотр:', preview)
