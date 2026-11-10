#!/usr/bin/env python3
"""Каменная арка в области «руины» (референс: два наклонных столба держат огромный валун, мох на уступах, валуны у подножия, песок спереди).

Запуск: ruins_arch.py <папка мира> [--write <новая папка копии>] [--dump файл.npz], игра закрыта. Без --write только отчёт. С --write все файлы
region копируются в новую папку. Не идемпотентен: повторный запуск поверх готового сначала вернуть копию.

Арка стоит на песчаной поляне области (центр CX, CZ), проём смотрит на север и юг, пролёт по x. Форма — неявные поверхности с шумом:
два столба (эллиптическое сечение, сужаются и наклоняются к середине, уступы-«пояса» каждые 4 блока), шапка — сплющенный снизу эллипсоид,
валуны у подножия. Порода: stone / andesite / tuff разводами; мох и трава на верхних гранях уступов и шапки. Деревья в рамке постройки
убираются вместе с осиротевшей листвой. Свет в затронутых чанках сбрасывается, карты высот пересчитываются.
"""
import math, os, shutil, sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca
import snow_and_clearing as S

NY, YMIN = S.NY, S.Y_MIN
OX, OZ, SX, SZ = 16, -416, 64, 64                      # рамка чтения: x 16..79, z -416..-353
CX, CZ = 51, -385                                      # центр арки
CLEAR = (30, 72, -399, -371)                           # где убираются деревья: x0, x1, z0, z1
SEED = 26110804

world = sys.argv[1]
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
backup, dump = arg('--write'), arg('--dump')
REG = os.path.join(world, 'dimensions/minecraft/overworld/region')

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
NEW = ['minecraft:stone', 'minecraft:andesite', 'minecraft:tuff', 'minecraft:moss_block', 'minecraft:short_grass', 'minecraft:fern', 'minecraft:sand',
       'minecraft:grass_block[snowy=false]', 'minecraft:cobblestone']
for k in NEW: idx.setdefault(k, len(idx))
nm_list = [None]*len(idx)
for k, i in idx.items(): nm_list[i] = k
ID = lambda k: idx[k]
orig = vol.copy()
bn = np.array([S.bname(k) for k in nm_list])
rng = np.random.default_rng(SEED)
vn = lambda x, z: np.asarray(S.vnoise(np.asarray(x, float), np.asarray(z, float)))


def cats():
    air = np.isin(bn, ['air', 'cave_air', 'void_air'])[vol]; water = (bn == 'water')[vol]
    leaf = np.array([b.endswith('leaves') for b in bn])[vol]; log = np.array([b.endswith('_log') for b in bn])[vol]
    plant = np.isin(bn, list(S.PLANTS))[vol]
    return air, water, leaf, log, plant


air, water, leaf, log, plant = cats()
gm = ~air & ~water & ~leaf & ~log & ~plant
G = np.where(gm.any(1), NY - 1 - np.argmax(gm[:, ::-1, :], axis=1) + YMIN, 0)          # верхний блок земли [x, z]
x0g, x1g = CX - 18 - OX, CX + 18 - OX; z0g, z1g = CZ - 10 - OZ, CZ + 10 - OZ
Y0 = int(np.median(G[x0g:x1g, z0g:z1g]))                                                # уровень земли под аркой
print(f'центр арки ({CX}, {CZ}), земля y={Y0} (в рамке арки {int(G[x0g:x1g, z0g:z1g].min())}..{int(G[x0g:x1g, z0g:z1g].max())})')

# ---------------------------------------------------------------- расчистка деревьев
cx0, cx1, cz0, cz1 = CLEAR
box = np.zeros(vol.shape, bool)
box[cx0 - OX:cx1 - OX, :, cz0 - OZ:cz1 - OZ] = True
yy = (np.arange(NY) + YMIN)[None, :, None]
rm = box & (leaf | log | plant) & (yy > G[:, None, :])
removed_logs = rm & log
rm_all = rm.copy()
# осиротевшая листва: дальше 6 блоков по листве от оставшихся стволов
st6 = ndimage.generate_binary_structure(3, 1)
logs_left = log & ~removed_logs
seen = logs_left.copy(); front = logs_left
for _ in range(6):
    front = ndimage.binary_dilation(front, structure=st6) & leaf & ~seen & ~rm
    seen |= front
orphan = leaf & ~seen & ~rm_all & ndimage.binary_dilation(rm_all, structure=np.ones((3, 3, 3)), iterations=7)
vol[rm_all | orphan] = 0
print(f'расчистка: деревьев/растений убрано {int(rm_all.sum())}, осиротевшей листвы {int(orphan.sum())}')

# ---------------------------------------------------------------- форма арки
xs = (np.arange(SX) + OX)[:, None, None] + 0.5 - CX
ts = (np.arange(NY) + YMIN)[None, :, None] - Y0 + 0.0
zs = (np.arange(SZ) + OZ)[None, None, :] + 0.5 - CZ
nz = (np.asarray(vn(xs/4.2 + ts/5.5, zs/4.2 - ts/6.5)) + 0.5*np.asarray(vn(xs/2.1 + 7.0 + ts/3.0, zs/2.1 - 3.0)))/1.5   # -1..1 шероховатость
band = ((np.floor(ts/4.0) % 2) - 0.5)*0.9                       # пояса: чередуются широкие и узкие по 4 блока, получаются уступы


def pillar(sign, base_u, top_u, a0, a1, b0, b1, v_off):
    s = np.clip(ts/17.0, 0, 1); s = s*s*(3 - 2*s)*0.55 + s*0.45
    uc = sign*(base_u + (top_u - base_u)*s)
    a = a0 + (a1 - a0)*s + band*0.8
    b = b0 + (b1 - b0)*s + band*0.6
    vc = v_off*s
    r2 = ((xs - uc)/a)**2 + ((zs - vc)/b)**2
    return (r2 <= 1.0 + 0.30*nz) & (ts >= -4) & (ts <= 18.5 + 0.8*nz)


left = pillar(-1, 10.6, 7.0, 5.6, 4.0, 5.3, 3.9, 0.6)
right = pillar(+1, 10.9, 7.6, 6.2, 4.4, 5.6, 4.1, -0.5)
# шапка: сплющенный снизу эллипсоид (плоское дно, округлая макушка)
dt = ts - 19.6
rt = np.where(dt < 0, 5.6, 6.6)
cap_val = (xs - 0.4)**2/12.0**2 + (zs/6.6)**2 + (np.abs(dt)/rt)**2.6
cap = cap_val <= 1.0 + 0.22*nz
solid = left | right | cap
# валуны у подножия: (u, v, высота центра, радиусы)
BOULDERS = [(14.5, 3.0, 1.0, 3.6, 3.0, 2.6), (16.5, -3.5, 0.5, 2.8, 2.6, 2.0), (11.5, 6.8, 0.2, 2.4, 2.0, 1.8), (-14.5, 3.2, 0.6, 3.2, 2.8, 2.2),
            (-12.0, -6.4, 0.2, 2.2, 1.8, 1.6), (0.5, 7.5, -0.4, 1.6, 1.3, 1.2), (-5.5, 6.5, -0.6, 1.3, 1.1, 1.0)]
for (bu, bv, bh, ru, rv, rh) in BOULDERS:
    solid |= ((xs - bu)/ru)**2 + ((zs - bv)/rv)**2 + ((ts - bh)/rh)**2 <= 1.0 + 0.25*nz
solid &= ~((ts < -0.5) & (np.arange(NY)[None, :, None] + YMIN <= G[:, None, :] - 3))          # внизу не трогаем породу под землёй, только добавляем над ней
print('арка: блоков формы', int(solid.sum()), ', высота до y =', int(np.nonzero(solid.any(axis=(0, 2)))[0].max()) + YMIN)

# ---------------------------------------------------------------- материалы
X3 = (np.arange(SX) + OX)[:, None, None] + 0.0
Y3 = (np.arange(NY) + YMIN)[None, :, None] + 0.0
Z3 = (np.arange(SZ) + OZ)[None, None, :] + 0.0
# разводы: наклонные трещины (тёмный туф), светлее на поверхностях выше
n = 0.65*np.asarray(vn(X3/3.4 - Y3/7.0, Z3/3.4 + Y3/9.0)) + 0.35*np.asarray(vn(X3/9.0 + 13.0, Z3/9.0 + Y3/12.0))
mat = np.full(vol.shape, ID('minecraft:stone'), np.uint16)
mat = np.where(n < -0.32, ID('minecraft:tuff'), mat)
mat = np.where((n >= -0.32) & (n < -0.04), ID('minecraft:andesite'), mat)
mat = np.where((n >= 0.46), ID('minecraft:andesite'), mat)
vol[solid] = mat[solid]
# мох и трава на верхних гранях выше земли (кроме шапки целиком: пятнами)
air2 = np.isin(bn, ['air', 'cave_air', 'void_air'])[vol]
topface = solid & np.roll(air2, -1, axis=1) & (Y3 > Y0 + 1.5)
mossn = np.asarray(vn(X3/3.2 + 21.0 + 0*Y3, Z3/3.2 - 9.0 + 0*Y3))
moss = topface & (mossn > 0.12)
vol[moss] = ID('minecraft:moss_block')
for i, j, k in np.argwhere(moss):
    if vol[i, j + 1, k] == 0 and rng.random() < 0.7: vol[i, j + 1, k] = ID('minecraft:short_grass') if rng.random() < 0.75 else ID('minecraft:fern')
# низ у земли: трава вокруг подножия
print('мха на арке:', int(moss.sum()))

chg = vol != orig
print('всего изменено блоков:', int(chg.sum()))
if dump: np.savez_compressed(dump, vol=vol, orig=orig, names=np.array(nm_list, dtype=object), OX=OX, OZ=OZ)
if not backup:
    print('без --write мир не изменён'); sys.exit()

os.makedirs(backup, exist_ok=False)
for f in os.listdir(REG):
    if f.endswith('.mca'): shutil.copy2(os.path.join(REG, f), backup)
print('копия региона:', backup)
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
