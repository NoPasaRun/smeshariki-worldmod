#!/usr/bin/env python3
"""Пещера в конце тоннеля: неровные своды и стены, площадка под выходом (y=−55, как была), лавовое озеро, берег за ним, две тарзанки.

Запуск: final_cave.py <папка мира> [--preview файл.png] [--write <новая папка копии>]. Без --write мир не меняется. Игра должна быть закрыта.
Тоннель (всё до края обрыва) не трогает. После chasm_world.py запускать заново: тот строит прежнее плато с водой.
"""
import math, os, shutil, struct, sys, uuid
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, gaussian_filter1d
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca, tunnel_field as TF

SEED = 20261103
Y0, YTOP = -64, 16
FLOOR = TF.PLATEAU_Y                   # верх площадки и берега, y=−55
LAVA_TOP, LAVA_BOTTOM = FLOOR - 1, FLOOR - 3
# длина вдоль пещеры от края обрыва: площадка — куда долетают с края (без воды 20,8 блока при падении на 30), озеро, берег
LAKE_FROM, LAKE_TO, END = 25.0, 53.0, 73.0
ROPES = [(29.0, -1.0), (41.5, 1.0)]    # вдоль, поперёк; с W размах за верёвку около 12–14 блоков, дотянуться можно на 4
HAND_Y = FLOOR + 4.0                   # куда опускается конец верёвки: с площадки достаётся, над лавой ноги не заденут
ROPE_MAX = 24.0                        # RopeEntity.MAX_LENGTH

world = sys.argv[1]
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
backup, preview = arg('--write'), arg('--preview')
D = os.path.join(world, 'dimensions/minecraft/overworld/'); REG = D + 'region/'; ENT = D + 'entities/'
X0, X1, Z0, Z1 = 96, 207, -304, -209
cx0, cx1, cz0, cz1 = X0 >> 4, X1 >> 4, Z0 >> 4, Z1 >> 4
ox, oz = cx0*16, cz0*16; sx, sz = (cx1 - cx0 + 1)*16, (cz1 - cz0 + 1)*16; NY = YTOP - Y0
vol = np.zeros((sx, NY, sz), np.uint16); names = {'minecraft:air': 0}; lst = ['minecraft:air']; regs = {}; chunks = {}
def gid(k):
    if k not in names: names[k] = len(lst); lst.append(k)
    return names[k]
for cx in range(cx0, cx1 + 1):
    for cz in range(cz0, cz1 + 1):
        p = REG + f'r.{cx >> 5}.{cz >> 5}.mca'
        regs.setdefault(p, mca.Region(p)); b = regs[p].raw(cx, cz)
        if b is None: raise SystemExit(f'нет чанка {cx} {cz}')
        nm, c = mca.parse(b); chunks[(cx, cz)] = (p, nm, c)
        for sec in c[b'sections'][1][1]:
            y = int.from_bytes(sec[b'Y'][1], 'big', signed=True); bs = sec.get(b'block_states')
            if bs is None or not Y0 <= y*16 < YTOP: continue
            pal = np.array([gid(mca.key(q)) for q in bs[1][b'palette'][1][1]])
            ids = pal[mca.unpack(bs[1][b'data'][1], len(pal))].reshape(16, 16, 16) if b'data' in bs[1] else np.full((16, 16, 16), pal[0])
            vol[(cx - cx0)*16:(cx - cx0 + 1)*16, y*16 - Y0:y*16 - Y0 + 16, (cz - cz0)*16:(cz - cz0 + 1)*16] = ids.transpose(2, 0, 1)
orig = vol.copy()
base = lambda i: lst[i].split('[')[0].split(':')[1]
def isin(*n): return np.isin(vol, [i for i in range(len(lst)) if base(i) in n])

# ---------- оси пещеры
k = np.searchsorted(TF.ts, TF.LIP_S); lip = TF.tc[k]
ax = (TF.tc[-1] - lip)/np.hypot(*(TF.tc[-1] - lip))
gx, gz = np.meshgrid(np.arange(sx) + ox + .5, np.arange(sz) + oz + .5, indexing='ij')
lon = (gx - lip[0])*ax[0] + (gz - lip[1])*ax[1]; lat = -(gx - lip[0])*ax[1] + (gz - lip[1])*ax[0]
_, near = cKDTree(TF.tc).query(np.c_[gx.ravel(), gz.ravel()]); tunnel_side = (TF.ts[near].reshape(gx.shape) < TF.LIP_S) | (lon < 0.3)
rng = np.random.default_rng(SEED)
def field(sig, shape=(sx, NY, sz)):
    a = gaussian_filter(rng.normal(size=shape), sig, mode='wrap'); return a/a.std()

# ---------- объём: свод ниже над озером, чтобы верёвки доставали, и выше у выхода; стены и свод со складками
half = np.interp(lon, [0, 10, 28, 56, END], [8, 15, 18, 17, 13])
cap = np.clip((lon - (END - 12))/12, 0, 1); half = half*np.sqrt(np.maximum(1 - cap*cap, 0))
ceil = np.interp(lon, [0, 10, 24, 54, END], [-13, -17, -29, -30, -35])
yy = (np.arange(NY) + Y0)[None, :, None]
t = np.clip((yy - FLOOR)/(ceil[:, None, :] - FLOOR), 0, 1.2)
wall = np.sqrt(np.maximum(1 - t**2.4, 0))          # свод: стена у пола, сверху смыкается
n = 0.26*field(6) + 0.12*field(2.5) + 0.04*field(1.2)      # складки, выступы и ниши
inside = (np.abs(lat)[:, None, :]/np.maximum(half[:, None, :], 0.1) < wall + n) & (yy < ceil[:, None, :] + 7*n) & (lon[:, None, :] > 0.3) & (lon[:, None, :] < END + 8*n) & (yy > FLOOR) & ~tunnel_side[:, None, :]
# кочки на площадке и берегу, шагом в один блок
bumps = gaussian_filter(rng.normal(size=(sx, sz)), 3.0, mode='wrap'); bumps /= bumps.std()
edge = gaussian_filter1d(rng.normal(size=sz*4), 7, mode='wrap'); edge /= edge.std()
def wob(sign): return 1.6*np.interp(lat*sign, np.arange(-sz*2, sz*2), edge)
lake = (lon > LAKE_FROM + wob(1)) & (lon < LAKE_TO + wob(-1))
footprint = inside[:, FLOOR + 1 - Y0, :]
lake = lake & (binary_dilation(footprint, iterations=2))
mound = (bumps > 1.25) & ~lake & (np.abs(lat) > 8) & ~binary_dilation(lake, iterations=4)
inside[:, FLOOR + 1 - Y0, :] &= ~mound

# ---------- блоки
air_id, bed = 0, [i for i in range(len(lst)) if base(i) == 'bedrock']
is_bed = np.isin(vol, bed)
keep = np.repeat(tunnel_side[:, None, :], NY, 1) | is_bed
old_open = isin('air', 'cave_air', 'water', 'lava', 'sea_lantern') & ~keep
vol[old_open & ~inside] = gid('minecraft:deepslate')
vol[inside] = air_id
lava_id = gid('minecraft:lava[level=0]')
for y in range(LAVA_BOTTOM, LAVA_TOP + 1):
    vol[:, y - Y0, :][lake] = lava_id
vol[:, FLOOR - Y0, :][lake] = air_id
# над лавой уровень y=−55 пуст: у края озера он может быть в камне, его открывает lake; лава не вытечет, вокруг камень
solid = vol != air_id
lava_mask = vol == lava_id
open_ = (vol == air_id) | lava_mask
shell = binary_dilation(open_, iterations=2) & ~open_ & ~keep
# жар: у озера и низко чёрный камень, выше лёд; граница неровная
dist_lake = np.minimum(np.maximum(LAKE_FROM - lon, lon - LAKE_TO), 99)
heat = (-(yy - (FLOOR + 9))/7.0 + (-np.maximum(dist_lake, 0)/14.0 + 0.6)[:, None, :] + 0.45*field(3.5))
r = rng.random(vol.shape)
ice_p, ice_b = gid('minecraft:packed_ice'), gid('minecraft:blue_ice')
black = [gid('minecraft:blackstone'), gid('minecraft:basalt[axis=y]'), gid('minecraft:tuff'), gid('minecraft:deepslate')]
hot = shell & (heat > 0.5)
cold = shell & ~hot
vol[cold] = np.where(r[cold] < 0.25, ice_b, ice_p)
vol[hot] = np.array(black)[(r[hot]*4).astype(int)]
# пол: площадка и берег из камня: лёд скользит, а сюда прилетают и отсюда разбегаются
top = np.zeros((sx, sz), int) - 999
fl = (vol != air_id) & (vol != lava_id)
for i, j in zip(*np.nonzero(footprint & ~lake)):
    for y in (FLOOR + 1, FLOOR):
        if vol[i, y - Y0, j] != air_id and vol[i, y - Y0 + 1, j] == air_id:
            vol[i, y - Y0, j] = black[int(r[i, y - Y0, j]*3)]; top[i, j] = y; break
# фонари: в холодной части стен и свода
lamp = gid('minecraft:sea_lantern')
exposed = binary_dilation(open_, iterations=1) & (np.isin(vol, [ice_p, ice_b]))
pts = np.argwhere(exposed)
pts = pts[pts[:, 1] + Y0 > FLOOR + 6]
rng.shuffle(pts); lamps = []
for p in pts:
    if len(lamps) >= 60: break
    if all((p[0] - q[0])**2 + (p[1] - q[1])**2 + (p[2] - q[2])**2 > 49 for q in lamps): vol[p[0], p[1], p[2]] = lamp; lamps.append(p)

# ---------- тарзанки: верёвка не длиннее предела, при необходимости свод опускается каменным наплывом; путь размаха чист
ropes = []
carved = hung = 0
def ij_at(lo, la):
    x = lip[0] + ax[0]*lo - ax[1]*la; z = lip[1] + ax[1]*lo + ax[0]*la
    return int(math.floor(x)) - ox, int(math.floor(z)) - oz, x, z
for lo, la in ROPES:
    i, j, x, z = ij_at(lo, la)
    y = FLOOR + 5
    while y < YTOP and vol[i, y - Y0, j] in (air_id, lava_id): y += 1
    ry = y                                            # первый твёрдый блок над верёвкой
    want = min(ROPE_MAX - 0.5, ry - HAND_Y)
    bottom = int(math.ceil(HAND_Y + want)) + 0
    while ry > bottom:                                # наплыв: конус из льда вниз до нужной высоты
        ry -= 1; w = (y - ry)/2 + 1; hung += 1
        for a in range(-3, 4):
            for b in range(-3, 4):
                if a*a + b*b <= 1 + (ry - bottom)*0.6 and vol[i + a, ry - Y0, j + b] == air_id: vol[i + a, ry - Y0, j + b] = ice_p
    L = float(round(ry - 0.05 - HAND_Y, 1))
    assert L <= ROPE_MAX, L
    # размах вдоль пещеры: ±14 блоков, тело ниже руки на 2,5 и шириной 1
    for u in np.arange(-14, 14.01, 0.5):
        h = L - math.sqrt(max(L*L - u*u, 0))
        for d in np.arange(0, L + 0.01, 1.0):
            f = d/L
            xx, zz = x + ax[0]*u*f, z + ax[1]*u*f; yc = ry - 0.05 - (L - h)*f
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    q, w_ = int(math.floor(xx)) + da - ox, int(math.floor(zz)) + db - oz
                    for yb in (int(math.floor(yc)) - 1, int(math.floor(yc))):
                        if yb < ry - 1 and vol[q, yb - Y0, w_] not in (air_id, lava_id) and yb > FLOOR and not (q == i and w_ == j and yb >= ry):
                            vol[q, yb - Y0, w_] = air_id; carved += 1
        low = ry - 0.05 - L + h
        for yb in range(int(math.floor(low)) - 3, int(math.floor(low)) + 1):
            xx, zz = x + ax[0]*u, z + ax[1]*u
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    q, w_ = int(math.floor(xx)) + da - ox, int(math.floor(zz)) + db - oz
                    if yb > FLOOR and vol[q, yb - Y0, w_] not in (air_id, lava_id): vol[q, yb - Y0, w_] = air_id; carved += 1
    ropes.append((i + ox, ry, j + oz, L, lo))

# ---------- украшения: колонны и колья у стен, не на линии полёта и не у края озера
placed = []; columns = []
walk = np.argwhere((top > -999) & ~binary_dilation(lake, iterations=5))
rng.shuffle(walk)
for i, j in walk:
    if abs(lat[i, j]) < 8 or len(columns) >= 6: continue
    if not (abs(lat[i, j]) > half[i, j]*0.55): continue
    if any(math.hypot(i - a, j - b) < 9 for a, b in columns): continue
    y = top[i, j]; h = 0
    while vol[i, y + 1 + h - Y0, j] == air_id: h += 1
    if not 4 <= h <= 24: continue
    mat = 'ice' if h > 12 and rng.random() < 0.5 else 'stone'
    vol[i, y + 1 - Y0, j] = gid(f'gameplaymod:{mat}_column[part=bottom]')
    for q in range(2, h): vol[i, y + q - Y0, j] = gid(f'gameplaymod:{mat}_column[part=middle]')
    vol[i, y + h - Y0, j] = gid(f'gameplaymod:{mat}_column[part=top]')
    columns.append((i, j))
SP = (('small_', 2, .25), ('', 6, .75), ('big_', 10, 1.25))
for i, j in walk:
    if abs(lat[i, j]) < 6.5 or len(placed) >= 26: continue
    if any(math.hypot(i - a, j - b) < 2.5 for a, b, _ in placed) or any(math.hypot(i - a, j - b) < 3 for a, b in columns): continue
    y = top[i, j]; h = 0
    while vol[i, y + 1 + h - Y0, j] == air_id: h += 1
    size = SP[int(rng.choice(3, p=(.6, .32, .08)))]
    if size[1] + 1 > h: continue
    vol[i, y + 1 - Y0, j] = gid(f'gameplaymod:{size[0]}stone_spike[tip=true]')
    placed.append((i, j, size))
changed = vol != orig

# ---------- отчёт
footprint2 = (vol[:, FLOOR + 1 - Y0, :] != air_id) * 0
print('изменено блоков:', int(changed.sum()), 'лавы клеток:', int((vol == lava_id).sum()), 'поверхность озера:', int(lake.sum()))
print('площадка клеток:', int(((top > -999) & (lon < LAKE_FROM)).sum()), 'берег:', int(((top > -999) & (lon > LAKE_TO)).sum()), 'колонн', len(columns), 'кольев', len(placed), 'фонарей', len(lamps))
print('верёвки (x, y крепления, z, длина, вдоль):', ropes, 'наплывов рядов', hung, 'расчищено', carved)
print('воды осталось в пещере:', int((np.isin(vol, [i for i in range(len(lst)) if base(i) == 'water']) & ~keep).sum()))
# лава: у каждой клетки сбоку и снизу только лава или твёрдое
L = vol == lava_id; a = vol == air_id
leak = 0
for ax_, s_ in ((0, 1), (0, -1), (2, 1), (2, -1)):
    leak += int((L & np.roll(a, s_, ax_)).sum())
leak += int((L & np.roll(a, 1, 1)).sum())
print('клеток лавы с воздухом сбоку или снизу:', leak)
op = (vol == air_id) | L
bound = binary_dilation(inside | L, iterations=3) & op & ~inside & ~L & ~keep
bound[:, FLOOR - Y0, :] &= ~lake                                   # над лавой пусто нарочно
print('чужого воздуха рядом с пещерой:', int(bound.sum()))
# щель выхода: у края обрыва есть воздух
li, lj = int(lip[0] + ax[0]*1.5) - ox, int(lip[1] + ax[1]*1.5) - oz
print('высоты у края обрыва:', [y + Y0 for y in range(NY) if vol[li, y, lj] == air_id][:1], '...', [y + Y0 for y in range(NY) if vol[li, y, lj] == air_id][-1:])
lam = np.array([p[1] + Y0 for p in lamps]); print('фонари y', lam.min(), lam.max())
if preview:
    from PIL import Image
    pal = {air_id: (20, 20, 30), lava_id: (255, 110, 20), lamp: (190, 240, 255), ice_p: (150, 190, 230), ice_b: (90, 140, 240)}
    def colour(v):
        if v in pal: return pal[v]
        n_ = base(v)
        if 'column' in n_: return (240, 230, 90)
        if 'spike' in n_: return (240, 90, 200)
        if n_ in ('blackstone', 'basalt', 'tuff'): return (55, 50, 60)
        return (95, 95, 100)
    rows = []
    for la in (0, 9, -9):   # разрезы вдоль
        im = np.zeros((NY*4, 90*4, 3), np.uint8)
        for lo_ in range(-8, 82):
            x = lip[0] + ax[0]*lo_ - ax[1]*la; z = lip[1] + ax[1]*lo_ + ax[0]*la
            i, j = int(x) - ox, int(z) - oz
            if not (0 <= i < sx and 0 <= j < sz): continue
            for y in range(NY): im[(NY - 1 - y)*4:(NY - y)*4, (lo_ + 8)*4:(lo_ + 9)*4] = colour(vol[i, y, j])
        rows.append(im)
    ims = []
    for lo_ in (12, 35, 56):  # поперёк
        im = np.zeros((NY*4, 60*4, 3), np.uint8)
        for la in range(-30, 30):
            x = lip[0] + ax[0]*lo_ - ax[1]*la; z = lip[1] + ax[1]*lo_ + ax[0]*la
            i, j = int(x) - ox, int(z) - oz
            if not (0 <= i < sx and 0 <= j < sz): continue
            for y in range(NY): im[(NY - 1 - y)*4:(NY - y)*4, (la + 30)*4:(la + 31)*4] = colour(vol[i, y, j])
        ims.append(im)
    top_im = np.zeros((sz*2, sx*2, 3), np.uint8)
    for i in range(sx):
        for j in range(sz):
            yy_ = FLOOR + 1 - Y0
            v = vol[i, yy_ - 2, j] if vol[i, yy_ - 2, j] == lava_id else vol[i, yy_, j]
            top_im[j*2:j*2 + 2, i*2:i*2 + 2] = colour(v)
    W = max(rows[0].shape[1], sum(i.shape[1] for i in ims), top_im.shape[1])
    pad = lambda a: np.pad(a, ((0, 4), (0, W - a.shape[1]), (0, 0)))
    Image.fromarray(np.vstack([pad(r) for r in rows[:2]] + [pad(np.hstack(ims)), pad(top_im)])).save(preview)
if not backup:
    print('без --write мир не изменён'); sys.exit()

# ---------- запись
os.makedirs(os.path.join(backup, 'region')); os.makedirs(os.path.join(backup, 'entities'))
for p in {v[0] for v in chunks.values()}: shutil.copy2(p, os.path.join(backup, 'region'))
touched = 0
for (cx, cz), (p, nm, c) in chunks.items():
    a, b = (cx - cx0)*16, (cz - cz0)*16
    if not changed[a:a + 16, :, b:b + 16].any(): continue
    touched += 1
    for sec in c[b'sections'][1][1]:
        y = int.from_bytes(sec[b'Y'][1], 'big', signed=True); bs = sec.get(b'block_states')
        sec.pop(b'SkyLight', None); sec.pop(b'BlockLight', None)
        if bs is None or not Y0 <= y*16 < YTOP: continue
        oy = y*16 - Y0
        if not changed[a:a + 16, oy:oy + 16, b:b + 16].any(): continue
        arr = vol[a:a + 16, oy:oy + 16, b:b + 16].transpose(1, 2, 0).reshape(4096)
        u, inv = np.unique(arr, return_inverse=True)
        bs[1][b'palette'] = (9, (10, [mca.state(lst[i]) for i in u]))
        if len(u) == 1: bs[1].pop(b'data', None)
        else: bs[1][b'data'] = (12, mca.pack(inv, len(u)))
    c[b'isLightOn'] = (1, b'\0')
    regs[p].put(cx, cz, mca.dump(nm, c))
for r in regs.values(): r.save()
eregs = {}; dv = next(iter(chunks.values()))[2][b'DataVersion']
dbl = lambda v: struct.pack('>d', v)
for rx, ry, rz, L_, _lo in ropes:
    cx, cz = rx >> 4, rz >> 4; p = ENT + f'r.{cx >> 5}.{cz >> 5}.mca'
    if p not in eregs:
        if os.path.exists(p): shutil.copy2(p, os.path.join(backup, 'entities'))
        eregs[p] = mca.Region(p)
    b = eregs[p].raw(cx, cz)
    if b is None: nm, c = b'', {b'Position': (11, struct.pack('>ii', cx, cz)), b'DataVersion': dv, b'Entities': (9, (10, []))}
    else: nm, c = mca.parse(b)
    if c[b'Entities'][1][0] == 0: c[b'Entities'] = (9, (10, c[b'Entities'][1][1]))
    ax_, ay_, az_ = rx + 0.5, ry - 0.05, rz + 0.5
    c[b'Entities'][1][1].append({b'id': (8, b'gameplaymod:tarzan_rope'), b'Pos': (9, (6, [dbl(ax_), dbl(ay_ - L_), dbl(az_)])), b'Motion': (9, (6, [dbl(0.0)]*3)),
        b'Rotation': (9, (5, [struct.pack('>f', 0.0)]*2)), b'UUID': (11, uuid.uuid4().bytes),
        b'AX': (6, dbl(ax_)), b'AY': (6, dbl(ay_)), b'AZ': (6, dbl(az_)), b'Length': (5, struct.pack('>f', float(L_))),
        b'BX': (3, struct.pack('>i', rx)), b'BY': (3, struct.pack('>i', ry)), b'BZ': (3, struct.pack('>i', rz)), b'Attached': (3, struct.pack('>i', 0))})
    eregs[p].put(cx, cz, mca.dump(nm, c))
for r in eregs.values(): r.save()
print('записано чанков:', touched, 'верёвок:', len(ropes), 'копия:', backup)
