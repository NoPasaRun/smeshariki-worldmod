#!/usr/bin/env python3
"""Паркур над лавовым озером в пещере конца тоннеля: бортик озера, столбы, площадка жерла и шахта до поверхности.

Запуск: lava_parkour.py <папка мира> [--preview файл.png] [--write <новая папка копии>]. Без --write мир не меняется. Игра должна быть закрыта.
Работает поверх final_cave.py. Повторно запускать только на копии: столбы и шахта уже будут в мире, а маркеры добавятся заново.
Шахта идёт над полублоком в своде, он отмечал её место. Сущности: gameplaymod:launcher на площадке и gameplaymod:lava_tide над озером.
"""
import math, os, shutil, struct, sys, uuid
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca, tunnel_field as TF

SEED = 20261104
Y0 = -64; Y1 = 208                  # по границе секции
LAVA_BASE, LAVA_PEAK = -56, -48       # верхняя клетка лавы внизу и наверху прилива: наверху вровень с площадкой жерла
RIM_TOP = -47                         # верхняя клетка бортика: лава на всю высоту прилива закрыта
RIM_WIDTH = 2.3
HOLD_LOW, STEP_TICKS, HOLD_HIGH = 600, 120, 200
SLAB = (131, -43, -258)               # отмеченное место шахты
# столбы вдоль и поперёк пещеры: (вдоль, поперёк, сторона, поверхность, особенность). Лава поднимается и заливает столбы снизу вверх
PADS = [
    (28.5, 2.0, 3, -48, None), (32.5, -0.5, 2, -49, None),            # спуск с бортика
    (36.0, -3.0, 3, -50, 'spike'),                                    # кол посередине: садиться на край
    (40.0, -1.5, 1, -50, None),                                       # точный прыжок
    (42.5, 1.0, 2, -51, 'launcher'),                                  # маленькое жерло через широкий пролёт
    (46.5, 6.0, 3, -51, 'column'),                                    # посадка, колонна в углу
    (49.5, 3.5, 1, -50, None), (51.5, 1.0, 1, -49, None),             # узкие столбы вверх
    (53.5, 3.5, 2, -48, None),
]
FINAL = (51.5, 7.5, 4, -47)           # площадка жерла, полублок над ней
TOP_CLEAR = 3                         # над поверхностью столба свободно

world = sys.argv[1]
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
backup, preview = arg('--write'), arg('--preview')
D = os.path.join(world, 'dimensions/minecraft/overworld/'); REG = D + 'region/'; ENT = D + 'entities/'
X0, X1, Z0, Z1 = 96, 207, -304, -209
cx0, cx1, cz0, cz1 = X0 >> 4, X1 >> 4, Z0 >> 4, Z1 >> 4
ox, oz = cx0*16, cz0*16; sx, sz = (cx1 - cx0 + 1)*16, (cz1 - cz0 + 1)*16; NY = Y1 - Y0
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
            if bs is None or not Y0 <= y*16 < Y1: continue
            pal = np.array([gid(mca.key(q)) for q in bs[1][b'palette'][1][1]])
            ids = pal[mca.unpack(bs[1][b'data'][1], len(pal))].reshape(16, 16, 16) if b'data' in bs[1] else np.full((16, 16, 16), pal[0])
            vol[(cx - cx0)*16:(cx - cx0 + 1)*16, y*16 - Y0:y*16 - Y0 + 16, (cz - cz0)*16:(cz - cz0 + 1)*16] = ids.transpose(2, 0, 1)
orig = vol.copy()
base = lambda i: lst[i].split('[')[0].split(':')[1]
lava_id = gid('minecraft:lava[level=0]'); air = 0
assert any(base(i) == 'oak_slab' for i in range(len(lst))) or '--force' in sys.argv, 'нет полублока: похоже, скрипт уже запускался'
k = np.searchsorted(TF.ts, TF.LIP_S); lip = TF.tc[k]; ax = (TF.tc[-1] - lip)/np.hypot(*(TF.tc[-1] - lip))
gx, gz = np.meshgrid(np.arange(sx) + ox + .5, np.arange(sz) + oz + .5, indexing='ij')
lon = (gx - lip[0])*ax[0] + (gz - lip[1])*ax[1]; lat = -(gx - lip[0])*ax[1] + (gz - lip[1])*ax[0]
def world_xz(lo, la): return lip[0] + ax[0]*lo - ax[1]*la, lip[1] + ax[1]*lo + ax[0]*la
rng = np.random.default_rng(SEED)
black = [gid('minecraft:blackstone'), gid('minecraft:basalt[axis=y]'), gid('minecraft:tuff'), gid('minecraft:deepslate')]
yi = lambda y: y - Y0
lake = vol[:, yi(LAVA_BASE), :] == lava_id
print('озеро клеток', int(lake.sum()), 'вдоль', round(lon[lake].min(), 1), round(lon[lake].max(), 1), 'поперёк', round(lat[lake].min(), 1), round(lat[lake].max(), 1))

# ---------- бортик и подъём на него со стороны площадки и берега
inner = distance_transform_edt(lake)
rim = lake & (inner <= RIM_WIDTH)
floor_top = np.full((sx, sz), -999)
for y in (-54, -55):
    m = (vol[:, yi(y), :] != air) & (vol[:, yi(y), :] != lava_id) & (vol[:, yi(y) + 1, :] == air) & (floor_top == -999)
    floor_top[m] = y
walk = (floor_top > -999) & ~lake
out = distance_transform_edt(~rim)
def fill(i, j, top):
    for y in range(-58, top + 1):
        if base(vol[i, yi(y), j]) != 'bedrock': vol[i, yi(y), j] = black[int(rng.random()*4)]
for i, j in zip(*np.nonzero(rim)): fill(i, j, RIM_TOP)
ramp = walk & (out <= (RIM_TOP - (-55)))
for i, j in zip(*np.nonzero(ramp | rim)):                # колья и колонны, оказавшиеся на подъёме, снимаются целиком
    for y in range(-55, -30):
        n_ = base(vol[i, yi(y), j])
        if 'spike' in n_ or 'column' in n_: vol[i, yi(y), j] = air
bumped = 0
for i, j in zip(*np.nonzero(ramp)):
    top = max(floor_top[i, j], int(RIM_TOP - math.ceil(out[i, j] - 0.5)))
    for y in range(floor_top[i, j], top + 1):
        if vol[i, yi(y), j] == air or 'spike' in base(vol[i, yi(y), j]) or 'column' in base(vol[i, yi(y), j]):
            vol[i, yi(y), j] = black[int(rng.random()*4)]; bumped += 1
# боковые щели: лава не должна уйти поверх бортика в полости стен на этих высотах
lake_open = lake & ~rim
seed = np.unravel_index(np.argmax(distance_transform_edt(lake_open)), lake.shape)
seed_lon, seed_lat = lon[seed], lat[seed]

# ---------- столбы
def pad(lo, la, size, surf):
    cells = []
    for i in range(sx):
        pass
    x, z = world_xz(lo, la)
    a0 = int(round(x - size/2)); b0 = int(round(z - size/2))
    for a in range(a0, a0 + size):
        for b in range(b0, b0 + size):
            i, j = a - ox, b - oz
            for y in range(-58, surf):
                if base(vol[i, yi(y), j]) != 'bedrock': vol[i, yi(y), j] = black[int(rng.random()*4)]
            vol[i, yi(surf - 1), j] = black[int(rng.random()*2)]
            for y in range(surf, surf + TOP_CLEAR):
                if vol[i, yi(y), j] not in (air, lava_id): vol[i, yi(y), j] = air
            cells.append((a, b))
    return (x, z, size, surf, cells)
pads = [pad(*p[:4]) for p in PADS]
final = pad(*FINAL)
extras = []
for spec, q in zip(PADS, pads):
    if spec[4] == 'spike':
        a, b = q[4][len(q[4])//2]; vol[a - ox, yi(q[3]), b - oz] = gid('gameplaymod:small_stone_spike[tip=true]')
    elif spec[4] == 'column':
        a, b = q[4][-1]; y = q[3]
        while vol[a - ox, yi(y), b - oz] == air: y += 1
        top_ = y - 1
        vol[a - ox, yi(q[3]), b - oz] = gid('gameplaymod:stone_column[part=bottom]')
        for t_ in range(q[3] + 1, top_): vol[a - ox, yi(t_), b - oz] = gid('gameplaymod:stone_column[part=middle]')
        vol[a - ox, yi(top_), b - oz] = gid('gameplaymod:stone_column[part=top]')
        print('колонна', a, q[3], b, 'высота', top_ - q[3] + 1)
HOP = None
for n_, spec in enumerate(PADS):
    if spec[4] == 'launcher':
        q = pads[n_]; t_ = pads[n_ + 1]
        yaw_ = math.degrees(math.atan2(-(t_[0] - q[0]), t_[1] - q[1]))
        HOP = (q[0], q[3], q[1], yaw_, math.hypot(t_[0] - q[0], t_[1] - q[1]))
        print(f'маленькое жерло: {q[0]:.1f} {q[3]} {q[1]:.1f}, курс {yaw_:.0f}, дальность {HOP[4]:.1f}, высота 3')
# в шахте полублок снимается; площадка должна накрыть его столб
slab_i, slab_j = SLAB[0] - ox, SLAB[2] - oz
assert (SLAB[0], SLAB[2]) in final[4], f'площадка не под полублоком: {final[4][:4]}'
print('путь: край до края, подъём; 3,6 — прыжок с разбега по ровному, на подъём 1 меньше')
near_rim = min(((gx[i, j] - pads[0][0])**2 + (gz[i, j] - pads[0][1])**2, gx[i, j], gz[i, j]) for i, j in zip(*np.nonzero(rim)))
route = [('бортик', (near_rim[1], near_rim[2], 1, RIM_TOP + 1))] + [(f'столб {n}', (q[0], q[1], q[2], q[3])) for n, q in enumerate(pads)] + [('площадка', (final[0], final[1], final[2], final[3]))]
worst = 0
for (na, a), (nb, b) in zip(route, route[1:]):
    d = math.hypot(a[0] - b[0], a[1] - b[1]); gap = d - (a[2] + b[2])/2; rise = b[3] - a[3]
    print(f'  {na} -> {nb}: зазор {gap:.1f}, подъём {rise:+d}, поверхность {b[3]}')
    if rise > 1: raise SystemExit(f'подъём больше прыжка: {na} -> {nb}')

# ---------- шахта: тело летит по дуге, шахта вокруг неё, выход над землёй, падение за края
fx, fz = SLAB[0] + .5, SLAB[2] + .5
ground = -999
COURSE = (0.0, -1.0); YAW = 180.0                    # на север, в море: берег около z=−300
def surface_at(i, j):
    k_ = np.nonzero(vol[i, :, j] != air)[0]
    return int(k_.max()) + Y0
area = [surface_at(slab_i + a, slab_j + b) for a in range(-9, 10) for b in range(-22, 7)]
ground = max(area)
G, DY = 0.08, 0.98
def apex(v):
    y = 0
    while v > 0: y += v; v = (v - G)*DY
    return y
def speed_for(h):
    lo_, hi_ = 0.0, 16.0
    for _ in range(40):
        m_ = (lo_ + hi_)/2
        if apex(m_) < h: lo_ = m_
        else: hi_ = m_
    return hi_
def arc(h):
    v_ = speed_for(h); ys = [0.0]; v0 = v_
    for t_ in range(1, 600):
        ys.append(ys[-1] + v_); v_ = (v_ - G)*DY
        if ys[-1] <= 0 and t_ > 1: break
    return v0, ys
SEA_Y, SHORE_Z = 62, -300
WANT_LAND = SHORE_Z - 8                               # в воду с запасом от берега; дальше Distance не позволяет
sh = ground + 1 - final[3]; water = SEA_Y + 1 - final[3]
choice = None
for h in range(int(sh + 12), 300, 4):
    vy0, ys = arc(h)
    if vy0 > 9.5: break                               # быстрее не кидаем: сервер считает слишком быструю езду нарушением
    T = len(ys) - 1
    up = next(n for n, y_ in enumerate(ys) if y_ >= sh); down = next(n for n, y_ in enumerate(ys) if n > up and y_ < water)
    need = (fz - WANT_LAND)*T/down                    # дальность жерла, дающая посадку в нужной точке
    if need > 64: continue
    xe = need*up/T; R = xe/2 + 2.8
    choice = (h, vy0, T, up, down, need, xe, R); break
assert choice, 'до воды не долететь'
H, vy0, T, up, down, DIST, xe, R = choice
DIST = float(DIST); cen = xe/2; xl = DIST*down/T
cx_, cz_ = fx + COURSE[0]*cen, fz + COURSE[1]*cen
print(f'жерло: Height {H}, скорость вверх {vy0:.2f}, Distance {DIST:.1f}, полёт до уровня жерла {T} тиков, до воды {down}; выход на {xe:.1f} от жерла, в воду на {xl:.1f} (z={fz - xl:.0f}); шахта центр ({cx_:.1f}, {cz_:.1f}) радиус {R:.1f}, земля y={ground}')
clear_min = 1e9
for t_ in range(up, down + 1):
    z_ = fz - DIST*t_/T; y_ = final[3] + ys[t_]; j_ = int(math.floor(z_)) - oz
    if 0 <= j_ < sz: clear_min = min(clear_min, y_ - max(surface_at(slab_i + a, j_) for a in (-2, 0, 2)))
print(f'над скалой на пути в воду запас не меньше {clear_min:.1f} блока (в рамке чтения, до z={oz})')
noise = gaussian_filter(rng.normal(size=(NY, 96)), (3, 4), mode='wrap'); noise /= noise.std()
lamp = gid('minecraft:sea_lantern')
carved = 0; lamps = 0
y_from = final[3] + 2
for y in range(y_from, min(Y1, ground + 14)):
    for i in range(int(cx_ - R - 3) - ox, int(cx_ + R + 3) - ox + 1):
        for j in range(int(cz_ - R - 3) - oz, int(cz_ + R + 3) - oz + 1):
            dx, dz = i + ox + .5 - cx_, j + oz + .5 - cz_
            r_ = math.hypot(dx, dz); a_ = int((math.atan2(dz, dx) + math.pi)/(2*math.pi)*96) % 96
            lim = R + 0.7*noise[yi(y), a_]
            if r_ <= lim and vol[i, yi(y), j] != air:
                vol[i, yi(y), j] = air; carved += 1
            elif y < ground - 2 and (y - y_from) % 15 == 8 and lim < r_ <= lim + 1 and a_ % 24 == 3 and vol[i, yi(y), j] != air:
                vol[i, yi(y), j] = lamp; lamps += 1
# верх шахты: снять траву и прочее над краем уже сделано вырезанием до ground + 14
tide_x, tide_z = seed[0] + ox + .5, seed[1] + oz + .5
print('шахта: блоков', carved, 'фонарей', lamps, 'маркер прилива в столбе', (int(tide_x), int(tide_z)), 'вдоль', round(seed_lon, 1), 'поперёк', round(seed_lat, 1))
vol[int(tide_x) - ox, yi(LAVA_BASE + 1):yi(LAVA_PEAK) + 1, int(tide_z) - oz] = air

# ---------- проверки
def flood(y):
    start = (int(tide_x) - ox, int(tide_z) - oz); seen = {start}; st = [start]
    while st:
        i, j = st.pop()
        for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (i + a, j + b)
            if q in seen or not (0 <= q[0] < sx and 0 <= q[1] < sz): continue
            v = vol[q[0], yi(y), q[1]]
            if v == air or v == lava_id:
                seen.add(q); st.append(q)
                if len(seen) > 6000: return None
    return seen
leak = False
for y in range(LAVA_BASE + 1, LAVA_PEAK + 1):
    cells = flood(y)
    if cells is None: print('прилив течёт на y', y, ': слой больше 6000'); leak = True; continue
    land = sum(1 for i, j in cells if lon[i, j] < 22 or lon[i, j] > 58)
    print('прилив y', y, 'клеток', len(cells), 'вдоль', round(min(lon[i, j] for i, j in cells), 1), round(max(lon[i, j] for i, j in cells), 1), 'за пределами озера', land)
    if land: leak = True
    for p_ in pads + [final]:   # над столбом лава и должна быть, когда прилив выше него; в самом столбе нет
        for a, b in p_[4]:
            if y < p_[3] and (a - ox, b - oz) in cells: print('столб протёк', a, b, y); leak = True
# свод над столбами
for n, p_ in enumerate(pads + [final]):
    low = min(next((y for y in range(p_[3], Y1) if vol[a - ox, yi(y), b - oz] != air), Y1) - p_[3] for a, b in p_[4])
    if low < TOP_CLEAR: print('низко над столбом', n, low)
changed = vol != orig
print('изменено блоков', int(changed.sum()), 'бортик', int(rim.sum()), 'подъём', int(ramp.sum()), 'добавлено на подъёме', bumped)
if preview:
    from PIL import Image
    im = np.zeros((sz*4, sx*4, 3), np.uint8)
    for i in range(sx):
        for j in range(sz):
            yy = next((y for y in range(-40, -60, -1) if vol[i, yi(y), j] != air), None)
            if yy is None: col = (20, 20, 25)
            elif vol[i, yi(yy), j] == lava_id: col = (255, 110, 20)
            elif yy >= -44: col = (150, 190, 230)
            else: c_ = 60 + (yy + 58)*9; col = (c_, c_, c_ + 8)
            im[j*4:j*4 + 4, i*4:i*4 + 4] = col
    for p_ in pads:
        for a, b in p_[4]: im[(b - oz)*4:(b - oz)*4 + 4, (a - ox)*4:(a - ox)*4 + 4] = (240, 230, 90) if p_[3] >= -51 else (240, 110, 200)
    for a, b in final[4]: im[(b - oz)*4:(b - oz)*4 + 4, (a - ox)*4:(a - ox)*4 + 4] = (80, 240, 120)
    Image.fromarray(im[:, 0:]).crop(((110 - ox)*4, (-285 - oz)*4, (195 - ox)*4, (-232 - oz)*4)).save(preview)
if leak: raise SystemExit('лава протекает: в мир не пишу')
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
        if bs is None or not Y0 <= y*16 < Y1: continue
        oy = y*16 - Y0
        if not changed[a:a + 16, oy:oy + 16, b:b + 16].any(): continue
        arr = vol[a:a + 16, oy:oy + 16, b:b + 16].transpose(1, 2, 0).reshape(4096)
        u, inv = np.unique(arr, return_inverse=True)
        bs[1][b'palette'] = (9, (10, [mca.state(lst[i]) for i in u]))
        if len(u) == 1: bs[1].pop(b'data', None)
        else: bs[1][b'data'] = (12, mca.pack(inv, len(u)))
    c[b'isLightOn'] = (1, b'\0')
    # карты высот: шахта выходит на поверхность
    leaf = [i for i in range(len(lst)) if base(i).endswith('_leaves')]
    noblock = [i for i in range(len(lst)) if base(i) in ('air', 'vine', 'tall_grass', 'short_grass', 'fern', 'large_fern', 'cocoa', 'snow', 'dead_bush') or base(i).endswith('_spike')]
    fluid = [i for i in range(len(lst)) if base(i) in ('water', 'lava')]
    w_ = {t: np.zeros(math.ceil(256/7), np.uint64) for t in (0, 1, 2, 3)}
    for lx in range(16):
        for lz in range(16):
            col = vol[a + lx, :, b + lz]; n = col != 0; mb = ~np.isin(col, noblock); fl = np.isin(col, fluid)
            hs = []
            for m in (n, mb | fl, (mb & ~np.isin(col, leaf)) | fl, mb & ~fl):
                idx = np.nonzero(m)[0]; hs.append(int(idx[-1]) + 1 if len(idx) else 0)
            q_ = lz*16 + lx
            for t in (0, 1, 2, 3): w_[t][q_//7] |= np.uint64(hs[t] << ((q_ % 7)*9))
    for nmk, t in {b'WORLD_SURFACE': 0, b'MOTION_BLOCKING': 1, b'MOTION_BLOCKING_NO_LEAVES': 2, b'OCEAN_FLOOR': 3}.items():
        c[b'Heightmaps'][1][nmk] = (12, w_[t].astype('>u8').tobytes())
    regs[p].put(cx, cz, mca.dump(nm, c))
for r in regs.values(): r.save()
dv = next(iter(chunks.values()))[2][b'DataVersion']
dbl = lambda v: struct.pack('>d', v); flt = lambda v: (5, struct.pack('>f', float(v))); int_ = lambda v: (3, struct.pack('>i', int(v)))
eregs = {}
def add(entity, x, y, z):
    cx, cz = int(math.floor(x)) >> 4, int(math.floor(z)) >> 4; p = ENT + f'r.{cx >> 5}.{cz >> 5}.mca'
    if p not in eregs:
        if os.path.exists(p): shutil.copy2(p, os.path.join(backup, 'entities'))
        eregs[p] = mca.Region(p)
    b = eregs[p].raw(cx, cz)
    if b is None: nm, c = b'', {b'Position': (11, struct.pack('>ii', cx, cz)), b'DataVersion': dv, b'Entities': (9, (10, []))}
    else: nm, c = mca.parse(b)
    if c[b'Entities'][1][0] == 0: c[b'Entities'] = (9, (10, c[b'Entities'][1][1]))
    e = {b'Pos': (9, (6, [dbl(x), dbl(y), dbl(z)])), b'Motion': (9, (6, [dbl(0.0)]*3)), b'Rotation': (9, (5, [struct.pack('>f', 0.0)]*2)), b'UUID': (11, uuid.uuid4().bytes)}
    e.update(entity); c[b'Entities'][1][1].append(e); eregs[p].put(cx, cz, mca.dump(nm, c))
add({b'id': (8, b'gameplaymod:launcher'), b'Yaw': flt(YAW), b'Distance': flt(DIST), b'Height': flt(H)}, fx, final[3], fz)
if HOP: add({b'id': (8, b'gameplaymod:launcher'), b'Yaw': flt(HOP[3]), b'Distance': flt(HOP[4]), b'Height': flt(3)}, HOP[0], HOP[1], HOP[2])
add({b'id': (8, b'gameplaymod:lava_tide'), b'BaseY': int_(LAVA_BASE), b'PeakY': int_(LAVA_PEAK), b'HoldLow': int_(HOLD_LOW), b'StepTicks': int_(STEP_TICKS), b'HoldHigh': int_(HOLD_HIGH), b'Radius': flt(72)}, tide_x, LAVA_PEAK + 2, tide_z)
for r in eregs.values(): r.save()
print('записано чанков', touched, 'копия', backup)
