#!/usr/bin/env python3
"""Поперечная расщелина через лес: гигантские деревья с тарзанками и сужающийся тоннель со склонами трёх углов и сталактитами под гору.
Правит сохранённый мир напрямую (игра закрыта): chasm_world.py <мир> [--restore <папка с r.*.mca до правки>] [--preview файл.png]"""
import sys, os, math, struct, collections, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mca
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d, binary_dilation, distance_transform_edt

world = sys.argv[1]; D = os.path.join(world, 'dimensions/minecraft/overworld/'); REG = D + 'region/'; ENT = D + 'entities/'
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
preview, restore = arg('--preview'), arg('--restore')
X0, X1, Z0, Z1 = 96, 671, -416, -192          # рамка (границы чанков)
Y0, NY = -64, 384
rng = np.random.default_rng(1101)
cx0, cx1, cz0, cz1 = X0 >> 4, X1 >> 4, Z0 >> 4, Z1 >> 4
sx, sz = (cx1 - cx0 + 1)*16, (cz1 - cz0 + 1)*16; ox, oz = cx0*16, cz0*16
# ---------- уровень: игрок идёт с юга на север
XC = 500; RIM_S, RIM_N = -268, -308; FLOOR = 68; START_Y = 122; VEIL_Y = 104
# верёвки без площадок: воронка к входу: разброс 75 у старта, 51 (ширина входа) у последнего ряда; концы верёвок не выше стартового берега
ROPE_BANDS = [-275, -281, -287, -293, -301]; ROPE_COUNT = [7, 6, 6, 5, 5]; ROPE_Y0 = 135; BANK_Y = 122; SPREAD0, SPREAD1 = 37.5, 25.5
def spread(z): return SPREAD0 + (SPREAD1 - SPREAD0)*np.clip((ROPE_BANDS[0] - z)/(ROPE_BANDS[0] - ROPE_BANDS[-1]), 0, 1)
RIM_TOP_Z = -316                                               # южная грань первого блока верхнего края над входом
LEDGE_Y = 105                                                  # с последней верёвки игрок падает ко входу: верх склона под ногами y≈109
# тоннель: посадочный склон 22,5°, крутой спуск, 45°, дальше непрерывное поле высот (tunnel_field.py: уклон плавно с 22,5° на 11,25°), у конца обрыв на 30 и плато
import tunnel_field as TF
from tunnel_field import FLAT_N, STEEP_N, MID_N, Z_STEEP, Z_MID, Z_GENTLE, F_STEEP, F_MID, F_GENTLE, PLATEAU_Y, DROP, PLATEAU_LEN, TUNNEL, tc, ts, TLEN, LIP_S, F_LIP_T, S1, FS
PLATEAU_HALF = 14; POOL_LEN = 45                             # при скорости ×3 с обрыва выносит далеко: вода должна быть там, куда падают
# материалы трассы по длине: земля, камень, лёд; на границах блоки перемешаны, доля меняется плавно на TRANS блоках в каждую сторону
S_END = FLAT_N + STEEP_N + MID_N + LIP_S; M1, M2 = S_END/3, 2*S_END/3; TRANS = 18.0
HALF0, R1 = 25, 3.75; H0, H1 = 14, 5                         # 51×14 у входа -> 8×5 к концу склона
from scipy.spatial import cKDTree
NARROW = LIP_S
_chunks = {(int(a) >> 4, int(b) >> 4) for a, b in tc[::8]}
need = {(a + da, b + db) for a, b in _chunks for da in (-2, -1, 0, 1, 2) for db in (-2, -1, 0, 1, 2)}
# ---------- откат прежней расщелины
if restore:
    n = 0
    for f in sorted(os.listdir(restore)):
        a, b = mca.Region(REG + f), mca.Region(os.path.join(restore, f)); rx, rz = int(f.split('.')[1]), int(f.split('.')[2])
        for i in range(32):
            for j in range(32):
                cx, cz = rx*32 + i, rz*32 + j
                if not (-176 >> 4 <= cx <= 671 >> 4 and -416 >> 4 <= cz <= -160 >> 4): continue
                ra, rb = a.raw(cx, cz), b.raw(cx, cz)
                if rb is None or ra == rb: continue
                if cx*16 < 176 and not any(k in ra for k in (b'gameplaymod:slide', b'minecraft:blue_ice', b'minecraft:glowstone', b'minecraft:water')): continue   # западнее — только чанки прежнего тоннеля
                ca, cb = mca.parse(ra)[1], mca.parse(rb)[1]
                sa = [s.get(b'block_states') for s in ca[b'sections'][1][1]]; sb = [s.get(b'block_states') for s in cb[b'sections'][1][1]]
                if sa != sb: a.put(cx, cz, rb); n += 1
        a.save()
    print('восстановлено чанков из копии:', n)
# ---------- чтение
vol = np.zeros((sx, NY, sz), np.uint16); names = {'minecraft:air': 0}; lst = ['minecraft:air']; regs = {}; chunks = {}
def gid(k):
    if k not in names: names[k] = len(lst); lst.append(k)
    return names[k]
for cx in range(cx0, cx1 + 1):
    for cz in range(cz0, cz1 + 1):
        if cx*16 < 336 and (cx, cz) not in need: continue
        p = REG + f'r.{cx >> 5}.{cz >> 5}.mca'
        regs.setdefault(p, mca.Region(p)); b = regs[p].raw(cx, cz)
        if b is None: continue
        nm, c = mca.parse(b); chunks[(cx, cz)] = (p, nm, c)
        for sec in c[b'sections'][1][1]:
            y = int.from_bytes(sec[b'Y'][1], 'big', signed=True); bs = sec.get(b'block_states')
            if bs is None or not -4 <= y < 20: continue
            pal = np.array([gid(mca.key(q)) for q in bs[1][b'palette'][1][1]])
            ids = pal[mca.unpack(bs[1][b'data'][1], len(pal))].reshape(16, 16, 16) if b'data' in bs[1] else np.full((16, 16, 16), pal[0])
            vol[(cx - cx0)*16:(cx - cx0 + 1)*16, (y + 4)*16:(y + 5)*16, (cz - cz0)*16:(cz - cz0 + 1)*16] = ids.transpose(2, 0, 1)
orig = vol.copy(); A = 0
def bn(i): return lst[i].split('[')[0].replace('minecraft:', '')
def ids_of(*n): return [i for i in range(len(lst)) if bn(i) in n]
GROUND = ('stone', 'deepslate', 'dirt', 'andesite', 'grass_block', 'tuff', 'gravel', 'sand', 'snow_block', 'bedrock', 'calcite', 'ice')
isg = np.isin(vol, ids_of(*GROUND)); G = NY - 1 - np.argmax(isg[:, ::-1, :], axis=1) + Y0
natural = set(GROUND) | {'air', 'water', 'jungle_leaves', 'vine', 'jungle_log', 'bamboo', 'snow', 'tall_grass', 'short_grass', 'fern', 'large_fern', 'cocoa', 'melon', 'oak_leaves', 'oak_log', 'spruce_leaves', 'spruce_log', 'azure_bluet', 'oxeye_daisy', 'dead_bush', 'cactus'}
gx, gz = np.meshgrid(np.arange(sx) + ox, np.arange(sz) + oz, indexing='ij')
def n1(n, sig): a = gaussian_filter1d(rng.normal(size=n), sig, mode='wrap'); return a/a.std()
ss = lambda x: np.clip(x, 0, 1)**2*(3 - 2*np.clip(x, 0, 1))
# ---------- расщелина в плане: шире всего у тарзанок (41), к западу и востоку сужается до 3 блоков; глубже всего у концов
xs = (np.arange(sx) + ox).astype(float)
HW, HEND = 20.0, 1.5; KEEP = 70; XW_T, XE_T = 385.0, 608.0; XW_END, XE_END = 352.0, 660.0
uw = np.clip(((XC - KEEP) - xs)/((XC - KEEP) - XW_T), 0, 1); ue = np.clip((xs - (XC + KEEP))/(XE_T - (XC + KEEP)), 0, 1)
u = np.maximum(uw, ue)
hw = HEND + (HW - HEND)*(1 - ss(u))
zc = -288 + 40*ss(((XC - KEEP) - xs)/70.0) + 26*ss((xs - (XC + KEEP))/45.0)
amp = np.clip((hw - 3)/15.0, 0, 1)*np.where(np.abs(xs - XC) <= KEEP, 0, 1)
half = np.floor(hw); zmid = np.round(zc)                                     # в самом узком месте ровно 3 блока
zS = zmid + half - np.round(amp*np.abs(3*n1(sx, 14) + 1.5*n1(sx, 5))); zN = zmid - half + np.round(amp*np.abs(3*n1(sx, 14) + 1.5*n1(sx, 5)))     # неровности только внутрь: шире, чем у тарзанок, нигде нет
# над входом в тоннель верхний край отодвинут на север: склон идёт от него к уступу, дальше к дну, и не мешает последней верёвке
wMouth = 1 - ss((np.abs(xs - XC) - 40)/25.0); RIM_BACK = 8   # последний ряд верёвок (z=-305, длина 10): при полном размахе до края остаётся 1 блок, вход в тоннель ниже края
zNtop = zN - np.round(RIM_BACK*wMouth)
mask = (gz >= zNtop[gx - ox]) & (gz <= zS[gx - ox]) & (gx >= XW_END) & (gx <= XE_END)
fl = gaussian_filter(rng.normal(size=gx.shape), 6); fl /= fl.std()
FL_C, FL_E = 40.0, 10.0                                                     # дно: мельче у тарзанок, глубже у концов
floor = (FL_C + (FL_E - FL_C)*ss(u))[gx - ox] + 1.5*fl
wat2 = np.isin(vol, ids_of('water')).any(1)
dsea = distance_transform_edt(~wat2)
floor = np.maximum(floor, np.where(dsea < 12, 66.0, 66.0 - (dsea - 12)*3))     # у берега порог выше моря, чтобы вода не зашла
floor = np.round(floor).astype(int)
mask &= (G > floor + 2)
assert not (mask & (gz > -212)).any()
dIn = distance_transform_edt(mask)
# порог перемешивания материалов: пятна и зерно, равномерно от 0 до 1; свой генератор, чтобы не менять случайности остальной расщелины
_mr = np.random.default_rng(79); _d = gaussian_filter(_mr.normal(size=gx.shape), 2.0)/0.14 + 1.0*_mr.normal(size=gx.shape)
dth = np.empty(gx.shape); dth.ravel()[np.argsort(_d.ravel())] = np.linspace(0, 1, _d.size)
def track_material(s, i, j):
    """Материал ската на расстоянии s по трассе: суффикс имени блока."""
    return '_dirt' if dth[i, j] >= ss((s - M1 + TRANS)/(2*TRANS)) else '_stone' if dth[i, j] >= ss((s - M2 + TRANS)/(2*TRANS)) else ''
def track_field(st, s, i, j): return st.replace('slide_field[', 'slide_field' + track_material(s, i, j) + '[')
stone = gid('minecraft:stone'); andes = gid('minecraft:andesite'); tuff = gid('minecraft:tuff'); moss = gid('minecraft:moss_block'); grav = gid('minecraft:gravel'); glow = gid('minecraft:glowstone')
grass = gid('minecraft:grass_block[snowy=false]'); dirt = gid('minecraft:dirt'); planks = gid('minecraft:jungle_planks')
logY = gid('minecraft:jungle_log[axis=y]'); logX = gid('minecraft:jungle_log[axis=x]'); logZ = gid('minecraft:jungle_log[axis=z]')
leafS = gid('minecraft:jungle_leaves[distance=7,persistent=true,waterlogged=false]')
def stratum(y):
    band = y >> 2; k = (band*7 + (band >> 2)*5 + (band >> 3)*3) % 11
    return stone if k < 5 else andes if k < 9 else tuff
strata = np.array([stratum(y + Y0) for y in range(NY)], np.uint16)
G0 = G
# берега: от края полого к дну; внизу щель 3 блока вдоль всей расщелины
zb = zmid[gx - ox]
tS = np.clip((gz - zb - 1)/np.maximum(zS[gx - ox] - zb - 1, 1), 0, 1); tN = np.clip((zb - 1 - gz)/np.maximum(zb - 1 - zNtop[gx - ox], 1), 0, 1)
t = np.maximum(tS, tN)
bump = gaussian_filter(rng.normal(size=gx.shape), 3); bump /= bump.std()
newtop = floor + (G - floor)*(1 - (1 - t)**2) + 1.0*bump*(t > 0)*(t < 0.9)     # выпукло: у края полого, ниже всё круче
# северный берег у входа: край -> уступ (y LEDGE_Y на z=RIM_N-1) -> дно на 20 блоков южнее
Z_KNEE = RIM_N - 1
KNEE_Y = LEDGE_Y + 5
knee = np.where(gz <= Z_KNEE, KNEE_Y + (G - KNEE_Y)*np.clip((Z_KNEE - gz)/np.maximum(Z_KNEE - zNtop[gx - ox], 1), 0, 1), floor + (KNEE_Y - floor)*(1 - (1 - np.clip((Z_KNEE + 20 - gz)/20.0, 0, 1))**2))
w = wMouth[gx - ox]*(gz < zb)
newtop = np.round(newtop*(1 - w) + knee*w).astype(int)
newtop = np.where(mask, np.minimum(newtop, G), newtop)
# северный склон (промах мимо входа): весь из скатов на юг, стоять негде, игрок уезжает в расщелину. Высота в каждом столбце идёт только вниз, уклон — ближайший из четырёх к прежнему профилю; круче 2,5 — обрыв
BANK_STONE_Y = 104; BANK_X0, BANK_X1 = 385, 615; bank = mask & (gz < zb) & (gx >= BANK_X0) & (gx <= BANK_X1) & ~((np.abs(gx - XC) <= HALF0) & (gz <= RIM_N - 1))
BANK_START = F_STEEP + TF.RISE*FLAT_N                          # перед входом склон начинается с гребня: верх посадочного склона
bank_cells = {}
for i in np.where(bank.any(1))[0]:
    js = np.where(bank[i])[0]
    H_ = BANK_START if abs(i + ox - XC) <= HALF0 else float(G[i, js[0]] + 1)
    for j in js:
        if H_ < floor[i, j] + 4: break
        want = H_ - (newtop[i, min(j + 1, sz - 1)] + 1)
        kind = min(((TF.RISE2, 'shallow'), (TF.RISE, 'gentle'), (1.0, '45'), (2.5, 'steep')), key=lambda q: abs(q[0] - want))[1]
        if kind in ('shallow', 'gentle'):
            low = H_ - (TF.RISE2 if kind == 'shallow' else TF.RISE); y = int(math.floor(low + 1e-9)); p = int(round((low - y)*16))
            if p == 16: y += 1; p = 0
            bank_cells[(i, j)] = [(y, kind, p)]; H_ = y + p/16; newtop[i, j] = y
        elif kind == '45':
            top = int(math.floor(H_)); bank_cells[(i, j)] = [(top - 1, kind, 0)]; H_ = top - 1; newtop[i, j] = top - 1
        else:
            F = math.floor(H_*2)/2; y = int(math.floor(F - 1e-6)); st_ = []
            while (F - y - 1)/2.5 < 1 - 1e-6: st_.append((y, kind, int(round(((F - y - 1)/2.5 + 0.2)/0.2)))); y -= 1
            bank_cells[(i, j)] = st_; H_ = F - 2.5; newtop[i, j] = st_[0][0]
bank = np.zeros(gx.shape, bool)
for i, j in bank_cells: bank[i, j] = True
# убираются только деревья со стволом внутри расщелины; деревья на берегах остаются целиком, их кроны нависают
logs = np.isin(vol, ids_of('jungle_log', 'oak_log', 'spruce_log'))
trunk = np.zeros(gx.shape, bool)
for i, j in zip(*np.where(logs.any(1))):
    y = G0[i, j] - Y0 + 1
    if y < NY and logs[i, y, j]: trunk[i, j] = True
gone = binary_dilation(trunk & mask, iterations=2) & trunk
di, (ni, nj) = distance_transform_edt(~trunk, return_indices=True)
leafy = np.isin(vol, ids_of('jungle_leaves', 'oak_leaves', 'spruce_leaves')); loggy = np.isin(vol, ids_of('jungle_log', 'oak_log', 'spruce_log'))
plant = np.isin(vol, ids_of('bamboo', 'tall_grass', 'short_grass', 'fern', 'large_fern', 'cocoa', 'melon', 'snow', 'azure_bluet', 'oxeye_daisy', 'dead_bush', 'vine'))
for i, j in zip(*np.where(gone[ni, nj] & (di <= 8))):
    g = G0[i, j] - Y0; col = vol[i, g + 1:, j]; col[leafy[i, g + 1:, j] | loggy[i, g + 1:, j]] = A
bamboo = ids_of('bamboo'); low_plants = np.isin(vol, ids_of('tall_grass', 'short_grass', 'fern', 'large_fern', 'melon', 'snow', 'azure_bluet', 'oxeye_daisy', 'dead_bush', 'cocoa'))
for i, j in zip(*np.where(mask)):                                             # растения на вырезанной земле
    g = G0[i, j] - Y0
    vol[i, g + 1:g + 4, j][low_plants[i, g + 1:g + 4, j]] = A
    col = vol[i, g + 1:, j]; col[np.isin(col, bamboo)] = A
# лианы и какао без опоры
vn = ids_of('vine', 'cocoa'); east = vol[336 - ox:, :, :]
for _ in range(14):
    v = np.isin(east, vn); sup = (east != A) & ~v
    side = np.zeros_like(v); above = np.zeros_like(v)
    above[:, :-1, :] = v[:, 1:, :] | sup[:, 1:, :]
    side[1:, :, :] |= sup[:-1, :, :]; side[:-1, :, :] |= sup[1:, :, :]; side[:, :, 1:] |= sup[:, :, :-1]; side[:, :, :-1] |= sup[:, :, 1:]
    bad = v & ~(side | above)
    if not bad.any(): break
    east[bad] = A
# вырезать: снимается только земля; листва с берега остаётся висеть над расщелиной
grd = ids_of(*GROUND)
for i, j in zip(*np.where(mask)):
    tp = newtop[i, j]; ty = tp - Y0; col = vol[i, ty + 1:, j]
    col[np.isin(col, grd)] = A
    if tp <= floor[i, j] + 3: vol[i, ty, j] = grav if (i*7 + j*13) % 5 else moss
    elif t[i, j] > 0.55:                                                       # плечо: джунгли стекают по склону
        vol[i, ty, j] = grass; vol[i, max(ty - 3, 0):ty, j] = dirt
for (i, j), st_ in bank_cells.items():                                        # скаты северного склона: сверху земля, ниже камень
    if vol[i, st_[0][0] - Y0, j] == A: vol[i, max(st_[0][0] - Y0 - 3, 0):st_[0][0] - Y0, j] = dirt
    for y, kind, p in st_:
        mat = '_dirt' if dth[i, j] >= ss((BANK_STONE_Y - y)/16.0) else '_stone'
        vol[i, y - Y0, j] = gid({'shallow': f'gameplaymod:slide_shallow{mat}[facing=south,part={p}]', 'gentle': f'gameplaymod:slide_gentle{mat}[facing=south,part={p}]', '45': f'gameplaymod:slide{mat}[facing=south]', 'steep': f'gameplaymod:slide_steep{mat}[facing=south,part={p}]'}[kind])
# ---------- деревья
def ij(x, z): return x - ox, z - oz
def setblk(x, y, z, b):
    i, j = ij(x, z)
    if 0 <= i < sx and 0 <= j < sz: vol[i, y - Y0, j] = b
import random
def plan():
    """Случайная расстановка воронкой: верёвки (x, y, z, L), стволы (x, z, верх, радиус кроны, параметры изгиба), ветки. Отбрасывает варианты, где ветка или ствол задевают чужое."""
    R_ = random.Random(20261101)
    for attempt in range(4000):
        rs = []
        for k, zb in enumerate(ROPE_BANDS):
            n = ROPE_COUNT[k]; hw_ = float(spread(zb))
            for q in range(n):
                x = int(round(np.clip(XC + hw_*(2*q/(n - 1) - 1) + R_.randint(-2, 2), XC - hw_, XC + hw_)))
                if k == len(ROPE_BANDS) - 1:                                  # последний ряд: при полном размахе до края не хватает 1 блока
                    y = ROPE_Y0; l = R_.randint(y - BANK_Y, y - BANK_Y + 2); z = int(math.ceil(RIM_TOP_Z + l + 1.3 - 0.5))
                else:
                    y = ROPE_Y0 + R_.randint(-2, 2); l = R_.randint(y - BANK_Y, y - BANK_Y + 7); z = zb + R_.randint(-2, 2)
                rs.append((x, y, z, l))
        ts_ = []
        for _ in range(6000):
            if len(ts_) == 12: break
            z = R_.randint(-300, -272); h_ = float(spread(z)); x = int(round(XC + R_.uniform(-h_, h_)))
            if all(math.hypot(x - a[0], z - a[1]) >= 10 for a in ts_):
                ts_.append((x, z, R_.randint(152, 166), R_.uniform(6, 9), R_.uniform(1.8, 4.0), R_.uniform(0, 6.28), R_.uniform(1.8, 4.0), R_.uniform(0, 6.28), R_.uniform(0.7, 1.4)))
        if len(ts_) < 12: continue
        def centre(t_, y):
            u_ = (y - 40)/(t_[2] - 40.0)
            return (t_[0] + t_[4]*math.sin(t_[8]*3.0*u_ + t_[5]) - t_[4]*math.sin(t_[5]), t_[1] + t_[6]*math.sin(t_[8]*2.4*u_ + t_[7]) - t_[6]*math.sin(t_[7]))
        ok = True; cells = []; owner = {}
        for t_ in ts_:
            for rx, ry, rz, l in rs:
                for y in range(ry - l, ry):
                    cx_, cz_ = centre(t_, y)
                    if math.hypot(cx_ - rx, cz_ - rz) < 3.5: ok = False; break
                if not ok: break
            if not ok: break
        if not ok: continue
        for rx, ry, rz, l in rs:
            ti = min(range(12), key=lambda q: math.hypot(centre(ts_[q], ry)[0] - rx, centre(ts_[q], ry)[1] - rz))
            tx, tz = centre(ts_[ti], ry); n = max(int(abs(rx - tx)), int(abs(rz - tz)), 1); lg = logX if abs(rx - tx) >= abs(rz - tz) else logZ
            for q in range(n + 1):
                c = (int(round(tx + (rx - tx)*q/n)), ry, int(round(tz + (rz - tz)*q/n))); cells.append((c, lg, ti))
                for a in (-1, 0, 1):
                    for b in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            o2 = owner.get((c[0] + a, c[1] + dy, c[2] + b))
                            if o2 is not None and o2 != ti: ok = False
                owner[c] = ti
        if not ok: continue
        for c, lg, ti in cells:
            for rx, ry, rz, l in rs:
                if c[0] == rx and c[2] == rz and ry - l <= c[1] < ry: ok = False
            for tj, t2 in enumerate(ts_):
                if tj != ti:
                    cx_, cz_ = centre(t2, c[1])
                    if math.hypot(c[0] - cx_, c[2] - cz_) < 3.5: ok = False
        if ok: return rs, ts_, cells, centre
    raise SystemExit('расстановка не сложилась')
ropes, trunks, branch_cells, centre = plan()
def vine_state(d):
    return gid('minecraft:vine[east=%s,north=%s,south=%s,up=false,west=%s]' % tuple('true' if d == k else 'false' for k in ('east', 'north', 'south', 'west')))
VSIDE = [(1, 0, 'west'), (-1, 0, 'east'), (0, 1, 'north'), (0, -1, 'south')]               # клетка лианы и сторона, где её опора
for t_ in trunks:                                                             # изогнутые стволы со дна, толще внизу; между собой не связаны
    x0, z, top, cr = t_[:4]; i0, j0 = ij(x0, z); base = int(newtop[i0, j0]); flare = 0.0 if bank[i0, j0] else 1.6   # на скользком склоне без корней: их кольца — полки, на которых можно застрять
    for y in range(base - 2, top + 1):
        cx_, cz_ = centre(t_, y); u_ = (y - base)/max(top - base, 1)
        r_ = 2.3 - 1.3*u_ + (flare*(1 - (y - base)/6.0) if y < base + 6 else 0)
        ia, ja = int(round(cx_)) - ox, int(round(cz_)) - oz; ri = int(r_) + 2
        for i in range(ia - ri, ia + ri + 1):
            for j in range(ja - ri, ja + ri + 1):
                if math.hypot(i + ox - cx_, j + oz - cz_) <= r_ and y >= newtop[i, j]: vol[i, y - Y0, j] = logY
    cx_, cz_ = centre(t_, top); i0, j0 = int(round(cx_)) - ox, int(round(cz_)) - oz; ci = int(cr) + 1
    for i in range(i0 - ci, i0 + ci + 1):
        for j in range(j0 - ci, j0 + ci + 1):
            for y in range(top - 3, top + 5):
                e = ((i - i0)/cr)**2 + ((j - j0)/cr)**2 + ((y - top)/4.0)**2
                if e <= 1 and rng.random() > 0.08 and vol[i, y - Y0, j] == A: vol[i, y - Y0, j] = leafS
for (cx_, cy_, cz_), lg, _t in branch_cells:
    i, j = ij(cx_, cz_); vol[i, cy_ - Y0, j] = lg
for rx, ry, rz, l in ropes:
    i, j = ij(rx, rz)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if vol[i + a, ry + 1 - Y0, j + b] == A: vol[i + a, ry + 1 - Y0, j + b] = leafS
# лианы на стволах
tvr = np.random.default_rng(77)
for t_ in trunks:
    for y in range(int(newtop[t_[0] - ox, t_[1] - oz]) + 6, t_[2]):
        cx_, cz_ = centre(t_, y); ia, ja = int(round(cx_)) - ox, int(round(cz_)) - oz
        for i in range(ia - 4, ia + 5):
            for j in range(ja - 4, ja + 5):
                if vol[i, y - Y0, j] != A: continue
                for da, db, d in VSIDE:
                    if vol[i - da, y - Y0, j - db] == logY and tvr.random() < 0.10: vol[i, y - Y0, j] = vine_state(d); break
# джунгли на плече: новые деревья и подлесок, ниже — лианы по склону
sh = np.random.default_rng(78); keep_clear = [(r[0], r[2]) for r in ropes]
sg, sf = gid('minecraft:short_grass'), gid('minecraft:fern'); lv = leafS; newtrees = 0
for i, j in zip(*np.where(mask & ~bank & (t > 0.55) & (newtop > floor + 3))):
    x, z = i + ox, j + oz; y = newtop[i, j] - Y0; r0 = sh.random()
    if vol[i, y + 1, j] != A: continue
    near_rope = min(math.hypot(x - a, z - b) for a, b in keep_clear) < 6
    in_mouth = abs(x - XC) <= HALF0 + 3 and z < RIM_N + 6
    if r0 < 0.012 and t[i, j] > 0.72 and not near_rope and not in_mouth:
        h = int(sh.integers(6, 12))
        vol[i, y + 1:y + 1 + h, j] = logY
        for da in range(-3, 4):
            for db in range(-3, 4):
                for dy in range(-2, 2):
                    if da*da + db*db <= 9 - dy*dy and 0 <= i + da < sx and 0 <= j + db < sz and vol[i + da, y + h + dy, j + db] == A and sh.random() > 0.12: vol[i + da, y + h + dy, j + db] = lv
        newtrees += 1
    elif r0 < 0.13: vol[i, y + 1, j] = sg
    elif r0 < 0.19: vol[i, y + 1, j] = sf
for i, j in zip(*np.where(mask & ~bank & (t > 0.3) & (t <= 0.55))):
    if sh.random() < 0.05:
        y = newtop[i, j] - Y0 + 1
        for da, db, d in VSIDE:
            if 0 <= i - da < sx and 0 <= j - db < sz and newtop[i - da, j - db] > newtop[i, j] + 1:
                for k_ in range(int(sh.integers(1, 5))):
                    if vol[i, y - k_ + 2, j] == A: vol[i, y - k_ + 2, j] = vine_state(d)
                break
print('новых деревьев на плече', newtrees)
# столбы, где висят верёвки, свободны от листвы и лиан с берегов
for rx, ry, rz, l in ropes:
    i, j = ij(rx, rz); col = vol[i, ry - l - Y0:ry - Y0, j]
    col[np.isin(col, [leafS] + ids_of('jungle_leaves', 'oak_leaves', 'spruce_leaves', 'vine'))] = A
# ---------- тоннель
stg = lambda name, f, p: gid(f'gameplaymod:{name}[facing={f},part={p}]')
s45 = gid('gameplaymod:slide_dirt[facing=north]')
tun = np.zeros(gx.shape, bool); floorT = np.zeros(gx.shape, int); topT = np.zeros(gx.shape, int); pathS = np.zeros(gx.shape, float); kinds = collections.Counter()
def hgt(s): return H0 + (H1 - H0)*ss(s/NARROW)
def carve(i, j, solid_y, top_surface, s, h):
    """воздух от блока над solid_y до top_surface + h"""
    a0 = solid_y - Y0 + 1; a1 = int(math.ceil(top_surface)) - Y0 + h
    vol[i, a0:a1, j] = A; tun[i, j] = True; floorT[i, j] = solid_y; topT[i, j] = a1 - 1 + Y0; pathS[i, j] = s
def roof(s, o, r): return max(3, int(round(hgt(s)*math.sqrt(max(0, 1 - (o/(r + 0.5))**2)))))
_L = TF.landing(XC - HALF0, XC + HALF0, RIM_N - FLAT_N, RIM_N - 1)
for k in range(FLAT_N):                                                       # склон 22,5° там, куда падают с последней верёвки
    z = RIM_N - 1 - k
    for o in range(-HALF0, HALF0 + 1):
        q = _L[o + HALF0:o + HALF0 + 2, z - (RIM_N - FLAT_N):z - (RIM_N - FLAT_N) + 2]; y, st, _c = TF.cell_state(q[0, 0], q[1, 0], q[0, 1], q[1, 1])
        i, j = ij(XC + o, z); carve(i, j, y, y + 1, k, roof(k, o, HALF0)); vol[i, y - Y0, j] = gid(track_field(st, k, i, j)); kinds['22'] += 1
for k in range(STEEP_N):                                                      # 68°: 2,5 блока вниз на блок пути
    z = Z_STEEP - k; F = F_STEEP - 2.5*k; s = FLAT_N + k
    for o in range(-HALF0, HALF0 + 1):
        i, j = ij(XC + o, z); y = int(math.floor(F - 1e-6))
        carve(i, j, y, F, s, roof(s, o, HALF0))
        while True:
            a = (F - y - 1)/2.5
            if a >= 1 - 1e-6: break
            vol[i, y - Y0, j] = stg('slide_steep_dirt', 'north', int(round((a + 0.2)/0.2))); kinds['68'] += 1; floorT[i, j] = y; y -= 1
for k in range(MID_N):                                                        # 45°
    z = Z_MID - k; y = int(F_MID) - 1 - k; s = FLAT_N + STEEP_N + k
    for o in range(-HALF0, HALF0 + 1):
        i, j = ij(XC + o, z); carve(i, j, y, y + 1, s, roof(s, o, HALF0)); vol[i, y - Y0, j] = s45; kinds['45'] += 1
S0 = FLAT_N + STEEP_N + MID_N
dT, iT = cKDTree(tc).query(np.c_[gx.ravel(), gz.ravel()]); dT = dT.reshape(gx.shape); iT = iT.reshape(gx.shape); sG = ts[iT]
rG = HALF0 + 0.5 + (R1 - HALF0 - 0.5)*ss(sG/NARROW)
slope = (dT <= rG) & (sG <= LIP_S) & ((gz < Z_GENTLE + 1) | (sG > 30)) & ~tun
assert not (slope & mask).any()
_ii, _jj = np.where(slope); _xa, _xb, _za, _zb = int(_ii.min()) + ox, int(_ii.max()) + ox, int(_jj.min()) + oz, int(_jj.max()) + oz
_H = TF.corners(_xa, _xb, _za, _zb, [(int(a) + ox, int(b) + oz) for a, b in zip(_ii, _jj)])
for i, j in zip(_ii, _jj):                                                    # соседние клетки делят углы: ступеней нет
    a, b = i + ox - _xa, j + oz - _za; y, st, _c = TF.cell_state(_H[a, b], _H[a + 1, b], _H[a, b + 1], _H[a + 1, b + 1])
    s = S0 + sG[i, j]
    carve(i, j, y, y + 1, s, roof(s, dT[i, j], rG[i, j]) + 1); vol[i, y - Y0, j] = gid(track_field(st, s, i, j)); kinds['поле'] += 1
# обрыв на 30 и плато под озеро: шире коридора, пол ровный, свод выше края
F_LIP = int(round(FS(LIP_S)))
rP = R1 + (PLATEAU_HALF - R1)*ss((sG - LIP_S)/12.0)
pl = (dT <= rP) & (sG > LIP_S) & ~tun
assert not (pl & mask).any()
water = gid('minecraft:water[level=0]')
for i, j in zip(*np.where(pl)):
    y = PLATEAU_Y; vol[i, y - Y0 + 1:F_LIP - Y0 + 8, j] = A; vol[i, y - Y0, j] = strata[y - Y0]
    if sG[i, j] < LIP_S + POOL_LEN and dT[i, j] < rP[i, j] - 1: vol[i, y - Y0, j] = water     # 30 блоков падения смертельны: под обрывом вода в один блок
    tun[i, j] = True; floorT[i, j] = y; topT[i, j] = F_LIP + 7; pathS[i, j] = 0
gaps = [k for k in range(0, len(tc) - 8, 8) if not tun[int(round(tc[k][0])) - ox, int(round(tc[k][1])) - oz]]
assert not gaps, f'тоннель прерывается: {[tc[k].round().tolist() for k in gaps[:5]]}'
# светокамень в своде; ледяные колья торчат из склона: 1–3 блока, над кончиком остаётся проход
gl = rng.random(gx.shape); sp_b = gid('gameplaymod:ice_spike[tip=false]'); sp_t = gid('gameplaymod:ice_spike[tip=true]')
cave = np.zeros(vol.shape, bool); nstal = 0
ice_w = ss((S0 + sG - M2 + TRANS)/(2*TRANS)); lantern = gid('minecraft:sea_lantern')   # доля льда по трассе; за обрывом 1
for i, j in zip(*np.where(tun)):
    y1 = topT[i, j] - Y0; y0 = floorT[i, j] - Y0
    cave[i, y0 + 1:y1 + 1, j] = True
    if gl[i, j] < 0.02: vol[i, y1 + 1, j] = glow if dth[i, j] >= ice_w[i, j] else lantern
    elif gl[i, j] < 0.035 and pathS[i, j] > S0 + 6:
        n = min(1 + int(gl[i, j]*2000) % 3, y1 - y0 - 3)
        if n >= 1:
            for q in range(n): vol[i, y0 + 1 + q, j] = sp_t if q == n - 1 else sp_b
            nstal += 1
heightT = topT - floorT
end = tc[-1].round().astype(int)
lip = tc[np.searchsorted(ts, LIP_S)].round().astype(int)
print('край обрыва', tuple(lip.tolist()), 'y', F_LIP, '-> плато y', PLATEAU_Y, 'конец', tuple(end.tolist()), 'длина', round(S0 + TLEN), 'блоков склона', dict(kinds), 'кольев', nstal)
for xx in (355, 370, 390, 420, 450, 500, 550, 580, 600, 608, 612, 616, 620):
    col = np.where(mask[xx - ox])[0]
    print('  x', xx, 'ширина', len(col), 'дно', int(newtop[xx - ox, col].min()) if len(col) else None)
print('верёвок', len(ropes), 'деревьев', len(trunks), 'тоннель клеток', int(tun.sum()), 'переход 22,5->11,25 на s', round(S1))
# ---------- завеса: выше неё расщелина освещена, ниже темно
veil = gid('gameplaymod:dark_veil'); nv = 0
for i, j in zip(*np.where(mask)):
    if vol[i, VEIL_Y - Y0, j] == A and newtop[i, j] < VEIL_Y: vol[i, VEIL_Y - Y0, j] = veil; nv += 1
print('завеса: клеток', nv)
# ---------- слои породы на открывшихся стенах
newair = (vol == A) & (orig != A) & np.isin(orig, ids_of('stone', 'deepslate', 'dirt', 'andesite', 'tuff', 'gravel', 'grass_block'))
expo = binary_dilation(newair | cave, iterations=2)
plain = np.isin(vol, ids_of('stone', 'deepslate')) & expo
ys = np.arange(NY)
for i, j in zip(*np.where(plain.any(1))):
    k = plain[i, :, j]; vol[i, k, j] = strata[ys[k]]
# ---------- ледяная часть: стены, свод, пол и вся финальная локация; на границе порода и лёд перемешаны
packed, blue = gid('minecraft:packed_ice'), gid('minecraft:blue_ice'); rock = np.isin(vol, ids_of('stone', 'deepslate', 'andesite', 'tuff', 'dirt', 'gravel', 'grass_block', 'calcite', 'moss_block'))
shell = binary_dilation(cave, iterations=2) & rock; ir = np.random.default_rng(80); nice = 0
for i, y, j in np.argwhere(shell):
    if ice_w[i, j] > 0 and ir.random() < ice_w[i, j]: vol[i, y, j] = blue if ir.random() < 0.25 else packed; nice += 1
# свод финальной локации в 38 блоках над полом и свет от него не доходит: морские фонари в стенах над полом, иначе лёд не видно
near = np.zeros(cave.shape, bool); near[1:] |= cave[:-1]; near[:-1] |= cave[1:]; near[:, :, 1:] |= cave[:, :, :-1]; near[:, :, :-1] |= cave[:, :, 1:]
lamps = 0; final = (sG > LIP_S) & (dT <= rP + 2)
for i, y, j in np.argwhere(near & np.isin(vol, [packed, blue])):
    if final[i, j] and PLATEAU_Y + 3 <= y + Y0 <= PLATEAU_Y + 12 and ir.random() < 0.06: vol[i, y, j] = lantern; lamps += 1
print('блоков льда в стенах', nice, 'фонарей в финальной локации', lamps)
# ---------- проверки
changed = vol != orig
print('изменено блоков', int(changed.sum()), 'построек затронуто:', int(sum((orig[changed] == i).sum() for i in range(len(lst)) if bn(i) not in natural)))
nearw = binary_dilation((vol == A) & (orig != A), iterations=3) & np.isin(orig, ids_of('water'))
print('воды рядом с новым воздухом:', int(nearw.sum()))
loose = np.isin(vol, ids_of('gravel', 'sand')); below = np.zeros_like(loose); below[:, 1:, :] = vol[:, :-1, :] == A
print('незакреплённых песка/гравия:', int((loose & below).sum()))
under = tun; cover = (G - topT)[under]; print('порода над тоннелем: мин', int(cover.min()), 'сред', int(cover.mean()))
for (rx, ry, rz, l) in ropes:
    i, j = ij(rx, rz); assert bn(vol[i, ry - Y0, j]) == 'jungle_log'; assert np.isin(vol[i, ry - Y0 - l:ry - Y0, j], [A, veil]).all(), (rx, ry, rz)
if nearw.any():
    w = np.argwhere(nearw)[:6]; sys.exit('вода рядом, останов, мир не изменён: ' + str([(a + ox, b + Y0, c + oz) for a, b, c in w]))
if preview:
    surf = NY - 1 - np.argmax(vol[:, ::-1, :] != A, axis=1)
    img = np.clip((surf - 40)*1.3, 0, 255).astype(np.uint8); rgb = np.dstack([img]*3)
    rgb[mask & (newtop <= FLOOR + 6)] = (120, 60, 40); rgb[tun] = (240, 120, 30)
    rgb[isg.any(1) & (surf > NY)] = 0
    for (rx, ry, rz, l) in ropes: i, j = ij(rx, rz); rgb[i - 1:i + 2, j - 1:j + 2] = (255, 40, 40)
    Image.fromarray(rgb.transpose(1, 0, 2)).resize((sx*3, sz*3), Image.NEAREST).save(preview)
    i = XC - ox; sec = np.where((vol[i] != A)[..., None], (110, 110, 110), (190, 220, 245)).astype(np.uint8)
    sec = sec[:, ::-1, :].transpose(1, 0, 2)[120:230]
    Image.fromarray(sec).resize((sec.shape[1]*4, sec.shape[0]*4), Image.NEAREST).save(preview.replace('.png', '_разрез.png'))
    print('разрез x=500, z: верх земли', [(z, int(newtop[XC - ox, z - oz])) for z in (-262, -268, -272, -280, -287, -288, -289, -297, -305, -309, -316, -322)])
# ---------- запись блоков
leaf = ids_of('jungle_leaves', 'oak_leaves', 'spruce_leaves')
noblock = ids_of('air', 'dark_veil', 'ice_spike', 'vine', 'tall_grass', 'short_grass', 'fern', 'large_fern', 'cocoa', 'snow', 'azure_bluet', 'oxeye_daisy', 'dead_bush') + [A]
fluid = ids_of('water')
TYPES = {b'WORLD_SURFACE': 0, b'MOTION_BLOCKING': 1, b'MOTION_BLOCKING_NO_LEAVES': 2, b'OCEAN_FLOOR': 3}
def hm(col):
    r = []; n = col != A; mb = ~np.isin(col, noblock); water = np.isin(col, fluid)
    for m in (n, mb | water, (mb & ~np.isin(col, leaf)) | water, mb & ~water):
        idx = np.nonzero(m)[0]; r.append(int(idx[-1]) + 1 if len(idx) else 0)
    return r
touched = 0
for (cx, cz), (p, nm, c) in chunks.items():
    a, b = (cx - cx0)*16, (cz - cz0)*16
    if not changed[a:a + 16, :, b:b + 16].any(): continue
    touched += 1
    for sec in c[b'sections'][1][1]:
        y = int.from_bytes(sec[b'Y'][1], 'big', signed=True); bs = sec.get(b'block_states')
        sec.pop(b'SkyLight', None); sec.pop(b'BlockLight', None)
        if bs is None or not -4 <= y < 20: continue
        oy = (y + 4)*16
        if not changed[a:a + 16, oy:oy + 16, b:b + 16].any(): continue
        arr = vol[a:a + 16, oy:oy + 16, b:b + 16].transpose(1, 2, 0).reshape(4096)
        u, inv = np.unique(arr, return_inverse=True)
        bs[1][b'palette'] = (9, (10, [mca.state(lst[i]) for i in u]))
        if len(u) == 1: bs[1].pop(b'data', None)
        else: bs[1][b'data'] = (12, mca.pack(inv, len(u)))
    c[b'isLightOn'] = (1, b'\0')
    hs = np.zeros((4, 256), int)
    for lx in range(16):
        for lz in range(16): hs[:, lz*16 + lx] = hm(vol[a + lx, :, b + lz])
    per = 7
    for nmk, t in TYPES.items():
        w = np.zeros(math.ceil(256/per), np.uint64)
        for q in range(256): w[q//per] |= np.uint64(int(hs[t, q]) << ((q % per)*9))
        c[b'Heightmaps'][1][nmk] = (12, w.astype('>u8').tobytes())
    regs[p].put(cx, cz, mca.dump(nm, c))
for r in regs.values(): r.save()
# ---------- верёвки (сущности gameplaymod:tarzan_rope, свободно висят)
eregs = {}
dv = next(iter(chunks.values()))[2][b'DataVersion']
removed = 0                                                                  # прежние верёвки этой расщелины
for cx in range(336 >> 4, cx1 + 1):
    for cz in range(cz0, cz1 + 1):
        p = ENT + f'r.{cx >> 5}.{cz >> 5}.mca'
        if not os.path.exists(p): continue
        eregs.setdefault(p, mca.Region(p)); b = eregs[p].raw(cx, cz)
        if b is None: continue
        nm, c = mca.parse(b); li = c[b'Entities'][1][1]; kept = [e for e in li if e[b'id'][1] != b'gameplaymod:tarzan_rope']
        if len(kept) != len(li):
            removed += len(li) - len(kept); c[b'Entities'] = (9, (10, kept)); eregs[p].put(cx, cz, mca.dump(nm, c))
print('прежних верёвок убрано', removed)
dbl = lambda v: struct.pack('>d', v)
for rx, ry, rz, L in ropes:
    ax, ay, az = rx + 0.5, ry - 0.05, rz + 0.5
    cx, cz = rx >> 4, rz >> 4; p = ENT + f'r.{cx >> 5}.{cz >> 5}.mca'
    if p not in eregs: eregs[p] = mca.Region(p)
    b = eregs[p].raw(cx, cz)
    if b is None:
        nm, c = b'', {b'Position': (11, struct.pack('>ii', cx, cz)), b'DataVersion': dv, b'Entities': (9, (10, []))}
    else: nm, c = mca.parse(b)
    if c[b'Entities'][1][0] == 0: c[b'Entities'] = (9, (10, c[b'Entities'][1][1]))
    u = uuid.uuid4().bytes
    c[b'Entities'][1][1].append({b'id': (8, b'gameplaymod:tarzan_rope'), b'Pos': (9, (6, [dbl(ax), dbl(ay - L), dbl(az)])), b'Motion': (9, (6, [dbl(0.0)]*3)),
        b'Rotation': (9, (5, [struct.pack('>f', 0.0)]*2)), b'UUID': (11, u),
        b'AX': (6, dbl(ax)), b'AY': (6, dbl(ay)), b'AZ': (6, dbl(az)), b'Length': (5, struct.pack('>f', float(L))),
        b'BX': (3, struct.pack('>i', rx)), b'BY': (3, struct.pack('>i', ry)), b'BZ': (3, struct.pack('>i', rz)), b'Attached': (3, struct.pack('>i', 0))})
    eregs[p].put(cx, cz, mca.dump(nm, c))
for r in eregs.values(): r.save()
print('чанков записано', touched, 'верёвок записано', len(ropes))
last = ropes[-5:]; print('последний ряд: зазор до края', [round(rz + 0.5 - RIM_TOP_Z - (l + 0.3), 1) for rx, ry, rz, l in last], 'длины', sorted({r[3] for r in ropes}), 'высоты', sorted({r[1] for r in ropes}), 'стволы', [(q[0], q[1]) for q in trunks], 'концы верёвок y', min(r[1] - r[3] for r in ropes), '..', max(r[1] - r[3] for r in ropes), 'разброс по рядам', [(min(r[0] for r in ropes if abs(r[2] - b) <= 3), max(r[0] for r in ropes if abs(r[2] - b) <= 3)) for b in ROPE_BANDS])
