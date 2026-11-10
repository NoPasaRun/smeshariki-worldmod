#!/usr/bin/env python3
"""Колья и колонны в тоннеле расщелины: снимает все колья в рамке и ставит группы кольев и редкие колонны от пола до свода (GameplayMod, SpikeBlock, ColumnBlock).

Запуск: tunnel_spikes.py <папка мира> [--preview файл.png] [--write <новая папка копии>]. Без --write мир не меняется. Игра должна быть закрыта.
После пересборки chasm_world.py запускать заново: тот ставит прежние пары блоков. Колонна занимает клетку пола, поэтому в мире с колоннами повторный запуск отказывается: сначала вернуть копию.

Вид: кол по материалу пола под ним (лёд на льду, камень на земле и камне), размер по своду, группы — большой кол и мелкие вокруг, у стен чаще.
Сложность: всюду есть проход — плавно виляющая линия с просветом, который сужается от земли ко льду. Виляет так, чтобы хватало подруливания.
"""
import math, os, pickle, shutil, sys
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca, tunnel_field as TF

SEED = 20261102
X0, X1, Z0, Z1 = 96, 671, -416, -192
Y0, NY = -64, 384
S_FROM, S_TO = 18.0, TF.LIP_S - 22   # посадка и разгон свободны, перед обрывом тоже: туда надо успеть вырулить
M1, M2 = 127.0, 254.0                # границы материалов трассы, как S_END/3 и 2·S_END/3 в chasm_world.py
SIZES = (('small_', 2, 0.25), ('', 6, 0.75), ('big_', 10, 1.25))  # приставка блока, высота, радиус (SpikeBlock)
PLAYER = 0.3                         # половина лежащего игрока
STEER = 0.036                        # SlopeSurface.STEER, блок/тик²
SPEED = 1.2                          # блок/тик: вся трасса проходится примерно за 1 блок/тик, с запасом
TURN = 0.5*STEER/SPEED**2            # линия требует не больше половины подруливания

def zone(s):
    """Просвет линии (половина, без игрока) и шаг групп вдоль тоннеля: от земли ко льду сложнее."""
    if s < M1: return 2.6, (14, 22)
    if s < M2: return 2.0, (11, 17)
    return 1.3, (8, 13)

world = sys.argv[1]
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
backup, preview = arg('--write'), arg('--preview')
REG = os.path.join(world, 'dimensions/minecraft/overworld/region/')
cx0, cx1, cz0, cz1 = X0 >> 4, X1 >> 4, Z0 >> 4, Z1 >> 4
sx, sz = (cx1 - cx0 + 1)*16, (cz1 - cz0 + 1)*16; ox, oz = cx0*16, cz0*16
vol = np.zeros((sx, NY, sz), np.uint16); names = {'minecraft:air': 0}; lst = ['minecraft:air']; regs = {}; chunks = {}
def gid(k):
    if k not in names: names[k] = len(lst); lst.append(k)
    return names[k]
for cx in range(cx0, cx1 + 1):
    for cz in range(cz0, cz1 + 1):
        p = REG + f'r.{cx >> 5}.{cz >> 5}.mca'
        if not os.path.exists(p): continue
        regs.setdefault(p, mca.Region(p)); b = regs[p].raw(cx, cz)
        if b is None: continue
        nm, c = mca.parse(b); chunks[(cx, cz)] = (p, nm, c)
        for sec in c[b'sections'][1][1]:
            y = int.from_bytes(sec[b'Y'][1], 'big', signed=True); bs = sec.get(b'block_states')
            if bs is None or not -4 <= y < 20: continue
            pal = np.array([gid(mca.key(q)) for q in bs[1][b'palette'][1][1]])
            ids = pal[mca.unpack(bs[1][b'data'][1], len(pal))].reshape(16, 16, 16) if b'data' in bs[1] else np.full((16, 16, 16), pal[0])
            vol[(cx - cx0)*16:(cx - cx0 + 1)*16, (y + 4)*16:(y + 5)*16, (cz - cz0)*16:(cz - cz0 + 1)*16] = ids.transpose(2, 0, 1)
orig = vol.copy()
base = [l.split('[')[0].split(':')[1] for l in lst]
OLD = [i for i, b in enumerate(base) if b.endswith('_spike')]
assert not any(b.endswith('_column') and (vol == i).any() for i, b in enumerate(base)), 'в мире уже есть колонны: колонна заняла клетку пола, сначала верните копию мира'
slide = np.array(['slide_field' in b for b in base])

# ---------- прежние колья
removed = int(np.isin(vol, OLD).sum())
vol[np.isin(vol, OLD)] = 0
air = np.array([b in ('air', 'dark_veil') for b in base])

# ---------- пол и свод: клетки поля под сводом (склоны у неба в расщелине не считаются)
tree = cKDTree(TF.tc)
tan = np.gradient(TF.tc, axis=0); tan /= np.hypot(*tan.T)[:, None]
sl = slide[vol]; has = sl.any(1); top = NY - 1 - np.argmax(sl[:, ::-1, :], axis=1)
cells = {}
for i, j in zip(*np.nonzero(has)):
    y = top[i, j]; h = 0
    while y + 1 + h < NY and air[vol[i, y + 1 + h, j]]: h += 1
    if not 3 <= h <= 16 or y + 1 + h >= NY: continue
    x, z = i + ox + .5, j + oz + .5
    d, k = tree.query((x, z))
    lat = (x - TF.tc[k, 0])*-tan[k, 1] + (z - TF.tc[k, 1])*tan[k, 0]
    cells[(int(i), int(j))] = (TF.ts[k], lat, y, h)
arr = np.array([v[:2] for v in cells.values()])
grid = np.arange(0, TF.LIP_S + 1, 1.0)
lo, hi = [], []
for g in grid:
    m = np.abs(arr[:, 0] - g) < 2
    lo.append(arr[m, 1].min() if m.any() else np.nan); hi.append(arr[m, 1].max() if m.any() else np.nan)
lo, hi = np.array(lo), np.array(hi)
ok = ~np.isnan(lo); lo = np.interp(grid, grid[ok], lo[ok]); hi = np.interp(grid, grid[ok], hi[ok])
lo = gaussian_filter1d(lo, 4); hi = gaussian_filter1d(hi, 4)

# ---------- линия прохода: размах от ширины, длина волны такая, чтобы хватало подруливания
rng = np.random.default_rng(SEED)
mid = gaussian_filter1d((lo + hi)/2, 12); half = (hi - lo)/2   # края неровные, без сглаживания линия дрожала бы
amp = np.clip(0.55*(half - 1.5), 0.3, 7)
in_run = (grid >= S_FROM) & (grid <= S_TO)              # по краям коридора поле неровное, а колья там не стоят
ph0, ph1 = rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi)
for stretch in np.arange(1.2, 4, 0.05):                # волна длиннее, пока кривизна не войдёт в предел подруливания
    wave = 2*np.pi*np.sqrt(stretch*amp/TURN)
    phase = np.cumsum(2*np.pi/wave) + ph0
    line = mid + amp*(0.85*np.sin(phase) + 0.15*np.sin(1.7*phase + ph1))   # вторая гармоника: волна неровная, как в настоящей пещере
    line = gaussian_filter1d(line, 3)
    if np.abs(np.gradient(np.gradient(line))[in_run]).max() <= TURN: break
print(f'растяжение волны {stretch:.2f}')
def line_at(s): return np.interp(s, grid, line)
def clear(s): return zone(s)[0]
print(f'линия: наибольшая кривизна {np.abs(np.gradient(np.gradient(line))[in_run]).max():.4f}, предел {TURN:.4f} блок⁻¹; размах линии ±{amp[in_run].max():.1f}..{amp[in_run].min():.1f}, волна {wave[in_run].min():.0f}–{wave[in_run].max():.0f}')

# ---------- группы
placed = []  # i, j, y, приставка, радиус, высота, s, lat
cell_keys = np.array(list(cells.keys())); cell_tree = cKDTree(np.c_[arr[:, 0], arr[:, 1]])
def fits(key, size):
    if key not in cells: return False
    s, lat, y, h = cells[key]; _, height, r = size
    if not S_FROM <= s <= S_TO or height + 1 > h: return False                             # кончик под сводом
    if abs(lat - line_at(s)) < clear(s) + PLAYER + r: return False                         # проход
    if lat - r < np.interp(s, grid, lo) - 0.5 or lat + r > np.interp(s, grid, hi) + 0.5: return False   # не в стену
    for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        o = cells.get((key[0] + a, key[1] + b))
        if o and abs(o[2] - y) > 1: return False                                           # основание не повиснет
    for p in placed:
        if math.hypot(key[0] - p[0], key[1] - p[1]) < max(r + p[4], 1.5): return False     # не сливаются
    return True
def put(key, size):
    s, lat, y, h = cells[key]; placed.append((key[0], key[1], y, size[0], size[2], size[1], s, lat))
def near(s, lat):
    d, k = cell_tree.query((s, lat)); return tuple(cell_keys[k]) if d < 1.5 else None
def group(s, lat, sizes, extra):
    """Большой кол и мелкие вокруг него. Возвращает, получилось ли."""
    for size in sizes:
        key = near(s, lat)
        if key is None or not fits(key, size): continue
        put(key, size); cs, cl = cells[key][:2]
        left = extra
        for _ in range(extra*5):
            if left == 0: break
            small = SIZES[0] if size is not SIZES[2] or rng.random() < 0.6 else SIZES[1]
            d = size[2] + small[2] + rng.uniform(0.6, 3.0); a = rng.uniform(0, 2*np.pi)
            k = near(cs + d*math.cos(a), cl + d*math.sin(a))
            if k is not None and fits(k, small): put(k, small); left -= 1
        return True
    return False
def beside(s, side, r):
    """Положение кола радиуса r сбоку от линии: вплотную к просвету, с небольшим разбросом."""
    room = (np.interp(s, grid, hi) if side > 0 else -np.interp(s, grid, lo)) - side*line_at(s)
    need = clear(s) + PLAYER + r
    if room < need + r: return None
    return line_at(s) + side*(need + rng.uniform(0.1, min(2.5, room - need - r + 0.1)))

# заросли у стен: на линию не влияют, это вид; большие там, где свод позволяет
fringe = 0
for wall, inward in ((lo, 1), (hi, -1)):
    s = S_FROM + rng.uniform(0, 5)
    while s < S_TO:
        w = float(np.interp(s, grid, wall)); width = float(np.interp(s, grid, half))
        lat = w + inward*rng.uniform(0.8, 1.5 + 0.22*width)
        if group(s, lat, [SIZES[1], SIZES[0]], int(rng.integers(0, 3))): fringe += 1
        s += rng.uniform(8, 16) if width > 6 else rng.uniform(10, 20)

# великаны: только в зале, где свод выше 11, в стороне от линии
giants = 0
s = S_FROM + 4
while s < M1 + 40:
    room_l = line_at(s) - np.interp(s, grid, lo); room_r = np.interp(s, grid, hi) - line_at(s)
    sd = 1 if room_r >= room_l else -1
    for off in (5.0, 4.0, 3.0):
        lat = line_at(s) + sd*(clear(s) + PLAYER + SIZES[2][2] + off)
        if group(s, lat, [SIZES[2]], int(rng.integers(2, 5))): giants += 1; break
    s += rng.uniform(16, 26)

# препятствия у линии: одна сторона, чаще та, куда линия только что ушла, или ворота с двух сторон
s = S_FROM; side = 1; rest = 70.0; groups = gates = 0
while s < S_TO:
    _, (g0, g1) = zone(s)
    if s > rest:                                       # передышка: на долгих участках пусто, можно выровняться
        s += 10; rest = s + rng.uniform(60, 90)
    room_l = line_at(s) - np.interp(s, grid, lo); room_r = np.interp(s, grid, hi) - line_at(s)
    side = (1 if room_r >= room_l else -1) if rng.random() < 0.7 else -side
    made = False
    for sd in ((side, -side) if s >= M1 and rng.random() < 0.45 else (side,)):
        for size in (SIZES[1], SIZES[0]):
            lat = beside(s, sd, size[2])
            if lat is None: continue
            if group(s + rng.uniform(-1.5, 1.5), lat, [size], int(rng.integers(1, 4))):
                made = True; groups += 1; break
    gates += made and s >= M1
    s += rng.uniform(g0, g1)

# без руля по оси тоннеля не больше UNSTEERED кольев: с 10 сердцами и 6 урона за кол выживает, но на последнем сердце
UNSTEERED = 3
mid_at = lambda g: float(np.interp(g, grid, mid))
axis_hits = [p for p in placed if abs(p[7] - mid_at(p[6])) < p[4] + PLAYER]
axis_hits.sort(key=lambda p: p[6])
drop = axis_hits[:-UNSTEERED] if len(axis_hits) > UNSTEERED else []
placed = [p for p in placed if p not in drop]

# ---------- колонны: кое-где, в стороне от линии и от оси: кол только ранит, а в колонну можно врезаться
COLUMN_R = 0.5
rng2 = np.random.default_rng(SEED + 1)
columns = []
s = S_FROM + rng2.uniform(4, 12)
while s < S_TO:
    best = None
    for _ in range(60):
        w0, w1 = float(np.interp(s, grid, lo)), float(np.interp(s, grid, hi))
        lat = rng2.uniform(w0 + 1.0, w1 - 1.0)
        key = near(s + rng2.uniform(-3, 3), lat)
        if key is None or key not in cells: continue
        cs, cl, y, h = cells[key]
        if not S_FROM <= cs <= S_TO or not 4 <= h <= 16: continue
        if abs(cl - line_at(cs)) < clear(cs) + PLAYER + COLUMN_R + 0.8: continue
        if abs(cl - mid_at(cs)) < PLAYER + COLUMN_R + 1.0: continue
        if not w0 + 0.8 <= cl <= w1 - 0.8: continue
        around = [cells.get((key[0] + a, key[1] + b)) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        if any(o is None or abs(o[2] - y) > 1 for o in around): continue                     # пол вокруг целый
        if any(math.hypot(key[0] - p[0], key[1] - p[1]) < p[4] + COLUMN_R + 0.3 for p in placed): continue
        if any(math.hypot(key[0] - c[0], key[1] - c[1]) < 10 for c in columns): continue
        best = key; break
    if best is not None: columns.append(best)
    s += rng2.uniform(14, 28)

# ---------- блоки: материал по полу
kinds = {}
for key in columns:
    i, j = key; cs, cl, y, h = cells[key]
    mat = 'stone' if base[vol[i, y, j]].endswith(('_dirt', '_stone')) else 'ice'
    vol[i, y, j] = gid(f'gameplaymod:{mat}_column[part=bottom]')
    for q in range(1, h): vol[i, y + q, j] = gid(f'gameplaymod:{mat}_column[part=middle]')
    vol[i, y + h, j] = gid(f'gameplaymod:{mat}_column[part=top]')
    kinds[f'колонна {mat}'] = kinds.get(f'колонна {mat}', 0) + 1
for i, j, y, pre, r, height, s_, lat in placed:
    floor = base[vol[i, y, j]]
    mat = 'stone' if floor.endswith(('_dirt', '_stone')) else 'ice'
    b = f'gameplaymod:{pre}{mat}_spike[tip=true]'
    vol[i, y + 1, j] = gid(b)
    kinds[f'{pre}{mat}'] = kinds.get(f'{pre}{mat}', 0) + 1
changed = vol != orig

# ---------- отчёт
P = np.array([(p[6], p[7], p[4], p[5]) for p in placed])
print('снято блоков прежних кольев:', removed, 'групп у линии:', groups, 'у стен:', fringe, 'великанов:', giants, 'поставлено кольев:', len(placed), kinds)
for name, a, b in (('земля', S_FROM, M1), ('камень', M1, M2), ('лёд', M2, S_TO)):
    m = (P[:, 0] >= a) & (P[:, 0] < b); print(f'  {name}: кольев {int(m.sum())}, на 10 блоков {10*m.sum()/(b - a):.1f}, по размерам', {int(h): int((P[m, 3] == h).sum()) for h in (2, 6, 10)})
gap = np.abs(P[:, 1] - line_at(P[:, 0])) - P[:, 2]
print('от линии до края ближайшего кола: мин', gap.min().round(2), 'по зонам', [round(float(gap[(P[:, 0] >= a) & (P[:, 0] < b)].min()), 2) for a, b in ((S_FROM, M1), (M1, M2), (M2, S_TO))])
print('высота самого большого над полом против свода, запас мин:', min(cells[(p[0], p[1])][3] - p[5] for p in placed))
def hits(path):
    n = 0; seen = set()
    for g in np.arange(S_FROM, S_TO, 0.25):
        q = path(g)
        for k, (ps, pl, r, _) in enumerate(P):
            if k not in seen and (ps - g)**2 + (pl - q)**2 < (r + PLAYER)**2: seen.add(k); n += 1
    return n
print('колонн:', len(columns), 'высоты', sorted(cells[k][3] + 1 for k in columns), 'от оси', sorted(round(abs(cells[k][1] - mid_at(cells[k][0])), 1) for k in columns), 'от линии', sorted(round(abs(cells[k][1] - line_at(cells[k][0])), 1) for k in columns), 'расстояния по трассе', sorted(round(cells[k][0]) for k in columns))
print('без руля, прямо по оси тоннеля: задел кольев', hits(lambda g: float(np.interp(g, grid, mid))), '; по линии:', hits(line_at))
if preview:
    from PIL import Image
    W, H = int(TF.LIP_S) + 1, 90
    im = np.full((H*4, W*2, 3), 30, np.uint8)
    for (s_, l_, h, _) in cells.values():
        u, v = int(s_*2), int((l_ + 45)*4)
        if 0 <= v < H*4 and u < W*2: im[v:v + 4, u:u + 2] = (70, 70, 80)
    for g in range(len(grid)):
        v = int((line[g] + 45)*4)
        if 0 <= v < H*4: im[v:v + 2, min(g*2, W*2 - 1)] = (80, 220, 120)
    for s_, l_, r, h in P:
        u, v = int(s_*2), int((l_ + 45)*4); rr = max(1, int(r*4))
        col = {2: (255, 200, 80), 6: (255, 130, 60), 10: (255, 60, 60)}[int(h)]
        im[max(0, v - rr):v + rr, max(0, u - rr//2):u + max(1, rr//2)] = col
    for k in columns:
        u, v = int(cells[k][0]*2), int((cells[k][1] + 45)*4)
        im[max(0, v - 4):v + 4, max(0, u - 2):u + 2] = (80, 200, 255)
    Image.fromarray(im).save(preview)
if not backup:
    print('без --write мир не изменён'); sys.exit()

# ---------- запись
os.makedirs(backup, exist_ok=False)
for p in {v[0] for k, v in chunks.items() if changed[(k[0] - cx0)*16:(k[0] - cx0 + 1)*16, :, (k[1] - cz0)*16:(k[1] - cz0 + 1)*16].any()}: shutil.copy2(p, backup)
touched = 0
for (cx, cz), (p, nm, c) in chunks.items():
    a, b = (cx - cx0)*16, (cz - cz0)*16
    if not changed[a:a + 16, :, b:b + 16].any(): continue
    touched += 1
    for sec in c[b'sections'][1][1]:
        y = int.from_bytes(sec[b'Y'][1], 'big', signed=True); bs = sec.get(b'block_states')
        if bs is None or not -4 <= y < 20: continue
        oy = (y + 4)*16
        if not changed[a:a + 16, oy:oy + 16, b:b + 16].any(): continue
        arr_ = vol[a:a + 16, oy:oy + 16, b:b + 16].transpose(1, 2, 0).reshape(4096)
        u, inv = np.unique(arr_, return_inverse=True)
        bs[1][b'palette'] = (9, (10, [mca.state(lst[i]) for i in u]))
        if len(u) == 1: bs[1].pop(b'data', None)
        else: bs[1][b'data'] = (12, mca.pack(inv, len(u)))
    regs[p].put(cx, cz, mca.dump(nm, c))
for r in regs.values(): r.save()
print('записано чанков:', touched, 'копия:', backup)
