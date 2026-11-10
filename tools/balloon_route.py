#!/usr/bin/env python3
"""Ветер для полёта на шаре от области «шар» до «засада_туземцев» (LevelMod): площадки ветра с промежутками.

Запуск: balloon_route.py <папка мира> [--preview файл.png] [--write <новая папка копии>]. Без --write мир не меняется. Игра закрыта.
2026-11-05, ветер 25 блоков/с (BalloonEntity.WIND_SPEED, было 5), по просьбе: «не последовательной дорожкой, а интервалами; можно по
диагонали». Ветер не дорожкой, а короткими площадками (1–3 маркера через PAD_STEP), между ними шар идёт по инерции около GAP блоков
(при таком ветре он проходит по инерции до 40). Курсы — с шагом 30° (поворот маркера), в том числе диагональные, путь — почти прямая
ломаная PATH: над морем на северо-восток к протоке, по ней на север, подъём наклонными ветрами (CLIMB_PITCHES вместо отвесного столба)
к верхнему ветру Y_HIGH, на северо-восток к засаде и вдоль её полосы на восток; последняя площадка рассчитана так, чтобы шар по инерции
остановился в полосе (не ближе AMB_MARGIN к краю).

Расстановка — жадный подбор по модели: шар летит; когда он ушёл от площадки на GAP (или почти встал), перебираются площадки у его
носа (12 курсов × длина × наклон), каждая прогоняется вперёд, берётся лучшая (продвижение вдоль PATH, отход от пути не больше CORR,
просвет над рельефом не меньше MIN_CLEAR; на подъёме — ещё набранная высота; в конце — точка остановки у середины засады). Потом полёт
проверяется точной моделью со стенами.

Симулятор повторяет BalloonEntity.simulate (сцепление 0,03, подъём 3 блока/с, снижение цели SINK, толчок руля) и Winds.at (среднее
направлений площадок 3×3, задевших шар). Пилот модели: стропа высоты не чаще раза в CLICK_EVERY тиков, когда плоскость следующей
площадки выше корпуса или плот под шаром подходит к рельефу; руль — в разрыве, к ближайшему маркеру следующей площадки, если по инерции
туда не доходит. Каждой стропе пишется FUEL_X × её расход (руль не меньше HELM_MIN, всё не больше MAX_FUEL) — и в мир, и в снимок
миссии идущего прохождения LevelMod (иначе сброс миссии вернёт прежнее топливо).
Старт — шар из снимка миссии (если прохождение идёт), иначе из мира; плот, висящий под шаром, учитывается в просвете над рельефом.
"""
import gzip, math, os, shutil, struct, sys, uuid
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca

WIND, GRIP, CLIMB, CLIMB_GRIP, GAIN, LEAD, SINK = 25/20, 0.03, 3/20, 0.2, 0.05, 6.0, 0.5/20
NUDGE = GRIP/(1 - GRIP)         # толчок руля: шар проходит 1 блок и встаёт (BalloonEntity.NUDGE_SPEED)
MAX_SIDE = 0.5
RADIUS, EDGE = 5.0, 0.2           # зона маркера и сила на её краю (доля от WIND), WindMarkerEntity.RADIUS, EDGE_STRENGTH
CLICK_EVERY = 5                 # каждая стропа щёлкает не чаще раза в 5 тиков
FUEL_X = 2.0                    # топливо стропам: во столько раз больше расхода на кратчайшем пути
HELM_MIN = 16                   # руль не меньше: по расчёту он почти не нужен, но человек промахивается
MAX_FUEL = 256                  # BalloonRopeItem.MAX_FUEL
GAP, GAP_MIN, V_MIN = 18.0, 10.0, 0.4       # по инерции от площадки до следующей; короче — только если шар почти встал (блоков/тик)
PAD_SIZES, PAD_STEP, PAD_AHEAD = (1, 2, 3), 4.0, 4.0   # маркеров в площадке, шаг между ними, первый — на столько впереди центра шара
CLIMB_PITCHES = (30.0, 60.0)    # наклоны ветров подъёма
CORR, MIN_CLEAR = 10.0, 3.0     # отход от пути без штрафа; просвет над рельефом в подборе
AMB_MARGIN = 5.0                # остановка в засаде не ближе к краю полосы
HELM_SLACK = 3.0                # руль молчит, если по инерции шар проходит не дальше этого от маркера
RAFT_CLEAR = 2.0                # просвет под висящим плотом, ниже — пилот подбрасывает высоту
Y_LOW, Y_HIGH = 126, 172        # высоты ветров: над морем и протокой, над джунглями (деревья до ~157)
GROUP = 'flight'

world = sys.argv[1]
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
preview, backup = arg('--preview'), arg('--write')
D = os.path.join(world, 'dimensions/minecraft/overworld/'); REG, ENT = D + 'region/', D + 'entities/'

# ---------- рельеф: карта MOTION_BLOCKING (листва считается)
_ch = {}
def chunk(cx, cz):
    if (cx, cz) not in _ch:
        p = REG + f'r.{cx >> 5}.{cz >> 5}.mca'
        nm, c = mca.parse(mca.Region(p).raw(cx, cz))
        raw = np.frombuffer(c[b'Heightmaps'][1][b'MOTION_BLOCKING'][1], '>u8')
        v = [(int(w) >> (9*k)) & 511 for w in raw for k in range(7)][:256]
        _ch[(cx, cz)] = np.array(v).reshape(16, 16) - 65          # [z][x] → верхний блок
    return _ch[(cx, cz)]
def ground(x, z): return int(chunk(x >> 4, z >> 4)[z & 15, x & 15])

# ---------- области и прохождение LevelMod
LEVELS = os.path.join(world, 'data/levelmod/levels.dat')
lv_name, lv_root = mca.parse(gzip.open(LEVELS).read())
areas = lv_root[b'data'][1][b'location_areas'][1]
def loop(name):
    return [tuple(struct.unpack('>d', c)[0] for c in p[1]) for p in areas[name.encode()][1][b'loops'][1][1][0][1]]
AMBUSH = loop('засада_туземцев')
def inside(poly, x, z):
    c = False
    for (x1, z1), (x2, z2) in zip(poly, poly[1:] + poly[:1]):
        if (z1 > z) != (z2 > z) and x < (x2 - x1)*(z - z1)/(z2 - z1) + x1: c = not c
    return c
d_ = lambda v: struct.unpack('>d', v)[0]; i_ = lambda v: struct.unpack('>i', v)[0]
def rafts_snapshots():
    """Снимки шаров группы GROUP в идущих прохождениях: словари gameplaymod:rafts."""
    out = []
    for run in lv_root[b'data'][1].get(b'runs', (9, (10, [])))[1][1]:
        snap = run.get(b'snapshots', (10, {}))[1].get(GROUP.encode())
        if snap and b'gameplaymod:rafts' in snap[1]: out.append(snap[1][b'gameplaymod:rafts'][1])
    return out
snaps = rafts_snapshots()

# ---------- шар и стропы из мира
def entities_near(x0, z0, r):
    out = []
    for cx in range((x0 - r) >> 4, ((x0 + r) >> 4) + 1):
        for cz in range((z0 - r) >> 4, ((z0 + r) >> 4) + 1):
            p = ENT + f'r.{cx >> 5}.{cz >> 5}.mca'
            if not os.path.exists(p): continue
            b = mca.Region(p).raw(cx, cz)
            if b is None: continue
            nm, c = mca.parse(b)
            for e in c[b'Entities'][1][1]: out.append(((cx, cz), e))
    return out
ents = entities_near(241, 237, 48)
world_balloon = next(e for _, e in ents if e[b'id'][1] == b'gameplaymod:balloon' and e.get(b'Group', (8, b''))[1] == GROUP.encode())
if snaps:
    balloon = next(e for e in snaps[0][b'Balloons'][1][1] if e[b'id'][1] == b'gameplaymod:balloon' and e.get(b'Group', (8, b''))[1] == GROUP.encode())
    rafts = [e for e in snaps[0][b'Balloons'][1][1] if e[b'id'][1] == b'gameplaymod:raft']
    hung = bool(rafts) and any(e[b'id'][1] == b'gameplaymod:calm_rope' and b'EndRaft' in e for e in snaps[0][b'Ropes'][1][1])
    src = 'снимок миссии'
else:
    balloon = world_balloon; rafts = []; hung = False; src = 'мир'
BX, BY, BZ = (d_(v) for v in balloon[b'Pos'][1][1])
SX, SY, SZ = i_(balloon[b'SX'][1]), i_(balloon[b'SY'][1]), i_(balloon[b'SZ'][1])
HEAD = i_(balloon[b'Heading'][1]) if b'Heading' in balloon else 0
FRONT = np.array([-math.sin(math.radians(HEAD*90)), math.cos(math.radians(HEAD*90))]); LEFT = np.array([math.cos(math.radians(HEAD*90)), math.sin(math.radians(HEAD*90))])
if hung:
    rf = rafts[0]; RX, RY, RZ = (d_(v) for v in rf[b'Pos'][1][1]); RAFT_DROP = BY - RY       # низ плота ниже низа шара
    RSX, RSZ = i_(rf[b'SX'][1]), i_(rf[b'SZ'][1])
else: RAFT_DROP, RSX, RSZ = 0.0, 0, 0
print(f'шар ({BX:.1f}, {BY:.1f}, {BZ:.1f}), {SX}×{SY}×{SZ}, нос {HEAD}, из: {src}' + (f'; под ним плот {RSX}×{RSZ} на {RAFT_DROP:.1f} ниже' if hung else ''))

# ---------- маркеры: зона — RADIUS от центра до ближайшей точки корпуса, сила к краю линейно до EDGE (WindMarkerEntity.strength)
def course(yaw, pitch=0.0):
    r, p = math.radians(yaw), math.radians(pitch)
    return np.array([-math.sin(r)*math.cos(p), math.sin(p), math.cos(r)*math.cos(p)])
def strength(m, lo, hi):
    c = np.array([m[0], m[1] + 0.5, m[2]]); d = np.linalg.norm(np.maximum(np.maximum(lo - c, 0), c - hi))
    return 0.0 if d > RADIUS else 1 - (1 - EDGE)*d/RADIUS
REACH = max(SX, SZ)/2 + RADIUS + 1
def cell(x, y, z, yw, pt): return (math.floor(x) + .5, math.floor(y), math.floor(z) + .5, yw, pt)

# ---------- рельеф для расчёта: верх под каждой клеткой и наибольший под следом шара (и плота)
GX0, GX1, GZ0, GZ1 = 150, 660, -220, 300
HG = np.array([[ground(x, z) for x in range(GX0, GX1)] for z in range(GZ0, GZ1)])
def gnd(x, z): return HG[z - GZ0, x - GX0] if GX0 <= x < GX1 and GZ0 <= z < GZ1 else ground(x, z)
from scipy.ndimage import maximum_filter
assert max(SX, RSX) % 2 == 0
FOOT = max(SX, RSX) + 1             # under(x, z, r) берёт клетки floor(x - r)..floor(x + r)
HMAX = maximum_filter(HG, size=FOOT, mode='nearest')
def under(x, z):
    r = max(SX, RSX)/2; gx, gz = math.floor(x - r) + FOOT//2, math.floor(z - r) + FOOT//2
    return int(HMAX[gz - GZ0, gx - GX0])
def free(x, y, z):
    for gx in range(math.floor(x - SX/2 + 1e-3), math.floor(x + SX/2 - 1e-3) + 1):
        for gz in range(math.floor(z - SZ/2 + 1e-3), math.floor(z + SZ/2 - 1e-3) + 1):
            if gnd(gx, gz) + 1 > y + 1e-3: return False
    return True

# ---------- движение шара (BalloonEntity.simulate) и пилот модели
class Ball:
    __slots__ = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'target', 'lift', 'helm', 'last_l', 'last_h', 'aloft', 't', 'clear')
    def __init__(s, x, y, z):
        s.x, s.y, s.z = x, y, z; s.vx = s.vy = s.vz = 0.0; s.target = y; s.lift = s.helm = 0; s.last_l = s.last_h = -99
        s.aloft = False; s.t = 0; s.clear = 1e9
    def copy(s):
        c = Ball.__new__(Ball)
        for k in Ball.__slots__: setattr(c, k, getattr(s, k))
        return c
def hits(b, ms):
    """Задевшие корпус маркеры: {маркер: сила}."""
    lo = np.array([b.x - SX/2, b.y, b.z - SZ/2]); hi = np.array([b.x + SX/2, b.y + SY, b.z + SZ/2]); out = {}
    for m in ms:
        if abs(m[0] - b.x) < REACH and abs(m[2] - b.z) < REACH and abs(m[1] - b.y - SY/2) < SY/2 + RADIUS + 1:
            k = strength(m, lo, hi)
            if k > 0: out[m] = k
    return out
def step(b, hit, need, fast=False, lift_limit=10**9, every_lift=CLICK_EVERY, stop_raft=False):
    """Тик: стропа высоты (пилот), ветер, движение. need — высота плоскости ветра, до которой пилот держит корпус."""
    w = np.zeros(3)                 # Winds.at: направление — сумма с весом по силе, длина — наибольшая сила
    for m, k in hit.items(): w += course(m[3], m[4])*k
    w = w/np.linalg.norm(w)*max(hit.values()) if np.linalg.norm(w) > 1e-3 else np.zeros(3)
    landed = w[1] <= 0 and (b.y - 0.05 < under(b.x, b.z) + 1 if fast else not free(b.x, b.y - 0.05, b.z))
    low_raft = False
    if hung and b.aloft and not stop_raft:
        g_ahead = under(b.x + b.vx*40, b.z + b.vz*40)
        low_raft = b.y - RAFT_DROP < g_ahead + 1 + RAFT_CLEAR
    if w[1] <= 0 and b.t - b.last_l >= every_lift and b.lift < lift_limit and (low_raft or (need is not None and need > b.y + SY - 1.5)):
        n2 = min(b.target + 1, b.y + LEAD)
        if n2 != b.target: b.target = n2; b.lift += 1
        b.last_l = b.t
    if landed: b.target = max(b.target, b.y)
    else: b.target -= SINK
    want = max(-CLIMB, min(CLIMB, (b.target - b.y)*GAIN))
    if not landed:
        want += w[1]*WIND
        if w[1] != 0: b.target += w[1]*WIND
    b.vy += (want - b.vy)*CLIMB_GRIP
    if landed: b.vx *= 0.5; b.vz *= 0.5
    else: b.vx += (w[0]*WIND - b.vx)*GRIP; b.vz += (w[2]*WIND - b.vz)*GRIP
    if fast:                        # в подборе: стены не считаем, касание рельефа ловится по просвету
        b.x += b.vx; b.z += b.vz; b.y += b.vy
    else:
        if b.vx and free(b.x + b.vx, b.y, b.z): b.x += b.vx
        else: b.vx = 0
        if b.vz and free(b.x, b.y, b.z + b.vz): b.z += b.vz
        else: b.vz = 0
        if b.vy and free(b.x, b.y + b.vy, b.z): b.y += b.vy
        else: b.vy = 0
    g = under(b.x, b.z)
    if not b.aloft and b.y - RAFT_DROP > g + 1 + RAFT_CLEAR: b.aloft = True
    if b.aloft: b.clear = min(b.clear, b.y - RAFT_DROP - (g + 1))
    b.t += 1
    return w

# ---------- путь (ломаная; курсы ветров — любые с шагом 30°, в том числе по диагонали)
TARGET = np.array([505.0, -86.0])           # середина полосы засады; к ней — вдоль полосы, с запада
PATH = np.array([(BX, BZ), (372, 40), (378, -30), (420, -100), tuple(TARGET)])
S_CLIMB = None
_seg = PATH[1:] - PATH[:-1]; _len = np.linalg.norm(_seg, axis=1); _cum = np.concatenate([[0], np.cumsum(_len)])
def proj(x, z):
    p = np.array([x, z]); u = np.clip(((p - PATH[:-1])*_seg).sum(1)/_len**2, 0, 1)
    foot = PATH[:-1] + _seg*u[:, None]; d = np.linalg.norm(foot - p, axis=1); k = int(np.argmin(d))
    return _cum[k] + u[k]*_len[k], d[k]
S_CLIMB = proj(374, 15)[0]                  # отсюда наклонные ветра поднимают шар к верхнему ветру
def inside_m(x, z, mg=AMB_MARGIN):
    return all(inside(AMBUSH, x + dx, z + dz) for dx, dz in ((0, 0), (mg, 0), (-mg, 0), (0, mg), (0, -mg)))
def coast_end(b): k = (1 - GRIP)/GRIP; return b.x + b.vx*k, b.z + b.vz*k

def mode_of(b, s):
    if b.y + SY - 1.5 >= Y_HIGH - 2 or b.target >= Y_HIGH - SY + 3: return 'high'
    return 'climb' if s >= S_CLIMB else 'low'
MODES = ['low', 'climb', 'high']
def need_of(mode): return {'low': Y_LOW, 'climb': None, 'high': Y_HIGH}[mode]

def candidates(b, mode):
    sp = math.hypot(b.vx, b.vz); dx, dz = (b.vx/sp, b.vz/sp) if sp > 1e-3 else (0.0, 0.0)
    cx, cz = b.x + dx*PAD_AHEAD, b.z + dz*PAD_AHEAD
    out = []
    pitches = CLIMB_PITCHES if mode == 'climb' else (0.0,)
    for pt in pitches:
        y0 = (Y_LOW if mode == 'low' else Y_HIGH) - 0.5 if pt == 0 else b.y + SY/2 - 0.5
        for yw in range(0, 360, 30):
            d = course(yw, pt)
            for k in PAD_SIZES:
                out.append([cell(cx + d[0]*PAD_STEP*i, y0 + d[1]*PAD_STEP*i, cz + d[2]*PAD_STEP*i, yw, pt) for i in range(k)])
    return out

def trial(b, ms, pad, mode):
    """Шар после площадки: до следующего срабатывания (ушёл от площадки на GAP) или 200 тиков. Оценка и просвет."""
    c = b.copy(); allm = ms + pad; left = False; run = 0.0; dev = 0.0; s0 = proj(c.x, c.z)[0]; y0 = c.y
    for _ in range(200):
        h = hits(c, [m for m in allm if abs(m[0] - c.x) < REACH and abs(m[2] - c.z) < REACH])
        step(c, h, need_of(mode), fast=True)
        dev = max(dev, proj(c.x, c.z)[1])
        if h: left = True; run = 0.0
        elif left:
            run += math.hypot(c.vx, c.vz)
            if run >= GAP: break
    s, d = proj(c.x + c.vx*10, c.z + c.vz*10)
    sc = s - 2.0*max(0.0, dev - CORR) - 0.3*len(pad) - (0 if left else 500)
    if mode == 'climb': sc += 2.0*(min(c.target, Y_HIGH) - y0)
    if mode == 'high':
        ex, ez = coast_end(c); sc = -math.hypot(ex - TARGET[0], ez - TARGET[1]) - 2.0*max(0.0, dev - CORR) - 0.3*len(pad) + (300 if inside_m(ex, ez) else 0)
    if c.clear < MIN_CLEAR: sc -= 1e4
    return sc

def plan():
    b = Ball(BX, BY, BZ); pads = []
    while b.y + SY - 1.5 < Y_LOW + 0.5: step(b, {}, Y_LOW)           # подъём со стоянки к нижнему ветру
    left = True; run = 1e9; done = False; mode = 'low'
    for _ in range(20*240):
        ms = [m for p in pads for m in p]
        h = hits(b, [m for m in ms if abs(m[0] - b.x) < REACH and abs(m[2] - b.z) < REACH])
        s = proj(b.x, b.z)[0]; mode = max(mode, mode_of(b, s), key=MODES.index)
        if h: run = 0.0; left = False
        else:
            if not left: left = True; run = 0.0
            run += math.hypot(b.vx, b.vz)
            if mode == 'high' and pads and pads[-1][0][4] == 0 and inside_m(*coast_end(b)): done = True; break
            sp = math.hypot(b.vx, b.vz)
            if run >= GAP or (sp < V_MIN and run >= GAP_MIN) or sp < 0.05 or not pads:
                best = max(candidates(b, mode), key=lambda pad: trial(b, ms, pad, mode))
                pads.append(best); ms = ms + best
                h = hits(b, best)
        step(b, h, need_of(mode), fast=True)
    return pads, done

ISL, planned = plan()
M = [m for g in ISL for m in g]
OWNER = {m: k for k, g in enumerate(ISL) for m in g}
print(f'площадок {len(ISL)}, маркеров {len(M)}; размеры', [len(g) for g in ISL], 'курсы', [g[0][3] for g in ISL], 'наклоны', [g[0][4] for g in ISL],
      '' if planned else '— ПОДБОР НЕ ДОВЁЛ ДО ЗАСАДЫ')

# ---------- полёт по поставленным ветрам (точная модель со стенами)
def coasts_into(b, isl):
    """Дойдёт ли шар по инерции (без руля и ветра) до площадки isl: тогда руль не нужен."""
    c = b.copy()
    for _ in range(90):
        c.vx *= 1 - GRIP; c.vz *= 1 - GRIP; c.x += c.vx; c.z += c.vz
        if hits(c, isl): return True
        if abs(c.vx) + abs(c.vz) < 0.02: return False
    return False
def fly(lift_limit=10**9, helm_limit=10**9, every_lift=CLICK_EVERY, every_helm=CLICK_EVERY, burst=1):
    """burst > 1: руль пачками (приказ напарнику: burst щелчков разом, следующий не раньше every_helm)."""
    b = Ball(BX, BY, BZ); trace = []; reached = None; cur = -1
    for t in range(20*600):
        hit = hits(b, [m for m in M if abs(m[0] - b.x) < REACH and abs(m[2] - b.z) < REACH])
        if hit: cur = max(cur, max(OWNER[m] for m in hit))
        nxt = ISL[cur + 1] if cur + 1 < len(ISL) else []
        # высота: плоскость следующей площадки (или той, в которой шар), если она горизонтальная
        ref = nxt if nxt else (ISL[cur] if hit and cur >= 0 else [])
        need = ref[0][1] + 0.5 if ref and ref[0][4] == 0 else None
        landed = not free(b.x, b.y - 0.05, b.z)
        # руль: в разрыве к ближайшему маркеру следующей площадки, если по инерции туда не доходит
        if not hit and nxt and cur >= 0 and not landed and b.t - b.last_h >= every_helm and b.helm < helm_limit:
            aim = min(nxt, key=lambda m: math.hypot(m[0] - b.x, m[2] - b.z))
            R = np.array([aim[0] - b.x, aim[2] - b.z]) - np.array([b.vx, b.vz])*(1 - GRIP)/GRIP
            if not coasts_into(b, nxt) and np.linalg.norm(R) > HELM_SLACK:
                ax = FRONT if abs(R @ FRONT) >= abs(R @ LEFT) else LEFT
                sgn = 1 if R @ ax > 0 else -1
                for _ in range(burst):
                    if b.helm < helm_limit:
                        b.vx = max(-MAX_SIDE, min(MAX_SIDE, b.vx + ax[0]*sgn*NUDGE)); b.vz = max(-MAX_SIDE, min(MAX_SIDE, b.vz + ax[1]*sgn*NUDGE)); b.helm += 1
                b.last_h = b.t
        step(b, hit, need, lift_limit=lift_limit, every_lift=every_lift, stop_raft=reached is not None)
        if t % 20 == 0: trace.append((t//20, round(b.x, 1), round(b.y, 1), round(b.z, 1), b.lift, b.helm, cur))
        if reached is None and not nxt and inside(AMBUSH, b.x, b.z): reached = (t/20, round(b.x, 1), round(b.y, 1), round(b.z, 1), b.lift, b.helm)
        if reached is None: clear = b.clear
        if t > 200 and not free(b.x, b.y - 0.05, b.z) and abs(b.vx) + abs(b.vz) < 1e-3: break
    return dict(end=(round(b.x, 1), round(b.y, 1), round(b.z, 1)), t=t/20, lift=b.lift, helm=b.helm, reached=reached, trace=trace,
                clear=clear, landed_inside=inside(AMBUSH, b.x, b.z), cur=cur)

r = fly()
print(f"кратчайший путь: высота {r['lift']}, руль {r['helm']}, засада {'достигнута ' + str(r['reached']) if r['reached'] else 'НЕТ'}, "
      f"конец {r['end']} за {r['t']:.0f} с, сел в засаде: {r['landed_inside']}, просвет под {'плотом' if hung else 'шаром'} до засады не меньше {r['clear']:.1f}")
for row in r['trace'][::5]:
    print('   ', row)
    if row[0] > r['t'] - 1: break
LIFT = min(MAX_FUEL, int(math.ceil(r['lift']*FUEL_X))); HELM = min(MAX_FUEL, max(HELM_MIN, int(math.ceil(r['helm']*FUEL_X))))
capped = [n for n, v, u in (('высоты', LIFT, r['lift']), ('руля', HELM, r['helm'])) if v < u*FUEL_X]
print(f'топливо: высота {LIFT}, руль {HELM} (×{FUEL_X:g} от {r["lift"]} и {r["helm"]}, руль не меньше {HELM_MIN})' + (f'; упёрлось в предел {MAX_FUEL}: {", ".join(capped)}' if capped else ''))
# проверки с записанным топливом: руль у напарника (приказ — 2 щелчка, приказ не чаще раза в 3 с), игрок щёлкает вдвое реже
checks = []
def check(name, **kw):
    q = fly(lift_limit=LIFT, helm_limit=HELM, **kw); good = bool(q['reached']) and q['landed_inside']
    checks.append(good)
    print(f"  {name}: {'долетает' if good else 'НЕ долетает, конец ' + str(q['end'])}; высота {q['lift']}, руль {q['helm']}, просвет {q['clear']:.1f}, {q['t']:.0f} с")
check('руль у напарника', every_helm=60, burst=2)
check('щелчки вдвое реже', every_lift=10, every_helm=10)
ok = planned and bool(r['reached']) and r['landed_inside'] and r['clear'] > 0.5

if preview:
    from PIL import Image, ImageDraw
    X0, X1, Z0, Z1 = 200, 620, -170, 290
    im = Image.new('RGB', ((X1 - X0)*2, (Z1 - Z0)*2)); px = im.load()
    for gx in range(X0, X1):
        for gz in range(Z0, Z1):
            h = ground(gx, gz); c = (30, 60, 140) if h < 66 else (int(min(255, 40 + (h - 60)*1.6)),) * 3
            for a in range(2):
                for b in range(2): px[(gx - X0)*2 + a, (gz - Z0)*2 + b] = c
    dr = ImageDraw.Draw(im)
    dr.polygon([((ax - X0)*2, (az - Z0)*2) for ax, az in AMBUSH], outline=(0, 255, 255))
    for m in M: dr.rectangle([((m[0] - X0)*2 - 2, (m[2] - Z0)*2 - 2), ((m[0] - X0)*2 + 2, (m[2] - Z0)*2 + 2)], outline=(255, 255, 0) if m[4] == 0 else (255, 0, 255))
    pts = [((q[1] - X0)*2, (q[3] - Z0)*2) for q in r['trace']]
    dr.line(pts, fill=(255, 60, 60), width=2)
    im.save(preview); print('предпросмотр', preview)
if not ok: raise SystemExit('полёт не доходит или задевает рельеф: в мир не пишу')
if not backup: print('без --write мир не изменён'); sys.exit()

# ---------- запись: маркеры ветра (прежние на маршруте убираются), топливо строп в мире и в снимках миссии
os.makedirs(os.path.join(backup, 'entities'))
flt = lambda v: (5, struct.pack('>f', float(v))); dbl = lambda v: struct.pack('>d', float(v))
def kind_of(e): return struct.unpack('>b', e[b'Kind'][1])[0] if e[b'Kind'][0] == 1 else i_(e[b'Kind'][1])
def fuel_for(e): return LIFT if kind_of(e) == 0 else HELM
touched = {}
def chunk_ents(cx, cz):
    if (cx, cz) not in touched:
        p = ENT + f'r.{cx >> 5}.{cz >> 5}.mca'
        if os.path.exists(p) and not os.path.exists(os.path.join(backup, 'entities', os.path.basename(p))): shutil.copy2(p, os.path.join(backup, 'entities'))
        b = mca.Region(p).raw(cx, cz) if os.path.exists(p) else None
        if b is None: nm, c = b'', {b'Position': (11, struct.pack('>ii', cx, cz)), b'DataVersion': balloon_dv, b'Entities': (9, (10, []))}
        else: nm, c = mca.parse(b)
        if c[b'Entities'][1][0] == 0: c[b'Entities'] = (9, (10, c[b'Entities'][1][1]))
        touched[(cx, cz)] = (p, nm, c)
    return touched[(cx, cz)][2]
balloon_dv = next(mca.parse(mca.Region(ENT + f'r.{ck[0] >> 5}.{ck[1] >> 5}.mca').raw(*ck))[1][b'DataVersion'] for ck, e in ents if e is world_balloon)
xs = [m[0] for m in M] + [220, 470]; zs = [m[2] for m in M] + [-145, 250]      # и вся полоса прежних маршрутов (сплошной ветер, островки)
removed = 0
for cx in range((int(min(xs)) - 8) >> 4, ((int(max(xs)) + 8) >> 4) + 1):
    for cz in range((int(min(zs)) - 8) >> 4, ((int(max(zs)) + 8) >> 4) + 1):
        p = ENT + f'r.{cx >> 5}.{cz >> 5}.mca'
        if not os.path.exists(p) or mca.Region(p).raw(cx, cz) is None: continue
        c = chunk_ents(cx, cz); before = len(c[b'Entities'][1][1])
        c[b'Entities'] = (9, (10, [e for e in c[b'Entities'][1][1] if e[b'id'][1] != b'gameplaymod:wind_marker']))
        removed += before - len(c[b'Entities'][1][1])
for mx, my, mz, yw, pt in M:
    c = chunk_ents(math.floor(mx) >> 4, math.floor(mz) >> 4)
    c[b'Entities'][1][1].append({b'id': (8, b'gameplaymod:wind_marker'), b'Pos': (9, (6, [dbl(mx), dbl(my), dbl(mz)])), b'Motion': (9, (6, [dbl(0)]*3)),
        b'Rotation': (9, (5, [struct.pack('>f', 0.0)]*2)), b'UUID': (11, uuid.uuid4().bytes), b'Yaw': flt(yw), b'Pitch': flt(pt)})
ropes_world = 0
for ck, e in ents:
    if e[b'id'][1] == b'gameplaymod:balloon_rope':
        c = chunk_ents(*ck)
        for e2 in c[b'Entities'][1][1]:
            if e2[b'UUID'] == e[b'UUID']: e2[b'Fuel'] = (3, struct.pack('>i', fuel_for(e))); ropes_world += 1
for (cx, cz), (p, nm, c) in touched.items():
    r_ = mca.Region(p); r_.put(cx, cz, mca.dump(nm, c)); r_.save()
ropes_snap = 0
if snaps:
    shutil.copy2(LEVELS, os.path.join(backup, 'levels.dat'))
    for sn in snaps:
        for e in sn[b'Ropes'][1][1]:
            if e[b'id'][1] == b'gameplaymod:balloon_rope': e[b'Fuel'] = (3, struct.pack('>i', fuel_for(e))); ropes_snap += 1
    tmp = LEVELS + '.tmp'
    with gzip.open(tmp, 'wb') as f: f.write(mca.dump(lv_name, lv_root))
    os.replace(tmp, LEVELS)
print(f'записано: маркеров {len(M)} в {len(ISL)} островках, прежних маркеров убрано {removed}, топливо: высота {LIFT}, руль {HELM} '
      f'(строп в мире {ropes_world}, в снимках миссии {ropes_snap}); копия {backup}')
