"""Трасса тоннеля и непрерывное поле высот его ската. Высоты заданы в углах клеток с шагом 1/16: соседние клетки делят углы, поэтому стыков нет."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

XC = 500; RIM_N = -308; LEDGE_Y = 105
FLAT_N, STEEP_N, MID_N = 8, 6, 3
Z_STEEP = RIM_N - 1 - FLAT_N; Z_MID = Z_STEEP - STEEP_N; Z_GENTLE = Z_MID - MID_N
F_STEEP = LEDGE_Y + 1; F_MID = F_STEEP - 2.5*STEEP_N; F_GENTLE = F_MID - MID_N
RISE, RISE2 = 0.41421356, 0.19891237                         # 22,5° и 11,25°
PLATEAU_Y, DROP, PLATEAU_LEN = -55, 30, 55
F_LIP_T = PLATEAU_Y + 1 + DROP
TUNNEL = [(XC, Z_GENTLE), (487, -358), (455, -371), (380, -345), (290, -305), (200, -272), (130, -250)]
_p = np.array(TUNNEL, float); _t = np.r_[0, np.cumsum(np.hypot(*np.diff(_p, axis=0).T))]
_c = CubicSpline(_t, _p, bc_type=((1, (0.0, -1.0)), 'natural'))
tc = _c(np.arange(0, _t[-1], 0.25)); ts = np.r_[0, np.cumsum(np.hypot(*np.diff(tc, axis=0).T))]
TLEN = ts[-1]; LIP_S = TLEN - PLATEAU_LEN
BLEND = 30.0                                                 # на такой длине уклон плавно меняется с 22,5° на 11,25°
# расстояние 0 — нижний край ската 45° (z = Z_GENTLE + 1), сплайн начинается на блок дальше
_sg = np.arange(-1.0, TLEN + 2, 0.25)
def _profile(s1):
    w = np.clip((_sg - s1 + BLEND/2)/BLEND, 0, 1); slope = RISE + (RISE2 - RISE)*w*w*(3 - 2*w)
    return F_GENTLE - np.r_[0, np.cumsum((slope[1:] + slope[:-1])/2*0.25)]
_lo, _hi = BLEND, LIP_S - BLEND                               # середина перехода подбирается так, чтобы край обрыва остался на прежней высоте
for _ in range(50):
    S1 = (_lo + _hi)/2
    if np.interp(LIP_S, _sg + 1, _profile(S1) + RISE) > F_LIP_T: _lo = S1
    else: _hi = S1
_prof = _profile(S1) + RISE                                   # _sg + 1 считается от начала сплайна
def FS(s):
    """Высота поверхности на расстоянии s от начала сплайна; до начала продолжается уклоном 22,5°."""
    s = np.asarray(s, float); return np.interp(s, _sg + 1, _prof) + np.where(s < 0, -RISE*s, 0)
_tree = cKDTree(tc)
Z_LAND = Z_STEEP + 1                                         # нижний край посадочного склона: здесь начинается крутой спуск
GRADE = 0.7                                                  # предел уклона; у внутренней стены изгиба путь короче, без предела там круче 45°
PAD = 40
def _limit(h, m):
    """Верхняя огибающая на маске m: точка не ниже соседней выше по склону минус GRADE на расстояние. Внутренний берег изгиба приподнят, скат идёт от него к внешнему."""
    N = [(a, b, GRADE*np.hypot(a, b)) for a in (-2, -1, 0, 1, 2) for b in (-2, -1, 0, 1, 2) if (a or b) and np.gcd(a, b) == 1]
    h = h.copy()
    for _ in range(4000):
        o = h.copy()
        for a, b, d in N:
            v = np.roll(np.roll(h, a, 0), b, 1) - d; ok = np.roll(np.roll(m, a, 0), b, 1) & m
            h[ok] = np.maximum(h[ok], v[ok])
        if np.abs(h - o).max() < 1e-4: return h
    raise SystemExit('ограничение уклона не сошлось')
def _soft(h, m, sigma):
    """Гауссово сглаживание только по точкам маски."""
    w = gaussian_filter(m.astype(float), sigma); v = gaussian_filter(np.where(m, h, 0.0), sigma)
    return np.where(m, v/np.maximum(w, 1e-6), h)
def corners(x0, x1, z0, z1, cells):
    """Высоты углов клеток в 1/16 блока: массив [x - x0, z - z0] для углов x0..x1+1, z0..z1+1. cells — клетки ската (x, z)."""
    xs = np.arange(x0 - PAD, x1 + 2 + PAD); zs = np.arange(z0 - PAD, z1 + 2 + PAD)
    gx, gz = np.meshgrid(xs, zs, indexing='ij')
    _, k = _tree.query(np.c_[gx.ravel(), gz.ravel()]); s = ts[k].reshape(gx.shape)
    m = np.zeros(gx.shape, bool)
    for x, z in cells: m[x - xs[0]:x - xs[0] + 2, z - zs[0]:z - zs[0] + 2] = True
    h = gaussian_filter(FS(s), 3.0, mode='nearest')
    plane = F_GENTLE + RISE*(gz - (Z_GENTLE + 1))             # начало трассы: строгая плоскость, продолжающая скат 45°
    w = np.clip(s/25.0, 0, 1); w = w*w*(3 - 2*w)
    h = plane*(1 - w) + h*w
    h = _soft(_limit(h, m), m, 1.5)
    start = np.clip((gz - (Z_GENTLE - 2))/2.0, 0, 1)*(s < 15)  # у самого стыка со скатом 45° — точная плоскость
    h = h*(1 - start) + plane*start
    return np.round(h[PAD:-PAD, PAD:-PAD]*16).astype(int)
def landing(x0, x1, z0, z1):
    """Посадочный склон: плоскость 22,5° на север, нижний край ровно на верхе крутого спуска."""
    gz = np.arange(z0, z1 + 2)[None, :].repeat(x1 - x0 + 2, 0)
    return np.round((F_STEEP + RISE*(gz - Z_LAND))*16).astype(int)
DX_MAX, DZ_MIN, DZ_MAX = 12, -4, 12
def cell_state(nw, ne, sw, se):
    """Углы клетки в 1/16 блока -> (y блока, имя состояния, была ли правка под пределы формы). Свойства — как в FieldSlideBlock."""
    dx = int(np.clip(ne - nw, 0, DX_MAX)); dz = int(np.clip(sw - nw, DZ_MIN, DZ_MAX)); tw = int(np.clip(se - ne - sw + nw, -1, 1))
    clipped = dx != ne - nw or dz != sw - nw or tw != se - ne - sw + nw
    low = int(nw) + min(0, dx, dz, dx + dz + tw); y = low >> 4
    return y, f'gameplaymod:slide_field[dx={dx},dz={dz - DZ_MIN},part={low & 15},twist={tw + 1}]', clipped
def is_landing(x, z):
    """Клетка посадочного склона: первые ряды у входа; остальная трасса — поле corners()."""
    return Z_LAND <= z <= RIM_N - 1 and x >= XC - 30
