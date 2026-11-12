import numpy as np, time, os
from PIL import Image
from scipy.ndimage import gaussian_filter, distance_transform_edt, label, binary_dilation, binary_closing, binary_fill_holes
t0 = time.time()
D = '/home/bogdan/Изображения/'
H, W = 768, 1362
XW, Z0, Wc, Hc = 420, 600, 2600, 1990
rng = np.random.default_rng(5)

# ---------- исходные карты центра
mraw = np.array(Image.open(D+'materials.png').convert('RGB')).astype(int)
def is_(c): return np.abs(mraw - np.array(c)).sum(2) < 60
mat = np.full((H, W), 2, np.uint8)               # 0 горы 1 глубокая вода 2 луг 3 песок 4 тропа 5 мелководье 6 джунгли
for k, c in enumerate([(128, 128, 128), (30, 80, 220), (40, 170, 40), (240, 220, 60), (140, 90, 40)]): mat[is_(c)] = k
hh = 200*(1 - np.array(Image.open(D+'Карта_высот.png')).astype(np.float32)/255)
lat = Image.open(D+'latest.png').convert('RGB'); hsv = np.array(lat.convert('HSV')).astype(float)
hue, sat, val = hsv[..., 0]*360/255, hsv[..., 1]/255, hsv[..., 2]/255
dark = gaussian_filter(val, 2) < 0.52
forest = np.where(dark & (mat != 1) & (mat != 0), np.where(hue > 130, 2, 1), 0).astype(np.uint8)
# плато (заливка контура обрыва), как при построении высот
art = np.array(Image.open(D+'artefacts.png').convert('RGBA')).astype(int)
red = (art[..., 0] > 180) & (art[..., 1] < 90) & (art[..., 2] < 90) & (art[..., 3] > 100)
lab, _ = label(red); sz = np.bincount(lab.ravel()); sz[0] = 0
line = lab == sz.argmax(); ys, xs = np.where(line)
bar = binary_dilation(line, iterations=2)
bar[ys[xs.argmin()]:ys[xs.argmin()]+3, :xs.min()+2] = True
top = ys.argmin(); bar[:ys[top]+1, max(0, xs[top]-1):xs[top]+3] = True
l2, _ = label(~bar); plat = (l2 == l2[25, 150]); plat = binary_dilation(plat, iterations=2) | plat
print('inputs', round(time.time()-t0, 1))

# ---------- разворот на 45°: g = y/sin + h*cos/sin  (для 45° это 1,414*y + h)
S = 4; cells = []; Hs = []; Ms = []; Fs = []; Ps = []
xsr = np.arange(W)
for y in range(H-1):
    for s in range(S):
        f = s/S; hv = hh[y]*(1-f) + hh[y+1]*f; src = y if f < 0.5 else y+1
        g = np.round(1.41421*(y+f) + hv).astype(int) + Z0
        cells.append(g*Wc + XW + xsr); Hs.append(hv); Ms.append(mat[src]); Fs.append(forest[src]); Ps.append(plat[src])
cells = np.concatenate(cells); Hs = np.concatenate(Hs).astype(np.float32); Ms = np.concatenate(Ms); Fs = np.concatenate(Fs); Ps = np.concatenate(Ps)
order = np.lexsort((Hs, cells)); cells, Hs, Ms, Fs, Ps = cells[order], Hs[order], Ms[order], Fs[order], Ps[order]
last = np.r_[cells[1:] != cells[:-1], True]       # из совпавших ячеек побеждает более высокий пиксель
cells, Hs, Ms, Fs, Ps = cells[last], Hs[last], Ms[last], Fs[last], Ps[last]
N = Wc*Hc
Hg = np.zeros(N, np.float32); Mg = np.full(N, 2, np.uint8); Fg = np.zeros(N, np.uint8); Pg = np.zeros(N, bool); K = np.zeros(N, bool)
Hg[cells], Mg[cells], Fg[cells], Pg[cells], K[cells] = Hs, Ms, Fs, Ps, True
Hg, Mg, Fg, Pg, K = [a.reshape(Hc, Wc) for a in (Hg, Mg, Fg, Pg, K)]
print('unwrap', round(time.time()-t0, 1), K.sum())

# ---------- заполнение: дыры внутри (тень за горами) и продление наружу
hull = binary_fill_holes(binary_closing(K, structure=np.ones((3, 3), bool), iterations=3))
dk, (iy, ix) = distance_transform_edt(~hull, return_indices=True)
inside_fill = hull & ~K
iy0, ix0 = distance_transform_edt(~K, return_indices=True)[1]
for a in (Hg, Mg, Fg, Pg):
    a[inside_fill] = a[iy0, ix0][inside_fill]
he = Hg[iy, ix]; mE = Mg[iy, ix]; fE = Fg[iy, ix]; pE = Pg[iy, ix]
def noise(sig, shape=(Hc, Wc)):
    n = gaussian_filter(rng.normal(size=shape).astype(np.float32), sig); return n/n.std()
nc = noise(28)
margin = 45 + 0.9*he
tt = np.clip((dk + 28*nc*(dk > 0))/np.maximum(margin, 1), 0, 9)
ss = lambda x: np.clip(x, 0, 1)**2*(3 - 2*np.clip(x, 0, 1))
out = ~hull
# западный берег — полуэллипс вдоль всей западной кромки (вместо прямой полосы)
Xg, Zg = np.meshgrid(np.arange(Wc, dtype=np.float32), np.arange(Hc, dtype=np.float32))
rows = np.where(hull[:, XW + 3])[0]; zmin, zmax = rows.min(), rows.max()
zc = (zmin + zmax)/2; bb = (zmax - zmin)/2*1.12; aW = 230
rr = np.clip((Zg - zc)/bb, -0.999, 0.999)
xcoast = XW - aW*np.sqrt(1 - rr**2)
west = out & (Xg < XW) & (Zg >= zmin) & (Zg <= zmax)
tt = np.where(west, np.clip((XW - Xg)/np.maximum(XW - xcoast, 1) + 0.05*nc, 0, 9), tt)
Hf = np.where(out, he*(1 - ss(tt)), Hg)
Mf = np.where(out, mE, Mg)
Ff = np.where(out, np.where((tt < 0.65) & (noise(10) + (0.75 - tt)*1.6 > 0), fE, 0), Fg)
Pf = np.where(out, pE, Pg)
desert_ext = out & pE                         # продолжение пустыни — только песок (без продолжения оазиса)
Mf = np.where(desert_ext & (Mf == 2), 3, Mf)
Ff = np.where(desert_ext, 0, Ff)
sea = out & (tt >= 1)
print('extend', round(time.time()-t0, 1))

# ---------- пропасть: оба края на одной высоте (разница ≤ 3), дно — вода на уровне моря
T = 24.0                                         # высота краёв пропасти
plat_cells = Pf & (Hf > 1)
Hf = np.where(plat_cells, Hf*(T/52.0), Hf)        # плато (и продолжение пустыни) опущено до T вместе с береговым спуском
dpl = distance_transform_edt(~Pf)
land0 = Hf.copy()
near_ok = (~Pf) & (land0 > 1) & (dpl <= 80)
rim = np.clip(T + 1.2*noise(10), T - 2.5, T + 2.5)
mixw = ss((dpl - 12)/55)
land0 = np.where(near_ok, rim*(1 - mixw) + land0*mixw, land0)     # луг у пропасти подведён к T, дальше плавно возвращается к рельефу
band = (~Pf) & (dpl <= 14) & (Hf > 1)
ramp = ss((dpl - 9)/5)
Hf = np.where(near_ok, land0, Hf)
Hf = np.where(band, land0*ramp, Hf)
Mf = np.where(band & (dpl < 9), 5, np.where(band & (dpl < 14) & (Mf != 5), 0, Mf))
Ff = np.where(band, 0, Ff)

# ---------- тропический лес: полумесяц от северо-востока к центральному востоку, вплотную к горам
pts0 = np.array([(1460, 575), (1620, 520), (1790, 535), (1910, 610), (1955, 760), (1965, 920), (1945, 1080), (1910, 1210)], np.float32)
cum0 = np.r_[0, np.cumsum(np.hypot(*np.diff(pts0, axis=0).T))]; h0 = 20 + 64*np.maximum(np.sin(np.pi*cum0/cum0[-1]), 0)**0.7
un = pts0 - np.array([1500, 850], np.float32); un /= np.hypot(un[:, 0], un[:, 1])[:, None]
pts = (pts0 + 2*h0[:, None]*un).astype(np.float32)     # ширина ×3, внутренний край у гор на месте
seg = np.diff(pts, axis=0); sl = np.hypot(seg[:, 0], seg[:, 1]); cum = np.r_[0, np.cumsum(sl)]
X, Z = np.meshgrid(np.arange(Wc, dtype=np.float32), np.arange(Hc, dtype=np.float32))
bestd = np.full((Hc, Wc), 1e9, np.float32); bests = np.zeros((Hc, Wc), np.float32)
for i in range(len(seg)):
    px, pz = X - pts[i, 0], Z - pts[i, 1]
    t = np.clip((px*seg[i, 0] + pz*seg[i, 1])/(sl[i]**2), 0, 1)
    d = np.hypot(px - t*seg[i, 0], pz - t*seg[i, 1])
    m = d < bestd; bestd = np.where(m, d, bestd); bests = np.where(m, (cum[i] + t*sl[i])/cum[-1], bests)
half = 3*(20 + 64*np.maximum(np.sin(np.pi*np.clip(bests, 0, 1)), 0)**0.7)
isl = bestd < half + 14*noise(20)
land_tot = isl | ((Hf > 0.5) & ~sea)
dsea = distance_transform_edt(land_tot)
HJ = float(os.environ.get('HJ', 61.0))        # ровное плато тропиков
Hj = HJ + 1.5*noise(30)
Hj = Hj - 8*(1 - ss(dsea/50))                  # к обрыву чуть снижается
Hjc = Hj*ss(dsea/12)                          # обрыв только со стороны моря
Hnew = np.where(isl, np.maximum(Hf, Hjc), Hf)
jungle = isl & (Hjc >= Hf - 1) & (Hjc > 1)
Hf = Hnew
Mf = np.where(isl & (dsea < 5), 0, np.where(jungle, 6, Mf))
Ff = np.where(jungle, 3, Ff)
sea = sea & ~isl
# ---------- пустыня на севере: песчаное плато севернее и западнее старого, площадью как у тропиков
dcx, dcz, da, db = 570, 480, 200, 175
dun = ((Xg - dcx)/da)**2 + ((Zg - dcz)/db)**2 + 0.2*noise(40)
blob = (dun < 1) & ~isl
land2 = blob | ((Hf > 0.5) & ~sea)
dsea2 = distance_transform_edt(land2)
Hd = np.clip(T + 2*noise(14) + 3*noise(40), T - 6, T + 6)
apply = blob & ((sea) | (Mf == 3) | (Mf == 2)) & (Mf != 6)
Hf = np.where(apply, np.maximum(Hf, Hd*ss(dsea2/45)), Hf)
Mf = np.where(apply, 3, Mf)
Ff = np.where(apply, 0, Ff)
sea = sea & ~apply
from scipy.ndimage import label as _lab
_l, _ = _lab((Mf == 3) & (Xg < 900) & (Zg < 900))
_c = _l[int(dcz), int(dcx)]
print('desert px', int((_l == _c).sum()), 'tropics px', int(isl.sum()))
print('island px', int(isl.sum()), 'share', round(isl.sum()/N, 4))

# ---------- материалы побережья
land = (Hf > 0.5) & (Mf != 1) & (Mf != 5)
Hf = np.where(sea & ~isl, 0, Hf)
dsea_c = distance_transform_edt(~sea)
shore = out & (~sea) & (dsea_c < 10) & (Hf < 6) & (Mf != 1) & (Mf != 5) & (Mf != 6) & (Mf != 0) & (Mf != 4)
Mf = np.where(shore, 3, Mf)
Mf = np.where((Mf == 0) & (Hf < 30) & out & ~isl, 2, Mf)     # скала у воды — не гора, а луг/пляж
waterish = (Mf == 1) | (Mf == 5) | (sea & ~isl)
dl = distance_transform_edt(~land)
Mf = np.where(waterish & ~isl, np.where(dl < 28, 5, 1), Mf)
Mf = np.where(sea & ~isl & ~land, np.where(dl < 40, 5, 1), Mf)
Hf = np.clip(np.where(waterish & ~isl & (Hf < 0.5), 0, Hf), 0, 250)

# ---------- оазис в пустыне: луг на плато и берег его пруда — зона пальм (id 4 на карте лесов)
oasis_water = Pf & ((Mf == 1) | (Mf == 5))
oasis = (Pf & (Mf == 2)) | (binary_dilation(oasis_water, iterations=16) & Pf & (Mf != 1) & (Mf != 5))
Ff = np.where(oasis & (Hf > 1), 4, Ff).astype(np.uint8)
print('оазис px', int((Ff == 4).sum()))
# ---------- уменьшение вдвое и ограничение перепада высот (соседние блоки — не более 3), кроме обрывов каньона и тропиков
cliffzone = (isl & (dsea < 16)) | band
exempt = binary_dilation(cliffzone, iterations=8) | ((distance_transform_edt(~isl) < 80) & (Hf < 3))   # море у тропиков не ограничивает плато
h2, w2 = Hc//2, Wc//2
def blocks(a): return a[:h2*2, :w2*2].reshape(h2, 2, w2, 2).transpose(0, 2, 1, 3).reshape(h2, w2, 4)
H2 = blocks(Hf).mean(-1)
Mb = blocks(Mf); cnt = np.stack([(Mb == k).sum(-1) for k in range(7)])
M2 = cnt.argmax(0).astype(np.uint8)
wcnt = cnt[1] + cnt[5]
M2 = np.where(wcnt >= 2, np.where(cnt[5] >= cnt[1], 5, 1), M2)
M2 = np.where((wcnt < 2) & (cnt[4] >= 2), 4, M2).astype(np.uint8)
Fb = blocks(Ff); fc = np.stack([(Fb == k).sum(-1) for k in (1, 2, 3, 4)])
F2 = np.where(fc.max(0) >= 2, fc.argmax(0) + 1, 0).astype(np.uint8)
EX = blocks(exempt.astype(np.uint8)).max(-1).astype(bool)

# ---------- правки 2026-10-25: западный берег (извилистый + пляж), полуостров на севере, горы слоями до ~300
HS = 250.0                                         # шкала высот в картах: 0..HS блоков над морем
rg = np.random.default_rng(2610)
def n2(sig, shape=(h2, w2)):
    n = gaussian_filter(rg.normal(size=shape).astype(np.float32), sig); return n/n.std()
Xh, Zh = np.meshgrid(np.arange(w2, dtype=np.float32), np.arange(h2, dtype=np.float32))
from scipy.ndimage import map_coordinates
wgt = ss((Zh - 380)/70)*ss((880 - Zh)/70)*ss((270 - Xh)/150)        # где гнём берег: западная кромка, плавно к краям
dxw = (36*n2(22) + 20*n2(52) + 12*n2(11))*wgt
dzw = 12*n2(30)*wgt
cz_, cx_ = Zh + dzw, Xh + dxw
H2 = map_coordinates(H2, [cz_, cx_], order=1, mode='nearest').astype(np.float32)
M2 = map_coordinates(M2, [cz_, cx_], order=0, mode='nearest').astype(np.uint8)
F2 = map_coordinates(F2, [cz_, cx_], order=0, mode='nearest').astype(np.uint8)
landm = (H2 > 0.3) & (M2 != 1) & (M2 != 5)
from scipy.ndimage import label as _label2
from scipy.ndimage import binary_opening
_wl, _ = _label2(binary_opening(~landm, iterations=6)); seaw = _wl == _wl[500, 5]                      # внешнее море (реки и озёра — не берег)
dws = distance_transform_edt(~seaw)
bw = 13*ss((Zh - 395)/45)*ss((900 - Zh)/70)*ss((330 - Xh)/60)           # ширина пляжа, к краям участка сходит на нет
beach = (M2 != 1) & (M2 != 5) & (dws < bw + 2.5*n2(7)) & (bw > 1.5)
H2 = np.where(beach, np.minimum(H2, 0.35 + 0.5*dws), H2).astype(np.float32)
M2 = np.where(beach, 3, M2).astype(np.uint8); F2 = np.where(beach, 0, F2).astype(np.uint8)
_ll, _nn = _label2((M2 != 1) & (M2 != 5)); _sz = np.bincount(_ll.ravel()); _tiny = (_sz[_ll] < 150) & (_ll > 0) & (Xh < 330)
M2 = np.where(_tiny, 5, M2).astype(np.uint8); H2 = np.where(_tiny, 0, H2).astype(np.float32); F2 = np.where(_tiny, 0, F2).astype(np.uint8)    # мелкие островки от изгиба берега — в воду
# полуостров на правом берегу северной бухты, 10-15 блоков над морем
q = ((Xh - 590)/66)**2 + ((Zh - 250)/98)**2 + 0.28*n2(20)
pen = q < 1
land_all = pen | ((H2 > 0.3) & (M2 != 1) & (M2 != 5))
din = distance_transform_edt(land_all)
Hp = np.clip(12.5 + 1.3*n2(12), 10, 15)*ss(din/14)
H2 = np.where(pen, np.maximum(H2, Hp), H2).astype(np.float32)
newland = pen & (Hp > 0.3) & (M2 != 0)
M2 = np.where(newland, np.where(din < 8, 3, 2), M2).astype(np.uint8)
F2 = np.where(newland, np.where((din > 12) & (n2(9) > 0.2), 1, 0), F2).astype(np.uint8)
dland = distance_transform_edt(~((H2 > 0.3) & (M2 != 1) & (M2 != 5)))
M2 = np.where((M2 == 1) & (dland < 16) & (Xh > 480) & (Xh < 700) & (Zh < 360), 5, M2).astype(np.uint8)
# горы: ярусы — плато 8 блоков, подъём 1:1 на 12 блоков (≈0,8 блока на блок), снова плато; вокруг массива предгорья
TIER = 12.0; SHELF = 8.0; RISE = 15.0; PER = SHELF + RISE; RF = float(os.environ.get('RF', 55.0))
from scipy.ndimage import binary_opening, binary_fill_holes
m0 = binary_opening(M2 == 0, iterations=3)                         # кромка скал вдоль берега тропиков узкая — отсекается
ml, _ = _label2(m0); msz = np.bincount(ml.ravel()); msz[0] = 0
main = binary_fill_holes(binary_dilation(np.isin(ml, np.where(msz >= 600)[0]), iterations=3) & (M2 == 0))
brn = ((M2 == 1) | (M2 == 5)) & (Xh >= 790) & (Xh <= 885) & (Zh >= 400) & (Zh <= 462)      # ветка реки к озеру, заходящая в массив, — убирается
M2 = np.where(brn, 2, M2).astype(np.uint8)
nwb = main & (Xh < 600)                                              # северо-западная глыба: заменяется хребтами (см. ниже)
main = main & ~nwb
dwater = ((M2 == 1) | (M2 == 5)) & ~binary_dilation(main, iterations=2)      # реки внутри массива убираются (остаётся только река-водопад)
src = ~main & ~dwater & ~binary_dilation(nwb, iterations=3) & (H2 > 0.5)                                 # земля вокруг массива — основание ярусов
_, (sy, sx) = distance_transform_edt(~src, return_indices=True)
Bz = gaussian_filter(H2[sy, sx], 8)
d_in = distance_transform_edt(main); d_out = distance_transform_edt(~main)
allowed = ~dwater & (M2 != 6) & (M2 != 3)
sgrid = np.where(main, RF + d_in, np.where(allowed & (d_out < RF), RF - d_out, -1.0))
sw = np.where(sgrid > 0, np.maximum(sgrid + 4*n2(18) + 5*n2(50), 0), 0)
kt = np.floor(sw/PER); gt = TIER*kt + TIER*np.clip((sw - kt*PER - SHELF)/RISE, 0, 1)
wt = ss(sgrid/14)*(~dwater)
H2 = (H2*(1 - wt) + (Bz + gt)*wt).astype(np.float32)
mtn = (wt*gt >= 11.5) | (main & ~dwater)
M2 = np.where(mtn & ~dwater, 0, M2).astype(np.uint8); F2 = np.where(mtn, 0, F2).astype(np.uint8)
# северо-западная глыба -> луг на основании; река внутри неё садится на новый рельеф
rwat = ((M2 == 1) | (M2 == 5)) & binary_dilation(nwb, iterations=2)
H2 = np.where(nwb & ~rwat, Bz, np.where(rwat, np.clip(np.minimum(H2, Bz - 1.5), 0.8, None), H2)).astype(np.float32)
M2 = np.where(nwb & ~rwat, 2, M2).astype(np.uint8); F2 = np.where(nwb, 0, F2).astype(np.uint8)
# хребты на севере: полоса в 40–70 блоках севернее реки (z≈460), не доходящая до обрыва каньона (≥ 13 блоков запаса)
rz = np.full(w2, np.nan)
for xx in range(250, 410):
    zz = np.where(((M2[:, xx] == 5) | (M2[:, xx] == 1)) & (H2[:, xx] > 3) & (np.arange(h2) > 430) & (np.arange(h2) < 480))[0]
    if len(zz): rz[xx] = np.median(zz)
okk = ~np.isnan(rz); rz = np.interp(np.arange(w2), np.where(okk)[0], rz[okk]); rz = gaussian_filter(rz, 6)
dN = rz[None, :] - Zh                                             # расстояние к северу от реки
RWR = 15.0; D0 = 55.0 + 4*n2(25)
# вся область до хребтов и 40–50 блоков перед ними — на уровне лугов (без предгорий и холмов от прежней глыбы)
fz = ss((Xh - 235)/20)*ss((440 - Xh)/20)*ss((dN + 40)/15)*ss((75 - dN)/10)
tgt = 14.0 + 1.5*n2(30)
isw = (M2 == 1) | (M2 == 5)
H2 = np.where(isw, H2*(1 - fz) + (tgt - 1.5)*fz, np.where((M2 != 3), H2*(1 - fz) + tgt*fz, H2)).astype(np.float32)
tap = ss((Xh - 255)/30)*ss((400 - Xh)/30)
sR = np.maximum((RWR - np.abs(dN - D0))*tap, 0) + 3*n2(12)*tap*(np.abs(dN - D0) < RWR)
sR = np.maximum(sR, 0)
kr = np.floor(sR/11.0); gr = 12.0*kr + 12.0*np.clip((sR - kr*11.0 - 2.0)/9.0, 0, 1)
okr = (sR > 0) & (M2 != 1) & (M2 != 5) & (M2 != 3)
H2 = np.where(okr, H2 + gr*ss(sR/4), H2).astype(np.float32)
rm_ = okr & (gr*ss(sR/4) >= 6)
M2 = np.where(rm_, 0, M2).astype(np.uint8); F2 = np.where(rm_, 0, F2).astype(np.uint8)
print('хребты: клеток', int(okr.sum()), 'макс. высота над основанием', float((gr*ss(sR/4)).max().round(1)), 'река z на x=320:', round(float(rz[320]), 1))
print('горы: пик до ограничителя', H2.max().round(1), 'основной массив px', int(main.sum()), 'глубина', d_in.max().round(1))
def lip(S, st=1.0):
    h = S.copy(); xi = np.arange(h.shape[1])*st; zi = (np.arange(h.shape[0])*st)[:, None]
    h = np.minimum.accumulate(h - xi, axis=1) + xi
    g = h[:, ::-1]; g = np.minimum.accumulate(g - xi, axis=1) + xi; h = g[:, ::-1]
    h = np.minimum.accumulate(h - zi, axis=0) + zi
    g = h[::-1, :]; g = np.minimum.accumulate(g - zi, axis=0) + zi; h = g[::-1, :]
    return h
HB = np.array([0, 70, 150, 400.0]); PB = np.array([0, 35, 35 + 80/4, 35 + 80/4 + 250/6])    # φ(h): уклон 2 до 70, 4 до 150, 6 выше
phi = lambda h: np.interp(h, HB, PB)
phinv = lambda p: np.interp(p, PB, HB)
S = np.where(EX, 1e5, phi(H2))
H3 = np.minimum(H2, phinv(np.minimum(phi(H2), lip(S))))
lim = lambda lo: np.where(lo < 70, 2.0, np.where(lo < 150, 4.0, 6.0))
okx_ = ~(EX[:, 1:] | EX[:, :-1]); okz_ = ~(EX[1:, :] | EX[:-1, :])
def strict(H, iters=600):
    for _ in range(iters):
        before = H.copy()
        for ax in (1, 0):
            sl_a = (slice(None), slice(0, -1)) if ax == 1 else (slice(0, -1), slice(None))
            sl_b = (slice(None), slice(1, None)) if ax == 1 else (slice(1, None), slice(None))
            ok = okx_ if ax == 1 else okz_
            lo, hi = H[sl_a], H[sl_b]
            H[sl_b] = np.where(ok, np.minimum(hi, lo + lim(lo)), hi)       # вперёд
            lo, hi = H[sl_b], H[sl_a]
            H[sl_a] = np.where(ok, np.minimum(hi, lo + lim(lo)), hi)       # назад
        if np.array_equal(before, H): break
    return H
H3 = strict(H3.copy())

# ---------- убрать отросток старой реки: её верхний участок у разветвления в горах (z 408..445) — суша, исток теперь ниже по склону
stub = ((M2 == 1) | (M2 == 5)) & (H3 > 3) & (Xh >= 800) & (Xh <= 870) & (Zh >= 408) & (Zh <= 445)
_, (ky, kx) = distance_transform_edt(((M2 == 1) | (M2 == 5)), return_indices=True)
H3 = np.where(stub, H3[ky, kx], H3).astype(np.float32); M2 = np.where(stub, 0, M2).astype(np.uint8)
# ---------- река из гор в тропики и водопад с обрыва: извилистое русло около лучшего хребта, вода только вниз, склоны подгоняются под уклон
from scipy.ndimage import gaussian_filter1d
isl2 = blocks(isl.astype(np.uint8)).max(-1).astype(bool)
XS2 = 1590//2
AMM = float(os.environ.get('AMM', 0.7)); AMJ = float(os.environ.get('AMJ', 1.0))
LH = HJ + 17.0                                         # уровень воды в тоннеле и отверстии над ущельем
LT = HJ - 3.0                                          # уровень воды у выхода из тоннеля и на входе в тропики
xjr = lambda zr: XS2 + int(np.argmax(isl2[zr, XS2:]))
best = None
for zr in range(300, 470, 2):                          # ряд трассы: над тоннелем должна быть высокая гора (≥ LT+9 на 15..75 блоков до тропиков)
    xj2 = xjr(zr)
    if xj2 <= XS2 or xj2 - XS2 > 450: continue
    cl = H3[zr - 5:zr + 6, xj2 - 130:xj2 - 50].min()
    if best is None or cl > best[0]: best = (cl, zr, xj2)
_, ZD2, XJ2s = best
rm = np.random.default_rng(31)
xs = np.arange(XS2, w2)
nn = gaussian_filter1d(rm.normal(size=w2).astype(np.float64), 18)[XS2:]; nn /= nn.std()
amp = (AMM + (AMJ - AMM)*ss((xs - 950)/60))*ss((xs - XS2)/45)*(1 - np.exp(-((xs - 940)/26.0)**2))
zpath = ZD2 + amp*(15*nn + 8*np.sin(2*np.pi*(xs - XS2)/100 + 0.7) + 4*np.sin(2*np.pi*(xs - XS2)/47 + 2.1))
zi_ = np.clip(np.round(zpath).astype(int), 2, h2 - 3)
KJ = int(np.argmax(np.array([isl2[zi_[k], xs[k]] for k in range(len(xs))]))); XJ2 = XS2 + KJ
def gmin(k, rr=4): z = zi_[k]; return H3[z - rr:z + rr + 1, xs[k]].min()
# тоннель: от места, где над трассой есть ≥ 7 блоков породы выше воды (LT+7), на запад до TL блоков подряд
TL = int(os.environ.get('TL', 70))
gm = np.array([gmin(k) for k in range(len(xs))]); good = gm >= LH + 7
k_out = None
for k in range(KJ, TL, -1):                                   # самая восточная точка, перед которой TL клеток подряд породы достаточно
    if good[k - TL:k + 1].all(): k_out = k; break
if k_out is None:
    k_out = KJ
    while k_out > 0 and not good[k_out]: k_out -= 1
k_in = k_out
while k_in > k_out - TL and k_in > 50 and good[k_in - 1]: k_in -= 1
print('тоннель: вход x', xs[k_in], 'выход x', xs[k_out], 'длина', k_out - k_in, 'вход в тропики', XJ2, 'ряд', ZD2)
have_tun = k_out - k_in >= 25
if not have_tun: k_in = k_out = KJ
# ширина русла: 5 в горах, 14..16 в тропиках (радиус 2,2 -> 7,5 ± шум)
nw = gaussian_filter1d(rm.normal(size=len(xs)), 10); nw /= nw.std(); ik = np.arange(len(xs))
rw = 2.2 + 5.4*ss((ik - KJ + 15)/55) + 0.6*nw*ss((ik - KJ)/30)
undw = np.array([H3[zi_[k] - int(rw[k]) - 1:zi_[k] + int(rw[k]) + 2, xs[k]].min() for k in range(len(xs))])
# река выходит из отверстия в отвесной стене ущелья (на LH выше воды джунглей) и водопадом падает в озеро; дальше течёт в тропики и расширяется
Lv = np.full(len(xs), np.nan)
k0 = k_in
Lv[k_in:k_out] = LH
Lp = LT; xend2 = None
for k in range(k_out, len(xs)):
    L = LT if k <= KJ else min(Lp, max(undw[k] - 1.5, 0.6)); Lv[k] = L; Lp = L
    if k > KJ and undw[k] <= 0.6: xend2 = xs[k]; break
if xend2 is None: xend2 = xs[-1]
Lv = Lv[:xend2 - XS2 + 1]
PROF = Lv[:KJ + 1]
GW = float(os.environ.get('GW', 13))                      # половина ширины ущелья
chan = np.zeros(M2.shape, bool); Lgrid = np.full(M2.shape, np.inf)
chan2 = np.zeros(M2.shape, bool); Lgrid2 = np.full(M2.shape, np.inf); gorge = np.zeros(M2.shape, bool)
tun_w = np.zeros(M2.shape, bool); tun_l = np.zeros(M2.shape, bool); tun_z = np.zeros(M2.shape, bool)
H3pre = H3.copy(); xo = int(xs[k_out]); zo = float(zpath[k_out])
for xf in np.arange(xs[k0], xend2 + 0.01, 0.5):
    z = np.interp(xf, xs, zpath); L = np.interp(xf, xs[k0:len(Lv)], Lv[k0:]); r = np.interp(xf, xs, rw)
    intun = have_tun and xs[k_in] + 1 <= xf <= xs[k_out] - 1
    ingor = have_tun and xo <= xf <= xs[KJ] + 4
    R = int(np.ceil(max(r, 3.8))) + 6 if intun else (int(GW) + 2 if ingor else int(np.ceil(r)) + 1)
    for cx in range(int(xf) - R, int(xf) + R + 1):
        for cz in range(int(z) - R, int(z) + R + 1):
            if not (0 <= cz < h2 and 0 <= cx < w2): continue
            dd = np.hypot(cx - xf, cz - z)
            if intun:
                if dd <= 9: tun_z[cz, cx] = True
                if xs[k_in] + 1 <= cx <= xs[k_out] - 1:
                    if dd <= 2.2: tun_w[cz, cx] = True
                    elif dd <= 3.8: tun_l[cz, cx] = True
            elif ingor:
                if dd <= GW and cx >= xo: gorge[cz, cx] = True
                if dd <= r and cx >= xo: chan2[cz, cx] = True; Lgrid2[cz, cx] = min(Lgrid2[cz, cx], L)
            elif dd <= r:
                chan[cz, cx] = True; Lgrid[cz, cx] = min(Lgrid[cz, cx], L)
tun_l &= ~tun_w
M2 = np.where(chan, 5, M2).astype(np.uint8); H3 = np.where(chan, Lgrid, H3).astype(np.float32)
print('река: вход в тропики', XJ2, 'уровень у входа', round(float(PROF[-1]), 1), 'ширина в тропиках', round(2*float(rw[-1]), 1), 'обрыв моря', xend2)
S = np.where(EX, 1e5, phi(H3))
H3 = strict(np.minimum(H3, phinv(np.minimum(phi(H3), lip(S)))))
H3 = strict(np.where(chan & (Xh <= XJ2), np.minimum(H3, Lgrid), H3).astype(np.float32))      # русло — на своём уровне (strict гасит микронарушения от округлений)
H3 = np.where(tun_z, H3pre, H3).astype(np.float32)        # гора над тоннелем не трогается ограничителем
# тоннель: карта верха породы (TUN) — генератор оставляет под ней проход; пол и вода на уровне LH
TUN = np.where(tun_w | tun_l, H3pre, np.nan).astype(np.float32)
H3 = np.where(tun_w, LH, np.where(tun_l, LH + 1.0, H3)).astype(np.float32)
M2 = np.where(tun_w, 5, np.where(tun_l, 0, M2)).astype(np.uint8); F2 = np.where(tun_w | tun_l, 0, F2).astype(np.uint8)
# ущелье: вырез в горе до уровня плато (отвесные стены), озеро у стены и водопад из отверстия
gfloor = HJ + 0.7*n2(25)
cut = gorge & (H3 > gfloor + 1.0)
M2 = np.where(cut, 0, M2).astype(np.uint8); F2 = np.where(gorge, 0, F2).astype(np.uint8)
H3 = np.where(gorge, np.minimum(H3, gfloor), H3).astype(np.float32)
Mid = (Xh >= xo + 1) & (Xh <= xo + 6)
pool = (np.hypot(Xh - (xo + 6), Zh - np.interp(xo + 6, xs, zpath)) <= 6.5) & (Xh >= xo) & gorge
chan2 |= pool; Lgrid2 = np.where(pool, np.minimum(Lgrid2, LT), Lgrid2)
M2 = np.where(chan2, 5, M2).astype(np.uint8); H3 = np.where(chan2, Lgrid2, H3).astype(np.float32)
fall = (Xh >= xo) & (Xh <= xo + 1) & (np.abs(Zh - zo) <= 3.4) & gorge
M2 = np.where(fall, 7, M2).astype(np.uint8); H3 = np.where(fall, LH, H3).astype(np.float32)
FALLD = np.where(fall, LH - (LT - 3.0), 0)
EX = EX | binary_dilation(tun_z | gorge, iterations=2)
print('тоннель: клеток', int((tun_w | tun_l).sum()), 'мин. верх породы', np.nanmin(TUN) if have_tun else None, 'LH', LH, 'ущелье клеток', int(gorge.sum()), 'водопад клеток', int(fall.sum()))
# ---------- второй водопад — там, где ветка реки раньше взбиралась в горы (юго-запад массива, у «треугольника» рек):
# обрыв с зелёным верхом, отверстие в стене, озеро у подножия, тоннель вглубь горы (на север) с тупиком
FW = 8.0; LW = FW + 17.0; LTW = FW - 3.0; TLW = 60
best_w = None
for xc in range(806, 824, 2):
    gz = np.array([H3[z, xc - 4:xc + 5].min() for z in range(h2)]); gd = gz >= LW + 7
    for zp in range(480, 400, -1):
        if gd[zp - TLW:zp + 1].all():
            if best_w is None or zp > best_w[0]: best_w = (zp, xc)
            break
if best_w is not None:
    z_pw, XCW = best_w
    dX = np.abs(Xh - XCW)
    tw_w = (dX <= 2.2) & (Zh >= z_pw - TLW) & (Zh <= z_pw - 1)
    tw_l = (dX <= 3.8) & (Zh >= z_pw - TLW) & (Zh <= z_pw - 1) & ~tw_w
    gorge_w = (dX <= GW) & (Zh >= z_pw) & (Zh <= z_pw + 45)
    TUN = np.where(tw_w | tw_l, H3, TUN).astype(np.float32)
    H3 = np.where(tw_w, LW, np.where(tw_l, LW + 1.0, H3)).astype(np.float32)
    M2 = np.where(tw_w, 5, np.where(tw_l, 0, M2)).astype(np.uint8); F2 = np.where(tw_w | tw_l, 0, F2).astype(np.uint8)
    cut_w = gorge_w & (H3 > FW + 1.0)
    rim_w = binary_dilation(gorge_w | tw_w | tw_l, iterations=7) & ~gorge_w & ~tw_w & ~tw_l & (M2 == 0) & ~((M2 == 5) | (M2 == 1))
    M2 = np.where(rim_w, 2, M2).astype(np.uint8)                                  # зелёный верх обрыва
    M2 = np.where(gorge_w, 2, M2).astype(np.uint8); F2 = np.where(gorge_w, 0, F2).astype(np.uint8)          # дно ущелья — трава
    H3 = np.where(gorge_w, np.minimum(H3, FW + 0.5*n2(20)), H3).astype(np.float32)
    pool_w = (np.hypot(Xh - XCW, Zh - (z_pw + 7)) <= 6.5) & gorge_w
    M2 = np.where(pool_w, 5, M2).astype(np.uint8); H3 = np.where(pool_w, LTW, H3).astype(np.float32)
    fall_w = (Zh >= z_pw) & (Zh <= z_pw + 1) & (dX <= 3.4) & gorge_w
    M2 = np.where(fall_w, 7, M2).astype(np.uint8); H3 = np.where(fall_w, LW, H3).astype(np.float32)
    FALLD = np.where(fall_w, LW - (LTW - 3.0), FALLD)
    # ручей из озера к ближайшей воде реки (вода по линейному уклону вниз)
    wat = ((M2 == 1) | (M2 == 5)) & ~binary_dilation(pool_w | gorge_w, iterations=1) & (H3 < 4)
    dd_, (iy2, ix2) = distance_transform_edt(~wat, return_indices=True)
    py_, px_ = z_pw + 7, XCW
    ty, tx = int(iy2[py_, px_]), int(ix2[py_, px_]); Lt0 = float(H3[ty, tx])
    for t in np.linspace(0, 1, 120):
        zc_, xc_ = py_ + t*(ty - py_), px_ + t*(tx - px_); Lc = LTW + t*(Lt0 - LTW)
        for dz_ in range(-2, 3):
            for dx_ in range(-2, 3):
                cz_, cx_ = int(round(zc_)) + dz_, int(round(xc_)) + dx_
                if np.hypot(dz_, dx_) <= 2.3 and 0 <= cz_ < h2 and 0 <= cx_ < w2 and M2[cz_, cx_] != 7 and not (tw_w | tw_l)[cz_, cx_]:
                    M2[cz_, cx_] = 5; H3[cz_, cx_] = min(H3[cz_, cx_], Lc) if M2[cz_, cx_] == 5 and H3[cz_, cx_] < 0.5 else Lc
    EX = EX | binary_dilation(gorge_w | tw_w | tw_l | rim_w, iterations=2)
    print('запад: портал z', z_pw, 'x', XCW, 'тоннель z', z_pw - TLW, '..', z_pw - 1, 'ущелье клеток', int(gorge_w.sum()), 'водопад клеток', int(fall_w.sum()), 'вода', LW, 'озеро', LTW, 'ручей до', (tx, ty), round(Lt0, 1))
else:
    print('запад: подходящего места для тоннеля нет')
dxx = np.abs(np.diff(H3, axis=1)); dzz = np.abs(np.diff(H3, axis=0))
okx = ~(EX[:, 1:] | EX[:, :-1]); okz = ~(EX[1:, :] | EX[:-1, :])
vx = (dxx > lim(np.minimum(H3[:, 1:], H3[:, :-1])) + 1e-3) & okx; vz = (dzz > lim(np.minimum(H3[1:, :], H3[:-1, :])) + 1e-3) & okz
print('нарушений уклона вне обрывов:', int(vx.sum() + vz.sum()), ' пик до/после:', H2.max().round(1), H3.max().round(1))
Hf, Mf, Ff, Hc, Wc = H3.astype(np.float32), M2, F2, h2, w2

# ---------- вывод
pal = np.array([(128, 128, 128), (30, 80, 220), (40, 170, 40), (240, 220, 60), (140, 90, 40), (80, 160, 240), (20, 100, 40), (200, 240, 255)], np.uint8)
Image.fromarray(pal[Mf]).save(D+'Материалы_сверху.png')
Image.fromarray(np.round(255*(1 - Hf/250)).astype(np.uint8)).save(D+'Высоты_сверху.png')
Image.fromarray((np.array([0, 85, 170, 255, 210], np.uint8))[Ff]).save(D+'Леса_сверху.png')
gy, gx = np.gradient(gaussian_filter(Hf, 1.5)); shade = np.clip(0.62 + (-gx - gy)*0.5, 0.25, 1.15)
rgb = np.clip(pal[Mf].astype(np.float32)*shade[..., None], 0, 255).astype(np.uint8)
Image.fromarray(rgb).resize((Wc//2, Hc//2), Image.BILINEAR).save('island_prev.png')
print('done', round(time.time()-t0, 1), 'max h', Hf.max(), Wc, Hc)

# ---------- данные для мода (индексы, 8 бит): src/main/resources/island/*.png
import os
R = '/home/bogdan/Документы/IdeaProjects/SmesharikiProject/WorldMod/src/main/resources/island/'
os.makedirs(R, exist_ok=True)
solid = (Hf > 0.5) & (Mf != 1) & (Mf != 5) & (Mf != 7)
dist = distance_transform_edt(~solid)
depth = np.where((Mf == 1) | (Mf == 5), np.clip(2 + dist*0.35, 2, 40), 0)
depth = np.where(Mf == 7, FALLD, depth)
Image.fromarray(np.round(65535*(1 - Hf/250)).astype(np.uint16)).save(R+'heights.png')   # 16 бит: шаг 0,003 блока, перепад ≤3 сохраняется после округления
Image.fromarray(np.where(np.isnan(TUN), 65535, np.round(65535*(1 - np.nan_to_num(TUN)/250))).astype(np.uint16)).save(R+'tunnel.png')   # верх породы над тоннелем (65535 — тоннеля нет)
Image.fromarray(Mf.astype(np.uint8), 'L').save(R+'materials.png', optimize=True)
Image.fromarray(Ff.astype(np.uint8), 'L').save(R+'forest.png', optimize=True)
Image.fromarray(np.round(depth).astype(np.uint8), 'L').save(R+'depth.png', optimize=True)
print('data maps saved; centre cell material', Mf[575, 550], 'canvas', Wc, Hc)
