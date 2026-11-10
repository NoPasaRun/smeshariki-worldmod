"""Чтение и запись чанков Anvil (.mca) без зависимостей: NBT разбирается в кортежи (тип, значение), порядок и типы сохраняются байт в байт."""
import struct, zlib, gzip, os, math, time
import numpy as np

class R:
    def __init__(s, b): s.b, s.p = b, 0
    def take(s, n): v = s.b[s.p:s.p+n]; s.p += n; return v
    def i8(s): return struct.unpack('>b', s.take(1))[0]
    def i32(s): return struct.unpack('>i', s.take(4))[0]
    def name(s): n = struct.unpack('>H', s.take(2))[0]; return s.take(n)

def rd(r, t):
    if t == 1: return r.take(1)
    if t == 2: return r.take(2)
    if t == 3: return r.take(4)
    if t == 4: return r.take(8)
    if t == 5: return r.take(4)
    if t == 6: return r.take(8)
    if t == 7: n = r.i32(); return r.take(n)
    if t == 8: return r.name()
    if t == 9:
        e = r.i8(); n = r.i32(); return (e, [rd(r, e) for _ in range(n)])
    if t == 10:
        d = {}
        while True:
            k = r.i8()
            if k == 0: return d
            nm = r.name(); d[nm] = (k, rd(r, k))
    if t == 11: n = r.i32(); return r.take(4*n)
    if t == 12: n = r.i32(); return r.take(8*n)
    raise ValueError(t)

def wr(o, t, v):
    if t in (1, 2, 3, 4, 5, 6): o.append(v)
    elif t == 7: o.append(struct.pack('>i', len(v))); o.append(v)
    elif t == 8: o.append(struct.pack('>H', len(v))); o.append(v)
    elif t == 9:
        o.append(struct.pack('>bi', v[0], len(v[1])))
        for x in v[1]: wr(o, v[0], x)
    elif t == 10:
        for k, (kt, kv) in v.items():
            o.append(struct.pack('>bH', kt, len(k))); o.append(k); wr(o, kt, kv)
        o.append(b'\0')
    elif t == 11: o.append(struct.pack('>i', len(v)//4)); o.append(v)
    elif t == 12: o.append(struct.pack('>i', len(v)//8)); o.append(v)

def parse(b):
    r = R(b); t = r.i8(); nm = r.name(); return nm, rd(r, t)
def dump(nm, v):
    o = [struct.pack('>bH', 10, len(nm)), nm]; wr(o, 10, v); return b''.join(o)

def key(state):
    """Блок палитры -> строка 'name[a=b,c=d]'."""
    n = state[b'Name'][1].decode()
    p = state.get(b'Properties')
    if not p: return n
    return n + '[' + ','.join(f'{k.decode()}={v[1].decode()}' for k, v in sorted(p[1].items())) + ']'
def state(s):
    if '[' not in s: return {b'Name': (8, s.encode())}
    n, a = s[:-1].split('[')
    props = {}
    for kv in a.split(','):
        k, v = kv.split('='); props[k.encode()] = (8, v.encode())
    return {b'Name': (8, n.encode()), b'Properties': (10, props)}

def unpack(data, n):
    """long-массив секции -> 4096 индексов (y<<8|z<<4|x)."""
    bits = max(4, math.ceil(math.log2(n)))
    per = 64//bits
    w = np.frombuffer(data, '>u8').astype(np.uint64)
    idx = np.arange(4096)
    return ((w[idx//per] >> ((idx % per)*bits).astype(np.uint64)) & np.uint64((1 << bits) - 1)).astype(np.int32)
def pack(ids, n):
    bits = max(4, math.ceil(math.log2(n)))
    per = 64//bits
    w = np.zeros(math.ceil(4096/per), np.uint64)
    idx = np.arange(4096)
    np.bitwise_or.at(w, idx//per, ids.astype(np.uint64) << ((idx % per)*bits).astype(np.uint64))
    return w.astype('>u8').tobytes()

class Region:
    def __init__(s, path):
        s.path = path
        with open(path, 'rb') as f: s.buf = bytearray(f.read())
    def present(s, cx, cz): return struct.unpack_from('>I', s.buf, 4*((cx & 31) + (cz & 31)*32))[0] != 0
    def raw(s, cx, cz):
        o = struct.unpack_from('>I', s.buf, 4*((cx & 31) + (cz & 31)*32))[0]
        if not o: return None
        p = (o >> 8)*4096; ln, ct = struct.unpack_from('>IB', s.buf, p)
        d = bytes(s.buf[p+5:p+4+ln])
        if ct == 2: return zlib.decompress(d)
        if ct == 1: return gzip.decompress(d)
        if ct == 3: return d
        raise ValueError(f'сжатие {ct} в {s.path}')
    def put(s, cx, cz, b):
        """Дописать чанк в конец файла (старые сектора просто не используются; файл валиден)."""
        d = zlib.compress(b); pl = struct.pack('>IB', len(d)+1, 2) + d
        secs = math.ceil(len(pl)/4096)
        if secs > 255: raise ValueError('чанк слишком большой')
        pos = math.ceil(len(s.buf)/4096)
        s.buf.extend(b'\0'*(pos*4096 - len(s.buf))); s.buf.extend(pl); s.buf.extend(b'\0'*(secs*4096 - len(pl)))
        i = (cx & 31) + (cz & 31)*32
        struct.pack_into('>I', s.buf, 4*i, (pos << 8) | secs)
        struct.pack_into('>I', s.buf, 4096 + 4*i, int(time.time()))
    def save(s):
        tmp = s.path + '.tmp'
        with open(tmp, 'wb') as f: f.write(s.buf)
        os.replace(tmp, s.path)
