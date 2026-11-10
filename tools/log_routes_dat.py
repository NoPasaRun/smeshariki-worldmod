"""Файл маршрутов брёвен GameplayMod в мире: data/gameplaymod/log_routes.dat (NBT, gzip).

Структура как у SavedData: {data: {routes: [{name, nodes: [x, y, z, полуширина, ...], wavelength, gap_min, gap_max,
lanes: [{center, phase, shift, sway}]}]}, DataVersion}. Узлы — float. Поток брёвен (ряды, сдвиги, паузы) — настройка реки, она лежит в данных
мира, а не в коде GameplayMod: `DEFAULT_FLOW` ниже — то, что до 2026-11-10 было зашито в `LogRiver` для реки «Ромашковой Долины».
Писать только при закрытой игре: мод файл не пишет, только читает при запуске мира.
"""
import gzip, os, shutil, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import mca

# Поток брёвен «реки» Ромашковой Долины (был в GameplayMod, LogRiver): три ряда — два у берегов и один по средней линии.
DEFAULT_FLOW = {
    'wavelength': 70.0, 'gap_min': 12, 'gap_max': 36,
    'lanes': [
        {'center': -0.7, 'phase': 0.0, 'shift': -2.0, 'sway': 0.3},
        {'center': 0.7, 'phase': math.pi / 2.0, 'shift': 2.0, 'sway': 0.3},
        {'center': 0.0, 'phase': 0.0, 'shift': 0.0, 'sway': 0.0},
    ],
}

def path(world):
    return os.path.join(world, 'data/gameplaymod/log_routes.dat')

def read(world):
    """Маршруты мира: {имя: [[x, y, z, полуширина], ...]}."""
    p = path(world)
    if not os.path.exists(p):
        return {}
    _, root = mca.parse(gzip.open(p).read())
    out = {}
    for r in root[b'data'][1].get(b'routes', (10, (10, [])))[1][1]:
        vals = [struct.unpack('>f', v)[0] for v in r[b'nodes'][1][1]]
        out[r[b'name'][1].decode()] = [vals[i:i + 4] for i in range(0, len(vals) - len(vals) % 4, 4)]
    return out

def _data_version(world):
    _, root = mca.parse(gzip.open(os.path.join(world, 'level.dat')).read())
    return root[b'Data'][1][b'DataVersion'][1]

def _flow_tag(flow):
    lanes = [{k.encode(): (5, struct.pack('>f', float(v))) for k, v in lane.items()} for lane in flow['lanes']]
    return {
        b'wavelength': (5, struct.pack('>f', float(flow['wavelength']))),
        b'gap_min': (3, struct.pack('>i', int(flow['gap_min']))),
        b'gap_max': (3, struct.pack('>i', int(flow['gap_max']))),
        b'lanes': (9, (10, lanes)),
    }

def read_flow(world, name):
    """Поток брёвен маршрута name из файла или None, если в записи его нет (мод тогда пустит один ряд по средней линии)."""
    p = path(world)
    if not os.path.exists(p):
        return None
    _, root = mca.parse(gzip.open(p).read())
    for r in root[b'data'][1].get(b'routes', (10, (10, [])))[1][1]:
        if r[b'name'][1].decode() != name or b'lanes' not in r:
            continue
        f = lambda t: struct.unpack('>f', t[1])[0]
        return {'wavelength': f(r[b'wavelength']), 'gap_min': struct.unpack('>i', r[b'gap_min'][1])[0],
                'gap_max': struct.unpack('>i', r[b'gap_max'][1])[0],
                'lanes': [{k.decode(): struct.unpack('>f', v[1])[0] for k, v in lane.items()} for lane in r[b'lanes'][1][1]]}
    return None

def write(world, name, nodes, backup=None, flow=None):
    """Записать или заменить маршрут name; прежний файл, если был, копируется в backup.

    Поток брёвен: flow, иначе тот, что уже записан у маршрута, иначе DEFAULT_FLOW.
    """
    p = path(world)
    if backup and os.path.exists(p):
        os.makedirs(backup, exist_ok=True)
        shutil.copy2(p, os.path.join(backup, 'log_routes.dat'))
    routes = read(world)
    flows = {n: read_flow(world, n) for n in routes}
    routes[name] = nodes
    flows[name] = flow or flows.get(name) or DEFAULT_FLOW
    lst = []
    for n, ns in routes.items():
        flat = [struct.pack('>f', float(v)) for node in ns for v in node]
        entry = {b'name': (8, n.encode()), b'nodes': (9, (5, flat))}
        if flows.get(n):
            entry.update(_flow_tag(flows[n]))
        lst.append(entry)
    root = {b'data': (10, {b'routes': (9, (10, lst))}), b'DataVersion': (3, _data_version(world))}
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with gzip.open(p + '.tmp', 'wb') as f:
        f.write(mca.dump(b'', root))
    os.replace(p + '.tmp', p)
