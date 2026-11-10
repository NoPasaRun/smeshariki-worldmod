"""Файл маршрутов брёвен GameplayMod в мире: data/gameplaymod/log_routes.dat (NBT, gzip).

Структура как у SavedData: {data: {routes: [{name, nodes: [x, y, z, полуширина, ...]}]}, DataVersion}. Узлы — float.
Писать только при закрытой игре: мод файл не пишет, только читает при запуске мира.
"""
import gzip, os, shutil, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca

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

def write(world, name, nodes, backup=None):
    """Записать или заменить маршрут name; прежний файл, если был, копируется в backup."""
    p = path(world)
    if backup and os.path.exists(p):
        os.makedirs(backup, exist_ok=True)
        shutil.copy2(p, os.path.join(backup, 'log_routes.dat'))
    routes = read(world)
    routes[name] = nodes
    lst = []
    for n, ns in routes.items():
        flat = [struct.pack('>f', float(v)) for node in ns for v in node]
        lst.append({b'name': (8, n.encode()), b'nodes': (9, (5, flat))})
    root = {b'data': (10, {b'routes': (9, (10, lst))}), b'DataVersion': (3, _data_version(world))}
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with gzip.open(p + '.tmp', 'wb') as f:
        f.write(mca.dump(b'', root))
    os.replace(p + '.tmp', p)
