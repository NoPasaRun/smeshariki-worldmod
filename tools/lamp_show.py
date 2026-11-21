#!/usr/bin/env python3
"""Датапак «световое шоу» для поля ламп-переключателей BuildMod под крашеным ковром.

python3 tools/lamp_show.py <папка мира>   — пишет <мир>/datapacks/smeshariki_lamps/ (прежний пак заменяется).

Поле 7x7: лампы на y=81 (x -15..-9, z -17..-11), ковёр над ними на y=82. Лампа ставится `/setblock` сразу
нужным состоянием (facing=up: светится вверх, в ковёр), без редстоуна. Кадр — функция с командами только для
ламп, изменившихся с прошлого кадра, и `schedule` следующего кадра; последний кадр (всё выключено) зовёт первый.
Функции: smeshariki:lamps/start (с начала), smeshariki:lamps/stop (остановить и погасить).
Анимации по порядку: кольца внутрь, крест от центра к углам с поворотом на каждом круге, волна туда-обратно,
шахматы, волны от каждого угла к противоположному, мигание всего поля.
"""
import math, os, shutil, sys

X0, Z0, Y, N = -15, -17, 81, 7
NS = 'smeshariki'
DIR = 'lamps'
C = (N - 1) / 2
ALL = frozenset((i, j) for i in range(N) for j in range(N))

frames = []  # (набор горящих (i, j), тиков до следующего кадра)


def add(lit, ticks):
    frames.append((frozenset(lit), ticks))


# 1. кольца внутрь (по расстоянию от центра, 5 колец: углы, ..., центр), два раза
for _ in range(2):
    for band in range(4, -1, -1):
        add({p for p in ALL if min(4, round(math.hypot(p[0] - C, p[1] - C))) == band}, 4)
add(set(), 4)

# 2. крест от центра к углам, на каждом круге повёрнут на 22,5° (два полных оборота креста: 45° → 225°)
def cell(o):
    """Смещение от центра → целое; половинки к центру, одинаково во все стороны (крест симметричен)."""
    return math.copysign(math.floor(abs(o) + 0.5 - 1e-6), o)


for k in range(8):
    th = math.radians(45 + 22.5 * k)
    for s in range(4):
        lit = set()
        for a in range(4):
            ang = th + a * math.pi / 2
            c, sn = math.cos(ang), math.sin(ang)
            reach = C / max(abs(c), abs(sn))          # до края поля (на диагонали — до угла)
            for t in range(0, 31):
                r = reach * s / 3 * t / 30
                lit.add((int(C + cell(r * c)), int(C + cell(r * sn))))
        add(lit, 3 if s < 3 else 5)
    add(set(), 2)

# 3. волна шириной 2 с запада на восток и обратно, два раза
for _ in range(2):
    for t in list(range(0, N + 1)) + list(range(N - 1, -2, -1)):
        add({p for p in ALL if p[0] in (t, t - 1)}, 3)
add(set(), 4)

# 4. шахматы
for k in range(8):
    add({p for p in ALL if (p[0] + p[1]) % 2 == k % 2}, 6)
add(set(), 4)

# 5. волны от каждого угла к противоположному: СЗ, СВ, ЮВ, ЮЗ
for ci, cj in ((0, 0), (N - 1, 0), (N - 1, N - 1), (0, N - 1)):
    for t in range(0, 2 * N):
        add({p for p in ALL if abs(p[0] - ci) + abs(p[1] - cj) in (t, t - 1)}, 2)
    add(set(), 3)

# 6. мигание всего поля
for _ in range(5):
    add(ALL, 5)
    add(set(), 5)

add(set(), 10)  # пауза и по новой


def lamp(p, on):
    return f'setblock {X0 + p[0]} {Y} {Z0 + p[1]} buildmod:pulse_lamp[facing=up,lit={"true" if on else "false"},powered=false]'


def main():
    world = sys.argv[1]
    root = os.path.join(world, 'datapacks', 'smeshariki_lamps')
    if os.path.exists(root):
        shutil.rmtree(root)
    fn = os.path.join(root, 'data', NS, 'function', DIR)
    os.makedirs(fn)
    with open(os.path.join(root, 'pack.mcmeta'), 'w') as f:
        f.write('{ "pack": { "description": "Световое шоу ламп", "min_format": [107, 1], "max_format": [107, 1] } }\n')
    name = lambda k: f'f{k:03d}'
    prev = frames[-1][0]
    for k, (lit, ticks) in enumerate(frames):
        lines = [lamp(p, p in lit) for p in sorted(lit ^ prev)]
        lines.append(f'schedule function {NS}:{DIR}/{name((k + 1) % len(frames))} {ticks}t replace')
        with open(os.path.join(fn, name(k) + '.mcfunction'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        prev = lit
    stop = [f'schedule clear {NS}:{DIR}/{name(k)}' for k in range(len(frames))]
    stop += [lamp(p, False) for p in sorted(ALL)]
    with open(os.path.join(fn, 'stop.mcfunction'), 'w') as f:
        f.write('\n'.join(stop) + '\n')
    with open(os.path.join(fn, 'start.mcfunction'), 'w') as f:
        f.write(f'function {NS}:{DIR}/stop\nfunction {NS}:{DIR}/{name(0)}\n')
    total = sum(t for _, t in frames)
    print(f'{len(frames)} кадров, круг {total} тиков ({total / 20:.1f} с) → {root}')


if __name__ == '__main__':
    main()
