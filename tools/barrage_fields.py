#!/usr/bin/env python3
"""Дописывает маркерам обстрела (gameplaymod:barrage) параметры строгости, которые до 2026-11-10 были константами и умолчаниями GameplayMod.

Запуск: barrage_fields.py <мир> [--write <папка копии>], игра закрыта (pgrep -af KnotClient). Без --write только считает, что изменит.
Что дописывается, если у маркера поля нет: MinSpeed 5, Grace 10, SlowMargin 10, SlowRecovery 3 (то есть «стрелы только за торможение»,
как было в коде) и Density — прежний профиль по умолчанию. Теперь умолчания мода нейтральные (стрелы всегда, 1/с со всех сторон),
поэтому маркеры погони должны нести свои числа сами. Повторный запуск ничего не меняет. Копии изменённых файлов сущностей — в папку копии.
"""
import os, shutil, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca

FIELDS = {b'MinSpeed': (5, struct.pack('>f', 5.0)), b'Grace': (3, struct.pack('>i', 10)),
          b'SlowMargin': (3, struct.pack('>i', 10)), b'SlowRecovery': (3, struct.pack('>i', 3))}
DENSITY = [0, 0, 0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0.2, 0]   # прежний DEFAULT_DENSITY мода

world = sys.argv[1]
backup = sys.argv[sys.argv.index('--write') + 1] if '--write' in sys.argv else None
dims = ['overworld', 'the_nether', 'the_end']
total = changed_markers = 0
for dim in dims:
    d = os.path.join(world, 'dimensions/minecraft', dim, 'entities')
    if not os.path.isdir(d): continue
    for name in sorted(os.listdir(d)):
        if not name.endswith('.mca'): continue
        path = os.path.join(d, name)
        if os.path.getsize(path) < 8192: continue   # пустой файл области
        region = mca.Region(path)
        touched = False
        for cx in range(32):
            for cz in range(32):
                if not region.present(cx, cz): continue
                nm, c = mca.parse(region.raw(cx, cz))
                dirty = False
                for e in c[b'Entities'][1][1]:
                    if e[b'id'][1] != b'gameplaymod:barrage': continue
                    total += 1
                    add = {k: v for k, v in FIELDS.items() if k not in e}
                    if b'Density' not in e: add[b'Density'] = (9, (5, [struct.pack('>f', float(v)) for v in DENSITY]))
                    if not add: continue
                    for k, v in add.items(): e[k] = v
                    dirty = True; changed_markers += 1
                if dirty and backup:
                    region.put(cx, cz, mca.dump(nm, c)); touched = True
        if touched:
            os.makedirs(os.path.join(backup, dim), exist_ok=True)
            shutil.copy2(path, os.path.join(backup, dim, name))
            region.save()
print(f'маркеров обстрела {total}, нужно дополнить {changed_markers}' + (f'; записано, копии в {backup}' if backup else '; без --write мир не изменён'))
