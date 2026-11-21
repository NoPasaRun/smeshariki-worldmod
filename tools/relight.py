#!/usr/bin/env python3
"""Пересчёт света: в чанках рамки удалить SkyLight/BlockLight и поставить isLightOn=0 — игра посчитает свет при загрузке.

Запуск: relight.py <мир> <x0> <z0> <x1> <z1> [--only-changed <папка region копии>], игра закрыта. С --only-changed трогаются только
чанки, которые отличаются от копии. Нужен после скриптов, которые сбрасывали свет, не снимая isLightOn (до 2026-11-16): чанки темнели.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mca

world = sys.argv[1]; x0, z0, x1, z1 = map(int, sys.argv[2:6])
ref = sys.argv[sys.argv.index('--only-changed') + 1] if '--only-changed' in sys.argv else None
REG = os.path.join(world, 'dimensions/minecraft/overworld/region')
regs, refs, n = {}, {}, 0
for cx in range(x0 >> 4, (x1 >> 4) + 1):
    for cz in range(z0 >> 4, (z1 >> 4) + 1):
        f = f'r.{cx >> 5}.{cz >> 5}.mca'
        if not os.path.exists(os.path.join(REG, f)): continue
        r = regs.setdefault(f, mca.Region(os.path.join(REG, f)))
        raw = r.raw(cx, cz)
        if raw is None: continue
        if ref:
            if f not in refs: refs[f] = mca.Region(os.path.join(ref, f)) if os.path.exists(os.path.join(ref, f)) else None
            if refs[f] is not None and refs[f].raw(cx, cz) == raw: continue
        nm, c = mca.parse(raw)
        c[b'isLightOn'] = (1, b'\x00')
        for sec in c[b'sections'][1][1]:
            sec.pop(b'SkyLight', None); sec.pop(b'BlockLight', None)
        r.put(cx, cz, mca.dump(nm, c)); n += 1
for r in regs.values(): r.save()
print('чанков на пересчёт света:', n)
