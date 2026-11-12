# WorldMod

Ландшафт и отображение: типы мира «Остров Смешариков» и «Мир-океан», шейдерпак Iris, яркость неба и поток брёвен на реке.

Часть набора модов «Смешарики» для Minecraft 26.2 (Fabric, Java 25). Общее описание, установка и состав набора — в [корневом README](https://github.com/NoPasaRun/smeshariki-demo#readme).

Типы мира появляются в окне создания мира. Шейдерпак ставится в `shaderpacks/` сам. Данные маршрутов реки лежат в мире (`data/gameplaymod/log_routes.dat`). Свои карты острова можно положить в `data/worldmod/island/`.

## Сборка

```
./gradlew jar --no-daemon -Dorg.gradle.jvmargs=-Xmx900m
```

Результат — `build/libs/worldmod-1.0.0.jar`. Моды компилируются против соседних jar в `../<Мод>/build/libs/`, порядок сборки: UiLib → GameplayMod, ManagementMod → WorldMod → BehaviorMod, BuildMod → LevelMod.

## Лицензия

Код — MIT ([LICENSE](LICENSE)). Это **некоммерческий фан-проект**; всё, что относится к «Смешарикам», лицензией не покрывается ([NOTICE.md](NOTICE.md)).
