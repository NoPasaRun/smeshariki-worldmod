package com.smeshariki.world;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.fabricmc.loader.api.FabricLoader;

/**
 * Где лежат данные мода вне мира: {@code <папка игры>/data/worldmod/} (у выделенного сервера — папка сервера). Настройки по-прежнему в {@code config/}, данные
 * мира — внутри мира. Прежние места ({@code config/…} и т. п.) переносятся сюда один раз при первом обращении: {@link #moved}.
 */
public final class ModData {
	private static final Logger LOGGER = LoggerFactory.getLogger("worldmod");

	private ModData() {
	}

	/** Корень данных мода. */
	public static Path dir() {
		return FabricLoader.getInstance().getGameDir().resolve("data").resolve("worldmod");
	}

	/** Прежнее место в папке игры. */
	public static Path game(String... parts) {
		Path p = FabricLoader.getInstance().getGameDir();
		for (String part : parts) p = p.resolve(part);
		return p;
	}

	/** Прежнее место в {@code config/}. */
	public static Path config(String... parts) {
		Path p = FabricLoader.getInstance().getConfigDir();
		for (String part : parts) p = p.resolve(part);
		return p;
	}

	/**
	 * Путь, которым пользоваться: {@code now}. Если там ещё пусто, а на прежнем месте {@code old} что-то лежит, оно переносится (переименованием, без копирования).
	 * Не вышло перенести — работаем по старому пути, ничего не теряется.
	 */
	public static Path moved(Path old, Path now) {
		try {
			if (!Files.exists(now) && Files.exists(old)) {
				Files.createDirectories(now.getParent());
				Files.move(old, now);
				LOGGER.info("Данные перенесены: {} -> {}", old, now);
			}
			return now;
		} catch (IOException | RuntimeException e) {
			LOGGER.warn("Не удалось перенести {} в {}: {}; работаем по старому пути", old, now, e.toString());
			return Files.exists(old) ? old : now;
		}
	}
}
