package com.smeshariki.world;

import java.lang.reflect.Method;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.screen.v1.ScreenEvents;
import net.fabricmc.fabric.api.client.screen.v1.Screens;
import net.minecraft.client.gui.components.Button;
import net.minecraft.client.gui.screens.PauseScreen;
import net.minecraft.network.chat.Component;

/**
 * Кнопка в меню паузы (Esc): быстро выключить и включить шейдеры Iris, не заходя в настройки видео.
 * Iris вызывается через рефлексию (без зависимости при сборке): без Iris кнопки нет.
 */
public final class SmesharikiWorldClient implements ClientModInitializer {
	private static final int W = 150, H = 20, MARGIN = 6;

	@Override
	public void onInitializeClient() {
		ScreenEvents.AFTER_INIT.register((client, screen, width, height) -> {
			if (!(screen instanceof PauseScreen) || !Iris.available()) return;
			Button[] holder = new Button[1];
			holder[0] = Button.builder(label(), b -> {
				Iris.setEnabled(!Iris.enabled());
				b.setMessage(label());
			}).bounds(MARGIN, height - H - MARGIN, W, H).build();
			Screens.getWidgets(screen).add(holder[0]);
		});
	}

	private static Component label() {
		return Component.literal(Iris.enabled() ? "Выключить шейдеры" : "Включить шейдеры");
	}

	/** Тонкая обёртка над IrisApi v0: getConfig().areShadersEnabled() / setShadersEnabledAndApply(boolean). */
	private static final class Iris {
		private static Object config;
		private static Method are, set;
		private static boolean tried;

		private static boolean available() {
			if (!tried) {
				tried = true;
				try {
					Object api = Class.forName("net.irisshaders.iris.api.v0.IrisApi").getMethod("getInstance").invoke(null);
					config = api.getClass().getMethod("getConfig").invoke(api);
					Class<?> cfg = Class.forName("net.irisshaders.iris.api.v0.IrisApiConfig");
					are = cfg.getMethod("areShadersEnabled");
					set = cfg.getMethod("setShadersEnabledAndApply", boolean.class);
				} catch (ReflectiveOperationException | LinkageError e) {
					config = null;
				}
			}
			return config != null;
		}

		private static boolean enabled() {
			try {
				return (boolean) are.invoke(config);
			} catch (ReflectiveOperationException e) {
				return false;
			}
		}

		private static void setEnabled(boolean on) {
			try {
				set.invoke(config, on);
			} catch (ReflectiveOperationException e) {
				System.err.println("[smesharikiworld] не удалось переключить шейдеры: " + e);
			}
		}
	}
}
