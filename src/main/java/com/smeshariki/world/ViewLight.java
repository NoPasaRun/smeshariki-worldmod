package com.smeshariki.world;

/**
 * Приглушение света мира на время кат-сцены (BehaviorMod; раньше было подсказкой направления в LevelMod). Свет — «отображение», поэтому живёт в моде мира, а не в механиках. Только клиент. {@code strength} — насколько включён режим подсказки (0 — обычный свет, 1 — полностью:
 * свет неба слабее, тени под кронами заметны), {@code darkness} внутри режима — от 0 (светло) до 1 (самое тёмное). Значения ставит тот, кто ведёт подсказку,
 * каждый тик; плавность — на его стороне.
 */
public final class ViewLight {
	private static volatile float strength, darkness;

	private ViewLight() {
	}

	public static void set(float strength, float darkness) {
		ViewLight.strength = Math.clamp(strength, 0f, 1f);
		ViewLight.darkness = Math.clamp(darkness, 0f, 1f);
	}

	public static float strength() {
		return strength;
	}

	public static float darkness() {
		return darkness;
	}
}
