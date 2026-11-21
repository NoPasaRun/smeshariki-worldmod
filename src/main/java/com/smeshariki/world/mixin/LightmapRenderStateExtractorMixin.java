package com.smeshariki.world.mixin;

import org.joml.Vector3f;
import org.joml.Vector3fc;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import com.smeshariki.world.ViewLight;

import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.LightmapRenderStateExtractor;
import net.minecraft.client.renderer.state.LightmapRenderState;

/**
 * Делает дневной свет неба ярче: в тенях и под навесами светлее, чем в ванили. Подсказка направления ({@link ViewLight}) идёт в обратную сторону:
 * чем она темнее, тем слабее свет неба и тем глубже тени (ниже и свет неба, и рассеянный свет в тени).
 */
@Mixin(LightmapRenderStateExtractor.class)
public class LightmapRenderStateExtractorMixin {
	/** Во сколько раз усиливается свет неба (1 — как в ванили). */
	@Unique
	private static final float SUN_BOOST = 6.25f;
	/**
	 * Усиление света неба в режиме подсказки: «светло» (бежит к цели) и «самое тёмное» (прочь от цели). Выше единицы всё, что светлее уровня около 5, упирается в максимум
	 * и тени под кронами исчезают (так и устроено «яркое солнце»), поэтому в режиме подсказки свет неба слабее ванильного: под листвой настоящая тень, на просветах светло.
	 */
	@Unique
	private static final float GUIDE_LIGHT = 0.9f, GUIDE_DARK = 0.25f;
	/** Какая доля рассеянного света остаётся в тени в режиме подсказки: при «светло» и при «самом тёмном». */
	@Unique
	private static final float AMBIENT_LIGHT = 0.9f, AMBIENT_DARK = 0.3f;
	/**
	 * Во сколько раз усиливается свет неба в сильный дождь (между этим и {@link #SUN_BOOST} — по уровню дождя). Ваниль в дождь сама снижает свет неба, но шестикратное усиление
	 * заставляет его упереться в максимум и дождь не темнит мир вовсе; в дождь усиление почти снимается, и под открытым небом становится серо и тускло, а в тени всё ещё видно.
	 */
	@Unique
	private static final float RAIN_BOOST = 1.5f;
	/**
	 * Ночь (просьба 2026-11-17: «за 10 блоков ничего не видно или очень смутно»): ночью дневное усиление снимается, свет неба (луна) ослабляется до этой доли ванильного,
	 * рассеянный свет почти гаснет, яркость из настроек (гамма) не действует. Свет от блоков не трогается. Тьму по расстоянию добавляет шейдерпак ({@code NIGHT_VIS}).
	 */
	@Unique
	private static final float NIGHT_SKY = 0.35f, NIGHT_AMBIENT = 0.1f;
	@Unique
	private Float smesharikiworld$vanillaBrightness;
	@Unique
	private float smesharikiworld$lastBrightness = Float.NaN;
	/** Что мы сами записали в {@code skyFactor} и {@code ambientColor} в прошлый раз и из чего: если там то же, значение не пересчитывалось и применять к нему поверх нельзя. */
	@Unique
	private float smesharikiworld$lastBoosted = Float.NaN, smesharikiworld$lastBase;
	@Unique
	private Vector3fc smesharikiworld$lastAmbient, smesharikiworld$lastAmbientBase;
	@Unique
	private float smesharikiworld$lastDarkness = -1;

	@Inject(method = "extract", at = @At("RETURN"))
	private void smesharikiworld$brighterSun(LightmapRenderState state, float partialTick, CallbackInfo ci) {
		float dark = ViewLight.darkness(), strength = ViewLight.strength();
		float rain = 0f;
		var level = Minecraft.getInstance().level;
		float night = 0f;
		if (level != null) {
			rain = Math.max(0f, Math.min(1f, level.getRainLevel(partialTick)));
			if (level.dimensionType().hasSkyLight() && !level.dimensionType().hasCeiling()) night = smesharikiworld$night(level.getOverworldClockTime());
		}
		float key = strength * 2 + dark + rain * 4 + Math.round(night * 50) * 8;
		boolean fresh = state.skyFactor != smesharikiworld$lastBoosted;
		// ваниль пересчитала значения: берём новые за основу; иначе основа — то, из чего мы получили прошлый результат
		float base = fresh ? state.skyFactor : smesharikiworld$lastBase;
		Vector3fc ambient = state.ambientColor == smesharikiworld$lastAmbient && smesharikiworld$lastAmbientBase != null ? smesharikiworld$lastAmbientBase : state.ambientColor;
		if (!state.needsUpdate && !fresh && Math.abs(key - smesharikiworld$lastDarkness) < 0.002f) return;
		if (!state.needsUpdate && fresh) return;
		// множители перемножаются, а не складываются: яркость на глаз меняется кратно, и «светло» с «тёмным» не сливаются
		float guide = (float) (GUIDE_LIGHT * Math.pow(GUIDE_DARK / GUIDE_LIGHT, dark));
		float sun = (float) (SUN_BOOST * Math.pow(RAIN_BOOST / SUN_BOOST, rain)); // в дождь дневное усиление слабее
		sun = (float) (sun * Math.pow(NIGHT_SKY / sun, night)); // ночью вместо усиления — ослабление
		float boost = (float) (sun * Math.pow(guide / sun, strength));
		state.skyFactor = base * boost;
		float shade = AMBIENT_LIGHT + (AMBIENT_DARK - AMBIENT_LIGHT) * dark;
		float ambMul = (1f + (shade - 1f) * strength) * (1f + (NIGHT_AMBIENT - 1f) * night);
		Vector3fc dimmed = ambMul != 1f ? new Vector3f(ambient).mul(ambMul) : ambient;
		// гамма из настроек: ночью снимается (иначе «яркость» вытягивает тьму); своё значение отличаем от ванильного, чтобы не уменьшать повторно
		if (state.brightness != smesharikiworld$lastBrightness) smesharikiworld$vanillaBrightness = state.brightness;
		if (smesharikiworld$vanillaBrightness != null) state.brightness = smesharikiworld$vanillaBrightness * (1f - night);
		smesharikiworld$lastBrightness = state.brightness;
		state.ambientColor = dimmed;
		// изменилось только наше затемнение: ваниль сама текстуру не обновит
		state.needsUpdate = true;
		smesharikiworld$lastBase = base;
		smesharikiworld$lastBoosted = state.skyFactor;
		smesharikiworld$lastAmbientBase = ambient;
		smesharikiworld$lastAmbient = dimmed;
		smesharikiworld$lastDarkness = key;
	}

	/** Доля ночи по времени суток: 0 днём, 1 глубокой ночью; переход в сумерках (солнце от 3° над горизонтом до 15° под ним). */
	@Unique
	private static float smesharikiworld$night(long clock) {
		double sunH = Math.sin(2 * Math.PI * Math.floorMod(clock, 24000L) / 24000.0); // 0 — восход, 6000 — полдень, 18000 — полночь
		double t = Math.max(0, Math.min(1, (sunH + 0.25) / 0.30));
		return (float) (1 - t * t * (3 - 2 * t));
	}
}
