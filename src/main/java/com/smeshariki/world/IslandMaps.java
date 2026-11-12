package com.smeshariki.world;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import javax.imageio.ImageIO;

import net.fabricmc.loader.api.FabricLoader;

/**
 * Карты мира «Остров Смешариков»: пять изображений (одно — верх породы над тоннелем реки) (1 пиксель = 1 блок): высоты (16 бит: 65535 = уровень моря, 0 = +250 блоков),
 * материалы (индексы 0 горы, 1 глубокая вода, 2 луг, 3 песок, 4 тропа, 5 мелководье, 6 джунгли), леса (0 нет, 1 лиственный, 2 хвойный,
 * 3 тропический) и глубина дна. Лежат в jar ({@code /island/*.png}); файлы из {@code data/worldmod/island/} имеют приоритет.
 * Мировая точка (0, 0) соответствует пикселю ({@link #OX}, {@link #OZ}) — середина лугов; за картой — глубокий океан.
 */
public final class IslandMaps {
	public static final int OX = 550, OZ = 575;

	private static short[] heights, tunnel;
	private static byte[] materials, forest, depth;
	private static int width, height;

	private IslandMaps() {
	}

	private static synchronized void ensure() {
		if (heights != null) return;
		try {
			short[] hs = readWide("heights.png"), ts = readWide("tunnel.png");
			byte[] ms = read("materials.png"), fs = read("forest.png"), ds = read("depth.png");
			materials = ms;
			forest = fs;
			depth = ds;
			heights = hs;
			tunnel = ts;
		} catch (IOException e) {
			throw new IllegalStateException("Не удалось загрузить карты острова", e);
		}
	}

	/** 16-битная карта высот: 65535 = уровень моря, 0 = +250 блоков. */
	private static short[] readWide(String name) throws IOException {
		int[] samples = samples(name);
		short[] out = new short[samples.length];
		for (int i = 0; i < out.length; i++) out[i] = (short) samples[i];
		return out;
	}

	private static byte[] read(String name) throws IOException {
		int[] samples = samples(name);
		byte[] out = new byte[samples.length];
		for (int i = 0; i < out.length; i++) out[i] = (byte) samples[i];
		return out;
	}

	private static int[] samples(String name) throws IOException {
		Path override = ModData.moved(ModData.config("buildmod_island"), ModData.dir().resolve("island")).resolve(name);
		try (InputStream in = Files.exists(override) ? Files.newInputStream(override) : IslandMaps.class.getResourceAsStream("/island/" + name)) {
			if (in == null) throw new IOException("нет " + name);
			BufferedImage image = ImageIO.read(in);
			width = image.getWidth();
			height = image.getHeight();
			return image.getRaster().getSamples(0, 0, width, height, 0, (int[]) null);
		}
	}

	/** Индекс пикселя для мировых координат или -1 за пределами карты. */
	public static int index(int wx, int wz) {
		ensure();
		int x = wx + OX, z = wz + OZ;
		if (x < 0 || z < 0 || x >= width || z >= height) return -1;
		return z * width + x;
	}

	/** Высота над уровнем моря в блоках (0..250). */
	public static float height(int i) {
		return i < 0 ? 0 : 250f * (1f - (heights[i] & 0xFFFF) / 65535f);
	}

	/** Верх породы над тоннелем реки (блоков над уровнем моря) или 0, если здесь тоннеля нет; та же шкала, что у высот (65535 — нет). */
	public static float tunnelTop(int i) {
		if (i < 0) return 0;
		int v = tunnel[i] & 0xFFFF;
		return v == 65535 ? 0 : 250f * (1f - v / 65535f);
	}

	public static int material(int i) {
		return i < 0 ? 1 : materials[i] & 255;
	}

	public static int forest(int i) {
		return i < 0 ? 0 : forest[i] & 255;
	}

	public static int depth(int i) {
		return i < 0 ? 40 : depth[i] & 255;
	}
}
