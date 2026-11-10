package com.smeshariki.world;

import java.util.List;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Stream;

import com.mojang.serialization.MapCodec;
import com.mojang.serialization.codecs.RecordCodecBuilder;

import net.minecraft.core.BlockPos;
import net.minecraft.core.HolderLookup;
import net.minecraft.server.level.WorldGenRegion;
import net.minecraft.world.level.LevelHeightAccessor;
import net.minecraft.world.level.NoiseColumn;
import net.minecraft.world.level.StructureManager;
import net.minecraft.world.level.WorldGenLevel;
import net.minecraft.world.level.biome.BiomeGenerationSettings;
import net.minecraft.world.level.biome.BiomeManager;
import net.minecraft.world.level.biome.BiomeSource;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.LeavesBlock;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.ChunkGenerator;
import net.minecraft.world.level.chunk.ChunkGeneratorStructureState;
import net.minecraft.world.level.chunk.LevelChunkSection;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.blending.Blender;
import net.minecraft.world.level.levelgen.structure.StructureSet;

/**
 * Генератор мира «Остров Смешариков»: рельеф, материалы, воду и леса берёт из карт {@link IslandMaps}. Сами столбцы строятся в
 * {@link #fillFromNoise}, деревья и трава — в {@link #applyBiomeDecoration} (там можно писать за границу чанка). Структур, пещер и
 * ванильных украшений нет.
 */
public final class IslandChunkGenerator extends ChunkGenerator {
	public static final MapCodec<IslandChunkGenerator> CODEC = RecordCodecBuilder.mapCodec(i -> i.group(
			BiomeSource.CODEC.fieldOf("biome_source").forGetter(generator -> generator.getBiomeSource())).apply(i, IslandChunkGenerator::new));

	public static final int SEA = 62, MIN_Y = -64, DEPTH = 384;

	private static final BlockState STONE = Blocks.STONE.defaultBlockState(), DEEPSLATE = Blocks.DEEPSLATE.defaultBlockState(),
			BEDROCK = Blocks.BEDROCK.defaultBlockState(), GRASS = Blocks.GRASS_BLOCK.defaultBlockState(), DIRT = Blocks.DIRT.defaultBlockState(),
			PATH = Blocks.DIRT_PATH.defaultBlockState(), SAND = Blocks.SAND.defaultBlockState(), SANDSTONE = Blocks.SANDSTONE.defaultBlockState(),
			GRAVEL = Blocks.GRAVEL.defaultBlockState(), SNOW = Blocks.SNOW_BLOCK.defaultBlockState(), ANDESITE = Blocks.ANDESITE.defaultBlockState(), TUFF = Blocks.TUFF.defaultBlockState(), CALCITE = Blocks.CALCITE.defaultBlockState(),
			WATER = Blocks.WATER.defaultBlockState(), AIR = Blocks.AIR.defaultBlockState();

	public IslandChunkGenerator(BiomeSource biomeSource) {
		super(biomeSource, biome -> BiomeGenerationSettings.EMPTY);
	}

	@Override
	protected MapCodec<? extends ChunkGenerator> codec() {
		return CODEC;
	}

	@Override
	public ChunkGeneratorStructureState createState(HolderLookup<StructureSet> structureSets, RandomState randomState, long seed) {
		return ChunkGeneratorStructureState.createForFlat(randomState, seed, getBiomeSource(), Stream.empty());
	}

	/** Что в столбце: верх твёрдого, верх воды (= верх твёрдого, если воды нет), материал поверхности, мелкая ли вода. */
	private record Column(int solidTop, int waterTop, int material, int depth, boolean water, boolean steep, int tunnelTop) {
	}

	/** Верх породы над тоннелем реки (абсолютный y) или 0. */
	private static int tunnel(int i) {
		float t = IslandMaps.tunnelTop(i);
		return t > 0 ? SEA + Math.round(t) : 0;
	}

	/** Верхняя граница прохода в тоннеле: 5 блоков над водой (4 над берегом). */
	private static int roofY(Column c) {
		return c.water() ? c.waterTop() + 5 : c.solidTop() + 4;
	}

	private static Column column(int wx, int wz) {
		int i = IslandMaps.index(wx, wz);
		float h = IslandMaps.height(i);
		int material = IslandMaps.material(i);
		if (material == 1 || material == 5 || material == 7) { // 7 — столб воды водопада: высота столба в карте глубин
			if (h > 0.5f) { // река выше моря: русло глубиной 3
				int top = SEA + Math.round(h);
				return new Column(material == 7 ? top - IslandMaps.depth(i) : top - 3, top, material, 2, true, false, tunnel(i));
			}
			int depth = IslandMaps.depth(i);
			return new Column(SEA - depth, SEA, material, depth, true, false, 0);
		}
		int top = SEA + Math.max(1, Math.round(h));
		boolean steep = false;
		if (material == 0) { // обрыв слоя: снег на нём не держится, виден камень
			float d = Math.max(Math.max(Math.abs(h - IslandMaps.height(IslandMaps.index(wx + 1, wz))), Math.abs(h - IslandMaps.height(IslandMaps.index(wx - 1, wz)))),
					Math.max(Math.abs(h - IslandMaps.height(IslandMaps.index(wx, wz + 1))), Math.abs(h - IslandMaps.height(IslandMaps.index(wx, wz - 1)))));
			steep = d >= 2.5f;
		}
		return new Column(top, top, material, 0, false, steep, tunnel(i));
	}

	/** Снег на горах лежит пятнами и гуще с высотой: ниже SNOW_FROM его нет совсем, к вершинам покрыто около 60%. */
	private static final int SNOW_FROM = 55;

	private static float hash(int x, int z) {
		int h = x * 374761393 + z * 668265263;
		h = (h ^ (h >>> 13)) * 1274126177;
		h ^= h >>> 16;
		return (h & 0xFFFFFF) / 16777216f;
	}

	/** Гладкий шум 0..1 с пятнами около 22 блоков. */
	private static float patch(int x, int z) {
		float fx = x / 22f, fz = z / 22f;
		int ix = (int) Math.floor(fx), iz = (int) Math.floor(fz);
		float tx = fx - ix, tz = fz - iz;
		tx = tx * tx * (3 - 2 * tx);
		tz = tz * tz * (3 - 2 * tz);
		float a = hash(ix, iz) * (1 - tx) + hash(ix + 1, iz) * tx, b = hash(ix, iz + 1) * (1 - tx) + hash(ix + 1, iz + 1) * tx;
		return a * (1 - tz) + b * tz;
	}

	private static boolean snowy(Column c, int wx, int wz) {
		int over = c.solidTop() - SEA;
		if (c.steep() || over < SNOW_FROM) return false;
		float cover = Math.min(0.6f, (over - SNOW_FROM) / 60f * 0.6f);
		return patch(wx, wz) < cover;
	}

	/** Слои горной породы по высоте: полосы по 4 блока, одинаковые у всех столбцов, поэтому на обрывах видны пласты. */
	private static BlockState stratum(int y) {
		int band = Math.floorDiv(y, 4);
		int k = Math.floorMod(band * 7 + (band >> 2) * 5 + (band >> 3) * 3, 11);
		return k < 5 ? STONE : k < 9 ? ANDESITE : TUFF;
	}

	private static BlockState stateAt(Column c, int wx, int wz, int y) {
		if (y == MIN_Y) return BEDROCK;
		if (c.tunnelTop() > 0 && y > roofY(c)) { // порода над тоннелем; у невысоких обрывов (нижний водопад) сверху трава
			if (y > c.tunnelTop()) return AIR;
			boolean green = c.tunnelTop() <= SEA + 70;
			return green ? (y == c.tunnelTop() ? GRASS : y >= c.tunnelTop() - 3 ? DIRT : stratum(y)) : stratum(y);
		}
		if (y > c.solidTop()) return y <= c.waterTop() ? WATER : AIR;
		int d = c.solidTop() - y;
		BlockState deep = y < 0 ? DEEPSLATE : STONE;
		if (c.water()) return d <= 2 ? (c.depth() < 8 && c.waterTop() == SEA ? SAND : GRAVEL) : deep;
		switch (c.material()) {
			case 0:
				if (y < 0) return deep;
				if (d <= 1 && snowy(c, wx, wz)) return SNOW;
				return d <= 8 ? stratum(y) : deep;
			case 3:
				return d <= 3 ? SAND : d <= 6 ? SANDSTONE : deep;
			case 4:
				return d == 0 ? PATH : d <= 3 ? DIRT : deep;
			default: // луга и джунгли
				return d == 0 ? GRASS : d <= 3 ? DIRT : deep;
		}
	}

	@Override
	public CompletableFuture<ChunkAccess> fillFromNoise(Blender blender, RandomState randomState, StructureManager structures, ChunkAccess chunk) {
		Heightmap floor = chunk.getOrCreateHeightmapUnprimed(Heightmap.Types.OCEAN_FLOOR_WG);
		Heightmap surface = chunk.getOrCreateHeightmapUnprimed(Heightmap.Types.WORLD_SURFACE_WG);
		int x0 = chunk.getPos().getMinBlockX(), z0 = chunk.getPos().getMinBlockZ();
		for (int lx = 0; lx < 16; lx++) {
			for (int lz = 0; lz < 16; lz++) {
				Column c = column(x0 + lx, z0 + lz);
				int wx = x0 + lx, wz = z0 + lz;
				int top = Math.max(c.waterTop(), c.tunnelTop());
				LevelChunkSection section = null;
				int sectionIndex = Integer.MIN_VALUE;
				for (int y = MIN_Y; y <= top; y++) {
					int index = chunk.getSectionIndex(y);
					if (index != sectionIndex) {
						section = chunk.getSection(index);
						sectionIndex = index;
					}
					section.setBlockState(lx, y & 15, lz, stateAt(c, wx, wz, y), false);
				}
				int floorTop = c.tunnelTop() > 0 ? c.tunnelTop() : c.solidTop();
				floor.update(lx, floorTop, lz, stateAt(c, wx, wz, floorTop));
				surface.update(lx, top, lz, stateAt(c, wx, wz, top));
			}
		}
		return CompletableFuture.completedFuture(chunk);
	}

	// ---------------------------------------------------------------- деревья и трава

	@Override
	public void applyBiomeDecoration(WorldGenLevel level, ChunkAccess chunk, StructureManager structures) {
		int x0 = chunk.getPos().getMinBlockX(), z0 = chunk.getPos().getMinBlockZ();
		Random rnd = new Random(level.getSeed() ^ (x0 * 341873128712L) ^ (z0 * 132897987541L));
		boolean[][] used = new boolean[16][16];
		BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();
		for (int lx = 0; lx < 16; lx++) {
			for (int lz = 0; lz < 16; lz++) {
				int wx = x0 + lx, wz = z0 + lz;
				int i = IslandMaps.index(wx, wz);
				int material = IslandMaps.material(i);
				int forest = IslandMaps.forest(i);
				float height = IslandMaps.height(i);
				int ground = SEA + Math.max(1, Math.round(height));
				if (forest == 4 && (material == 2 || material == 3)) { // оазис: пальмы и трава
					if (rnd.nextFloat() < 0.035f && free(used, lx, lz)) {
						used[lx][lz] = true;
						palm(level, wx, ground + 1, wz, rnd);
					} else if (material == 2) {
						groundCover(level, pos, wx, ground + 1, wz, rnd, 0.06f, 0.05f);
					}
					continue;
				}
				if (material == 3) { // пустыня: редкие кактусы и сухие кусты
					if (height <= 4) continue;
					float r = rnd.nextFloat();
					if (r < 0.004f && free(used, lx, lz)) {
						used[lx][lz] = true;
						cactus(level, pos, wx, ground + 1, wz, 1 + rnd.nextInt(3));
					} else if (r < 0.0075f) {
						put(level, pos, wx, ground + 1, wz, Blocks.DEAD_BUSH.defaultBlockState(), true);
					}
					continue;
				}
				if (material != 2 && material != 6) continue;
				if (material == 6 || forest == 3) {
					jungle(level, pos, wx, ground + 1, wz, lx, lz, used, rnd);
					continue;
				}
				float density = forest == 2 ? 0.055f : forest == 1 ? 0.06f : 0.003f;
				if (rnd.nextFloat() < density && free(used, lx, lz)) {
					used[lx][lz] = true;
					if (forest == 2) spruce(level, wx, ground + 1, wz, rnd);
					else oak(level, wx, ground + 1, wz, rnd);
					continue;
				}
				float r = rnd.nextFloat();
				if (r < 0.07f) level.setBlock(pos.set(wx, ground + 1, wz), Blocks.SHORT_GRASS.defaultBlockState(), 2);
				else if (r < 0.085f) level.setBlock(pos.set(wx, ground + 1, wz), (rnd.nextBoolean() ? Blocks.OXEYE_DAISY : Blocks.AZURE_BLUET).defaultBlockState(), 2);
			}
		}
	}

	/** Трава и папоротники у земли: высокая трава и большой папоротник — двойные блоки. */
	private static void groundCover(WorldGenLevel level, BlockPos.MutableBlockPos pos, int x, int y, int z, Random rnd, float grass, float fern) {
		float r = rnd.nextFloat();
		if (r < grass * 0.45f) doublePlant(level, pos, x, y, z, Blocks.TALL_GRASS);
		else if (r < grass) put(level, pos, x, y, z, Blocks.SHORT_GRASS.defaultBlockState(), true);
		else if (r < grass + fern * 0.35f) doublePlant(level, pos, x, y, z, Blocks.LARGE_FERN);
		else if (r < grass + fern) put(level, pos, x, y, z, Blocks.FERN.defaultBlockState(), true);
	}

	private static void doublePlant(WorldGenLevel level, BlockPos.MutableBlockPos pos, int x, int y, int z, net.minecraft.world.level.block.Block block) {
		if (!level.getBlockState(pos.set(x, y, z)).isAir() || !level.getBlockState(pos.set(x, y + 1, z)).isAir()) return;
		level.setBlock(pos.set(x, y, z), block.defaultBlockState().setValue(net.minecraft.world.level.block.DoublePlantBlock.HALF, net.minecraft.world.level.block.state.properties.DoubleBlockHalf.LOWER), 2);
		level.setBlock(pos.set(x, y + 1, z), block.defaultBlockState().setValue(net.minecraft.world.level.block.DoublePlantBlock.HALF, net.minecraft.world.level.block.state.properties.DoubleBlockHalf.UPPER), 2);
	}

	private static final net.minecraft.core.Direction[] HORIZONTAL = {net.minecraft.core.Direction.NORTH, net.minecraft.core.Direction.EAST,
			net.minecraft.core.Direction.SOUTH, net.minecraft.core.Direction.WEST};

	/** Лиана в клетке pos, прикреплённая гранью к блоку в направлении {@code toward}. */
	private static void vine(WorldGenLevel level, BlockPos.MutableBlockPos pos, int x, int y, int z, net.minecraft.core.Direction toward) {
		put(level, pos, x, y, z, Blocks.VINE.defaultBlockState().setValue(net.minecraft.world.level.block.VineBlock.PROPERTY_BY_DIRECTION.get(toward), true), true);
	}

	private static void cactus(WorldGenLevel level, BlockPos.MutableBlockPos pos, int x, int y, int z, int height) {
		for (int h = 0; h < height; h++) {
			boolean clear = true;
			for (net.minecraft.core.Direction d : HORIZONTAL) clear &= level.getBlockState(pos.set(x + d.getStepX(), y + h, z + d.getStepZ())).isAir();
			if (!clear) return;
			put(level, pos, x, y + h, z, Blocks.CACTUS.defaultBlockState(), true);
		}
	}

	/** Флора джунглей: деревья (с лианами и какао), редкие огромные деревья 2×2, кусты, бамбук, арбузы, папоротники, трава. */
	private static void jungle(WorldGenLevel level, BlockPos.MutableBlockPos pos, int x, int y, int z, int lx, int lz, boolean[][] used, Random rnd) {
		float r = rnd.nextFloat();
		boolean room = free(used, lx, lz);
		if (r < 0.003f && room && lx < 14 && lz < 14) {
			used[lx][lz] = used[lx + 1][lz] = used[lx][lz + 1] = used[lx + 1][lz + 1] = true;
			megaJungleTree(level, pos, x, y, z, rnd);
			return;
		}
		r -= 0.003f;
		if (r < 0.04f && room) {
			used[lx][lz] = true;
			jungleTree(level, pos, x, y, z, rnd);
			return;
		}
		r -= 0.04f;
		if (r < 0.03f) { // куст: бревно с шапкой из листвы
			put(level, pos, x, y, z, Blocks.JUNGLE_LOG.defaultBlockState(), true);
			BlockState leaves = leaves(Blocks.JUNGLE_LEAVES);
			for (int dx = -2; dx <= 2; dx++)
				for (int dz = -2; dz <= 2; dz++)
					for (int dy = 0; dy <= 1; dy++) {
						if (Math.abs(dx) + Math.abs(dz) + dy > 3 || (dx == 0 && dz == 0 && dy == 0)) continue;
						put(level, pos, x + dx, y + dy + (dx == 0 && dz == 0 ? 1 : 0), z + dz, leaves, true);
					}
			return;
		}
		r -= 0.03f;
		if (r < 0.006f) { // бамбуковая рощица
			int stalks = 4 + rnd.nextInt(5);
			for (int k = 0; k < stalks; k++) {
				int bx = x + rnd.nextInt(5) - 2, bz = z + rnd.nextInt(5) - 2;
				int ground = SEA + Math.max(1, Math.round(IslandMaps.height(IslandMaps.index(bx, bz)))) + 1;
				if (IslandMaps.material(IslandMaps.index(bx, bz)) != 6 && IslandMaps.material(IslandMaps.index(bx, bz)) != 2) continue;
				int top = 6 + rnd.nextInt(9);
				for (int h = 0; h < top; h++) {
					var leaves = h >= top - 2 ? net.minecraft.world.level.block.state.properties.BambooLeaves.LARGE
							: h == top - 3 ? net.minecraft.world.level.block.state.properties.BambooLeaves.SMALL : net.minecraft.world.level.block.state.properties.BambooLeaves.NONE;
					put(level, pos, bx, ground + h, bz, Blocks.BAMBOO.defaultBlockState().setValue(net.minecraft.world.level.block.BambooStalkBlock.LEAVES, leaves), true);
				}
			}
			return;
		}
		r -= 0.006f;
		if (r < 0.004f) {
			put(level, pos, x, y, z, Blocks.MELON.defaultBlockState(), true);
			return;
		}
		groundCover(level, pos, x, y, z, rnd, 0.11f, 0.075f);
	}

	private static void jungleTree(WorldGenLevel level, BlockPos.MutableBlockPos pos, int x, int y, int z, Random rnd) {
		int trunk = 8 + rnd.nextInt(8);
		BlockState log = Blocks.JUNGLE_LOG.defaultBlockState();
		for (int i = 0; i < trunk; i++) put(level, pos, x, y + i, z, log, false);
		BlockState leaves = leaves(Blocks.JUNGLE_LEAVES);
		for (int dy = -2; dy <= 1; dy++) {
			int r = dy < 0 ? 3 : dy == 0 ? 3 : 2;
			for (int dx = -r; dx <= r; dx++)
				for (int dz = -r; dz <= r; dz++) {
					if (dx * dx + dz * dz > r * r + 1) continue;
					if (rnd.nextInt(8) == 0) continue;
					put(level, pos, x + dx, y + trunk + dy, z + dz, leaves, true);
				}
		}
		// лианы по стволу и какао на нём
		for (int i = 1; i < trunk; i++)
			for (net.minecraft.core.Direction d : HORIZONTAL) {
				if (rnd.nextInt(4) == 0) vine(level, pos, x + d.getStepX(), y + i, z + d.getStepZ(), d.getOpposite());
				else if (i > 2 && i < trunk - 2 && rnd.nextInt(28) == 0)
					put(level, pos, x + d.getStepX(), y + i, z + d.getStepZ(), Blocks.COCOA.defaultBlockState()
							.setValue(net.minecraft.world.level.block.CocoaBlock.AGE, rnd.nextInt(3))
							.setValue(net.minecraft.world.level.block.state.properties.BlockStateProperties.HORIZONTAL_FACING, d.getOpposite()), true);
			}
		hangVines(level, pos, x, y + trunk - 2, z, 3, rnd);
	}

	/** Свисающие лианы по краю кроны: у листвы, граничащей с воздухом. */
	private static void hangVines(WorldGenLevel level, BlockPos.MutableBlockPos pos, int x, int y, int z, int radius, Random rnd) {
		for (int dx = -radius - 1; dx <= radius + 1; dx++)
			for (int dz = -radius - 1; dz <= radius + 1; dz++)
				for (net.minecraft.core.Direction d : HORIZONTAL) {
					if (rnd.nextInt(5) != 0) continue;
					int lx = x + dx, lz = z + dz;
					if (!level.getBlockState(pos.set(lx, y, lz)).is(net.minecraft.tags.BlockTags.LEAVES)) continue;
					int vx = lx + d.getStepX(), vz = lz + d.getStepZ();
					int length = 1 + rnd.nextInt(5);
					for (int k = 0; k < length; k++) vine(level, pos, vx, y - k, vz, d.getOpposite());
				}
	}

	/** Огромное джунглевое дерево: ствол 2×2, широкая шапка и лианы. */
	private static void megaJungleTree(WorldGenLevel level, BlockPos.MutableBlockPos pos, int x, int y, int z, Random rnd) {
		int trunk = 20 + rnd.nextInt(10);
		BlockState log = Blocks.JUNGLE_LOG.defaultBlockState();
		for (int i = 0; i < trunk; i++)
			for (int dx = 0; dx <= 1; dx++)
				for (int dz = 0; dz <= 1; dz++) put(level, pos, x + dx, y + i, z + dz, log, false);
		BlockState leaves = leaves(Blocks.JUNGLE_LEAVES);
		int[][] layers = {{-2, 6}, {-1, 6}, {0, 5}, {1, 4}, {2, 2}};
		for (int[] layer : layers) {
			int r = layer[1];
			for (int dx = -r + 1; dx <= r; dx++)
				for (int dz = -r + 1; dz <= r; dz++) {
					double dist = Math.hypot(dx - 0.5, dz - 0.5);
					if (dist > r + 0.3 || (dist > r - 1 && rnd.nextInt(3) == 0)) continue;
					put(level, pos, x + dx, y + trunk + layer[0], z + dz, leaves, true);
				}
		}
		for (int i = 1; i < trunk; i++)
			if (rnd.nextInt(3) == 0) {
				net.minecraft.core.Direction d = HORIZONTAL[rnd.nextInt(4)];
				int vx = x + (d.getStepX() > 0 ? 2 : d.getStepX() < 0 ? -1 : rnd.nextInt(2)), vz = z + (d.getStepZ() > 0 ? 2 : d.getStepZ() < 0 ? -1 : rnd.nextInt(2));
				vine(level, pos, vx, y + i, vz, d.getOpposite());
			}
		hangVines(level, pos, x, y + trunk - 2, z, 6, rnd);
	}

	/** Пальма: наклонный ствол из джунглевых брёвен и шапка из «перьев» листвы, свисающих вниз. */
	private static void palm(WorldGenLevel level, int x, int y, int z, Random rnd) {
		BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();
		int trunk = 7 + rnd.nextInt(4);
		net.minecraft.core.Direction lean = HORIZONTAL[rnd.nextInt(4)];
		BlockState log = Blocks.JUNGLE_LOG.defaultBlockState();
		int tx = x, tz = z;
		for (int i = 0; i < trunk; i++) {
			int off = (int) Math.floor(2.4 * (i / (double) trunk) * (i / (double) trunk));
			tx = x + lean.getStepX() * off;
			tz = z + lean.getStepZ() * off;
			put(level, pos, tx, y + i, tz, log, false);
		}
		int ty = y + trunk - 1;
		BlockState leaves = leaves(Blocks.JUNGLE_LEAVES);
		put(level, pos, tx, ty + 1, tz, leaves, true);
		for (int dx = -1; dx <= 1; dx++)
			for (int dz = -1; dz <= 1; dz++) {
				if (dx == 0 && dz == 0) continue;
				int length = dx != 0 && dz != 0 ? 3 : 4;
				for (int k = 1; k <= length; k++) {
					int drop = k >= 3 ? (k == 3 ? 1 : 2) : 0;
					put(level, pos, tx + dx * k, ty + (k == 1 ? 1 : 0) - drop, tz + dz * k, leaves, true);
				}
			}
	}

	private static boolean free(boolean[][] used, int lx, int lz) {
		for (int dx = -2; dx <= 2; dx++)
			for (int dz = -2; dz <= 2; dz++) {
				int x = lx + dx, z = lz + dz;
				if (x >= 0 && x < 16 && z >= 0 && z < 16 && used[x][z]) return false;
			}
		return true;
	}

	private static void put(WorldGenLevel level, BlockPos.MutableBlockPos pos, int x, int y, int z, BlockState state, boolean onlyAir) {
		pos.set(x, y, z);
		if (onlyAir && !level.getBlockState(pos).isAir()) return;
		level.setBlock(pos, state, 2);
	}

	private static BlockState leaves(net.minecraft.world.level.block.Block block) {
		return block.defaultBlockState().setValue(LeavesBlock.PERSISTENT, true);
	}

	private static void oak(WorldGenLevel level, int x, int y, int z, Random rnd) {
		BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();
		int trunk = 4 + rnd.nextInt(3);
		for (int i = 0; i < trunk; i++) put(level, pos, x, y + i, z, Blocks.OAK_LOG.defaultBlockState(), false);
		BlockState leaves = leaves(Blocks.OAK_LEAVES);
		for (int dy = -2; dy <= 1; dy++) {
			int r = dy < 1 ? 2 : 1;
			for (int dx = -r; dx <= r; dx++)
				for (int dz = -r; dz <= r; dz++) {
					if (Math.abs(dx) == r && Math.abs(dz) == r && rnd.nextBoolean()) continue;
					put(level, pos, x + dx, y + trunk + dy, z + dz, leaves, true);
				}
		}
	}

	private static void spruce(WorldGenLevel level, int x, int y, int z, Random rnd) {
		BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();
		int trunk = 6 + rnd.nextInt(4);
		for (int i = 0; i < trunk; i++) put(level, pos, x, y + i, z, Blocks.SPRUCE_LOG.defaultBlockState(), false);
		BlockState leaves = leaves(Blocks.SPRUCE_LEAVES);
		for (int i = 2; i < trunk - 1; i++) {
			int r = Math.max(0, Math.min(3, (trunk - i) / 2 - 1 + (i % 2 == 0 ? 1 : 0)));
			for (int dx = -r; dx <= r; dx++)
				for (int dz = -r; dz <= r; dz++)
					if (Math.abs(dx) + Math.abs(dz) <= r + 1) put(level, pos, x + dx, y + i, z + dz, leaves, true);
		}
		put(level, pos, x, y + trunk, z, leaves, true);
	}

	// ---------------------------------------------------------------- остальное

	@Override
	public void applyCarvers(WorldGenRegion region, long seed, RandomState randomState, BiomeManager biomes, StructureManager structures, ChunkAccess chunk) {
	}

	@Override
	public void buildSurface(WorldGenRegion region, StructureManager structures, RandomState randomState, ChunkAccess chunk) {
	}

	@Override
	public void spawnOriginalMobs(WorldGenRegion region) {
	}

	@Override
	public int getGenDepth() {
		return DEPTH;
	}

	@Override
	public int getSeaLevel() {
		return SEA;
	}

	@Override
	public int getMinY() {
		return MIN_Y;
	}

	@Override
	public int getBaseHeight(int x, int z, Heightmap.Types type, LevelHeightAccessor level, RandomState randomState) {
		Column c = column(x, z);
		boolean solidOnly = type == Heightmap.Types.OCEAN_FLOOR || type == Heightmap.Types.OCEAN_FLOOR_WG || type == Heightmap.Types.MOTION_BLOCKING_NO_LEAVES;
		return Math.max(solidOnly ? c.solidTop() : c.waterTop(), c.tunnelTop()) + 1;
	}

	@Override
	public NoiseColumn getBaseColumn(int x, int z, LevelHeightAccessor level, RandomState randomState) {
		Column c = column(x, z);
		BlockState[] states = new BlockState[level.getHeight()];
		for (int i = 0; i < states.length; i++) states[i] = stateAt(c, x, z, level.getMinY() + i);
		return new NoiseColumn(level.getMinY(), states);
	}

	@Override
	public void addDebugScreenInfo(List<String> info, RandomState randomState, BlockPos pos) {
	}
}
