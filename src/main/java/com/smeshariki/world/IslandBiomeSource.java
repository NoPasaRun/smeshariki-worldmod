package com.smeshariki.world;

import java.util.List;
import java.util.stream.Stream;

import com.mojang.serialization.MapCodec;
import com.mojang.serialization.codecs.RecordCodecBuilder;

import net.minecraft.core.Holder;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.BiomeSource;
import net.minecraft.world.level.biome.Climate;

/**
 * Биомы острова по картам: порядок списка {@code biomes} в json: 0 равнина, 1 лес, 2 тайга, 3 джунгли, 4 пустыня, 5 пляж, 6 океан,
 * 7 глубокий океан, 8 река, 9 каменистые пики, 10 заснеженные пики.
 */
public final class IslandBiomeSource extends BiomeSource {
	public static final MapCodec<IslandBiomeSource> CODEC = RecordCodecBuilder.mapCodec(i -> i.group(
			Biome.CODEC.listOf().fieldOf("biomes").forGetter(source -> source.biomes)).apply(i, IslandBiomeSource::new));

	private final List<Holder<Biome>> biomes;

	public IslandBiomeSource(List<Holder<Biome>> biomes) {
		this.biomes = biomes;
	}

	@Override
	protected MapCodec<? extends BiomeSource> codec() {
		return CODEC;
	}

	@Override
	protected Stream<Holder<Biome>> collectPossibleBiomes() {
		return biomes.stream();
	}

	@Override
	public Holder<Biome> getNoiseBiome(int qx, int qy, int qz, Climate.Sampler sampler) {
		int i = IslandMaps.index(qx * 4 + 2, qz * 4 + 2);
		float h = IslandMaps.height(i);
		int index;
		switch (IslandMaps.material(i)) {
			case 0 -> index = h >= 90 ? 10 : 9;
			case 1 -> index = h > 0.5f ? 8 : IslandMaps.depth(i) >= 12 ? 7 : 6;
			case 5 -> index = h > 0.5f ? 8 : 6;
			case 3 -> index = h < 4 ? 5 : 4;
			case 6 -> index = 3;
			case 7 -> index = 8;
			default -> {
				int f = IslandMaps.forest(i);
				index = f == 1 ? 1 : f == 2 ? 2 : 0;
			}
		}
		return biomes.get(index);
	}
}
