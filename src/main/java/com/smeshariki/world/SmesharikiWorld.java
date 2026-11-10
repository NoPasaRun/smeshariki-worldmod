package com.smeshariki.world;

import net.fabricmc.api.ModInitializer;
import net.minecraft.core.Registry;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;

public class SmesharikiWorld implements ModInitializer {
	/** Пространство имён оставлено прежним (buildmod): на него ссылаются пресеты и уже созданные миры. */
	public static final String NAMESPACE = "buildmod";

	@Override
	public void onInitialize() {
		Registry.register(BuiltInRegistries.CHUNK_GENERATOR, Identifier.fromNamespaceAndPath(NAMESPACE, "island"), IslandChunkGenerator.CODEC);
		Registry.register(BuiltInRegistries.BIOME_SOURCE, Identifier.fromNamespaceAndPath(NAMESPACE, "island"), IslandBiomeSource.CODEC);
		ShaderPackInstaller.install();
	}
}
