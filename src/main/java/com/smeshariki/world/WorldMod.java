package com.smeshariki.world;

import net.fabricmc.api.ModInitializer;
import net.minecraft.core.Registry;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;

public class WorldMod implements ModInitializer {
	/** Пространство имён оставлено прежним (buildmod): на него ссылаются пресеты и уже созданные миры. */
	public static final String NAMESPACE = "buildmod";
	/** Идентификатор мода: пакеты и билеты реки. */
	public static final String MOD_ID = "worldmod";

	@Override
	public void onInitialize() {
		Registry.register(BuiltInRegistries.CHUNK_GENERATOR, Identifier.fromNamespaceAndPath(NAMESPACE, "island"), IslandChunkGenerator.CODEC);
		Registry.register(BuiltInRegistries.BIOME_SOURCE, Identifier.fromNamespaceAndPath(NAMESPACE, "island"), IslandBiomeSource.CODEC);
		ShaderPackInstaller.install();
		// река: маршруты и поток брёвен; само бревно (тип сущности, движение) — GameplayMod
		com.smeshariki.gameplay.raft.Logs.setPaths(com.smeshariki.world.river.LogRoutes::get);
		com.smeshariki.world.river.LogRiver.init();
	}
}
