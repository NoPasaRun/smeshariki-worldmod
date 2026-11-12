package com.smeshariki.world;

import net.fabricmc.api.ClientModInitializer;

/** Клиентская часть WorldMod: приём маршрутов реки (путь бревна клиент считает сам). */
public final class WorldModClient implements ClientModInitializer {
	@Override
	public void onInitializeClient() {
		net.fabricmc.fabric.api.client.networking.v1.ClientPlayNetworking.registerGlobalReceiver(com.smeshariki.world.river.LogRoutes.Sync.TYPE, (msg, ctx) -> com.smeshariki.world.river.LogRoutes.receive(msg));
		net.fabricmc.fabric.api.client.networking.v1.ClientPlayConnectionEvents.DISCONNECT.register((handler, client) -> com.smeshariki.world.river.LogRoutes.clearClient());
	}
}
