package com.smeshariki.world;

import net.fabricmc.api.EnvType;
import net.fabricmc.loader.api.FabricLoader;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.HexFormat;

/**
 * Кладёт шейдерпак Iris (исходники в shaderpack/, zip собирает Gradle) в папку shaderpacks клиента.
 * Только на клиенте; файл переписывается, когда его содержимое отличается от пака в jar. Запись атомарная
 * (временный файл рядом и переименование), чтобы Iris не прочитал недописанный архив.
 */
final class ShaderPackInstaller {
	private static final String NAME = "Smeshariki_Shaders.zip";

	static void install() {
		if (FabricLoader.getInstance().getEnvironmentType() != EnvType.CLIENT) return;
		try (InputStream in = ShaderPackInstaller.class.getResourceAsStream("/shaderpack/" + NAME)) {
			if (in == null) return;
			byte[] pack = in.readAllBytes();
			Path dir = FabricLoader.getInstance().getGameDir().resolve("shaderpacks");
			Files.createDirectories(dir);
			Path target = dir.resolve(NAME);
			if (Files.exists(target) && sameHash(Files.readAllBytes(target), pack)) return;
			Path tmp = dir.resolve(NAME + ".tmp");
			Files.write(tmp, pack);
			Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
		} catch (Exception e) {
			System.err.println("[smesharikiworld] не удалось установить шейдерпак: " + e);
		}
	}

	private static boolean sameHash(byte[] a, byte[] b) throws Exception {
		MessageDigest md = MessageDigest.getInstance("SHA-256");
		return HexFormat.of().formatHex(md.digest(a)).equals(HexFormat.of().formatHex(md.digest(b)));
	}
}
