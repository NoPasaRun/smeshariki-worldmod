// Colour of light by time of day and block light. sunH = sin of the sun elevation (sunPosition . up).
// In rain the sky light becomes cool, grey and dimmer; light from blocks and covered places (lm.y low) is not touched.
vec3 lightTint(vec2 lm, float sunH) {
	float night = 1.0 - smoothstep(-0.30, -0.02, sunH);
	float golden = (1.0 - smoothstep(0.03, 0.30, sunH)) * (1.0 - night);
	vec3 sky = mix(vec3(1.0), vec3(1.18, 0.92, 0.70), golden * (1.0 - RAIN_W));
	sky = mix(sky, vec3(0.70, 0.82, 1.20), night);
	sky = mix(sky, vec3(0.80, 0.87, 0.96) * 0.86, RAIN_W * (1.0 - night) * smoothstep(0.2, 1.0, lm.y));
	vec3 torch = vec3(1.15, 0.85, 0.55);
	float torchW = smoothstep(0.0, 1.0, lm.x) * (1.0 - lm.y * (1.0 - night) * 0.85);
	return mix(sky, torch, clamp(torchW, 0.0, 1.0));
}

// Overexposure near lamps: multiplier > 1 for strong block light (levels ~12..15), fading out in open daylight.
// Colour above 1 clips to white in the 8-bit target: the texture "burns out" right next to the lamp.
float lampGlow(vec2 lmc, float sunH) {
	float night = 1.0 - smoothstep(-0.30, -0.02, sunH);
	float w = smoothstep(0.70, 0.97, lmc.x);
	w *= w;
	w *= 1.0 - smoothstep(0.5, 1.0, lmc.y) * (1.0 - night) * 0.85;
	return 1.0 + LAMP_GLOW * w;
}
