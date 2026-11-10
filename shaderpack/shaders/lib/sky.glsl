// Own sky palette: soft gradient from zenith to horizon, warm at sunset, dark at night.
// sunH = sin of the sun elevation. Weights: day / golden hour / night.
void skyWeights(float sunH, out float day, out float golden, out float night) {
	night = 1.0 - smoothstep(-0.30, -0.02, sunH);
	golden = (1.0 - smoothstep(0.03, 0.30, sunH)) * (1.0 - night);
	day = max(1.0 - golden - night, 0.0);
}

vec3 skyHorizon(float sunH) {
	float d, g, n;
	skyWeights(sunH, d, g, n);
	return vec3(0.76, 0.87, 0.97) * d + vec3(1.0, 0.74, 0.52) * g + vec3(0.05, 0.07, 0.15) * n;
}

vec3 skyGradient(vec3 dir, float sunH, vec3 sunDirW) {
	float d, g, n;
	skyWeights(sunH, d, g, n);
	vec3 zenith = vec3(0.44, 0.66, 0.93) * d + vec3(0.40, 0.48, 0.78) * g + vec3(0.012, 0.02, 0.06) * n;
	float h = clamp((dir.y + 0.08) / 1.08, 0.0, 1.0);
	vec3 col = mix(skyHorizon(sunH), zenith, 1.0 - exp(-3.0 * h)); // no kink at the horizon
	float s = max(dot(dir, sunDirW), 0.0);
	col += vec3(1.0, 0.55, 0.25) * pow(s, 5.0) * g * 0.5;   // sunset glow around the sun
	return col;
}
