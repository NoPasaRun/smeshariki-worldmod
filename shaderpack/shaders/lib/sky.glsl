// Own sky palette: soft gradient from zenith to horizon, warm at sunset, dark at night.
// sunH = sin of the sun elevation. Weights: day / golden hour / night.
// In rain (RAIN_W, see settings.glsl) the sky turns into a flat grey-blue overcast: no sun glow, no warm sunset, brighter night (clouds reflect town light).
void skyWeights(float sunH, out float day, out float golden, out float night) {
	night = 1.0 - smoothstep(-0.30, -0.02, sunH);
	golden = (1.0 - smoothstep(0.03, 0.30, sunH)) * (1.0 - night);
	day = max(1.0 - golden - night, 0.0);
}

vec3 skyHorizon(float sunH) {
	float d, g, n;
	skyWeights(sunH, d, g, n);
	vec3 clear = vec3(0.76, 0.87, 0.97) * d + vec3(1.0, 0.74, 0.52) * g + vec3(0.05, 0.07, 0.15) * n;
	vec3 rainy = vec3(0.63, 0.67, 0.71) * d + vec3(0.58, 0.54, 0.54) * g + vec3(0.045, 0.052, 0.07) * n;
	return mix(clear, rainy, RAIN_W);
}

vec3 skyGradient(vec3 dir, float sunH, vec3 sunDirW) {
	float d, g, n;
	skyWeights(sunH, d, g, n);
	vec3 zenith = vec3(0.44, 0.66, 0.93) * d + vec3(0.40, 0.48, 0.78) * g + vec3(0.012, 0.02, 0.06) * n;
	vec3 zenithRain = vec3(0.37, 0.41, 0.47) * d + vec3(0.34, 0.35, 0.40) * g + vec3(0.02, 0.024, 0.036) * n; // darker overhead: thick cloud
	zenith = mix(zenith, zenithRain, RAIN_W);
	float h = clamp((dir.y + 0.08) / 1.08, 0.0, 1.0);
	vec3 col = mix(skyHorizon(sunH), zenith, 1.0 - exp(-mix(3.0, 1.8, RAIN_W) * h)); // no kink at the horizon
	float s = max(dot(dir, sunDirW), 0.0);
	col += vec3(1.0, 0.55, 0.25) * pow(s, 5.0) * g * 0.5 * (1.0 - RAIN_W);   // sunset glow around the sun
	return col;
}
