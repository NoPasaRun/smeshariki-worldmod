#version 330 compatibility
#include "/lib/settings.glsl"

// Iris buffer formats: read by Iris from this comment (the GLSL compiler must not see them)
/*
const int colortex3Format = RGBA16F;
const int colortex4Format = RGBA16F;
*/

uniform sampler2D colortex0;
uniform sampler2D colortex3;
uniform sampler2D depthtex0;
uniform mat4 gbufferProjectionInverse;
uniform vec3 sunPosition;
uniform vec3 upPosition;

in vec2 texcoord;

layout(location = 0) out vec4 outColor;

float viewDist(ivec2 px, float depth) {
	vec2 uv = (vec2(px) + 0.5) / vec2(textureSize(depthtex0, 0));
	vec4 v = gbufferProjectionInverse * vec4(vec3(uv, depth) * 2.0 - 1.0, 1.0);
	return -v.z / v.w;
}

// smooth shoulder: untouched below 0.7, bright values fold into 1 instead of burning out
vec3 shoulder(vec3 x) {
	vec3 over = max(x - 0.7, 0.0);
	return min(x, vec3(0.7)) + 0.3 * (1.0 - exp(-over / 0.3));
}

void main() {
	ivec2 px = ivec2(gl_FragCoord.xy);
	vec3 c = texelFetch(colortex0, px, 0).rgb;

#ifdef BLOOM
	c += texture(colortex3, texcoord).rgb * BLOOM_STRENGTH;
#endif

	// soft outline where a near object stands in front of a much farther one (never against the sky)
	if (OUTLINE_STRENGTH > 0.0) {
		float d0 = texelFetch(depthtex0, px, 0).r;
		if (d0 < 1.0) {
			float z0 = viewDist(px, d0);
			float edge = 0.0;
			for (int k = 0; k < 4; k++) {
				ivec2 o = k == 0 ? ivec2(1, 0) : (k == 1 ? ivec2(-1, 0) : (k == 2 ? ivec2(0, 1) : ivec2(0, -1)));
				float dn = texelFetch(depthtex0, px + o, 0).r;
				if (dn < 1.0) {
					float zn = viewDist(px + o, dn);
					edge = max(edge, smoothstep(0.06, 0.2, (zn - z0) / (z0 * 0.1 + 0.5)));
				}
			}
			c *= 1.0 - edge * OUTLINE_STRENGTH * (1.0 - smoothstep(60.0, 160.0, z0));
		}
	}

#ifdef TIME_TINT
	{
		float sunH = dot(normalize(sunPosition), normalize(upPosition));
		float night = 1.0 - smoothstep(-0.30, -0.02, sunH);
		float golden = (1.0 - smoothstep(0.03, 0.30, sunH)) * (1.0 - night);
		c *= mix(vec3(1.0), vec3(1.07, 0.97, 0.86), golden * 0.7 * (1.0 - RAIN_W));
		c *= mix(vec3(1.0), vec3(0.88, 0.95, 1.10), night * 0.6);
	}
#endif

	float luma = dot(c, vec3(0.299, 0.587, 0.114));
	c = mix(vec3(luma), c, mix(SATURATION, 0.78, RAIN_W)); // rain washes the colours out
	c = (c - 0.5) * mix(CONTRAST, 0.97, RAIN_W) + 0.5;
	c *= mix(vec3(1.0 + 0.12 * WARMTH, 1.0 + 0.03 * WARMTH, 1.0 - 0.10 * WARMTH), vec3(0.97, 1.0, 1.04), RAIN_W);
#ifdef TONEMAP
	c = shoulder(max(c, 0.0));
#endif

	vec2 d = (texcoord - 0.5) * vec2(1.0, 0.85);
	c *= 1.0 - VIGNETTE * pow(clamp(length(d) * 1.6, 0.0, 1.0), 2.5);

	outColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
