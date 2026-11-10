#version 330 compatibility
#include "/lib/settings.glsl"

uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D depthtex0;
uniform mat4 gbufferProjectionInverse;

in vec2 texcoord;

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 outColor;

float viewDist(ivec2 px, float depth) {
	vec2 uv = (vec2(px) + 0.5) / vec2(textureSize(depthtex0, 0));
	vec4 v = gbufferProjectionInverse * vec4(vec3(uv, depth) * 2.0 - 1.0, 1.0);
	return -v.z / v.w;
}

void main() {
	ivec2 px = ivec2(gl_FragCoord.xy);
	vec3 color = texelFetch(colortex0, px, 0).rgb;
#ifdef SSAO
	float d0 = texelFetch(depthtex0, px, 0).r;
	if (d0 < 1.0) {
		float z0 = viewDist(px, d0);
		float sum = 0.0;
		float wsum = 0.0;
		// 4x4 window = one full period of the ordered pattern in composite; weighted by depth so edges stay sharp
		for (int j = -2; j <= 1; j++) {
			for (int i = -2; i <= 1; i++) {
				ivec2 q = px + ivec2(i, j);
				float z = viewDist(q, texelFetch(depthtex0, q, 0).r);
				float w = exp(-abs(z - z0) / (0.05 * z0 + 0.05));
				sum += texelFetch(colortex1, q, 0).r * w;
				wsum += w;
			}
		}
		color *= sum / wsum;
	}
#endif
	outColor = vec4(color, 1.0);
}
