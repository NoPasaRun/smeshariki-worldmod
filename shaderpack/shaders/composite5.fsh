#version 330 compatibility
#include "/lib/settings.glsl"

uniform sampler2D colortex0;
uniform sampler2D depthtex0;
uniform mat4 gbufferProjection;
uniform vec3 sunPosition;
uniform vec3 upPosition;

in vec2 texcoord;

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 outColor;

const float bayer4[16] = float[16](
	0.0, 8.0, 2.0, 10.0,
	12.0, 4.0, 14.0, 6.0,
	3.0, 11.0, 1.0, 9.0,
	15.0, 7.0, 13.0, 5.0);

void main() {
	vec3 color = texture(colortex0, texcoord).rgb;
#ifdef GODRAYS
	if (sunPosition.z < 0.0) {
		float sunH = dot(normalize(sunPosition), normalize(upPosition));
		float vis = smoothstep(0.0, 0.2, sunH);
		vec4 c = gbufferProjection * vec4(sunPosition, 1.0);
		vec2 suv = c.xy / c.w * 0.5 + 0.5;
		vec2 dir = suv - texcoord;
		float len = length(dir);
		ivec2 fp = ivec2(gl_FragCoord.xy) & 3;
		float jitter = (bayer4[fp.x + fp.y * 4] + 0.5) / 16.0;
		float acc = 0.0;
		for (int i = 0; i < 16; i++) {
			vec2 uv = texcoord + dir * ((float(i) + jitter) / 16.0);
			if (uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0)
				acc += texture(depthtex0, uv).r >= 1.0 ? 1.0 : 0.0;
		}
		float falloff = pow(max(1.0 - len / 1.4, 0.0), 2.0);
		float golden = 1.0 - smoothstep(0.1, 0.6, sunH);
		vec3 rayCol = mix(vec3(1.0, 0.92, 0.75), vec3(1.0, 0.65, 0.35), golden);
		color += rayCol * (acc / 16.0) * falloff * vis * GODRAYS_STRENGTH * 0.45;
	}
#endif
	outColor = vec4(color, 1.0);
}
