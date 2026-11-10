#version 330 compatibility
#include "/lib/settings.glsl"

uniform sampler2D colortex4;
uniform float viewWidth;
uniform float viewHeight;

in vec2 texcoord;

/* RENDERTARGETS: 3 */
layout(location = 0) out vec4 outColor;

void main() {
#ifdef BLOOM
	const float w[5] = float[5](0.2270270, 0.1945946, 0.1216216, 0.0540541, 0.0162162);
	vec2 stepv = vec2(0.0, 1.0) * float(BLOOM_RADIUS) / vec2(viewWidth, viewHeight);
	vec3 sum = texture(colortex4, texcoord).rgb * w[0];
	for (int i = 1; i < 5; i++) {
		sum += texture(colortex4, texcoord + stepv * float(i)).rgb * w[i];
		sum += texture(colortex4, texcoord - stepv * float(i)).rgb * w[i];
	}
	outColor = vec4(sum, 1.0);
#else
	outColor = vec4(0.0, 0.0, 0.0, 1.0);
#endif
}
