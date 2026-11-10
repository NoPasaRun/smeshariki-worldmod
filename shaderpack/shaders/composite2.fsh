#version 330 compatibility
#include "/lib/settings.glsl"

// Iris buffer formats: read by Iris from this comment (the GLSL compiler must not see them)
/*
const int colortex3Format = RGBA16F;
const int colortex4Format = RGBA16F;
*/

uniform sampler2D colortex0;
uniform sampler2D colortex2;
uniform sampler2D depthtex0;

in vec2 texcoord;

/* RENDERTARGETS: 3 */
layout(location = 0) out vec4 outColor;

void main() {
#ifdef BLOOM
	vec3 c = texture(colortex0, texcoord).rgb;
	float em = texture(colortex2, texcoord).g;
	float luma = dot(c, vec3(0.299, 0.587, 0.114));
	float sun = texture(depthtex0, texcoord).r >= 1.0 ? smoothstep(0.88, 1.0, luma) : 0.0; // sun and moon disc
	outColor = vec4(c * max(em, sun) * 1.5, 1.0);
#else
	outColor = vec4(0.0, 0.0, 0.0, 1.0);
#endif
}
