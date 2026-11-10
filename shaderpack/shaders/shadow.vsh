#version 330 compatibility
#include "/lib/settings.glsl"

out vec2 texcoord;
out vec4 glcolor;

void main() {
	texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
	glcolor = gl_Color;
	gl_Position = ftransform();
#ifdef SHADOWS
	gl_Position.xy = distortShadow(gl_Position.xy);
#else
	gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // shadows off: geometry outside the map, nothing to rasterize
#endif
}
