#version 330 compatibility
#include "/lib/settings.glsl"
#include "/lib/light.glsl"

uniform sampler2D lightmap;
uniform sampler2D gtexture;
uniform float alphaTestRef;

in vec2 texcoord;
in vec2 lmcoord;
in vec4 glcolor;
flat in float emissive;
flat in float isWater;
flat in float sunH;

/* RENDERTARGETS: 0,2 */
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outId; // r = water

void main() {
	vec4 color = texture(gtexture, texcoord) * glcolor;
	vec3 lm = texture(lightmap, lmcoord).rgb;
#ifdef COLOR_LIGHT
	lm *= lightTint(lmcoord, sunH);
#endif
	lm *= lampGlow(lmcoord, sunH); // overexposure next to lamps
	color.rgb *= lm;
	if (color.a < alphaTestRef) discard;
	if (isWater > 0.5) color.a *= WATER_OPACITY; // clearer water: the bed shows through
	outColor = color;
	outId = vec4(isWater, 0.0, lmcoord.x, 1.0);
}
