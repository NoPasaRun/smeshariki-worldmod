#version 330 compatibility
#include "/lib/settings.glsl"

uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;
uniform vec3 cameraPosition;
uniform float frameTimeCounter;
uniform vec3 sunPosition;
uniform vec3 upPosition;

in vec4 mc_Entity;
in vec4 mc_midTexCoord;

out vec2 texcoord;
out vec2 lmcoord;
out vec4 glcolor;
flat out float emissive;
flat out float isWater;
flat out float sunH;
out vec3 viewPosV;

void main() {
	texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
	lmcoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
	lmcoord = lmcoord * 33.05 / 32.0 - 1.05 / 32.0;
	glcolor = gl_Color;
	float id = mc_Entity.x;
	emissive = id == 10002.0 ? 1.0 : 0.0;
	isWater = id == 10001.0 ? 1.0 : 0.0;
	sunH = dot(normalize(sunPosition), normalize(upPosition));

	vec4 vp = gl_ModelViewMatrix * gl_Vertex;
#ifdef WIND
	float sway = 0.0;
	if (id == 10003.0) sway = 0.6;                                     // leaves: the whole block
	else if (id == 10004.0 && texcoord.y < mc_midTexCoord.y) sway = 1.0; // grass and flowers: top only
	if (sway > 0.0) {
		vec3 feet = (gbufferModelViewInverse * vp).xyz;
		vec3 w = feet + cameraPosition;
		float t = frameTimeCounter;
		feet.xz += vec2(sin(t * 1.4 + w.x * 0.6 + w.z * 0.4), sin(t * 1.1 + w.z * 0.7 - w.x * 0.3)) * 0.045 * WIND_STRENGTH * sway;
		vp = gbufferModelView * vec4(feet, 1.0);
	}
#endif
	viewPosV = vp.xyz;
	gl_Position = gl_ProjectionMatrix * vp;
}
