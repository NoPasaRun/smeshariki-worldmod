#version 330 compatibility
#include "/lib/settings.glsl"
#include "/lib/light.glsl"

uniform sampler2D lightmap;
uniform sampler2D gtexture;
uniform sampler2D normals;
uniform sampler2D specular;
uniform float alphaTestRef;
uniform vec3 shadowLightPosition;
uniform vec3 upPosition;
uniform float sunAngle;

in vec2 texcoord;
in vec2 lmcoord;
in vec4 glcolor;
flat in float emissive;
flat in float isWater;
flat in float sunH;
in vec3 viewPosV;

/* RENDERTARGETS: 0,2,5,6 */
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outId;   // r = water, g = emissive (for bloom)
layout(location = 3) out vec4 outNormal; // view-space face normal for shadows and AO (a = 1: valid)
layout(location = 2) out vec4 outSpec; // sun specular highlight, added in composite with the shadow applied (stored halved)

void main() {
	vec4 albedo = texture(gtexture, texcoord) * glcolor;
	if (albedo.a < alphaTestRef) discard;
	vec3 lm = texture(lightmap, lmcoord).rgb;
	albedo.rgb *= 1.0 - 0.22 * WET_W * smoothstep(0.5, 1.0, lmcoord.y); // wet ground and leaves are darker (only where the rain reaches)
	// face normal from interpolated positions: exact (the depth buffer would be noisy at grazing angles)
	vec3 dp1 = dFdx(viewPosV);
	vec3 dp2 = dFdy(viewPosV);
	vec2 duv1 = dFdx(texcoord);
	vec2 duv2 = dFdy(texcoord);
	vec3 Ng = normalize(cross(dp1, dp2));
	if (dot(Ng, viewPosV) > 0.0) Ng = -Ng; // face the camera
#ifdef COLOR_LIGHT
	lm *= lightTint(lmcoord, sunH);
#endif
	float em = emissive;
	vec3 spec = vec3(0.0);

#ifdef PBR
	{
		vec4 sp = texture(specular, texcoord);       // LabPBR: r smoothness, g reflectance (>= 230: metal), a emission (255: none)
		vec3 nt = texture(normals, texcoord).rgb * 2.0 - 1.0;
		nt.xy *= PBR_BUMP;
#ifdef PBR_FLIP_Y
		nt.y = -nt.y;
#endif
		nt.z = sqrt(max(1.0 - dot(nt.xy, nt.xy), 0.0)); // z is rebuilt: works for packs that store AO or z in blue
		vec3 dp2perp = cross(dp2, Ng);
		vec3 dp1perp = cross(Ng, dp1);
		vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
		vec3 B = -(dp2perp * duv1.y + dp1perp * duv2.y);  // atlas v grows downwards, map +y is up
		float im = max(dot(T, T), dot(B, B));
		vec3 Nm = Ng;
		if (im > 1e-14) Nm = normalize(mat3(T * inversesqrt(im), B * inversesqrt(im), Ng) * nt);
		vec3 L = normalize(shadowLightPosition);
		vec3 V = normalize(-viewPosV);

		float sunVis = smoothstep(0.02, 0.25, dot(L, normalize(upPosition))) * (sunAngle < 0.5 ? 1.0 : 0.15);
		float skyW = smoothstep(0.3, 0.9, lmcoord.y);     // no sun highlights in caves and under roofs
		float k = sunVis * skyW * (1.0 - RAIN_W);

		// relief: the normal map changes how the sun lies on the surface relative to the flat face
		float ndlG = max(dot(Ng, L), 0.0);
		float ndlM = max(dot(Nm, L), 0.0);
		albedo.rgb *= clamp(1.0 + (ndlM - ndlG) * 0.6 * k, 0.6, 1.4);

		// highlight: GGX with Schlick fresnel; smoothness from the pack, F0 4% for non-metals
		float rough = pow(1.0 - sp.r, 2.0);
		float a = max(rough, 0.04);
		float a2 = a * a;
		vec3 H = normalize(L + V);
		float NH = max(dot(Nm, H), 0.0);
		float d = NH * NH * (a2 - 1.0) + 1.0;
		float D = a2 / (3.14159265 * d * d);
		float metal = step(230.0 / 255.0, sp.g);
		vec3 f0 = metal > 0.5 ? albedo.rgb : vec3(max(sp.g * 1.1, 0.04));
		vec3 F = f0 + (1.0 - f0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
		float vis = ndlM * step(0.0, dot(Ng, L));
		spec = min(D * 0.25 * vis, 6.0) * F * k * SPECULAR_STRENGTH * vec3(1.0, 0.95, 0.85);

		if (sp.a < 0.999) em = max(em, sp.a * 255.0 / 254.0); // emission from the pack, if it has any
	}
#endif

	albedo.rgb *= mix(lm, vec3(1.0), em * 0.5);
	outColor = albedo;
	outId = vec4(isWater, em, 0.0, 1.0);
	outSpec = vec4(spec * 0.5, 1.0);
	outNormal = vec4(Ng * 0.5 + 0.5, 1.0);
}
