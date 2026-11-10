#version 330 compatibility
#include "/lib/settings.glsl"
#include "/lib/sky.glsl"

uniform sampler2D colortex0;
uniform sampler2D depthtex0;
uniform sampler2D depthtex1;
uniform sampler2D colortex2;
uniform sampler2D colortex5;
uniform sampler2D colortex6;
uniform sampler2D shadowtex0;

uniform mat4 gbufferProjection;
uniform mat4 gbufferModelView;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;
uniform vec3 shadowLightPosition;
uniform vec3 upPosition;
uniform vec3 fogColor;
uniform vec3 skyColor;
uniform vec3 sunPosition;
uniform vec3 cameraPosition;
uniform float frameTimeCounter;
uniform float sunAngle;
uniform float viewWidth;
uniform float viewHeight;
uniform int isEyeInWater;

in vec2 texcoord;

/* RENDERTARGETS: 0,1 */
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outAO; // AO is blurred and applied in composite1

const float bayer4[16] = float[16](
	0.0, 8.0, 2.0, 10.0,
	12.0, 4.0, 14.0, 6.0,
	3.0, 11.0, 1.0, 9.0,
	15.0, 7.0, 13.0, 5.0);

vec3 viewPos(vec2 uv, float depth) {
	vec4 v = gbufferProjectionInverse * vec4(vec3(uv, depth) * 2.0 - 1.0, 1.0);
	return v.xyz / v.w;
}

float viewDist(vec2 uv, float depth) {
	return -viewPos(uv, depth).z;
}

float waveH(vec2 p, float t) {
	return sin(p.x * 0.9 + t * 1.3) * 0.5 + sin(p.y * 1.2 - t * 1.1) * 0.4 + sin((p.x + p.y) * 0.5 + t * 0.7) * 0.6
		+ sin((p.x - p.y) * 2.1 + t * 2.0) * 0.15;
}

// ---------- clouds: a layer of rounded blocks, ray-marched cell by cell (only over sky pixels)
const float CELL = 16.0;
const float CR = 2.4; // corner radius

float hash21(vec2 p) {
	p = fract(p * vec2(123.34, 456.21));
	p += dot(p, p + 45.32);
	return fract(p.x * p.y);
}

float vnoise(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), f.x), mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), f.x), f.y);
}

float cellDensity(vec2 c) {
	return 0.65 * vnoise(c * 0.21 + vec2(3.1, 7.7)) + 0.35 * vnoise(c * 0.53 + 11.0);
}

// box of one cell; sides facing a present neighbour are pushed CR into it, so adjacent blocks merge without a groove
float sdCell(vec3 q, vec3 b, vec4 ext) { // ext = (-x, +x, -z, +z) extension
	vec3 ctr = vec3((ext.y - ext.x) * 0.5, 0.0, (ext.w - ext.z) * 0.5);
	vec3 bb = b + vec3((ext.x + ext.y) * 0.5, 0.0, (ext.z + ext.w) * 0.5);
	vec3 e = max(abs(q - ctr) - (bb - CR), 0.0);
	return length(e) - CR;
}

vec3 cloudEdgeColor(float sunH) {
	float dd, gg, nn;
	skyWeights(sunH, dd, gg, nn);
	vec3 col = vec3(0.86, 0.89, 0.95);
	col = mix(col, col * vec3(1.2, 0.80, 0.55), gg * 0.8);
	return mix(col, col * vec3(0.30, 0.36, 0.55), nn);
}

vec3 addClouds(vec3 sky, vec3 dir, float sunH, vec3 Lw) {
	float yb = CLOUD_HEIGHT;
	float camY = cameraPosition.y;
	if (dir.y < 0.02 || camY > yb - 2.0) return sky;
	float t0 = (yb - camY) / dir.y;
	float t1 = (yb + 16.0 - camY) / dir.y;
	vec2 wind = vec2(frameTimeCounter * 2.0, frameTimeCounter * 0.7);
	vec3 ro = vec3(cameraPosition.x - wind.x, camY, cameraPosition.z - wind.y);
	float th = 0.74 - CLOUD_COVERAGE * 0.5;

	vec2 p = ro.xz + dir.xz * t0;
	vec2 cell = floor(p / CELL);
	vec2 stp = vec2(dir.x >= 0.0 ? 1.0 : -1.0, dir.z >= 0.0 ? 1.0 : -1.0);
	vec2 ad = max(abs(dir.xz), vec2(1e-5));
	vec2 tn = t0 + vec2(dir.x >= 0.0 ? (cell.x + 1.0) * CELL - p.x : p.x - cell.x * CELL,
	                    dir.z >= 0.0 ? (cell.y + 1.0) * CELL - p.y : p.y - cell.y * CELL) / ad;
	vec2 td = vec2(CELL) / ad;
	float tcur = t0;

	float edgeA = 0.0;
	for (int i = 0; i < 28; i++) {
		float tex = min(min(tn.x, tn.y), t1);
		float dn = cellDensity(cell);
		if (dn > th) {
			float hh = 2.6 + 2.5 * hash21(cell + 3.3) + 3.5 * smoothstep(th, th + 0.2, dn);
			vec3 ctr = vec3((cell.x + 0.5) * CELL, yb + hh, (cell.y + 0.5) * CELL);
			vec3 b = vec3(CELL * 0.5, hh, CELL * 0.5);
			vec4 ext = vec4(cellDensity(cell + vec2(-1.0, 0.0)) > th ? CR : 0.0, cellDensity(cell + vec2(1.0, 0.0)) > th ? CR : 0.0,
			                cellDensity(cell + vec2(0.0, -1.0)) > th ? CR : 0.0, cellDensity(cell + vec2(0.0, 1.0)) > th ? CR : 0.0);
			float t = tcur;
			float dmin = 1e9;
			bool hit = false;
			float eps = max(0.04, 0.0012 * t);
			for (int k = 0; k < 10; k++) {
				float d = sdCell(ro + dir * t - ctr, b, ext);
				dmin = min(dmin, d);
				if (d < eps) { hit = true; break; }
				t += max(d, 0.05);
				if (t > tex + CR + 1.0) break;
			}
			if (hit) {
				vec3 pos = ro + dir * t - ctr;
				vec2 e = vec2(0.06, 0.0);
				vec3 n = normalize(vec3(sdCell(pos + e.xyy, b, ext) - sdCell(pos - e.xyy, b, ext),
				                        sdCell(pos + e.yxy, b, ext) - sdCell(pos - e.yxy, b, ext),
				                        sdCell(pos + e.yyx, b, ext) - sdCell(pos - e.yyx, b, ext)));
				float dif = clamp(dot(n, Lw) * 0.5 + 0.5, 0.0, 1.0);
				// undersides are what you mostly see: light, with a soft difference between blocks; tops catch the sun
				vec3 under = vec3(0.95, 0.97, 1.0) * mix(0.88, 1.0, dif);
				under *= 1.0 - 0.07 * clamp((hh - 2.6) / 6.0, 0.0, 1.0);
				under *= 0.95 + 0.07 * hash21(cell + 9.1);
				vec3 col = mix(under, vec3(1.0, 0.985, 0.96), smoothstep(0.25, 0.9, n.y));
				col *= mix(0.88, 1.0, smoothstep(0.0, hh * 1.4, pos.y + hh));
				float dd, gg, nn;
				skyWeights(sunH, dd, gg, nn);
				col = mix(col, col * vec3(1.2, 0.80, 0.55), gg * 0.8);
				col = mix(col, col * vec3(0.30, 0.36, 0.55), nn);
				col = mix(col, skyHorizon(sunH), smoothstep(160.0, 460.0, t));
				float a = 1.0 - smoothstep(300.0, 470.0, t);
				return mix(mix(sky, col, a), cloudEdgeColor(sunH), edgeA * a);
			}
			// near miss: soft antialiased edge
			edgeA = max(edgeA, 1.0 - smoothstep(eps, eps * 3.0, dmin));
		}
		if (tex >= t1) break;
		if (tn.x < tn.y) { cell.x += stp.x; tcur = tn.x; tn.x += td.x; }
		else { cell.y += stp.y; tcur = tn.y; tn.y += td.y; }
		if (tcur > 470.0) break;
	}
	return mix(sky, cloudEdgeColor(sunH), edgeA * (1.0 - smoothstep(300.0, 470.0, tcur)));
}

void main() {
	float aoOut = 1.0;
	vec3 color = texture(colortex0, texcoord).rgb;
	float depth = texture(depthtex0, texcoord).r;
	vec3 gn = cross(dFdx(viewPos(texcoord, depth)), dFdy(viewPos(texcoord, depth))); // fallback normal (entities, hand); derivatives must be outside branches
	vec4 nTex = texture(colortex6, texcoord);
	vec3 faceN = nTex.a > 0.5 ? normalize(nTex.rgb * 2.0 - 1.0) : normalize(gn);
	if (dot(faceN, viewPos(texcoord, depth)) > 0.0) faceN = -faceN; // face the camera by the ray to the point (n.z is wrong: it flips when the camera looks above the horizon)

	if (depth < 1.0 && isEyeInWater == 0) {
		vec3 vpos = viewPos(texcoord, depth);
		float dist = length(vpos);
		float litSun = 1.0; // sun visibility from the shadow map (1 without shadows)

#ifdef SSAO
		{
				float zc = -vpos.z;
				vec3 nrm = faceN;
				float radiusPx = clamp(0.8 * gbufferProjection[1][1] * 0.5 * viewHeight / zc, 2.0, 40.0);
				ivec2 fp = ivec2(gl_FragCoord.xy) & 3;
				float rot = (bayer4[fp.x + fp.y * 4] + 0.5) / 16.0 * 6.2831853; // 4x4 ordered pattern: removed exactly by the 4x4 blur
				float occ = 0.0;
				for (int i = 0; i < 8; i++) {
					float a = rot + float(i) * 0.7853982;
					float r = radiusPx * (0.35 + 0.65 * fract(float(i) * 0.618034 + 0.3));
					vec2 uv = texcoord + vec2(cos(a), sin(a)) * r / vec2(viewWidth, viewHeight);
					float ds = texture(depthtex0, uv).r;
					if (ds < 1.0) {
						// occludes only if the neighbour lies above the surface plane, not merely closer to the camera
						vec3 v = viewPos(uv, ds) - vpos;
						float d = length(v);
						float elev = dot(v, nrm) / max(d, 0.0001);
						occ += (d > 0.02 && d < 1.6 && elev > 0.12) ? (1.0 - d / 1.6) * smoothstep(0.12, 0.5, elev) : 0.0;
					}
				}
			aoOut = 1.0 - clamp(occ / 8.0, 0.0, 1.0) * SSAO_STRENGTH;
		}
#endif

#ifdef SHADOWS
		{
			vec3 L = normalize(shadowLightPosition);
			float up = dot(L, normalize(upPosition));
			float lightVis = smoothstep(0.02, 0.25, up) * (sunAngle < 0.5 ? 1.0 : 0.35);
			if (lightVis > 0.0) {
				vec3 n = faceN;
				float ndl = dot(n, L);
				float bias = 0.04 + 0.002 * dist;
				vec3 feet = (gbufferModelViewInverse * vec4(vpos + n * bias, 1.0)).xyz;
				vec4 sc = shadowProjection * (shadowModelView * vec4(feet, 1.0));
				sc.xyz /= sc.w;
				float edge = max(abs(sc.x), abs(sc.y));
				sc.xy = distortShadow(sc.xy);
				vec3 s = sc.xyz * 0.5 + 0.5;
				float lit = 1.0;
				if (s.x > 0.0 && s.x < 1.0 && s.y > 0.0 && s.y < 1.0 && s.z < 1.0) {
					float slope = clamp(1.0 - ndl, 0.0, 1.0);
					lit = (texture(shadowtex0, s.xy).r + 0.0004 + 0.0016 * slope < s.z) ? 0.0 : 1.0;
				}
				lit *= smoothstep(-0.02, 0.15, ndl);          // faces turned away from the sun are shaded even if the map misses them
				lit = mix(lit, 1.0, smoothstep(0.9, 1.0, edge)); // shadow fades out towards the map edge
				litSun = lit;
				float shade = (1.0 - lit) * SHADOW_STRENGTH * lightVis;
				color *= mix(vec3(1.0), vec3(0.62, 0.70, 0.90), shade);
			}
		}
#endif


#ifdef PBR
		color += texture(colortex5, texcoord).rgb * 2.0 * litSun * (1.0 - step(0.5, texture(colortex2, texcoord).r)); // sun highlights from the PBR maps: hidden in shadow, not under water
#endif

#ifdef WATER_FX
		if (texture(colortex2, texcoord).r > 0.5 && depth < texture(depthtex1, texcoord).r) {
			float d1 = texture(depthtex1, texcoord).r;
			float thick = d1 < 1.0 ? max(viewDist(texcoord, d1) - viewDist(texcoord, depth), 0.0) : 64.0;
			float dayF = smoothstep(-0.1, 0.25, dot(normalize(sunPosition), normalize(upPosition)));

			// depth tint: clear at the shore, deeper blue further out
			vec3 deep = vec3(0.03, 0.20, 0.34) * (0.3 + 0.7 * dayF);
			color = mix(color, deep, clamp(1.0 - exp(-thick * 0.15), 0.0, 0.88) * 0.9 * WATER_TINT);

			// small procedural waves for the normal
			vec3 feet = (gbufferModelViewInverse * vec4(vpos, 1.0)).xyz;
			vec3 vd = normalize(feet);
			vec2 wp = feet.xz + cameraPosition.xz;
			float t = frameTimeCounter;
			vec2 e = vec2(0.15, 0.0);
			float h0 = waveH(wp, t);
			vec2 grad = vec2(waveH(wp + e.xy, t) - h0, waveH(wp + e.yx, t) - h0) / e.x;
			vec3 n = normalize(vec3(-grad.x * 0.05 * WATER_WAVES, 1.0, -grad.y * 0.05 * WATER_WAVES));

			float cosT = clamp(dot(n, -vd), 0.0, 1.0);
			float fres = 0.02 + 0.98 * pow(1.0 - cosT, 5.0);
			vec3 r = reflect(vd, n);
			vec3 refl = skyGradient(r, dot(normalize(sunPosition), normalize(upPosition)), normalize(mat3(gbufferModelViewInverse) * normalize(sunPosition)));

#ifdef WATER_SSR
			{
				vec3 nv = normalize(mat3(gbufferModelView) * n);
				vec3 rv = reflect(normalize(vpos), nv);
				if (rv.z < 0.0) {
					vec3 p = vpos;
					float sl = 0.4;
					for (int i = 0; i < WATER_SSR_STEPS; i++) {
						p += rv * sl;
						vec4 c = gbufferProjection * vec4(p, 1.0);
						if (c.w <= 0.0) break;
						vec2 uv = c.xy / c.w * 0.5 + 0.5;
						if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) break;
						float ds = texture(depthtex1, uv).r;
						if (ds < 1.0) {
							float sz = viewDist(uv, ds);
							float rz = -p.z;
							if (rz > sz && rz - sz < sl * 1.5 + 0.3) {
								float k = 1.0 - smoothstep(0.7, 1.0, max(abs(uv.x * 2.0 - 1.0), abs(uv.y * 2.0 - 1.0)));
								refl = mix(refl, texture(colortex0, uv).rgb, k);
								break;
							}
						}
						sl *= 1.7;
					}
				}
			}
#endif

			color = mix(color, refl, clamp(fres * WATER_REFLECT, 0.0, 0.9));
			// sun glint
			vec3 Lw = normalize(mat3(gbufferModelViewInverse) * normalize(sunPosition));
			color += vec3(1.0, 0.88, 0.65) * pow(max(dot(r, Lw), 0.0), 250.0) * 2.5 * dayF;
		}
#endif

		float haze = (1.0 - exp(-max(dist - 48.0, 0.0) * 0.004)) * 0.45 * HAZE;
		color = mix(color, skyHorizon(dot(normalize(sunPosition), normalize(upPosition))), clamp(haze, 0.0, 0.6));
	} else if (depth >= 1.0 && isEyeInWater == 0) {
		vec3 dir = normalize(mat3(gbufferModelViewInverse) * viewPos(texcoord, 1.0));
		// the whole sky is drawn here (soft gradient, no band, no grey below the horizon); sun, moon and stars are kept from the game
		float sunHs = dot(normalize(sunPosition), normalize(upPosition));
		vec3 Lw = normalize(mat3(gbufferModelViewInverse) * normalize(sunPosition));
		float dd, gg, nn;
		skyWeights(sunHs, dd, gg, nn);
		float luma = dot(color, vec3(0.299, 0.587, 0.114));
		float keep = smoothstep(mix(0.90, 0.28, nn), mix(0.98, 0.45, nn), luma);
		color = mix(skyGradient(dir, sunHs, Lw), color, keep);
#ifdef CLOUDS
		color = addClouds(color, dir, sunHs, Lw);
#endif
	}

	outColor = vec4(color, 1.0);
	outAO = vec4(aoOut, 0.0, 0.0, 1.0);
}
