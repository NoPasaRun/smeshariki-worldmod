// Pack options (shown by Iris in the shader menu as sliders and switches).
// By default only colour grading and haze are on: shadows and SSAO are off for weak GPUs.

// ---- colour
#define WARMTH 0.5 //[0.0 0.25 0.5 0.75 1.0 1.5]
#define SATURATION 1.15 //[0.8 0.9 1.0 1.1 1.15 1.25 1.4]
#define CONTRAST 1.05 //[0.9 1.0 1.05 1.1 1.2]
#define VIGNETTE 0.25 //[0.0 0.15 0.25 0.4 0.6]
#define HAZE 1.0 //[0.0 0.5 1.0 1.5 2.0]

// ---- shadows (second geometry pass from the sun: the most expensive part)
//#define SHADOWS
#define SHADOW_STRENGTH 0.7 //[0.3 0.5 0.7 0.85 1.0]
const int shadowMapResolution = 512; //[256 512 1024 2048]
const float shadowDistance = 48.0; //[24.0 32.0 48.0 64.0 96.0 128.0 192.0]
const float shadowIntervalSize = 2.0;
const bool shadowHardwareFiltering = false;

// ---- water (cheap tier: depth tint, sky fresnel, sun glint; SSR is optional and costs more)
#define WATER_FX
#define WATER_OPACITY 0.45 //[0.2 0.3 0.45 0.6 0.8 1.0]
#define WATER_TINT 0.6 //[0.0 0.3 0.6 0.9 1.2]
#define WATER_REFLECT 0.9 //[0.3 0.5 0.7 0.9 1.0]
#define WATER_WAVES 1.0 //[0.0 0.5 1.0 1.5 2.0]
//#define WATER_SSR
#define WATER_SSR_STEPS 8 //[6 8 12 16]

// ---- light and look
#define COLOR_LIGHT
#define TIME_TINT
#define TONEMAP
#define OUTLINE_STRENGTH 0.25 //[0.0 0.15 0.25 0.4 0.6]
#define WIND
#define WIND_STRENGTH 1.0 //[0.5 1.0 1.5 2.0]
//#define BLOOM
#define BLOOM_STRENGTH 0.8 //[0.4 0.6 0.8 1.0 1.5]
#define BLOOM_RADIUS 3 //[2 3 4 6]
//#define GODRAYS
#define GODRAYS_STRENGTH 0.6 //[0.3 0.6 0.9 1.2]
// buffer formats (colortex3/4 = RGBA16F for bloom) are declared in final.fsh and composite2.fsh: Iris reads them only from the pass file itself

// ---- sky: own sky gradient and rounded-block clouds
#define CLOUDS
#define CLOUD_COVERAGE 0.5 //[0.3 0.4 0.5 0.6]
#define CLOUD_HEIGHT 230.0 //[180.0 230.0 280.0]

// ---- PBR maps of the resource pack (LabPBR: _s smoothness, _n normals): relief and specular highlights
#define PBR
#define PBR_BUMP 0.8 //[0.4 0.6 0.8 1.0 1.4]
#define SPECULAR_STRENGTH 1.0 //[0.4 0.7 1.0 1.5 2.0]
//#define PBR_FLIP_Y

// ---- screen-space ambient occlusion
//#define SSAO
#define SSAO_STRENGTH 0.8 //[0.4 0.6 0.8 1.0 1.3]

// ---- shadow map distortion towards the player (sharper near, coarser far)
#define SHADOW_DISTORT 0.15

vec2 distortShadow(vec2 xy) {
	float f = length(xy) * (1.0 - SHADOW_DISTORT) + SHADOW_DISTORT;
	return xy / f;
}
