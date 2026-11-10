#version 330 compatibility

uniform sampler2D gtexture;

in vec2 texcoord;
in vec4 glcolor;

layout(location = 0) out vec4 outColor;

void main() {
	outColor = texture(gtexture, texcoord) * glcolor;
	if (outColor.a < 0.2) discard; // leaves and glass do not cast square shadows
}
