uniform sampler2D tex0, tex1;

void main()
{
	vec4 accessibility, texel;

	accessibility = texture2D( tex0, gl_TexCoord[0].st);
	texel = texture2D( tex1, gl_TexCoord[1].st);

	gl_FragColor = accessibility * texel;
}
