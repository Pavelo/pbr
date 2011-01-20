uniform sampler2D tex;

void main()
{
	vec4 accessibility;

	accessibility = texture2D( tex, gl_TexCoord[0].st);

	gl_FragColor = accessibility;
}
