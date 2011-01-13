uniform sampler2D tex;

void main()
{
	float accessibility;

	accessibility = texture2D( tex, gl_TexCoord[0].st).w;

	gl_FragColor = vec4( vec3(accessibility), 1.0);
}
