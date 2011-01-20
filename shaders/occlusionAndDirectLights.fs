uniform sampler2D tex;

varying vec3 normal, lightDir;
varying vec4 diffuse, ambient;

// equivalent to OpenGL's GL_MODULATE
void main()
{
	float intensity, af;
	vec3 n, cf, accessibility;

	n = normalize(normal);
	intensity = max( dot( n, lightDir), 0.0);
	accessibility = texture2D( tex, gl_TexCoord[0].st).rgb;

	cf = accessibility * ambient.rgb + diffuse.rgb * intensity; // frag color
	af = ambient.a + diffuse.a; // frag alpha

	gl_FragColor = vec4(cf, af);
}