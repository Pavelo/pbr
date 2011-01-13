uniform sampler2D tex;

varying vec3 normal, lightDir;
varying vec4 diffuse, ambient;

void main()
{
	float intensity, af;
	vec3 n, cf;
	float accessibility;

	n = normalize(normal);
	intensity = max( dot( n, lightDir), 0.0);
	accessibility = texture2D( tex, gl_TexCoord[0].st).a;

	cf = accessibility * ambient.rgb + diffuse.rgb * intensity; // frag color
	af = ambient.a + diffuse.a; // frag alpha

	gl_FragColor = vec4(cf, af);
}
