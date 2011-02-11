uniform sampler2D tex0, tex1;

varying vec3 normal, lightDir;
varying vec4 diffuse, ambient;

// equivalent to OpenGL's GL_MODULATE
void main()
{
	float intensity, af, at;
	vec3 n, cf, ct, accessibility;
	vec4 texel1;

	n = normalize(normal);
	intensity = max( dot( n, lightDir), 0.0);
	accessibility = texture2D( tex0, gl_TexCoord[0].st).rgb;
	texel1 = texture2D( tex1, gl_TexCoord[1].st);

	ct = texel1.rgb;
	at = texel1.a;
	cf = accessibility * ambient.rgb + diffuse.rgb * intensity; // frag color
	af = diffuse.a; // frag alpha

	gl_FragColor = vec4(ct * cf, at * af);
}
