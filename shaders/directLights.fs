uniform sampler2D tex1;

varying vec3 normal, lightDir;
varying vec4 diffuse, ambient;

void main()
{
	float intensity, af, at;
	vec3 n, cf, ct;
	vec4 texel1;

	n = normalize(normal);
	intensity = max( dot( n, lightDir), 0.0);
	texel1 = texture2D( tex1, gl_TexCoord[1].st);

	cf = ambient.rgb + diffuse.rgb * intensity;
	af = diffuse.a;
	ct = texel1.rgb;
	at = texel1.a;
	
	gl_FragColor = vec4(ct * cf, at * af);
}