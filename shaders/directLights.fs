varying vec3 normal, lightDir;
varying vec4 diffuse, ambient;

void main()
{
	float NdotL;
	vec3 n;
	vec4 color;

	n = normalize(normal);
	NdotL = max( dot( n, lightDir), 0.0);

	color = ambient + diffuse * NdotL;

	gl_FragColor = color;
}