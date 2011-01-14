uniform sampler2D tex;

varying vec3 lightDir;
varying vec4 diffuse, ambient;

// equivalent to OpenGL's GL_MODULATE
void main()
{
	float intensity, af;
	vec3 normal, cf;
	float accessibility;

	// decompress normals range
	normal = vec3(2.0) * (texture2D( tex, gl_TexCoord[0].st).xyz - vec3(0.5));
	normal = normalize( gl_NormalMatrix * normal);
	intensity = max( dot( normal, lightDir), 0.0);
	accessibility = texture2D( tex, gl_TexCoord[0].st).w;

	cf = accessibility * ambient.rgb + intensity * diffuse.rgb; // frag color
	af = ambient.a + diffuse.a; // frag alpha

	gl_FragColor = vec4(cf, af);
}
