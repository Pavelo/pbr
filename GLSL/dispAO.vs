attribute float accessibility;

void main()
{
	vec3 normal, lightDir;
	vec4 globalAmbient, ambient, diffuse;
	float NdotL;

	normal = normalize( gl_NormalMatrix * gl_Normal);
	lightDir = normalize( vec3( gl_LightSource[0].position));
	NdotL = max( dot( normal, lightDir), 0.0);

	globalAmbient = gl_FrontMaterial.ambient * gl_LightModel.ambient;
	ambient = gl_FrontMaterial.ambient * gl_LightSource[0].ambient;
	diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;

//	gl_FrontColor = accessibility * (ambient + globalAmbient + NdotL * diffuse);
	globalAmbient *= accessibility;
	gl_FrontColor = ambient + globalAmbient + NdotL * diffuse;
//	gl_FrontColor = accessibility * NdotL * diffuse;

	gl_Position = ftransform();
}
