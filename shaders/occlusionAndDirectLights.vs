varying vec3 normal, lightDir;
varying vec4 ambient, diffuse;

void main()
{
	normal = normalize( gl_NormalMatrix * gl_Normal);
	lightDir = normalize( vec3( gl_LightSource[0].position)); // for directional lights position is direction!

	// global ambient + per-light ambient
	ambient = gl_FrontMaterial.ambient * (gl_LightSource[0].ambient + gl_LightModel.ambient);
	diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;

	gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;
	gl_TexCoord[1] = gl_TextureMatrix[1] * gl_MultiTexCoord1;
	gl_Position = ftransform();
}
