varying vec3 lightDir;
varying vec4 ambient, diffuse;

void main()
{
	lightDir = normalize( vec3( gl_LightSource[0].position)); // for directional lights position is direction!

	// global ambient + per-light ambient
	ambient = gl_FrontMaterial.ambient * (gl_LightSource[0].ambient + gl_LightModel.ambient);
	diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;

	gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;
	gl_Position = ftransform();
}
