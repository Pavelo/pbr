attribute float accessibility;

varying vec3 normal, lightDir;
varying vec4 ambient, diffuse;
varying float accPerFrag;

void main()
{
	normal = normalize( gl_NormalMatrix * gl_Normal);
	lightDir = normalize( vec3( gl_LightSource[0].position)); // for directional lights position is direction!
	accPerFrag = accessibility;

	// global ambient + per-light ambient
	ambient = gl_FrontMaterial.ambient * (gl_LightSource[0].ambient + gl_LightModel.ambient);
	diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;

	gl_Position = ftransform();
}