#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#define ILUT_USE_OPENGL

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <vector>

// includes, GL
#include <GL/glew.h>

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// includes, DevIL
#include <IL/il.h>
#include <IL/ilu.h>
#include <IL/ilut.h>

// includes, CUDA
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <rendercheck_gl.h>

using namespace std;

// definitions
#define PI 3.14159265358979323846f
#define INIT_ROTATE_X 15.0f
#define INIT_ROTATE_Y 330.0f
#define INIT_LIGHT_ROTATE_X 0.0f
#define INIT_LIGHT_ROTATE_Y 0.0f

// enumerated types
enum ViewMode
{
	POLY_CUBE     = 0,
	POINTS        = 1,
	NORMALS       = 2,
	S_DIR_LIGHTS  = 3,
	S_OCCLUSION   = 4,
	S_OCC_AND_DIR = 5
};

////////////////////////////////////////////////////////////////////////////////
// data structures
typedef struct _Vertex    Vertex;
typedef struct _Face      Face;
typedef struct _Solid     Solid;
typedef struct _Scene     Scene;
typedef struct _Material  Material;
typedef struct _Surfel    Surfel;

struct _Vertex
{
	float3 pos;
};

struct _Face
{
	uint3 v;
	uint3 t;
	uint3 n;
	float area;
	float tArea;
	float3 centroid;
	float3 normal;
};

struct _Solid
{
	string name;
	unsigned int faceBegin;
	unsigned int faceEnd;
	GLuint aotId;             // ambient occlusion texture id
	Material *mat;
	float surfaceArea;
	float textureArea;
};

struct _Scene
{
	vector<Vertex> v;
	vector<float2> vt;
	vector<float3> vn;
	vector<Face> f;
	vector<Solid> s;
	vector<Material> m;
};

struct _Material
{
	string name;
	
	float ambient[3];
	float diffuse[3];
	float specular[3];
	float shininess;
	
	GLuint textureId;
	string textureName;
	string texturePath;
};

struct _Surfel
{
	float3 pos;
	float3 normal;
	int texelId;
	unsigned int faceId;
	unsigned int solidId;
	float area;
};

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 512;
const unsigned int window_height = 512;

// global variables
unsigned int timer = 0;
float anim = 0.0;
Scene* scn;
vector<Surfel> pointCloud;
ViewMode view_model = S_DIR_LIGHTS;
float light_rotate_x = INIT_LIGHT_ROTATE_X, light_rotate_y = INIT_LIGHT_ROTATE_Y;
float light_orientation[] = {0.0, 0.2, 0.8, 0.0};
bool altPressed = false;
GLuint shaderSelected = 0, shaderID[8];
ILuint *imgId;
int counter = 0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = INIT_ROTATE_X, rotate_y = INIT_ROTATE_Y;
float3 translate = make_float3( .0f, .0f, -10.0f);

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
unsigned int frameCount = 0;
bool g_Verify = false;

// CheckFBO/BackBuffer class objects
CheckRender       *g_CheckRender = NULL;

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// kernels
extern "C" void faceArea(int n_faces, int4* face_v_id, float3* vertex, float* face_area, float4* service);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
string help();
float semiperimeter( vector<float> length);
float3 faceCentroid( Face* f);
float3 faceNormal( Face* f);
float3 getVector( float3 tail, float3 head);
float3 reverseVector( float3 v);
CUTBoolean preprocessing( int argc, char** argv);
CUTBoolean run( int argc, char** argv);
void cleanup();
void setLighting();
void setAOTextures();
float faceArea( float2 v0, float2 v1, float2 v2);
float faceArea( float3 v0, float3 v1, float3 v2);
int balanceMapResolution( int desiredMapResolution, float solidSurfaceArea, float solidTextureArea, float m_solidSuraceArea, float m_solidTextureArea);
CUTBoolean createPointCloud( int desiredMapResolution, int *balancedMapResolution, vector<Surfel> &pc);
CUTBoolean growPointCloud( int mapResolution, vector<Surfel> &pc, Solid &object, int aoTextureId);
CUTBoolean savePointCloud( vector<Surfel> &pc, const char* path, int *balancedMapResolution);
CUTBoolean loadPointCloud( const char* path, vector<Surfel> &pc, int *balancedMapResolution);
float2 texelCentre( int dx, int dy, float du);
bool pointInTriangle( float2 p, float2 t0, float2 t1, float2 t2, float &beta, float &gamma);
float4 rasterizeCoordinates( int faceId, float2 texelUV, float2 t0, float2 t1, float2 t2, float4 itself);
float4 rasterizeBorders( int faceId, float2 texelUV, vector<float2> &vt, vector<Face> &f);
void dilatePatchesBorders( unsigned int currentMap, int textureDim, float *originalTexture);
CUTBoolean occlusion( int passes, vector<Surfel> &pc, float* accessibility, unsigned int solidId);
float formFactor_pA( Surfel* receiver, Face* emitterF);
float surfelShadow( Surfel* receiver, Face* emitter);

// IL functionality
bool initIL();
bool handleDevILErrors();

// GL functionality
CUTBoolean initGL( int argc, char** argv);
CUTBoolean loadMTL( const char* path);
CUTBoolean loadOBJ( const char* path);
void drawCube( float size);
void drawScene();
void drawSolid( Solid &mesh);
void drawFaceNormals();
void drawPointCloud( vector<Surfel> &cloud);
void drawPoint( Surfel* sf);
void noiseTexture( float* texel, int dim);
void checkersTexture( float* texel, int dim, float color0 = 0.0, float color1 = 1.0);

// GLSL functionality
char* textFileRead(char *fn);
int textFileWrite(char *fn, char *s);
void printShaderInfoLog(GLuint obj);
void printProgramInfoLog(GLuint obj);
GLuint setShaders( char* vertexShaderPath, char* fragmentShaderPath);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void specialKeys(int key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

// Cuda functionality
void runCuda();
void runAutoTest();

// utilities
float rad( float deg);
float deg( float rad);
float norm( float3 vec);
float3 normalizeVector( float3 vec);
float dot( float2 v1, float2 v2);
float dot( float3 v1, float3 v2);
float3 cross( float3 v1, float3 v2);
float clamp( float val, float inf = 0.0f, float sup = 1.0f);
float abs( float n);
float round( float n);
float2 add( float2 a, float2 b);
float2 sub( float2 a, float2 b);
float2 mul( float2 a, float2 b);
float2 div( float2 a, float2 b);
float2 mul( float c, float2 a);
float3 add( float3 a, float3 b);
float3 sub( float3 a, float3 b);
float3 mul( float3 a, float3 b);
float3 div( float3 a, float3 b);
float3 mul( float c, float3 a);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	if ( !preprocessing(argc, argv) )
		return 1;
	
    run(argc, argv);
    
    cudaThreadExit();
    
    cutilExit(argc, argv);
	
	return 0;
}

void AutoQATest()
{
    if (g_CheckRender && g_CheckRender->IsQAReadback()) {
        char temp[256];
        sprintf(temp, "AutoTest: Cuda GL Interop (VBO)");
	glutSetWindowTitle(temp);
	exit(0);
    }
}

void computeFPS()
{
    frameCount++;
    fpsCount++;
    if (fpsCount == fpsLimit-1) {
        g_Verify = true;
    }
    if (fpsCount == fpsLimit) {
        char fps[256];
        float ifps = 1.f / (cutGetAverageTimerValue(timer) * .001f);
        sprintf(fps, "%sAmbient Occlusion @ %3.1f fps",
                ((g_CheckRender && g_CheckRender->IsQAReadback()) ? "AutoTest: " : ""), ifps);  

        glutSetWindowTitle(fps);
        fpsCount = 0; 
        if (g_CheckRender && !g_CheckRender->IsQAReadback()) 
	    fpsLimit = (int)MAX(ifps, 1.f);

        cutilCheckError(cutResetTimer(timer));  

        AutoQATest();
    }
 }

// initialize DevIL
bool initIL()
{
	// Check version number to know if it's all updated
	if (ilGetInteger(IL_VERSION_NUM) < IL_VERSION ||
		iluGetInteger(ILU_VERSION_NUM) < ILU_VERSION ||
		ilutGetInteger(ILUT_VERSION_NUM) < ILUT_VERSION)
	{
		cerr << "DevIL library is out of date! Please upgrade\n";
		return false;
	}
	// Needed to initialize DeviIL
	ilInit();
	iluInit();
	
	// GL cannot use palettes anyway, so convert early.
	ilEnable (IL_CONV_PAL);
	
	// Gets rid of dithering on some nVidia-based cards.
	ilutEnable (ILUT_OPENGL_CONV);
	
	return true;
}

bool handleDevILErrors()
{
	ILenum error = ilGetError();
	
	if (error != IL_NO_ERROR)
	{
		while ( error != IL_NO_ERROR )
		{
			cerr << "DevIL error: " << iluErrorString(error) << endl;
			error = ilGetError();
		}
		return false;
	}
	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
CUTBoolean initGL(int argc, char **argv)
{
	// functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Ambient Occlusion");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialKeys);
    glutMotionFunc(motion);

	// initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported("GL_VERSION_2_0 ")) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return CUTFalse;
    }
	if (!glewIsSupported("GL_ARB_fragment_shader"))
	{
		cerr << "GL_ARB_fragment_shader extension is not available!" << endl;
	}
	if (!glewIsSupported("GL_ARB_vertex_shader"))
	{
		cerr << "GL_ARB_vertex_shader extension is not available!" << endl;
	}
	if (!glewIsSupported("GL_ARB_shader_objects"))
	{
		cerr << "GL_ARB_shader_objects extension is not available!" << endl;
	}

    // default initialization
    glClearColor(0.0, 0.52, 0.78, 1.0);
    glEnable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (GLfloat) window_width / (GLfloat) window_height, 0.1, 60.0);
	
	// lighting
	setLighting();

	// shading
	string vs_path = "/Developer/GPU Computing/C/src/occlusion/shaders/directLights.vs";
	string fs_path = "/Developer/GPU Computing/C/src/occlusion/shaders/directLights.fs";
	shaderID[0] = setShaders( (char*)vs_path.c_str(), (char*)fs_path.c_str());

	vs_path = "/Developer/GPU Computing/C/src/occlusion/shaders/occlusionOnly.vs";
	fs_path = "/Developer/GPU Computing/C/src/occlusion/shaders/occlusionOnly.fs";
	shaderID[1] = setShaders( (char*)vs_path.c_str(), (char*)fs_path.c_str());
	
	vs_path = "/Developer/GPU Computing/C/src/occlusion/shaders/occlusionAndDirectLights.vs";
	fs_path = "/Developer/GPU Computing/C/src/occlusion/shaders/occlusionAndDirectLights.fs";
	shaderID[2] = setShaders( (char*)vs_path.c_str(), (char*)fs_path.c_str());
	
	shaderSelected = shaderID[0];
	
	// texturing
	setAOTextures();

    CUT_CHECK_ERROR_GL();

    return CUTTrue;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
CUTBoolean run(int argc, char** argv)
{
    // Create the CUTIL timer
    cutilCheckError( cutCreateTimer( &timer));

	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	if (CUTFalse == initGL(argc, argv)) {
		return CUTFalse;
	}
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
		cutilGLDeviceInit(argc, argv);
	} else {
		cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
 	}
	
	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	
	// run the cuda part
//   	runCuda();
	
	// things to do when the program exit
	atexit(cleanup);
	
	// show help
	cout << "Press H for help" << endl;
	
	// start rendering mainloop
	glutMainLoop();

    cudaThreadExit();

	return CUTTrue;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda()
{
//	unsigned int mem_size;
//	
//	// allocate memory and copy vertexes on device
//	mem_size = h_imesh.v.size() * sizeof(float3);
//	
//	float3* d_ivertex;
//	cutilSafeCall( cudaMalloc( (void**) &d_ivertex, mem_size));
////	cutilSafeCall( cudaMemcpy( d_ivertex, h_imesh.v, mem_size, cudaMemcpyHostToDevice));
//	
////	float3* h_overtex = (float3*) malloc(mem_size);
////	cutilSafeCall( cudaMemcpy( h_overtex, d_ivertex, mem_size, cudaMemcpyDeviceToHost));
//	
//	// allocate memory and copy faces on device
//	mem_size = h_imesh.f.size() * sizeof(int3);
//	
//	int3* h_iface = (int3*) malloc(mem_size);
//	for (unsigned int i=0; i<h_imesh.f.size(); i++) {
//		h_iface[i].x = h_imesh.f[i].v.x;
//		h_iface[i].y = h_imesh.f[i].v.y;
//		h_iface[i].z = h_imesh.f[i].v.z;
//	}
//	
//	int3* d_iface;
//	cutilSafeCall( cudaMalloc( (void**) &d_iface, mem_size));
//	cutilSafeCall( cudaMemcpy( d_iface, h_iface, mem_size, cudaMemcpyHostToDevice));
//	
//	float* d_ofaceArea;
//	mem_size = h_imesh.f.size() * sizeof(float);
//	cutilSafeCall( cudaMalloc( (void**) &d_ofaceArea, mem_size));
//
////	for (int i = 0; i < h_imesh.f.size(); i++) {
////		cout << h_oface[i].v.x << "/" << h_oface[i].t.x << "/" << h_oface[i].n.x << "\t";
////		cout << h_oface[i].v.y << "/" << h_oface[i].t.y << "/" << h_oface[i].n.y << "\t";
////		cout << h_oface[i].v.z << "/" << h_oface[i].t.z << "/" << h_oface[i].n.z << "\t";
////	}
//	
//	float4* d_serv;
//	cutilSafeCall( cudaMalloc( (void**) &d_serv, h_imesh.f.size()*sizeof(float4)));
//	
////	faceArea( h_imesh.f.size(), d_iface, d_ivertex, d_ofaceArea, d_serv);
//	
//	float4* h_serv = (float4*) malloc(h_imesh.f.size()*sizeof(float4));
//	cutilSafeCall( cudaMemcpy( h_serv, d_serv, h_imesh.f.size()*sizeof(float4), cudaMemcpyDeviceToHost));
//	
//	for (unsigned int i=0; i<h_imesh.f.size(); i++) {
//		cout << h_serv[i].x << "\t" << h_serv[i].y << "\t" << h_serv[i].z << "\t" << h_serv[i].w << endl;
//	}
//	
//	float* h_ofaceArea = (float*) malloc(mem_size);
//	
//	// free memory
//	cutilSafeCall( cudaFree( d_ivertex));
//	cutilSafeCall( cudaFree( d_iface));
//	cutilSafeCall( cudaFree( d_ofaceArea));
//	cutilSafeCall( cudaFree( d_serv));
//	free( h_iface);
////	free( h_overtex);
//	free( h_serv);
//	free( h_ofaceArea);
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    cutilCheckError(cutStartTimer(timer));  

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
	
    glTranslatef(translate.x, translate.y, translate.z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
	
	glPushMatrix();
		glRotatef( light_rotate_y, 0.0, 1.0, 0.0);
		glRotatef( light_rotate_x, 1.0, 0.0, 0.0);
		glLightfv(GL_LIGHT0, GL_POSITION, light_orientation);
	glPopMatrix();

	glUseProgram(shaderSelected);
	
	switch (view_model)
	{
		case POINTS:
			drawPointCloud(pointCloud);
			break;

		case NORMALS:
			drawFaceNormals();
			break;

		case S_DIR_LIGHTS:
		case S_OCCLUSION:
		case S_OCC_AND_DIR:
			drawScene();
			break;

		case POLY_CUBE:
			drawCube(1.5f);
			break;

		default:
			break;
	}

    glutSwapBuffers();
    glutPostRedisplay();

    anim += 0.01;

    cutilCheckError(cutStopTimer(timer));  
    computeFPS();
}

void cleanup()
{
    cutilCheckError( cutDeleteTimer( timer));

    if (g_CheckRender) {
        delete g_CheckRender; g_CheckRender = NULL;
    }
	
	free(scn);
	free(imgId);
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) {

	// quit application
		case 27 :
		case 'q' :
		case 'Q' :
			exit(0);
			break;
			
	// pretty useless polygonal cube
		case '0':
			view_model = POLY_CUBE;
			shaderSelected = 0;
			break;

	// points/vertexes view
		case '1':
			view_model = POINTS;
			shaderSelected = 0;
			break;
			
	// surfel view
		case '2':
			view_model = NORMALS;
			shaderSelected = 0;
			break;
			
	// polygonal view, direct lights rendering (shader)
		case '3':
			view_model = S_DIR_LIGHTS;
			shaderSelected = shaderID[0];
			break;

	// polygonal view, occlusion rendering (shader)
		case '4':
			view_model = S_OCCLUSION;
			shaderSelected = shaderID[1];
			break;

	// polygonal view, ambient occlusion and direct lights rendering (shader)
		case '5':
			view_model = S_OCC_AND_DIR;
			shaderSelected = shaderID[2];
			break;
			
	// rotate light
		case 'l':
		case 'L':
			light_rotate_y += 9.0f;
			break;

	// rotate light in big steps
		case 'k':
		case 'K':
			light_rotate_y += 90.0f - fmod( light_rotate_y, 90.0f);
			break;
			
	// use shaders
//		case 's':
//		case 'S':
//			shaderSelected = shaderSelected ? 0 : shaderID;
//			break;

	// press space to reset camera view, or alt+space to reset lights position
		case 32:
			if ( glutGetModifiers() == GLUT_ACTIVE_ALT )
			{
				light_rotate_x = INIT_LIGHT_ROTATE_X;
				light_rotate_y = INIT_LIGHT_ROTATE_Y;
			}
			else
			{
				rotate_x = INIT_ROTATE_X;
				rotate_y = INIT_ROTATE_Y;
				translate.x = translate.y = 0.0;
				translate.z = -10.0;
			}
			break;

	// show help
		case 'H':
		case 'h':
			cout << help();
			break;

		default:
			break;
    }
}

void specialKeys(int key, int, int)
{
	switch (key) {
		case GLUT_KEY_UP:
			translate.y -= .1f;
			break;
		case GLUT_KEY_DOWN:
			translate.y += .1f;
			break;
		case GLUT_KEY_RIGHT:
			translate.x -= .1f;
			break;
		case GLUT_KEY_LEFT:
			translate.x += .1f;
			break;
		default:
			break;
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (glutGetModifiers() == GLUT_ACTIVE_ALT) {
		altPressed = true;
	} else {
		altPressed = false;
	}

    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - mouse_old_x;
    dy = y - mouse_old_y;

    if (mouse_buttons & 1)
	{
		if (altPressed)
		{
			light_rotate_x += dy * 0.2;
			light_rotate_y += dx * 0.2;
		}
		else
		{
			rotate_x += dy * 0.2;
			rotate_y += dx * 0.2;
		}
    }
	else if (mouse_buttons & 4)
	{
		if (altPressed)
		{
		}
		else
		{
			translate.z += dy * 0.02;
		}
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void drawCube(float size)
{
	glFrontFace(GL_CW);
	
	float c = size*0.5f;
	
	glBegin(GL_QUADS);
	//faccia in basso
	glNormal3f(0.0f, -1.0f, 0.0f);
	glVertex3f(-c,-c,-c);
	glVertex3f(-c,-c, c);
	glVertex3f( c,-c, c);
	glVertex3f( c,-c,-c);
	
	//faccia in alto
	glNormal3f( 0.0f,1.0f, 0.0f);
	glVertex3f(-c, c,-c);
	glVertex3f( c, c,-c);
	glVertex3f( c, c, c);
	glVertex3f(-c, c, c);
	
	//faccia davanti
	glNormal3f( 0.0f, 0.0f, 1.0f);
	glVertex3f(-c, c, c);
	glVertex3f( c, c, c);
	glVertex3f( c,-c, c);
	glVertex3f(-c,-c, c);
	
	//faccia dietro
	glNormal3f( 0.0f, 0.0f,-1.0f);
	glVertex3f(-c, c,-c);
	glVertex3f(-c,-c,-c);
	glVertex3f( c,-c,-c);
	glVertex3f( c, c,-c);
	
	//faccia sinistra
	glNormal3f( -1.0f, 0.0f, 0.0f);
	glVertex3f(-c, c, c);
	glVertex3f(-c,-c, c);
	glVertex3f(-c,-c,-c);
	glVertex3f(-c, c,-c);
	
	//faccia destra
	glNormal3f(1.0f, 0.0f, 0.0f);
	glVertex3f( c, c,-c);
	glVertex3f( c,-c,-c);
	glVertex3f( c,-c, c);
	glVertex3f( c, c, c);
	glEnd();
	
	glFrontFace(GL_CCW);
}

CUTBoolean loadMTL(const char* path)
{
	Material mtl;
	CUTBoolean loaded;
	const char* line;
	string s_line;

	ifstream fp(path);
	
	if (fp.is_open())
	{
		while (!fp.eof())
		{
			getline(fp, s_line);
			line = s_line.c_str();
			
//			if (s_line.find("map_Kd") == 0)
//			{
//				tn = s_line.substr(7);
//				tp = "textures/" + tn;
//				mtl.textureName = &tn;
//				mtl.texturePath = &tp;
				// manca il caricamento della texture da TGA
//				strncpy(mtl.textureName, line + 7, strlen(line)-8);
//				printf(" [%s ", mtl.textureName);
//				sprintf(mtl.texturePath, "texture/%s", mtl.textureName);
//				printf("%d]", loadTGA(mtl.texturePath, mtl.textureId));
//			}
			if (s_line.find("newmtl") == 0)
			{
				// old material
				scn->m.push_back(mtl);
				
				// new material
				mtl.name = s_line.substr(7);
			}
			else if (s_line.find("Kd") == 0)
			{
				sscanf(line, "%*c%*c %f %f %f", &mtl.diffuse[0], &mtl.diffuse[1], &mtl.diffuse[2]);
			}
			else if (s_line.find("Ka") == 0)
			{
				sscanf(line, "%*c%*c %f %f %f", &mtl.ambient[0], &mtl.ambient[1], &mtl.ambient[2]);
			}
			else if (s_line.find("Ks") == 0)
			{
				sscanf(line, "%*c%*c %f %f %f", &mtl.specular[0], &mtl.specular[1], &mtl.specular[2]);
			}
			else if (s_line.find("Ns") == 0)
			{
				sscanf(line, "%*c%*c %f", &mtl.shininess);
			}
		}
		scn->m.push_back(mtl);
		scn->m.erase(scn->m.begin());
		
		loaded = CUTTrue;
	}
	else
	{
		loaded = CUTFalse;
	}
	
	fp.close();
	
	return loaded;
}

CUTBoolean loadOBJ(const char* path)
{
	Solid mesh;
	CUTBoolean loaded;
	string s_line, s_path, mName;
	const char *line;
	vector<Material>::iterator material;
	
	vector<string> vtn;
	vector<int> vtn_parsed;
	string vtn_element, mtllibName, mtllibPath;
	unsigned int pos;
	unsigned int beg;
	unsigned int end;
	
	Vertex v_tmp;
	float2 vt_tmp;
	float3 vn_tmp;
	Face f_tmp;
	
	ifstream fp(path);
	
	s_path = path;
	
	if (fp.is_open())
	{
		while (!fp.eof())
		{
			getline(fp, s_line);
			line = s_line.c_str();
			
			if (line[0] == 'v')
			{
				// texture vertex
				if (line[1] == 't')
				{
					sscanf(line, "%*c%*c %f %f", &vt_tmp.x, &vt_tmp.y);
					scn->vt.push_back(vt_tmp);
				}
				// normal vertex
				else if (line[1] == 'n')
				{
					sscanf(line, "%*c%*c %f %f %f", &vn_tmp.x, &vn_tmp.y, &vn_tmp.z);
					scn->vn.push_back(vn_tmp);
				}
				// vertex
				else
				{
					sscanf(line, "%*c %f %f %f", &v_tmp.pos.x, &v_tmp.pos.y, &v_tmp.pos.z);
					scn->v.push_back(v_tmp);
				}
			}
			// face
			else if (s_line[0] == 'f')
			{
				pos = s_line.find(" ") + 1;
				
				while (pos < s_line.length())
				{
					vtn_element.clear();
					while (s_line[pos] != ' ' && pos < s_line.length())
					{
						vtn_element.push_back(s_line[pos]);
						pos++;
					}
					vtn.push_back(vtn_element);
					pos++;
				}
				
				for (unsigned int i=0; i < vtn.size(); i++)
				{
					vtn_element = vtn[i];
					beg = end = 0;
					while (end < vtn_element.size()) {
						end = vtn_element.find("/", beg);
						vtn_parsed.push_back( atoi( vtn_element.substr(beg, end-beg).c_str()));
						beg = end+1;
					}
				}
				
				// lettura degli indici per la triangolarizzazione
				for (unsigned int i=2; i < vtn.size(); i++) {
					f_tmp.v.x = vtn_parsed[ 0 ];	// primo vertice, rimane fisso
					f_tmp.v.y = vtn_parsed[ 3*(i-1) ];
					f_tmp.v.z = vtn_parsed[ 3*i ];
					
					f_tmp.t.x = vtn_parsed[ 1 ];	// primo vertice texture, rimane fisso
					f_tmp.t.y = vtn_parsed[ 3*(i-1) +1 ];
					f_tmp.t.z = vtn_parsed[ 3*i +1 ];
					
					f_tmp.n.x = vtn_parsed[ 2 ];	// primo vertice normale, rimane fisso
					f_tmp.n.y = vtn_parsed[ 3*(i-1) +2 ];
					f_tmp.n.z = vtn_parsed[ 3*i +2 ];
					
					scn->f.push_back(f_tmp);
				}
								
				vtn_parsed.clear();
				vtn.clear();
			}
			else if ( s_line.find("mtllib") == 0 )
			{
				mtllibName = s_line.substr(7);
				mtllibPath = s_path.substr( 0, s_path.find_last_of( '/') + 1) + mtllibName;
				loadMTL( mtllibPath.c_str());
			}
			else if ( s_line.find("usemtl") == 0 )
			{
				// assign material to mesh
				mName = s_line.substr(7);
				material = scn->m.begin();
				while ( material != scn->m.end() )
				{
					if ( mName == material->name )
					{
						mesh.mat = &(*material);
					}
					++material;
				}
			}
			else if ( s_line[0] == 'g' )
			{
				// old mesh
				mesh.faceEnd = scn->f.size() - 1;
				scn->s.push_back(mesh);
				
				// new mesh
				mesh.name = s_line.substr(2);
				mesh.faceBegin = scn->f.size();
			}
		}
		// last mesh
		mesh.faceEnd = scn->f.size() - 1;
		scn->s.push_back(mesh);
		scn->s.erase( scn->s.begin());
		
		loaded = CUTTrue;
	}
	else
	{
		loaded = CUTFalse;
	}
	
	fp.close();
	
	return loaded;
}

void drawSolid(Solid &mesh)
{
	glMaterialfv(GL_FRONT, GL_AMBIENT, mesh.mat->ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mesh.mat->diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mesh.mat->specular);
	glMaterialf(GL_FRONT, GL_SHININESS, mesh.mat->shininess);
	
	glBindTexture(GL_TEXTURE_2D, mesh.aotId);

	glBegin(GL_TRIANGLES);
	for (unsigned int i = mesh.faceBegin; i <= mesh.faceEnd; i++)
	{
		glNormal3f(scn->vn[scn->f[i].n.x-1].x, scn->vn[scn->f[i].n.x-1].y, scn->vn[scn->f[i].n.x-1].z);
		glTexCoord2f(scn->vt[scn->f[i].t.x-1].x, scn->vt[scn->f[i].t.x-1].y);
		glVertex3f(scn->v[scn->f[i].v.x-1].pos.x, scn->v[scn->f[i].v.x-1].pos.y, scn->v[scn->f[i].v.x-1].pos.z);
		
		glNormal3f(scn->vn[scn->f[i].n.y-1].x, scn->vn[scn->f[i].n.y-1].y, scn->vn[scn->f[i].n.y-1].z);
		glTexCoord2f(scn->vt[scn->f[i].t.y-1].x, scn->vt[scn->f[i].t.y-1].y);
		glVertex3f(scn->v[scn->f[i].v.y-1].pos.x, scn->v[scn->f[i].v.y-1].pos.y, scn->v[scn->f[i].v.y-1].pos.z);
		
		glNormal3f(scn->vn[scn->f[i].n.z-1].x, scn->vn[scn->f[i].n.z-1].y, scn->vn[scn->f[i].n.z-1].z);
		glTexCoord2f(scn->vt[scn->f[i].t.z-1].x, scn->vt[scn->f[i].t.z-1].y);
		glVertex3f(scn->v[scn->f[i].v.z-1].pos.x, scn->v[scn->f[i].v.z-1].pos.y, scn->v[scn->f[i].v.z-1].pos.z);
	}
	glEnd();
}

void drawScene()
{
	vector<Solid>::iterator object;
	
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_LIGHTING);

	object = scn->s.begin();
	while ( object != scn->s.end() )
	{
		drawSolid( *object);
		++object;
	}

	glDisable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);
}

float semiperimeter( vector<float> length)
{
	float perimeter = 0;
	
	for (unsigned int i=0; i < length.size(); i++) {
		perimeter += length.at(i);
	}
	
	return perimeter * .5f;
}

float norm(float3 vec)
{
	return sqrt( vec.x*vec.x + vec.y*vec.y + vec.z*vec.z );
}

float3 normalizeVector(float3 vec)
{
	float length = norm( vec);
	
	if ( length == 0.0 ) {
		return make_float3( 0.0, 0.0, 0.0 );
	}
	return make_float3( vec.x/length, vec.y/length, vec.z/length );
}

float clamp(float val, float inf, float sup)
{
	if (val < inf) {
		return inf;
	}
	else if (val > sup) {
		return sup;
	}
	else {
		return val;
	}
}

float2 add(float2 a, float2 b)
{
	return make_float2( a.x + b.x, a.y + b.y);
}

float3 add(float3 a, float3 b)
{
	return make_float3( a.x + b.x, a.y + b.y, a.z + b.z);
}

float2 sub(float2 a, float2 b)
{
	return make_float2( a.x - b.x, a.y - b.y);
}

float3 sub(float3 a, float3 b)
{
	return make_float3( a.x - b.x, a.y - b.y, a.z - b.z);
}

float2 mul(float c, float2 a)
{
	return make_float2( c * a.x, c * a.y);
}

float3 mul(float c, float3 a)
{
	return make_float3( c * a.x, c * a.y, c * a.z);
}

float2 mul(float2 a, float2 b)
{
	return make_float2( a.x * b.x, a.y * b.y);
}

float3 mul(float3 a, float3 b)
{
	return make_float3( a.x * b.x, a.y * b.y, a.z * b.z);
}

float2 div(float2 a, float2 b)
{
	return make_float2( a.x / b.x, a.y / b.y);
}

float3 div(float3 a, float3 b)
{
	return make_float3( a.x / b.x, a.y / b.y, a.z / b.z);
}

float abs(float n)
{
	return max( -n, n);
}

float round(float n)
{
	return floor( n + 0.5);
}

CUTBoolean preprocessing(int argc, char** argv)
{
	int surfelMapDim, *balancedSurfelMapDim;

	// load poly mesh
	string dir, filename, path, spath, msg;
	stringstream pathStream;
	char** cfilename;
	
	cfilename = (char**) malloc(sizeof(char));
	scn = (Scene*) malloc(sizeof(Scene));
	
	if ( !initIL() ) // initialize DevIL
		return CUTFalse;
	
	dir = "/Developer/GPU Computing/C/src/occlusion/polyModels/";
	if ( cutCheckCmdLineFlag(argc, (const char**)argv, "mesh")) {
		cutGetCmdLineArgumentstr( argc, (const char**)argv, "mesh", cfilename);
	} else {
		cerr << "Please, specify a valid OBJ filename passed as command-line argument!\nEXAMPLE:\n--mesh=filename.obj" << endl;
		return CUTFalse;
	}
	filename = *cfilename;
	path = dir + filename;
	cout << "Loading scene... ";
	cout.flush();
	if ( !loadOBJ(path.c_str()) )
	{
		cerr << "File \"" << filename << "\" not found!" << endl;
		return CUTFalse;
	}
	
	if ( scn->vt.empty() )
	{
		cerr << "UVs not mapped!\nPlease, load a mesh with mapped texture coordinates." << endl;
		return CUTFalse;
	}
	// calculate face centroid and normal
	Face* tempF;
	for (unsigned int i=0;  i < scn->f.size(); i++) {
		tempF = &scn->f[i];
		tempF->centroid = faceCentroid( tempF);
		tempF->normal = faceNormal( tempF);
	}
	cout << "done" << endl;
	
	// create or load point cloud
	unsigned int nSolids = scn->s.size();
	if ( cutCheckCmdLineFlag( argc, (const char**)argv, "map")) {
		cutGetCmdLineArgumenti( argc, (const char**)argv, "map", &surfelMapDim);
	} else {
		surfelMapDim = 64;
	}
	balancedSurfelMapDim = (int*) malloc( nSolids * sizeof(int));

	cout << "Loading surfels cloud... ";
	cout.flush();
	dir = "/Developer/GPU Computing/C/src/occlusion/pointClouds/";
	if ( cutCheckCmdLineFlag(argc, (const char**)argv, "cloud"))
	{ // point cloud passed by argument
		cutGetCmdLineArgumentstr( argc, (const char**)argv, "cloud", cfilename );
		filename = *cfilename;
		path = dir + filename;

		if ( !loadPointCloud( path.c_str(), pointCloud, balancedSurfelMapDim))
		{ // point cloud doesn't exist
			do {
				cout << "File \"" << filename << "\" does not exist! Create it? ";
				getline(cin, msg);
				
				if (msg[0] == 'y' || msg[0] == 'Y') {
					cout << "Creating surfels cloud..." << endl;
					createPointCloud( surfelMapDim, balancedSurfelMapDim, pointCloud);
					cout << "Saving to \"" << filename << "\"..." << endl;
					savePointCloud( pointCloud, path.c_str(), balancedSurfelMapDim);
				} else if (msg[0] == 'n' || msg[0] == 'N') {
					cerr << "Cannot load any surfel cloud. Aborting..." << endl;
					return CUTFalse;
				} else {
					cout << "Answer with 'yes' or 'no'. ";
				}
			} while (msg.find_first_of("ynYN") != 0);
		}
		else
		{ // point cloud exists
			cout << "\"" << filename << "\" loaded correctly" << endl;
		}
	}
	else
	{ // default point cloud
		filename.replace( filename.length()-4, filename.length(), ".sfc" );
		path = dir + filename;
		
		if ( !loadPointCloud( path.c_str(), pointCloud, balancedSurfelMapDim))
		{ // point cloud doesn't exist
			cout << "Creating surfels cloud... " << endl;
			createPointCloud( surfelMapDim, balancedSurfelMapDim, pointCloud);
			cout << "Saving to \"" << filename << "\"..." << endl;
			savePointCloud( pointCloud, path.c_str(), balancedSurfelMapDim);
		}
		else
		{ // point cloud exists
			cout << "\"" << filename << "\" loaded correctly" << endl;
		}
	}

	// compute or load ambient occlusion texture
	int multipass;
	float* acc;
	
	if ( cutCheckCmdLineFlag( argc, (const char**)argv, "multipass")) {
		cutGetCmdLineArgumenti( argc, (const char**)argv, "multipass", &multipass);
	} else {
		multipass = 1;
	}
	dir = "/Developer/GPU Computing/C/src/occlusion/occlusionTextures/";

	if ( cutCheckCmdLineFlag(argc, (const char**)argv, "texturePrefix"))
	{ // occlusion map is passed by argument
		cutGetCmdLineArgumentstr( argc, (const char**)argv, "texturePrefix", cfilename);
		filename = *cfilename;
	}
	else
	{ // default occlusion map
		cutGetCmdLineArgumentstr( argc, (const char**)argv, "mesh", cfilename);
		filename = *cfilename;
		filename = filename.substr(0,filename.size()-4);
	}
	spath = dir + filename;

	acc = (float*) malloc(0);
	imgId = (ILuint*) malloc( nSolids * sizeof(ILuint));
	ilGenImages( nSolids, imgId);
	
	for (unsigned int i=0; i < nSolids; i++)
	{
		acc = (float*) realloc( (void*)acc, balancedSurfelMapDim[i] * balancedSurfelMapDim[i] * sizeof(float));
		for (int j=0; j < balancedSurfelMapDim[i] * balancedSurfelMapDim[i]; j++) {
			acc[j] = 1.0;
		}
		
		pathStream << spath << i << ".tga";
		path = pathStream.str();
		pathStream.str("");
		
		ilBindImage( imgId[i]);
		if ( !ilLoadImage( path.c_str()) )
		{ // occlusion map doesn't exist
			handleDevILErrors();
			cout << "Computing ambient occlusion... ";
			cout.flush();
			occlusion( multipass, pointCloud, acc, i);
			dilatePatchesBorders( i, balancedSurfelMapDim[i], acc);
			cout << "done" << endl;
			cout << "Saving to \"" << filename << i << ".tga" << "\"... ";
			cout.flush();
			
			ilTexImage (balancedSurfelMapDim[i], // width
						balancedSurfelMapDim[i], // height
						1,                       // depth
						1,                       // Bpp
						IL_LUMINANCE,            // format
						IL_FLOAT,                // type
						acc);                    // data
			// turn image the right direction
			iluRotate(90);
			iluMirror();
			
			ilEnable(IL_FILE_OVERWRITE);
			ilSave( IL_TGA, path.c_str());
			handleDevILErrors();
			cout << "done" << endl;
		}
		else
		{ // occlusion map exists
			cout << "Ambient occlusion map \"" << filename << i << ".tga" << "\" loaded correctly" << endl;
		}
		ilConvertImage(IL_LUMINANCE, IL_FLOAT);
	}

	free(cfilename);
	free(acc);
	free(balancedSurfelMapDim);
	
	return CUTTrue;
}

void setLighting()
{
	float lightModelAmbient[] = { 1.0, 1.0, 1.0, 1.0 };
	
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	glColorMaterial(GL_FRONT, GL_DIFFUSE);
	glShadeModel(GL_SMOOTH);
	glLightModelfv( GL_LIGHT_MODEL_AMBIENT, lightModelAmbient);
	
	glEnable(GL_CULL_FACE);
	glEnable(GL_LIGHT0);
}

void checkersTexture(float* texel, int dim, float color0, float color1)
{
	float finalColor;
	
	for (int i=0; i < dim; i++)
	{
		for (int j=0; j < dim; j++)
		{
			finalColor = (i + j) % 2;
			texel[i*dim+j] = clamp( finalColor, color0, color1);
		}
	}
}

void noiseTexture(float* texel, int dim)
{
	for (int i=0; i < dim; i++)
	{
		for (int j=0; j < dim; j++)
		{
			texel[i*dim+j] = (float)rand() / (float)RAND_MAX;
		}
	}
}

void setAOTextures()
{
	unsigned int nSolids = scn->s.size();
	GLuint tid[nSolids];
	ILubyte* occImg;
	
	glGenTextures (nSolids, tid);
	
	for (unsigned int i=0; i < nSolids; i++)
	{
		ilBindImage( imgId[i]);
		occImg = ilGetData();
		scn->s[i].aotId = tid[i];
		glBindTexture (GL_TEXTURE_2D, tid[i]);
		glPixelStorei (GL_UNPACK_ALIGNMENT, 1);
		glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
//		glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//		glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
		glTexImage2D(GL_TEXTURE_2D,
					 0,
					 GL_LUMINANCE,
					 ilGetInteger( IL_IMAGE_WIDTH),
					 ilGetInteger( IL_IMAGE_HEIGHT),
					 0,
					 GL_LUMINANCE,
					 GL_FLOAT,
					 occImg);
	}
}

void drawFaceNormals()
{
	glBegin(GL_LINES);
		glColor3f( 1.f, 1.f, 1.f);
	for (unsigned int i=0; i < scn->f.size(); i++) {
		glVertex3f( scn->f[i].centroid.x, scn->f[i].centroid.y, scn->f[i].centroid.z);
		glVertex3f( scn->f[i].normal.x+scn->f[i].centroid.x, scn->f[i].normal.y+scn->f[i].centroid.y, scn->f[i].normal.z+scn->f[i].centroid.z);
	}
	glEnd();
}

void drawPointCloud(vector<Surfel> &cloud)
{
	glBegin(GL_POINTS);
	for (unsigned int i=0; i < cloud.size(); i++)
		drawPoint( &cloud[i]);
	glEnd();
}

void drawPoint(Surfel* sf)
{
	glColor3f( 1.f, 1.f, 1.f);
	glVertex3f( sf->pos.x, sf->pos.y, sf->pos.z );
}

// dot product of R^2 vectors
float dot(float2 v1, float2 v2)
{
	return v1.x*v2.x + v1.y*v2.y;
}

// dot product of R^3 vectors
float dot(float3 v1, float3 v2)
{
	return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

// cross product of R^3 vectors
float3 cross(float3 v1, float3 v2)
{
	return make_float3( v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x );
}

float deg(float rad)
{
	return rad * 180.f / PI;
}

float rad(float deg)
{
	return deg * PI / 180.f;
}

// Get the vector from tail to head (which are points)
float3 getVector(float3 tail, float3 head)
{
	float3 res;
	
	res.x = head.x - tail.x;
	res.y = head.y - tail.y;
	res.z = head.z - tail.z;
	
	return res;
}

float3 reverseVector(float3 v)
{
	return make_float3( -v.x, -v.y, -v.z);
}

CUTBoolean savePointCloud(vector<Surfel> &pc, const char* path, int *balancedMapResolution)
{
	ofstream file;
	string line;
	
	file.open(path);
	
	if (file.is_open()) {
		line = "";
		
		file << "SFC" << endl;
		file << pc.size() << endl;
		for (unsigned int i=0; i < scn->s.size(); i++)
		{
			file << balancedMapResolution[i] << endl;
		}
		for (unsigned int i=0; i < pc.size(); i++) {
			file << pc[i].pos.x<<" "<<pc[i].pos.y<<" "<<pc[i].pos.z << endl;
		}
		for (unsigned int i=0; i < pc.size(); i++) {
			file << pc[i].normal.x<<" "<<pc[i].normal.y<<" "<<pc[i].normal.z << endl;
		}
		for (unsigned int i=0; i < pc.size(); i++) {
			file << pc[i].texelId <<" "<< pc[i].faceId <<" "<< pc[i].solidId << endl;
		}
		
	} else {
		cerr << "Error opening file \"" << path << "\"" << endl;
		return CUTFalse;
	}

	file.close();
	
	cout << "Surfels cloud saved correctly" << endl;

	return CUTTrue;
}

CUTBoolean growPointCloud(int mapResolution, vector<Surfel> &pc, Solid &object, int solidId)
{
	int faceId;
	float alpha, beta, gamma;
	float2 vt0, vt1, vt2, **texelUV;
	float3 v0, v1, v2, n0, n1, n2;
	float4 **barycentricCooAndId;
	float du = 1.0 / (float)mapResolution;
	Surfel point;

	// allocate mem for 2D arrays
	texelUV = (float2**) malloc( mapResolution * sizeof(float2*));
	barycentricCooAndId = (float4**) malloc( mapResolution * sizeof(float4*));
	for (int i=0; i < mapResolution; i++)
	{
		texelUV[i] = (float2*) malloc( mapResolution * sizeof(float2));
		barycentricCooAndId[i] = (float4*) malloc( mapResolution * sizeof(float4));
	}
	
	// place a surfel in the centre of each texel and store its UVs;
	// initialize the barycentric coordinates and face id array.
	for (int i=0; i < mapResolution; i++)
	{
		for (int j=0; j < mapResolution; j++)
		{
			texelUV[i][j] = texelCentre( i, j, du);
			barycentricCooAndId[i][j] = make_float4( 0.0, 0.0, 0.0, -1);
		}
	}

	// rasterize faces
	for (unsigned int id = object.faceBegin; id <= object.faceEnd; id++)
	{
		vt0 = scn->vt[ scn->f[id].t.x-1 ];
		vt1 = scn->vt[ scn->f[id].t.y-1 ];
		vt2 = scn->vt[ scn->f[id].t.z-1 ];
		
		for (int i=0; i < mapResolution; i++)
		{
			for (int j=0; j < mapResolution; j++)
			{
				barycentricCooAndId[i][j] = rasterizeCoordinates( id,
																  texelUV[i][j],
																  vt0,
																  vt1,
																  vt2,
																  barycentricCooAndId[i][j]);
			}
		}
	}

	for (int i=0; i < mapResolution; i++)
	{
		for (int j=0; j < mapResolution; j++)
		{
			faceId = barycentricCooAndId[i][j].w;
			if ( faceId != -1 )
			{
				alpha = barycentricCooAndId[i][j].x;
				beta  = barycentricCooAndId[i][j].y;
				gamma = barycentricCooAndId[i][j].z;

				// interpolate surfel position ( alpha*A + beta*B + gamma*C )
				v0 = scn->v[ scn->f[ faceId ].v.x-1 ].pos;
				v1 = scn->v[ scn->f[ faceId ].v.y-1 ].pos;
				v2 = scn->v[ scn->f[ faceId ].v.z-1 ].pos;
				point.pos = add( add( mul( alpha, v0), mul( beta, v1)), mul( gamma, v2));

				// interpolate surfel normal ( alpha*Na + beta*Nb + gamma*Nc )
				n0 = scn->vn[ scn->f[ faceId ].n.x-1 ];
				n1 = scn->vn[ scn->f[ faceId ].n.y-1 ];
				n2 = scn->vn[ scn->f[ faceId ].n.z-1 ];
				point.normal = normalizeVector( add( add( mul( alpha, n0), mul( beta, n1)), mul( gamma, n2)));

				// store surfel
				point.texelId = i * mapResolution + j;
				point.faceId = faceId;
				point.solidId = solidId;
				pc.push_back( point);
			}
		}
	}
	
	// deallocate mem
	for (int i=0; i < mapResolution; i++) {
		free(texelUV[i]);
		free(barycentricCooAndId[i]);
	}
	free(texelUV);
	free(barycentricCooAndId);

	return CUTTrue;
}

float faceArea(float3 v0, float3 v1, float3 v2)
{
	float s, a, b, c;
	float3 va, vb, vc;
	
	// get the vectors along edges
	va = sub( v0, v1);
	vb = sub( v0, v2);
	vc = sub( v1, v2);
	
	// get vectors length
	a = sqrt( dot( va, va));
	b = sqrt( dot( vb, vb));
	c = sqrt( dot( vc, vc));
	
	// get semiperimeter
	s = (a + b + c) * 0.5;
	
	// get area with Heron's formula
	return sqrt( s * (s-a) * (s-b) * (s-c) );
}

float faceArea(float2 v0, float2 v1, float2 v2)
{
	return faceArea(make_float3(v0.x, v0.y, 0.0), make_float3(v1.x, v1.y, 0.0), make_float3(v2.x, v2.y, 0.0));
}

int balanceMapResolution(int desiredMapResolution,
						 float solidSurfaceArea,
						 float solidTextureArea,
						 float m_solidSuraceArea,
						 float m_solidTextureArea)
{
	float ct, cw;
	
	ct = m_solidTextureArea / solidTextureArea;
	cw = solidSurfaceArea / m_solidSuraceArea;
	
	return (int)round( desiredMapResolution * sqrt( ct * cw));
}

CUTBoolean createPointCloud(int desiredMapResolution, int *balancedMapResolution, vector<Surfel> &pc)
{
	int sid, mapResolution;
	float m_sArea, sArea, m_tArea, tArea;
	vector<Solid>::iterator object;
	
	m_sArea = m_tArea = 0.0;
	object = scn->s.begin();
	while ( object != scn->s.end() )
	{
		sArea = tArea = 0.0;
		for (unsigned int i = object->faceBegin; i <= object->faceEnd; i++)
		{
			scn->f[i].area = faceArea(scn->v[ scn->f[i].v.x-1 ].pos,
									  scn->v[ scn->f[i].v.y-1 ].pos,
									  scn->v[ scn->f[i].v.z-1 ].pos);
			
			scn->f[i].tArea = faceArea(scn->vt[ scn->f[i].t.x-1 ],
									   scn->vt[ scn->f[i].t.y-1 ],
									   scn->vt[ scn->f[i].t.z-1 ]);
			sArea += scn->f[i].area;
			tArea += scn->f[i].tArea;
		}
		object->surfaceArea = sArea;
		object->textureArea = tArea;
		m_sArea += sArea;
		m_tArea += tArea;
		++object;
	}
	m_sArea /= (float) scn->s.size();
	m_tArea /= (float) scn->s.size();
	
	sid = 0;
	object = scn->s.begin();
	while ( object != scn->s.end() )
	{
		mapResolution = balanceMapResolution( desiredMapResolution, object->surfaceArea, object->textureArea, m_sArea, m_tArea);
		balancedMapResolution[sid] = mapResolution;
		growPointCloud( mapResolution, pc, *object, sid++);
		++object;
	}
	
	return CUTTrue;
}

void dilatePatchesBorders(unsigned int currentMap, int textureDim, float *originalTexture)
{
	int nexti, nextj, previ, prevj, *rasterizedFaceId;
	float **faceMask, **dilatedMask, intorno[8];
	
	// allocate mem
	rasterizedFaceId = (int*) malloc( textureDim * textureDim * sizeof(int));
	faceMask = (float**) malloc( textureDim * sizeof(float*));
	dilatedMask = (float**) malloc( textureDim * sizeof(float*));
	for (int i=0; i < textureDim; i++)
	{
		faceMask[i] = (float*) malloc( textureDim * sizeof(float));
		dilatedMask[i] = (float*) malloc( textureDim * sizeof(float));
	}
	
	// recreate face id map
	for (int i=0; i < textureDim * textureDim; i++)
	{
		rasterizedFaceId[i] = -1;
	}
	for (unsigned int i=0; i < pointCloud.size(); i++)
	{
		if ( pointCloud[i].solidId == currentMap )
		{
			rasterizedFaceId[ pointCloud[i].texelId ] = pointCloud[i].faceId;
		}
	}
	
	// create mask: white faces on black background
	for (int i=0; i < textureDim; i++)
	{
		for (int j=0; j < textureDim; j++)
		{
			dilatedMask[i][j] = faceMask[i][j] = ( rasterizedFaceId[ i * textureDim + j ] > -1 ) ? 1.0 : 0.0;
		}
	}
	
	// create mask: dilate on white by 1 pixel border
	nexti = nextj = previ = prevj = 0;
	for (int i=0; i < textureDim; i++)
	{
		nexti = (i+1) % textureDim;
		previ = (textureDim+i-1) % textureDim;
		for (int j=0; j < textureDim; j++)
		{
			nextj = (j+1) % textureDim;
			prevj = (textureDim+j-1) % textureDim;
			if ( faceMask[i][j] == 1.0 )
			{
				dilatedMask[nexti][j] = 1.0;
				dilatedMask[previ][j] = 1.0;
				dilatedMask[i][nextj] = 1.0;
				dilatedMask[i][prevj] = 1.0;
				dilatedMask[nexti][nextj] = 1.0;
				dilatedMask[nexti][prevj] = 1.0;
				dilatedMask[previ][nextj] = 1.0;
				dilatedMask[previ][prevj] = 1.0;
			}
		}
	}
	
	// masks difference
	for (int i=0; i < textureDim; i++)
	{
		for (int j=0; j < textureDim; j++)
		{
			dilatedMask[i][j] -= faceMask[i][j];
		}
	}
	
	// apply interpolation mask
	nexti = nextj = previ = prevj = 0;
	for (int i=0; i < textureDim; i++)
	{
		nexti = (i+1) % textureDim;
		previ = (textureDim+i-1) % textureDim;
		for (int j=0; j < textureDim; j++)
		{
			faceMask[i][j] = originalTexture[ i * textureDim + j ];
			nextj = (j+1) % textureDim;
			prevj = (textureDim+j-1) % textureDim;
			if ( dilatedMask[i][j] == 1.0 )
			{
				intorno[0] = originalTexture[nexti * textureDim + j];
				intorno[1] = originalTexture[previ * textureDim + j];
				intorno[2] = originalTexture[i * textureDim + nextj];
				intorno[3] = originalTexture[i * textureDim + prevj];
				intorno[4] = originalTexture[nexti * textureDim + nextj];
				intorno[5] = originalTexture[nexti * textureDim + prevj];
				intorno[6] = originalTexture[previ * textureDim + nextj];
				intorno[7] = originalTexture[previ * textureDim + prevj];
				
				faceMask[i][j] = *min_element( intorno, intorno+7);
			}
		}
	}
	
	// copy temp data to final data structure
	for (int i=0; i < textureDim; i++)
	{
		for (int j=0; j < textureDim; j++)
		{
			originalTexture[ i * textureDim + j ] = faceMask[i][j];
		}
	}
	
	// deallocate mem
	for (int i=0; i < textureDim; i++)
	{
		free(faceMask[i]);
		free(dilatedMask[i]);
	}
	free(faceMask);
	free(dilatedMask);
	free(rasterizedFaceId);
}

bool pointInTriangle(float2 p, float2 t0, float2 t1, float2 t2, float &beta, float &gamma)
{
	float2 v0, v1, v2;
	float dot00, dot01, dot02, dot11, dot12, invDenom;

	// Compute vectors
	v0 = sub( t2, t0);
	v1 = sub( t1, t0);
	v2 = sub( p , t0);

	// Compute dot products
	dot00 = dot(v0, v0);
	dot01 = dot(v0, v1);
	dot02 = dot(v0, v2);
	dot11 = dot(v1, v1);
	dot12 = dot(v1, v2);

	// Compute barycentric coordinates
	invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
	beta = (dot00 * dot12 - dot01 * dot02) * invDenom;
	gamma = (dot11 * dot02 - dot01 * dot12) * invDenom;

	// Check if point is in triangle
	return (beta >= 0) && (gamma >= 0) && (beta + gamma <= 1);
}

float4 rasterizeCoordinates(int faceId, float2 texelUV, float2 t0, float2 t1, float2 t2, float4 itself)
{
	float alpha, beta, gamma;
	
	if ( pointInTriangle( texelUV, t0, t1, t2, beta, gamma) )
	{
		alpha = 1 - beta - gamma;
		return make_float4( alpha, beta, gamma, faceId);
	}
	else
	{
		return itself;
	}
}

float4 rasterizeBorders(int faceId, float2 texelUV, vector<float2> &vt, vector<Face> &f)
{
	float alpha, beta, gamma;
	float2 vt0, vt1, vt2;
	vt0 = vt[ f[faceId].t.x-1 ];
	vt1 = vt[ f[faceId].t.y-1 ];
	vt2 = vt[ f[faceId].t.z-1 ];
	pointInTriangle( texelUV, vt0, vt1, vt2, beta, gamma);
	alpha = 1 - beta - gamma;
	
	return make_float4( alpha, beta, gamma, faceId);
}

float2 texelCentre(int dx, int dy, float du)
{
	float2 uv;
	
	uv.x = dx * du + du * 0.5;
	uv.y = dy * du + du * 0.5;
	
	return uv;
}

float3 faceCentroid(Face* f)
{
	float3 p, v0, v1, v2;

	v0 = scn->v[ f->v.x-1 ].pos;
	v1 = scn->v[ f->v.y-1 ].pos;
	v2 = scn->v[ f->v.z-1 ].pos;
	p = add( add( v0, v1), v2);
	p.x /= 3.0;
	p.y /= 3.0;
	p.z /= 3.0;

	return p;
}

float3 faceNormal(Face* f)
{
	float3 v0, v1, v2, a, b;
	
	v0 = scn->v[ f->v.x-1 ].pos;
	v1 = scn->v[ f->v.y-1 ].pos;
	v2 = scn->v[ f->v.z-1 ].pos;
	
	a = getVector( v0, v1);
	b = getVector( v0, v2);
	
	return normalizeVector( cross( a, b));
}

void visibleQuad(float3 p, float3 n, float3 v0, float3 v1, float3 v2, float3 &q0, float3 &q1, float3 &q2, float3 &q3)
{
	const float epsilon = 1e-6;
	float d = dot( n, p);
	float3 sd;
	
	// Compute the signed distances from the vertices to the plane
	sd.x = dot( n, v0) - d;
	sd.y = dot( n, v1) - d;
	sd.z = dot( n, v2) - d;
	if ( abs(sd.x) <= epsilon ) sd.x = 0.0;
	if ( abs(sd.y) <= epsilon ) sd.y = 0.0;
	if ( abs(sd.z) <= epsilon ) sd.z = 0.0;
	
	// Determine case
	if ( sd.x > 0.0 )
	{
		if ( sd.y > 0.0 )
		{
			if ( sd.z > 0.0 )
			{
				// +++ & ++0
				q0 = v0;
				q1 = v1;
				q2 = v2;
				q3 = q2;
			}
			else
			{
				// ++-
				q0 = v0;
				q1 = v1;
				q2 = add( v1, mul( sd.y/(sd.y - sd.z), sub(v2, v1))); // <-- ray-plane equations
				q3 = add( v0, mul( sd.x/(sd.x - sd.z), sub(v2, v0))); // <-'
			}
		}
		else if ( sd.y < 0.0 )
		{
			if ( sd.z > 0.0 )
			{
				// +-+
				q0 = v0;
				q1 = add( v0, mul( sd.x/(sd.x - sd.y), sub( v1, v0)));
				q2 = add( v1, mul( sd.y/(sd.y - sd.z), sub( v2, v1)));
				q3 = v2;
			}
			else if ( sd.z < 0.0 )
			{
				// +--
				q0 = v0;
				q1 = add( v0, mul( sd.x/(sd.x - sd.y), sub( v1, v0)));
				q2 = add( v0, mul( sd.x/(sd.x - sd.z), sub( v2, v0)));
				q3 = q2;
			}
			else
			{
				// +-0
				q0 = v0;
				q1 = add( v0, mul( sd.x/(sd.x - sd.y), sub( v1, v0)));
				q2 = v2;
				q3 = q2;
			}
		}
		else
		{
			if ( sd.z < 0.0 ) {
				//+0-
				q0 = v0;
				q1 = v1;
				q2 = add( v0, mul( sd.x/(sd.x - sd.z), sub( v2, v0)));
				q3 = q2;
			}
			else
			{
				// +0+
				q0 = v0;
				q1 = v1;
				q2 = v2;
				q3 = q2;
			}
		}
	}
	else if ( sd.x < 0.0 )
	{
		if ( sd.y > 0.0 )
		{
			if ( sd.z > 0.0 )
			{
				// -++
				q0 = add( v0, mul( sd.x/(sd.x - sd.y), sub( v1, v0)));
				q1 = v1;
				q2 = v2;
				q3 = add( v0, mul( sd.x/(sd.x - sd.z), sub( v2, v0)));
			}
			else if ( sd.z < 0.0 )
			{
				// -+-
				q0 = add( v0, mul( sd.x/(sd.x - sd.y), sub( v1, v0)));
				q1 = v1;
				q2 = add( v1, mul( sd.y/(sd.y - sd.z), sub( v2, v1)));
				q3 = q2;
			}
			else
			{
				// -+0
				q0 = add( v0, mul( sd.x/(sd.x - sd.y), sub( v1, v0)));
				q1 = v1;
				q2 = v2;
				q3 = q2;
			}
		}
		else if ( sd.y < 0.0 )
		{
			if ( sd.z > 0.0 )
			{
				// --+
				q0 = add( v0, mul( sd.x/(sd.x - sd.z), sub( v2, v0)));
				q1 = add( v1, mul( sd.y/(sd.y - sd.z), sub( v2, v1)));
				q2 = v2;
				q3 = q2;
			}
			else
			{
				// --- & --0
				q0 = p;
				q1 = p;
				q2 = p;
				q3 = p;
			}
		}
		else
		{
			if ( sd.z > 0.0 )
			{
				// -0+
				q0 = add( v0, mul( sd.x/(sd.x - sd.z), sub( v2, v0)));
				q1 = v1;
				q2 = v2;
				q3 = q2;
			}
			else
			{
				// -0- & -00
				q0 = p;
				q1 = p;
				q2 = p;
				q3 = p;
			}
		}
	}
	else
	{
		if ( sd.y > 0.0 )
		{
			if ( sd.z < 0.0 )
			{
				// 0+-
				q0 = v0;
				q1 = v1;
				q2 = add( v1, mul( sd.y/(sd.y - sd.z), sub( v2, v1)));
				q3 = q2;
			}
			else
			{
				// 0+0 & 0+-
				q0 = v0;
				q1 = v1;
				q2 = v2;
				q3 = q2;
			}
		}
		else if ( sd.y < 0.0 )
		{
			if ( sd.z > 0.0 )
			{
				// 0-+
				q0 = v0;
				q1 = add( v1, mul( sd.y/(sd.y - sd.z), sub( v2, v1)));
				q2 = v2;
				q3 = q2;
			}
			else
			{
				// 0-- & 0-0
				q0 = p;
				q1 = p;
				q2 = p;
				q3 = p;
			}
		}
		else
		{
			if ( sd.z > 0.0 )
			{
				// 00+
				q0 = v0;
				q1 = v1;
				q2 = v2;
				q3 = q2;
			}
			else
			{
				// 00- & 000
				q0 = p;
				q1 = p;
				q2 = p;
				q3 = p;
			}
		}
	}
}

float formFactor_pA(Surfel* receiver, float3 q0, float3 q1, float3 q2, float3 q3)
{
	float fpa;
	float3 r[4], g, p;
	
	p = receiver->pos;
	r[0] = normalizeVector( getVector( p, q0));
	r[1] = normalizeVector( getVector( p, q1));
	r[2] = normalizeVector( getVector( p, q2));
	r[3] = normalizeVector( getVector( p, q3));
	
	fpa = 0.0;
	for (unsigned int i=0; i < 4; i++)
	{
		g = normalizeVector( cross( r[(i+1)%4], r[i]));
		fpa += acos( clamp( dot( r[(i+1)%4], r[i]), -1.0, 1.0))
				* clamp( dot( receiver->normal, g), -1.0, 1.0);
	}
	fpa /= 2.0 * PI;
	
	return fpa;
}

float surfelShadow(Surfel* receiver, Face* emitter)
{
	float3 emitterVector, q0, q1, q2, q3;
	
	emitterVector = normalizeVector( getVector( emitter->centroid, receiver->pos));
	
	visibleQuad( receiver->pos,
				 receiver->normal,
				 scn->v[ emitter->v.x-1 ].pos,
				 scn->v[ emitter->v.y-1 ].pos,
				 scn->v[ emitter->v.z-1 ].pos,
				 q0, q1, q2, q3);

	if ( dot( emitter->normal, emitterVector) >= 0.0 )
	{
		return formFactor_pA( receiver, q0, q1, q2, q3);
	}
	else
	{
		return 0.0;
	}
}

CUTBoolean occlusion(int passes, vector<Surfel> &pc, float* accessibility, unsigned int solidId)
{
	float sshadow_total, **ss_global;
	
	ss_global = (float**) malloc( pc.size() * sizeof(float*));
	for (unsigned int i=0; i < pc.size(); i++) {
		ss_global[i] = (float*) malloc( scn->f.size() * sizeof(float));
	}

	for (int k=1; k <= passes; k++)
	{
		for (unsigned int i=0; i < pc.size(); i++)
		{
			if ( pc[i].solidId == solidId )
			{
				sshadow_total = .0f;
				for (unsigned int j=0; j < scn->f.size(); j++)
				{
					if ( pc[i].faceId != j )
					{ // if surfel lays on a face except the current face
						if (k == 1)
						{
							ss_global[i][j] = surfelShadow( &pc[i], &scn->f[j]);
							sshadow_total += ss_global[i][j];
						}
						else
						{
							sshadow_total += ss_global[i][j] * accessibility[ pc[i].texelId ];
						}
					}
				}
				accessibility[ pc[i].texelId ] = 1.f - sshadow_total;
			}
		}
	}
	
	for (unsigned int i=0; i < pc.size(); i++) {
		free(ss_global[i]);
	}
	free(ss_global);

	return CUTTrue;
}

CUTBoolean loadPointCloud(const char* path, vector<Surfel> &pc, int *balancedMapResolution)
{
	string s_line;
	const char* line;
	unsigned int n_surfels;
	
	ifstream file(path);
	
	if (file.is_open())
	{
		getline(file, s_line);
		if (s_line == "SFC")
		{
			getline(file, s_line);
			n_surfels = atoi(s_line.c_str());
			for (unsigned int i=0; i < scn->s.size(); i++)
			{
				getline(file, s_line);
				balancedMapResolution[i] = atoi(s_line.c_str());
			}
		}
		else {
			cerr << "Wrong file type!" << endl;
			return CUTFalse;
		}
		pc.clear();
		pc.resize(n_surfels);
		for (unsigned int i=0; i < n_surfels; i++)
		{
			getline(file, s_line);
			line = s_line.c_str();
			sscanf( line, "%g %g %g", &pc[i].pos.x, &pc[i].pos.y, &pc[i].pos.z);
		}
		for (unsigned int i=0; i < n_surfels; i++)
		{
			getline(file, s_line);
			line = s_line.c_str();
			sscanf( line, "%g %g %g", &pc[i].normal.x, &pc[i].normal.y, &pc[i].normal.z);
		}
		for (unsigned int i=0; i < n_surfels; i++) {
			getline(file, s_line);
			line = s_line.c_str();
			sscanf( line, "%d %d %d", &pc[i].texelId, &pc[i].faceId, &pc[i].solidId);
		}
	} else {
		return CUTFalse;
	}

	file.close();

	return CUTTrue;
}

string help()
{
	string msg;
	
	msg  = "--------\n";
	msg += "Controls\n";
	msg += "--------\n";
	msg += "\n";
	msg += "Keyboard:\n";
	msg += "1 ............ view mesh vertexes (equivalent to surfels position)\n";
	msg += "2 ............ view surfel cloud representation\n";
	msg += "3 ............ view polygonal mesh\n";
	msg += "4 ............ view single pass occlusion representation\n";
	msg += "5 ............ view double pass occlusion representation\n";
	msg += "6 ............ view shading using bent normals\n";
	msg += "l ............ rotate scene light\n";
	msg += "k ............ rotate scene light in big steps\n";
	msg += "SPACE ........ reset camera view to initial values\n";
	msg += "alt + SPACE .. reset light position to initial values\n";
	msg += "\n";
	msg += "Mouse:\n";
	msg += "primary button (hold down while dragging) .......... rotate scene\n";
	msg += "secondary button (hold down while dragging) ........ zoom-in or zoom-out\n";
	msg += "alt + primary button (hold down while dragging) .... rotate light\n";
	
	return msg;
}

char* textFileRead(char *fn)
{
	FILE *fp;
	char *content = NULL;
	
	int count=0;
	
	if (fn != NULL) {
		fp = fopen(fn,"rt");
		
		if (fp != NULL) {
			
			fseek(fp, 0, SEEK_END);
			count = ftell(fp);
			rewind(fp);
			
			if (count > 0) {
				content = (char *)malloc(sizeof(char) * (count+1));
				count = fread(content,sizeof(char),count,fp);
				content[count] = '\0';
			}
			fclose(fp);
		}
	}
	return content;
}

int textFileWrite(char *fn, char *s)
{
	FILE *fp;
	int status = 0;
	
	if (fn != NULL) {
		fp = fopen(fn,"w");
		
		if (fp != NULL) {
			
			if (fwrite(s,sizeof(char),strlen(s),fp) == strlen(s))
				status = 1;
			fclose(fp);
		}
	}
	return(status);
}

GLuint setShaders(char* vertexShaderPath, char* fragmentShaderPath)
{
	GLuint v, f, p;
	char *vs = NULL, *fs = NULL;
	
	v = glCreateShader(GL_VERTEX_SHADER);
	f = glCreateShader(GL_FRAGMENT_SHADER);
	
	vs = textFileRead(vertexShaderPath);
	fs = textFileRead(fragmentShaderPath);
	
	const char * ff = fs;
	const char * vv = vs;
	
	glShaderSource(v, 1, &vv, NULL);
	glShaderSource(f, 1, &ff, NULL);
	
	free(vs);
	free(fs);
	
	glCompileShader(v);
	glCompileShader(f);
	
	p = glCreateProgram();
	
	glAttachShader(p,v);
	glAttachShader(p,f);
	
	glLinkProgram(p);
	
//	printShaderInfoLog(v);
//	printShaderInfoLog(f);
//	printProgramInfoLog(p);
	
	return p;
}

// Prints out shader info (debugging!)
void printShaderInfoLog(GLuint obj)
{
    GLint infologLength = 0;
    GLsizei charsWritten = 0;
    char *infoLog;
	
    glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &infologLength);
	
    if (infologLength > 0)
    {
        infoLog = (char *)malloc(infologLength);
        glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
		printf("printShaderInfoLog: %s\n",infoLog);
        free(infoLog);
	}else{
		printf("Shader Info Log: OK\n");
	}
}

// Prints out shader info (debugging!)
void printProgramInfoLog(GLuint obj)
{
    GLint infologLength = 0;
    GLsizei charsWritten = 0;
    char *infoLog;
	
	glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &infologLength);
	
    if (infologLength > 0)
    {
        infoLog = (char *)malloc(infologLength);
        glGetProgramInfoLog(obj, infologLength, &charsWritten, infoLog);
		printf("printProgramInfoLog: %s\n",infoLog);
        free(infoLog);
    }else{
		printf("Program Info Log: OK\n");
	}
}
