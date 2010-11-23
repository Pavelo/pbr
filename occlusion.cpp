#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
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

// includes
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
#define POLY_CUBE  0
#define POINTS     1
#define SURFELS    2
#define POLYS      3
#define OCCLUSION  4
#define OCC_DOUBLE 5
#define BENT_NORM  6

////////////////////////////////////////////////////////////////////////////////
// data structures
typedef struct _Vertex    Vertex;
typedef struct _Halfedge  Halfedge;
typedef struct _Face      Face;
typedef struct _Solid     Solid;
typedef struct _Surfel    Surfel;

struct _Vertex
{
	float3 pos;
	Halfedge* he;	// one of the half-edges starting from this vertex
};

struct _Halfedge
{
	Vertex* vert;		// vertex at the beginning of the half-edge
	float3* vn;			// vertex normal of the face the half-edge belongs to
	Face* face;			// face bordered by the half-edge
	Halfedge* twin;		// the half-edge adjacent and opposed
	Halfedge* next;		// next half-edge along the same face
	Halfedge* prev;		// previous half-edge along the same face
	float length;
};

struct _Face
{
	uint3 v;
	uint3 t;
	uint3 n;
	Halfedge* he;	// one of the half-edges bordering the face
	float area;
};

struct _Solid
{
	vector<Vertex> v;
	vector<float2> vt;
	vector<float3> vn;
	vector<Face> f;
	vector<Halfedge> he;
	
	float ambient[3];
	float diffuse[3];
	float specular[3];
	float shininess;
	
	int textureId;
	string* textureName;
	string* texturePath;
};

struct _Surfel {
	float3 pos;
	float3 normal;
	float3 bentNormal;
	float area;
	float radius;         // radius of a circle with this surfel area (for displaying purpose)
	float phi;            // angle between initial normal and actual normal (for displaying purpose)
	float3 rot_axis;      // roation axis needed to correctly orient the surfel representation
	float accessibility;  // accessibility value: percentage of the hemisphere above each surfel not occluded by geometry
	float acc_2nd_pass;   // accessibility value got with two-passes computation
};

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 512;
const unsigned int window_height = 512;

// global variables
unsigned int timer = 0;
float anim = 0.0;
Solid* h_imesh;
Solid* h_omesh;
vector<Surfel> pointCloud;
int slices;
float theta;
unsigned int view_model = POLYS;
float light_rotate_x = 0.0, light_rotate_y = 0.0f;
float light_orientation[] = {0, 1, 1, 0};
bool altPressed = false;
GLuint shaderSelected = 0, shaderID = 0;
int counter = 0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
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
CUTBoolean createHalfedgeList( Solid* s);
Halfedge* findTwin( Halfedge* hedge, Solid* s);
float halfedgeLength( Vertex* v1, Vertex* v2);
float semiperimeter( vector<float> length);
CUTBoolean faceArea( Solid* s);
float surfelArea( Vertex* v);
float3 getVector( Halfedge* he);
float3 getVector( float3 tail, float3 head);
float3 reverseVector( float3 v);
CUTBoolean preprocessing( int argc, char** argv);
CUTBoolean run( int argc, char** argv);
void cleanup();
void setLighting();
CUTBoolean createPointCloud( Solid* s, vector<Surfel> &pc);
CUTBoolean savePointCloud( vector<Surfel> &pc, const char* path);
CUTBoolean loadPointCloud( const char* path, vector<Surfel> &pc);
CUTBoolean occlusion( int passes, vector<Surfel> &pc);
float surfelShadow( Surfel* emitter, Surfel* receiver, float3 &receiverVector);

// GL functionality
CUTBoolean initGL( int argc, char** argv);
CUTBoolean loadMTL( const char* path, Solid* model);
CUTBoolean loadOBJ( const char* path, Solid* model);
void drawCube( float size);
void drawSolid( Solid* model);
void drawPointCloud( vector<Surfel> &cloud);
void drawSurfel( Surfel* sf);
void drawPoint( Surfel* sf);
void drawCircle();
void displayOcclusion( Solid* s, vector<Surfel> &pc);
void displayOcclusionDoublePass( Solid* s, vector<Surfel> &pc);
void displayBentNormal( Solid* model, vector<Surfel> &pc);

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
float magnitude( float3 vec);
float3 normalizeVector( float3 vec);
float dotProduct( float3 v1, float3 v2);
float3 crossProduct( float3 v1, float3 v2);
float3 normalsAverage( vector<float3> normals, vector<float> weights);
float clamp( float val, float inf = 0.0f, float sup = 1.0f);

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
        sprintf(fps, "%sCuda GL Interop (polygon to cloud): %3.1f fps", 
                ((g_CheckRender && g_CheckRender->IsQAReadback()) ? "AutoTest: " : ""), ifps);  

        glutSetWindowTitle(fps);
        fpsCount = 0; 
        if (g_CheckRender && !g_CheckRender->IsQAReadback()) 
	    fpsLimit = (int)MAX(ifps, 1.f);

        cutilCheckError(cutResetTimer(timer));  

        AutoQATest();
    }
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
    glutCreateWindow("Cuda GL Interop (polygon to cloud)");
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
    glClearColor(0.1, .3, .3, 1.0);
    glEnable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat) window_width / (GLfloat) window_height, 0.1, 30.0);
	
	// lighting
	setLighting();
	glEnable(GL_LIGHTING);
	
	// shading
	char vs_path[] = "/Developer/GPU Computing/C/src/occlusion/GLSL/toon.vert";
	char fs_path[] = "/Developer/GPU Computing/C/src/occlusion/GLSL/toon.frag";
	shaderID = setShaders( vs_path, fs_path);
	
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
		glRotatef( light_rotate_x, 1.0, 0.0, 0.0);
		glRotatef( light_rotate_y, 0.0, 1.0, 0.0);
		glLightfv(GL_LIGHT0, GL_POSITION, light_orientation);
	glPopMatrix();

	glUseProgram(shaderSelected);
	
	switch (view_model) {
		case POLYS:
			drawSolid(h_imesh);
			break;

		case SURFELS:
		case POINTS:
			drawPointCloud(pointCloud);
			break;

		case POLY_CUBE:
			drawCube(1.5f);
			break;

		case OCCLUSION:
			displayOcclusion(h_imesh, pointCloud);
			break;

		case OCC_DOUBLE:
			displayOcclusionDoublePass(h_imesh, pointCloud);
			break;

		case BENT_NORM:
			displayBentNormal(h_imesh, pointCloud);
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
	
	free(h_imesh);
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
			break;

	// points/vertexes view
		case '1':
			view_model = POINTS;
			break;
			
	// surfel view
		case '2':
			view_model = SURFELS;
			break;
			
	// polygonal view, occlusion rendering
		case '3':
			view_model = OCCLUSION;
			break;

	// polygonal view, double pass occlusion
		case '4':
			view_model = OCC_DOUBLE;
			break;

	// polygonal view
		case '5':
			view_model = POLYS;
			break;

	// polygonal view, bent normal rendering
		case '6':
			view_model = BENT_NORM;
			break;

	// polygonal view, occlusion and lights rendering
//		case '5':
//			view_model = GLOBAL;
//			break;
		
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
		case 's':
		case 'S':
			shaderSelected = shaderSelected ? 0 : shaderID;
			break;

	// press space to reset camera view, or alt+space to reset lights position
		case 32:
			if ( glutGetModifiers() == GLUT_ACTIVE_ALT )
			{
				light_rotate_x = light_rotate_y = 0.0;
			}
			else
			{
				rotate_x = rotate_y = translate.x = translate.y = 0.0;
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

CUTBoolean loadMTL(const char* path, Solid* model)
{
	CUTBoolean loaded;
	const char* line;
	string s_line, tn, tp;

	//assegno un identificativo per la texture da caricare per ogni modello
//	model->textureId = txtId;
//	txtId++;
	
	ifstream fp(path);
	
	if (fp.is_open())
	{
		while (!fp.eof())
		{
			getline(fp, s_line);
			line = s_line.c_str();
			
			if (s_line.find("map_Kd") == 0)
			{
				tn = s_line.substr(7);
				tp = "textures/" + tn;
				model->textureName = &tn;
				model->texturePath = &tp;
				// manca il caricamento della texture da TGA
//				strncpy(model->textureName, line + 7, strlen(line)-8);
//				printf(" [%s ", model->textureName);
//				sprintf(model->texturePath, "texture/%s", model->textureName);
//				printf("%d]", loadTGA(model->texturePath, model->textureId));
			}
			else if (s_line.find("Kd") == 0)
			{
				sscanf(line, "%*c%*c %f %f %f", &model->diffuse[0], &model->diffuse[1], &model->diffuse[2]);
			}
			else if (s_line.find("Ka") == 0)
			{
				sscanf(line, "%*c%*c %f %f %f", &model->ambient[0], &model->ambient[1], &model->ambient[2]);
			}
			else if (s_line.find("Ks") == 0)
			{
				sscanf(line, "%*c%*c %f %f %f", &model->specular[0], &model->specular[1], &model->specular[2]);
			}
			else if (s_line.find("Ns") == 0)
			{
				sscanf(line, "%*c%*c %f", &model->shininess);
			}
		}
		loaded = CUTTrue;
	}
	else
	{
		loaded = CUTFalse;
	}
	
	fp.close();
	
	return loaded;
}

CUTBoolean loadOBJ(const char* path, Solid* model)
{
	CUTBoolean loaded;
	string s_line, s_path;
	const char *line;
	
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
					model->vt.push_back(vt_tmp);
				}
				// normal vertex
				else if (line[1] == 'n')
				{
					sscanf(line, "%*c%*c %f %f %f", &vn_tmp.x, &vn_tmp.y, &vn_tmp.z);
					model->vn.push_back(vn_tmp);
				}
				// vertex
				else
				{
					sscanf(line, "%*c %f %f %f", &v_tmp.pos.x, &v_tmp.pos.y, &v_tmp.pos.z);
					model->v.push_back(v_tmp);
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
					
					model->f.push_back(f_tmp);
				}
								
				vtn_parsed.clear();
				vtn.clear();
			}
			else if ( s_line.find("mtllib") == 0 )
			{
				mtllibName = s_line.substr(7);
				mtllibPath = s_path.substr( 0, s_path.find_last_of( '/') + 1) + mtllibName;
				loadMTL( mtllibPath.c_str(), h_imesh);
			}
		}
		loaded = CUTTrue;
	}
	else
	{
		loaded = CUTFalse;
	}
	
	fp.close();
	
	return loaded;
}

void drawSolid(Solid* model)
{
//	glMaterialfv(GL_FRONT, GL_AMBIENT, model->ambient);
//	glMaterialfv(GL_FRONT, GL_DIFFUSE, model->diffuse);
//	glMaterialfv(GL_FRONT, GL_SPECULAR, model->specular);
//	glMaterialf(GL_FRONT, GL_SHININESS, model->shininess);
	
//	glEnable(GL_TEXTURE_2D);
//	glBindTexture(GL_TEXTURE_2D, model->textureId);
	
	glBegin(GL_TRIANGLES);
	for (unsigned int i=0; i < model->f.size(); i++)
	{
		glNormal3f(model->vn[model->f[i].n.x-1].x, model->vn[model->f[i].n.x-1].y, model->vn[model->f[i].n.x-1].z);
//		glTexCoord2f(model->vt[model->f[i].t.x-1].x, model->vt[model->f[i].t.x-1].y);
		glVertex3f(model->v[model->f[i].v.x-1].pos.x, model->v[model->f[i].v.x-1].pos.y, model->v[model->f[i].v.x-1].pos.z);
		
		glNormal3f(model->vn[model->f[i].n.y-1].x, model->vn[model->f[i].n.y-1].y, model->vn[model->f[i].n.y-1].z);
//		glTexCoord2f(model->vt[model->f[i].t.y-1].x, model->vt[model->f[i].t.y-1].y);
		glVertex3f(model->v[model->f[i].v.y-1].pos.x, model->v[model->f[i].v.y-1].pos.y, model->v[model->f[i].v.y-1].pos.z);
		
		glNormal3f(model->vn[model->f[i].n.z-1].x, model->vn[model->f[i].n.z-1].y, model->vn[model->f[i].n.z-1].z);
//		glTexCoord2f(model->vt[model->f[i].t.z-1].x, model->vt[model->f[i].t.z-1].y);
		glVertex3f(model->v[model->f[i].v.z-1].pos.x, model->v[model->f[i].v.z-1].pos.y, model->v[model->f[i].v.z-1].pos.z);
		
//		if (counter == 1)
//			printf("f %d/%d/%f %d/%d/%f %d/%d/%f\n"
//				   , model->f[i].v.x, model->f[i].t.x, model->vn[model->f[i].n.x].x
//				   , model->f[i].v.y, model->f[i].t.y, model->vn[model->f[i].n.y].y
//				   , model->f[i].v.z, model->f[i].t.z, model->vn[model->f[i].n.z].z);
	}
	
//	counter++;
	
	glEnd();
	
//	glDisable(GL_TEXTURE_2D);
}

void displayBentNormal(Solid* model, vector<Surfel> &pc)
{
//	glMaterialfv(GL_FRONT, GL_DIFFUSE, model->diffuse);

	glBegin(GL_TRIANGLES);
	for (unsigned int i=0; i < model->f.size(); i++)
	{
		glNormal3f(pc[model->f[i].v.x-1].bentNormal.x, pc[model->f[i].v.x-1].bentNormal.y, pc[model->f[i].v.x-1].bentNormal.z);
		glVertex3f(model->v[model->f[i].v.x-1].pos.x, model->v[model->f[i].v.x-1].pos.y, model->v[model->f[i].v.x-1].pos.z);
		
		glNormal3f(pc[model->f[i].v.y-1].bentNormal.x, pc[model->f[i].v.y-1].bentNormal.y, pc[model->f[i].v.y-1].bentNormal.z);
		glVertex3f(model->v[model->f[i].v.y-1].pos.x, model->v[model->f[i].v.y-1].pos.y, model->v[model->f[i].v.y-1].pos.z);
		
		glNormal3f(pc[model->f[i].v.z-1].bentNormal.x, pc[model->f[i].v.z-1].bentNormal.y, pc[model->f[i].v.z-1].bentNormal.z);
		glVertex3f(model->v[model->f[i].v.z-1].pos.x, model->v[model->f[i].v.z-1].pos.y, model->v[model->f[i].v.z-1].pos.z);
	}
	glEnd();
}

CUTBoolean createHalfedgeList(Solid* s)
{
	if ( s->v.empty() || s->f.empty()) {
		cerr << "vertex list or face list is empty!" << endl;
		return CUTFalse;
	}
	if ( !s->he.empty()) {
		cerr << "half-edge list already exists!" << endl;
		return CUTFalse;
	}
	
	Halfedge hedge, border, *start, *current, *border_head;
	s->he.reserve( 6 * s->f.size());
	// *** number of faces is multiplied by 6 because in the worst case
	// *** each face has 3 border half-edges in addition to its 3 ones.
	for (unsigned int i=0; i < s->f.size(); i++) {
		hedge.vert = &s->v[ s->f[i].v.x-1 ];
		hedge.vn = &s->vn[ s->f[i].n.x-1 ];
		hedge.face = &s->f[i];
		s->he.push_back(hedge);
		hedge.vert = &s->v[ s->f[i].v.y-1 ];
		hedge.vn = &s->vn[ s->f[i].n.y-1 ];
		hedge.face = &s->f[i];
		s->he.push_back(hedge);
		hedge.vert = &s->v[ s->f[i].v.z-1 ];
		hedge.vn = &s->vn[ s->f[i].n.z-1 ];
		hedge.face = &s->f[i];
		s->he.push_back(hedge);
		
		s->f[i].he = &s->he.back();	// half-edge pointer of the face
	}
	
	// find half-edge successor and predecessor
	for (unsigned int i=0; i < s->he.size(); i+=3) {
		s->he[ i   ].next = &s->he[ i+1 ];
		s->he[ i+1 ].next = &s->he[ i+2 ];
		s->he[ i+2 ].next = &s->he[ i   ];
		
		s->he[ i   ].prev = &s->he[ i+2 ];
		s->he[ i+1 ].prev = &s->he[ i   ];
		s->he[ i+2 ].prev = &s->he[ i+1 ];
	}
	
	// fill half-edge pointer of the vertex
	for (unsigned int i=0; i < s->he.size(); i++) {
		s->he[i].vert->he = &s->he[i];
	}
	
	// find the half-edge adjacent and opposed to the current half-edge
	for (unsigned int i=0; i < s->he.size(); i++) {
		s->he[i].twin = findTwin(&s->he[i], s);
	}
	
	// build any border halfe-edge of non-manifold meshes
	for (unsigned int i=0; i < s->v.size(); i++)
	{
		border_head = NULL;
		start = current = s->v[i].he;
		do {
			if ( !current->twin )
			{
				border.vert = current->next->vert;
				border.vn = NULL;
				border.face = NULL;
				border.twin = current;
				border.next = &s->he.back();
				border.prev = NULL;
				
				s->he.push_back(border);
				if ( border_head == NULL ) {
					border_head = &s->he.back();
				} else {
				}
				border_head->next = &s->he.back();

				current->twin = &s->he.back();
				current = current->next;
			}
			else {
				current = current->twin->next;
			}
		} while (current != start);
		
		current = &s->he.back();
		while ( !current->next->prev ) {
			current->next->prev = current;
			current = current->next;
		}
	}
	
//	for (unsigned int i=0; i < s->v.size(); i++) {
//		printf("v %3i %8p, he start %8p\n",i+1,&s->v[i],s->v[i].he);
//	}
//	for (unsigned int i=0; i < s->he.size(); i++) {
//		printf("he %3i %8p, v %8p ",i,&s->he[i],s->he[i].vert);
//		if (!s->he[i].face) {
//			printf("border\n");
//		} else {
//			printf("\n");
//		}
//
//	}

	return CUTTrue;
}

Halfedge* findTwin(Halfedge* hedge, Solid* s)
{
	Halfedge* twin_candidate;
	Vertex* start, *end;
	
	start = hedge->vert;
	end = hedge->next->vert;

	for (unsigned int i=0; i < s->f.size(); i++) {
		twin_candidate = s->f[i].he;
		do {
			if (start == twin_candidate->next->vert && end == twin_candidate->vert) {
				return twin_candidate;
			}
			twin_candidate = twin_candidate->next;
		} while (s->f[i].he != twin_candidate);
	}
	
	return NULL;
}

float halfedgeLength(Vertex* v1, Vertex* v2)
{
	
	float x_diff, y_diff, z_diff;
	
	x_diff = v1->pos.x - v2->pos.x;
	y_diff = v1->pos.y - v2->pos.y;
	z_diff = v1->pos.z - v2->pos.z;
	
	return sqrt( x_diff*x_diff + y_diff*y_diff + z_diff*z_diff );
}

float semiperimeter( vector<float> length)
{
	float perimeter = 0;
	
	for (unsigned int i=0; i < length.size(); i++) {
		perimeter += length.at(i);
	}
	
	return perimeter * .5f;
}

CUTBoolean faceArea(Solid* s)
{
	Halfedge* he_temp;
	vector<float> he_length;
	float sp;

	// calculate half-edge length
	for (unsigned int i=0; i < s->v.size(); i++) {
		he_temp = s->v[i].he;
		do {
			he_temp->length = halfedgeLength( he_temp->vert, he_temp->twin->vert);
			he_temp = he_temp->twin->next;
		} while (he_temp != s->v[i].he);
	}
	
	// for each face calculate its area
	for (unsigned int i=0; i < s->f.size(); i++) {
		he_length.clear();
		he_temp = s->f[i].he;
		do {
			he_length.push_back( he_temp->length);
			he_temp = he_temp->next;
		} while (he_temp != s->f[i].he);
		
		sp = semiperimeter( he_length);
		
		// area is obtained with Heron's formula
		s->f[i].area = sqrt( sp * (sp - he_length.at(0)) * (sp - he_length.at(1)) * (sp - he_length.at(2)) );
	}
	
	return CUTTrue;
}

// normalized average of a group of normals
float3 normalsAverage(vector<float3> normals, vector<float> weights)
{
	float3 sum = make_float3(0,0,0);
	unsigned int nn = normals.size();

	for (unsigned int i=0; i < nn; i++) {
		sum.x += weights[i] * normals[i].x;
		sum.y += weights[i] * normals[i].y;
		sum.z += weights[i] * normals[i].z;
	}

	return normalizeVector( sum);
}

float magnitude(float3 vec)
{
	return sqrt( vec.x*vec.x + vec.y*vec.y + vec.z*vec.z );
}

float3 normalizeVector(float3 vec)
{
	float mag = magnitude( vec);
	
	return make_float3( vec.x/mag, vec.y/mag, vec.z/mag );
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

float surfelArea(Vertex* v)
{
	Halfedge* hedge = v->he;
	float area_sum = .0f;
	
	do {
		if ( hedge->face )
		{
			area_sum += hedge->face->area;
		}
		hedge = hedge->twin->next;
	} while (hedge != v->he);
	
	return area_sum * .3333333333f;
}

CUTBoolean preprocessing(int argc, char** argv)
{
	// load poly mesh
	string dir, filename, path, msg;
	char** cfilename;
	
	cfilename = (char**) malloc(sizeof(char));
	h_imesh = (Solid*) malloc(sizeof(Solid));
	
	dir = "/Developer/GPU Computing/C/src/occlusion/polyModels/";
	if ( cutCheckCmdLineFlag(argc, (const char**)argv, "mesh")) {
		cutGetCmdLineArgumentstr( argc, (const char**)argv, "mesh", cfilename);
	} else {
		cerr << "Please, specify a valid OBJ filename passed as command-line argument!\nSYNTAX:\n--mesh=filename.obj" << endl;
		return CUTFalse;
	}
	filename = *cfilename;
	path = dir + filename;
	cout << "Loading mesh... ";
	cout.flush();
	if ( !loadOBJ(path.c_str(), h_imesh) )
	{
		cerr << "File \"" << filename << "\" not found!" << endl;
		return CUTFalse;
	}
	cout << "done" << endl;
	
	// half-edge structure creation
	cout << "Creating half-edge structure... ";
	cout.flush();
	createHalfedgeList(h_imesh);
	cout << "done" << endl;
	
	// calculate face area (offline)
	faceArea(h_imesh);
	
	// create or load point cloud
	cout << "Loading surfels cloud... ";
	cout.flush();
	dir = "/Developer/GPU Computing/C/src/occlusion/pointClouds/";
	if ( cutCheckCmdLineFlag(argc, (const char**)argv, "cloud")) {
		cutGetCmdLineArgumentstr( argc, (const char**)argv, "cloud", cfilename );
		filename = *cfilename;
		path = dir + filename;

		if ( !loadPointCloud(path.c_str(), pointCloud)) {
			do {
				cout << "File \"" << filename << "\" does not exist! Create it? ";
				getline(cin, msg);
				
				if (msg[0] == 'y' || msg[0] == 'Y') {
					cout << "Creating surfels cloud..." << endl;
					createPointCloud( h_imesh, pointCloud);
					cout << "Saving it to \"" << filename << "\"..." << endl;
					savePointCloud( pointCloud, path.c_str());
				} else if (msg[0] == 'n' || msg[0] == 'N') {
					cerr << "Cannot load any surfel cloud. Aborting..." << endl;
					return CUTFalse;
				} else {
					cout << "Answer with 'yes' or 'no'. ";
				}
			} while (msg.find_first_of("ynYN") != 0);
		} else {
			cout << "\"" << filename << "\" loaded correctly" << endl;
		}
	} else {
		filename.replace( filename.length()-4, filename.length(), ".sfc" );
		path = dir + filename;
		
		if ( !loadPointCloud( path.c_str(), pointCloud)) {
			cout << "Creating surfels cloud..." << endl;
			createPointCloud( h_imesh, pointCloud);
			cout << "Saving it to \"" << filename << "\"..." << endl;
			savePointCloud( pointCloud, path.c_str());
		} else {
			cout << "\"" << filename << "\" loaded correctly" << endl;
		}
	}

	// calculate occlusion
	int multipass;
	if ( cutCheckCmdLineFlag( argc, (const char**)argv, "multipass")) {
		cutGetCmdLineArgumenti( argc, (const char**)argv, "multipass", &multipass);
	} else {
		multipass = 1;
	}
	cout << "Computing occlusion... ";
	cout.flush();
	occlusion( multipass, pointCloud);
	cout << "done" << endl;

	// calculate theta angle of the slices of circle for surfel representation
	if ( cutCheckCmdLineFlag(argc, (const char**)argv, "surfel_slices")) {
		cutGetCmdLineArgumenti( argc, (const char**)argv, "surfel_slices", &slices);
	} else {
		slices = 24;
	}
	slices = clamp( slices, 4, 32);
	theta = 2*PI / (float)slices;
	
	free(cfilename);
	
	return CUTTrue;
}

void setLighting()
{
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	glColorMaterial(GL_FRONT, GL_DIFFUSE);
	glShadeModel(GL_SMOOTH);
	
	glEnable(GL_CULL_FACE);
	glEnable(GL_LIGHT0);
}

void drawPointCloud(vector<Surfel> &cloud)
{
		glDisable(GL_LIGHTING);
	if (view_model == POINTS) {
		glBegin(GL_POINTS);
	}
	for (unsigned int i=0; i < cloud.size(); i++)
	{
		if (view_model == SURFELS)
		{
			glColor3f( cloud[i].accessibility, cloud[i].accessibility, cloud[i].accessibility);
			drawSurfel( &cloud[i]);
		}
		else if (view_model == POINTS)
		{
			drawPoint( &cloud[i]);
		}
	}
	if (view_model == POINTS) {
		glEnd();
	}
		glEnable(GL_LIGHTING);
}

void drawSurfel(Surfel* sf)
{
	glPushMatrix();
		glTranslatef( sf->pos.x, sf->pos.y, sf->pos.z );
		glRotatef( sf->phi, sf->rot_axis.x, sf->rot_axis.y, sf->rot_axis.z );
		glScalef( sf->radius, sf->radius, 1.f);
		drawCircle();
	glPopMatrix();
}

void drawCircle()
{
	glBegin(GL_TRIANGLE_FAN);
		glNormal3f( .0f, .0f, 1.f );
		glVertex3f( .0f, .0f, .0f );
	for (int i=0; i <= slices; i++) {
		glVertex3f( cos( i * theta ), sin( i * theta ), .0f );
	}
	glEnd();
}

void drawPoint(Surfel* sf)
{
	glColor3f( 1.f, 1.f, 1.f);
	glVertex3f( sf->pos.x, sf->pos.y, sf->pos.z );
	
}

// dot product of NORMALIZED vectors
float dotProduct(float3 v1, float3 v2)
{
	return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

// cross product of NORMALIZED vectors
float3 crossProduct(float3 v1, float3 v2)
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

// Get the normalized joint vector between the two vertexes connected by the half-edge
float3 getVector(Halfedge* he)
{
	float3 head, tail, res;
	
	tail = he->vert->pos;
	head = he->twin->vert->pos;
	res.x = head.x - tail.x;
	res.y = head.y - tail.y;
	res.z = head.z - tail.z;

	return normalizeVector( res);
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

CUTBoolean savePointCloud(vector<Surfel> &pc, const char* path)
{
	ofstream file;
	string line;
	
	file.open(path);
	
	if (file.is_open()) {
		line = "";
		
		file << "SFC" << endl;
		file << pc.size() << endl;
		for (unsigned int i=0; i < pc.size(); i++) {
			file << pc[i].pos.x<<" "<<pc[i].pos.y<<" "<<pc[i].pos.z << endl;
		}
		for (unsigned int i=0; i < pc.size(); i++) {
			file << pc[i].normal.x<<" "<<pc[i].normal.y<<" "<<pc[i].normal.z << endl;
		}
		for (unsigned int i=0; i < pc.size(); i++) {
			file << pc[i].area<<" "<<pc[i].radius<<" "<<pc[i].phi << endl;
		}
		for (unsigned int i=0; i < pc.size(); i++) {
			file << pc[i].rot_axis.x<<" "<<pc[i].rot_axis.y<<" "<<pc[i].rot_axis.z << endl;
		}
		
	} else {
		cerr << "Error opening file \"" << path << "\"" << endl;
		return CUTFalse;
	}

	file.close();
	
	cout << "Surfels cloud saved correctly" << endl;

	return CUTTrue;
}

CUTBoolean createPointCloud(Solid* s, vector<Surfel> &pc)
{
	Surfel point;
	Halfedge* cur_he, *next_he;
	vector<float3> vnorm;
	vector<float> weight;
	float w;
	float3 zeta = make_float3( 0,0,1 );
	
	for (unsigned int i=0; i < s->v.size(); i++) {
		point.pos = s->v[i].pos;
		
		// averaging vertex normals
		vnorm.clear();
		weight.clear();
		cur_he = s->v[i].he;
		do {
			next_he = cur_he->prev->twin;
			if ( cur_he->vn ) {
				// collect vertex normals
				vnorm.push_back( *cur_he->vn);
				// evaluate the angle between the two edges starting from the vertex
				w = acos( dotProduct( getVector( cur_he), getVector( next_he)));
				// multiply the angle and the face area to get the weigth
//				w *= cur_he->face->area;
				weight.push_back(w);
			}
			cur_he = next_he;
		} while ( cur_he != s->v[i].he);

		point.normal = normalsAverage( vnorm, weight);
		
		// calculate surfel area and radius
		point.area = surfelArea( &s->v[i]);
		point.radius = sqrt( point.area / PI );
		
		// rotation angle and axis to draw surfel
		point.phi = deg( acos( dotProduct( zeta, point.normal)));
		point.rot_axis = crossProduct( zeta, point.normal);
		
		pc.push_back( point );
	}
	
	return CUTTrue;
}

float surfelShadow(Surfel* receiver, Surfel* emitter, float3 &receiverVector)
{
	float distance, dSquared;
	float3 v, emitterVector;
	
	v = getVector( emitter->pos, receiver->pos);
	distance = magnitude( v);
	dSquared = distance * distance;
	emitterVector = normalizeVector( v);
	receiverVector = reverseVector( emitterVector);

	return (1 - 1 / sqrt( (emitter->area / PI) / dSquared + 1))
			* clamp( dotProduct( emitter->normal, emitterVector))
			* clamp( 4 * dotProduct( receiver->normal, receiverVector));
}

CUTBoolean occlusion(int passes, vector<Surfel> &pc)
{
	float sshadow, sshadow_total;
	float3 recVec;

	sshadow = 0.0f;
	for (int k=1; k <= passes; k++)
	{
		for (unsigned int i=0; i < pc.size(); i++)
		{
			sshadow_total = .0f;
			pc[i].bentNormal = pc[i].normal;
			for (unsigned int j=0; j < pc.size(); j++)
			{
				if (i!=j) {
					if (k == 1)
					{
						sshadow = surfelShadow( &pc[i], &pc[j], recVec);
						sshadow_total += sshadow;
					}
					else if (k == 2)
					{
						sshadow = surfelShadow( &pc[i], &pc[j], recVec) * pc[j].accessibility;
						sshadow_total += sshadow;
						pc[i].bentNormal.x -= sshadow * recVec.x;
						pc[i].bentNormal.y -= sshadow * recVec.y;
						pc[i].bentNormal.z -= sshadow * recVec.z;
					}
				}
			}
			if (k == 1)
			{
				pc[i].accessibility = 1.f - sshadow_total;
				
			}
			else if (k == 2)
			{
				pc[i].acc_2nd_pass = 1.f - sshadow_total;
				pc[i].bentNormal = normalizeVector( pc[i].bentNormal);
			}
		}
	}

	return CUTTrue;
}

void displayOcclusion(Solid* s, vector<Surfel> &pc)
{
	glDisable(GL_LIGHTING);
	
	glBegin(GL_TRIANGLES);
	for (unsigned int i=0; i < s->f.size(); i++)
	{
		glColor3f( pc[s->f[i].v.x-1].accessibility, pc[s->f[i].v.x-1].accessibility, pc[s->f[i].v.x-1].accessibility);
		glVertex3f(s->v[s->f[i].v.x-1].pos.x, s->v[s->f[i].v.x-1].pos.y, s->v[s->f[i].v.x-1].pos.z);
		
		glColor3f( pc[s->f[i].v.y-1].accessibility, pc[s->f[i].v.y-1].accessibility, pc[s->f[i].v.y-1].accessibility);
		glVertex3f(s->v[s->f[i].v.y-1].pos.x, s->v[s->f[i].v.y-1].pos.y, s->v[s->f[i].v.y-1].pos.z);
		
		glColor3f( pc[s->f[i].v.z-1].accessibility, pc[s->f[i].v.z-1].accessibility, pc[s->f[i].v.z-1].accessibility);
		glVertex3f(s->v[s->f[i].v.z-1].pos.x, s->v[s->f[i].v.z-1].pos.y, s->v[s->f[i].v.z-1].pos.z);
	}
	glEnd();
	
	glEnable(GL_LIGHTING);
}

void displayOcclusionDoublePass(Solid* s, vector<Surfel> &pc)
{
	glDisable(GL_LIGHTING);
	
	glBegin(GL_TRIANGLES);
	for (unsigned int i=0; i < s->f.size(); i++)
	{
		glColor3f( pc[s->f[i].v.x-1].acc_2nd_pass, pc[s->f[i].v.x-1].acc_2nd_pass, pc[s->f[i].v.x-1].acc_2nd_pass);
		glVertex3f(s->v[s->f[i].v.x-1].pos.x, s->v[s->f[i].v.x-1].pos.y, s->v[s->f[i].v.x-1].pos.z);
		
		glColor3f( pc[s->f[i].v.y-1].acc_2nd_pass, pc[s->f[i].v.y-1].acc_2nd_pass, pc[s->f[i].v.y-1].acc_2nd_pass);
		glVertex3f(s->v[s->f[i].v.y-1].pos.x, s->v[s->f[i].v.y-1].pos.y, s->v[s->f[i].v.y-1].pos.z);
		
		glColor3f( pc[s->f[i].v.z-1].acc_2nd_pass, pc[s->f[i].v.z-1].acc_2nd_pass, pc[s->f[i].v.z-1].acc_2nd_pass);
		glVertex3f(s->v[s->f[i].v.z-1].pos.x, s->v[s->f[i].v.z-1].pos.y, s->v[s->f[i].v.z-1].pos.z);
	}
	glEnd();
	
	glEnable(GL_LIGHTING);
}

CUTBoolean loadPointCloud(const char* path, vector<Surfel> &pc)
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
		for (unsigned int i=0; i < n_surfels; i++)
		{
			getline(file, s_line);
			line = s_line.c_str();
			sscanf( line, "%g %g %g", &pc[i].area, &pc[i].radius, &pc[i].phi);
		}
		for (unsigned int i=0; i < n_surfels; i++)
		{
			getline(file, s_line);
			line = s_line.c_str();
			sscanf( line, "%g %g %g", &pc[i].rot_axis.x, &pc[i].rot_axis.y, &pc[i].rot_axis.z);
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
	glUseProgram(p);
	
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
