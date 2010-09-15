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
#define POLY_CUBE 0
#define POLYS     1
#define SURFELS   2
#define POINTS    3

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
	Face* face;			// face bordered by the half-edge
	Halfedge* twin;		// the half-edge adjacent and opposed
	Halfedge* next;		// next half-edge along the same face
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
	char textureName[32];
	char texturePath[32];
};

struct _Surfel {
	float3 pos;
	float3 normal;
	float area;
	float radius;       // radius of a circle with this surfel area (for displaying purpose)
	float phi;          // angle between initial normal and actual normal (for displaying purpose)
	float3 rot_axis;    // roation axis needed to correctly orient the surfel representation
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
int counter = 0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -10.0;

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
CUTBoolean createHalfedgeList( Solid* s);
Halfedge* findTwin( Halfedge* hedge, Solid* s);
float halfedgeLength( Vertex* v1, Vertex* v2);
float semiperimeter( vector<float> length);
CUTBoolean faceArea( Solid* s);
float3 normalsAverage( vector<float3> normals);
float surfelArea( Vertex* v);
CUTBoolean preprocessing( int argc, char** argv);
CUTBoolean run( int argc, char** argv);
void cleanup();
void setLighting();
CUTBoolean createPointCloud( Solid* mesh, vector<Surfel> &pc);
CUTBoolean savePointCloud( vector<Surfel> &pc, const char* path);
CUTBoolean loadPointCloud( const char* path, vector<Surfel> &pc);

// GL functionality
CUTBoolean initGL( int argc, char** argv);
CUTBoolean loadOBJ( const char* path, Solid* model);
void drawCube( float size);
void drawSolid( Solid* model);
void drawPointCloud( vector<Surfel> &cloud);
void drawSurfel( Surfel* sf);
void drawPoint( Surfel* sf);
void drawCircle( float radius);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
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
    glutMotionFunc(motion);

	// initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported("GL_VERSION_2_0 ")) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return CUTFalse;
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
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

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

	// polygonal view
		case '1':
			view_model = POLYS;
			break;
			
	// surfel view
		case '2':
			view_model = SURFELS;
			break;
			
	// point view
		case '3':
			view_model = POINTS;
			break;

	// press space to reset camera view
		case 32:
			rotate_x = rotate_y = 0.0;
			translate_z = -10.0;
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

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2;
        rotate_y += dx * 0.2;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01;
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

CUTBoolean loadOBJ(const char* path, Solid* model)
{
	CUTBoolean loaded;
	string s_line;
	const char *line;
	
	vector<string> vtn;
	vector<int> vtn_parsed;
	string vtn_element;
	unsigned int pos;
	unsigned int beg;
	unsigned int end;
	
	Vertex v_tmp;
	float2 vt_tmp;
	float3 vn_tmp;
	Face f_tmp;
	
	ifstream fp(path);
	
// 	char mtllibName[32];
// 	char mtllibPath[32];
	
	// inizializzo le stringhe
// 	int i;
// 	for (i=0; i<32; i++)
// 	{
// 		mtllibName[i] = '\0';
// 		mtllibPath[i] = '\0';
// 	}
	
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
// 			else if (strstr(line, "mtllib") != NULL)
// 			{
// 				strncpy(mtllibName, line+7, strlen(line)-8);
// 				printf("%s", mtllibName);
// 				sprintf(mtllibPath, "obj/%s", mtllibName);
// 				printf(" %d\n",loadMTL(mtllibPath,model));
// 				
// 				// reinizializzo le stringhe
// 				for (i=0; i<32; i++)
// 				{
// 					mtllibName[i] = '\0';
// 					mtllibPath[i] = '\0';
// 				}
// 			}
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
		glTexCoord2f(model->vt[model->f[i].t.x-1].x, model->vt[model->f[i].t.x-1].y);
		glVertex3f(model->v[model->f[i].v.x-1].pos.x, model->v[model->f[i].v.x-1].pos.y, model->v[model->f[i].v.x-1].pos.z);
		
		glNormal3f(model->vn[model->f[i].n.y-1].x, model->vn[model->f[i].n.y-1].y, model->vn[model->f[i].n.y-1].z);
		glTexCoord2f(model->vt[model->f[i].t.y-1].x, model->vt[model->f[i].t.y-1].y);
		glVertex3f(model->v[model->f[i].v.y-1].pos.x, model->v[model->f[i].v.y-1].pos.y, model->v[model->f[i].v.y-1].pos.z);
		
		glNormal3f(model->vn[model->f[i].n.z-1].x, model->vn[model->f[i].n.z-1].y, model->vn[model->f[i].n.z-1].z);
		glTexCoord2f(model->vt[model->f[i].t.z-1].x, model->vt[model->f[i].t.z-1].y);
		glVertex3f(model->v[model->f[i].v.z-1].pos.x, model->v[model->f[i].v.z-1].pos.y, model->v[model->f[i].v.z-1].pos.z);
		
//		if (counter == 1)
//			printf("f %d/%d/%f %d/%d/%f %d/%d/%f\n"
//				   , model->f[i].v.x, model->f[i].t.x, model->vn[model->f[i].n.x].x
//				   , model->f[i].v.y, model->f[i].t.y, model->vn[model->f[i].n.y].y
//				   , model->f[i].v.z, model->f[i].t.z, model->vn[model->f[i].n.z].z);
	}
	
//	counter++;
	
	glEnd();
	
	glDisable(GL_TEXTURE_2D);
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
	
	Halfedge hedge;
	s->he.reserve( 3 * s->f.size());
	for (unsigned int i=0; i < s->f.size(); i++) {
		hedge.vert = &s->v[ s->f[i].v.x-1 ];
		hedge.face = &s->f[i];
		s->he.push_back(hedge);
		hedge.vert = &s->v[ s->f[i].v.y-1 ];
		hedge.face = &s->f[i];
		s->he.push_back(hedge);
		hedge.vert = &s->v[ s->f[i].v.z-1 ];
		hedge.face = &s->f[i];
		s->he.push_back(hedge);
		
		s->f[i].he = &s->he.back();	// half-edge pointer of the face
	}
	
	// find half-edge successor
	for (unsigned int i=0; i < s->he.size(); i+=3) {
		s->he[ i   ].next = &s->he[ i+1 ];
		s->he[ i+1 ].next = &s->he[ i+2 ];
		s->he[ i+2 ].next = &s->he[ i   ];
	}
	
	// fill half-edge pointer of the vertex
	for (unsigned int i=0; i < s->he.size(); i++) {
		s->he[i].vert->he = &s->he[i];
	}
	
	// find the half-edge adjacent and opposed to the current half-edge
	for (unsigned int i=0; i < s->he.size(); i++) {
		s->he[i].twin = findTwin(&s->he[i], s);
	}

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

float3 normalsAverage(vector<float3> normals)
{
	float3 sum = make_float3(0,0,0);
	float nn = normals.size();
	
	for (unsigned int i=0; i < nn; i++) {
		sum.x += normals[i].x;
		sum.y += normals[i].y;
		sum.z += normals[i].z;
	}
	
	return make_float3( sum.x/nn, sum.y/nn, sum.z/nn);
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

float surfelArea(Vertex* v)
{
	Halfedge* hedge = v->he;
	float area_sum = .0f;
	
	do {
		area_sum += hedge->face->area;
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
	
	dir = "/Developer/GPU Computing/C/src/pbrSurfelsCloud/polyModels/";
	if ( cutCheckCmdLineFlag(argc, (const char**)argv, "mesh")) {
		cutGetCmdLineArgumentstr( argc, (const char**)argv, "mesh", cfilename);
	} else {
		cerr << "Please, specify a valid OBJ filename passed as command-line argument!\nSYNTAX:\n--mesh=filename.obj" << endl;
		return CUTFalse;
	}
	filename = *cfilename;
	path = dir + filename;
	if ( !loadOBJ(path.c_str(), h_imesh) )
	{
		cerr << "File \"" << filename << "\" not found!" << endl;
		return CUTFalse;
	}
	
	// winged-edge structure creation
	createHalfedgeList(h_imesh);
	
	// calculate face area (offline)
	faceArea(h_imesh);
	
	// create or load point cloud
	dir = "/Developer/GPU Computing/C/src/pbrSurfelsCloud/pointClouds/";
	if ( cutCheckCmdLineFlag(argc, (const char**)argv, "cloud_forced")) {
		cutGetCmdLineArgumentstr( argc, (const char**)argv, "cloud_forced", cfilename );
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

//	for (unsigned int i=0; i < pointCloud.size(); i++) {
//		printf("surfel %u:\t( %12g , %12g , %12g )   ||   normal( %12g , %12g , %12g )   ||   area: %g\n",
//			   i, pointCloud[i].pos.x, pointCloud[i].pos.y, pointCloud[i].pos.z,
//			   pointCloud[i].normal.x, pointCloud[i].normal.y, pointCloud[i].normal.z, pointCloud[i].area);
//	}

	// calculate theta angle of the slices of circle for surfel representation
	if ( cutCheckCmdLineFlag(argc, (const char**)argv, "surfel_slices")) {
		cutGetCmdLineArgumenti( argc, (const char**)argv, "surfel_slices", &slices);
	} else {
		slices = 12;
	}
	slices = (slices < 4 ) ? 4  : slices;
	slices = (slices > 32) ? 32 : slices;
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
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_LIGHT0);
}

void drawPointCloud(vector<Surfel> &cloud)
{
	for (unsigned int i=0; i < cloud.size(); i++) {
		if (view_model == SURFELS) {
			drawSurfel( &cloud[i]);
		} else if (view_model == POINTS) {
			drawPoint( &cloud[i]);
		}
	}
}

void drawSurfel(Surfel* sf)
{
	glPushMatrix();
		glTranslatef( sf->pos.x, sf->pos.y, sf->pos.z );
		glRotatef( sf->phi, sf->rot_axis.x, sf->rot_axis.y, sf->rot_axis.z );
		drawCircle( sf->radius );
	glPopMatrix();
}

void drawCircle(float radius)
{
	glBegin(GL_TRIANGLES);
	for (int i=0; i < slices; i++) {
		glNormal3f( .0f ,.0f, 1.f );
		glVertex3f( .0f, .0f, .0f );
		glVertex3f( radius * cos( i * theta ), radius * sin( i * theta ), .0f );
		glVertex3f( radius * cos( (i+1) * theta ), radius * sin( (i+1) * theta ), .0f );
	}
	glEnd();
}

void drawPoint(Surfel* sf)
{
	glDisable(GL_LIGHTING);
	
	glBegin(GL_POINTS);
	glVertex3f( sf->pos.x, sf->pos.y, sf->pos.z );
	glEnd();
	
	glEnable(GL_LIGHTING);
}

float dotProduct(float3 v1, float3 v2)
{
	return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

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

CUTBoolean createPointCloud(Solid* mesh, vector<Surfel> &pc)
{
	Surfel point;
	vector<float3> vnorm;
	float3 normal;
	float3 zeta = make_float3( 0,0,1 );
	
	for (unsigned int i=0; i < mesh->v.size(); i++) {
		point.pos = mesh->v[i].pos;
		
		// averaging vertex normals
		vnorm.clear();
		for (unsigned int j=0; j < mesh->f.size(); j++) {
			if (mesh->f[j].v.x-1 == i) {
				normal = mesh->vn[ mesh->f[j].n.x-1 ];
				vnorm.push_back( normal );
			} else if (mesh->f[j].v.y-1 == i) {
				normal = mesh->vn[ mesh->f[j].n.y-1 ];
				vnorm.push_back( normal );
			} else if (mesh->f[j].v.z-1 == i) {
				normal = mesh->vn[ mesh->f[j].n.z-1 ];
				vnorm.push_back( normal );
			}
		}
		normal = normalsAverage( vnorm);
		point.normal = normalizeVector( normal);
		
		// calculate surfel area and radius
		point.area = surfelArea( &mesh->v[i]);
		point.radius = sqrt( point.area / PI );
		
		// rotation angle and axis to draw surfel
		point.phi = deg( acos( dotProduct( zeta, normal)));
		point.rot_axis = crossProduct( zeta, normal);
		//		printf("glRotate( %10g, %10g, %10g, %10g )\n",point.phi,point.rot_axis.x,point.rot_axis.y,point.rot_axis.z);
		
		pc.push_back( point );
	}
	
	
	return CUTTrue;
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

	
	return CUTTrue;
}




























