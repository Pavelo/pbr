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
};

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 512;
const unsigned int window_height = 512;

float anim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

unsigned int timer = 0;

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
float magnitude( float3 vec);
float3 normalizeVector( float3 vec);
float3 normalsAverage( vector<float3> normals);
float surfelArea( Vertex* v);
CUTBoolean run( int argc, char** argv);
void cleanup();

// GL functionality
CUTBoolean initGL(int argc, char** argv);
void drawCube();
CUTBoolean loadOBJ(const char* path, Solid* model);
void drawSolid(Solid* model);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

// Cuda functionality
void runCuda();
void runAutoTest();

Solid* h_imesh;
Solid* h_omesh;

int counter=0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    run(argc, argv);
    
    cudaThreadExit();
    
    cutilExit(argc, argv);
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
        float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
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
	// load poly mesh
	string mesh_dir, mesh_name, obj_path;
	char** mesh_cname;
	
	mesh_cname = (char**) malloc(sizeof(char));
	h_imesh = (Solid*) malloc(sizeof(Solid));
	
	mesh_dir = "/Developer/GPU Computing/C/src/pbrSurfelsCloud/polyModels/";
	if ( cutCheckCmdLineFlag(argc, (const char**)argv, "mesh")) {
		cutGetCmdLineArgumentstr( argc, (const char**)argv, "mesh", mesh_cname);
	} else {
		cout << "Please, specify a valid OBJ filename passed as command-line argument!\nSYNTAX:\n--mesh=filename.obj" << endl;
		return CUTFalse;
	}
	mesh_name = *mesh_cname;
	obj_path = mesh_dir + mesh_name;
	loadOBJ(obj_path.c_str(), h_imesh);
	free(mesh_cname);
	
	// winged-edge structure creation
	createHalfedgeList(h_imesh);

	// calculate face area (offline)
	faceArea(h_imesh);
	
	// create point cloud
	vector<Surfel> pointCloud;
	Surfel point;
	vector<float3> vnorm;
	float3 normal;
	
	for (unsigned int i=0; i < h_imesh->v.size(); i++) {
		point.pos = h_imesh->v[i].pos;
		
		// averaging vertex normals
		vnorm.clear();
		for (unsigned int j=0; j < h_imesh->f.size(); j++) {
			if (h_imesh->f[j].v.x-1 == i) {
				normal = h_imesh->vn[ h_imesh->f[j].n.x-1 ];
				vnorm.push_back( normal );
			} else if (h_imesh->f[j].v.y-1 == i) {
				normal = h_imesh->vn[ h_imesh->f[j].n.y-1 ];
				vnorm.push_back( normal );
			} else if (h_imesh->f[j].v.z-1 == i) {
				normal = h_imesh->vn[ h_imesh->f[j].n.z-1 ];
				vnorm.push_back( normal );
			}
		}
		normal = normalsAverage( vnorm);
		point.normal = normalizeVector( normal);
		
		// calculate surfel area
		point.area = surfelArea( &h_imesh->v[i]);
		
		pointCloud.push_back( point );
	}
	
	for (unsigned int i=0; i < pointCloud.size(); i++) {
		printf("surfel %u:\t( %12g , %12g , %12g )   ||   normal( %12g , %12g , %12g )   ||   area: %g\n",
			   i, pointCloud[i].pos.x, pointCloud[i].pos.y, pointCloud[i].pos.z,
			   pointCloud[i].normal.x, pointCloud[i].normal.y, pointCloud[i].normal.z, pointCloud[i].area);
	}
	
	// functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
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
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat) window_width / (GLfloat) window_height, 0.1, 10.0);

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

// 	drawCube();
	
  	drawSolid(h_imesh);
	
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
    case 27 :
    case 'q' :
    case 'Q' :
        exit(0);
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

void drawCube()
{
	glBegin(GL_QUADS);
		glVertex3f( 1.0f, 1.0f,-1.0f);			// Top Right Of The Quad (Top)
		glVertex3f(-1.0f, 1.0f,-1.0f);			// Top Left Of The Quad (Top)
		glVertex3f(-1.0f, 1.0f, 1.0f);			// Bottom Left Of The Quad (Top)
		glVertex3f( 1.0f, 1.0f, 1.0f);			// Bottom Right Of The Quad (Top)

		glVertex3f( 1.0f,-1.0f, 1.0f);			// Top Right Of The Quad (Bottom)
		glVertex3f(-1.0f,-1.0f, 1.0f);			// Top Left Of The Quad (Bottom)
		glVertex3f(-1.0f,-1.0f,-1.0f);			// Bottom Left Of The Quad (Bottom)
		glVertex3f( 1.0f,-1.0f,-1.0f);			// Bottom Right Of The Quad (Bottom)
		
		glVertex3f( 1.0f, 1.0f, 1.0f);			// Top Right Of The Quad (Front)
		glVertex3f(-1.0f, 1.0f, 1.0f);			// Top Left Of The Quad (Front)
		glVertex3f(-1.0f,-1.0f, 1.0f);			// Bottom Left Of The Quad (Front)
		glVertex3f( 1.0f,-1.0f, 1.0f);			// Bottom Right Of The Quad (Front)
		
		glVertex3f( 1.0f,-1.0f,-1.0f);			// Bottom Left Of The Quad (Back)
		glVertex3f(-1.0f,-1.0f,-1.0f);			// Bottom Right Of The Quad (Back)
		glVertex3f(-1.0f, 1.0f,-1.0f);			// Top Right Of The Quad (Back)
		glVertex3f( 1.0f, 1.0f,-1.0f);			// Top Left Of The Quad (Back)

		glVertex3f(-1.0f, 1.0f, 1.0f);			// Top Right Of The Quad (Left)
		glVertex3f(-1.0f, 1.0f,-1.0f);			// Top Left Of The Quad (Left)
		glVertex3f(-1.0f,-1.0f,-1.0f);			// Bottom Left Of The Quad (Left)
		glVertex3f(-1.0f,-1.0f, 1.0f);			// Bottom Right Of The Quad (Left)
		
		glVertex3f( 1.0f, 1.0f,-1.0f);			// Top Right Of The Quad (Right)
		glVertex3f( 1.0f, 1.0f, 1.0f);			// Top Left Of The Quad (Right)
		glVertex3f( 1.0f,-1.0f, 1.0f);			// Bottom Left Of The Quad (Right)
		glVertex3f( 1.0f,-1.0f,-1.0f);			// Bottom Right Of The Quad (Right)
	glEnd();
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
	glMaterialfv(GL_FRONT, GL_AMBIENT, model->ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, model->diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, model->specular);
	glMaterialf(GL_FRONT, GL_SHININESS, model->shininess);
	
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, model->textureId);
	
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
		cout << "vertex list or face list is empty!" << endl;
		return CUTFalse;
	}
	if ( !s->he.empty()) {
		cout << "half-edge list already exists!" << endl;
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
