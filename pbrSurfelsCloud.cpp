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
struct Vertex
{
	float x;
	float y;
	float z;
};

struct VTexture
{
	float u;
	float v;
};

struct VNormal
{
	float x;
	float y;
	float z;
};

struct Face
{
	int v1;
	int t1;
	int n1;
	
	int v2;
	int t2;
	int n2;
	
	int v3;
	int t3;
	int n3;
};

struct obj
{
	vector<Vertex> v;
	vector<VTexture> vt;
	vector<VNormal> vn;
	vector<Face> f;
	
	float ambient[3];
	float diffuse[3];
	float specular[3];
	float shininess;
	
	int textureId;
	char textureName[32];
	char texturePath[32];
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
CUTBoolean run(int argc, char** argv);
void cleanup();

// GL functionality
CUTBoolean initGL(int argc, char** argv);
void drawCube();
obj loadOBJ(const char* path);
void drawOBJ(obj model);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

// Cuda functionality
void runCuda();
void runAutoTest();

obj h_imesh;
obj h_omesh;

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
	
	// load poly mesh
 	h_imesh = loadOBJ("/Developer/GPU Computing/C/src/pbrSurfelsCloud/polyModels/die.obj");
 	
	
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
	unsigned int mem_size;
	
	// allocate memory and copy vertexes on device
	mem_size = h_imesh.v.size() * sizeof(float3);
	
	float3* d_ivertex;
	cutilSafeCall( cudaMalloc( (void**) &d_ivertex, mem_size));
//	cutilSafeCall( cudaMemcpy( d_ivertex, h_imesh.v, mem_size, cudaMemcpyHostToDevice));
	
//	float3* h_overtex = (float3*) malloc(mem_size);
//	cutilSafeCall( cudaMemcpy( h_overtex, d_ivertex, mem_size, cudaMemcpyDeviceToHost));
	
	// allocate memory and copy faces on device
	mem_size = h_imesh.f.size() * sizeof(int3);
	
	int3* h_iface = (int3*) malloc(mem_size);
	for (unsigned int i=0; i<h_imesh.f.size(); i++) {
		h_iface[i].x = h_imesh.f[i].v1;
		h_iface[i].y = h_imesh.f[i].v2;
		h_iface[i].z = h_imesh.f[i].v3;
	}
	
	int3* d_iface;
	cutilSafeCall( cudaMalloc( (void**) &d_iface, mem_size));
	cutilSafeCall( cudaMemcpy( d_iface, h_iface, mem_size, cudaMemcpyHostToDevice));
	
	float* d_ofaceArea;
	mem_size = h_imesh.f.size() * sizeof(float);
	cutilSafeCall( cudaMalloc( (void**) &d_ofaceArea, mem_size));

//	for (int i = 0; i < h_imesh.f.size(); i++) {
//		cout << h_oface[i].v1 << "/" << h_oface[i].t1 << "/" << h_oface[i].n1 << "\t";
//		cout << h_oface[i].v2 << "/" << h_oface[i].t2 << "/" << h_oface[i].n2 << "\t";
//		cout << h_oface[i].v3 << "/" << h_oface[i].t3 << "/" << h_oface[i].n3 << "\t";
//	}
	
	float4* d_serv;
	cutilSafeCall( cudaMalloc( (void**) &d_serv, h_imesh.f.size()*sizeof(float4)));
	
//	faceArea( h_imesh.f.size(), d_iface, d_ivertex, d_ofaceArea, d_serv);
	
	float4* h_serv = (float4*) malloc(h_imesh.f.size()*sizeof(float4));
	cutilSafeCall( cudaMemcpy( h_serv, d_serv, h_imesh.f.size()*sizeof(float4), cudaMemcpyDeviceToHost));
	
	for (unsigned int i=0; i<h_imesh.f.size(); i++) {
		cout << h_serv[i].x << "\t" << h_serv[i].y << "\t" << h_serv[i].z << "\t" << h_serv[i].w << endl;
	}
	
	float* h_ofaceArea = (float*) malloc(mem_size);
	
	// free memory
	cutilSafeCall( cudaFree( d_ivertex));
	cutilSafeCall( cudaFree( d_iface));
	cutilSafeCall( cudaFree( d_ofaceArea));
	cutilSafeCall( cudaFree( d_serv));
	free( h_iface);
//	free( h_overtex);
	free( h_serv);
	free( h_ofaceArea);
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
	
  	drawOBJ(h_imesh);
	
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

obj loadOBJ(const char* path)
{
	int loaded;
	string s_line;
	const char *line;
	obj model;
//	model = (obj*) malloc(sizeof(obj));
	
	vector<string> vtn;
	vector<int> vtn_parsed;
	string vtn_element;
	unsigned int pos;
	unsigned int beg;
	unsigned int end;
	
	Vertex v_tmp;
	VTexture vt_tmp;
	VNormal vn_tmp;
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
					sscanf(line, "%*c%*c %f %f", &vt_tmp.u, &vt_tmp.v);
					model.vt.push_back(vt_tmp);
				}
				// normal vertex
				else if (line[1] == 'n')
				{
					sscanf(line, "%*c%*c %f %f %f", &vn_tmp.x, &vn_tmp.y, &vn_tmp.z);
					model.vn.push_back(vn_tmp);
				}
				// vertex
				else
				{
					sscanf(line, "%*c %f %f %f", &v_tmp.x, &v_tmp.y, &v_tmp.z);
					model.v.push_back(v_tmp);
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
					vtn_element = vtn.at(i);
					beg = end = 0;
					while (end < vtn_element.size()) {
						end = vtn_element.find("/", beg);
						vtn_parsed.push_back( atoi( vtn_element.substr(beg, end-beg).c_str()));
						beg = end+1;
					}
				}
				
				for (unsigned int i=2; i < vtn.size(); i++) {
					f_tmp.v1 = vtn_parsed.at( 0 );	// primo vertice, rimane fisso
					f_tmp.v2 = vtn_parsed.at( 3*(i-1) );
					f_tmp.v3 = vtn_parsed.at( 3*i );
					
					f_tmp.t1 = vtn_parsed.at( 1 );	// primo vertice texture, rimane fisso
					f_tmp.t2 = vtn_parsed.at( 3*(i-1) +1 );
					f_tmp.t3 = vtn_parsed.at( 3*i +1 );
					
					f_tmp.n1 = vtn_parsed.at( 2 );	// primo vertice normale, rimane fisso
					f_tmp.n2 = vtn_parsed.at( 3*(i-1) +2 );
					f_tmp.n3 = vtn_parsed.at( 3*i +2 );
					
					model.f.push_back(f_tmp);
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
		loaded = 1;
	}
	else
	{
		loaded = 0;
	}
	
	fp.close();
	
	return model;
}

void drawOBJ(obj model)
{
	glMaterialfv(GL_FRONT, GL_AMBIENT, model.ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, model.diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, model.specular);
	glMaterialf(GL_FRONT, GL_SHININESS, model.shininess);
	
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, model.textureId);
	
	glBegin(GL_TRIANGLES);
	
	for (unsigned int i=0; i < model.f.size(); i++)
	{
		glNormal3f(model.vn.at(model.f.at(i).n1-1).x, model.vn.at(model.f.at(i).n1-1).y, model.vn.at(model.f.at(i).n1-1).z);
		glTexCoord2f(model.vt.at(model.f.at(i).t1-1).u, model.vt.at(model.f.at(i).t1-1).v);
		glVertex3f(model.v.at(model.f.at(i).v1-1).x, model.v.at(model.f.at(i).v1-1).y, model.v.at(model.f.at(i).v1-1).z);
		
		glNormal3f(model.vn.at(model.f.at(i).n2-1).x, model.vn.at(model.f.at(i).n2-1).y, model.vn.at(model.f.at(i).n2-1).z);
		glTexCoord2f(model.vt.at(model.f.at(i).t2-1).u, model.vt.at(model.f.at(i).t2-1).v);
		glVertex3f(model.v.at(model.f.at(i).v2-1).x, model.v.at(model.f.at(i).v2-1).y, model.v.at(model.f.at(i).v2-1).z);
		
		glNormal3f(model.vn.at(model.f.at(i).n3-1).x, model.vn.at(model.f.at(i).n3-1).y, model.vn.at(model.f.at(i).n3-1).z);
		glTexCoord2f(model.vt.at(model.f.at(i).t3-1).u, model.vt.at(model.f.at(i).t3-1).v);
		glVertex3f(model.v.at(model.f.at(i).v3-1).x, model.v.at(model.f.at(i).v3-1).y, model.v.at(model.f.at(i).v3-1).z);
		
//			printf("f %d/%d/%d %d/%d/%d %d/%d/%d\n"
//				   , model.f.at(i).v1, model.f.at(i).t1, model.f.at(i).n1
//				   , model.f.at(i).v2, model.f.at(i).t2, model.f.at(i).n2
//				   , model.f.at(i).v3, model.f.at(i).t3, model.f.at(i).n3);
	}
	
	glEnd();
	
	glDisable(GL_TEXTURE_2D);
}
