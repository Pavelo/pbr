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
	
	int v4;
	int t4;
	int n4;
};

struct Face3
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
	Vertex v[50000];
	VTexture vt[50000];
	VNormal vn[50000];
	Face f[50000];
	Face3 f3[50000];
	int vCount;
	int fCount;
	int f3Count;
	
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
extern "C" void launch_kernel();

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
CUTBoolean runTest(int argc, char** argv);
void cleanup();

// GL functionality
CUTBoolean initGL(int argc, char** argv);
void drawCube();
obj* loadOBJ(const char* path);
void drawOBJ(obj* model);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

// Cuda functionality
void runCuda();
void runAutoTest();

obj* h_imesh;
obj* h_omesh;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    runTest(argc, argv);
    
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
CUTBoolean runTest(int argc, char** argv)
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
 	h_imesh = loadOBJ("/Developer/GPU Computing/C/src/pbrLoadPoly/polyModels/cactus.obj");
 	
	
	
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

obj* loadOBJ(const char* path)
{
	int loaded;
	string s_line;
	const char *line;
	obj* model;
	model = (obj*) malloc(sizeof(obj));
	
	model->vCount = 0;
	int vtCount = 0;
	int vnCount = 0;
	model->fCount = 0;
	model->f3Count = 0;
	
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
					sscanf(line, "%*c%*c %f %f", &model->vt[vtCount].u, &model->vt[vtCount].v);
					vtCount++;
				}
				// normal vertex
				else if (line[1] == 'n')
				{
					sscanf(line, "%*c%*c %f %f %f", &model->vn[vnCount].x, &model->vn[vnCount].y, &model->vn[vnCount].z);
					vnCount++;
				}
				// vertex
				else
				{
					sscanf(line, "%*c %f %f %f", &model->v[model->vCount].x, &model->v[model->vCount].y, &model->v[model->vCount].z);
					model->vCount++;
				}
			}
			// face
			else if (line[0] == 'f')
			{
				int n = 0;
				int spaceCount = 0;
				while (line[n] != '\0')
				{
					if (line[n]==' ')
						spaceCount++;
					n++;
				}
				if (spaceCount == 3)
				{
					sscanf(line, "%*c %d/%d/%d %d/%d/%d %d/%d/%d",
						   &model->f3[model->f3Count].v1, &model->f3[model->f3Count].t1, &model->f3[model->f3Count].n1,
						   &model->f3[model->f3Count].v2, &model->f3[model->f3Count].t2, &model->f3[model->f3Count].n2,
						   &model->f3[model->f3Count].v3, &model->f3[model->f3Count].t3, &model->f3[model->f3Count].n3);
					model->f3Count++;
				}
				else if (spaceCount == 4)
				{
					sscanf(line, "%*c %d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d",
						   &model->f[model->fCount].v1, &model->f[model->fCount].t1, &model->f[model->fCount].n1,
						   &model->f[model->fCount].v2, &model->f[model->fCount].t2, &model->f[model->fCount].n2,
						   &model->f[model->fCount].v3, &model->f[model->fCount].t3, &model->f[model->fCount].n3,
						   &model->f[model->fCount].v4, &model->f[model->fCount].t4, &model->f[model->fCount].n4);
					model->fCount++;
				}
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

void drawOBJ(obj* model)
{
	glMaterialfv(GL_FRONT, GL_AMBIENT, model->ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, model->diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, model->specular);
	glMaterialf(GL_FRONT, GL_SHININESS, model->shininess);
	
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, model->textureId);
	
	glBegin(GL_QUADS);
	
	int i;
	for (i=0; i<model->fCount; i++)
	{
		glNormal3f(model->vn[model->f[i].n1-1].x, model->vn[model->f[i].n1-1].y, model->vn[model->f[i].n1-1].z);
		glTexCoord2f(model->vt[model->f[i].t1-1].u, model->vt[model->f[i].t1-1].v);
		glVertex3f(model->v[model->f[i].v1-1].x, model->v[model->f[i].v1-1].y, model->v[model->f[i].v1-1].z);
		
		glNormal3f(model->vn[model->f[i].n2-1].x, model->vn[model->f[i].n2-1].y, model->vn[model->f[i].n2-1].z);
		glTexCoord2f(model->vt[model->f[i].t2-1].u, model->vt[model->f[i].t2-1].v);
		glVertex3f(model->v[model->f[i].v2-1].x, model->v[model->f[i].v2-1].y, model->v[model->f[i].v2-1].z);
		
		glNormal3f(model->vn[model->f[i].n3-1].x, model->vn[model->f[i].n3-1].y, model->vn[model->f[i].n3-1].z);
		glTexCoord2f(model->vt[model->f[i].t3-1].u, model->vt[model->f[i].t3-1].v);
		glVertex3f(model->v[model->f[i].v3-1].x, model->v[model->f[i].v3-1].y, model->v[model->f[i].v3-1].z);
		
		glNormal3f(model->vn[model->f[i].n4-1].x, model->vn[model->f[i].n4-1].y, model->vn[model->f[i].n4-1].z);
		glTexCoord2f(model->vt[model->f[i].t4-1].u, model->vt[model->f[i].t4-1].v);
		glVertex3f(model->v[model->f[i].v4-1].x, model->v[model->f[i].v4-1].y, model->v[model->f[i].v4-1].z);
	}
	
	glEnd();
	
	glBegin(GL_TRIANGLES);
	
	for (i=0; i<model->f3Count; i++)
	{
		glNormal3f(model->vn[model->f3[i].n1-1].x, model->vn[model->f3[i].n1-1].y, model->vn[model->f3[i].n1-1].z);
		glTexCoord2f(model->vt[model->f3[i].t1-1].u, model->vt[model->f3[i].t1-1].v);
		glVertex3f(model->v[model->f3[i].v1-1].x, model->v[model->f3[i].v1-1].y, model->v[model->f3[i].v1-1].z);
		
		glNormal3f(model->vn[model->f3[i].n2-1].x, model->vn[model->f3[i].n2-1].y, model->vn[model->f3[i].n2-1].z);
		glTexCoord2f(model->vt[model->f3[i].t2-1].u, model->vt[model->f3[i].t2-1].v);
		glVertex3f(model->v[model->f3[i].v2-1].x, model->v[model->f3[i].v2-1].y, model->v[model->f3[i].v2-1].z);
		
		glNormal3f(model->vn[model->f3[i].n3-1].x, model->vn[model->f3[i].n3-1].y, model->vn[model->f3[i].n3-1].z);
		glTexCoord2f(model->vt[model->f3[i].t3-1].u, model->vt[model->f3[i].t3-1].v);
		glVertex3f(model->v[model->f3[i].v3-1].x, model->v[model->f3[i].v3-1].y, model->v[model->f3[i].v3-1].z);
	}
	
	glEnd();
	
	glDisable(GL_TEXTURE_2D);
}
