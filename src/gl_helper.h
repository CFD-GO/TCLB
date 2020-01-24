/*
  This software contains source code provided by NVIDIA Corporation
  taken from materials for the book "CUDA by example"
  Modified by L Laniewski-Wollk for CLB project   
*/
      
#ifndef __GL_HELPER_H__
   #define __GL_HELPER_H__
   #ifdef _WIN64
      #define GLUT_NO_LIB_PRAGMA
      #pragma comment (lib, "opengl32.lib")
      #pragma comment (lib, "glut64.lib")
   #endif
   #ifdef _WIN32
      #include <GL/glut.h>
      #include <GL/glext.h>
      #define GET_PROC_ADDRESS( str ) wglGetProcAddress( str )
   #else
      #include <GL/glut.h>
      #include <GL/glext.h>
      #include <GL/glx.h>
      #include <GL/freeglut.h>
      #define GET_PROC_ADDRESS( str ) glXGetProcAddress( (const GLubyte *)str )
   #endif
#endif
