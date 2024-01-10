/*
  This software contains source code provided by NVIDIA Corporation
  taken from materials for the book "CUDA by example"
  Modified by L Laniewski-Wollk for CLB project
*/


#ifndef __GPU_ANIM_H__
#define __GPU_ANIM_H__

#include "../config.h"

#ifdef GRAPHICS

#include "gl_helper.h"

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <iostream>


extern PFNGLBINDBUFFERARBPROC    glBindBuffer;
extern PFNGLDELETEBUFFERSARBPROC glDeleteBuffers;
extern PFNGLGENBUFFERSARBPROC    glGenBuffers;
extern PFNGLBUFFERDATAARBPROC    glBufferData;


struct GPUAnimBitmap {
    GLuint  bufferObj;
    cudaGraphicsResource *resource;
    int     width, height;
    void    *dataBlock;
    int (*fAnim)(uchar4*,void*,int);
    void (*animExit)(void*);
    void (*clickDrag)(void*,int,int,int,int);
    void (*move)(void*,int,int,int,int);
    int     dragStartX, dragStartY;

    GPUAnimBitmap( int w, int h, void *d = NULL ) {
        width = w;
        height = h;
        dataBlock = d;
        clickDrag = NULL;
        move = NULL;

        // a bug in the Windows GLUT implementation prevents us from
        // passing zero arguments to glutInit()
        int c=1;
        char* dummy = "";
        glutInit( &c, &dummy );
        glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
        glutInitWindowSize( width, height );
        glutCreateWindow( "bitmap" );

        glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
        glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
        glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
        glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

        glGenBuffers( 1, &bufferObj );
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
        glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, width * height * 4,
                      NULL, GL_DYNAMIC_DRAW_ARB );

        HANDLE_ERROR( cudaGraphicsGLRegisterBuffer( &resource, bufferObj, cudaGraphicsMapFlagsNone ) );
    }

    ~GPUAnimBitmap() {
        free_resources();
    }

    void free_resources( void ) {
        HANDLE_ERROR( cudaGraphicsUnregisterResource( resource ) );

        glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
        glDeleteBuffers( 1, &bufferObj );
    }


    long image_size( void ) const { return width * height * 4; }

    void click_drag( void (*f)(void*,int,int,int,int)) {
        clickDrag = f;
    }

    void mouse_move( void (*f)(void*,int,int,int,int)) {
        move = f;
    }

    void anim_and_exit( int (*f)(uchar4*,void*,int), void(*e)(void*) ) {
        GPUAnimBitmap**   bitmap = get_bitmap_ptr();
        *bitmap = this;
        fAnim = f;
        animExit = e;

        glutKeyboardFunc( Key );
        glutDisplayFunc( Draw );
        if (clickDrag != NULL || move != NULL)
            glutMouseFunc( mouse_func );
        if (move != NULL)
            glutMotionFunc( move_func );
        glutIdleFunc( idle_func );
//        glutMainLoop();
    }

    // static method used for glut callbacks
    static GPUAnimBitmap** get_bitmap_ptr( void ) {
        static GPUAnimBitmap*   gBitmap;
        return &gBitmap;
    }

    // static method used for glut callbacks
    static void mouse_func( int button, int state,
                            int mx, int my ) {
        if (button == GLUT_LEFT_BUTTON) {
            GPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
            if (state == GLUT_DOWN) {
                bitmap->dragStartX = mx;
                bitmap->dragStartY = my;
            } else if (state == GLUT_UP) {
                if (bitmap->clickDrag != NULL) {
                bitmap->clickDrag( bitmap->dataBlock,
                                   bitmap->dragStartX,
                                   bitmap->dragStartY,
                                   mx, my );
                }
                if (bitmap->move != NULL) {
                bitmap->move( bitmap->dataBlock,
                                   bitmap->dragStartX,
                                   bitmap->dragStartY,
                                   mx, my );
                }
                bitmap->dragStartX = mx;
                bitmap->dragStartY = my;
            }
        }
    }

    static void move_func( int mx, int my ) {
            GPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
                if ((mx < 0) || (mx >= bitmap->width) || (my < 0) || (my >= bitmap->height)) return;
                bitmap->move( bitmap->dataBlock,
                                   bitmap->dragStartX,
                                   bitmap->dragStartY,
                                   mx, my );
                bitmap->dragStartX = mx;
                bitmap->dragStartY = my;
    }

    // static method used for glut callbacks
    static void idle_func( void ) {
        static int ticks = 1;
        GPUAnimBitmap*  bitmap = *(get_bitmap_ptr());
        uchar4*         devPtr;
        size_t  size;

        HANDLE_ERROR( cudaGraphicsMapResources( 1, &(bitmap->resource), NULL ) );
        HANDLE_ERROR( cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, bitmap->resource) );

        bitmap->fAnim( devPtr, bitmap->dataBlock, ticks++ );

        HANDLE_ERROR( cudaGraphicsUnmapResources( 1, &(bitmap->resource), NULL ) );

        glutPostRedisplay();
    }

    // static method used for glut callbacks
    static void Key(unsigned char key, int x, int y) {
        switch (key) {
            case 27:
                GPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
                if (bitmap->animExit) bitmap->animExit( bitmap->dataBlock );
                bitmap->free_resources();
                exit(0);
        }
    }

    // static method used for glut callbacks
    static void Draw( void ) {
        GPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
        glClearColor( 0.0, 0.0, 0.0, 1.0 );
        glClear( GL_COLOR_BUFFER_BIT );
        glDrawPixels( bitmap->width, bitmap->height, GL_RGBA,
                      GL_UNSIGNED_BYTE, 0 );
        glutSwapBuffers();
    }
};

#endif  // GRAPHICS
#endif  // __GPU_ANIM_H__

