/**
    \file Global.h
    Declaration of global variables and functions. Including MPI connectivity info.
*/

#ifndef GLOBAL_H
#define GLOBAL_H

#include "Consts.h"
#include "types.h"
#include "cross.h"

#include <string>

#ifdef _WIN32
  #define isatty(...) 1
#else
  #include <unistd.h>
#endif

    #define BOUNDARY_UX UX_mid
    
    #include "MPMD.hpp"
    extern MPMDHelper MPMD;
/*
#ifndef SETTINGS_H

void initSettings();

#define SETTINGS_H 1
#endif
*/
#ifndef DEBUG_H   
  #ifdef USE_STEADY_CLOCK
	#include <chrono>
	typedef std::chrono::time_point<std::chrono::steady_clock> time_point_t;
  #else
	#include <ctime>
	typedef std::clock_t time_point_t;
  #endif

    extern time_point_t global_start;

    inline double get_walltime() {
	#ifdef USE_STEADY_CLOCK
		time_point_t now = std::chrono::steady_clock::now();
		std::chrono::duration<double, std::milli> ms = now - global_start;
		return ms.count() / 1000.0;
	#else
		time_point_t now = std::clock();
		return (now - global_start) / (double)CLOCKS_PER_SEC;
	#endif
    }

    void start_walltime();

    extern int D_MPI_RANK;
    extern int D_TERMINAL;

    int myprint(int level, int all, const char *fmt, ...);
    int InitPrint(int print_level, int only_root_level, int error_level);
    #define dprint(x, ...) myprint(x, 0, __VA_ARGS__)
    #define debug0(...) DEBUG0(myprint(0, 0, __VA_ARGS__)) // 0-level Debug
    #define debug1(...) DEBUG1(myprint(1, 0, __VA_ARGS__)) // 1-level Debug
    #define debug2(...) DEBUG2(myprint(2, 0, __VA_ARGS__)) // 2-level Debug
    #define debug3(...) DEBUG3(myprint(3, 0, __VA_ARGS__)) // Output
    #define debug4(...) myprint(4, 0, __VA_ARGS__) // Notice
    #define debug5(...) myprint(5, 0, __VA_ARGS__) // Important Notice
    #define debug6(...) myprint(6, 0, __VA_ARGS__) // Warning
    #define debug7(...) myprint(7, 0, __VA_ARGS__) // Important Warning
    #define debug8(...) myprint(8, 0, __VA_ARGS__) // Error
    #define debug9(...) myprint(9, 0, __VA_ARGS__) // Fatal Error
    
    #define output_all(...) DEBUG3(myprint(3, -1, __VA_ARGS__))
    #define output(...) debug3(__VA_ARGS__)    
    #define notice(...) debug4(__VA_ARGS__)    
    #define NOTICE(...) debug5(__VA_ARGS__)    
    #define warning(...) debug6(__VA_ARGS__)    
    #define WARNING(...) debug7(__VA_ARGS__)    
    #define error(...) debug8(__VA_ARGS__)    
    #define ERROR(...) debug9(__VA_ARGS__)    
    

    #define DEBUG_SETRANK(x) {D_MPI_RANK = x; D_TERMINAL = isatty(1); }
#if DEBUG_LEVEL < 1
    #define DEBUG0(x) x; fflush(stdout);
    #define DEBUG1(x) x; fflush(stdout);
    #define DEBUG2(x) x; fflush(stdout);
    #define DEBUG3(x) x; fflush(stdout);
#elif DEBUG_LEVEL < 2
    #define DEBUG0(x)
    #define DEBUG1(x) x
    #define DEBUG2(x) x
    #define DEBUG3(x) x
#elif DEBUG_LEVEL < 3
    #define DEBUG0(x)
    #define DEBUG1(x)
    #define DEBUG2(x) x
    #define DEBUG3(x) x
#elif DEBUG_LEVEL < 4
    #define DEBUG0(x)
    #define DEBUG1(x)
    #define DEBUG2(x)
    #define DEBUG3(x) x
#else
    #define DEBUG0(x)
    #define DEBUG1(x)
    #define DEBUG2(x)
    #define DEBUG3(x)
#endif
    #define DEBUG_M debug0("%s(%d)\n", __FILE__, __LINE__ )

int kbhit(void);
	
#define DEBUG_H 1
#endif

#ifndef D2Q9_w_and_e

CudaConstantMemory const  real_t 	wt[5] = {2./6., 	1./6., 1./6., 1./6., 1./6.};
CudaConstantMemory const  real_t 	wf[9] = {4./9., 	1./9., 1./9., 1./9., 1./9., 	1./36., 1./36., 1./36., 1./36.};

	
	CudaConstantMemory real_t const  d2q9_ex[9] = {0,1,0,-1,0,1,-1,-1,1};
	CudaConstantMemory real_t const  d2q9_ey[9] = {0,0,1,0,-1,1,1,-1,-1};
	CudaConstantMemory real_t const  d2q5_ex[5] = {0,1,0,-1,0};
	CudaConstantMemory real_t const  d2q5_ey[5] = {0,0,1,0,-1};


#endif
#define D2Q9_w_and_e

#ifndef D3Q27_w_and_e
CudaConstantMemory real_t const wtConv[27] = {8./27.,
                                      2./27., 2./27., 2./27., 2./27., 2./27., 2./27.,
                                      1./54.,  1./54.,  1./54.,  1./54.,
                                      1./54.,  1./54.,  1./54.,  1./54.,
                                      1./54.,  1./54.,  1./54.,  1./54.,
  				      1./216., 1./216., 1./216., 1./216.,
                                      1./216., 1./216., 1./216., 1./216.};


CudaConstantMemory real_t const wg[27] = {8./27.,
                                      2./27., 2./27., 2./27., 2./27., 2./27., 2./27.,
                                      1./216., 1./216., 1./216., 1./216.,
                                      1./216., 1./216., 1./216., 1./216.,
                                      1./54.,  1./54.,  1./54.,  1./54.,
                                      1./54.,  1./54.,  1./54.,  1./54.,
                                      1./54.,  1./54.,  1./54.,  1./54. };

CudaConstantMemory real_t const wh[15] = {2./9.,
                                          1./9., 1./9., 1./9., 1./9., 1./9., 1./9.,
                                          1./72.,1./72.,1./72.,1./72.,1./72.,1./72.,1./72.,1./72.};

CudaConstantMemory real_t const d3q27_ex[27] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
CudaConstantMemory real_t const d3q27_ey[27] = {0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1};
CudaConstantMemory real_t const d3q27_ez[27] = {0, 0, 0, 0, 0, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1};

#endif
#define D3Q27_w_and_e

#endif // GLOBAL_H
