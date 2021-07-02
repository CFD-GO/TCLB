#include "Consts.h"

CudaSharedMemory int pushcontrol[MAX_THREADS];

#define pushcontrol1b(x__) pushcontrol1b_(x__)
#define pushControl1b(x__) pushcontrol1b_(x__)
#define pushcontrol2b(x__) pushcontrol2b_(x__)
#define pushControl2b(x__) pushcontrol2b_(x__)
#define pushcontrol3b(x__) pushcontrol3b_(x__)
#define pushControl3b(x__) pushcontrol3b_(x__)
#define popcontrol1b(x__)  popcontrol1b_(x__)
#define popControl1b(x__)  popcontrol1b_(x__)
#define popcontrol2b(x__)  popcontrol2b_(x__)
#define popControl2b(x__)  popcontrol2b_(x__)
#define popcontrol3b(x__)  popcontrol3b_(x__)
#define popControl3b(x__)  popcontrol3b_(x__)


inline CudaDeviceFunction void pushcontrol1b_(int i) {
	const int cths=CudaThread.x + CudaThread.y*CudaNumberOfThreads.x;
	pushcontrol[cths] <<= 1;
	pushcontrol[cths] += i;
}

inline CudaDeviceFunction void pushcontrol2b_(int i) {
	const int cths=CudaThread.x + CudaThread.y*CudaNumberOfThreads.x;
	pushcontrol[cths] <<= 2;
	pushcontrol[cths] += i;
}

inline CudaDeviceFunction void pushcontrol3b_(int i) {
	const int cths=CudaThread.x + CudaThread.y*CudaNumberOfThreads.x;
	pushcontrol[cths] <<= 3;
	pushcontrol[cths] += i;
}

inline CudaDeviceFunction void popcontrol1b_(int * i) {
	const int cths=CudaThread.x + CudaThread.y*CudaNumberOfThreads.x;
	*i = pushcontrol[cths] & 0x01;
	pushcontrol[cths] >>= 1;
}

inline CudaDeviceFunction void popcontrol2b_(int * i) {
	const int cths=CudaThread.x + CudaThread.y*CudaNumberOfThreads.x;
	*i = pushcontrol[cths] & 0x03;
	pushcontrol[cths] >>= 2;
}

inline CudaDeviceFunction void popcontrol3b_(int * i) {
	const int cths=CudaThread.x + CudaThread.y*CudaNumberOfThreads.x;
	*i = pushcontrol[cths] & 0x07;
	pushcontrol[cths] >>= 3;
}

/*	
CudaSharedMemory real_t pushval[MAX_THREADS*6];
CudaSharedMemory int pushvali[MAX_THREADS];

inline CudaDeviceFunction void ADclear () {
	pushvali[CudaThread.x] = 0;
}

inline CudaDeviceFunction void pushreal (real_t f) {
	pushval[CudaThread.x + pushvali[CudaThread.x]*CudaNumberOfThreads.x] = f;
	pushvali[CudaThread.x] ++;
}

inline CudaDeviceFunction void pushreal4_ (float f) { pushreal(f); }
inline CudaDeviceFunction void pushreal8_ (double f) { pushreal(f); }

inline CudaDeviceFunction void popreal (real_t * f) {
	pushvali[CudaThread.x] --;
	*f = pushval[CudaThread.x + pushvali[CudaThread.x]*CudaNumberOfThreads.x];
}

inline CudaDeviceFunction void popreal4_ (float  * f) { popreal((real_t*) f); }
inline CudaDeviceFunction void popreal8_ (double * f) { popreal((real_t*) f); }
*/
