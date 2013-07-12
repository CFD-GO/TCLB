CudaSharedMemory int pushcontrol[512];

inline CudaDeviceFunction void pushcontrol1b_(int i) {
	pushcontrol[CudaThread.x] <<= 1;
	pushcontrol[CudaThread.x] += i;
}

inline CudaDeviceFunction void pushcontrol2b_(int i) {
	pushcontrol[CudaThread.x] <<= 2;
	pushcontrol[CudaThread.x] += i;
}

inline CudaDeviceFunction void pushcontrol3b_(int i) {
	pushcontrol[CudaThread.x] <<= 3;
	pushcontrol[CudaThread.x] += i;
}

inline CudaDeviceFunction void popcontrol1b_(int * i) {
	*i = pushcontrol[CudaThread.x] & 0x01;
	pushcontrol[CudaThread.x] >>= 1;
}

inline CudaDeviceFunction void popcontrol2b_(int * i) {
	*i = pushcontrol[CudaThread.x] & 0x03;
	pushcontrol[CudaThread.x] >>= 2;
}

inline CudaDeviceFunction void popcontrol3b_(int * i) {
	*i = pushcontrol[CudaThread.x] & 0x07;
	pushcontrol[CudaThread.x] >>= 3;
}
	
CudaSharedMemory real_t pushval[512*6];
CudaSharedMemory int pushvali[512];

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

