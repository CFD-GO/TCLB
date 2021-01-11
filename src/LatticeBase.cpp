/*  File defining Lattice                                      */
/*     Lattice is the low level class defining functionality   */
/*       of Adjoint LBM solver. It realizes all the LBM        */
/*       calculations and data transfer                        */
/*-------------------------------------------------------------*/

#include "Consts.h"
#include "cross.h"
#include "types.h"
#include "Global.h"
#include "LatticeBase.h"
#include <mpi.h>
#include <assert.h>
#include "BallTree.hpp"

#ifdef ENABLE_NVPROF
	#include <nvToolsExt.h>
	#define DEBUG_PROF_PUSH(x__) nvtxRangePushA(x__)
	#define DEBUG_PROF_POP() nvtxRangePop()
#else
	#define DEBUG_PROF_PUSH(x__)
	#define DEBUG_PROF_POP()
#endif


LatticeBase::~LatticeBase() {

}

/// Set monitor callback
/**
        Sets the monitor callback which will be called every second or frame
*/
void LatticeBase::Callback(int(*cb)(int,int, void*), void* data) {
	callback = cb;
	callback_data = data;
}

/// Calculate the Snapshot level for the optimal Checkpoiting technique
/**
        C-crazed function for calculating number of zeros at
        the end of a number written in a binary system
        /param i The number
        /return number of zeros at the end of the number written in a binary system
*/
int SnapLevel(unsigned int i) {
	unsigned int j = 16;
	unsigned int w = 0;
	unsigned int k = 0xFFFFu;
	while(j) {
		if (i & k) {
			j = j >> 1;
			k = k >> j;
		} else {
			w = w + j;
			j = j >> 1;
			k = k << j;
		}
	}
	return w;
}

// Function for calculating the index of a Snapshot for an iteration
int LatticeBase::getSnap(int i) {
	int s = SnapLevel(i) + 1;
	return s;
}