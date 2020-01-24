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

