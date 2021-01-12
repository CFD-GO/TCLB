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

/// Starting of unsteady adjoint recording
/** 
	Starts tape recording of the iteration process including:
	all of the iterations done
	changes of settings
*/
void LatticeBase::startRecord() {
	if(reverse_save) {
		ERROR("Nested record! Called startRecord while recording\n");
		exit(-1);
	}
	if(Record_Iter != 0) {
		ERROR("Record tape is not rewound (iter = %d) maybe last adjoint didn't go all the way\n", Record_Iter);
		Record_Iter = 0;
	}
	debug2("Lattice is starting to record (unsteady adjoint)\n");
	if(Snap != 0) {
		warning("Snap = %d at startRecord\n", Snap);
	}
	for(int i = 0; i < maxSnaps; i++) {
		iSnaps[i] = -1;
	}
	// not sure if there should be an iterator or if statement here
	{
		char filename[4*STRING_LEN];
		sprintf(filename, "%s_%02d_%02d.dat", snapFileName, D_MPI_RANK, getSnap(0));
		iSnaps[getSnap(0)] = 0;
		save(Snaps[Snap], filename);
	}
	if(Snap != 0) {
		warning("Snap = %d. Go through disk\n", Snap);
	} else {
		iSnaps[Snap] = 0;
	}
	reverse_save = 1;
	clearAdjoint();
	clearGlobals();
	settings_record.clear();
	settings_i = 0;
}

/**
	Stops the Adjoint recording process
*/
void LatticeBase::rewindRecord() {
	Record_Iter = 0;
	IterateTill(Record_Iter, ITER_NORM);
	debug2("Rewind tape\n");
}

/**
	Stops the adjoint recording process
*/
void LatticeBase::stopRecord() {
	if(Record_Iter != 0) {
		WARNING("Record tape is not rewound (iter = %d)\n", Record_Iter);
		Record_Iter = 0;
	}
	reverse_save = 0;
	debug2("Stop recording\n");
}

/// Save a FTabs or AFTabs
int LatticeBase::save(FTabsBase& tab, const char * filename) {
	FILE * f = fopen(filename, "w");
	if (f == NULL) {
		ERROR("Cannot open %s for output\n", filename);
		assert(f == NULL);
		return -1;
	}

	void ** ptr;
	void * pt=NULL;
	size_t * size;
	size_t maxsize;
	int n;

	listTabs(tab, &n, &size, &ptr, &maxsize);
	CudaMallocHost(&pt,maxsize);

	for(int i=0; i<n; i++)
	{
        output("Saving data slice %d, size %d", i, size[i]);
		CudaMemcpy( pt, ptr[i], size[i], cudaMemcpyDeviceToHost);
		fwrite(pt, size[i], 1, f);
	}

	CudaFreeHost(pt);
	fclose(f);
	delete[] size;
	delete[] ptr;
	return 0;
}

/// Load a FTabs or AFTabs
int LatticeBase::load(FTabsBase& tab, const char * filename) {
	FILE * f = fopen(filename, "r");
	output("Loading Lattice data from %s\n", filename);
	if (f == NULL) {
		ERROR("Cannot open %s for output\n", filename);
		return -1;
	}

	void ** ptr;
	void * pt = NULL;
	size_t * size;
	size_t maxsize;
	int n;

	listTabs(tab, &n, &size, &ptr, &maxsize);
	CudaMallocHost(&pt,maxsize);

	for(int i=0; i<n; i++)
	{
		int ret = fread(pt, size[i], 1, f);
		if (ret != 1) ERROR("Could not read in ArbitraryLattice::load");
		CudaMemcpy( ptr[i], pt, size[i], cudaMemcpyHostToDevice);
	}

	CudaFreeHost(pt);
	fclose(f);
	delete[] size;
	delete[] ptr;
	return 0;
}