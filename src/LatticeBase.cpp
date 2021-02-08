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
		save(Snap, false, filename);
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
	Dump the primal and adjoint solutions to binary files
	\param filename Prefix/path for the dumped binary files
*/
void LatticeBase::saveSolution(const char *filename) {
	char fn[STRING_LEN];
	sprintf(fn, "%s_%d.pri", filename, D_MPI_RANK);
	save(Snap, false, fn); // write primal
#ifdef ADJOINT
	sprintf(fn, "%s_%d.adj", filename, D_MPI_RANK);
	save(aSnap, true, fn);
#endif
}

// Loads soln for binary files
/** 
	Loads the primal and adjoint solutions from binary files
	\param filename Prefix/path for the dumped binary files
*/
void LatticeBase::loadSolution(const char *filename) {
	char fn[STRING_LEN];
	sprintf(fn, "%s_%d.pri", filename, D_MPI_RANK);
	if(load(Snap, false, fn)) exit(-1);
#ifdef ADJOINT
	sprintf(fn, "%s_%d.adj", filename, D_MPI_RANK);
	load(aSnap, true, fn);
#endif
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
int LatticeBase::save(int snap, bool adjSnap, const char * filename) {
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

	listTabs(snap, adjSnap, &n, &size, &ptr, &maxsize);
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
int LatticeBase::load(int snap, bool adjSnap, const char * filename) {
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

	listTabs(snap, adjSnap, &n, &size, &ptr, &maxsize);
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

/**
	Push a setting and its value on the stack,
	during adjoint recording.
	\param i Index of the setting
	\param old Old value of the setting
	\param nw New value of the setting
	CBG
*/
void LatticeBase::push_setting(int i, real_t old, real_t nw) {
	settings_record.push_back(
		std::pair< int, std::pair<int, std::pair<real_t, real_t> > > (
			Record_Iter,
			std::pair<int, std::pair<real_t, real_t> > (
				i,
				std::pair<real_t, real_t> (
					old,
					nw
				)
			)
		)
	);
	settings_i = settings_record.size();
}

/**
	Reconstruct the values of all the settings
	in a specific iteration of the recorded solution
	from the setting stack.
	CBG
*/
void LatticeBase::pop_settings() {
	while(settings_i > 0) {
		if(settings_record[settings_i - 1].first <= Record_Iter) break;
		settings_i --;
		debug1("set %d to (back) %lf -> %lf\n", settings_record[settings_i].second.first, settings_record[settings_i].second.second.second, settings_record[settings_i].second.second.first);
		setSetting(settings_record[settings_i].second.first, settings_record[settings_i].second.second.first);
	}
	while(settings_i < settings_record.size()) {
		if(settings_record[settings_i].first > Record_Iter) break;
		debug1("set %d to (front) %lf -> %lf\n", settings_record[settings_i].second.first, settings_record[settings_i].second.second.first, settings_record[settings_i].second.second.second);
		setSetting(settings_record[settings_i].second.first, settings_record[settings_i].second.second.second);
		settings_i++;
	}
}

