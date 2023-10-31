#ifndef SAMPLER_H
#define SAMPLER_H

#include <mpi.h>

#include <cstdlib>
#include <vector>

#include "Consts.h"
#include "cross.h"
#include "Lists.h"
#include "Global.h"
#include "Region.h"
#include "types.h"
#include "unit.h"
#include "utils.h"

/* 
Class used for optimal storage and output of the evolution of a particular point/{set of points}.
Each point and corresponding output data is controlled by the mpi rank related to that point according to initial mesh division
*/

struct sreg {
	int rank;
	lbRegion location;
};
class Sampler {
       	typedef std::map< std::string , int > Location;
       	const Model* model;
       	public:
		Sampler(Model* model_, const UnitEnv* units_, int my_rank_) : model(model_), units(units_), mpi_rank(my_rank_) {}
		lbRegion position;
		real_t *gpu_buffer;
               	Location location;
               	name_set *quant;
               	int size = 0;
		const UnitEnv* units;
		std::vector <sreg> spoints; 
		int mpi_rank;
		int startIter = 0;
		int totalIter;
               	int initCSV(const char* name);
               	int writeHistory(int curr_iter);
               	int Allocate(name_set* quantities,int total_iter,int iter);
		int addPoint(lbRegion loc,int rank);
               	const char *filename;
		int Finish();
       	};
inline int csvWriteElement(FILE * f, float tmp) { return fprintf(f, ",%g" , tmp); }
inline int csvWriteElement(FILE * f, double tmp) { return fprintf(f, ",%lg" , tmp); }
inline int csvWriteElement(FILE * f, float3 tmp) { return fprintf(f, ",%g,%g,%g" , tmp.x, tmp.y, tmp.z); }
inline int csvWriteElement(FILE * f, double3 tmp) { return fprintf(f, ",%lg,%lg,%lg" , tmp.x, tmp.y, tmp.z); }
inline int csvWriteElement(FILE * f, vector_t tmp) {
       	csvWriteElement(f, tmp.x);
       	csvWriteElement(f, tmp.y);
       	return csvWriteElement(f, tmp.z);
}

#endif
