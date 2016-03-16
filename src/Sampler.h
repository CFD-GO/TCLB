#include "mpi.h"
#include "utils.h"
#include "Consts.h"
#include "Global.h"
#include "cross.h"
#include "types.h"
#include <math.h>
#include <stdlib.h>
#include <vector>
/* 
Class used for optimal storing and output of the evolution of the particular point/{set of points}.
Each point and output data is controlled, by the mpi rank related to that point, according to initial mesh division
*/

struct sreg {
	int rank;
	lbRegion location;
};
class Sampler {
       	typedef std::map< std::string , int > Location;
       	private:        
       //        	double * gpu_buffer; 
       	public:
		Sampler();
		lbRegion position;
		real_t *gpu_buffer;
               	Location location;
               	name_set *quant;
               	int size;
		std::vector <sreg> spoints; 
		MPIInfo mpis; 
		int startIter;
		int totalIter;
               	int initCSV(const char* name);
               	int writeHistory(int curr_iter);
               	int Allocate(name_set* quantities,int total_iter,int noidea);
		int addPoint(lbRegion loc,int rank);
               	const char *filename;
		int Finish();
       	};
inline int csvWriteElement(FILE * f, float tmp) { return fprintf(f, "%g\t" , tmp); }
inline int csvWriteElement(FILE * f, double tmp) { return fprintf(f, "%lg\t" , tmp); }
inline int csvWriteElement(FILE * f, float3 tmp) { return fprintf(f, "%g\t,%g\t,%g\t" , tmp.x, tmp.y, tmp.z); }
inline int csvWriteElement(FILE * f, double3 tmp) { return fprintf(f, "%lg\t%lg\t%lg\t" , tmp.x, tmp.y, tmp.z); }
inline int csvWriteElement(FILE * f, vector_t tmp) {
       	csvWriteElement(f, tmp.x);
       	fprintf(f,"\t");
       	csvWriteElement(f, tmp.y);
       	fprintf(f,"\t");
       	return csvWriteElement(f, tmp.z);
       	fprintf(f,"\t");
}

