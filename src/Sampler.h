#include "mpi.h"
#include "utils.h"
#include "Consts.h"
#include "Global.h"
#include "cross.h"
#include "types.h"
#include <math.h>
#include <stdlib.h>

class Sampler {
       	typedef std::map< std::string , int > Location;
       	private:        
       //        	double * gpu_buffer; 
       	public:
		Sampler();
		double *gpu_buffer;
               	Location location;
               	name_set *quant;
               	int size;
               	int initCSV(const char* name);
               	int writeHistory(int curr_iter);
               	int Allocate(name_set* quantities,int total_iter);
               	const char *filename;

       	};
inline int csvWriteElement(FILE * f, float tmp) { return fprintf(f, "%g\n" , tmp); }
inline int csvWriteElement(FILE * f, double tmp) { return fprintf(f, "%lg\n" , tmp); }
inline int csvWriteElement(FILE * f, float3 tmp) { return fprintf(f, "%g\t,%g\t,%g\n" , tmp.x, tmp.y, tmp.z); }
inline int csvWriteElement(FILE * f, double3 tmp) { return fprintf(f, "%lg\t%lg\t%lg\n" , tmp.x, tmp.y, tmp.z); }
inline int csvWriteElement(FILE * f, vector_t tmp) {
       	csvWriteElement(f, tmp.x);
       	fprintf(f,"\t");
       	csvWriteElement(f, tmp.y);
       	fprintf(f,"\t");
       	return csvWriteElement(f, tmp.z);
       	fprintf(f,"\n");
}

