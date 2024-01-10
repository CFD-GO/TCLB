#include <mpi.h>
#include "Consts.h"
#include "Global.h"
#include "cross.h"
#include "types.h"
#include <stdlib.h>
#include "Sampler.h"

int Sampler::initCSV(const char *name) 
     {
     filename = name;
     FILE * f = NULL;
     f = fopen(name, "wt");
     output("Initializing %s\n",filename);
     assert( f != NULL );
     fprintf(f,"Iteration,X,Y,Z");
	for (const Model::Quantity& it : model->quantities) {
		if (quant->in(it.name)) {
			const char* n = it.name.c_str();
			if (it.isVector) {
				fprintf(f,",%s.x,%s.y,%s.z", n, n, n);
			} else {
				fprintf(f,",%s", n);
			}
		}
	}
     fprintf(f,"\n");
     fclose(f); 
     return 0;
}
int Sampler::writeHistory(int curr_iter) {
     FILE* f = fopen(filename,"at");
     for (int i = startIter; i< curr_iter; i++){
	     for (size_t j = 0; j <  spoints.size(); j++) {
		if (mpi_rank == spoints[j].rank) {
			vector_t tmp_loc;
			tmp_loc.x = spoints[j].location.dx;
			tmp_loc.y = spoints[j].location.dy;
			tmp_loc.z = spoints[j].location.dz;
			fprintf(f,"%d",i);
			csvWriteElement(f,tmp_loc);
	for (const auto& quantity : model->quantities) {
			if (quant->in(quantity.name)) {
				real_t tmp;
				int comp = 1;
				if (quantity.isVector) comp = 3;
				CudaMemcpy(&tmp,&gpu_buffer[(location[quantity.name] + (i - startIter)*size + totalIter*j*size)],sizeof(real_t)*comp,CudaMemcpyDeviceToHost);
				csvWriteElement(f,tmp);
			}
		}
			fprintf(f,"\n");
			}
			}
	       } 
      fclose(f);
     return 0;
 }

int Sampler::Allocate(name_set* nquantities,int start,int iter) {
	totalIter = iter;
	int i = 0;
	startIter=start;
	quant = nquantities;
	for (const auto& quantity : model->quantities) {
		if (quant->in(quantity.name)) {
			location[quantity.name] = i;
			if (quantity.isVector) i = i + 3; else i = i + 1;
		}
	}
	CudaMalloc((void**)&gpu_buffer, i*totalIter*spoints.size()*sizeof(real_t)); 
	size = i;
	return 0;
}

int Sampler::addPoint(lbRegion loc,int rank){ 
	sreg temp;
	temp.location = loc;
	temp.rank = rank;
	spoints.push_back(temp);
	return 0;
}

int Sampler::Finish()
{
 CudaFree(gpu_buffer);
 size = 0;
 startIter = 0;
 spoints.clear();
 position = {};
 return 0;
}
