#include <mpi.h>
#include "Consts.h"
#include "Global.h"
#include "cross.h"
#include "types.h"
#include <stdlib.h>
#include "Sampler.h"
#include "Lattice.h"

Sampler::Sampler(Lattice *lattice_) : lattice(lattice_) { 
	size = 0;
	startIter = 0;
	position = lbRegion();
}

int Sampler::initCSV(const char *name) 
     {
     filename = name;
     FILE * f = NULL;
     f = fopen(name, "wt");
     output("Initializing %s\n",filename);
     assert( f != NULL );
     fprintf(f,"Iteration,X,Y,Z");
	for (const Model::Quantity& it : lattice->model->quantities) {
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
		if (mpis.rank == spoints[j].rank) { 
			vector_t tmp_loc;
			tmp_loc.x = spoints[j].location.dx;
			tmp_loc.y = spoints[j].location.dy;
			tmp_loc.z = spoints[j].location.dz;
			fprintf(f,"%d",i);
			csvWriteElement(f,tmp_loc);
	for (Model::Quantity& it : lattice->model->quantities) {
			if (quant->in(it.name)) {
				real_t tmp;
				int comp = 1;
				if (it.isVector) comp = 3;
				CudaMemcpy(&tmp,&gpu_buffer[(location[it.name] + (i - startIter)*size + totalIter*j*size)],sizeof(real_t)*comp,cudaMemcpyDeviceToHost); 
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
	for (Model::Quantity& it : lattice->model->quantities) {
		if (quant->in(it.name)) {
			location[it.name] = i;	
			if (it.isVector) i = i + 3; else i = i + 1;
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
 lbRegion pos;
 position = pos;
 return 0;
}
