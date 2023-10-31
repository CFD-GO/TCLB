#include "InternalTopology.h"
std::string InternalTopology::xmlname = "InternalTopology";
#include "../HandlerFactory.h"

int InternalTopology::Init () {
		Pars = -1;
                par_struct.Par_disp = std::make_unique<int[]>(solver->mpi_size);
                par_struct.Par_sizes = std::make_unique<int[]>(solver->mpi_size);
		return Design::Init();
	};


int InternalTopology::NumberOfParameters () {
		if (Pars < 0) {
                    const auto lattice = solver->getCartLattice();
		    Pars =  lattice->getPars(par_struct);
		}
		return Pars;
	};


int InternalTopology::Parameters (int type, double * tab) {
                const auto lattice = solver->getCartLattice();
		switch(type){
		case PAR_GET:
			return lattice->getPar(par_struct, tab);
		case PAR_SET:
			return lattice->setPar(par_struct, tab);
		case PAR_GRAD:
			return lattice->getDPar(par_struct, tab);
		case PAR_UPPER:
			for (int i=0;i<Pars;i++) tab[i]=1;
			return 0;
		case PAR_LOWER:
			for (int i=0;i<Pars;i++) tab[i]=0;
			return 0;
		default:
			ERROR("Unknown type %d in call to Parameters in %s\n",type,node.name());
			exit(-1);
		}
		return -1;	
	};


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< InternalTopology > >;
