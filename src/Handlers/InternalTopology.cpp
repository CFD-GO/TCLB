#include "InternalTopology.h"
#include "../HandlerFactory.h"

int InternalTopology::Init () {
		Pars = -1;
		return Design::Init();
	};


int InternalTopology::NumberOfParameters () {
		if (Pars < 0) {
			Pars =  solver->getPars();
		}
		return Pars;
	};


int InternalTopology::Parameters (int type, double * tab) {
		switch(type){
		case PAR_GET:
			return solver->getPar(tab);
		case PAR_SET:
			return solver->setPar(tab);
		case PAR_GRAD:
			return solver->getDPar(tab);
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


// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_InternalTopology(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "InternalTopology") {
		return new InternalTopology;
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_InternalTopology >;

