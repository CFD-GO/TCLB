#include "acInit.h"
std::string acInit::xmlname = "Init";
#include "../HandlerFactory.h"

int acInit::Init () {
		Action::Init();
		solver->lattice->initLattice();
		solver->iter = 0;
		return 0;
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acInit > >;
