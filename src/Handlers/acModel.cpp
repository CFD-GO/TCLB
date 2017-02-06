#include "acModel.h"
std::string acModel::xmlname = "Model";
#include "../HandlerFactory.h"

int acModel::Init () {
		GenericContainer::Init();
		solver->lattice->Init();
		solver->iter = 0;
		return 0;
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acModel > >;
