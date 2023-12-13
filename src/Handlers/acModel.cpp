#include "acModel.h"
std::string acModel::xmlname = "Model";
#include "../HandlerFactory.h"

int acModel::Init() {
    int ret = GenericContainer::Init();
    if (ret) return ret;
    solver->lattice->initLattice();
    solver->iter = 0;
    return 0;
}

// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register<GenericAsk<acModel> >;
