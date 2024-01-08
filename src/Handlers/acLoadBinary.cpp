#include "acLoadBinary.h"
std::string acLoadBinary::xmlname = "LoadBinary";
#include "../HandlerFactory.h"

int acLoadBinary::Init() {
    Action::Init();
    auto attr = node.attribute("file");
    if (!attr) {
        attr = node.attribute("filename");
        if (!attr) {
            error("No file specified in LoadBinary\n");
            return -1;
        }
    }
    const auto attr2 = node.attribute("comp");
    if (attr2) {
        error("LoadBinary with selected component was not implemented");
        return -1;
    } else {
        solver->lattice->loadSolution(attr.value());
    }
    return 0;
}

// Register the handler (based on xmlname) in the Handler Factory
template class HandlerFactory::Register<GenericAsk<acLoadBinary> >;
