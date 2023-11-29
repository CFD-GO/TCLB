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
        solver->lattice->loadComp(attr.value(), attr2.value());
    } else {
        solver->lattice->loadSolution(attr.value());
        error("Missing comp parameter in LoadBinary");
    }
    return 0;
}

// Register the handler (based on xmlname) in the Handler Factory
template class HandlerFactory::Register<GenericAsk<acLoadBinary> >;
