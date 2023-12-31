#include "cbSaveBinary.h"
std::string cbSaveBinary::xmlname = "SaveBinary";
#include "../HandlerFactory.h"

int cbSaveBinary::Init() {
    Callback::Init();
    auto attr = node.attribute("file");
    if (!attr) {
        attr = node.attribute("filename");
        if (!attr) {
            fn = solver->outIterFile("Save", "");
        } else {
            fn = attr.value();
        }
    } else {
        fn = solver->outpath + "_" + attr.value();
    }
    return 0;
}

int cbSaveBinary::DoIt() {
    Callback::DoIt();
    const auto attr = node.attribute("comp");
    if (attr) solver->lattice->saveComp(fn, attr.value());
    else
        solver->lattice->saveSolution(fn);
    return EXIT_SUCCESS;
};

// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register<GenericAsk<cbSaveBinary> >;
