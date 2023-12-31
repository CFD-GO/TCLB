#include "cbTXT.h"
std::string cbTXT::xmlname = "TXT";
#include "../HandlerFactory.h"
#include "../vtkLattice.h"

int cbTXT::Init() {
    Callback::Init();
    auto attr = node.attribute("name");
    nm = attr ? attr.value() : "TXT";
    attr = node.attribute("what");
    s.add_from_string(attr ? attr.value() : "all", ',');
    gzip = node.attribute("gzip").as_bool();
    txt_type = gzip ? 1 : 0;
    return EXIT_SUCCESS;
}

int cbTXT::DoIt() {
    Callback::DoIt();
    const auto filename = solver->outIterFile(nm, "");
    return std::visit([&](const auto lattice_ptr) { return txtWriteLattice(filename, *lattice_ptr, solver->units, s, txt_type); }, solver->getLatticeVariant());
};

// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register<GenericAsk<cbTXT> >;
