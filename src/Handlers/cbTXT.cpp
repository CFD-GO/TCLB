#include "cbTXT.h"
std::string cbTXT::xmlname = "TXT";
#include "../HandlerFactory.h"
#include "../vtkLattice.h"

int cbTXT::Init () {
		Callback::Init();
		pugi::xml_attribute attr = node.attribute("name");
		nm = "TXT";
		if (attr) nm = attr.value();
		attr = node.attribute("what");
		if (attr) {
		        s.add_from_string(attr.value(),',');
                } else {
                        s.add_from_string("all",',');
                }
		gzip = false;
		attr = node.attribute("gzip");
		if (attr) gzip = attr.as_bool();
		txt_type = 0;
		if (gzip) txt_type = 1;
		return 0;
	}

int cbTXT::DoIt() {
    Callback::DoIt();
    const auto filename = solver->outIterFile(nm, "");
    return std::visit([&](const auto lattice_ptr) { return txtWriteLattice(filename, *lattice_ptr, solver->units, s, txt_type); }, solver->getLatticeVariant());
};

// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbTXT > >;
