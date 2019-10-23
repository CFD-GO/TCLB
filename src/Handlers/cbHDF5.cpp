#include "cbHDF5.h"
std::string cbHDF5::xmlname = "HDF5";
#include "../HandlerFactory.h"
#include "../hdf5Lattice.h"

int cbHDF5::Init () {
		Callback::Init();
		pugi::xml_attribute attr = node.attribute("name");
		nm = "HDF5";
		if (attr) nm = attr.value();
		attr = node.attribute("what");
		if (attr) {
		        s.add_from_string(attr.value(),',');
                } else {
                        s.add_from_string("all",',');
                }
		return 0;
	}


int cbHDF5::DoIt () {
		Callback::DoIt();
		return hdf5WriteLattice(nm.c_str(), solver->lattice, solver->units, &s);
};


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbHDF5 > >;
