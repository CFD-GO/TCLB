#include "cbHDF5.h"
std::string cbHDF5::xmlname = "HDF5";
#include "../HandlerFactory.h"
#include "../hdf5Lattice.h"

int cbHDF5::Init () {
	Callback::Init();
#ifdef WITH_HDF5
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
#else
		ERROR("No hdf5 support at configure\n");
		return -1;
#endif
	}


int cbHDF5::DoIt () {
#ifdef WITH_HDF5
		Callback::DoIt();
		return hdf5WriteLattice(nm.c_str(), solver->lattice, solver->units, &s);
#else
		return -1;
#endif
};


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbHDF5 > >;
