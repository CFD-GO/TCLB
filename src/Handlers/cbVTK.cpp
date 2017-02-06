#include "cbVTK.h"
std::string cbVTK::xmlname = "VTK";
#include "../HandlerFactory.h"

int cbVTK::Init () {
		Callback::Init();
		pugi::xml_attribute attr = node.attribute("name");
		nm = "VTK";
		if (attr) nm = attr.value();
		attr = node.attribute("what");
		if (attr) {
		        s.add_from_string(attr.value(),',');
                } else {
                        s.add_from_string("all",',');
                }
		return 0;
	}


int cbVTK::DoIt () {
		Callback::DoIt();
		return solver->writeVTK(nm.c_str(), &s);
	};


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbVTK > >;
