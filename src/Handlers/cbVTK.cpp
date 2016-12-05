#include "cbVTK.h"

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

