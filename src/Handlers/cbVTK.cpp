#include "cbVTK.h"
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


// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_cbVTK(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "VTK") {
		return new cbVTK;
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_cbVTK >;

