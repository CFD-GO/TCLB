#include "acLoadBinary.h"
#include "../HandlerFactory.h"

int acLoadBinary::Init () {
		Action::Init();
		pugi::xml_attribute attr = node.attribute("file");
		if (!attr) {
			attr = node.attribute("filename");
			if (!attr) {
				error("No file specified in LoadBinary\n");
				return -1;
			}
		}
		pugi::xml_attribute attr2= node.attribute("comp");
		if (attr2) {
			solver->loadComp(attr.value(), attr2.value());
		} else {
		solver->lattice->loadSolution(attr.value());
            error("Missing comp parameter in LoadBinary");
		}
		return 0;
	}


// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_acLoadBinary(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "LoadBinary") {
		return new acLoadBinary;
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_acLoadBinary >;

