#include "acLoadMemoryDump.h"
#include "../HandlerFactory.h"

int acLoadMemoryDump::Init () {
		Action::Init();
		pugi::xml_attribute attr = node.attribute("file");
		if (!attr) {
			attr = node.attribute("filename");
			if (!attr) {
				error("No file specified in LoadMemoryDump\n");
				return -1;
			}
		}
		pugi::xml_attribute attr2= node.attribute("comp");
		if (attr2) {
            error("Depreceted API call. Use LoadBinary with comp parameter");
        }
		solver->lattice->loadSolution(attr.value());
		return 0;
	}


// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_acLoadMemoryDump(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "LoadMemoryDump") {
		return new acLoadMemoryDump;
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_acLoadMemoryDump >;

