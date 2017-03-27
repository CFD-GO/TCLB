#include "acLoadMemoryDump.h"
std::string acLoadMemoryDump::xmlname = "LoadMemoryDump";
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


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acLoadMemoryDump > >;
