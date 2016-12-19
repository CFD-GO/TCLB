#include "cbSaveMemoryDump.h"
#include "../HandlerFactory.h"

int cbSaveMemoryDump::Init () {
		Callback::Init();
		pugi::xml_attribute attr = node.attribute("file");
		if (!attr) {
			attr = node.attribute("filename");
			if (!attr) {
				char filename[STRING_LEN];
				solver->outIterFile("Save", "", filename);
				fn = filename;
			} else {
                fn = attr.value();
            }
		} else {
            fn = ((std::string) solver->info.outpath) + "_" + attr.value();
        }
		return 0;
	}


int cbSaveMemoryDump::DoIt () {
		Callback::DoIt();
		pugi::xml_attribute attr= node.attribute("comp");
		if (attr) {
            error("Depreceted API call. Use SaveBinary with comp parameter");
        }
		solver->lattice->saveSolution(fn.c_str());
		return 0;
	};


// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_cbSaveMemoryDump(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "SaveMemoryDump") {
		return new cbSaveMemoryDump;
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_cbSaveMemoryDump >;

