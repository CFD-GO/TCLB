#include "cbDumpSettings.h"
#include "../HandlerFactory.h"

int cbDumpSettings::Init () {
		Callback::Init();
		pugi::xml_attribute attr = node.attribute("name");
		filename = "ZonalSettings";
		if (attr) filename = attr.value();
		return 0;
	}


int cbDumpSettings::DoIt () {
		Callback::DoIt();
		char fn[STRING_LEN];
		solver->outIterFile(filename.c_str(), ".csv", fn);
		solver->lattice->zSet.dumpToFile(fn);
		return 0;
	}


int cbDumpSettings::Finish () {
		return Callback::Finish();
	}


// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_cbDumpSettings(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "DumpSettings") {
		return new cbDumpSettings;
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_cbDumpSettings >;

