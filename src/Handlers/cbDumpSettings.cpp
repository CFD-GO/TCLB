#include "cbDumpSettings.h"
std::string cbDumpSettings::xmlname = "DumpSettings";
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


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbDumpSettings > >;
