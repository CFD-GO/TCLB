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
                const auto fn = solver->outIterFile(filename, ".csv");
		solver->getCartLattice()->zSet.dumpToFile(fn.c_str());
		return 0;
	}


int cbDumpSettings::Finish () {
		return Callback::Finish();
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbDumpSettings > >;
