#include "cbDumpSettings.h"

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

