#include "cbBIN.h"

int cbBIN::Init () {
		Callback::Init();
		pugi::xml_attribute attr = node.attribute("name");
		nm = "BIN";
		if (attr) nm = attr.value();
		return 0;
	}


int cbBIN::DoIt () {
		Callback::DoIt();
		return solver->writeBIN(nm.c_str());
	};

