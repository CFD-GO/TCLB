#include "cbBIN.h"
std::string cbBIN::xmlname = "BIN";
#include "../HandlerFactory.h"

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


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbBIN > >;
