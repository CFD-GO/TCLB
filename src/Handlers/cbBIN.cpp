#include "cbBIN.h"
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


// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_cbBIN(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "BIN") {
		return new cbBIN;
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_cbBIN >;

