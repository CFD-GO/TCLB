#include "cbTXT.h"
#include "../HandlerFactory.h"

int cbTXT::Init () {
		Callback::Init();
		pugi::xml_attribute attr = node.attribute("name");
		nm = "TXT";
		if (attr) nm = attr.value();
		attr = node.attribute("what");
		if (attr) {
		        s.add_from_string(attr.value(),',');
                } else {
                        s.add_from_string("all",',');
                }
		txt_type = 0;
		attr = node.attribute("gzip");
		if (attr) {
		        txt_type = 1;
                } else {
                }
		return 0;
	}


int cbTXT::DoIt () {
		Callback::DoIt();
		return solver->writeTXT(nm.c_str(), &s, txt_type);
	};


// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_cbTXT(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "TXT") {
    	return new cbTXT;
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_cbTXT >;

