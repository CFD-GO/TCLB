#include "cbTXT.h"
std::string cbTXT::xmlname = "TXT";
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


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbTXT > >;
