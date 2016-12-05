#include "conControlParameter.h"

int conControlParameter::Init () {
		Action::Init();
		pugi::xml_attribute attr = node.attribute("field");
		if (!attr) {
			ERROR("No \"field\" attribute in GeometryParameter\n");
			return -1;
		}
		std::string str = attr.value();
		

		return 0;
	}

