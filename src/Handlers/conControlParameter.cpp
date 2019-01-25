#include "conControlParameter.h"
std::string conControlParameter::xmlname = "ControlParameter";
#include "../HandlerFactory.h"

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


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< conControlParameter > >;
