#include "conFieldParameter.h"
std::string conFieldParameter::xmlname = "FieldParameter";
#include "../HandlerFactory.h"

int conFieldParameter::Init () {
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
template class HandlerFactory::Register< GenericAsk< conFieldParameter > >;
