#include "acRepeat.h"
std::string acRepeat::xmlname = "Repeat";
#include "../HandlerFactory.h"

int acRepeat::Init () {
		GenericAction::Init();
		pugi::xml_attribute attr = node.attribute("Times");
		if (attr) {
			times = attr.as_int();
		} else {
			error("no Times parameter in %s\n");
			return -1;
		}
		for (int i=0; i < times; i++) {
        		if (GenericAction::ExecuteInternal()) return -1;
                }
		return 0;
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acRepeat > >;
