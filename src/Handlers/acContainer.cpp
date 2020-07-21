#include "acContainer.h"
std::string acContainer::xmlname = "Container";
#include "../HandlerFactory.h"

int acContainer::Init () {
		GenericAction::Init();
       		if (GenericAction::ExecuteInternal()) return -1;
		GenericAction::Unstack();
		return 0;
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acContainer > >;
