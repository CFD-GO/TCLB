#include "GenericContainer.h"
std::string GenericContainer::xmlname = "Units";
#include "../HandlerFactory.h"

int GenericContainer::Init () {
		GenericAction::Init();
		return GenericAction::ExecuteInternal();
	}


int GenericContainer::Finish () {
		GenericAction::Unstack();
		return GenericAction::Finish();
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< GenericContainer > >;
