#include "GenericContainer.h"

int GenericContainer::Init () {
		GenericAction::Init();
		return GenericAction::ExecuteInternal();
	}


int GenericContainer::Finish () {
		GenericAction::Unstack();
		return GenericAction::Finish();
	}

