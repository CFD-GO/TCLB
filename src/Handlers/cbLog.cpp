#include "cbLog.h"

int cbLog::Init () {
		char fn[STRING_LEN];
		Callback::Init();
		pugi::xml_attribute attr = node.attribute("name");
		std::string nm = "Log";
		if (attr) nm = attr.value();
		solver->outIterFile(nm.c_str(), ".csv", fn);
		filename = fn;
		solver->initLog(filename.c_str());
		old_iter_type = solver->iter_type;
		solver->iter_type |= ITER_LASTGLOB;
		return 0;
	}


int cbLog::DoIt () {
		Callback::DoIt();
		solver->writeLog(filename.c_str());
		return 0;
	}


int cbLog::Finish () {
		solver->iter_type = old_iter_type;
		return Callback::Finish();
	}

