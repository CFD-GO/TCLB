#include "vHandler.h"
#include "../CommonHandler.h"

vHandler::vHandler() {
	parSize= -1;
}

int vHandler::DoIt() {
	ERROR("Called virtual function (DoIt)!\n");
	exit(-1);
	return -1;
}

int vHandler::Init() {
	ERROR("Called virtual function (Init)!\n");
	exit(-1);
	return -1;
}

int vHandler::Finish() {
	ERROR("Called virtual function (Finish)!\n");
	exit(-1);
	return -1;
}

int vHandler::Type() {
	return 0;
}

int vHandler::NumberOfParameters () {
		output("Collecting parameters from Design elements\n");
		if (parSize < 0) {
			parSize = 0;
			for (size_t i=0; i<solver->hands.size(); i++) if (solver->hands[i].Type()  == HANDLER_DESIGN) {
				output("Getting number of parameters from %s\n", solver->hands[i]->node.name());
				int k = solver->hands[i]->NumberOfParameters();
				parSize += k;
			}
		} else {
			output("Done some time ago\n");
		}
		return parSize;
	};


int vHandler::Parameters (int type, double * tab) {
		int offset = 0, size = 0, ret=0;
		for (size_t i=0; i<solver->hands.size(); i++) if (solver->hands[i].Type()  == HANDLER_DESIGN) {
			output("Parameters from %s (%d)\n", solver->hands[i]->node.name(), type);
			size = solver->hands[i]->NumberOfParameters();
			if (offset + size > parSize) { offset = offset + size; break; }
			ret = solver->hands[i]->Parameters(type, tab+offset);
			if (ret) return ret;
			offset += size;
		}
		if (offset != parSize) {
				ERROR("Numer of parameters is inconsistent with first call to NumberOfParameters (in Parameters(%d)!", type);
				exit(-1);
				return -1;
		}
		return 0;
	};

