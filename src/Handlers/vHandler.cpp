#include "vHandler.h"
#include "../CommonHandler.h"

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

int vHandler::NumberOfParameters() {
	ERROR("Called virtual function (NumberOfParameters)!\n");
	exit(-1);
	return -1;
}
int vHandler::Parameters(int type, double * tab) {
	ERROR("Called virtual function (Parameters, type: %d)!\n", type);
	exit(-1);
	return -1;
}

