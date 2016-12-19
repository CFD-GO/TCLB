#include "vHandler.h"
#include "../CommonHandler.h"

#include "acFDTest.h"
#include "acGeometry.h"
#include "acInit.h"
#include "acLoadBinary.h"
#include "acLoadMemoryDump.h"
#include "acModel.h"
#include "acOptimize.h"
#include "acOptSolve.h"
#include "acParams.h"
#include "acRepeat.h"
#include "acSAdjoint.h"
#include "acSolve.h"
#include "acSyntheticTurbulence.h"
#include "acThreshold.h"
#include "acThresholdNow.h"
#include "Action.h"
#include "acUSAdjoint.h"
#include "BSpline.h"
#include "Callback.h"
#include "cbAveraging.h"
#include "cbBIN.h"
#include "cbCatalyst.h"
#include "cbDumpSettings.h"
#include "cbFailcheck.h"
#include "cbKeep.h"
#include "cbLog.h"
#include "cbPythonCall.h"
#include "cbSample.h"
#include "cbSaveBinary.h"
#include "cbSaveMemoryDump.h"
#include "cbStop.h"
#include "cbTXT.h"
#include "cbVTK.h"
#include "conControl.h"
#include "conControlParameter.h"
#include "conFieldParameter.h"
#include "Design.h"
#include "Fourier.h"
#include "GenericAction.h"
#include "GenericContainer.h"
#include "GenericOptimizer.h"
#include "InternalTopology.h"
#include "MainContainer.h"
#include "OptimalControl.h"
#include "OptimalControlSecond.h"
#include "RepeatControl.h"
#include "vHandler.h"
#include "NullHandler.h"

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

