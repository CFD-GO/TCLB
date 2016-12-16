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

vHandler * getHandler(pugi::xml_node node) {
	vHandler * ret = NULL;
	std::string name(node.name());
	debug1("Parsing xml element %s\n", name.c_str());
	if (name=="VTK") {
		ret = new cbVTK;
    } else if (name=="BIN") {
		ret = new cbBIN;
    } else if (name=="TXT") {
    	ret = new cbTXT;
    } else if (name=="Log") {
		ret = new cbLog;
    } else if (name=="Save") {
	//	ret = new cbSave;
        error("Depreceted API call, use SaveMemoryDump or SaveBinary");
    } else if (name=="Load") {
        error("Depreceted API call, use LoadMemoryDump or LoadBinary");
//		ret = new acLoad;
    } else if (name=="SaveMemoryDump") {
		ret = new cbSaveMemoryDump;
    } else if (name=="LoadMemoryDump") {
		ret = new acLoadMemoryDump;
    } else if (name=="SaveBinary") {
		ret = new cbSaveBinary;
    } else if (name=="LoadBinary") {
		ret = new acLoadBinary;
    } else if (name=="CallPython"){
        ret = new cbPythonCall;
    } else if (name=="Stop") {
		ret = new cbStop;
    } else if (name=="Keep") {
		ret = new cbKeep;
    } else if (name=="Failcheck") {
                ret = new cbFailcheck;
    } else if (name=="Average") {
		ret = new cbAveraging;
    } else if (name=="Sample") {
		ret = new cbSample;
    } else if (name=="Solve") {
		ret = new acSolve;
    } else if (name=="OptSolve") {
		ret = new acOptSolve;
    } else if (name=="Adjoint") {
		pugi::xml_attribute attr = node.attribute("type");
		if (attr) {
			std::string type(attr.value());
			if (type == "unsteady") {
				ret = new acUSAdjoint;
			} else if (type == "steady") {
				ret = new acSAdjoint;
			} else {
				error("Unknown type of adjoint in xml: %s", type.c_str());
			}
		} else {
			pugi::xml_attribute attr = node.attribute("Iterations");
                	if (attr) {
				ret = new acSAdjoint;
				WARNING("Making a steady adjoint, because you gave me Iterations - better to state type explicitly.\n");
			} else {
				WARNING("Default adjoint is unsteady - better state type explicitly next time.\n");
				ret = new acUSAdjoint;
			}
		}
    } else if (name=="Params") {
	    ret = new acParams;
    } else if (name=="Units") {
		ret = new GenericContainer;
    } else if (name=="Geometry") {
		ret = new acGeometry;
    } else if (name=="Repeat") {
		ret = new acRepeat;
    } else if (name=="Threshold") {
		ret = new acThreshold;
    } else if (name=="ThresholdNow") {
		ret = new acThresholdNow;
    } else if (name=="CLBConfig") {
		ret = new MainContainer;
    } else if (name=="Model") {
		ret = new acModel;
    } else if (name=="InternalTopology") {
		ret = new InternalTopology;
    } else if (name=="OptimalControl") {
		ret = new OptimalControl;
    } else if (name=="OptimalControlSecond") {
		ret = new OptimalControlSecond;
    } else if (name=="Fourier") {
		ret = new Fourier;
    } else if (name=="BSpline") {
		ret = new BSpline;
    } else if (name=="RepeatControl") {
		ret = new RepeatControl;
    } else if (name=="Optimize") {
#ifdef WITH_NLOPT
		ret = new acOptimize;
#else
                ERROR("No NLOpt support. configure with --with-nlopt to use Optimize element\n");
                exit(-1);
#endif
	} else if (name=="Catalyst") {
#ifdef WITH_CATALYST
		ret = new cbCatalyst;
#else
                ERROR("No Catalyst support. configure with --with-catalyst\n");
                exit(-1);
#endif
    } else if (name=="FDTest") {
		ret = new acFDTest;
    } else if (name=="DumpSettings") {
		ret = new cbDumpSettings;
    } else if (name=="Init") {
		ret = new acInit;
    } else if (name=="Control") {
		ret = new conControl;
    } else if (name=="FieldParameter") {
		ret = new conFieldParameter;
    } else if (name=="ControlParameter") {
		ret = new conControlParameter;
	} else if (name=="SyntheticTurbulence") {
		ret = new acSyntheticTurbulence;
    } else if (name=="Run") {
		output("Skipping 'Run' element");
    }else {
		ERROR("Unknown element '%s'\n", node.name());
		return NULL;
	}
    // end else-if
    // SERIOUSLY!!!!
    // CONSIDER switch-case maybe:)
    // some observator pattern may be in hand

	if (ret != NULL) ret->node = node;
	return ret;
}


