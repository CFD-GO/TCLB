#include "cbCatalyst.h"
std::string cbCatalyst::xmlname = "Catalyst";
#include "../HandlerFactory.h"

#ifdef WITH_CATALYST

        #include "Catalyst.h"


int cbCatalyst::Init () {
		Callback::Init();
		pugi::xml_attribute attr;
		attr = node.attribute("export");
		bool cellDataNew = cellData;
                if (attr) {
                        std::string val = attr.value();
                        if (val == "CellData") {
                                cellDataNew = true;
                        } else if (val == "PointData") {
                                cellDataNew = false;
                        } else {
                                error("Unknown export value in Catalyst xml element: %s. Can be CellData or PointData\n",val.c_str());
                                return -1;
                        }
                }
		if (script_number == 0) {
		        CatalystAdaptor::Initialize(cellDataNew);
		        cellData = cellDataNew;
                } else {
                        if (cellData != cellDataNew) {
                                ERROR("Cannot mix CellData and PointData Catalyst outputs in one xml config\n");
                                return -1;
                        }
                }
                script_number ++;
		attr = node.attribute("script");
		if (attr) {
		        nm = attr.value();
                } else {
                        error("No script provided in Catalyst");
                        return -1;
                }
                attr = node.attribute("preprocess");
                int preprocess = 1;
                if (attr) {
                        std::string val = attr.value();
                        if (val == "yes") {
                                preprocess = 1;
                        } else if (val == "no") {
                                preprocess = 0;
                        } else {
                                error("Unknown preprocess value in Catalyst xml element: %s. Can be yes or no\n",val.c_str());
                                return -1;
                        }
                }
                if (preprocess) {
                        char fn[STRING_LEN];
                        char short_nm[STRING_LEN];
                        sprintf(short_nm, "SCRIPT%d", script_number);
//            		solver->outGlobalFile(short_nm, ".py", fn);
                        sprintf(fn, "%s_%s.py", solver->info.outpath, short_nm);
                        if (D_MPI_RANK == 0) {
                                notice("Preprocessing script %s --> %s\n", nm.c_str(), fn);
                                std::string prefix = "";
                                for (char * buf = solver->info.outpath; *buf; buf++) {
                                        if ((*buf) == '/') prefix += '\\';
                                        if ((*buf) == '\\' || (*buf) == '&' || (*buf) == ';' || (*buf) == ' ') {
                                                ERROR("illegal character in output prefix: %c!\n",*buf);
                                                return -1;                                        
                                        }
                                        prefix += *buf;
                                }
                                prefix += "_";
                                std::string com = "cat " + nm + " | sed";
                              //  com = com + " -e 's/\\([.]CreateView([^,]*,[^\"]*\\)\"\\([^\"]*\\)\"/\\1\"" + prefix + "\\2\"/g'";
                              //  com = com + " -e 's/\\([.]CreateWriter([^,]*,[^\"]*\\)\"\\([^\"]*\\)\"/\\1\"" + prefix + "\\2\"/g'";
				com = com + " -e \"s/[^/'\\\"]*\\.\\(png\\|pvti\\|pvtp\\)['\\\"]/" + prefix + "\\0/g\"";
                                com = com + " > " + fn;
                                debug2("preprocessing command: %s\n", com.c_str());
                                int ret = system(com.c_str());
                                if (ret) {
                                        ERROR("py preprocessing command failed\n");
                                        ERROR("commandline: %s\n", com.c_str());
                                        return -1;
                                }
                        }
                        nm = fn;                                                                                
                        MPI_Barrier(MPI_COMM_WORLD);
                }
//		attr = node.attribute("what");
//		if (attr) {
//		        s.add_from_string(attr.value(),',');
//                } else {
//                        s.add_from_string("all",',');
//                }
                CatalystAdaptor::AddScript(nm.c_str());
		return 0;
	}


int cbCatalyst::DoIt () {
		Callback::DoIt();
		solver->print("running Catalyst");
		CatalystAdaptor::CoProcess(*solver, solver->iter, solver->iter, 0);
		return 0;
	};


int cbCatalyst::Finish () {
	        CatalystAdaptor::Finalize();
	        return 0;
	};

int cbCatalyst::script_number = 0;
bool cbCatalyst::cellData = true;

#else


int cbCatalyst::Init () {
                ERROR("Catalyst not supported\n");
                return -1;
	}

int cbCatalyst::DoIt () {
		return 0;
	};

int cbCatalyst::Finish () {
	        return 0;
	};

#endif

// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_cbCatalyst(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "Catalyst") {
#ifdef WITH_CATALYST
		return new cbCatalyst;
#else
                ERROR("No Catalyst support. configure with --with-catalyst\n");
                exit(-1);
#endif
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_cbCatalyst >;
