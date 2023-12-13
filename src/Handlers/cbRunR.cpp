#include "cbRunR.h"

#include <algorithm>

#ifdef WITH_R

#define rNull Rcpp::NumericVector(0)
template <typename T> Rcpp::IntegerVector SingleInteger(T i) { Rcpp::IntegerVector v(1); v[0] = i; return v; }

//RInside RunR::R(0,0,true,false,true);
RInside RunR::R(0,0,true,false,true);

class rWrapper { // Wrapper for all my R objects
public:
	Solver * solver;
	vHandler * hand;
	virtual SEXP Dollar(std::string name) { return rNull; };
	virtual void DollarAssign(std::string name, SEXP v) {};
	virtual Rcpp::CharacterVector Names() { return Rcpp::CharacterVector(0); };
	virtual std::string print() {
		char str[2048];
		sprintf(str,"rWrapper(%p)\n",solver);
		return std::string(str);
	}
	virtual SEXP Call(Rcpp::List) { ERROR("R: Called a non-callable rWrapper"); return rNull; }
	virtual ~rWrapper() { debug0("R: Destroying wrapper");};
	template <class T>
	SEXP rWrap(T * ptr) {
	        Rcpp::XPtr< rWrapper > a(ptr);
		a->solver = solver;
		a->hand = hand;
		Rcpp::Function wraper("CLBFunctionWrap");
		Rcpp::Function ra = wraper(a);
	        ra.attr("class") = "CLB";
		ra.attr("xptr") = a;
	        return ra;
	}
};

class rZoneSetting : public rWrapper {
	std::string name;
	int idx;
public:
	std::string print() { return name + " (ZoneSetting)"; }
	rZoneSetting(const char* name_, const int idx_): name(name_), idx(idx_) {};
	SEXP Dollar(std::string name) {
	  return Rcpp::NumericVector(0.0);
	}
	void DollarAssign(std::string zone, SEXP v_) {
		WARNING("in zone %s setting parameter %s\n", zone.c_str(), name.c_str());
		Rcpp::NumericVector v(v_);
	        int zone_number = -1;
                const auto zone_iter = solver->setting_zones.find(zone);
                if (zone_iter != solver->setting_zones.end())
                        zone_number = zone_iter->second;
                else {
                        WARNING("Unknown zone %s (found while setting parameter %s)\n", zone.c_str(), name.c_str());
                        return;
                }
		solver->lattice->zSet.set(idx, zone_number, v[0]);
	}
	Rcpp::CharacterVector Names() {
		Rcpp::CharacterVector ret;
                ret.reserve(solver->setting_zones.size());
                std::transform(solver->setting_zones.cbegin(), solver->setting_zones.cend(), std::back_inserter(ret), [](const auto& pair){ return pair.first; });
		return ret;
	}

};


class rSettings : public rWrapper {
public:
	std::string print() { return "Settings"; }
	SEXP Dollar(std::string name) {
		const Model::ZoneSetting& set = solver->lattice->model->zonesettings.by_name(name);
		if (set) return rWrap(new rZoneSetting(name.c_str(),set.id));
		return Rcpp::NumericVector(0);
	}

	void DollarAssign(std::string name, SEXP v_) {
		const Model::ZoneSetting& zset = solver->lattice->model->zonesettings.by_name(name);
		if (zset) {
			ERROR("R: ZoneSetting not supported in rSetting");
		}
		const Model::Setting& set = solver->lattice->model->settings.by_name(name);
		if (set) {
			Rcpp::NumericVector v(v_);
			solver->lattice->SetSetting(set, v[0]);
		} else {
			ERROR("R: Unknown setting");
		}
	}
	Rcpp::CharacterVector Names() {
		Rcpp::CharacterVector ret;
		for (const Model::Setting& it : solver->lattice->model->settings) {
			ret.push_back(it.name);
		}
		for (const Model::ZoneSetting& it : solver->lattice->model->zonesettings) {
			ret.push_back(it.name);
		}
		return ret;
	}
};

class rFields : public rWrapper {
public:
	std::string print() { return "Fields"; }
	SEXP Dollar(std::string name) {
		const Model::Field& it = solver->lattice->model->fields.by_name(name);
		if (!it) {
			ERROR("R: Unknown parameter");
			return Rcpp::NumericVector(0);
		}
		lbRegion reg = solver->lattice->region;
		Rcpp::NumericVector ret(reg.size());
		Rcpp::IntegerVector retdim(3);
		retdim[0] = reg.nx;
		retdim[1] = reg.ny;
		retdim[2] = reg.nz;
		ret.attr("dim") = retdim;
	    solver->lattice->Get_Field(it.id, &ret[0]); 
		return ret;
	}

	void DollarAssign(std::string name, SEXP v_) {
		const Model::Field& it = solver->lattice->model->fields.by_name(name);
		if (!it) {
			ERROR("R: Unknown parameter");
			return;
		}
		Rcpp::NumericVector v(v_);
		if (v.size() != solver->region.size()) {
			ERROR("Wrong size of the parameter field!");
			return;
		}
	        solver->lattice->Set_Field(it.id,&v[0]); 
		return;
	}
	Rcpp::CharacterVector Names() {
		Rcpp::CharacterVector ret;
		for (const Model::Field& it : solver->lattice->model->fields) {
			ret.push_back(it.name);
		}
		return ret;
	}
};

class rParameters : public rWrapper {
public:
	std::string print() { return "Parameters"; }
	SEXP Dollar(std::string name) {
		int len = hand->NumberOfParameters();
		Rcpp::NumericVector ret(len);
		if (name == "Values") {
			hand->Parameters(PAR_GET, &ret[0]);
		} else if (name == "Lower") {
			hand->Parameters(PAR_LOWER, &ret[0]);
		} else if (name == "Upper") {
			hand->Parameters(PAR_UPPER, &ret[0]);
		} else if (name == "Gradient") {
			hand->Parameters(PAR_GRAD, &ret[0]);
		} else if (name == "X") {
			hand->Parameters(PAR_X, &ret[0]);
		} else if (name == "Y") {
			hand->Parameters(PAR_Y, &ret[0]);
		} else if (name == "Z") {
			hand->Parameters(PAR_Z, &ret[0]);
		} else if (name == "T") {
			hand->Parameters(PAR_T, &ret[0]);
		} else {
			ERROR("R: Unknown parameter");
			return Rcpp::NumericVector(0);
		}
		return ret;
	}

	void DollarAssign(std::string name, SEXP v_) {
		Rcpp::NumericVector v(v_);
		int len = hand->NumberOfParameters();
		if (v.size() != len) {
			ERROR("R: Wrong number of parameters");
			return;
		}
		if (name == "Values") {
			hand->Parameters(PAR_SET, &v[0]);
		} else {
			ERROR("R: Cannot set anything but Values");
			return;
		}
		return;
	}
	Rcpp::CharacterVector Names() {
		Rcpp::CharacterVector ret;
		ret.push_back("Values");
		ret.push_back("Lower");
		ret.push_back("Upper");
		ret.push_back("Gradient");
		ret.push_back("X");
		ret.push_back("Y");
		ret.push_back("Z");
		ret.push_back("T");
		return ret;
	}
};

class rQuantities : public rWrapper {
public:
	std::string print() { return "Quantities"; }
	SEXP Dollar(std::string name) {
		lbRegion reg = solver->lattice->region;
		Rcpp::NumericVector ret;
		bool si = false;
		std::string quant = name;
		size_t last_index = name.find_last_not_of(".");
		if (last_index != std::string::npos) {
			std::string result = name.substr(last_index + 1);
			if (result == "si") {
				si = true;
				quant = name.substr(0, last_index);
			}
		}
		const Model::Quantity& it = solver->lattice->model->quantities.by_name(quant);
		if (!it) {
			ERROR("R: Unknown Quantity");
			return Rcpp::NumericVector(0);
		}
		double v = 1;
		if (si) v = solver->units.alt(it.unit);
		int comp = 1;
		if (it.isVector) comp = 3;
		real_t* tmp = new real_t[reg.size()*comp];
                solver->lattice->GetQuantity(it.id, reg, tmp, 1/v);
		ret = Rcpp::NumericVector(reg.size()*comp);
		if (comp != 1) {
			Rcpp::IntegerVector retdim(4);
			retdim[0] = comp;
			retdim[1] = reg.nx;
			retdim[2] = reg.ny;
			retdim[3] = reg.nz;
			ret.attr("dim") = retdim;
		} else {
			Rcpp::IntegerVector retdim(3);
			retdim[0] = reg.nx;
			retdim[1] = reg.ny;
			retdim[2] = reg.nz;
			ret.attr("dim") = retdim;
		}
		for (size_t i=0; i<reg.sizeL()*comp; i++) {
			ret[i] = tmp[i];
		}
		delete[] tmp;
		return ret;
	}
	Rcpp::CharacterVector Names() {
		Rcpp::CharacterVector ret;
		for (const Model::Quantity& it : solver->lattice->model->quantities) {
			if (it.isAdjoint) continue;
			ret.push_back(it.name);
			ret.push_back(it.name + ".si");
		}
		return ret;
	}
};

class rGlobals : public rWrapper {
public:
	std::string print() { return "Globals"; }
	SEXP Dollar(std::string name) {
		Rcpp::NumericVector ret(1);
		if (name == "Iteration") {
			ret[0] = solver->lattice->Iter;
			return ret;
		}

		bool si = false;
		std::string glob = name;
		size_t last_index = name.find_last_not_of(".");
		if (last_index != std::string::npos) {
			std::string result = name.substr(last_index + 1);
			if (result == "si") {
				si = true;
				glob = name.substr(0, last_index);
			}
		}
		const Model::Global& it = solver->lattice->model->globals.by_name(glob);
		if (!it) {
			ERROR("R: Unknown global");
			return Rcpp::NumericVector(0);
		}

		double v = 1;
		if (si) v = solver->units.alt(it.unit);
		ret[0] = v * solver->lattice->globals[it.id];
		return ret;
	}
	Rcpp::CharacterVector Names() {
		Rcpp::CharacterVector ret;
		ret.push_back("Iteration");
		for (const Model::Global& it : solver->lattice->model->globals) {
			if (it.isAdjoint) continue;
			ret.push_back(it.name);
			ret.push_back(it.name + ".si");
		}
		return ret;
	}
};

class rAction : public rWrapper {
	std::string name;
public:
	std::string print() { return name + " (Action)"; }

	rAction(const char* name_): name(name_) {};
	SEXP Call(Rcpp::List args) {
		int Snap = solver->lattice->Snap;
		const Model::Action& it = solver->lattice->model->actions.by_name(name);
		if (it) {
			solver->lattice->RunAction(it.id, solver->iter_type);
		} else {
			ERROR("R: Unknown Action");
		}
		return rNull;
	}
};

class rActions : public rWrapper {
	std::string print() { return "Actions"; }
public:
	SEXP Dollar(std::string name) {
		return rWrap(new rAction(name.c_str()));
	}
	Rcpp::CharacterVector Names() {
		Rcpp::CharacterVector ret;
		for (const Model::Action& it : solver->lattice->model->actions) {
			ret.push_back(it.name);
		}
		return ret;
	}
};

class rGeometry : public rWrapper {
public:
	std::string print() { return "Geometry"; }

	void DollarAssign(std::string name, SEXP v_) {
		Rcpp::IntegerVector v(v_);
		lbRegion reg = solver->lattice->region;
		size_t size = reg.sizeL();
		{
			flag_t * NodeType = new flag_t[size];
			solver->lattice->GetFlags(reg, NodeType);
			for (const Model::NodeTypeGroupFlag& it : solver->lattice->model->nodetypegroupflags) {
				bool some_na = false;
				for (size_t i=0;i<size;i++) {
					if (Rcpp::IntegerVector::is_na(v[i])) {
						some_na = true;
					} else {
						NodeType[i] = (NodeType[i] - (NodeType[i] & it.flag)) + ((v[i] - 1) << it.shift);
					}
				}
				if (some_na) {
					ERROR("Some NA in Geometry (%s) assignment", it.name.c_str());
				}
			}
			solver->lattice->FlagOverwrite(NodeType, reg);
			delete[] NodeType;
		}
		return;
	}

SEXP Dollar(std::string name) {
	lbRegion reg = solver->lattice->region;
	size_t size = reg.sizeL();
	if (name == "dx") return SingleInteger(reg.dx);
	if (name == "dy") return SingleInteger(reg.dy);
	if (name == "dz") return SingleInteger(reg.dz);
	if (name == "size") return SingleInteger(reg.size());
	Rcpp::IntegerVector retdim(3);
	retdim[0] = reg.nx;
	retdim[1] = reg.ny;
	retdim[2] = reg.nz;
	if (name == "dim") return retdim;
	if ((name == "X") || (name == "Y") || (name == "Z")) { // Positions
		double unit = 1/solver->units.alt("1m");
		int dir = -1;
		if (name == "X") dir = 0;
		if (name == "Y") dir = 1;
		if (name == "Z") dir = 2;
		Rcpp::NumericVector small(size);
		small.attr("dim") = retdim;
		size_t i=0;
		for (int z=0;z<reg.nz;z++)
		for (int y=0;y<reg.ny;y++)
		for (int x=0;x<reg.nx;x++) {
			double val=0;
			switch (dir) {
			case 0:	val = reg.dx + x; break;
			case 1: val = reg.dy + y; break;
			case 2: val = reg.dz + z; break;
			}
			val = (val + 0.5) * unit; 
			small[i] = val;
			i++;
		}
		return small;
	}

	const Model::NodeTypeGroupFlag& it = solver->lattice->model->nodetypegroupflags.by_name(name);
	if (it) { // Geometry components
		flag_t * NodeType = new flag_t[size];
		solver->lattice->GetFlags(reg, NodeType);
		Rcpp::IntegerVector small(size);
		small.attr("dim") = retdim;
		for (size_t i=0;i<size;i++) {
			small[i] = 1 + ((NodeType[i] & it.flag) >> it.shift);
		}
		Rcpp::CharacterVector levels(it.max+1);
		levels[0] = "None";
		for (const Model::NodeTypeFlag& it2 : solver->lattice->model->nodetypeflags) {
			if (it2.group_id == it.id) {
				int idx = it2.flag >> it.shift;
				if (idx < levels.size()) levels[idx] = it2.name;
			}
		}
		small.attr("levels") = levels;
		small.attr("class") = "factor";
		delete[] NodeType;
		return small;
	}
	ERROR("R: Unknown component of Geometry");
	return Rcpp::IntegerVector(0);
}
	virtual Rcpp::CharacterVector Names() {
		Rcpp::CharacterVector ret;
		ret.push_back("dx");
		ret.push_back("dy");
		ret.push_back("dz");
		ret.push_back("X");
		ret.push_back("Y");
		ret.push_back("Z");
		ret.push_back("size");
		ret.push_back("dim");
		for (const Model::NodeTypeGroupFlag& it : solver->lattice->model->nodetypegroupflags) {
			ret.push_back(it.name);
		}
		return ret;
	}

};


class rInfo: public rWrapper {
	public:
	std::string print() { return "Info"; }
	SEXP Dollar(std::string name) {
		Rcpp::CharacterVector ret;
		if (name == "OutputPath") {
			ret.push_back(this->solver->info.outpath);
			return ret;
		}
		ERROR("R: Not implemented!");
		return ret;
	}

	Rcpp::CharacterVector Names() {
		Rcpp::CharacterVector ret;
		ret.push_back("OutputPath");
		return ret;
	}
};


class rSolver : public rWrapper {
public:
	std::string print() { return "Solver"; }

	SEXP Dollar(std::string name) {
	  if (name == "Settings") {  
	    return rWrap(new rSettings());
	  } else if (name == "Fields") {  
	    return rWrap(new rFields());
	  } else if (name == "Parameters") {  
	    return rWrap(new rParameters());
	  } else if (name == "Quantities") {  
	    return rWrap(new rQuantities());
	  } else if (name == "Globals") {  
	    return rWrap(new rGlobals());
	  } else if (name == "Actions") {  
	    return rWrap(new rActions());
	  } else if (name == "Geometry") {  
	    return rWrap(new rGeometry());
	  } else if (name == "Info") {
		return rWrap(new rInfo());
	  }
	  return rNull;
	}
	Rcpp::CharacterVector Names() {
		Rcpp::CharacterVector ret;
		ret.push_back("Settings");
		ret.push_back("Fields");
		ret.push_back("Parameters");
		ret.push_back("Quantities");
		ret.push_back("Globals");
		ret.push_back("Actions");
		ret.push_back("Geometry");
		return ret;
	}
};

class rXMLNode : public rWrapper {
public:
	pugi::xml_node node;
	rXMLNode(const pugi::xml_node& node_): node(node_) {};
	std::string print() { return "XMLNode"; }
	SEXP Call(Rcpp::List args) {
		output("call to xml node: %s\n",node.name());
                Handler hand(node, solver);
                if (hand) {
                        if (hand.Type() & HANDLER_DESIGN) {
                                error("No support for DESIGN XML elements in RunR");
                        } else if (hand.Type() & HANDLER_CALLBACK) {
                                if ((hand.hand->everyIter != 0) || (hand.Type() & HANDLER_DESIGN)) {
	                                error("No support for CALLBACK XML elements with Iterations set in RunR");
                                } else {
                                        hand.DoIt();
                                }
                        }
                };
		return rNull;
	}

};




SEXP CLBFunctionCall(Rcpp::XPtr< rWrapper > obj, Rcpp::List args) {
	debug2("R: Calling %s",obj->print().c_str());
	return obj->Call(args);
}


SEXP CLBDollar(SEXP fobj_, std::string name) {
	Rcpp::Function fobj = fobj_;
	Rcpp::XPtr< rWrapper > obj = fobj.attr("xptr");
	debug2("R: Getting %s from %s",name.c_str(),obj->print().c_str());
	return obj->Dollar(name);
}

SEXP CLBPrint(SEXP fobj_) {
	Rcpp::Function fobj = fobj_;
	Rcpp::XPtr< rWrapper > obj = fobj.attr("xptr");
	std::string s = obj->print();
	notice("R: Printing %s",s.c_str());
	return Rcpp::CharacterVector(s);
}

SEXP CLBNames(SEXP fobj_) {
	Rcpp::Function fobj = fobj_;
	Rcpp::XPtr< rWrapper > obj = fobj.attr("xptr");
	return obj->Names();
}


SEXP CLBDollarAssign(SEXP fobj_, std::string name, SEXP v) {
	Rcpp::Function fobj = fobj_;
	Rcpp::XPtr< rWrapper > obj = fobj.attr("xptr");
	debug2("R: Setting %s from %s",name.c_str(),obj->print().c_str());
	obj->DollarAssign(name,v);
	return fobj_;
}

extern "C" {

void CLB_WriteConsoleLine( const char* message, int oType) {
	if (oType == 0) {
		output("R: %s",message);
	} else if (oType == 1) {
		error("R: %s", message);
	} else {
		notice("R: (%d) %s",oType,message);
	}
}

void CLB_WriteConsoleEx( const char* message, int len, int oType ){
	const int buf_size = 4000;
	static char buf[buf_size];
	static int pos = 0;
	static int oldType = 0;
	if (oldType != oType) {
		if (pos > 0) {
			buf[pos] = '\0';
			CLB_WriteConsoleLine(buf,oType);
			pos = 0;
		}
	}
	oldType = oType;
	while (*message) {
		buf[pos] = *message;
		message++;
		if (buf[pos] == '\n') {
			buf[pos] = '\0';
			CLB_WriteConsoleLine(buf,oType);
			pos = 0;
		} else {
			pos++;
			if (pos == buf_size - 1) {
				buf[pos] = '\0';
                        	CLB_WriteConsoleLine(buf,oType);
                        	pos = 0;
			}
		}		
	}
}
}

  #define R_INTERFACE_PTRS
  #include <Rinterface.h>

int RunR::Init() {
	Callback::Init();
	notice("R: Initializing R environment ...");

	R["CLBFunctionCall"] = Rcpp::InternalFunction( &CLBFunctionCall );
	R["$.CLB"]           = Rcpp::InternalFunction( &CLBDollar );
	R["[[.CLB"]          = Rcpp::InternalFunction( &CLBDollar );
	R["$<-.CLB"]         = Rcpp::InternalFunction( &CLBDollarAssign );
	R["print.CLB"]       = Rcpp::InternalFunction( &CLBPrint );
	R["names.CLB"]       = Rcpp::InternalFunction( &CLBNames );
	R.parseEval("'CLBFunctionWrap' <- function(obj) { function(...) CLBFunctionCall(obj, list(...)); }");

	rWrapper base;
	base.solver = solver;
	base.hand = this;
	R["Solver"]          = base.rWrap(new  rSolver ());

        ptr_R_WriteConsoleEx = CLB_WriteConsoleEx ;
        ptr_R_WriteConsole = NULL;
	R_Outputfile = NULL;
	R_Consolefile = NULL;

	R.parseEval("options(prompt='[  ] R:> ');");

	interactive = false;
	echo = true;

	pugi::xml_attribute attr;
	attr = node.attribute("interactive");
	if (attr) interactive = attr.as_bool();
	attr = node.attribute("echo");
	if (attr) echo = attr.as_bool();

	source = "";
        for (pugi::xml_node par = node.first_child(); par; par = par.next_sibling()) {
		if (par.type() == pugi::node_element) {
			char nd_name[20];
			sprintf(nd_name, "xml_%0zx", par.hash_value());
			R[nd_name]          = base.rWrap(new  rXMLNode (par));
			
			source = source + nd_name + "()\n";
			output("element\n");
		} else if (par.type() == pugi::node_pcdata) {
			output("pcdata\n");
			source += par.value();
		} else if (par.type() == pugi::node_cdata) {
			output("cdata\n");
			source += par.value();
		} else {
			output("Unknown\n");
		}
	}
//	output("----- RunR -----\n");
//	output("%s\n",source.c_str());
//	output("----------------\n");	
	
	return 0;
}


int RunR::DoIt() {
	try {
		if (source != "") {
			solver->print("Running R ...");
			if (echo) {
				output("----- RunR -----\n");
				output("%s\n",source.c_str());
				output("----------------\n");
			}
			R.parseEval(source);
		}
		if (!interactive) {
			if (echo) NOTICE("You can run interactive R session with Ctrl+X");
			int c = kbhit();
			if (c == 24) {
				int a = getchar();
				if (a == c) {
					interactive = true;
				}
			}
		}
		if (interactive) {
			R_ReplDLLinit();
			while( R_ReplDLLdo1() > 0 ) {}
		}
	} catch (...) {
		return -1;
	}
	return 0;
}


#endif // WITH_R

// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_RunR(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "RunR") {
#ifdef WITH_R
    return new RunR;
#else
    ERROR("No R support. configure with --enable-rinside\n");
    exit(-1);  
#endif
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_RunR >;


