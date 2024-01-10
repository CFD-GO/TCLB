#include "cbRunR.h"

#include <algorithm>

#ifdef WITH_R

#define rNull Rcpp::NumericVector(0)
template <typename T> Rcpp::IntegerVector SingleInteger(T i) { Rcpp::IntegerVector v(1); v[0] = i; return v; }

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
		for (const auto& [name, id] : solver->setting_zones) {
			ret.push_back(name);
		}
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
		size_t size = solver->lattice->getLocalSize();
		std::vector<int> retdim = solver->lattice->shape();
		std::vector<real_t> tmp = solver->lattice->getField(it); 
		Rcpp::NumericVector ret(tmp.begin(), tmp.end());
		ret.attr("dim") = Rcpp::IntegerVector(retdim.begin(), retdim.end());
		return ret;
	}

	void DollarAssign(std::string name, SEXP v_) {
		const Model::Field& it = solver->lattice->model->fields.by_name(name);
		if (!it) {
			ERROR("R: Unknown parameter");
			return;
		}
		Rcpp::NumericVector v(v_);
		size_t size = solver->lattice->getLocalSize();
		if ((size_t) v.size() != size) {
			ERROR("Wrong size of the parameter field!");
			return;
		}
		std::vector<real_t> tmp(v.begin(), v.end());
        solver->lattice->setField(it,tmp);
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
		bool si = false;
		std::string quant = name;
		size_t last_index = name.find_last_of(".");
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
		int comp = it.getComp();
		size_t size = solver->lattice->getLocalSize();
		std::vector<real_t> tmp = solver->lattice->getQuantity(it, 1/v);
		Rcpp::NumericVector ret(tmp.begin(), tmp.end());
		std::vector<int> retdim = solver->lattice->shape();
		if (comp != 1) {
			retdim.insert(retdim.begin(),1,comp);
		}
		ret.attr("dim") = Rcpp::IntegerVector(retdim.begin(), retdim.end());
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
		size_t size = solver->lattice->getLocalSize();
		const Model::NodeTypeGroupFlag& it = solver->lattice->model->nodetypegroupflags.by_name(name);
		if (it) {
			std::vector<big_flag_t> NodeType = solver->lattice->getFlags();
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
			return solver->lattice->setFlags(NodeType);
		}
		ERROR("R: Unknown component of Geometry");
		return;
	}

SEXP Dollar(std::string name) {
	size_t size = solver->lattice->getLocalSize();
	// if (name == "dx") return SingleInteger(reg.dx);
	// if (name == "dy") return SingleInteger(reg.dy);
	// if (name == "dz") return SingleInteger(reg.dz);
	if (name == "size") return SingleInteger(size);
	std::vector<int> retdim = solver->lattice->shape();
	Rcpp::IntegerVector r_retdim(retdim.begin(), retdim.end());
	if (name == "dim") return r_retdim;
	if ((name == "X") || (name == "Y") || (name == "Z")) { // Positions
		double unit = 1/solver->units.alt("1m");
		Model::Coord dir = solver->lattice->model->coords.by_name(name);
		std::vector<real_t> tmp = solver->lattice->getCoord(dir, unit);
		Rcpp::NumericVector ret(tmp.begin(),tmp.end());
		ret.attr("dim") = r_retdim;
		return ret;
	}

	const Model::NodeTypeGroupFlag& it = solver->lattice->model->nodetypegroupflags.by_name(name);
	if (it) { // Geometry components
		std::vector<big_flag_t> NodeType = solver->lattice->getFlags();
		Rcpp::IntegerVector small(size);
		small.attr("dim") = r_retdim;
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
		return small;
	}
	ERROR("R: Unknown component of Geometry");
	return Rcpp::IntegerVector(0);
}
	virtual Rcpp::CharacterVector Names() {
		Rcpp::CharacterVector ret;
		// ret.push_back("dx");
		// ret.push_back("dy");
		// ret.push_back("dz");
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
			ret.push_back(this->solver->outpath);
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
		ret.push_back("Info");
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


namespace RunR {
	RInside& GetR() {
		static RInside * Rptr;
		if (Rptr == NULL) {
			notice("R: Initializing R environment ...");
			Rptr = new RInside(0,0,true,false,true);
			RInside& R = *Rptr;

			R["CLBFunctionCall"] = Rcpp::InternalFunction( &CLBFunctionCall );
			R["$.CLB"]           = Rcpp::InternalFunction( &CLBDollar );
			R["[[.CLB"]          = Rcpp::InternalFunction( &CLBDollar );
			R["$<-.CLB"]         = Rcpp::InternalFunction( &CLBDollarAssign );
			R["[[<-.CLB"]        = Rcpp::InternalFunction( &CLBDollarAssign );
			R["print.CLB"]       = Rcpp::InternalFunction( &CLBPrint );
			R["names.CLB"]       = Rcpp::InternalFunction( &CLBNames );
			R.parseEval("'CLBFunctionWrap' <- function(obj) { function(...) CLBFunctionCall(obj, list(...)); }");
			ptr_R_WriteConsoleEx = CLB_WriteConsoleEx ;
			ptr_R_WriteConsole = NULL;
			R_Outputfile = NULL;
			R_Consolefile = NULL;
			R.parseEval("options(prompt='[  ] R:> ');");
		}
		return *Rptr;
	};

	void parseEval(const std::string& source) {
		RInside& R = GetR();
		R.parseEval(source);
	}

	int replInit() {
		R_ReplDLLinit();
		return 0;
	}

	int replDo() {
		return R_ReplDLLdo1();
	}

	SEXP wrap_solver(Solver* solver, vHandler * hand) {
		rWrapper base;
		base.solver = solver;
		base.hand = hand;
		return base.rWrap(new  rSolver ());
	}

	SEXP wrap_handler(Solver* solver, vHandler * hand, const pugi::xml_node& par) {
		rWrapper base;
		base.solver = solver;
		base.hand = hand;
		return base.rWrap(new  rXMLNode (par));
	}
};

namespace RunPython {
	bool has_reticulate = false;
	bool py_initialised = false;
	void initializePy();

	void parseEval(const std::string& source) {
		if (! py_initialised) initializePy();
		Rcpp::Function py_run_string("py_run_string");
		py_run_string(source);
		return;
	}	

	void initializePy() {
		if (py_initialised) return;
		RInside& R = RunR::GetR();
		has_reticulate = R.parseEval("require(reticulate, quietly=TRUE)");
		if (!has_reticulate) throw std::string("Tried to call Python, but no reticulate installed");
		py_initialised = true;
		R.parseEval(
			"py_names = function(obj) names(obj)                                   \n"
			"py_element = function(obj, name) {                                    \n"
			"  ret = `[[`(obj,name)                                                \n"
			"  if (is.factor(ret)) {                                               \n"
			"    as.integer(ret) - 1L                                              \n"
			"  } else {                                                            \n"
			"    ret                                                               \n"
			"  }                                                                   \n"
			"}                                                                     \n"
			"py_element_assign = function(obj, name, value) `[[<-`(obj,name,value) \n"
			"r_to_py.CLB = function(x, convert=FALSE) py$S3(reticulate:::py_capsule(x))\n"
		);
		parseEval(
			"class S3:                                                             \n"
			"  def __init__(self, obj):                                            \n"
			"    object.__setattr__(self,'obj',obj)                                \n"
			"  def print(self):                                                    \n"
			"    return r.print(self.obj)                                          \n"
			"  def __dir__(self):                                                  \n"
			"    return r.py_names(self.obj)                                       \n"
			"  def __iter__(self):                                                 \n"
			"    for n in r.py_names(self.obj):                                    \n"
			"      yield n, r.py_element(self.obj, n)                              \n"
			"  def __getattr__(self, index):                                       \n"
			"    if index.startswith('_'):                                         \n"
			"      return None                                                     \n"
			"    return r.py_element(self.obj, index)                              \n"
			"  def __setattr__(self, index, value):                                \n"
			"    return r.py_element_assign(self.obj, index, value)                \n"
			"  def __call__(self):                                                 \n"
			"    raise TypeError('not really callable')                            \n"
		);
		R.parseEval(
			"py$Solver = r_to_py(Solver)"
		);
	}

	int replRun() {
		if (! py_initialised) initializePy();
		Rcpp::Function repl_python("repl_python");
		repl_python();
		return 0;
	}
}

int cbRunR::Init() {
	Callback::Init();
	RInside& R = RunR::GetR();
	R["Solver"] = RunR::wrap_solver(solver,this);

	python = false;
	interactive = false;
	echo = true;

	std::string name = node.name();
	if (name == "RunPython") python = true;
	pugi::xml_attribute attr;
	attr = node.attribute("interactive");
	if (attr) interactive = attr.as_bool();
	attr = node.attribute("echo");
	if (attr) echo = attr.as_bool();

	s_tag++;
	tag = s_tag;
	
	source = "";
    for (pugi::xml_node par = node.first_child(); par; par = par.next_sibling()) {
		if (par.type() == pugi::node_element) {
			if (python) {
				ERROR("Code-embedded xml nodes not supported for python");
				return -1;
			}
			char nd_name[20];
			sprintf(nd_name, "xml_%0zx", par.hash_value());
			R[nd_name] = RunR::wrap_handler(solver,this,par);
			
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
	if (echo) {
		output("-----[ %9s code %03d ]-----\n", node.name(), tag);
		output("%s\n",source.c_str());
		output("--------------------------------\n");
	}
	return 0;
}

int cbRunR::s_tag = 0;

int cbRunR::DoIt() {
	try {
		if (source != "") {
			output("%8d it Executing %s code %03d\n", solver->iter, node.name(), tag);
			if (python) {
				RunPython::parseEval(source);
			} else {
				RunR::parseEval(source);
			}
		}
		if (!interactive) {
			if (echo) NOTICE("You can run interactive %s session with Ctrl+X", node.name());
			int c = kbhit();
			if (c == 24) {
				int a = getchar();
				if (a == c) {
					interactive = true;
				}
			}
		}
		if (interactive) {
			if (python) {
				RunPython::replRun();
			} else {
				RunR::replInit();
				while( RunR::replDo() > 0 ) {}
			}
		}
	} catch (Rcpp::exception& ex) {
		ERROR("Caught Rcpp exception");
		return -1;
	} catch(std::exception &ex) {	
		ERROR("Caught std exception: %s", ex.what());
		return -1;
	} catch (...) {
		ERROR("Caught uknown exception");
		return -1;
	}
	return 0;
}


#endif // WITH_R

// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_RunR(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "RunR" || name == "RunPython") {
#ifdef WITH_R
    return new cbRunR;
#else
    ERROR("No R support. configure with --enable-rinside\n");
    exit(-1);  
#endif
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_RunR >;


