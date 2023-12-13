#include "conControl.h"
#include <sstream>

std::string conControl::xmlname = "Control";

int conControl::Param (pugi::xml_node n) {
        std::string par, zone, value;
        bool permissive = false;
        int zone_number = -1;
        Action::Init();
        pugi::xml_attribute attr;
        attr = n.attribute("name");
        if (attr) par = attr.value();
        attr = n.attribute("zone");
        if (attr) zone = attr.value();
        attr = n.attribute("value");
        if (attr) {
		value = attr.value();
	} else {
                ERROR("Setting value not specified in Param element\n");
                return -2;
	}

	if (zone != "") {
                const auto zone_iter = solver->setting_zones.find(zone);
                if (zone_iter != solver->setting_zones.end())
                        zone_number = zone_iter->second;
		else {
			ERROR("Unknown zone %s (found while setting parameter %s)\n", zone.c_str(), par.c_str());
			return -1;
		}
	}

        if (par == "") {
                ERROR("Setting name not specified in Param element\n");
                return -2;
	} else {
		const Model::ZoneSetting& it = solver->lattice->model->zonesettings.by_name(par);
		if (it) {
                        std::vector<double> val;
                        get( context, value.c_str(), 1, val);
                        solver->lattice->zSet.set(it.id, zone_number, val);
                } else {
                        ERROR("Unknown setting %s\n", par.c_str());
                        return -3;
                }
	}
        return 0;
}


int conControl::get (Context& cont, const char * svar, double scale, std::vector<double>& fill) {
	        Context::iterator it;
	        it = cont.begin();
                if (it->second.size() != fill.size()) {
                        fill.resize(it->second.size());
                }
                for (size_t i=0; i < it->second.size(); i++) {
                        fill[i] = 0;
                }
                
	        std::istringstream s(svar);
                std::string var;
                int sum_i = 0;
                while (std::getline(s, var, '+'))
                {
                        std::istringstream ss(var);
                        std::string token;
                        std::getline(ss, token, '*');
                        Context::iterator it = cont.find(token);
                        double nscale;
                        bool ret = 0;
                        if (it == cont.end()) {
                                if (sum_i == 0) {
                                        error("Variable %s not found in context in Control in xml config\n", token.c_str());
                                        error("Syntax of time-dependent settings is: [Variable]*[scale with unit]");
                                        return -1;
                                }
                                ret = 1;
                        } else {
                                ret = static_cast<bool>( std::getline(ss, token, '*') );
                        }
                        if (ret) {
                                nscale = solver->units.alt(token);
                                if (std::getline(ss, token, '*')) {
                                        error("Too many \"*\" in Control in xml config\n");
                                        error("Syntax of time-dependent settings is: [Variable]*[scale with unit]");
                                        return -1;
                                }
                        } else {
                                nscale = 1;
                        }    
                        if (it == cont.end()) {
                                output("Setting 1 with a scale of %lf (*%lf)\n", nscale, scale);
                                for (size_t i=0; i < fill.size(); i++) {
                                        fill[i] += nscale * scale;
                                }
                        } else {
                                output("Getting %s from context with a scale of %lf (*%lf)\n", it->first.c_str(), nscale, scale);
                                for (size_t i=0; i < fill.size(); i++) {
                                        fill[i] += it->second[i] * nscale * scale;
                                }
                        }
                        sum_i++;
                }
                return 0;
	}


int conControl::Internal (pugi::xml_node n) {
	        const char * nm;
	        {
                        pugi::xml_attribute attr = n.attribute("file");
                        if (! attr) {
                                error("No file attribute in CSV in xml config\n");
                                return -1;
                        }
                        nm = attr.value();
                }
                {
                        std::vector< std::string > names;
                        Context csv_data;
                        std::string val;
                        std::string line;
                        std::ifstream myfile(nm);
                        if (! myfile.is_open()) {
                                error("Could not open file CSV %s\n", nm);
                                return -1;
                        }
                        if (! std::getline(myfile, line)) {
                                error("Empty file CSV %s\n", nm);
                                return -1;
                        }
                        {
                                std::istringstream ss(line);
                                std::string token;
                                while(std::getline(ss, token, ',')) {
                                        if (token[0] == '"') {
                                                if (token[token.size()-1] != '"') {
                                                        error("Column name in header started with \", but didn't end: %s\n", token.c_str());
                                                        return -1;
                                                }
                                                token = token.substr(1,token.size()-2);
                                        }
                                        names.push_back(token);
                                }                
                        }
                        {
                                int j=0;
                                while (std::getline(myfile, line)) {
                                        std::istringstream ss(line);
                                        std::string token;
                                        size_t i = 0;
                                        while(std::getline(ss, token, ',')) {
                                                if (i >= names.size()) {
                                                        error("More data then names in header in CSV file %s\n", nm);
                                                        return -1;
                                                }
                                                csv_data[ names[i] ].push_back( solver->units.alt(token.c_str()) );
                                                i++;
                                        }           
                                        if (i != names.size()) {
                                                error("Less data then names in header in CSV file %s\n", nm);
                                                return -1;
                                        }
                                        csv_data[ "_index" ].push_back( j );
                                        j++;     
                                }
                                output("Read %d rows from %s file\n", j, nm);
                        }
                        {
                                pugi::xml_attribute attr = n.attribute("Time");
                                const char * time_str;
                                double time_scale;
                                if (! attr) {
                                        time_scale = (double)iter/csv_data["_index"].size();
                                        time_str = "_index";
                                } else {
                                        time_str = attr.value();
                                        time_scale = 1;                                
                                }
                                csv_data["_time"];
                                if (get(csv_data, time_str, time_scale, csv_data["_time"])) {
                                        return -1;
                                }
                        }
                        int k = 0;
                        int max_k = csv_data[ "_time" ].size() - 2;
                        if (max_k < 0) {
                                error("Not enaugh records in CSV file or something went terribly wrong\n");
                                return -1;
                        }
                                
                        for (int i=0;i<iter; i++) {
                                double alpha;
                                while ((i > csv_data[ "_time" ][k+1]) && (k < max_k)) k++;
                                if (i < csv_data[ "_time" ][k]) {
                                        alpha = 0;
                                } else if (i > csv_data[ "_time" ][k+1]) {
                                        alpha = 1;
                                } else {
                                        alpha = (1.0 * i - csv_data[ "_time" ][k])/(csv_data[ "_time" ][k+1]-csv_data[ "_time" ][k]);
                                }
                                for (size_t j=0; j<names.size(); j++) {
                                        double val = csv_data[ names[j] ][k]*(1-alpha) + csv_data[ names[j] ][k+1]*alpha;
                                        context[ names[j] ].push_back(val);
                                }
                        }
                        
                }
	        for (pugi::xml_node n2 = n.first_child(); n2; n2 = n2.next_sibling()) {
                        std::string name2 = n2.name();
                        if (name2 == "Param") {
                                debug0("Param in CSV\n");
                                if (Param(n2)) return -1;
                        } else {
        		        error("Only Param allowed in %s in Control sub-element in config\n",n.name());
        		        return -1;
                        }
                }
                return 0;
        }


int conControl::Init () {
		Action::Init();
		iter = myround(everyIter);
		if (iter <= 0) {
		        error("Zero (or less) iterations in Control element in config\n");
		        return -1;
                }
                output("Setting iterations for time-dependent settings at %d\n",iter);
                warning("clearing old control\n");
                solver->lattice->zSet.setLen(iter);
                for (pugi::xml_node n = node.first_child(); n; n = n.next_sibling()) {
                        std::string name = n.name();
                        if (name == "Param") {
                	        debug2("Param in Control\n");
                                if (Param(n)) return -1;
                        } else if (name == "CSV") {
                                debug2("CSV in Control\n");
                                if (Internal(n)) return -1;
                        } else {
        		        error("Element %s not allowed in Control element in config\n", name.c_str());
        		        return -1;
                        }
                }
                std::vector<double> s;
                for (int i=0;i<iter; i++) {
                        s.push_back(sin(i/500.)*0.01);
                }
//                solver->lattice->zSet.set( ZONESETTINGS_InletVelocity, 0, s );
		return 0;
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< conControl > >;
