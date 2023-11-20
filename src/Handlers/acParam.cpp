#include "acParam.h"

std::string acParam::xmlname = "Param";

int acParam::Init () {
	        std::string par, zone, value, gauge;
		bool permissive = false;
	        int zone_number = -1;
		Action::Init();
		pugi::xml_attribute attr;
		attr = node.attribute("name");
		if (attr) par = attr.value();
		attr = node.attribute("zone");
		if (attr) zone = attr.value();
		attr = node.attribute("value");
		if (attr) value = attr.value();
		attr = node.attribute("gauge");
		if (attr) gauge = attr.value();
		attr = context_attribute("permissive");
		if (attr) permissive = attr.as_bool();
		if (zone != "") {
                        const auto zone_iter = solver->setting_zones.find(zone);
                        if (zone_iter != solver->setting_zones.end())
                                zone_number = zone_iter->second;
			else {
				ERROR("Unknown zone %s (found while setting parameter %s)\n", zone.c_str(), par.c_str());
				return -1;
			}
		}
		double val = solver->units.alt(value);
		if (par == "") {
			if (gauge != "") {
				output("Gauge without setting\n");
			} else {
				ERROR("Setting name not specified in Param element\n");
				return -2;
			}
		} else {
			const Model::Setting& it = solver->lattice->model->settings.by_name(par);
			const Model::ZoneSetting& zoneit = solver->lattice->model->zonesettings.by_name(par);
			if (it) {
                output("Setting %s to %s (%lf)\n", par.c_str(), value.c_str(), val);
				solver->lattice->SetSetting(it, val);
			} else if (zoneit) {
                output("Setting %s in zone %s (%d) to %s (%lf)\n", par.c_str(), zone.c_str(), zone_number, value.c_str(), val);
				solver->lattice->zSet.set(zoneit.id, zone_number, val);
			} else {
				if (permissive) {
					if (gauge != "") {
						notice("Unknown setting %s with gauge\n", par.c_str());
					} else {
						WARNING("Unknown setting %s\n", par.c_str());
					}
				} else {
					ERROR("Unknown setting %s\n", par.c_str());
					return -3;
				}
			}
		}
		return 0;
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acParam > >;
