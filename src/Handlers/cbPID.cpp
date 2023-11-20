#include "cbPID.h"

std::string cbPID::xmlname = "PID";

int cbPID::Init () {
		Callback::Init();
		std::string glob;
		double stop;
		pugi::xml_attribute attr;
		int whats=0;
		attr = node.attribute("integral");
		if (attr) {
			glob = attr.value();
		}
		attr = node.attribute("target");
		if (attr) {
			target = solver->units.alt(attr.value());
		} else {
			ERROR("No target value provided in the PID element");
			return -1;
		}
		
		if (glob == "") {	
			ERROR("Name of the global not provided in the PID element");
			return -1;
		} else {
			const Model::Global& it = solver->lattice->model->globals.by_name(glob);
			if (it) {
				what = it.id;
			} else {
				ERROR("Uknown global name in the PID element");
				return -1;			
			}
		}
		std::string par,zone;
		attr = node.attribute("control");
		if (attr) par = attr.value();
		attr = node.attribute("zone");
		if (attr) zone = attr.value();
		if (zone == "") {
			zone_number = -1;
		} else{
                        const auto zone_iter = solver->setting_zones.find(zone);
			if (zone_iter != solver->setting_zones.end())
				zone_number = zone_iter->second;
			else {
				ERROR("Unknown zone %s (found while setting parameter %s)\n", zone.c_str(), par.c_str());
				return -1;
			}
		}
		
		if (par == "") {
			error("No zonal setting supplied for control in %s\n", node.name());
			return -1;
		} else {
			const Model::ZoneSetting& it = solver->lattice->model->zonesettings.by_name(par);
			if (it) {
				setting = it.id;
			} else {
				error("%s is not a valid zonal setting supplied for control in %s\n", par.c_str(), node.name());
			}
		}

		attr = node.attribute("IntegrationTime");
		if (attr) {
			itime = solver->units.alt(attr.value());
		} else {
		        itime = solver->units.alt("1s");
		}
		attr = node.attribute("DerivativeTime");
		if (attr) {
			dtime = solver->units.alt(attr.value());
		} else {
		        dtime = solver->units.alt("1s");
		}
		attr = node.attribute("scale");
		if (attr) {
			scale = solver->units.alt(attr.value());
		} else {
		        scale = solver->units.alt("1s");
		}
		integral = 0.0;
		old_err = 0.0;
		DT = everyIter;
                if (zone_number < 0) {
                        sval = solver->lattice->zSet.get(setting, 0, (size_t) 0);
                } else {
                        sval = solver->lattice->zSet.get(setting, zone_number, (size_t) 0);
                }
		old_iter_type = solver->iter_type;
		solver->iter_type |= ITER_LASTGLOB;
		return 0;
	}


int cbPID::DoIt () {
		Callback::DoIt();
		int ret=0;
		pugi::xml_attribute attr;
		double control, derivative;
		
                if (solver->mpi_rank == 0) {
                        double val = solver->lattice->globals[ what ];
                        double err = target - val;
                        output("PID criterium : %lg --> %lg (error: %lg)\n", val, target, err);
			
			integral = integral + (old_err + err)*DT/2.0;
			derivative = (err - old_err)/DT;
			
			control = err + integral / itime + derivative * dtime;
			old_err = err;
			
		}
                MPI_Bcast(&control, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                double nval;
/*                if (zone_number < 0) {
                	sval = solver->lattice->zSet.get(setting, 0, (size_t) 0);
		} else {
                	sval = solver->lattice->zSet.get(setting, zone_number, (size_t) 0);
		}*/
		nval = sval + control * scale;
                output("PID setting   : %lg --> %lg (control: %lg)", sval, nval, control);
		solver->lattice->zSet.set(setting, zone_number, nval);
		
		return ret;
	}


int cbPID::Finish () {
		solver->iter_type = old_iter_type;
		return Callback::Finish();
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbPID > >;
