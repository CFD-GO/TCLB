#include "OptimalControl.h"

std::string OptimalControl::xmlname = "OptimalControl";

int OptimalControl::Init () {
		std::string par;
		std::string zone;
		zone_number = -10;
		par_index = -10;
		Pars = -1;
		pugi::xml_attribute attr = node.attribute("what");
		if (attr) {
	                par = attr.value();
                        size_t i = par.find_first_of('-');
                        if (i == std::string::npos) {
				ERROR("Can only optimal control a parameters in a specific zone\n");
				return -1;
                        } else {
                                zone = par.substr(i+1);
                                par = par.substr(0,i);
                                const auto zone_iter = solver->setting_zones.find(zone);
                                if (zone_iter != solver->setting_zones.end())
                                        zone_number = zone_iter->second;
                                else {
                                        ERROR("Unknown zone %s (found while setting parameter %s)\n", zone.c_str(), par.c_str());
					return -1;
                                }
                        }
			const Model::ZoneSetting& it = solver->lattice->model->zonesettings.by_name(par);
			if (!it) {
				error("Unknown param %s in OptimalControl\n", par.c_str());
				return -1;
			}
			par_index = it.id;
			output("Selected %s (%d) in zone \"%s\" (%d) for optimal control\n", par.c_str(), par_index, zone.c_str(), zone_number);
		} else {
			ERROR("Parameter \"what\" needed in %s\n",node.name());
			return -1;
		}
		Pars = solver->lattice->zSet.getLen(par_index, zone_number);
		output("Lenght of the control: %d\n", Pars);
		old_iter_type = solver->iter_type;
		solver->iter_type |= ITER_GLOBS;
		attr = node.attribute("lower");
		if (attr) {
			lower = solver->units.alt(attr.value());
		} else {
			notice("lower bound not set in %s - setting to -1\n",node.name());
			lower = -1;
		}
		attr = node.attribute("upper");
		if (attr) {
			upper = solver->units.alt(attr.value());
		} else {
			notice("upper bound not set in %s - setting to 1\n",node.name());
			upper = 1;
		}
		if (solver->mpi_rank == 0) {
                        const auto path = solver->outpath + "_OC_" + par + "_" + zone + ".dat";
			f = fopen(path.c_str(),"w");
			assert( f != NULL );
		} else {
			f = NULL;
		}
		tmptab = new double[Pars];
		return Design::Init();
	};


int OptimalControl::NumberOfParameters () {
		return Pars;
	};


int OptimalControl::Parameters (int type, double * tab) {
		if (solver->mpi_rank != 0) {
			tab = tmptab;
		}
		switch(type) {
		case PAR_GET:
			output("Getting the params from the zone\n");
			solver->lattice->zSet.get(par_index, zone_number, tab);
			if (f != NULL) {
				fprintf(f,"GET");
				for (int i=0;i<Pars;i++) fprintf(f,",%lg",(double) tab[i]);
				fprintf(f,"\n"); fflush(f);
			}
			return 0;
		case PAR_SET:
			output("Setting the params in the zone\n");
			MPI_Bcast(tab, Pars, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			if (f != NULL) {
				fprintf(f,"SET");
				for (int i=0;i<Pars;i++) fprintf(f,",%lg",(double) tab[i]);
				fprintf(f,"\n"); fflush(f);
			}
			solver->lattice->zSet.set(par_index, zone_number, tab);
			return 0;
		case PAR_GRAD:
			output("Getting gradient of a param in zone\n");
			solver->lattice->zSet.get_grad(par_index, zone_number, tmptab);
			MPI_Reduce(tmptab, tab, Pars, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			if (f != NULL) {
				fprintf(f,"GRAD");
				for (int i=0;i<Pars;i++) fprintf(f,",%lg",(double) tab[i]);
				fprintf(f,"\n"); fflush(f);
			}
			return 0;
		case PAR_UPPER:
			for (int i=0;i<Pars;i++) tab[i]=upper;
			return 0;
		case PAR_LOWER:
			for (int i=0;i<Pars;i++) tab[i]=lower;
			return 0;
		default:
			ERROR("Unknown type %d in call to Parameters in %s\n", type, node.name());
			exit(-1);
		}
	};


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< OptimalControl > >;
