#include "Fourier.h"
std::string Fourier::xmlname = "Fourier";
#include "../HandlerFactory.h"

int Fourier::Init () {
		Pars = -1;
		pi = atan(1) * 4;
		pugi::xml_attribute attr;
                pugi::xml_node par = node.first_child();
		if (! par) {
			ERROR("%s needs exacly one child! - none found\n", node.name());
			return -1;
		}
		hand = new Handler(par, solver);
		par = par.next_sibling();
		if (par) {
			ERROR("%s needs exacly one child! - more found\n", node.name());
			return -1;
		}
		if (! (*hand)) return -1;
		if ((*hand).Type() != HANDLER_DESIGN) {
			ERROR("%s needs child of design type!\n", node.name());
			return -1;
		}

		Pars2 = (*hand)->NumberOfParameters();
		attr = node.attribute("modes");
		if (attr) {
			Pars = attr.as_int();
		} else {
			Pars = 10;
			notice("number of modes not set in %s - setting to %d\n",node.name(), Pars);
		}
		if (Pars % 2 != 1) {
			Pars = Pars + 1;
			notice("number of modes in %s not odd - setting to %d\n",node.name(), Pars);
		}
		output("Lenght of time-resolved control: %d\n", Pars2);
		tab2 = new double[Pars2];
		output("Lenght of fourier control: %d\n", Pars);

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
		return Design::Init();
	};


int Fourier::Finish () {
		delete hand;
		return 0;
	}


int Fourier::NumberOfParameters () {
		return Pars;
	};


int Fourier::Parameters (int type, double * tab) {
		double s = 0.5*(upper - lower);
		switch(type) {
		case PAR_GET:
			output("Getting the params and making fourier decomposition\n");
			(*hand)->Parameters(type, tab2);
			for (int i=0; i<Pars;i++) {
				int i0 = (i+1)>>1; int i1 = i & 1;
				tab[i] = 0;
				for (int j=0; j<Pars2; j++) {
					if (i1) {
						tab[i] += sin(i0*pi*2*j/Pars2) * tab2[j];
					} else {
						tab[i] += cos(i0*pi*2*j/Pars2) * tab2[j];
					}
				}
				if (i == 0) {
					tab[i] = tab[i] / Pars2;
				} else {
					tab[i] = 2 * tab[i] / Pars2;
				}
				output("%s[%d] = %lf\n", node.name(), i, tab[i]);
			}
			return 0;
		case PAR_SET:
			output("Setting the params with a fourier series\n");
			for (int j=0; j<Pars2; j++) {
				tab2[j] = 0;
				for (int i=0; i<Pars;i++) {
					int i0 = (i+1)>>1; int i1 = i & 1;
					if (i1) {
						tab2[j] += sin(i0*pi*2*j/Pars2) * tab[i];
					} else {
						tab2[j] += cos(i0*pi*2*j/Pars2) * tab[i];
					}
				}
			}
			(*hand)->Parameters(type, tab2);
			return 0;
		case PAR_GRAD:
			output("Getting gradient and making fourier decomposition\n");
			(*hand)->Parameters(type, tab2);
			for (int i=0; i<Pars;i++) {
				int i0 = (i+1)>>1; int i1 = i & 1;
				tab[i] = 0;
				for (int j=0; j<Pars2; j++) {
					if (i1) {
						tab[i] += sin(i0*pi*2*j/Pars2) * tab2[j];
					} else {
						tab[i] += cos(i0*pi*2*j/Pars2) * tab2[j];
					}
				}
			}
			return 0;
		case PAR_UPPER:
			for (int i=0;i<Pars;i++) {
				int i0 = (i+1)>>1; int i1 = i & 1;
				tab[i]=s/i0;
			}
			tab[0] = upper;
			return 0;
		case PAR_LOWER:
			for (int i=0;i<Pars;i++) {
				int i0 = (i+1)>>1; int i1 = i & 1;
				tab[i]= -s/i0;
			}
			tab[0] = lower;
			return 0;
		default:
			ERROR("Unknown type %d in call to Parameters in %s\n", type, node.name());
			exit(-1);
		}
	};


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< Fourier > >;
