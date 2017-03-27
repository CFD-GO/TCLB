#include "RepeatControl.h"
std::string RepeatControl::xmlname = "RepeatControl";
#include "../HandlerFactory.h"

int RepeatControl::Init () {
		Pars = -1;
		flip_level = 0.0;
		flip = false;
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
		attr = node.attribute("length");
		if (attr) {
			Pars = solver->units.alt(attr.value());
		} else {
			Pars = 1;
		}
		output("Lenght of time-resolved control: %d\n", Pars2);
		tab2 = new double[Pars2];
		output("Lenght of repeated segment control: %d\n", Pars);

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
		attr = node.attribute("flip");
		if (attr) {
			flip = true;
			flip_level = solver->units.alt(attr.value());
		} else {
			flip = false;
		}
		return Design::Init();
	};


int RepeatControl::Finish () {
		delete hand;
		return 0;
	}


int RepeatControl::NumberOfParameters () {
		return Pars;
	};


double RepeatControl::Flip (double v, double l, int j) {
		if (!flip) return v;
		j -= j % Pars;
		j /= Pars;
		if (j % 2 == 0) return v;
		return l-v;
	};


int RepeatControl::Parameters (int type, double * tab) {
		switch(type) {
		case PAR_GET:
			output("Getting the params and making pariodic means\n");
			(*hand)->Parameters(type, tab2);
			for (int i=0; i<Pars;i++) tab[i] = 0;
			for (int j=0; j<Pars2; j++) tab[j % Pars] += Flip(tab2[j],flip_level,j);
			for (int i=0; i<Pars;i++) tab[i] = tab[i] / floor((Pars2 - i - 1.0)/Pars+1);
			return 0;
		case PAR_SET:
			output("Setting the params with a reapet control\n");
			for (int j=0; j<Pars2; j++) tab2[j] = Flip(tab[j % Pars],flip_level,j);
			(*hand)->Parameters(type, tab2);
			return 0;
		case PAR_GRAD:
			output("Getting gradient and making fourier decomposition\n");
			(*hand)->Parameters(type, tab2);
			for (int i=0; i<Pars;i++) tab[i] = 0;
			for (int j=0; j<Pars2; j++) tab[j % Pars] += Flip(tab2[j],0.0,j);
			return 0;
		case PAR_UPPER:
			for (int i=0;i<Pars;i++) tab[i] = upper;
			return 0;
		case PAR_LOWER:
			for (int i=0;i<Pars;i++) tab[i] = lower;
			return 0;
		default:
			ERROR("Unknown type %d in call to Parameters in %s\n", type, node.name());
			exit(-1);
		}
	};


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< RepeatControl > >;
