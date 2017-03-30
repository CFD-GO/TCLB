#include "conExtrude.h"
std::string conExtrude::xmlname = "Extrude";
#include "../HandlerFactory.h"
#include <algorithm>

struct conExtrudeCompare {
        double ** coords;
        int direction;
        bool positive;
        conExtrudeCompare(double ** coords_, int direction_, bool positive_): coords(coords_), direction(direction_), positive(positive_) {}
        bool operator()(int a, int b) const {
                for (int i=0;i<4;i++) if (i != direction) {
                        if (coords[i][a] < coords[i][b]) return true;
                        if (coords[i][a] > coords[i][b]) return false;
                }
                if (positive) return coords[direction][a] < coords[direction][b];
                return coords[direction][a] > coords[direction][b];
        }
};

bool conExtrude::next(size_t k) {
        if (! ((k+1) < Pars2)) return true;
        size_t a=idx[k], b=idx[k+1];
        for (int i=0;i<4;i++) if (i != direction) {
                if (coords[i][a] != coords[i][b]) return true;
        }
        return false;
}

int conExtrude::Init () {
		Pars = -1;
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
		direction=-1;
		attr = node.attribute("direction");
		if (attr) {
		        std::string dire = attr.value();
		        if (dire == "x") direction = 0;
		        if (dire == "y") direction = 1;
		        if (dire == "z") direction = 2;
		        if (dire == "t") direction = 3;
		        if (direction < 0) {
        			ERROR("%s needs proper direction - \"%s\" given!\n", node.name(), attr.value());
        			return -1;
                        }
		} else {
			ERROR("%s needs direction!\n", node.name());
			return -1;
		}
		theta = 1.0;
		attr = node.attribute("theta");
		if (attr) {
		        theta = solver->units.alt(attr.value());
		        output("Setting theta in %s to %lg\n",node.name(),theta);
                }
		margin = 1.0;
		attr = node.attribute("margin");
		if (attr) {
		        margin = solver->units.alt(attr.value());
		        output("Setting margin in %s to %lg\n",node.name(),margin);
                }
	        tab2 = new double[Pars2];
		for (int i=0;i<4;i++) {
		        coords[i] = new double[Pars2];
		        (*hand)->Parameters(PAR_X+i, coords[i]);
                }
                idx.resize(Pars2);
                for (size_t i=0; i<Pars2; i++) idx[i] = i;
                std::sort(idx.begin(),idx.end(),conExtrudeCompare(coords,direction,theta>0));
                Pars = 0;
                for (size_t i=0; i<Pars2; i++) if (next(i)) Pars++;
                output("%s with %d parameters\n", node.name(), Pars);
                Par = new double[Pars];
		return Design::Init();
	};


int conExtrude::Finish () {
	delete hand;
	return 0;
}


int conExtrude::NumberOfParameters () {
	return Pars;
};

double conExtrude::Fun(double x, double v) {
        x = (x-v)/theta;
        x = exp(x);
        x = x/(x+1);
        return x;
}

double conExtrude::FunD(double x, double v) {
        x = (x-v)/theta;
        x = exp(x);
        x = x/(x+1)/(x+1);
        return -x/theta;
}

int conExtrude::Parameters (int type, double * tab) {
                size_t k=0;
		switch(type) {
		case PAR_SET:
		        for (size_t i=0; i<Pars; i++) Par[i] = tab[i]; // We have to save the parameters, are our trasformation is non-linear
		        for (size_t i=0; i<Pars2; i++) {
		                size_t j = idx[i];
		                tab2[j] = Fun(coords[direction][j],tab[k]);
		                if (next(i)) k++;
                        }
                        assert(k == Pars);
                        (*hand)->Parameters(type,tab2);
			return 0;
		case PAR_GRAD:
                        (*hand)->Parameters(type,tab2);
                        for (size_t i=0; i<Pars; i++) tab[i] = 0;
		        for (size_t i=0; i<Pars2; i++) {
		                size_t j = idx[i];
		                tab[k] += FunD(coords[direction][j],Par[k]) * tab2[j];
		                if (next(i)) k++;
                        }
                        assert(k == Pars);
			return 0;
		case PAR_GET:
		        (*hand)->Parameters(type,tab2); // NOTE: no break - continues below
		case PAR_UPPER:
		case PAR_LOWER:
		        {
		                bool st=true;
        		        for (size_t i=0; i<Pars2; i++) {
        		                size_t j = idx[i];
        		                switch(type) {
        		                        case PAR_UPPER: if (tab[k] < coords[direction][j]) st=true; break;
        		                        case PAR_LOWER: if (tab[k] > coords[direction][j]) st=true; break;
        		                        case PAR_GET:   if (tab2[j] < 0.5) st=true; break;
                                        }
        		                if (st) {
        		                        tab[k] = coords[direction][j];
        		                        st = false;
                                        }
        		                if (next(i)) { k++; st = true; }
                                }
                                double offset = abs(margin*theta);
       		                switch(type) {
       		                        case PAR_UPPER: for (size_t i=0; i<Pars; i++) tab[i] += offset; break;
       		                        case PAR_LOWER: for (size_t i=0; i<Pars; i++) tab[i] -= offset; break;
       		                        case PAR_GET:   for (size_t i=0; i<Pars; i++) tab[i] -= offset; break;
                                }
                        }
			return 0;
		default:
			ERROR("Unknown type %d in call to Parameters in %s\n", type, node.name());
			exit(-1);
		}
	};


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< conExtrude > >;
