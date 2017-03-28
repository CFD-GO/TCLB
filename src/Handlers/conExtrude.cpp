#include "conExtrude.h"
std::string conExtrude::xmlname = "Extrude";
#include "../HandlerFactory.h"
#include <algorithm>

struct conExtrudeCompare {
        double ** coords;
        int direction;
        conExtrudeCompare(double ** coords_, int direction_): coords(coords_), direction(direction_) {}
        bool operator()(int a, int b) const {
                for (int i=0;i<4;i++) if (i != direction) {
                        if (coords[i][a] < coords[i][b]) return true;
                        if (coords[i][a] > coords[i][b]) return false;
                }
                return coords[direction][a] < coords[direction][b];
        }
};

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
		direction=0;
		attr = node.attribute("direction");
		if (attr) {
		        std::string dire = attr.value();
		        if (dire == "x") direction = PAR_X;
		        if (dire == "y") direction = PAR_Y;
		        if (dire == "z") direction = PAR_Z;
		        if (dire == "t") direction = PAR_T;
		        if (direction == 0) {
        			ERROR("%s needs proper direction - \"%s\" given!\n", node.name(), attr.value());
        			return -1;
                        }
		} else {
			ERROR("%s needs direction!\n", node.name());
			return -1;
		}
		Pars = 0;
	        tab2 = new double[Pars2];
		for (int i=0;i<4;i++) {
		        coords[i] = new double[Pars2];
		        (*hand)->Parameters(PAR_X+i, coords[i]);
                }
                idx.resize(Pars2);
                for (size_t i=0; i<Pars2; i++) idx[i] = i;
                std::sort(idx.begin(),idx.end(),conExtrudeCompare(coords,direction-PAR_X));
                for (size_t i=0; i<Pars2; i++) {
                        size_t j=idx[i];
                        printf("extr: %lf %lf %lf %lf\n", coords[0][j], coords[1][j], coords[2][j], coords[3][j]);
                }
		return Design::Init();
	};


int conExtrude::Finish () {
	delete hand;
	return 0;
}


int conExtrude::NumberOfParameters () {
	return Pars;
};


int conExtrude::Parameters (int type, double * tab) {
		switch(type) {
		case PAR_GET:
			return 0;
		case PAR_SET:
			return 0;
		case PAR_GRAD:
			return 0;
		case PAR_UPPER:
			return 0;
		case PAR_LOWER:
			return 0;
		default:
			ERROR("Unknown type %d in call to Parameters in %s\n", type, node.name());
			exit(-1);
		}
	};


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< conExtrude > >;
