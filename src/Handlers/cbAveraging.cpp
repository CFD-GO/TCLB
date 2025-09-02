#include "cbAveraging.h"
std::string cbAveraging::xmlname = "Average";
#include "../HandlerFactory.h"

int cbAveraging::Init () {
	return Callback::Init();
}


int cbAveraging::DoIt () {
	        Callback::DoIt();
	        const auto do_cartesian = [&](const Lattice<CartLattice>* lattice){
		        solver->getCartLattice()->resetAverage();
		        return EXIT_SUCCESS;
	        };
	        const auto do_arbitrary = [&](const Lattice<ArbLattice>* lattice){
		        solver->getArbLattice()->resetAverage();
		        return EXIT_SUCCESS;
	        };
	        return std::visit(OverloadSet{do_cartesian, do_arbitrary}, solver->getLatticeVariant());
        }


int cbAveraging::Finish () {
                return Callback::Finish();
        }


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbAveraging > >;
