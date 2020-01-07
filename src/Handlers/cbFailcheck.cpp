#include "cbFailcheck.h"
std::string cbFailcheck::xmlname = "Failcheck";

int cbFailcheck::Init () {
		Callback::Init();
		currentlyactive = false;
		reg.dx = solver->region.dx;
		reg.dy = solver->region.dy;
		reg.dz = solver->region.dz;
	


        pugi::xml_attribute attr = node.attribute("dx");
        if (attr) {
            reg.dx = solver->units.alt(attr.value());
        }
        attr = node.attribute("dy");
        if (attr) {
            reg.dy = solver->units.alt(attr.value());
        }
        attr = node.attribute("dz");
        if (attr) {
            reg.dz = solver->units.alt(attr.value());
        }


        attr = node.attribute("nx");
        if (attr) {
            reg.nx = solver->units.alt(attr.value());
        }
        attr = node.attribute("ny");
        if (attr) {
            reg.ny = solver->units.alt(attr.value());
        }
        attr = node.attribute("nz");
        if (attr) {
            reg.nz = solver->units.alt(attr.value());
        }

		return 0;
	}


int cbFailcheck::DoIt () {
		Callback::DoIt();
	if (currentlyactive) return 0;
	currentlyactive = true;
       	int ret = 0;
		int fin;
		fin = false;

        pugi::xml_attribute comp = node.attribute("what");
 
        name_set components;
        if(comp){
            components.add_from_string(comp.value(),',');
        } else {
            components.add_from_string("all",',');
        }

	for (ModelBase::Quantities::const_iterator it = solver->lattice->model->quantities.begin(); it != solver->lattice->model->quantities.end(); it++) {
	#ifndef ADJOINT
		if (it->isAdjoint) continue;
	#endif        
            if (components.in(it->name)) {
			int comp = 1;
			if (it->isVector) comp = 3;
                    real_t* tmp = new real_t[reg.size()*comp];
		    solver->lattice->GetQuantity(it->id, reg, tmp, 1);
                    bool cond = false;
                    for (int k = 0; k < reg.size()*comp; k++){  
	       		    cond = cond || (std::isnan(tmp[k]));
                    }
		    delete[] tmp;
			MPI_Allreduce(&cond,&fin,1,MPI_INT,MPI_LOR,MPMD.local);

                    if(fin ){
			notice("Checking %s discovered NaN", it->name);
			break;
			}
		}

        }
	    if (fin) {
			notice("NaN value discovered. Executing final actions from the Failcheck element before full stop...\n");
                for (pugi::xml_node par = node.first_child(); par; par = par.next_sibling()) {
                    Handler hand(par, solver);
                    if (hand) hand.DoIt();
                }
                notice("Stopping due to Nan value\n");
                ret = ITERATION_STOP;
            }
            return ret;
    }	


int cbFailcheck::Finish () {
		return Callback::Finish();
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbFailcheck > >;
