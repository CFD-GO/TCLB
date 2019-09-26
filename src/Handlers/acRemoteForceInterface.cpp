#include "acRemoteForceInterface.h"
std::string acRemoteForceInterface::xmlname = "RemoteForceInterface";
#include "../HandlerFactory.h"

int acRemoteForceInterface::Init () {
        Action::Init();
        pugi::xml_attribute attr = node.attribute("integrator");
        if (attr) return ConnectRemoteForceInterface(attr.value());
        ERROR("You must specify RemoteForceInterface integrator name\n");
        return -1;
}


int acRemoteForceInterface::ConnectRemoteForceInterface(std::string integrator_) {
        output("Connecting RFI to %s\n",integrator_.c_str());
        pugi::xml_attribute attr;
        double units[3];
        units[0] = solver->units.alt("1m");
        units[1] = solver->units.alt("1s");
        units[2] = solver->units.alt("1kg");
        
        solver->lattice->RFI.setUnits(units[0],units[1],units[2]);
        solver->lattice->RFI.CanCopeWithUnits(false);

        inter = MPMD[integrator_];
        if (! inter) {
                ERROR("Integrator %s not found in MPMD (that usualy means that you didn't run it)\n",integrator_.c_str());
                return -1;
        }
        integrator = integrator_;
        
        MPI_Barrier(MPMD.local);
        solver->lattice->RFI.Connect(MPMD.work,inter.work);
        
	return 0;
}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acRemoteForceInterface > >;
