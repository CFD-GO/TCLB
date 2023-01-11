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

        bool stats = false;
        std::string stats_prefix = solver->info.outpath;
        stats_prefix = stats_prefix + "_RFI";
        int stats_iter = 200;
        
        attr = node.attribute("stats");
        if (attr) stats = attr.as_bool();
        attr = node.attribute("stats_iter");
        if (attr) {
          stats_iter = solver->units.alt(attr.value());
          stats = true;
        }
        attr = node.attribute("stats_prefix");
        if (attr) {
          stats_prefix = attr.value();
          stats = true;
        }
        
        if (stats) {
          output("Asking for stats on RFI ( %s every %d it)\n", stats_prefix.c_str(), stats_iter);
          solver->lattice->RFI.enableStats(stats_prefix.c_str(), stats_iter);
        }

        inter = MPMD[integrator_];
        if (! inter) {
                ERROR("Integrator %s not found in MPMD (that usualy means that you didn't run it)\n",integrator_.c_str());
                return -1;
        }
        integrator = integrator_;

        bool use_box = true;
        attr = node.attribute("use_box");
        if (attr) use_box = attr.as_bool();
        
        if (use_box) {
          lbRegion reg = solver->lattice->region;
          double px = solver->lattice->px;
          double py = solver->lattice->py;
          double pz = solver->lattice->pz;
          solver->lattice->RFI.DeclareSimpleBox(
            px + reg.dx,
            px + reg.dx + reg.nx,
            py + reg.dy,
            py + reg.dy + reg.ny,
            pz + reg.dz,
            pz + reg.dz + reg.nz);
        }
        MPI_Barrier(MPMD.local);
        solver->lattice->RFI.Connect(MPMD.work,inter.work);
        
	return 0;
}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acRemoteForceInterface > >;
