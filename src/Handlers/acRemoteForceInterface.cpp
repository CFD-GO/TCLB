#include "acRemoteForceInterface.h"
std::string acRemoteForceInterface::xmlname = "RemoteForceInterface";
#include "../HandlerFactory.h"

#include <sstream>

int acRemoteForceInterface::Init () {
        Action::Init();
        pugi::xml_attribute attr = node.attribute("integrator");
        if (attr) return ConnectRemoteForceInterface(attr.value());
        ERROR("You must specify RemoteForceInterface integrator name\n");
        return -1;
}


int acRemoteForceInterface::ConnectRemoteForceInterface(std::string integrator_) {
        output("Connecting RFI to %s\n",integrator_.c_str());
        double units[3];
        units[0] = solver->units.alt("1m");
        units[1] = solver->units.alt("1s");
        units[2] = solver->units.alt("1kg");
        
        solver->lattice->RFI.setUnits(units[0],units[1],units[2]);
        solver->lattice->RFI.CanCopeWithUnits(false);

        solver->lattice->RFI.setVar("output", solver->info.outpath);
        
        std::string element_content;
        int node_children = 0;
        for (pugi::xml_node par = node.first_child(); par; par = par.next_sibling()) {
          node_children ++;
          if (node_children > 1) {
              ERROR("Only a single element/CDATA allowed inside of a RemoteForceInterface xml element\n");
              return -1;
          }
		      if ((par.type() == pugi::node_pcdata) || (par.type() == pugi::node_cdata)) {
            element_content = par.value();
		      } else {
            std::stringstream ss;
            par.print(ss);
            element_content = ss.str();
          }
	      }
        if (node_children > 0) solver->lattice->RFI.setVar("content", element_content);

        bool stats = false;
        std::string stats_prefix = solver->info.outpath;
        stats_prefix = stats_prefix + "_RFI";
        int stats_iter = 200;
        bool use_box = true;


        for (pugi::xml_attribute attr = node.first_attribute(); attr; attr = attr.next_attribute()) {
          std::string attr_name = attr.name();
          if (attr_name == "integrator") {
            // ignore
          } else if (attr_name == "stats") {
            stats = attr.as_bool();
          } else if (attr_name == "stats_iter") {
            stats_iter = solver->units.alt(attr.value());
            stats = true;
          } else if (attr_name == "stats_prefix") {
            stats_prefix = attr.value();
            stats = true;
          } else if (attr_name == "use_box") {
            use_box = attr.as_bool();
          } else if (attr_name == "omega") {
            solver->lattice->RFI_omega = attr.as_bool();
          } else if (attr_name == "torque") {
            solver->lattice->RFI_torque = attr.as_bool();
          } else {
            double val = solver->units.alt(attr.value());
            char str[STRING_LEN];
            sprintf(str, "%.15lg", val);
            solver->lattice->RFI.setVar(attr.name(), str);
          }
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

        if (use_box) {
          lbRegion reg = solver->lattice->region;
          double px = solver->lattice->px;
          double py = solver->lattice->py;
          double pz = solver->lattice->pz;
          solver->lattice->RFI.DeclareSimpleBox(
            px + reg.dx - PART_MAR_BOX,
            px + reg.dx + reg.nx + PART_MAR_BOX,
            py + reg.dy - PART_MAR_BOX,
            py + reg.dy + reg.ny + PART_MAR_BOX,
            pz + reg.dz - PART_MAR_BOX,
            pz + reg.dz + reg.nz + PART_MAR_BOX);
        }

        MPI_Barrier(MPMD.local);
        solver->lattice->RFI.Connect(MPMD.work,inter.work);
        
        return 0;
}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acRemoteForceInterface > >;
