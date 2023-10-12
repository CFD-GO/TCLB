#include "cbStop.h"

std::string cbStop::xmlname = "Stop";

void cbStop::AddStop(int what_, int stop_type_, double limit_) {
	what.push_back(what_);
	stop_type.push_back(stop_type_);
	limit.push_back(limit_);
	old.push_back(-12341234);
}

int cbStop::Init () {
		Callback::Init();
		double stop;
		pugi::xml_attribute attr;
		for (const Model::Global& it : solver->lattice->model->globals) {
			std::string nm;
			nm = it.name + "Change";
			attr = node.attribute(nm.c_str());
			if (attr) AddStop(it.id, STOP_CHANGE, attr.as_double());
			nm = it.name + "PercentChange";
			attr = node.attribute(nm.c_str());
			if (attr) AddStop(it.id, STOP_PERCENTCHANGE, attr.as_double());
			nm = it.name + "Above";
			attr = node.attribute(nm.c_str());
			if (attr) AddStop(it.id, STOP_ABOVE, attr.as_double());
			nm = it.name + "Below";
			attr = node.attribute(nm.c_str());
			if (attr) AddStop(it.id, STOP_BELOW, attr.as_double());
		}
		if (what.size() < 1) {
			error("No *Change attribute in %s\n", node.name());
			return -1;
		}
		attr = node.attribute("Times");
		if (attr) {
			times = attr.as_int();
			if (times < 1) {
			        error("Minimal number for Times attribute is 1\n");
                                return -1;
                        }
		} else {
		        times = 1;
		}
		score = 0;
		old_iter_type = solver->iter_type;
		solver->iter_type |= ITER_LASTGLOB;
		return 0;
	}


int cbStop::DoIt () {
		Callback::DoIt();
		int ret=0;
                if (solver->mpi_rank == 0) {
                        int any = 0;
                        output("Stop criterium:");
                        for (size_t i=0;i<what.size();i++) {
                                double v = solver->lattice->globals[ what[i] ];
				double value = 0;
				switch (stop_type[i]) {
				case STOP_CHANGE:
					if (fabs(old[i] - v) > limit[i]) any++;
					if (D_MPI_RANK == 0) output("    change:      %4lg < %4lg", fabs(old[i] - v), limit[i]);
					break;
				case STOP_PERCENTCHANGE:
					if (fabs((old[i] - v)/v) > limit[i]) any++;
					if (D_MPI_RANK == 0) output("    change:    %4lg%% < %4lg%%", 100.0 * fabs((old[i] - v)/v), 100.0 * limit[i]);
					break;
				case STOP_ABOVE:
					if (v < limit[i]) any++;
					if (D_MPI_RANK == 0) output("    limit:       %4lg > %4lg", v, limit[i]);
					break;
				case STOP_BELOW:
					if (v > limit[i]) any++;
					if (D_MPI_RANK == 0) output("    limit:       %4lg < %4lg", v, limit[i]);
					break;
				}
                                old[i] = v;
                        }
                        if (!any) {
                                score++;
                        } else {
                                score = 0;
                        }
                        if (D_MPI_RANK == 0) {
                                output("Score: %d\n", score);
                        }
                        if (score >= times) {
                                if (D_MPI_RANK == 0) notice("Stop.\n");
                                ret = ITERATION_STOP;
                                for (size_t i=0;i<what.size();i++) {
                                        old[i] = -12341234;
                                }
                                score=0;
                        }
                }
                MPI_Bcast(&ret, 1, MPI_INT, 0, MPMD.local);
		return ret;
	}


int cbStop::Finish () {
		solver->iter_type = old_iter_type;
		return Callback::Finish();
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbStop > >;
