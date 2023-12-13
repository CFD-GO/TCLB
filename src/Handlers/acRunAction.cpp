#include "acRunAction.h"

std::string acRunAction::xmlname = "RunAction";

int acRunAction::Init() {
    GenericAction::Init();
    pugi::xml_attribute attr = node.attribute("name");
    if (!attr) {
        ERROR("Have to specify the name of the Action in RunAction");
        return EXIT_FAILURE;
    }
    std::string action_name = attr.value();
    const Model::Action& act = solver->lattice->model->actions.by_name(action_name);
    if (!act) {
        ERROR("R: Unknown Action");
        return -1;
    }
    const int action = act.id;
    if (GenericAction::ExecuteInternal()) return EXIT_FAILURE;
    bool stop = false;
    do {
        int next_it = Next(solver->iter);
        for (auto& hand : solver->hands) {
            const int it = hand.Next(solver->iter);
            if (it > 0 && it < next_it) next_it = it;
        }
        solver->steps = next_it;
        MPI_Bcast(&solver->steps, 1, MPI_INT, 0, MPMD.local);
        solver->iter += solver->steps;
        solver->lattice->IterateAction(action, solver->steps, solver->iter_type);
        CudaDeviceSynchronize();
        MPI_Barrier(MPMD.local);
        for (auto& hand : solver->hands) {
            if (hand.Now(solver->iter)) {
                int ret = hand.DoIt();
                switch (ret) {
                    case ITERATION_STOP:
                        stop = true;
                    case 0:
                        break;
                    default:
                        return EXIT_FAILURE;
                }
            }
        }
        if (stop) break;
    } while (!Now(solver->iter));
    CudaDeviceSynchronize();
    MPI_Barrier(MPMD.local);
    GenericAction::Unstack();
    return EXIT_SUCCESS;
}

// Register the handler (based on xmlname) in the Handler Factory
template class HandlerFactory::Register<GenericAsk<acRunAction> >;
