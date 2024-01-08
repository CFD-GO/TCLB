#include "cbFailcheck.h"

std::string cbFailcheck::xmlname = "Failcheck";

int cbFailcheck::Init() {
    Callback::Init();
    currentlyactive = false;
    const auto init_cart = [&](const Lattice<CartLattice>* lattice) {
        reg.dx = lattice->getLocalRegion().dx;
        reg.dy = lattice->getLocalRegion().dy;
        reg.dz = lattice->getLocalRegion().dz;
        const auto set_if_present = [&](const char* name, auto& value) {
            const auto attribute = node.attribute(name);
            if (attribute) value = solver->units.alt(attribute.value());
        };
        set_if_present("dx", reg.dx);
        set_if_present("dy", reg.dy);
        set_if_present("dz", reg.dz);
        set_if_present("nx", reg.nx);
        set_if_present("ny", reg.ny);
        set_if_present("nz", reg.nz);
        return EXIT_SUCCESS;
    };
    const auto init_arb = [&](const Lattice<ArbLattice>*) { return EXIT_SUCCESS; };
    return std::visit(OverloadSet{init_cart, init_arb}, solver->getLatticeVariant());
}

int cbFailcheck::DoIt() {
    Callback::DoIt();
    if (currentlyactive) return EXIT_SUCCESS;
    currentlyactive = true;

    name_set components;
    const auto comp = node.attribute("what");
    components.add_from_string(comp ? comp.value() : "all", ',');

    const auto check_for_nans = [&](const Model::Quantity& quantity) -> bool {
        if (!components.in(quantity.name) || quantity.isAdjoint) return false;

        const auto get_quantity_vec = [&](const Model::Quantity& q) -> std::vector<real_t> {
            const auto get_from_cart = [&](Lattice<CartLattice>* lattice) {
                return lattice->getQuantity(q, reg, 1);
            };
            const auto get_from_arb = [&](Lattice<ArbLattice>* lattice) {
                return lattice->getQuantity(q, 1);
            };
            return std::visit(OverloadSet{get_from_cart, get_from_arb}, solver->getLatticeVariant());
        };

        const auto values = get_quantity_vec(quantity);
        int has_nans = std::any_of(values.begin(), values.end(), [](auto v) { return std::isnan(v); });
        MPI_Allreduce(MPI_IN_PLACE, &has_nans, 1, MPI_INT, MPI_LOR, MPMD.local);
        if (has_nans) notice("Discovered NaN values in %s", quantity.name.c_str());
        return has_nans;
    };

    // Note: std::any_of would break early, we want to print all quantities which have NaN values, hence std::transform_reduce
    const auto& quants = solver->lattice->model->quantities;
    if (std::transform_reduce(quants.begin(), quants.end(), false, std::logical_or{}, check_for_nans)) {
        notice("NaN value discovered. Executing final actions from the Failcheck element before full stop...\n");
        for (pugi::xml_node par = node.first_child(); par; par = par.next_sibling()) {
            Handler hand(par, solver);
            if (hand) hand.DoIt();
        }
        notice("Stopping due to NaN value\n");
        return ITERATION_STOP;
    }
    return EXIT_SUCCESS;
}

int cbFailcheck::Finish() {
    return Callback::Finish();
}

// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register<GenericAsk<cbFailcheck> >;
