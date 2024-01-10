/*  Main program file                                          */
/*     Here we have all the initialization and the main loop   */
/*-------------------------------------------------------------*/

#include "Consts.h"


#include <assert.h>
#include <mpi.h>

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "GetThreads.h"
#include "Global.h"
#include "Solver.h"
#include "glue.hpp"
#include "mpitools.hpp"
#include "toArb.h"
#include "unit.h"
#include "utils.h"
#include "xpath_modification.h"

CudaEvent_t start, stop;  // CUDA events to measure time

class MainCallback {
   public:
    MainCallback(Solver* solver_) : solver(solver_) {}
    int operator()(int seg, int tot) {
        int begin = tot == 0;
        int end = tot == seg;

        if (tot < last_tot) last_tot = 0;
        iter += tot - last_tot;
        last_tot = tot;
        if ((iter < steps) & (!end)) return -1;
        if (iter == 0) return -2;
        float elapsedTime = 0;  // Elapsed times
        CudaEventRecord(stop, 0);
        CudaEventSynchronize(stop);
        CudaEventElapsedTime(&elapsedTime, start, stop);

        if (D_MPI_RANK == 0) {
            float eTime;
            int all_iter;
            cum_time += elapsedTime;
            if (end) {
                eTime = cum_time;
                all_iter = tot;
            } else {
                eTime = elapsedTime;
                all_iter = iter;
            }
            int ups = (float)(1000. * all_iter) / eTime;  // Steps made per second
            const double lbups = static_cast<double>(solver->getGlobalLatticeSize() * iter / elapsedTime);
            int desired_steps = ups / desired_fps;  // Desired steps per frame (so that on next frame fps = desired_fps)
            char per[1000];
            char buf[1000];
            char left[1000];
            // int left_s = (cum_time * (seg - tot)) / ((tot+1) * 1000);
            int left_s = cum_time / 1000;
            if (left_s < 60) {
                sprintf(left, "%2ds", left_s);
            } else {
                int left_m = left_s / 60;
                left_s = left_s - left_m * 60;
                if (left_m < 60) {
                    sprintf(left, "%2dm %2ds", left_m, left_s);
                } else {
                    int left_h = left_m / 60;
                    left_m = left_m - left_h * 60;
                    sprintf(left, "%dh %2dm", left_h, left_m);
                }
            }
            sprintf(buf, "%8.1f MLBUps   %7.2f GB/s", lbups / 1000., (lbups * (2. * solver->lattice->model->fields.size() * sizeof(real_t) + sizeof(flag_t))) / 1e6);
            int per_len = 20;
            {
                int i = 0;
                per[i] = '[';
                i++;
                for (; i <= per_len; i++)
                    if (i * seg <= tot * per_len) per[i] = '=';
                    else
                        per[i] = ' ';
                per[i] = ']';
                i++;
                per[i] = 0;
            }
            if (end) {
                output("%s %s\n", buf, per);
                cum_time = 0;
            } else {
                if (D_TERMINAL) {
                    printf("[  ] %s %s %s\r", buf, per, left);
                } else {
                    printf("[  ] %s %s %s\n", buf, per, left);
                }
            }
            fflush(stdout);
            steps = desired_steps;
            if (steps < 1) steps = 1;
            if (steps % 2 == 1) steps++;
        }
        MPI_Bcast(&steps, 1, MPI_INT, 0, MPMD.local);
        solver->EventLoop();
        CudaEventRecord(start, 0);
        CudaEventSynchronize(start);
        iter = 0;
        return steps;
    }

   private:
    Solver* solver;
    int iter = 0;
    int last_tot = 0;
    double cum_time = 0;
    int steps = 1;
};

void handleDeprecatedParam(pugi::xml_node config) {
    pugi::xpath_node_set found = config.select_nodes("//Params");
    if (found.size() > 0) {
        WARNING("%ld depreciated Params elements found. Changing them to Param:\n", found.size());
        config.append_attribute("permissive").set_value("true");
        for (pugi::xpath_node_set::const_iterator it = found.begin(); it != found.end(); ++it) {
            pugi::xml_node node = it->node();
            pugi::xml_node after = node;
            std::string gauge = "";
            pugi::xml_attribute gauge_attr = node.attribute("gauge");
            if (gauge_attr) gauge = gauge_attr.value();
            for (pugi::xml_attribute attr = node.first_attribute(); attr; attr = attr.next_attribute())
                if (strcmp(attr.name(), "gauge") != 0) {
                    std::string par, zone;
                    par = attr.name();
                    size_t i = par.find_first_of('-');
                    if (i == std::string::npos) {
                        zone = "";
                    } else {
                        zone = par.substr(i + 1);
                        par = par.substr(0, i);
                    }
                    pugi::xml_node param = node.parent().insert_child_after("Param", after);
                    after = param;
                    param.append_attribute("name").set_value(par.c_str());
                    param.append_attribute("value").set_value(attr.value());
                    if (zone != "") param.append_attribute("zone").set_value(zone.c_str());
                    if (gauge != "") param.append_attribute("gauge").set_value(gauge.c_str());
                }
            node.parent().remove_child(node);
        }
    }
}

void deleteComments(pugi::xml_node config) {
    pugi::xpath_node_set found = config.select_nodes("//comment()");
    if (found.size() > 0) {
        output("Discarding %ld comments\n", found.size());
        for (const auto& xnode : found) {
            pugi::xml_node node = xnode.node();
            if (node) {
                node.parent().remove_child(node);
            } else {
                ERROR("Comment is not a node (this should not happen)\n");
            }
        }
    }
}

int selectDevice(pugi::xml_node config) {
#ifndef CROSS_CPU
    int count, dev;
    CudaGetDeviceCount(&count);
    {
        MPI_Comm comm = MPMD.local;
        std::string nodename = mpitools::MPI_Nodename();
        MPI_Comm nodecomm = mpitools::MPI_Split(nodename, comm);
        dev = mpitools::MPI_Rank(nodecomm);
        MPI_Comm_free(&nodecomm);
    }
    if (dev >= count) {
        const bool oversubscribe = config.attribute("oversubscribe_gpu").as_bool();
        if (!oversubscribe) {
            ERROR("Oversubscribing GPUs. This is not a good idea, but if you want to do it, add oversubscribe_gpu=\"true\" to the config file");
            return EXIT_FAILURE;
        } else {
            WARNING("Oversubscribing GPUs.");
        }
        dev = dev % count;
    }
    output_all("Selecting device %d/%d\n", dev, count);
    CudaSetDevice(dev);
    debug2("Initializing device\n");
    CudaFree(0);
    InitDim();
#else
    output_all("Running on CPU\n");
    CudaSetDevice(0);
#endif
    return EXIT_SUCCESS;
}

std::array<int, 3> readLatticeDims(const UnitEnv& units, pugi::xml_node geom) {
    return {myround(units.alt(geom.attribute("nx").value(), 1)), myround(units.alt(geom.attribute("ny").value(), 1)), myround(units.alt(geom.attribute("nz").value(), 1))};
}

class SolverBuilder {
    std::unique_ptr<Solver> solver = std::make_unique<Solver>();

   public:
    std::unique_ptr<Solver> build() {
        solver->setOutput("");
        return std::exchange(solver, {});
    }

    int setComm(MPI_Comm comm) {
        solver->mpi_comm = comm;
        MPI_Comm_rank(comm, &solver->mpi_rank);
        MPI_Comm_size(comm, &solver->mpi_size);
        return EXIT_SUCCESS;
    }
    pugi::xml_document* setConfFile(std::string_view conf_path) {
        solver->conffile_path.assign(conf_path);
        solver->setOutput("");
        const auto result = solver->configfile.load_file(conf_path.data(), pugi::parse_default | pugi::parse_comments);
        if (!result) error("Error while parsing %s: %s\n", conf_path.data(), result.description());
        return result ? &solver->configfile : nullptr;
    }
    int setUnits(pugi::xml_node config) {
        if (readUnits(config)) {
            ERROR("Wrong Units\n");
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
    void setZones(pugi::xml_node node) {  // This works for **both** the Geometry and ArbitraryLattice nodes
        for (auto n = node.first_child(); n; n = n.next_sibling()) {
            if (std::string(n.name()) != "Zone") {
                const auto attr = n.attribute("name");
                if (attr) insertZone(attr.value());
            }
        }
    }
    void setSnaps() {
        int num_snaps = 2;
        // Finding the adjoint element
        pugi::xml_node adj = solver->configfile.find_node([](pugi::xml_node node) { return std::string_view(node.name()) == "Adjoint" ? (std::string_view(node.attribute("type").value()) != "steady") : false; });
        if (adj) {
            const auto attr = adj.attribute("NumberOfSnaps");
            num_snaps = attr ? std::max(attr.as_int(), 2) : 10;
            solver->num_snaps = num_snaps;
            NOTICE("Will be running nonstationary adjoint at %d Snaps\n", D_MPI_RANK, num_snaps);
        }
    }
    int setGeometry(pugi::xml_node geom) {
        // Reading the size of mesh
        const auto [nx, ny, nz] = readLatticeDims(solver->units, geom);
        NOTICE("Mesh size in config file: %dx%dx%d\n", nx, ny, nz);

        // Initializing the lattice of a specific size
        return solver->initCartLattice(nx, ny, nz);
    }
    int setArbitrary(pugi::xml_node arb_node) { return solver->initArbLattice(arb_node); }
    void setCallback() { solver->lattice->setCallback(MainCallback(solver.get())); }

   private:
    int readUnits(pugi::xml_node config) {
        pugi::xml_node set = config.child("Units");
        if (!set) {
            warning("No \"Units\" element in config file\n");
            return 0;
        }
        int i = 1;
        for (pugi::xml_node node = set.child("Param"); node; node = node.next_sibling("Param")) {
            std::string par, value, gauge;
            pugi::xml_attribute attr;
            attr = node.attribute("value");
            if (attr) {
                value = attr.value();
            } else {
                ERROR("Value not provided in Param in Units\n");
                return -1;
            }
            attr = node.attribute("gauge");
            if (attr) {
                gauge = attr.value();
            } else {
                ERROR("Gauge not provided in Param in Units\n");
                return -1;
            }
            attr = node.attribute("name");
            if (attr) par = attr.value();
            else
                par = (Glue() << "unnamed" << i).str();
            debug2("Units: %s = %s = %s\n", par.c_str(), value.c_str(), gauge.c_str());
            solver->setUnit(par, value, gauge);
            i++;
        }
        solver->Gauge();
        return 0;
    }
    void insertZone(std::string_view name) {
        const auto [iter, was_inserted] = solver->setting_zones.emplace(name, static_cast<int>(solver->setting_zones.size()));
        if (was_inserted) {
            debug1("Setting new zone: %s -> %d\n", iter->first.c_str(), iter->second);
            assert(iter->second < ZONE_MAX);
        }
    }
};

// Invoke toArb based on the provided geometry and units
int convertToArbitrary(const std::unique_ptr<Solver>& solver, pugi::xml_node geo_xml, const Model& model) {
    if (solver->mpi_size != 1) {
        ERROR("toArb must be run with a single MPI rank");
        return EXIT_FAILURE;
    }
    const auto [nx, ny, nz] = readLatticeDims(solver->units, geo_xml);
    const auto region = lbRegion(0, 0, 0, nx, ny, nz);
    Geometry geometry(region, region, solver->units, &model);
    if (geometry.load(geo_xml, solver->setting_zones)) {
        ERROR("Error while loading geometry for toArb");
        return EXIT_FAILURE;
    }
    if (toArbitrary(*solver, geometry, model)) {
        ERROR("Error exporting to .cxn file");
        return EXIT_FAILURE;
    } else {
        NOTICE("toArb completed successfully");
        return EXIT_SUCCESS;
    }
}

using namespace std::string_view_literals;

// Main program function
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPMD.Init(MPI_COMM_WORLD, "TCLB");
    MPMD.Identify();

    DEBUG_SETRANK(MPMD.local_rank);
    DEBUG_M;
    InitPrint(DEBUG_LEVEL, 6, 8);
    MPI_Barrier(MPMD.local);

    start_walltime();
    if (MPMD.local_rank == 0) {
        NOTICE("-------------------------------------------------------------------------\n");
        NOTICE("-  CLB version: %25s                               -\n", VERSION);
        NOTICE("-        Model: %25s                               -\n", MODEL);
        NOTICE("-------------------------------------------------------------------------\n");
    }
    MPI_Barrier(MPMD.local);
    DEBUG_M;

    DEBUG0(debug0("0 level debug"); debug1("1 level debug"); debug2("2 level debug"); output("normal output"); notice("notice"); NOTICE("important notice"); warning("warning"); WARNING("important warning"); error("error"); ERROR("fatal error");)

    // Read arguments, at least 1 is required
    if (argc < 2) {
        error("Not enough parameters");
        notice("Usage: program configfile [device number]\n");
        return EXIT_FAILURE;
    }

    debug0("sizeof(size_t) = %ld\n", sizeof(size_t));
    debug0("sizeof(real_t) = %ld\n", sizeof(real_t));
    debug0("sizeof(vector_t) = %ld\n", sizeof(vector_t));
    debug0("sizeof(flag_t) = %ld\n", sizeof(flag_t));

    MPI_Barrier(MPMD.local);

    // The first arg is the path to the config file
    const auto conf_path = std::string_view(argv[1]);

    // The builder object used to initialize the solver piecemeal
    SolverBuilder solver_builder;

    // Pass the local communicator
    solver_builder.setComm(MPMD.local);

    // The config xml doc resides in the solver. Initialize it from the config file and retrieve a pointer
    const auto conf_doc_ptr = solver_builder.setConfFile(conf_path);
    if (!conf_doc_ptr) return EXIT_FAILURE;
    auto config = conf_doc_ptr->child("CLBConfig");
    if (!config) {
        const auto err_msg = formatAsString("No CLBconfig element in %s", conf_path.data());
        return EXIT_FAILURE;
    }
    handleDeprecatedParam(config);

    // xpath_modify with special handling for the return value -444
    if (argc > 2) {
        const int status = xpath_modify(*conf_doc_ptr, config, argc - 2, std::next(argv, 2));
        switch (status) {
            case 0:
                break;
            case -444:
                return solver_builder.setUnits(config);
            default:
                return status;
        }
    }

    deleteComments(config);
    if (selectDevice(config)) return EXIT_FAILURE;

    MPI_Barrier(MPMD.local);
    DEBUG_M;

    // Units
    if (solver_builder.setUnits(config)) return EXIT_FAILURE;

    // Geometry/ArbitraryLattice XML nodes responsible for most of initialization
    auto geo_xml = config.find_node([](auto node) { return "Geometry"sv == node.name(); });
    auto arb_xml = config.find_node([](auto node) { return "ArbitraryLattice"sv == node.name(); });

    // Generate arbitrary lattice files
    // This has to be called before lattice initialization, since we want to avoid storing the entire Cartesian lattice
    if (config.attribute("toArb")) {
        if (!geo_xml) {
            ERROR("Conversion to arbitrary lattice requested without providing the geometry");
            return EXIT_FAILURE;
        }
        solver_builder.setZones(geo_xml);
        return convertToArbitrary(solver_builder.build(), geo_xml, Model_m());  // TODO: model initialization where?
    }

    // Snaps
    solver_builder.setSnaps();

    // Initialize lattice according to the specified type (Cartesian or arbitrary)
    if (geo_xml && arb_xml) {
        ERROR("\"ArbitraryLattice\" and \"Geometry\" are mutually exclusive");
        return EXIT_FAILURE;
    } else if (!geo_xml && !arb_xml) {
        ERROR("Either \"ArbitraryLattice\" or \"Geometry\" must be specified");
        return EXIT_FAILURE;
    } else if (geo_xml) {
        NOTICE("Using Cartesian lattice");
        solver_builder.setZones(geo_xml);
        if (solver_builder.setGeometry(geo_xml)) return EXIT_FAILURE;
    } else if (arb_xml) {
        NOTICE("Using arbitrary lattice");
        solver_builder.setZones(arb_xml);
        if (solver_builder.setArbitrary(arb_xml)) return EXIT_FAILURE;
    }  // else __builtin_unreachable();

    // Setting main callback
    solver_builder.setCallback();

    {  // <- Respect this scope - the solver cannot outlive cuda finalization

        // The solver has been built!
        const auto solver = solver_builder.build();

        // Initializing CUDA events
        CudaEventCreate(&start);
        CudaEventCreate(&stop);
        CudaEventRecord(start, 0);

        // Running main handler (it makes all the magic)
        {
            Handler hand(config, solver.get());
            if (!hand) {
                error("Something went wrong in xml run!\n");
                return -1;
            }
        }

        // Finish and clean up
        debug2("CudaFree ...\n");
        CudaEventDestroy(start);
        CudaEventDestroy(stop);

        if (solver->mpi_rank == 0) {
            double duration = get_walltime();
            output("Total duration: %lf s = %lf min = %lf h\n", duration, duration / 60, duration / 60 / 60);
        }
    }

    CudaDeviceReset();
    MPI_Finalize();
}
