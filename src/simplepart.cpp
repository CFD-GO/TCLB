#include "Global.h"
#include "MPMD.hpp"
#include "RemoteForceInterface.hpp"
#include "pugixml.hpp"
#include <math.h>
#include <vector>

struct Particle {
  double x[3];
  double r;
  double m;
  double v[3];
  double f[3];
  Particle() {
    x[0] = 0;
    x[1] = 0;
    x[2] = 0;
    v[0] = 0;
    v[1] = 0;
    v[2] = 0;
    m = 0;
    r = 0;
  }
};

typedef std::vector<Particle> Particles;

int main(int argc, char *argv[]) {
  int ret;
  MPMDHelper MPMD;
  MPI_Init(&argc, &argv);
  MPMD.Init(MPI_COMM_WORLD, "SIMPLEPART");
  DEBUG_SETRANK(MPMD.local_rank);
  InitPrint(DEBUG_LEVEL, 6, 8);
  if (MPMD.local_size > 1) {
    ERROR("simplepart: Can only run on single MPI rank");
    return -1;
  }
  MPMD.Identify();
  rfi::RemoteForceInterface<rfi::ForceIntegrator, rfi::RotParticle> RFI;
  RFI.name = "SIMPLEPART";

  MPMDIntercomm inter = MPMD["TCLB"];
  ret = RFI.Connect(MPMD.work, inter.work);
  if (ret)
    return ret;
  assert(RFI.Connected());

  std::vector<size_t> wsize, windex;
  wsize.resize(RFI.Workers());
  windex.resize(RFI.Workers());
  Particles particles;

  double periodicity[3];
  bool periodic[3];
  for (int i = 0; i < 3; i++) {
    periodic[i] = false;
    periodicity[i] = 0.0;
  }

  if (argc != 2) {
    printf("Syntax: simplepart config.xml\n");
    MPI_Abort(MPI_COMM_WORLD,1);
    exit(1);
  }
  char * filename = argv[1];
  pugi::xml_document config;
  pugi::xml_parse_result result = config.load_file(filename, pugi::parse_default | pugi::parse_comments);
  if (!result) {
    ERROR("Error while parsing %s: %s\n", filename, result.description());
    return -1;
  }

  pugi::xml_node main_node = config.child("SimplePart");
  if (! main_node) {
    ERROR("No SimplePart element in %s", filename);
    return -1;
  }

  for (pugi::xml_node node = main_node.first_child(); node; node = node.next_sibling()) {
    std::string node_name = node.name();
    if (node_name == "Particle") {
      Particle p;
      for (pugi::xml_attribute attr = node.first_attribute(); attr; attr = attr.next_attribute()) {
        std::string attr_name = attr.name();
        if (attr_name == "x") {
          p.x[0] = attr.as_double();
        } else if (attr_name == "y") {
          p.x[1] = attr.as_double();
        } else if (attr_name == "z") {
          p.x[2] = attr.as_double();
        } else if (attr_name == "vx") {
          p.v[0] = attr.as_double();
        } else if (attr_name == "vy") {
          p.v[1] = attr.as_double();
        } else if (attr_name == "vz") {
          p.v[2] = attr.as_double();
        } else if (attr_name == "r") {
          p.r = attr.as_double();
        } else if (attr_name == "m") {
          p.m = attr.as_double();
        } else {
          ERROR("Unknown atribute '%s' in '%s'", attr.name(), node.name());
          return -1;
        }
      }
      if (p.r <= 0.0) {
        ERROR("Specify the radius with 'r' attribute");
        return -1;
      }
      particles.push_back(p);
    } else if (node_name == "Periodic") {
      for (pugi::xml_attribute attr = node.first_attribute(); attr; attr = attr.next_attribute()) {
        std::string attr_name = attr.name();
        if (attr_name == "x") {
          periodic[0] = true;
          periodicity[0] = attr.as_double();
        } else if (attr_name == "y") {
          periodic[1] = true;
          periodicity[1] = attr.as_double();
        } else if (attr_name == "z") {
          periodic[2] = true;
          periodicity[2] = attr.as_double();
        } else {
          ERROR("Unknown atribute '%s' in '%s'", attr.name(), node.name());
          return -1;
        }
      }
    } else {
      ERROR("Unknown node '%s' in '%s'", node.name(), main_node.name());
      return -1;
    }
  }

  while (RFI.Active()) {

    for (int phase = 0; phase < 3; phase++) {
      if (phase == 0) {
        for (int i = 0; i < RFI.Workers(); i++)
          wsize[i] = 0;
      } else {
        for (int i = 0; i < RFI.Workers(); i++)
          windex[i] = 0;
      }

      for (Particles::iterator p = particles.begin(); p != particles.end(); p++) {
        if (phase == 2) {
          p->f[0] = 0;
          p->f[1] = 0;
          p->f[2] = 0;
        }
        int minper[3], maxper[3], d[3];
        size_t offset = 0;
        for (int worker = 0; worker < RFI.Workers(); worker++) {
          for (int j = 0; j < 3; j++) {
            double prd = periodicity[j];
            double lower = 0;
            double upper = periodicity[j];
            if (RFI.WorkerBox(worker).declared) {
              lower = RFI.WorkerBox(worker).lower[j];
              upper = RFI.WorkerBox(worker).upper[j];
            }
            if (periodic[j]) {
              maxper[j] = floor((upper - p->x[j] + p->r) / prd);
              minper[j] = ceil((lower - p->x[j] - p->r) / prd);
            } else {
              if ((p->x[j] + p->r >= lower) && (p->x[j] - p->r <= upper)) {
                minper[j] = 0;
                maxper[j] = 0;
              } else {
                minper[j] = 0;
                maxper[j] = -1; // no balls
              }
            }
          }

          int copies = (maxper[0] - minper[0] + 1) * (maxper[1] - minper[1] + 1) * (maxper[2] - minper[2] + 1);
          for (d[0] = minper[0]; d[0] <= maxper[0]; d[0]++) {
            for (d[1] = minper[1]; d[1] <= maxper[1]; d[1]++) {
              for (d[2] = minper[2]; d[2] <= maxper[2]; d[2]++) {
                double px[3];
                for (int j = 0; j < 3; j++)
                  px[j] = p->x[j] + d[j] * periodicity[j];
                if (phase == 0) {
                  wsize[worker]++;
                } else {
                  size_t i = offset + windex[worker];
                  if (phase == 1) {
                    RFI.setData(i, RFI_DATA_R, p->r);
                    RFI.setData(i, RFI_DATA_POS + 0, px[0]);
                    RFI.setData(i, RFI_DATA_POS + 1, px[1]);
                    RFI.setData(i, RFI_DATA_POS + 2, px[2]);
                    RFI.setData(i, RFI_DATA_VEL + 0, p->v[0]);
                    RFI.setData(i, RFI_DATA_VEL + 1, p->v[1]);
                    RFI.setData(i, RFI_DATA_VEL + 2, p->v[2]);
                    if (RFI.Rot()) {
                      RFI.setData(i, RFI_DATA_ANGVEL + 0, 0.0);
                      RFI.setData(i, RFI_DATA_ANGVEL + 1, 0.0);
                      RFI.setData(i, RFI_DATA_ANGVEL + 2, 0.0);
                    }
                  } else {
                    p->f[0] += RFI.getData(i, RFI_DATA_FORCE + 0);
                    p->f[1] += RFI.getData(i, RFI_DATA_FORCE + 1);
                    p->f[2] += RFI.getData(i, RFI_DATA_FORCE + 2);
                  }
                  windex[worker]++;
                }
              }
            }
          }
          offset += wsize[worker];
        }
      }
      if (phase == 0) {
        for (int worker = 0; worker < RFI.Workers(); worker++)
          RFI.Size(worker) = wsize[worker];
        RFI.SendSizes();
        RFI.Alloc();
      } else if (phase == 1) {
        RFI.SendParticles();
        RFI.SendForces();
      } else {
      }
    }
  }

  if (RFI.Connected()) {
    RFI.Close();
    MPI_Finalize();
  }
  return 0;
}
