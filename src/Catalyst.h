#ifndef FEADAPTOR_HEADER
#define FEADAPTOR_HEADER

class Solver;

namespace CatalystAdaptor
{
  void Initialize(bool exportCellData);
  void AddScript(const char * script);
  
  void Finalize();

  void CoProcess(Solver& grid, double time,
                 unsigned int timeStep, bool lastTimeStep);
}

#endif
