#include <iostream>
#include "Catalyst.h"
#include "Solver.h"

#include <vtkCellData.h>
#include <vtkCellType.h>
#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkCPProcessor.h>
#include <vtkCPPythonScriptPipeline.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPointData.h>

#ifdef CALC_DOUBLE_PRECISION
  typedef vtkDoubleArray vtkRealTArray;
#else
  typedef vtkFloatArray vtkRealTArray;
#endif
              

namespace
{
  vtkCPProcessor* Processor = NULL;
  vtkImageData* VTKGrid = NULL;
  bool exportCellData;
  int extent[6];
  int wholeExtent[6];
  lbRegion reg;
  
  void BuildVTKGrid(Solver& solver)
  {
    // The grid structure isn't changing so we only build it
    // the first time it's needed. If we needed the memory
    // we could delete it and rebuild as necessary.
    if(VTKGrid == NULL)
      {
      VTKGrid = vtkImageData::New();
      extent[0] = solver.lattice->region.dx;
      extent[1] = solver.lattice->region.dx + solver.lattice->region.nx;
      extent[2] = solver.lattice->region.dy;
      extent[3] = solver.lattice->region.dy + solver.lattice->region.ny;
      extent[4] = solver.lattice->region.dz;
      extent[5] = solver.lattice->region.dz + solver.lattice->region.nz;
      wholeExtent[0] = solver.lattice->mpi.totalregion.dx;
      wholeExtent[1] = solver.lattice->mpi.totalregion.dx + solver.lattice->mpi.totalregion.nx;
      wholeExtent[2] = solver.lattice->mpi.totalregion.dy;
      wholeExtent[3] = solver.lattice->mpi.totalregion.dy + solver.lattice->mpi.totalregion.ny;
      wholeExtent[4] = solver.lattice->mpi.totalregion.dz;
      wholeExtent[5] = solver.lattice->mpi.totalregion.dz + solver.lattice->mpi.totalregion.nz;
      if (! exportCellData) {
        if (extent[1] == wholeExtent[1]) extent[1]--;
        wholeExtent[1]--;
        if (extent[3] == wholeExtent[3]) extent[3]--;
        wholeExtent[3]--;
        if (extent[5] == wholeExtent[5]) extent[5]--;
        wholeExtent[5]--;
      }
      reg.dx = extent[0];
      reg.nx = extent[1]-extent[0];
      reg.dy = extent[2];
      reg.ny = extent[3]-extent[2];
      reg.dz = extent[4];
      reg.nz = extent[5]-extent[4];
      if (! exportCellData) {
        reg.nx++;
        reg.ny++;
        reg.nz++;
      }
      VTKGrid->SetExtent(extent);
      double spacing[3];
      spacing[0] = spacing[1] = spacing[2] = 1/solver.units.alt("m");
      VTKGrid->SetSpacing(spacing);
      }
  }

  void fixPointData(lbRegion& old, lbRegion& reg, void* data_, int element_size) {
        int old_change = -1;
        int chunk = 0;
        int ind=0, ind2=0;
        char * data = (char*) data_;
        element_size = element_size / sizeof(char); // Just to be on the safe side
        for (int z=old.dz+old.nz; z>old.dz; z--) {
          for (int y=old.dy+old.ny; y>old.dy; y--) {
            for (int x=old.dx+old.nx; x>old.dx; x--) {
              int change = reg.offset(x-1,y-1,z-1) - old.offset(x-1,y-1,z-1);
              if (change != old_change) {
                if (old_change != -1) {
                  debug0("moving chunk of size %d from %d to %d\n", chunk, ind, ind2);
                  memmove(data + ind2*element_size, data + ind*element_size, chunk*element_size);
                }
                old_change = change;
                chunk = 0;
              }
              ind  = old.offset(x-1,y-1,z-1);
              ind2 = reg.offset(x-1,y-1,z-1);
              chunk++;
            }
          }
        }
        if (old_change != 0) {
          error("while moving data for PointData output something went wrong");
        }

  }

  void UpdateVTKAttributes(Solver& solver)
  {
    vtkIdType numberOf;
    if (exportCellData) {
      numberOf = VTKGrid->GetCellData()->GetNumberOfArrays();
    } else {
      numberOf = VTKGrid->GetPointData()->GetNumberOfArrays();
    }        
    if(numberOf == 0)
    {
      debug2("Creating Catalyst VTK objects\n");
	for (ModelBase::Quantities::const_container it=solver->lattice->model->quantities.begin(); it!=solver->lattice->model->quantities.end(); it++) {
#ifndef ADJOINT
	  if (it->isAdjoint) continue;
#endif
          debug1("Creating Catalyst VTK object for %s\n", it->name);
          vtkIdType size = static_cast<vtkIdType> (reg.size());
          vtkNew<vtkRealTArray> myArray;
          myArray->SetName(it->name.c_str());
	  int comp = 1;
	  if (it->isVector) comp = 3;
          myArray->SetNumberOfComponents(comp);
          myArray->SetNumberOfTuples(size);
          if (exportCellData) {
            VTKGrid->GetCellData()->AddArray(myArray.GetPointer());
          } else {
            VTKGrid->GetPointData()->AddArray(myArray.GetPointer());
          }        
	}
    }
	for (ModelBase::Quantities::const_container it=solver->lattice->model->quantities.begin(); it!=solver->lattice->model->quantities.end(); it++) {
#ifndef ADJOINT
	  if (it->isAdjoint) continue;
#endif
	  int comp = 1;
	  if (it->isVector) comp = 3;
      debug2("Filling Catalyst VTK object for %s\n", it->name);
      vtkIdType size = reg.size();
      vtkRealTArray* myArray;
      if (exportCellData) {
        myArray = vtkRealTArray::SafeDownCast(VTKGrid->GetCellData()->GetArray(it->name.c_str()));
      } else {
        myArray = vtkRealTArray::SafeDownCast(VTKGrid->GetPointData()->GetArray(it->name.c_str()));
      }      
      vtkIdType numTuples = myArray->GetNumberOfComponents();
      real_t * myArrayData = myArray->WritePointer(0,size * numTuples);
      double myUnit = solver.units.alt(it->unit);
      lbRegion old = solver.lattice->region;      
      solver.lattice->GetQuantity(old, myArrayData, 1/myUnit);
      
      if (! exportCellData) {
        fixPointData(old, reg, myArrayData, sizeof(real_t) * comp);
      }      
    }
  }

  void BuildVTKDataStructures(Solver& solver)
  {
    BuildVTKGrid(solver);
    UpdateVTKAttributes(solver);
  }
}

namespace CatalystAdaptor
{

  void Initialize(bool exportCellData_)
  {
    exportCellData = exportCellData_;
    if(Processor == NULL)
      {
      debug2("Initializing Catalyst coProcessor\n");
      Processor = vtkCPProcessor::New();
      Processor->Initialize();
      }
    else
      {
      debug2("Clearing Catalyst coProcessor\n");
      Processor->RemoveAllPipelines();
      }
  }
  
  void AddScript(const char * script)
  {
      notice("Adding Catalyst script: %s\n", script);
      vtkNew<vtkCPPythonScriptPipeline> pipeline;
      pipeline->Initialize(script);
      Processor->AddPipeline(pipeline.GetPointer());
  }

  void Finalize()
  {
    if(Processor)
    {
      Processor->Delete();
      Processor = NULL;
    }
    if(VTKGrid)
    {
      VTKGrid->Delete();
      VTKGrid = NULL;
    }
  }

  void CoProcess(Solver& solver, double time,
                 unsigned int timeStep, bool lastTimeStep)
  {
    debug2("Running Catalyst update procedure\n");
    vtkNew<vtkCPDataDescription> dataDescription;
    dataDescription->AddInput("input");
    dataDescription->SetTimeData(time, timeStep);
    if(lastTimeStep == true)
      {
      // assume that we want to all the pipelines to execute if it
      // is the last time step.
      dataDescription->ForceOutputOn();
      debug2("Last iteration in Catalyst\n");
      }
    if(Processor->RequestDataDescription(dataDescription.GetPointer()) != 0)
      {
      debug2("Data requested by Catalyst\n");
      BuildVTKDataStructures(solver);
      dataDescription->GetInputDescriptionByName("input")->SetGrid(VTKGrid);


      dataDescription->GetInputDescriptionByName("input")->SetWholeExtent(wholeExtent);
      debug2("Running Catalyst CoProcessing\n");
      Processor->CoProcess(dataDescription.GetPointer());
      }
      
  }
} // end of Catalyst namespace
