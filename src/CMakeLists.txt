cmake_minimum_required(VERSION 2.8.9)
project (main)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${CUDA_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${CUDA_LINK_FLAGS})

add_compile_options ( -O3 -Wno-write-strings )

#cuda_add_executable
cuda_add_executable(main
  RemoteForceInterface.cpp
  Handlers/acSyntheticTurbulence.cpp
  Handlers/acUSAdjoint.cpp
  Handlers/acLoadMemoryDump.cpp
  Handlers/vHandler.cpp
  Handlers/Callback.cpp
  Handlers/GenericContainer.cpp
  Handlers/acThresholdNow.cpp
  Handlers/cbTXT.cpp
  Handlers/acLoadBinary.cpp
  Handlers/MainContainer.cpp
  Handlers/cbKeep.cpp
  Handlers/acGeometry.cpp
  Handlers/cbSample.cpp
  Handlers/acESYSParticle.cpp
  Handlers/conExtrude.cpp
  Handlers/acInit.cpp
  Handlers/cbAveraging.cpp
  Handlers/acSAdjoint.cpp
  Handlers/BSpline.cpp
  Handlers/OptimalControl.cpp
  Handlers/acSolve.cpp
  Handlers/conControlParameter.cpp
  Handlers/OptimalControlSecond.cpp
  Handlers/cbPythonCall.cpp
  Handlers/IfContainer.cpp
  Handlers/cbPID.cpp
  Handlers/GenericOptimizer.cpp
  Handlers/acRepeat.cpp
  Handlers/Design.cpp
  Handlers/conFieldParameter.cpp
  Handlers/cbSaveBinary.cpp
  Handlers/RepeatControl.cpp
  Handlers/acOptSolve.cpp
  Handlers/cbBIN.cpp
  Handlers/GenericAction.cpp
  Handlers/acFDTest.cpp
  Handlers/acModel.cpp
  Handlers/acOptimize.cpp
  Handlers/acRemoteForceInterface.cpp
  Handlers/acAndersen.cpp
  Handlers/cbStop.cpp
  Handlers/cbRunR.cpp
  Handlers/cbLog.cpp
  Handlers/NullHandler.cpp
  Handlers/Action.cpp
  Handlers/cbSaveMemoryDump.cpp
  Handlers/acParam.cpp
  Handlers/cbDumpSettings.cpp
  Handlers/cbHDF5.cpp
  Handlers/acObjective.cpp
  Handlers/conControl.cpp
  Handlers/cbVTK.cpp
  Handlers/cbFailcheck.cpp
  Handlers/InternalTopology.cpp
  Handlers/Fourier.cpp
  Handlers/acThreshold.cpp
  Global.cpp
  Geometry.cpp
  Solver.cpp
  SyntheticTurbulence.cpp
  pugixml.cpp
  Sampler.cpp
  def.cpp
  vtkOutput.cpp
  ZoneSettings.cpp
  hdf5Lattice.cpp
  vtkLattice.cpp
  main.cpp
  Lattice.cu
  cross.cu
  BallTree.cpp
  unit.cpp
  xpath_modification.cpp
  cuda.cu
)

target_link_libraries (main ${MPI_LIBRARIES})
#target_link_libraries (main cudart)
target_link_libraries (main ${CUDA_LIBRARIES})
