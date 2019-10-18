#include "Global.h"
#include "RemoteForceInterface.hpp"

namespace rfi {
    template class RemoteForceInterface< ForceCalculator, RotParticle >;
    template class RemoteForceInterface< ForceCalculator, NRotParticle >;
    template class RemoteForceInterface< ForceCalculator, RotParticle, StructureOfArrays >;
    template class RemoteForceInterface< ForceCalculator, NRotParticle, StructureOfArrays >;
    template class RemoteForceInterface< ForceIntegrator, RotParticle >;
    template class RemoteForceInterface< ForceIntegrator, NRotParticle >;
    template class RemoteForceInterface< ForceIntegrator, RotParticle, StructureOfArrays >;
    template class RemoteForceInterface< ForceIntegrator, NRotParticle, StructureOfArrays >;
};