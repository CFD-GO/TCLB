#include "RemoteForceInterface.hpp"

namespace rfi {
    template class RemoteForceInterface< ForceCalculator, RotParticle >;
    template class RemoteForceInterface< ForceCalculator, NRotParticle >;
    template class RemoteForceInterface< ForceCalculator, RotParticle, StructureOfArrays >;
    template class RemoteForceInterface< ForceCalculator, NRotParticle, StructureOfArrays >;
};