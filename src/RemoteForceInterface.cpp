#include "Global.h"
#include "RemoteForceInterface.hpp"
#include "pinned_allocator.hpp"

namespace rfi {
    template class RemoteForceInterface< ForceCalculator, RotParticle, ArrayOfStructures, real_t, pinned_allocator<real_t> >;
};