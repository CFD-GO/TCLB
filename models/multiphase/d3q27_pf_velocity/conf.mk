ADJOINT=0
TEST=FALSE
OPT="q27*ML*OutFlow*altContactAngle*BGK*thermo*planarBenchmark*autosym"
# q27 - Q27 lattice structure for phasefield
# ML  - export densities for machine learning
# OutFlow - include extra velocity stencil for outflowing boundaries
# altContactAngle - geometric contact angle implementation, implemented by dmytro merged into code by travis
# BGK - single relaxation time operator
# thermo - include energy equation solver for temperature field, influences through
#        - the surface tension
# planarBenchmark - thermocapillary benchmark case
# autosym - symmetry boundary conditions