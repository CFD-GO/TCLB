ADJOINT=0
TEST=FALSE
OPT="(q27 + OutFlow  + BGK + thermo*planarBenchmark)*autosym*geometric*staircaseimp*isograd*tprec"
# q27 - Q27 lattice structure for phasefield
#
# OutFlow - include extra velocity stencil for outflowing boundaries
#
# BGK - single relaxation time operator, not really supported- educational purposes only
#
# thermo - include energy equation solver for temperature field, influences through
#        - the surface tension
#
# planarBenchmark - thermocapillary benchmark case, must be built with thermo
#
# autosym - symmetry boundary conditions
#
# geometric - use geometric boundary conditions instead of surface energy, sometimes gives more accurate results
#
# staircaseimp - use staircase improvement (applicable to both surface energy and geometric boundary conditions)
#
# isograd - use isotopic gradient also near boundaries when calculating phase field gradient, this
# 	        essentially uses the value of the phase field gradient at the boundary from the previous iteration
# 	        to calculate the phase field gradient at the boundary in the current iteration
#
# tprec - use more precise triangle for interpolation of phase field gradient (only applicable to geometric
# 	   	  boundary conditions with staircase improvement)
