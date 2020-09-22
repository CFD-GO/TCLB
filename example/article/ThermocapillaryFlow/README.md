Run files for LBM paper on thermocapillary flows
===
Responsible: @TravisMitchell

These files were developed for a publication,

T. Mitchell et al. Computational modelling of three-dimensional thermocapillary flow of recalcitrant bubbles using a coupled lattice Boltzmann-finite difference method

to be submitted to Physical Review E in 2020.

They are run with the solver d3q27_pf_velocity_thermo and d3q27_pf_velocity_thermo_layeredBenchmark

## Layered Thermocapillary
Is a pseudo-2D simulation using a 3D solver to verify the implementation in comparison with previous 2D models.

## Bubble Migration
Bubbles are placed within a temperature gradient and driven due to the thermal effects on surface temperature. This is then compared to the theoretical YGB velocity.

## Recalcitrant Bubble
Certain fluids exhibit a more complex relation between temperature gradient and surface tension. This can lead to unexpected bubble migration, which is investigated in these files.
