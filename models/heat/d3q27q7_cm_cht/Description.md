
# This model uses

* d3q27 cumulant collision kernel for hydrodynamics
* d3q7  cascaded collision kernel for heat
* Boussinesq approx to couple heat with hydrodynamics

## BC

* periodic
* Dirichlet - Equilibrium scheme (1st order convergence)
* Dirichlet - Anti Bounce Back (2nd order convergence)
* Dirichlet - Interpolated Anti Bounce Back (2nd order convergence)
* Neumann like bc - impose heat flux

* WVelocity inlet
* EPressure Outlet
* Neumann Outlet
* Convective Outlet
