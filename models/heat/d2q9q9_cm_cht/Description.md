
# This model uses

* d2q7 cumulant collision kernel for hydrodynamics
* d2q7 cumulant or cascaded collision kernel for heat
* Boussinesq approx to couple heat with hydrodynamics

It is a 2d version od d3q27q27_cm_cht

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
