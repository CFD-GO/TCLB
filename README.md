![TCLB Solver Header](https://raw.githubusercontent.com/CFD-GO/documents/master/assets/header.png)

TCLB Solver

===

- [![Build Status](https://travis-ci.org/CFD-GO/TCLB.svg?branch=master)](https://travis-ci.org/CFD-GO/TCLB) [![codecov](https://codecov.io/gh/CFD-GO/TCLB/branch/master/graph/badge.svg)](https://codecov.io/gh/CFD-GO/TCLB) [![documentation](https://raw.githubusercontent.com/CFD-GO/documents/master/assets/documentation.svg?sanitize=true)](https://docs.tclb.io/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3672102.svg)](https://doi.org/10.5281/zenodo.3973739) - Stable release [(`master` branch)](https://github.com/CFD-GO/TCLB/tree/master)
- [![Build Status](https://travis-ci.org/CFD-GO/TCLB.svg?branch=develop)](https://travis-ci.org/CFD-GO/TCLB) [![codecov](https://codecov.io/gh/CFD-GO/TCLB/branch/develop/graph/badge.svg)](https://codecov.io/gh/CFD-GO/TCLB) [![documentation](https://raw.githubusercontent.com/CFD-GO/documents/master/assets/documentation.svg?sanitize=true)](https://develop.docs.tclb.io/) - Current release [(`develop` branch)](https://github.com/CFD-GO/TCLB/tree/develop)

## Description
TCLB is a MPI+CUDA or MPI+CPU high-performance Conputational Fluid Dynamics simulation code, based on Lattice Boltzmann Method.
It provides a clear interface for calculation of complex physics, and the implementation of new models.

## How to use it

**Install**
```bash
git clone https://github.com/CFD-GO/TCLB.git
cd TCLB
```
**Configure**
```bash
make configure
./configure --enable-graphics --with-cuda-arch=sm_30
```
**Compile**
```bash
make d2q9
```
**Run**
```bash
CLB/d2q9/main example/flow/2d/karman.xml
```

## More information

### Documentation

The documentation (including tutorials) is published at
[docs.tclb.io](https://docs.tclb.io/).

For the `develop` version, most recent documentation can be found at
[develop.docs.tclb.io](https://develop.docs.tclb.io/).

You can contribute to the documentation at
[CFD-GO/TCLB_docs](https://github.com/CFD-GO/TCLB_docs).

### Dependencies

For the code to compile and work you'll need a few things:
- [R](https://www.r-project.org/) and some R packages ([optparse](https://cran.r-project.org/package=optparse), [numbers](https://cran.r-project.org/package=numbers), [rtemplate](https://github.com/llaniewski/rtemplate), [gvector](https://github.com/llaniewski/gvector), [polyAlgebra](https://github.com/llaniewski/polyAlgebra))
- [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface). We recommend [OpenMPI](http://www.open-mpi.org/)
- If you want to use your GPU, you'll need [nVidia CUDA](https://developer.nvidia.com/cuda-zone)
- If you want to integrate TCLB with Python, you'll need [python](https://www.python.org/), [numpy](http://www.numpy.org/) with headers and stuff
- If you want to develop a model using Python, you'll need [python](https://www.python.org/), [sympy](http://www.sympy.org/) and R package: [rPython](https://cran.r-project.org/package=rPython)

You can install many of these with the provided `tools/install.sh` script (note it requires sudo):
```bash
sudo tools/install.sh essentials   # Installs essential system packages needed by TCLB
sudo tools/install.sh r            # Installs R
sudo tools/install.sh openmpi      # Installs OpenMPI
     tools/install.sh rdep         # Installs needed R packages
sudo tools/install.sh cuda         # Installs CUDA (we recommend to do it on your own)
sudo tools/install.sh python-dev   # Installs Python libraries with headers
```

You can run the `tools/install.sh` script with the `--dry` option, which will print the commands to run, so you can run them on your own.
*We do not recommend running anything with sudo without checking*

### `develop` Branch:
If you want a more recent version, you could try the development branch with `git checkout develop`

### CPU
To compile the code for CPU, you can use the `--disable-cuda` option for `./configure`:
```
./configure --disable-cuda
```
### Parallel run
To run TCLB in parallel (both on multiple CPU and multiple GPU), you can use the standard syntax of MPI parallel run:
```bash
mpirun -np 8 CLB/d2q9/main example/flow/2d/karman.xml
```

## About

### Authors

TCLB is developed from 2012 with the aim at providing a framework for efficient CFD computations with LBM, mainly for research.

Author: [Łukasz Łaniewski-Wołłk](https://github.com/llaniewski)

Co-authors:
* [Michał Dzikowski](https://github.com/mdzik)

Contributors:
* [Wojtek Regulski](https://github.com/wojtasMEiL)
* [Zbigniew Gawłowicz](https://github.com/zgawlowicz)
* [Mariusz Rutkowski](https://github.com/mrutkowski-aero)
* [Dmytro Sashko](https://github.com/shkodm)
* [Travis Mitchell](https://github.com/TravisMitchell)
* [Paweł Obrępalski](https://github.com/PabloOb)
* [Grzegorz Gruszczyński](https://github.com/ggruszczynski)

Developed at:
- [Zakład Aerodynamiki](https://meil.pw.edu.pl/ZA/) at [Politechnika Warszawska (Warsaw University of Technology)](http://pw.edu.pl/)
- [School of Mechanical & Mining Engineering](https://www.mechmining.uq.edu.au/) at [University of Queensland](http://uq.edu.au/)

### License

This software is distributed under the [GPL v3 License](LICENSE).

If you need this software under a different license, please contact the main author.

Contact: llaniewski([monkey](https://en.wikipedia.org/wiki/At_sign#Names_in_other_languages))meil.pw.edu.pl
