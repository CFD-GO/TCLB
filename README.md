CudneLB - the templated version
===
[![Build Status](https://travis-ci.org/CFD-GO/TCLB.svg?branch=develop)](https://travis-ci.org/CFD-GO/TCLB) [![Coverage Status](https://coveralls.io/repos/github/CFD-GO/TCLB/badge.svg?branch=develop)](https://coveralls.io/github/CFD-GO/TCLB?branch=develop)

CudneLB is a MPI+CUDA or MPI+CPU high-performance CFD simulation code, based on Lattice Boltzmann Method.

It provides a clear interface for calculation of complex physics, and implementing new models.

## Installation

### Dependencies

You'll need:
- [R](https://www.r-project.org/)
- packages for R: [optparse](https://cran.r-project.org/package=optparse), [numbers](https://cran.r-project.org/package=numbers), [rtemplate](https://github.com/llaniewski/rtemplate), [gvector](https://github.com/llaniewski/gvector), [polyAlgebra](https://github.com/llaniewski/polyAlgebra)
- [nVidia CUDA](https://developer.nvidia.com/cuda-zone) (if you want to use GPU)
- [python](https://www.python.org/), [numpy](http://www.numpy.org/) (if you want to use the integrated python interpreter)
- [python](https://www.python.org/), [python](http://www.sympy.org/) and R package: [rPython](https://cran.r-project.org/package=rPython) (if you want to develop a model using python in place or R)
- [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) (e.g. [OpenMPI](http://www.open-mpi.org/))

You can install lot of these with the tools/install.sh script (if you are not afraid of running a script with sudo):
```bash
sudo tools/install.sh cuda 6.5-14
sudo tools/install.sh r
sudo tools/install.sh openmpi
     tools/install.sh rdep
sudo tools/install.sh python-dev
     tools/install.sh rpython
```
The `install.sh` script is designed to work on Ubuntu (e.g. on the [Travis-CI](https://travis-ci.org/CFD-GO/TCLB) VMs). You can install the **`sudo`** parts by yourself, and use script to install R packages: rdep and rpython.

### Compilation
This should work:
```bash
make configure
./configure --enable-double --enable-graphics --with-cuda-arch=sm_20
make d2q9
```

## Usage

This should also work:
```bash
CLB/d2q9/main example/karman.xml
```

## Authors

Author: [Łukasz Łaniewski-Wołłk](https://github.com/llaniewski)

Co-authors:
* [Michał Dzikowski](https://github.com/mdzik)

Contributors:
* [Wojtek Regulski](https://github.com/wojtasMEiL)
* [Zbigniew Gawłowicz](https://github.com/zgawlowicz)
* [Mariusz Rutkowski](https://github.com/mrutkowski-aero)
* [Dmytro Sashko](https://github.com/shkodm)

Developed at: [C-CFD Group](https://c-cfd.meil.pw.edu.pl/) at [Warsaw University of Technology](http://pw.edu.pl/) from 2012

## License
The software is free to use for non-commercial purposes.

*Any usage should be acknowledged accordingly (at least with a reference to this repository).*
* Author wanting to use this software in scientific publications should consult the author for the apropriate reference.
* Any commercial use should be consulted beforehead with the author
* Any derived code (even if only on a specific model) should be acknowledged

And, most importantly: if the software proved to be useful for you, please write!

Contact: llaniewski([monkey](https://en.wikipedia.org/wiki/At_sign#Names_in_other_languages))meil.pw.edu.pl
