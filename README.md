CudneLB - the templated version
===
- [![Build Status](https://travis-ci.org/CFD-GO/TCLB.svg?branch=develop)](https://travis-ci.org/CFD-GO/TCLB) [![codecov](https://codecov.io/gh/CFD-GO/TCLB/branch/master/graph/badge.svg)](https://codecov.io/gh/CFD-GO/TCLB) [![documentation](https://raw.githubusercontent.com/CFD-GO/documents/master/assets/documentation.svg?sanitize=true)](https://docs.tclb.io/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3672102.svg)](https://doi.org/10.5281/zenodo.3973739) - Stable release [(`master` branch)](https://github.com/CFD-GO/TCLB/tree/master)
- [![Build Status](https://travis-ci.org/CFD-GO/TCLB.svg?branch=develop)](https://travis-ci.org/CFD-GO/TCLB) [![codecov](https://codecov.io/gh/CFD-GO/TCLB/branch/develop/graph/badge.svg)](https://codecov.io/gh/CFD-GO/TCLB) [![documentation](https://raw.githubusercontent.com/CFD-GO/documents/master/assets/documentation.svg?sanitize=true)](https://develop.docs.tclb.io/) - Current release [(`develop` branch)](https://github.com/CFD-GO/TCLB/tree/develop)

CudneLB is a MPI+CUDA or MPI+CPU high-performance CFD simulation code, based on Lattice Boltzmann Method.

It provides a clear interface for calculation of complex physics, and the implementation of new models.

## Installation

Just clone the repo (or download the [zip file](https://github.com/CFD-GO/TCLB/archive/master.zip)):
```bash
git clone https://github.com/CFD-GO/TCLB.git
cd TCLB
```

If you want a more recent (but less stable) version, you could try the development branch:

```bash
git clone -b develop https://github.com/CFD-GO/TCLB.git
cd TCLB
```

### Dependencies

You'll need:
- [R](https://www.r-project.org/)
- packages for R: [optparse](https://cran.r-project.org/package=optparse), [numbers](https://cran.r-project.org/package=numbers), [rtemplate](https://github.com/llaniewski/rtemplate), [gvector](https://github.com/llaniewski/gvector), [polyAlgebra](https://github.com/llaniewski/polyAlgebra)
- [nVidia CUDA](https://developer.nvidia.com/cuda-zone) (if you want to use GPU)
- [python](https://www.python.org/), [numpy](http://www.numpy.org/) (if you want to use the integrated python interpreter)
- [python](https://www.python.org/), [sympy](http://www.sympy.org/) and R package: [rPython](https://cran.r-project.org/package=rPython) (if you want to develop a model using python instead of R)
- [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) (e.g. [OpenMPI](http://www.open-mpi.org/))

You can install many of these with the tools/install.sh script (note it requires sudo):

```bash
sudo tools/install.sh cuda 6.5-14 # only if your GPU supports cuda
sudo tools/install.sh r
sudo tools/install.sh openmpi
     tools/install.sh rdep
sudo tools/install.sh python-dev
     tools/install.sh rpython
sudo tools/install.sh module # only on CentOS
```

The `install.sh` script is designed to work on Ubuntu (e.g. on the [Travis-CI](https://travis-ci.org/CFD-GO/TCLB) VMs). 
The `install.sh` script should work on CentOS.
You can install the **`sudo`** parts by yourself, and use script to install R packages: rdep and rpython.


### Compilation
This should work:
```bash
module load mpi/openmpi-x86_64 # only on CentOS
make configure
./configure --enable-double --enable-graphics --with-cuda-arch=sm_20 
# only CPU ./configure --enable-double --disable-cuda
make d2q9
```

## Usage

This should also work:
```bash
CLB/d2q9/main example/flow/2d/karman.xml
```

## Documentation

The documentation (including tutorials) is published at
[docs.tclb.io](https://docs.tclb.io/). You can contribute at
[CFD-GO/TCLB_docs](https://github.com/CFD-GO/TCLB_docs).

For the `develop` version, most recent documentation can be found at
[develop.docs.tclb.io](https://develop.docs.tclb.io/).

## Authors

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

Developed at: [C-CFD Group](https://c-cfd.meil.pw.edu.pl/) at [Warsaw University of Technology](http://pw.edu.pl/) from 2012

## License

This software is distributed under the [GPL v3 License](LICENSE).

If you need this software under a different license, please contact the main author.

Contact: llaniewski([monkey](https://en.wikipedia.org/wiki/At_sign#Names_in_other_languages))meil.pw.edu.pl
