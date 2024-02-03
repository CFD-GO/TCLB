![TCLB Solver Header](https://raw.githubusercontent.com/CFD-GO/documents/master/assets/header.png)

TCLB Solver [![ZENADO DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3550331.svg)](https://doi.org/10.5281/zenodo.3550331) [![Article](https://zenodo.org/badge/DOI/10.1016/j.camwa.2015.12.043.svg)](https://doi.org/10.1016/j.camwa.2015.12.043)
===
TCLB is a MPI+CUDA, MPI+CPU or MPI+HIP high-performance Computational Fluid Dynamics simulation code, based on the Lattice Boltzmann Method.
It provides a clear interface for calculation of complex physics, and the implementation of new models.

**Stable release** [(`master` branch)](https://github.com/CFD-GO/TCLB/tree/master):<br/>[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/CFD-GO/TCLB?quickstart=1)<br/>[![CPU build status](https://github.com/CFD-GO/TCLB/actions/workflows/cpu_test.yml/badge.svg?branch=master)](https://github.com/CFD-GO/TCLB/actions/workflows/cpu_test.yml) [![CUDA build status](https://github.com/CFD-GO/TCLB/actions/workflows/gpu_comp.yml/badge.svg?branch=master)](https://github.com/CFD-GO/TCLB/actions/workflows/gpu_comp.yml) [![HIP build status](https://github.com/CFD-GO/TCLB/actions/workflows/hip_comp.yml/badge.svg?branch=develop)](https://github.com/CFD-GO/TCLB/actions/workflows/hip_comp.yml) [![codecov](https://codecov.io/gh/CFD-GO/TCLB/branch/master/graph/badge.svg)](https://codecov.io/gh/CFD-GO/TCLB) [![documentation](https://raw.githubusercontent.com/CFD-GO/documents/master/assets/documentation.svg?sanitize=true)](https://docs.tclb.io/)


**Current release** [(`develop` branch)](https://github.com/CFD-GO/TCLB/tree/develop):</br>[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/CFD-GO/TCLB/tree/develop?quickstart=1)<br/>[![CPU build status](https://github.com/CFD-GO/TCLB/actions/workflows/cpu_test.yml/badge.svg?branch=develop)](https://github.com/CFD-GO/TCLB/actions/workflows/cpu_test.yml) [![CUDA build status](https://github.com/CFD-GO/TCLB/actions/workflows/gpu_comp.yml/badge.svg?branch=develop)](https://github.com/CFD-GO/TCLB/actions/workflows/gpu_comp.yml) [![HIP build status](https://github.com/CFD-GO/TCLB/actions/workflows/hip_comp.yml/badge.svg?branch=develop)](https://github.com/CFD-GO/TCLB/actions/workflows/hip_comp.yml) [![codecov](https://codecov.io/gh/CFD-GO/TCLB/branch/develop/graph/badge.svg)](https://codecov.io/gh/CFD-GO/TCLB) [![documentation](https://raw.githubusercontent.com/CFD-GO/documents/master/assets/documentation.svg?sanitize=true)](https://develop.docs.tclb.io/)

## How to use it

**Install**
```bash
git clone https://github.com/CFD-GO/TCLB.git
cd TCLB
```
**Configure**
```bash
make configure
./configure
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

For the `develop` version, the most recent documentation can be found at
[develop.docs.tclb.io](https://develop.docs.tclb.io/).

You can contribute to the documentation at
[CFD-GO/TCLB_docs](https://github.com/CFD-GO/TCLB_docs).

### Supported architectures
This code is designed to run on **Linux** with **CUDA**. We strongly recommend using Linux for compilation, computation and postprocessing.

Nevertheless, TCLB can be compiled on Windows using the [Windows Subsystem for Linux](https://pl.wikipedia.org/wiki/Windows_Subsystem_for_Linux), with CUDA supported on some system configurations (see [nVidia's website](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) for more info).  It also can be compiled on MacOS (CPU only). Both Debian and Red Hat based Linux distributions are supported by the `install.sh` script described below, as is MacOS (with `brew` package manager).

### Dependencies

For the code to compile and work you'll need a few things:
- [R](https://www.r-project.org/) and some R packages ([optparse](https://cran.r-project.org/package=optparse), [rtemplate](https://github.com/llaniewski/rtemplate), [gvector](https://github.com/llaniewski/gvector), [polyAlgebra](https://github.com/llaniewski/polyAlgebra))
- [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface). We recommend [OpenMPI](http://www.open-mpi.org/)
- To use your GPU, you'll need [nVidia CUDA](https://developer.nvidia.com/cuda-zone) or [AMD HIP/ROCm](https://www.amd.com/en/graphics/servers-solutions-rocm)

Optionally, you may need:
- To integrate TCLB with R, you'll need R package [rinside](https://github.com/eddelbuettel/rinside)
- To integrate TCLB with Python, you'll need [python](https://www.python.org/), [numpy](http://www.numpy.org/) with libraries and headers
- To develop a model using Python, you'll need [python](https://www.python.org/), [sympy](http://www.sympy.org/) and R package [reticulate](https://cran.r-project.org/package=reticulate)

You can install many of these with the provided `tools/install.sh` script (note that this requires sudo):
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
### HIP
To compile the code for AMD GPUs (ROCm), you can use the `--enable-hip` option for `./configure`:
```
./configure --enable-hip
```
### Parallel run
To run TCLB in parallel (both on multiple CPU and multiple GPU), you can use the standard syntax of MPI parallel run:
```bash
mpirun -np 8 CLB/d2q9/main example/flow/2d/karman.xml
```

### Running on clusters
To assist with using TCLB on HPC clusters (SLURM/PBS), there are scripts provided in the [TCLB_cluster](https://github.com/CFD-GO/TCLB_cluster) repository.

### LBM-DEM computation
TCLB code can be coupled with Discrete Element Method (DEM) codes, to enable computation of flow with particles.

The DEM codes that TCLB can be integrated with are:
- [LIGGGHTS](https://www.cfdem.com/liggghts-open-source-discrete-element-method-particle-simulation-code)
- [LAMMPS](https://lammps.sandia.gov/)
- [ESYS-Particle](https://launchpad.net/esys-particle)

Refer to the documentation for instructions on compilation and coupling.

## Models
For users looking to apply existing LBM methods, common/supported models are below. Note extensions to these models exist using the [TCLB's overlay](https://github.com/CFD-GO/TCLB_overlay) framework and TCLB optional compile flags.

**Two-Dimensional**
- d2q9: MRT LBM for single-phase flow.
- d2q9_les: MRT LBM with Smagorinski LES turbulence model.
- d2q9q9_cm_cht: thermal LBM with Boussinesq approx for coupling and cumulant or cascaded relaxation kernels.
- d2q9_pf_velocity: multiphase LBM based on the phase field model and incompressible LBM.

**Three-Dimensional**
- d3q27_cumulant: cumulant LBM with options for:
     * Interpolated bounceback.
     * Smagorinski LES turbulence model.
- d3q27q27_cm_cht: thermal LBM with Boussinesq approx for coupling and cumulant or cascaded collision relaxation kernels.
- d3q27_pf_velocity: multiphase LBM based on the phase field model and incompressible LBM.
     * Options for various contact angle implementations (surface energy or geometric)
     * [Thermocapillary flow extension](https://github.com/TravisMitchell/thermocapillary)

**Particle (DEM) Coupled**
- d3q27_PSM: Applies the partially saturated model for coupling particles in single phase flow.
     * Options for Two-Relaxation-Time kernel
     * Option for Non-Equilibiurm Bounce-Back and Superposition for the DEM-LBM coupling.

## About

### Authors

TCLB began development in 2012 with the aim at providing a framework for efficient CFD computations with LBM, mainly for research.

Author: [Łukasz Łaniewski-Wołłk](https://github.com/llaniewski)

Major contributors:
* [Michał Dzikowski](https://github.com/mdzik)
* [Travis Mitchell](https://github.com/TravisMitchell)

Contributors:
* [Nathan Di Vaira](https://github.com/ndivaira)
* [Grzegorz Gruszczyński](https://github.com/ggruszczynski)
* [Bryce Hill](https://github.com/bhill23)
* [Jon McCullough](https://github.com/JonMcCullough)
* [Paweł Obrępalski](https://github.com/PabloOb)
* [Wojciech Regulski](https://github.com/wojtasMEiL)
* [Mariusz Rutkowski](https://github.com/mrutkowski-aero)
* [Dmytro Sashko](https://github.com/shkodm)

Developed at:
- [Zakład Aerodynamiki](https://meil.pw.edu.pl/ZA/) at [Politechnika Warszawska (Warsaw University of Technology)](http://pw.edu.pl/)
- [School of Mechanical & Mining Engineering](https://www.mechmining.uq.edu.au/) at [University of Queensland](http://uq.edu.au/)
- [Interdisciplinary Centre for Mathematical and Computational Modelling](https://icm.edu.pl/en) at [University of Warsaw](https://www.uw.edu.pl/)

### Citation
Please use **appropriate citations if using this software** in any research publication. The publication should cite [the original paper about TCLB](https://doi.org/10.1016/j.camwa.2015.12.043) and papers which describe the used LBM models. You can find the list of TCLB publications at [docs.tclb.io/general-info/publications/](https://docs.tclb.io/general-info/publications/). You can also find the information about published articles in the source code of the models.
The code can be cited additionally, by its [Zenodo DOI](https://doi.org/10.5281/zenodo.3550331).

### License

This software is distributed under the [GPL v3 License](LICENSE).

If you need this software under a different license, please contact the main author.

Contact: lukasz.laniewski([monkey](https://en.wikipedia.org/wiki/At_sign#Names_in_other_languages))pw.edu.pl
