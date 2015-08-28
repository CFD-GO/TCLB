CudneLB - the templated version
===
[![Build Status](https://travis-ci.org/llaniewski/TCLB.svg?branch=develop)](https://travis-ci.org/llaniewski/TCLB)
[![Coverage Status](https://coveralls.io/repos/llaniewski/TCLB/badge.svg?branch=develop&service=github)](https://coveralls.io/github/llaniewski/TCLB?branch=develop)

CudneLB is a MPI+CUDA or MPI+CPU high-performance CFD simulation code, based on Lattice Boltzmann Method.

It provides a clear interface for calculation of complex physics, and implementing new models.

## Installation

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
* Wojtek Regulski
* [Zbigniew Gawłowicz](https://github.com/zgawlowicz)
* [Mariusz Rutkowski](https://github.com/mrutkowski-aero)

Developed at: [C-CFD Group](https://c-cfd.meil.pw.edu.pl/) at [Warsaw University of Technology](http://pw.edu.pl/) from 2012

## License
The software is free to use for non-commercial purposes.

*Any usege should be acknowledged accordingly (at least with a reference to this repository).*
* Author wanting to use this software in scientific publications should consult the author for the apropriate reference.
* Any commercial use should be consulted beforehead with the author
* Any derivaed code (even if only on a specific model) should be acknowledged

And, most importantly: if the software proved to be useful for you, please write!

Contact: llaniewski([monkey](https://en.wikipedia.org/wiki/At_sign#Names_in_other_languages))meil.pw.edu.pl
