![TCLB Solver Header](https://raw.githubusercontent.com/CFD-GO/documents/master/assets/header.png)

TCLB Solver - Codespace (Small)
===

**Configure**
```bash
make configure
./configure --disable-cuda --with-openmp --enable-cpu-layout
```

**Compile**
```bash
make d2q9
```

**Run**
```bash
CLB/d2q9/main example/runr/karman.xml
```
