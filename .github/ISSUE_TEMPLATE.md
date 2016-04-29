I found that ... is not working

Please fill:

My configuration:
```bash
./configure --with-some-weird-options
```

I was running with CPU/GPU PARALLEL/SERIAL CUDA 7.0/...

I was running with:
```bash
mpirun -np 3 CLB/d2q9_weird/main some.weird.example.xml
```

The xml file looks like this:
```xml
<CLBConfig version="4.9">
	<Weird is="getting" even="weirder"/>
</CLBConfig>
```

I get this error:
```
ERROR: This is just too weird
```
