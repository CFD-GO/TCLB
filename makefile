include makefile.main

makefile.main:src/makefile.main.Rt src/*
	@tools/RT -q -f $< -o $@
