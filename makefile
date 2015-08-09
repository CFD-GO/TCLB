include makefile.main

makefile.main:src/makefile.main.Rt src/*
	@tools/RT -I src/ -q -f $< -o $@
