include config.main.mk
include makefile.main

makefile.main:src/makefile.main.Rt src/*
	@tools/RT -I src/ -q -f $< -o $@

config.main.mk:
	touch config.main.mk