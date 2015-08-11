include config.main.mk
include makefile.main

makefile.main:src/makefile.main.Rt src/*
	@tools/RT -I src/ -q -f $< -o $@

config.main.mk:
	touch config.main.mk

configure:src/configure.ac
	@echo "  AUTOCONF   $@"
	@autoconf -o $@ $< && rm -r autom4te.cache 2>/dev/null
	