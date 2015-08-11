include config.main.mk
include makefile.main

makefile.main:src/makefile.main.Rt src/*
	@echo "Making makefile.main"
	@tools/RT -I src/ -q -f $< -o $@

config.main.mk: configure
	@echo
	@echo "   #########################################"
	@echo "   #        Run ./configure [...]          #"
	@echo "   #########################################"
	@echo
	@exit 1

configure:src/configure.ac
	@echo "  AUTOCONF   $@"
	@autoconf -o $@ $< && rm -r autom4te.cache 2>/dev/null
	