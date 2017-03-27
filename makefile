ifneq ($(MAKECMDGOALS),configure)
	include config.main.mk
	include makefile.main
endif

makefile.main:src/makefile.main.Rt src/* models/* models/*/*
	@echo "  RT         $@"
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
	@autoconf --warnings=error -o $@ $< && rm -r autom4te.cache 2>/dev/null
	