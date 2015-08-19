F=$(find -name "*.gcda")
if test -z "$F"
then
	echo coveralls: No GCDA files
	echo Exiting without error
	exit 0
fi
lcov --compat-libtool --directory . --capture --output-file coverage.info
lcov --remove coverage.info "/usr*" -o coverage.info
lcov --list coverage.info
coveralls-lcov coverage.info
