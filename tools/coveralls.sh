F=$(find -name "*.gcda")
if test -z "$F"
then
	echo coveralls: No GCDA files
	echo Exiting without error
	exit 0
fi
lcov --compat-libtool --directory . --capture --output-file coverage.info
lcov --list coverage.info
echo coveralls-lcov coverage.info
