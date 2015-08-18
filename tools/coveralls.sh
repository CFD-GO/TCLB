lcov --compat-libtool --directory . --capture --output-file coverage.info
lcov --list coverage.info
echo coveralls-lcov coverage.info
