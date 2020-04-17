
# Usage

Doxygen is able to generate documentation which provide some insight to the code structure.

## Install doxygen (Ubuntu 18.04)

You can download doxygen from the official website <http://www.doxygen.nl/index.html>.

Or install via command line:

```.sh
sudo apt-get update -y
sudo apt-get install -y doxygen
```

## Generate documentation

To generate the documentation run the following command from the main TCLB folder

```.sh
doxygen docs-by-doxygen/DoxygenConfig
```

## Read documentation

Next, open `docs-by-doxygen/doxygen/html/index.html` in your browser.
