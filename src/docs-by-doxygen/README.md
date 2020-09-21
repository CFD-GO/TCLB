
# Usage

Doxygen is able to generate documentation which provide some insight to the code structure.

## Install doxygen (Ubuntu 18.04)

You can download doxygen from the official website <http://www.doxygen.nl/index.html>.

Or install via command line:

```.sh
sudo apt-get update -y
sudo apt-get install -y doxygen
sudo apt-get install -y graphviz

```

## Generate documentation

To generate the documentation for `d2q9` model, run the following command from the main TCLB folder

```.sh
make d2q9
make d2q9/docs
```

### Notice

Under the hood, the `doxygen CLB/d2q9/docs-by-doxygen/conf.doxygen` command is called.
The input path parameter from `conf.doxygen` points to the current directory (i.e. `d2q9`):

```.sh
INPUT                  =    #  Note: If this tag is empty the current directory is searched.
```

## Read documentation

Next, open `CLB/d2q9/docs-by-doxygen/output/html/index.html` in your browser.
