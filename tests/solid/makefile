
SRC = ../../src/
CXXFLAGS += -I$(SRC)
CXXFLAGS += -Wall -Wno-unused-but-set-variable -Wno-unused-variable
CXXFLAGS += -Werror -Wno-unknown-warning-option

all: main

run: main
	./main

main.o: main.cpp $(SRC)/SolidTree.hpp $(SRC)/SolidTree.h $(SRC)/SolidGrid.h $(SRC)/SolidGrid.hpp $(SRC)/SolidAll.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

main: main.o
	$(CXX) -o $@ $^
