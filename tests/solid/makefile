
SRC = ../../src/
CXXFLAGS += -I$(SRC)
CXXFLAGS += -Wall -Wno-unused-but-set-variable -Wno-unused-variable
CXXFLAGS += -Werror -Wno-unknown-warning-option
CXXFLAGS += $(ADD_FLAGS)

all: main

run: main
	./main

main.o: main.cpp $(SRC)/SolidTree.h $(SRC)/SolidGrid.h $(SRC)/SolidAll.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

main: main.o
	$(CXX) $(ADD_FLAGS) -o $@ $^
