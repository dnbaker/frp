CXX=g++-7

INCLUDES=-I`python3-config --includes` -I../pybind11/include -I../include -I..
OBJS=$(patsubst %.cpp,%.cpython-36m-darwin.so,$(wildcard *.cpp))

all: $(OBJS)

MFLAGS=-msse4.1 -mavx2 -march=native

%.cpython-36m-darwin.so: %.cpp
	$(CXX) -undefined dynamic_lookup $(INCLUDES) -O3 -Wall $(MFLAGS) -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` $< -o $@
