CXX=g++-7

INCLUDES=-I`python3-config --includes` -I../pybind11/include -I../include -I.. -I../sleef/build/include
SUF=`python3-config --extension-suffix`
OBJS=$(patsubst %.cpp,%.cpython.so,$(wildcard *.cpp))


all: $(OBJS) python

python:
	python -c "import subprocess;import site; subprocess.check_call('cp jl.py "*`python3-config --extension-suffix`" %s' % site.getsitepackages()[0], shell=True)"

INC = ../fht.o ../fast_copy.o

MFLAGS=-msse4.1 -mavx2 -march=native

../%.o:
	cd .. && make $*.o

%.cpython.so: %.cpp ../fht.o ../fast_copy.o
	$(CXX) -undefined dynamic_lookup $(INCLUDES) -O3 -Wall $(MFLAGS) $(INC) -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` $< -o $*$(SUF) && \
    ln -fs $*$(SUF) $@

clean:
	rm -f *cpython.so
