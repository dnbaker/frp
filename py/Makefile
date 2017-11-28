CXX=g++-7

SUF=`python3-config --extension-suffix`
OBJS=$(patsubst %.cpp,%.cpython.so,$(wildcard *.cpp))


all: $(OBJS) python

python:
	python -c "import subprocess;import site; subprocess.check_call('cp jl.py "*`python3-config --extension-suffix`" %s' % site.getsitepackages()[0], shell=True)"

INC = ../fht.o ../fast_copy.o

BOOST_DIRS=math config random utility assert static_assert \
    integer type_traits mpl core preprocessor exception throw_exception \
    range iterator io predef concept_check detail lexical_cast \
    numeric_conversion functional array container move thread smart_ptr

BOOST_INCS=$(patsubst %,-I../%/include,$(BOOST_DIRS))
INCLUDES=-I`python3-config --includes` -I../pybind11/include -I../include -I.. -I../sleef/build/include $(BOOST_INCS)


MFLAGS=-msse4.1 -mavx2 -march=native

../%.o:
	cd .. && make $*.o

%.cpython.so: %.cpp ../fht.o ../fast_copy.o
	$(CXX) -undefined dynamic_lookup $(INCLUDES) -O3 -Wall $(MFLAGS) $(INC) -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` $< -o $*$(SUF) && \
    ln -fs $*$(SUF) $@

clean:
	rm -f *cpython.so
