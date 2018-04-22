.PHONY=all tests clean obj python
ifndef CXX
CXX=g++
endif
ifndef CC
CC=gcc
endif
ifndef STD
STD=c++17
endif
WARNINGS=-Wall -Wextra -Wno-char-subscripts \
		 -Wpointer-arith -Wwrite-strings -Wdisabled-optimization \
		 -Wformat -Wcast-align -Wno-unused-function -Wunused-variable -Wno-ignored-qualifiers# -Wconversion -Werror -Wno-float-conversion
DBG:= # -DNDEBUG
OPT:= -O3 -funroll-loops -pipe -fno-strict-aliasing -mavx2 -march=native -fopenmp -DUSE_FASTRANGE \
      -funsafe-math-optimizations -ftree-vectorize
OS:=$(shell uname)

EXTRA=
OPT:=$(OPT) $(FLAGS)
XXFLAGS=-fno-rtti
CXXFLAGS=$(OPT) $(XXFLAGS) -std=$(STD) $(WARNINGS) -DRADEM_LUT $(EXTRA)
CCFLAGS=$(OPT) -std=c11 $(WARNINGS)
LIB=-lz -pthread -lfftw3 -lfftw3l -lfftw3f -lstdc++fs -lsleef -llapack
LD=-L. -Lfftw-3.3.7/lib -Lvec/sleef/build/lib

OBJS=$(patsubst %.cpp,%.o,$(wildcard lib/*.cpp))
TEST_OBJS=$(patsubst %.cpp,%.o,$(wildcard test/*.cpp))
EXEC_OBJS=$(patsubst %.cpp,%.o,$(wildcard src/*.cpp)) $(patsubst %.cpp,%.fo,$(wildcard src/*.cpp))

EX=$(patsubst src/%.fo,%f,$(EXEC_OBJS)) $(patsubst src/%.o,%,$(EXEC_OBJS))
BOOST_DIRS=math config random utility assert static_assert \
    integer type_traits mpl core preprocessor exception throw_exception \
    range iterator io predef concept_check detail lexical_cast \
    numeric_conversion functional array container move thread smart_ptr

BOOST_INCS=$(patsubst %,-Iboost/%/include,$(BOOST_DIRS))


# If compiling with c++ < 17 and your compiler does not provide
# bessel functions with c++14, you must compile against boost.

INCLUDE=-I. -Iinclude -Ivec/blaze -Ithirdparty -Irandom/include/ -Ifftw-3.3.7/include -I vec/sleef/build/include/ $(BOOST_INCS) -I/usr//local/Cellar/zlib/1.2.11/include

ifdef BOOST_INCLUDE_PATH
INCLUDE += -I$(BOOST_INCLUDE_PATH)
endif

OBJS:=$(OBJS) fht.o fast_copy.o  vec/sleef/build/include/sleef.h

all: $(OBJS) $(EX) python
print-%  : ; @echo $* = $($*)

obj: $(OBJS) $(EXEC_OBJS)

test/%.o: test/%.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LD) $(OBJS) -c $< -o $@ $(LIB)

%.fo: %.cpp
	$(CXX) $(CXXFLAGS) -DFLOAT_TYPE=float $(DBG) $(INCLUDE) $(LD) -c $< -o $@ $(LIB)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -DFLOAT_TYPE=double $(DBG) $(INCLUDE) $(LD) -c $< -o $@ $(LIB)

%: src/%.cpp $(OBJS) fftw3.h
	$(CXX) $(CXXFLAGS) -DFLOAT_TYPE=double $(DBG) $(INCLUDE) $(LD) $(OBJS) $< -o $@ $(LIB)

%f: src/%.cpp $(OBJS) fftw3.h
	$(CXX) $(CXXFLAGS) -DFLOAT_TYPE=float $(DBG) $(INCLUDE) $(LD) $(OBJS) $< -o $@ $(LIB)

%.o: %.c
	$(CC) $(CCFLAGS) -Wno-sign-compare $(DBG) $(INCLUDE) $(LD) -c $< -o $@ $(LIB)

%.o: FFHT/%.c
	cd FFHT && make $@ && cp $@ .. && cd ..

fftw-3.3.7: fftw-3.3.7.tar.gz
	tar -zxvf fftw-3.3.7.tar.gz

fftw3.h: fftw-3.3.7
	cd fftw-3.3.7 && \
	./configure --enable-avx2 --prefix=$$PWD && make && make install && \
	./configure --prefix=$$PWD --enable-long-double && make && make install &&\
	./configure --enable-avx2 --prefix=$$PWD --enable-single && make && make install &&\
	cp api/fftw3.h .. && cd ..

python:
	cd py && make


tests: clean unit

unit: $(OBJS) $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(TEST_OBJS) $(LD) $(OBJS) -o $@ $(LIB)

vec/sleef/build: vec/sleef
	mkdir -p vec/sleef/build

vec/sleef/build/include/sleef.h: vec/sleef/build
	cd $< && cmake .. && make && cd ../..

clean:
	rm -f $(EXEC_OBJS) $(OBJS) $(EX) $(TEST_OBJS) unit lib/*o frp/src/*o && cd FFHT && make clean && cd ..

mostlyclean: clean
