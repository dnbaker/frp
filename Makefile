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
		 -Wformat -Wcast-align -Wno-unused-function -Wunused-variable -Wno-ignored-qualifiers -Wsuggest-attribute=const \
        # -Wconversion -Werror -Wno-float-conversion
DBG:= # -DNDEBUG
OPT:= -O3 -funroll-loops -pipe -fno-strict-aliasing -march=native -fopenmp -DUSE_FASTRANGE \
      -funsafe-math-optimizations -ftree-vectorize \
        -DBOOST_NO_RTTI
OS:=$(shell uname)

EXTRA=
BLAS_LINKING_FLAGS?=
OPT:=$(OPT) $(FLAGS)
XXFLAGS=-fno-rtti
CBLASFILE?=cblas.h
BLAZEFLAGS= -DBLAZE_RANDOM_NUMBER_GENERATOR='::wy::WyHash<uint32_t, 8>' -DBLAZE_BLAS_MODE=1 \
    -DBLAZE_USE_BLAS_MATRIX_VECTOR_MULTIPLICATION=1 -DBLAZE_BLAS_INCLUDE_FILE='"$(CBLASFILE)"' \
     $(BLAS_LINKING_FLAGS)
CXXFLAGS=$(OPT) $(XXFLAGS) -std=$(STD) $(WARNINGS) -DRADEM_LUT $(EXTRA) $(BLAZEFLAGS)
CCFLAGS=$(OPT) -std=c11 $(WARNINGS)
LIB=-lz -pthread -lfftw3 -lfftw3l -lfftw3f -lstdc++fs -lsleef -llapack
LD=-L. -Lfftw-3.3.7/lib -Lvec/sleef/build/lib

OBJS=$(patsubst %.cpp,%.o,$(wildcard lib/*.cpp)) clhash/clhash.o
TEST_OBJS=$(patsubst %.cpp,%.o,$(wildcard test/*.cpp))
EXEC_OBJS=$(patsubst %.cpp,%.o,$(wildcard src/*.cpp)) $(patsubst %.cpp,%.fo,$(wildcard src/*.cpp))

clhash/clhash.o:
	cd clhash && make && cd ..

EX=$(patsubst src/%.fo,%f,$(EXEC_OBJS)) $(patsubst src/%.o,%,$(EXEC_OBJS))
BOOST_DIRS=math config random utility assert static_assert \
    integer type_traits mpl core preprocessor exception throw_exception \
    range iterator io predef concept_check detail lexical_cast \
    numeric_conversion functional array container move thread smart_ptr

SAN=-fsanitize=address -fsanitize=undefined

BOOST_INCS=$(patsubst %,-Iboost/%/include,$(BOOST_DIRS))


# If compiling with c++ < 17 and your compiler does not provide
# bessel functions with c++14, you must compile against boost.

INCLUDE=-I. -Iinclude -Ivec/blaze -Ithirdparty -Irandom/include/\
      -Ifftw-3.3.7/include -I vec/sleef/build/include/ $(BOOST_INCS) \
    -I/usr//local/Cellar/zlib/1.2.11/include -Ifastrange -Idistmat -Iaesctr \
        -Iinclude/frp -Iclhash/include # asdfnfkjhqefkjhdafs

ifdef BOOST_INCLUDE_PATH
INCLUDE += -I$(BOOST_INCLUDE_PATH)
endif

OBJS:=$(OBJS) vec/sleef/build/include/sleef.h fht.o FFHT/fast_copy.o

all: $(OBJS) $(EX) python
print-%  : ; @echo $* = $($*)

obj: $(OBJS) $(EXEC_OBJS)

HEADERS=$(wildcard include/frp/*.h)

fht.o: FFHT/fht.c
	cd FFHT && make fht.o && cp fht.o ..

HEADERS=$(wildcard include/frp/*.h)

test/%.o: test/%.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LD) $(OBJS) -c $< -o $@ $(LIB)

%.fo: %.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) -DFLOAT_TYPE=float $(DBG) $(INCLUDE) $(LD) -c $< -o $@ $(LIB)

%.o: %.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) -DFLOAT_TYPE=double $(DBG) $(INCLUDE) $(LD) -c $< -o $@ $(LIB)

%: src/%.cpp $(OBJS) fftw3.h $(HEADERS)
	$(CXX) $(CXXFLAGS) -DFLOAT_TYPE=double $(DBG) $(INCLUDE) $(LD) $(OBJS) $< -o $@ $(LIB)
pcatest: src/pcatest.cpp $(OBJS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -DFLOAT_TYPE=double $(DBG) $(INCLUDE) $(LD) $(OBJS) $< -o $@ $(LIB)
dcitest: src/dcitest.cpp $(OBJS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -DFLOAT_TYPE=double $(DBG) $(INCLUDE) $(LD) $(OBJS) $< -o $@ -lz -pthread -fopenmp -llapack -DTIME_ADDITIONS #$(SAN)
dcitestf: src/dcitest.cpp $(OBJS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -DFLOAT_TYPE=float $(DBG) $(INCLUDE) $(LD) $(OBJS) $< -o $@ -lz -pthread -fopenmp -llapack -DTIME_ADDITIONS #$(SAN)

%f: src/%.cpp $(OBJS) fftw3.h
	$(CXX) $(CXXFLAGS) -DFLOAT_TYPE=float $(DBG) $(INCLUDE) $(LD) $(OBJS) $< -o $@ $(LIB)

%.o: %.c
	$(CC) $(CCFLAGS) -Wno-sign-compare $(DBG) $(INCLUDE) $(LD) -c $< -o $@ $(LIB)

%.o: FFHT/%.c $(OBJS) fftw3.h
	+cd FFHT && make $@ && cp $@ .. && cd ..

fftw-3.3.7: fftw-3.3.7.tar.gz
	tar -zxvf fftw-3.3.7.tar.gz

fftw-3.3.7.exist: fftw-3.3.7.tar.gz
	tar -zxvf fftw-3.3.7.tar.gz && touch fftw-3.3.7.exist

PLATFORM_CONF_STR?=--enable-avx2

fftw3.h: fftw-3.3.7/lib/libfftw3l.a fftw-3.3.7/lib/libfftw3.a fftw-3.3.7/lib/libfftw3f.a
	cp fftw-3.3.7/api/fftw3.h .

python:
	cd py && make

fftw-3.3.7/lib/libfftw3.a: fftw-3.3.7.exist
	+cd fftw-3.3.7 &&\
	./configure $(PLATFORM_CONF_STR) --prefix=$$PWD && make && make install
fftw-3.3.7/lib/libfftw3f.a: fftw-3.3.7.exist fftw-3.3.7/lib/libfftw3.a
	+cd fftw-3.3.7 &&\
	./configure $(PLATFORM_CONF_STR) --prefix=$$PWD --enable-single && make && make install
fftw-3.3.7/lib/libfftw3l.a: fftw-3.3.7.exist fftw-3.3.7/lib/libfftw3f.a
	+cd fftw-3.3.7 &&\
	./configure --prefix=$$PWD --enable-long-double && make && make install && cp api/fftw3.h ..


tests: clean unit

unit: $(OBJS) $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(TEST_OBJS) $(LD) $(OBJS) -o $@ $(LIB)

vec/sleef/build: vec/sleef
	mkdir -p vec/sleef/build

vec/sleef/build/include/sleef.h: vec/sleef/build
	cd $< && cmake .. && make && cd ../..

sleef.h:vec/sleef/build/include/sleef.h
	cp vec/sleef/build/include/sleef.h sleef.h

clean:
	+rm -f $(EXEC_OBJS) $(OBJS) $(EX) $(TEST_OBJS) fftw3.h unit lib/*o frp/src/*o && cd FFHT && make clean && cd ..

mostlyclean: clean
