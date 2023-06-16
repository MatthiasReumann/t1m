CCX = g++
CFLAGS = -std=c++17 -O3 -mavx2

INCLUDE += -I../include
INCLUDE += -I../src/external/marray/marray
INCLUDE += -I../src/external/blis/include/haswell

LIBS     = -L../src/external/blis/lib/haswell/ -lblis2

all: clean real complex

real:
	${CCX} ${CFLAGS} benchmark_real.cpp ${INCLUDE} ${LIBS} -o  $@.benchmark.o

complex:
	${CCX} ${CFLAGS} benchmark_complex.cpp ${INCLUDE} ${LIBS} -o  $@.benchmark.o

clean:
	rm *.o ||:
	