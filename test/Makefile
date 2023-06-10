CCX = g++
CFLAGS = -std=c++17 -O3

INCLUDE += -I../include
INCLUDE += -I../src/external/marray/marray
INCLUDE += -I../src/external/blis/include/haswell

LIBS     = -lblis -lhptt

all: clean complex real

complex:
	${CCX} ${CFLAGS} test_complex.cpp ${INCLUDE} ${LIBS} -o $@.test.o

real:
	${CCX} ${CFLAGS} test_real.cpp ${INCLUDE} ${LIBS} -o $@.test.o

debug:
	${CCX} -std=c++17 test.cpp ${INCLUDE} ${LIBS} -o $@.o

clean:
	rm *.o ||: