CCX = g++
CFLAGS = -std=c++17 -O3

INCLUDE += -I../include
INCLUDE += -I../src/external/marray/marray
INCLUDE += -I../src/external/blis/include/haswell

LIBS     = -L../src/external/blis/lib/haswell/ -lblis2

all: clean build

build:
	${CCX} ${CFLAGS} benchmark.cpp ${INCLUDE} ${LIBS} -o benchmark.o

clean:
	rm test.o ||: