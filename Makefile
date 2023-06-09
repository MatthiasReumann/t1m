CCX = g++
CFLAGS = -std=c++17 -O3

INCLUDE += -I./include
INCLUDE += -I./src/external/marray/marray
INCLUDE += -I./src/external/blis/include/haswell

LIBS     = -L./src/external/blis/lib/haswell/ -lblis2

all: clean build

build:
	${CCX} ${CFLAGS} test.cpp ${INCLUDE} ${LIBS} -mavx2 -o test.o

debug:
	${CCX} -std=c++17 -g test.cpp ${INCLUDE} ${LIBS} -mavx2 -o debug.o

clean:
	rm test.o ||:
