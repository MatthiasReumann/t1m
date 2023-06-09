CCX = g++
CFLAGS = -std=c++17 -O3

INCLUDE += -I./include
INCLUDE += -I./src/external/marray/marray
INCLUDE += -I./src/external/blis/include/haswell

LIBS     = -L./src/external/blis/lib/haswell/ -lblis2

all: clean build

build:
	${CCX} ${CFLAGS} main.cpp ${INCLUDE} ${LIBS} -mavx2 -o tfctc.o

debug:
	${CCX} -std=c++17 -g main.cpp ${INCLUDE} ${LIBS} -mavx2 -o debug.o

clean:
	rm debug.o tfctc.o ||:
