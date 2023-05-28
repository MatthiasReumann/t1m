CCX = g++
CFLAGS = -std=c++17 -O3

INCLUDE += -I./include
INCLUDE += -I./src/external/marray/marray
INCLUDE += -I./src/external/blis/include/haswell

LIBS     = -L./src/external/blis/lib/haswell/ -lblis2

all: clean build

build:
	${CCX} ${CFLAGS} test.cpp ${INCLUDE} ${LIBS} -o test.o

clean:
	rm test.o ||: