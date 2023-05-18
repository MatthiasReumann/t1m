CCX = g++
CFLAGS = -std=c++17 -O3

INCLUDE += -I./include
INCLUDE += -I./src/external/marray/marray

all: clean build

build:
	${CCX} ${CFLAGS} test.cpp ${INCLUDE} -o test.o

clean:
	rm test.o ||: