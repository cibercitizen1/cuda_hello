SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

LIB_DIRS= -L.. 
LIB_NAMES= -lutil 

CC = g++
CFLAGS = -I..

help:
	@ echo  help: make test, make clean

ensure_libs:
	make lib -C ..

.cpp.o:
	$(CC) $(CFLAGS) -c $<

test: $(OBJECTS) ensure_libs
	$(CC) $(LIB_DIRS) $(OBJECTS) -o run.test $(LIB_NAMES)

clean:
	rm -f $(OBJECTS) run.test
