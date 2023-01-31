SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
#OBJECTS = $(subst .cpp, .o, $(SOURCES))

LIB_NAME=images

CC = g++
CFLAGS = -I.. -I../../80_util

help:
	@ echo  ayuda: make test, make clean

.cpp.o:
	$(CC) $(CFLAGS) -c $<

test: $(OBJECTS)
	(cd .. ; make lib)
	$(CC) -L.. $(OBJECTS) -l$(LIB_NAME) -o run.test

clean:
	rm -f $(OBJECTS) 
	rm -f kk*





