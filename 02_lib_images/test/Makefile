SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
#OBJECTS = $(subst .cpp, .o, $(SOURCES))

NOMBRE_LIB=Imagenes

CC = g++
CFLAGS = -I..

help:
	@ echo  ayuda: make test, make clean

.cpp.o:
	$(CC) $(CFLAGS) -c $<

test: $(OBJECTS)
	(cd .. ; make lib)
	$(CC) -L.. $(OBJECTS) -l$(NOMBRE_LIB) -o run.test

clean:
	rm -f $(OBJECTS) 





