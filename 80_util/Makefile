SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
NOMBRE_LIB=util

CC = g++
CFLAGS = -I. 
AR=ar 

help:
	@ echo  help: make lib, make clean

.cpp.o:
	$(CC) $(CFLAGS) -c $<

lib: clean $(OBJECTS)
	$(AR) r lib$(NOMBRE_LIB).a $(OBJECTS)
	@ echo "library contents: ..................."
	@ $(AR) t lib$(NOMBRE_LIB).a 

clean:
	rm -f $(OBJECTS) *.a
