SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
NOMBRE_LIB=Imagenes
#OBJECTS = $(subst .cpp, .o, $(SOURCES))

CC = g++
#CC = nvcc
CFLAGS = -I. 
AR=ar 

help:
	@ echo  ayuda: make lib, make cudalib, make clean

.cpp.o:
	$(CC) $(CFLAGS) -c $<

lib: clean $(OBJECTS)
	$(AR) r lib$(NOMBRE_LIB).a $(OBJECTS)
	@ echo "contenido de la biblioteca ..................."
	@ $(AR) t lib$(NOMBRE_LIB).a 

cudalib: clean 
	nvcc -c $(CFLAGS) $(SOURCES)
	$(AR) r lib$(NOMBRE_LIB)Cuda.a $(OBJECTS)
	@ echo "contenido de la biblioteca ..................."
	@ $(AR) t lib$(NOMBRE_LIB)Cuda.a 

clean:
	rm -f $(OBJECTS) *.a
