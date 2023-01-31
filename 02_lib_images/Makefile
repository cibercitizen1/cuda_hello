SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
LIB_NAME=images
#OBJECTS = $(subst .cpp, .o, $(SOURCES))

CC 		= g++
NVCC 	= nvcc
INCLUDE_FLAGS	= -I. -I../80_util
AR		= ar 

help:
	@ echo  make lib, make cudalib, make clean

lib: clean 
	$(CC) -c $(INCLUDE_FLAGS) $(SOURCES)
	$(AR) r lib$(LIB_NAME).a $(OBJECTS)
	@ echo "library contents lib ..................."
	@ $(AR) t lib$(LIB_NAME).a 

cudalib: clean 
	$(NVCC) -c $(INCLUDE_FLAGS) $(SOURCES)
	$(AR) r lib$(LIB_NAME)Cuda.a $(OBJECTS)
	@ echo "library contents cudalib ..................."
	@ $(AR) t lib$(LIB_NAME)Cuda.a 

clean:
	rm -f $(OBJECTS) *.a
