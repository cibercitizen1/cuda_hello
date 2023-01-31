CPP_SOURCES 	= $(wildcard *.cpp)
CUDA_SOURCES 	= $(wildcard *.cu)
OBJECTS 		= $(CPP_SOURCES:.cpp=.o) $(CUDA_SOURCES:.cu=.o)
LIB_NAME		= myutil

CC 		= g++
NVCC 	= nvcc
INCLUDE = -I. 
AR		= ar 

help:
	@ echo  help: make cudalib, make lib, make clean

lib: clean 
	$(CC) $(INCLUDE) -c $(CPP_SOURCES)
	$(AR) r lib$(LIB_NAME).a $(OBJECTS)
	@ echo "library contents: lib ..................."
	@ $(AR) t lib$(LIB_NAME).a 

cudalib: clean 
	$(NVCC)  $(INCLUDE) -c $(CPP_SOURCES) $(CUDA_SOURCES)
	$(AR) r lib$(LIB_NAME)Cuda.a $(OBJECTS)
	@ echo "library contents: cudalib ..................."
	@ $(AR) t lib$(LIB_NAME)Cuda.a 

clean:
	rm -f $(OBJECTS) *.a
