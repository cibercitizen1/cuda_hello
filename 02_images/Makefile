
include ../makefile_h

NVCC				= nvcc
UTIL_DIR			= ../80_util
IMAGE_DIR			= ../02_lib_images
INCLUDE_FLAG	= -I. -I$(UTIL_DIR) -I$(IMAGE_DIR) 
DIR_LIBS_FLAG	= -L$(UTIL_DIR) -L$(IMAGE_DIR)
LIBS			= -lmyutil -limages
CUDA_LIBS			= -lmyutilCuda -limagesCuda

compile: clean mainGeneral.cu
	make -C $(UTIL_DIR) cudalib
	make -C $(IMAGE_DIR) cudalib
	$(NVCC) $(INCLUDE_FLAG) mainGeneral.cu	$(DIR_LIBS_FLAG) $(CUDA_LIBS)

alinna: publish
	ssh -t bataller@alinna.dsic.upv.es "cd cuda/cuda_hello/02_images ; make download_force ; make run NVCC=/usr/local/cuda/bin/nvcc"
	# getimage
	scp bataller@alinna.dsic.upv.es:/tmp/corrected.bmp /media/sf_compartit


run: compile
	./a.out image=../90_data/patoC.bmp 
	mv corrected.bmp /tmp/corrected.bmp 

getimage:
	scp bataller@alinna.dsic.upv.es:/tmp/corrected.bmp /media/sf_compartit


clean:
	rm -f a.out
	rm -f corrected.bmp

