
include ../makefile_h

compile: mainGeneral.cu
	nvcc mainGeneral.cu	

a.out: mainGeneral.cu
	git pull
	rm a.out || true
	nvcc mainGeneral.cu	
	./a.out
