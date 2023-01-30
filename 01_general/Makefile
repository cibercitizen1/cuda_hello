
include ../makefile_h

help:
	@echo "make send"
	@echo "make a.out"

a.out: mainGeneral.cu
	git pull
	rm a.out || true
	nvcc mainGeneral.cu	
	./a.out
