clean:
	rm a.out

all: a.out

a.out: mainGeneral.cu
	git pull
	rm a.out || true
	nvcc mainGeneral.cu	
	./a.out

send:
	git commit -a -m "automatic commit"
	git push
