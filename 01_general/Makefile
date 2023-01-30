
help:
	@echo "make send"
	@echo "make a.out"

a.out: mainGeneral.cu
	git pull
	rm a.out || true
	nvcc mainGeneral.cu	
	./a.out

send:
	git commit -a -m "automatic commit"
	git push

download:
	git fetch
	git reset --hard
	git merge
