.PHONY: build clean run

cc = gcc
main = main.c
out = mandle

build: $(main)
	$(cc) $(main) -o $(out)

clean:
	-\rm $(out)

run: $(out)
	./$(out)
