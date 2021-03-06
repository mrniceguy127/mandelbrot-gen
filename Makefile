.PHONY: build clean run

cc = gcc
main = main.c writepng.c
libs = png -lm

build_dir = ./build
out = $(build_dir)/mandel

build: $(main)
	mkdir -p $(build_dir)
	$(cc) $(main) -o $(out) -l $(libs)
	chmod +x $(out)

clean:
	-\rm $(out)
	-\rm -r $(build_dir)

run: $(out)
	$(out)
