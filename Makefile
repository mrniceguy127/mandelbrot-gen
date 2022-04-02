.PHONY: build clean run

cc = nvcc
main = main.cu writepng.cu
libs = -lpng
opts = --expt-relaxed-constexpr

build_dir = ./build
out = $(build_dir)/mandel

build: $(main)
	mkdir -p $(build_dir)
	$(cc) $(main) -o $(out) $(libs) $(opts)
	chmod +x $(out)

clean:
	-\rm $(out)
	-\rm -r $(build_dir)

run: $(out)
	$(out)
