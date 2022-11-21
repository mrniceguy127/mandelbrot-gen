# mandelbrot-gen
Mandelbrot fractal generator.

## Status
Working on this whenever I feel like it. Plans include:

- Deep zooms -- This is pretty straight forward and I already have progress.
  - Just want to speed up the program. (Done?)
  - The final step!!! Then (finished) YouTube vid.
- Considering the state of the GPU market and my old card, time to switch to OpenCL?

## Dependencies
- CUDA -- You'll need an NVIDIA card. See the `cpu-threaded` branch for cpu threaded rendering (that branch will NOT be updated).
- gcc
- libpng

## Building
`make clean && make build`

## Running
`make run` -- All this does is run the executable that was outputted in the build dir.

Open the mandel.png file in your current working directory, and there it is!

## Optional CLI arguments
- `-i` - Specify number of iterates.
- `-x` - Offset from minimum x ("all the way" to the left in default settings) as a point on the mandelbrot coordinate plane and NOT PIXELS.
- `-n` - Offset from maximum x ("all the way" to the right in default settings) as a point on the mandelbrot coordinate plane and NOT PIXELS.
- `-y` - Offset from minimum y (all the way to the bottom in default settings) as a point on the mandelbrot coordinate plane and NOT PIXELS.
- `-m` - Offset from maximum y ("all the way" to the top in default settings) as a point on the mandelbrot coordinate plane and NOT PIXELS.

Example: `./build/mandel -i 10000 -x -2 -n 1 -y -1.5 -m 1.5`

![mandel.png](readme-assets/mandel.png "mandel.png")
![zoom.png](readme-assets/zoom.png "zoom.png")


## ZOOOOO
[OOOOOM](https://www.youtube.com/watch?v=bwpxdjsLIlw)


## License
[GNU GPL v3.0](LICENSE)
