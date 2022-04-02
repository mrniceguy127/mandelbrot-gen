# mandelbrot-gen
Mandelbrot fractal generator.

## Status
Working on this whenever I feel like it. Plans include:

- Other coloring related additions to the algorithm
- CLI interface (not difficult, just haven't done it yet for some reason)
- Deep zooms
- Parallel processing
  - For learning purposes and speed: Parallel processing using a GPU -- IN PROGRESS, WORKING, JUST TRYING TO FIND MORE OPTIMIZATION!!!
- Better algorithm

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
- `-z` - Zoom scale. e.g. `-z 2` to double zoom on both axes.
- `-x` - Offset from minimum x (all the way to the left in default settings) as a point on the mandelbrot coordinate plane and NOT PIXELS.
- `-y` - Offset from minimum y (all the way to the bottom in default settings) as a point on the mandelbrot coordinate plane and NOT PIXELS.

Example: `./build/mandel -i 10000 -x 1.001013 -y 0.817897 -z 80000`

![mandel.png](readme-assets/mandel.png "mandel.png")
![zoom.png](readme-assets/zoom.png "zoom.png")


## ZOOOOO
[OOOOOM](https://www.youtube.com/watch?v=bwpxdjsLIlw)


## License
[GNU GPL v3.0](LICENSE)
