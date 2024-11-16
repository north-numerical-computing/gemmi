# Matrix–matrix multiply using integer Ozaki scheme

[![Testing](https://github.com/north-numerical-computing/gemmi/actions/workflows/run_cpp_tests.yml/badge.svg?branch=main)](https://github.com/north-numerical-computing/gemmi/actions/workflows/run_cpp_tests.yml)

## Dependencies

The recommended system to build this project is `CMake`. Compilation of the C++ example requires a g++ compiler that supports the C++20 standard. In order to build the MEX interface, MATLAB should be installed and available on the search path.

## Build

The C++ example program and the MEX interface can be compiled with:
```bash
mkdir build
cmake -B build
cd build && make
```
If the compilation is successful, these commands will produce the executable `build/example` and the MEX interface `build/gemmi.$(mexext)`, where `$(mexext)` is `mexa64` on Linux, `mexmaci64` or `mexmaca64` on MacOS , and `mexw64` on Windows. The file `build/gemmi.m` contains the documentation for the MATLAB `gemmi` function.

## Configuration

Additional options can be passed to `CMake` during the configuration stage. To use a compiler other than the default one, for example, one can use:
```bash
cmake -B build -D CMAKE_CXX_COMPILER=<preferred_compiler>
```
Here, `<preferred_compiler>` must be either an executable on the search path or the path to the chosen compiler.

## Notes on the MEX interface

Systems without `CMake` can use the function `mex/compile_mex.m` to compile the MEX interface. In MATLAB, this can be achieved by using the following commands:
```matlab
cd mex
compile_mex
```
The MEX interface and the corresponding documentation will be in the local directory (`mex/`).
