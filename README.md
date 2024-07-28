# Integer matrix–matrix multiplication using Ozaki scheme

## Build

The recommended system to build this project is `CMake`. The C++ example program and the MEX interface can be compiled with:
```bash
mkdir build
cmake -B build
cd build && make
```
If the compilation is successful, the C++ example will be the executable `build/example`. The MEX interface will be named `build/gemmi.$(mexext)`, where `$(mexext)` is `mexa64` on Linux, `mexmaci64` or `mexmaca64` on MacOS , and `mexw64` on Windows. The file `build/gemmi.m` will contain the documentation for the MATLAB `gemmi` function.

## Notes on the MEX interface

Systems without `CMake` can use the function `mex/compile_mex.m` to compile the
MEX interface. In MATLAB, this can be achieved by using the following commands:
```matlab
cd mex
compile_mex
```
The MEX interface and the corresponding documentation will be in the local directory (`mex/`).
