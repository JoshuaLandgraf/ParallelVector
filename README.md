# ParallelVector

Nothing much right now, but this repo will hold the code for my parallel C++ vector implementation.

## build on OS X
clang++ -o example example.cpp -framework OpenCL -std=c++11

## run with
./example

## for extra OpenCL debgging info, run as
CL_LOG_ERRORS=stdout ./example