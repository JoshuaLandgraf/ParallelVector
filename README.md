# ParallelVector

A C++ library / DSL for vector parallelism.

## build on OS X
clang++ -o example example.cpp -framework OpenCL -std=c++11

## run with
./example

## for extra OpenCL debgging info, run as
CL_LOG_ERRORS=stdout ./example
