# ParallelVector

A C++ library / DSL for vector parallelism through OpenCL.

## Installing and building

#### Download ParallelVector

  git clone https://github.com/JoshuaLandgraf/ParallelVector.git

### Install OpenCL and Build

#### OS X

Nothing to do, OpenCL is already provided by Apple. Can use the Makefile provided.

#### Windows and Linux

OpenCL drivers are easily available from [Intel](https://software.intel.com/en-us/articles/opencl-drivers), [AMD](http://support.amd.com/en-us/kb-articles/Pages/OpenCL2-Driver.aspx), and [Nvidia](http://www.nvidia.com/Download/index.aspx?lang=en-us). [This site](https://www.fixstars.com/en/opencl/book/OpenCLProgrammingBook/first-opencl-program/) has instructions on where the files are installed and how to include them in your build system.

### Running and Debugging Programs

Examplpe programs can be run (on OS X and Linux) with

  ./test
  ./kmeans
  ./keys

For extra OpenCL debgging info, adding `CL_LOG_ERRORS=stdout` before running the command can reveal more specific errors about which part of the implmenetation is breaking and what went wrong.
