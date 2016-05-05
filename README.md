# ParallelVector

A C++ library / DSL for vector parallelism through OpenCL.

## Installing and building

#### Download ParallelVector
```
git clone https://github.com/JoshuaLandgraf/ParallelVector.git
```
#### Install OpenCL and Build

##### OS X

Nothing to do, OpenCL is already provided by Apple. Can use the Makefile provided.

##### Windows and Linux

OpenCL drivers are easily available from [Intel](https://software.intel.com/en-us/articles/opencl-drivers), [AMD](http://support.amd.com/en-us/kb-articles/Pages/OpenCL2-Driver.aspx), and [Nvidia](http://www.nvidia.com/Download/index.aspx?lang=en-us). [This site](https://www.fixstars.com/en/opencl/book/OpenCLProgrammingBook/first-opencl-program/) has instructions on where the files are installed and how to include them in your build system.

#### Running and Debugging Programs

Examplpe programs can be run (on OS X and Linux) with
```
./test
./kmeans
./keys
```

For extra OpenCL debgging info, adding `CL_LOG_ERRORS=stdout` before running the command can reveal more specific errors about which part of the implmenetation is breaking and what went wrong.

##Coding in ParallelVector

#### Constructors

| Constructor   | Code                               | Description                                       |
|---------------|------------------------------------|---------------------------------------------------|
| Default       | `PV::Vector<T>()`                  | Uninitialized Vector                              |
| Allocation    | `PV::Vector<T>(length)`            | Space for `length` elements                       |
| Fill          | `PV::Vector<T>(length, value)`     | Initializes all `length` elements to `value`      |
| Range         | `PV::Vector<T>(begin, end)`        | Initializes Vector with data from iterators       |
| Range         | `PV::Vector<T>(pointer, length)`   | Initializes Vector with data from pointer         |
| Copy          | `PV::Vector<T>(Vector)`            | Initializes Vector with data from Vector          |
| `std::vector` | `PV::Vector<T>(vector)`            | Initializes Vector with data from a `std::vector` |
| Move          | `PV::Vector<T>(std::move(Vector))` | Moves Vector into constructed Vector              |
| Indices       | `PV::indices_Vector<T>(length)`    | Returns Vector with values 0 through `length-1`   |

#### Operators

| Operator                             | Description                                                                                   | Special Notes                                            |
|--------------------------------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------|
| `Vector[index]`                      | Returns value at index                                                                        | Value is not a reference                                 |
| `Vector.front()`                     | Returns first element in Vector                                                               | Value is not a reference                                 |
| `Vector.back()`                      | Returns last element in Vector                                                                | Value is not a reference                                 |
| `Vector.get(index)`                  | Returns value at index                                                                        | Value is not a reference                                 |
| `Vector.get(index, pointer, length)` | Copies `length` elements starting at `index`to `pointer`                                      |                                                          |
| `Vector.get(index, vector)`          | Copies `length` elements starting at `index`into `std::vector`                                |                                                          |
| `Vector.get(index, begin, end)`      | Copies elements starting at `index` into iterators                                            |                                                          |
| `Vector.set(index, value)`           | Sets element `index` to `value`                                                               |                                                          |
| `Vector.set(index, pointer, length)` | Copies `length` elements from `pointer` into Vector                                           | `Index` specifies where to start in Vector               |
| `Vector.set(index, vector)`          | Copies `length` elements from `std::vector` into Vector                                       | `Index` specifies where to start in Vector               |
| `Vector.set(index, begin, end)`      | Copies `length` elements from iterators into Vector                                           |                                                          |
| `Vector.push_back(values)`           | Adds `value` to the end of the Vector                                                         | Changes Vector size Potentially very expensive operation |
| `Vector.pop_back()`                  | Removes last element from Vector                                                              | Changes Vector size                                      |
| `Vector + Vector`                    | Self-explanatory                                                                              |                                                          |
| `Vector += Vector`                   | Self-explanatory                                                                              | No return value                                          |
| `Vector - Vector`                    | Self-explanatory                                                                              |                                                          |
| `Vector -= Vector`                   | Self-explanatory                                                                              | No return value                                          |
| `Vector * Vector`                    | Self-explanatory                                                                              |                                                          |
| `Vector *= Vector`                   | Self-explanatory                                                                              | No return value                                          |
| `Vector / Vector`                    | Self-explanatory                                                                              |                                                          |
| `Vector /= Vector`                   | Self-explanatory                                                                              | No return value                                          |
| `Vector % Vector`                    | Self-explanatory                                                                              |                                                          |
| `Vector %= Vector`                   | Self-explanatory                                                                              | No return value                                          |
| `Vector++`                           | Self-explanatory                                                                              | Returns copy of original Vector                          |
| `++Vector`                           | Self-explanatory                                                                              |                                                          |
| `Vector--`                           | Self-explanatory                                                                              | Returns copy of original Vector                          |
| `--Vector`                           | Self-explanatory                                                                              |                                                          |
| `-Vector`                            | Returns negated Vector                                                                        |                                                          |
| `Vector == Vector`                   | Self-explanatory                                                                              |                                                          |
| `Vector != Vector`                   | Self-explanatory                                                                              |                                                          |
| `Vector > Vector`                    | Self-explanatory                                                                              |                                                          |
| `Vector >= Vector`                   | Self-explanatory                                                                              |                                                          |
| `Vector < Vector`                    | Self-explanatory                                                                              |                                                          |
| `Vector <= Vector`                   | Self-explanatory                                                                              |                                                          |
| `Vector && Vector`                   | Self-explanatory                                                                              | No short-circuiting                                      |
| `Vector || Vector`                   | Self-explanatory                                                                              | No short-circuiting                                      |
| `!Vector`                            | Returns logical not of original vector                                                        |                                                          |
| `Vector & Vector`                    | Self-explanatory                                                                              |                                                          |
| `Vector &= Vector`                   | Self-explanatory                                                                              | No return value                                          |
| `Vector | Vector`                    | Self-explanatory                                                                              |                                                          |
| `Vector |= Vector`                   | Self-explanatory                                                                              | No return value                                          |
| `Vector ^ Vector`                    | Self-explanatory                                                                              |                                                          |
| `Vector ^= Vector`                   | Self-explanatory                                                                              | No return value                                          |
| `~Vector`                            | Returns bit-flipped Vector                                                                    |                                                          |
| `Vector << Vector`                   | Self-explanatory                                                                              |                                                          |
| `Vector <<= Vector`                  | Self-explanatory                                                                              | No return value                                          |
| `Vector >>= Vector`                  | Self-explanatory                                                                              |                                                          |
| `Vector >>= Vector`                  | Self-explanatory                                                                              | No return value                                          |
| `Vector.choose(Vector, Vector)`      | Returns ternary operator of the Vectors Think of `A.choose(B,C)` as `A ? B : C`               |                                                          |
| `Vector.sum()`                       | Returns sum of elements in Vector                                                             | Requires non-empty Vector                                |
| `Vector.product()`                   | Returns product of elements in Vector                                                         | Requires non-empty Vector                                |
| `Vector.filterBy(Vector)`            | Returns elements in first Vector whose corresponding elements in the second Vector are `true` | Result Vector can be empty                               |
| `Vector.rotateBy(rotation)`          | Moves elements to the right by `rotation`                                                     | Negative values rotate left Elements wrap around         |

#### Misc Methods

| Method                   | Description                                                         | Special Notes                                                                   |
|--------------------------|---------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `Vector.size()`          | Returns number of elements in Vector                                |                                                                                 |
| `Vector.resize(length)`  | Changes number of elements in Vector                                | Elements not initialized to any value when growing Vector                       |
| `Vector.reserve(length)` | Guarantees Vector has allocated enough space for `length` elements  | Does not change Vector size but is useful for improving performance with `push_back()` |
