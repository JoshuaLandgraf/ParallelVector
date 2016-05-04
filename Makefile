all: tests kmeans keys

tests: tests.cpp ParallelVector.hpp cl.hpp
	clang++ -o tests tests.cpp -framework OpenCL -std=c++11 -O3

kmeans: kmeans.cpp ParallelVector.hpp cl.hpp
	clang++ -o kmeans kmeans.cpp -framework OpenCL -std=c++11 -O3

keys: keys.cpp ParallelVector.hpp cl.hpp
	clang++ -o keys keys.cpp -framework OpenCL -std=c++11 -O3
