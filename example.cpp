// Example program that uses ParallelVector to do some simple computations

#include <iostream>
#include <vector>
#include "ParallelVector.hpp"

int main(int argc, char *argv[]) {
	try {
	PV::Vector<int> vec1(10,1);
	
	// arithmetic tests
	PV::Vector<int> vec2 = vec1 + vec1;
	printf("here\n");
	PV::Vector<int> vec4 = vec2 * vec2;
	printf("here\n");
	PV::Vector<int> vec3 = vec4 - vec1;
	printf("here\n");
	PV::Vector<int> vec6 = (vec4 * vec4 - vec4) / vec2;
	printf("here\n");
	PV::Vector<int> vec5 = (vec4 * vec3 - vec1) % vec6;
	printf("here\n");
	PV::Vector<int> vecI1(10);
	printf("here\n");
	for (int i = 0; i < vecI1.size(); ++i) vecI1[i] = i;
	printf("here\n");
	PV::Vector<int> vecI2 = vecI1++;
	printf("here\n");
	PV::Vector<int> vecI3 = --vecI1;
	printf("here\n");
	PV::Vector<int> vecI4 = ++vecI1;
	printf("here\n");
	PV::Vector<int> vecI5 = vec1 + vecI1--;
	printf("here\n");
	vecI5 -= vec1;
	printf("here\n");
	PV::Vector<int> vecI6 = (vecI5 += vec2);
	printf("here\n");
	vecI5 -= vec1;
	printf("here\n");
	
	printf("%d %d %d %d %d %d %d %d %d %d %d %d\n1 2 3 4 5 6 7 8 9 10 11 12\n", vec1[0], vec2[1], vec3[2], vec4[3], vec5[4], vec6[5], vecI1[7], vecI2[8], vecI3[9], vecI4[9], vecI5[9], vecI6[9]);
	printf("here\n");
	// comparison and logical tests
	PV::Vector<bool> vecB1 = vec1 == vec1;
	printf("here\n");
	PV::Vector<bool> vecB2 = vec1 != vec1;
	printf("here\n");
	PV::Vector<bool> vecB3 = vec2 > vec1;
	printf("here\n");
	PV::Vector<bool> vecB4 = vec2 < vec1;
	printf("here\n");
	PV::Vector<bool> vecB5 = vec3 >= vec2;
	printf("here\n");
	PV::Vector<bool> vecB6 = vec3 <= vec2;
	
	PV::Vector<bool> vecB7 = vecB1 && vecB2;
	PV::Vector<bool> vecB8 = vecB1 || vecB2;
	PV::Vector<bool> vecB9 = !vecB3;
	
	printf("%d %d %d %d %d %d %d %d %d\n1 0 1 0 1 0 0 1 0\n", vecB1.at(0), vecB2.at(1), vecB3.at(2), vecB4.at(3), vecB5.at(4), vecB6.at(5), vecB7.at(6), vecB8.at(7), vecB9.at(8));
	
	// bitwise tests
	PV::Vector<int> vec0 = vec1 ^ vec1;
	PV::Vector<int> vec01 = vec1 & vec1;
	PV::Vector<int> vec001 = vec01 | vec0;
	PV::Vector<int> vecNeg1 = ~vec0;
	
	printf("%d %d %d %d\n0 0 1 -1\n", vec0[0], vec01[1], vec001[2], vecNeg1[3]);
	} catch (char const * error) {
		printf("[Error] %s\n", error);
		exit(1);
	}
}