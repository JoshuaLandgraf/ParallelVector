// Simple key-breaking code using ParallelVector

#include <iostream>
#include <math.h> 
#include "ParallelVector.hpp"

int main(int argc, char *argv[]) {
	// create key using "large" prime numbers
	const size_t prime_1 = 32452841;
	const size_t prime_2 = 32452843;
	const size_t key = prime_1 * prime_2;
	
	// determine bound on numbers to search through
	const size_t bound = (size_t)(sqrt((double)key) + 1);
	
	// initialize ParallelVector constants
	PV::Vector<size_t> zeros(bound-1, 0);
	PV::Vector<size_t> twos(bound-1, 2);
	PV::Vector<size_t> keys(bound-1, key);
	
	// get all numbers from 2 to bound
	PV::Vector<size_t> nums = PV::indices_Vector<size_t>(bound-1) + twos;
	
	// find number that evenly divides key via brute force
	PV::Vector<size_t> result = nums.filterBy(keys % nums == zeros);
	
	// generate the original primes
	const size_t result_prime_1 = result[0];
	const size_t result_prime_2 = key / result_prime_1;
	
	printf("Original primes: %zu, %zu\n", prime_1, prime_2);
	printf("Found primes: %zu, %zu\n", result_prime_1, result_prime_2);
}