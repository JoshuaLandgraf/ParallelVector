// Example program that tests various ParallelVector operations

#include <iostream>
#include <vector>
#include <assert.h>
#include "ParallelVector.hpp"

int main(int argc, char *argv[]) {
	try {
		const unsigned test_size = 25000000;
		
		// test constructors, getters, and setters
		{
			std::vector<int> nums(test_size, 1);
			std::vector<int> nums2(test_size);
			
			// constructors
			PV::Vector<int> test1 = PV::Vector<int>();
			PV::Vector<int> test2(test_size);
			PV::Vector<int> test3(test_size, 0);
			assert(test3[0] == 0);
			PV::Vector<int> test4(nums.begin(), nums.end());
			assert(test4[1] == 1);
			PV::Vector<int> test5(&nums[0], nums.size());
			assert(test5[2] == 1);
			PV::Vector<int> test6(test5);
			assert(test6[3] == 1);
			PV::Vector<int> test7(nums);
			PV::Vector<int> test8(test7);
			assert(test8[4] == 1);
			PV::Vector<int> test9;
			test9 = test8;
			assert(test9[5] == 1);
			PV::Vector<int> test10(std::move(test9));
			assert(test10[6] == 1);
			PV::Vector<int> indices = PV::indices_Vector<int>(test_size);
			assert(indices[0] == 0);
			assert(indices[1] == 1);
			assert(indices.back() == indices.size()-1);
			
			// getters
			assert(test3.get(0) == 0);
			test4.get(0, nums2.begin(), nums2.begin()+1);
			assert(nums2[0] == 1);
			test4.get(1, &nums2[1], 1);
			assert(nums2[1] == 1);
			test4.get(0, nums2);
			assert(nums2[2] == 1);
			
			// setters
			PV::Vector<int> test11(test_size);
			test11.set(0, 1);
			assert(test11[0] == 1);
			test11.set(1, nums.begin(), nums.begin()+1);
			assert(test11[1] == 1);
			test11.set(2, &nums[0], 1);
			assert(test11[2] == 1);
			test11.set(0, nums);
			assert(test11[3] == 1);
		}
		
		// test operations on numbers
		{
			PV::Vector<int> ones(test_size, 1);
			PV::Vector<int> twos(test_size, 2);
			PV::Vector<int> threes(test_size, 3);
			PV::Vector<int> eights(test_size, 8);
			
			// arithmetic
			PV::Vector<int> test1 = ones + twos;
			assert(test1[0] == 3);
			PV::Vector<int> test2 = threes - twos;
			assert(test2[1] == 1);
			PV::Vector<int> test3 = twos * threes;
			assert(test3[2] == 6);
			PV::Vector<int> test4 = eights / twos;
			assert(test4[3] == 4);
			PV::Vector<int> test5 = threes % twos;
			assert(test5[4] == 1);
			PV::Vector<int> test6 = -ones;
			assert(test6[5] == -1);
			PV::Vector<int> test7 = ones;
			PV::Vector<int> test7_2 = test7++;
			assert(test7[6] == 2);
			assert(test7_2[6] == 1);
			PV::Vector<int> test8 = ones; ++test8;
			assert(test8[7] == 2);
			PV::Vector<int> test9 = ones;
			PV::Vector<int> test9_2 = test9--;
			assert(test9[8] == 0);
			assert(test9_2[8] == 1);
			PV::Vector<int> test10 = ones; --test10;
			assert(test10[9] == 0);
			PV::Vector<int> test11 = ones; test11 += ones;
			assert(test11[10] == 2);
			PV::Vector<int> test12 = ones; test12 -= ones;
			assert(test12[11] == 0);
			PV::Vector<int> test13 = twos; test13 *= threes;
			assert(test13[12] == 6);
			PV::Vector<int> test14 = eights; test14 /= twos;
			assert(test14[13] == 4);
			
			// comparisons
			PV::Vector<bool> test15 = ones == twos;
			assert(test15[14] == false);
			PV::Vector<bool> test16 = ones != twos;
			assert(test16[15] == true);
			PV::Vector<bool> test17 = ones > twos;
			assert(test17[16] == false);
			PV::Vector<bool> test18 = ones < twos;
			assert(test18[17] == true);
			PV::Vector<bool> test19 = ones >= twos;
			assert(test19[18] == false);
			PV::Vector<bool> test20 = ones <= twos;
			assert(test20[19] == true);
			
			// bitwise
			PV::Vector<int> test21 = ones & threes;
			assert(test21[20] == 1);
			PV::Vector<int> test22 = ones; test22 &= threes;
			assert(test22[21] == 1);
			PV::Vector<int> test23 = ones | twos;
			assert(test23[22] == 3);
			PV::Vector<int> test24 = ones; test24 |= twos;
			assert(test24[23] == 3);
			PV::Vector<int> test25 = ones ^ ones;
			assert(test25[24] == 0);
			PV::Vector<int> test26 = ones; test26 ^= ones;
			assert(test26[25] == 0);
			PV::Vector<int> test27 = ~ones;
			assert(test27[26] == -2);
			PV::Vector<int> test28 = ones << ones;
			assert(test28[27] == 2);
			PV::Vector<int> test29 = ones; test29 <<= ones;
			assert(test29[28] == 2);
			PV::Vector<int> test30 = twos >> ones;
			assert(test30[29] == 1);
			PV::Vector<int> test31 = twos; test31 >>= ones;
			assert(test31[30] == 1);
			
			// ternary
			PV::Vector<int> test32 = (ones == twos).choose(ones, twos);
			assert(test32[31] == 2);
		}
		
		// test operations on bools
		{
			PV::Vector<bool> falses(test_size, false);
			PV::Vector<bool> trues(test_size, true);
			
			PV::Vector<bool> test1 = falses && trues;
			assert(test1[0] == false);
			PV::Vector<bool> test2 = falses || trues;
			assert(test2[1] == true);
			PV::Vector<bool> test3 = !falses;
			assert(test3[2] == true);
		}
		
		// test operations on datatypes
		{
			PV::Vector<bool> test1(test_size, true);
			test1 &= test1;
			assert(test1[0] == true);
			PV::Vector<char> test2(test_size, 1);
			test2 += test2;
			assert(test2[1] == 2);
			PV::Vector<signed char> test3(test_size, 1);
			test3 += test3;
			assert(test3[2] == 2);
			PV::Vector<unsigned char> test4(test_size, 1);
			test4 += test4;
			assert(test4[3] == 2);
			PV::Vector<short> test5(test_size, 1);
			test5 += test5;
			assert(test5[4] == 2);
			PV::Vector<uint16_t> test6(test_size, 1);
			test6 += test6;
			assert(test6[5] == 2);
			PV::Vector<int> test7(test_size, 1);
			test7 += test7;
			assert(test7[6] == 2);
			PV::Vector<unsigned int> test8(test_size, 1);
			test8 += test8;
			assert(test8[7] == 2);
			PV::Vector<long> test9(test_size, 1);
			test9 += test9;
			assert(test9[8] == 2);
			PV::Vector<unsigned long> test10(test_size, 1);
			test10 += test10;
			assert(test10[9] == 2);
			PV::Vector<long long> test11(test_size, 1);
			test11 += test11;
			assert(test11[10] == 2);
			PV::Vector<unsigned long long> test12(test_size, 1);
			test12 += test12;
			assert(test12[11] == 2);
			PV::Vector<float> test13(test_size, 1);
			test13 -= test13;
			assert(test13[12] == 0);
		}
		
		// test reductions and rotations
		{
			PV::Vector<int> zeros(test_size, 0);
			PV::Vector<int> ones(test_size, 1);
			PV::Vector<int> twos(test_size, 2);
			PV::Vector<float> not_ones(test_size, 1.0000001);
			
			// sum and product
			assert(ones.sum() == test_size);
			assert(not_ones.product() > 1.0000001);
			
			// filter
			PV::Vector<int> nums(test_size, 5);
			nums.set(0,0);
			nums.set(1,1);
			nums.set(2,2);
			nums.set(3,3);
			PV::Vector<int> evens = nums.filterBy(nums % twos == zeros);
			assert(evens.size() == 2);
			assert(evens.sum() == 2);
			
			// rotate
			nums.set(nums.size()-1,4);
			PV::Vector<int> rotated_nums1 = nums.rotateBy(1);
			PV::Vector<int> rotated_nums2 = nums.rotateBy(-1);
			assert(rotated_nums1[0] == 4);
			assert(rotated_nums1[1] == 0);
			assert(rotated_nums2.back() == 0);
			assert(rotated_nums2[0] == 1);
		}
		
		// test other operations
		{
			PV::Vector<int> nums(test_size);
			assert(nums.size() == test_size);
			nums.set(0,1);
			assert(nums[0] == 1);
			nums.reserve(test_size+1);
			nums.resize(test_size+1);
			assert(nums[0] == 1);
			nums.set(test_size,2);
			assert(nums[test_size] == 2);
			assert(nums.front() == 1);
			assert(nums.back() == 2);
			nums.pop_back();
			assert(nums.size() == test_size);
			nums.push_back(3);
			assert(nums.size() == test_size + 1);
			assert(nums.back() == 3);

		}
		
	} catch (char const * error) {
		printf("[Error] %s\n", error);
		exit(1);
	}
	
	printf("All tests completed successfully!\n");
}