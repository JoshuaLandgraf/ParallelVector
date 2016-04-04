// ParallelVector header library
// Joshua Landgraf

#include <Vector>

namespace PV {
	template<class T>
	class Vector {
		typedef size_t size_type;
		
		public:
			// CONSTRUCTORS
			// default constructor
			Vector() : data() {};
			
			// fill constructors
			Vector(size_type length) : data(length) {};
			Vector(size_type length, T fill_value) : data(length, fill_value) {};
			
			// range constructors
			template<class iterator_type>
			Vector(iterator_type start, iterator_type end) : data(start, end) {};
			Vector(T* data_in, size_type length) : data(data_in, data_in + length) {};
			
			// copy constructors
			Vector(const Vector& vec) : data(vec.data) {};
			Vector(const std::vector<T>& vec) : data(vec) {};
			
			// move constructor
			Vector(const Vector&& vec) : data(std::move(vec.data)) {};
			
			
			// OPERATORS
			// data accessor(s)
			T& operator[] (size_type index) {
				return data[index];
			}
			T at(size_type index) {
				return data.at(index);
			} // this is just a hack for now to access the bools
			
			// arithmetic operators
			// plus
			Vector operator+ (const Vector& vec) {
				return do_operation<T>(vec, std::plus<T>());
			}
			// minus
			Vector operator- (const Vector& vec) {
				return do_operation<T>(vec, std::minus<T>());
			}
			// times
			Vector operator* (const Vector& vec) {
				return do_operation<T>(vec, std::multiplies<T>());
			}
			// divide
			Vector operator/ (const Vector& vec) {
				return do_operation<T>(vec, std::divides<T>());
			}
			// mod
			Vector operator% (const Vector& vec) {
				return do_operation<T>(vec, std::modulus<T>());
			}
			// negate
			Vector operator- () {
				return do_operation<T>(std::negate<T>());
			}
			// prefix increment
			Vector operator++ () {
				for (size_type i = 0; i < data.size(); ++i) {
					++data[i];
				}
				return (*this);
			}
			// postfix increment
			Vector operator++ (int value) {
				Vector<T> data_out(data);
				T increment_by = value == 0 ? 1 : value;
				for (size_type i = 0; i < data.size(); ++i) {
					data[i] += increment_by;
				}
				return data_out;
			}
			// prefix decrement
			Vector operator-- () {
				for (size_type i = 0; i < data.size(); ++i) {
					--data[i];
				}
				return (*this);
			}
			// postfix decrement
			Vector operator-- (int value) {
				Vector<T> data_out(data);
				T decrement_by = value == 0 ? 1 : value;
				for (size_type i = 0; i < data.size(); ++i) {
					data[i] -= decrement_by;
				}
				return data_out;
			}
			// plus equals
			Vector operator+= (const Vector& vec) {
				for (size_type i = 0; i < data.size(); ++i) {
					data[i] += vec.data[i];
				}
				return (*this);
			}
			// minus equals
			Vector operator-= (const Vector& vec) {
				for (size_type i = 0; i < data.size(); ++i) {
					data[i] -= vec.data[i];
				}
				return (*this);
			}
			
			//comparison operators
			// equals
			Vector<bool> operator== (const Vector& vec) {
				return do_operation<bool>(vec, std::equal_to<T>());
			}
			// not equals
			Vector<bool> operator!= (const Vector& vec) {
				return do_operation<bool>(vec, std::not_equal_to<T>());
			}
			// greater than
			Vector<bool> operator> (const Vector& vec) {
				return do_operation<bool>(vec, std::greater<T>());
			}
			// less than
			Vector<bool> operator< (const Vector& vec) {
				return do_operation<bool>(vec, std::less<T>());
			}
			// greater than or equal to
			Vector<bool> operator>= (const Vector& vec) {
				return do_operation<bool>(vec, std::greater_equal<T>());
			}
			// less than or equal to
			Vector<bool> operator<= (const Vector& vec) {
				return do_operation<bool>(vec, std::less_equal<T>());
			}
			
			//logical operators
			// and
			Vector<bool> operator&& (const Vector& vec) {
				return do_operation<bool>(vec, std::logical_and<T>());
			}
			// or
			Vector<bool> operator|| (const Vector& vec) {
				return do_operation<bool>(vec, std::logical_or<T>());
			}
			// not
			Vector<bool> operator! () {
				return do_operation<T>(std::logical_not<T>());
			}
			
			// bitwise operators
			// and
			Vector operator& (const Vector& vec) {
				return do_operation<T>(vec, std::bit_and<T>());
			}
			// and equals
			Vector operator&= (const Vector& vec) {
				for (size_type i = 0; i < data.size(); ++i) {
					data[i] &= vec.data[i];
				}
				return (*this);
			}
			// or
			Vector operator| (const Vector& vec) {
				return do_operation<T>(vec, std::bit_or<T>());
			}
			// or equals
			Vector operator|= (const Vector& vec) {
				for (size_type i = 0; i < data.size(); ++i) {
					data[i] |= vec.data[i];
				}
				return (*this);
			}
			// xor
			Vector operator^ (const Vector& vec) {
				return do_operation<T>(vec, std::bit_xor<T>());
			}
			// xor equals
			Vector operator^= (const Vector& vec) {
				for (size_type i = 0; i < data.size(); ++i) {
					data[i] ^= vec.data[i];
				}
				return (*this);
			}
			// not
			Vector operator~ () {
				std::vector<T> data_out(data.size());
				for (size_type i = 0; i < data.size(); ++i) {
					data_out[i] = ~data[i];
				}
				return Vector<T>(std::move(data_out));
			}
			// shift left
			Vector operator<< (const Vector& vec) {
				Vector<T> data_out(data);
				for (size_type i = 0; i < data.size(); ++i) {
					data_out[i] = data[i] << vec.data[i];
				}
				return data_out;
			}
			// shift left equals
			Vector operator<<= (const Vector& vec) {
				for (size_type i = 0; i < data.size(); ++i) {
					data[i] <<= vec.data[i];
				}
				return (*this);
			}
			// shift right
			Vector operator>> (const Vector& vec) {
				Vector<T> data_out(data);
				for (size_type i = 0; i < data.size(); ++i) {
					data_out[i] = data[i] >> vec.data[i];
				}
				return data_out;
			}
			// shift right equals
			Vector operator>>= (const Vector& vec) {
				for (size_type i = 0; i < data.size(); ++i) {
					data[i] >>= vec.data[i];
				}
				return (*this);
			}
			
			// ternary operator will require hacks to work "natively"
			// may just use a method instead (e.g. ternary(A,B,C), choose(A,B,C), switch(A,B,C), pick(A,B,C), etc.)
			// another possibility would be something like if(A).then(B).else(C), A.isTrue(B).isFalse(C), or B.if(A).otherwise(C)
			
			
			// OTHER METHODS
			// get size of vector
			size_type size() {
				return data.size();
			}
			// get max size of vector
			size_type max_size() {
				return 1 << 33;   // 8GB for now, this may change with implementation
			}
			// resize vector
			void resize(size_type new_size) {
				data.resize(new_size);
			}
			// guarantee the vector has space for the given number of elements
			void reserve(size_type reservation) {
				data.reserve(reservation);
			}
			// reference to first element in vector
			T& front() {
				return data.front();
			}
			// reference to last element in vector
			T& back() {
				return data.back();
			}
			// push an element onto the vector
			void push_back(const T& val) {
				data.push_back(val);
			}
			/* left out because C++11 only
			void push_back(T&& val) {
				data.push_back(val);
			}*/
			// removes the last element from the vector
			void pop_back() {
				data.pop_back();
			}
		private:
			// handle arbitrary 2-input, 1-output operation on data
			template<class output_type, class operation_type>
			Vector<output_type> do_operation(const Vector& vec, operation_type operation) {
				if (data.size() != vec.data.size()) throw "ParallelVector size mismatch";
				
				std::vector<output_type> data_out(data.size());
				for (size_type i = 0; i < data.size(); ++i) {
					data_out[i] = operation(data[i], vec.data[i]);
				}
				
				return Vector<output_type>(std::move(data_out));
			}
			
			// handle arbitrary 1-input, 1-output operation on data
			template<class output_type, class operation_type>
			Vector<output_type> do_operation(operation_type operation) {
				std::vector<output_type> data_out(data.size());
				for (size_type i = 0; i < data.size(); ++i) {
					data_out[i] = operation(data[i]);
				}
				
				return Vector<output_type>(std::move(data_out));
			}
			
			std::vector<T> data;
	};
}