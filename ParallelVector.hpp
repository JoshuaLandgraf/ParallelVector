// ParallelVector DSL / header library
// Joshua Landgraf

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include <vector>
#include <string>
#include <exception>
#include <iostream>

namespace PV {
	typedef size_t size_type;
	
	enum operation {
		plus,
		minus,
		times,
		divide,
		mod,
		negate,
		increment,
		decrement,
		equals,
		not_equals,
		greater,
		lesser,
		greater_equal,
		lesser_equal,
		logical_and,
		logical_or,
		logical_not,
		bitwise_and,
		bitwise_or,
		bitwise_xor,
		bitwise_not,
		left_shift,
		right_shift,
		num_ops
	};
	
	const char* const op_to_str[] = {"c = a + b;",
	                                 "c = a - b;",
	                                 "c = a * b;",
	                                 "c = a / b;",
	                                 "c = a % b;",
	                                 "b = -a;",
	                                 "b = a + 1;",
	                                 "b = a - 1;",
	                                 "c = (a == b);",
	                                 "c = (a != b);",
	                                 "c = (a > b);",
	                                 "c = (a < b);",
	                                 "c = (a >= b);",
	                                 "c = (a <= b);",
	                                 "c = a && b;",
	                                 "c = a || b;",
	                                 "b = !a;",
	                                 "c = a & b;",
	                                 "c = a | b;",
	                                 "c = a ^ b;",
	                                 "b = ~a;",
	                                 "c = a << b;",
	                                 "c = a >> b;"};
	
	// the following section of code is made possible by a lot of the tricks learned from the source below
	//    https://gist.github.com/9prady9/848e7774c86a03febe4a
	
	// catch supported data types and convert to opencl-supported string
	template<typename T> static const char* typeToStr() { throw "Unsupported PV::Vector datatype"; return NULL; }
	template<> const char* typeToStr<bool>() { return "bool"; }
	template<> const char* typeToStr<char>() { return "char"; }
	template<> const char* typeToStr<signed char>() { return "char"; }
	template<> const char* typeToStr<unsigned char>() { return "unsigned char"; }
	template<> const char* typeToStr<short>() { return "short"; }
	template<> const char* typeToStr<uint16_t>() { return "unsigned char"; }
	template<> const char* typeToStr<int>() { return "int"; }
	template<> const char* typeToStr<unsigned int>() { return "unsigned int"; }
	template<> const char* typeToStr<long>() { return "long"; }
	template<> const char* typeToStr<unsigned long>() { return "unsigned long"; }
	template<> const char* typeToStr<long long>() { return "long"; }
	template<> const char* typeToStr<unsigned long long>() { return "unsigned long"; }
	template<> const char* typeToStr<float>() { return "float"; }
	
	class opencl_helper {
		public:
		void init() {
			if (!initialized) {
				cl_int err = CL_SUCCESS;
				CPU_available = true;
				CPU_context = cl::Context(CL_DEVICE_TYPE_CPU, NULL, NULL, &err);
				if (err != CL_SUCCESS) CPU_available = false;
				else CPU_queue = cl::CommandQueue(CPU_context, 0, &err);
				if (err != CL_SUCCESS) CPU_available = false;
				
				err = CL_SUCCESS;
				GPU_available = true;
				GPU_context = cl::Context(CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
				if (err != CL_SUCCESS) GPU_available = false;
				else {
					std::vector<cl::Device> devices = GPU_context.getInfo<CL_CONTEXT_DEVICES>();
					GPU_context = cl::Context(devices.back(), NULL, NULL, NULL, &err);
					if (err != CL_SUCCESS) GPU_available = false;
					else GPU_queue = cl::CommandQueue(GPU_context, 0, &err);
					if (err != CL_SUCCESS) GPU_available = false;
				}
				
				initialized = true;
			}
		}
		template<typename T>
		cl::Buffer CPU_buffer(size_type size) {
			cl_int err = CL_SUCCESS;
			cl::Buffer buffer = cl::Buffer(get_CPU_context(), CL_MEM_READ_WRITE, sizeof(T)*size, NULL, &err);
			if (err == CL_SUCCESS) return buffer;
			else throw "CPU buffer creation failed";
		}
		template<typename T>
		cl::Buffer GPU_buffer(size_type size) {
			cl_int err = CL_SUCCESS;
			cl::Buffer buffer = cl::Buffer(get_GPU_context(), CL_MEM_READ_WRITE, sizeof(T)*size, NULL, &err);
			if (err == CL_SUCCESS) return buffer;
			else throw "GPU buffer creation failed";
		}
		template<typename T>
		cl::Buffer CPU_buffer(size_type size, T fill_value) {
			cl::Buffer buffer = CPU_buffer<T>(size);
			cl::Event event;
			cl_int err = get_CPU_queue().enqueueFillBuffer(buffer, fill_value, 0, size * sizeof(T), NULL, &event);
			if (err == CL_SUCCESS) {
				event.wait();
				return buffer;
			} else throw "error filling CPU buffer";
		}
		template<typename T>
		cl::Buffer GPU_buffer(size_type size, T fill_value) {
			cl::Buffer buffer = GPU_buffer<T>(size);
			cl::Event event;
			cl_int err = get_GPU_queue().enqueueFillBuffer(buffer, fill_value, 0, size * sizeof(T), NULL, &event);
			if (err == CL_SUCCESS) {
				event.wait();
				return buffer;
			} else throw "error filling CPU buffer";
		}
		template<class iterator_type>
		cl::Buffer CPU_buffer(iterator_type start, iterator_type end) {
			typedef typename std::iterator_traits<iterator_type>::value_type T;
			cl::Buffer buffer = CPU_buffer<T>(sizeof(T)*(end-start));
			cl::Event event;
			cl_int err = cl::copy(get_CPU_queue(), start, end, buffer);
			if (err == CL_SUCCESS) {
				return buffer;
			} else throw "error creating CPU buffer from iterators";
		}
		template<class iterator_type>
		cl::Buffer GPU_buffer(iterator_type start, iterator_type end) {
			typedef typename std::iterator_traits<iterator_type>::value_type T;
			cl::Buffer buffer = GPU_buffer<T>(sizeof(T)*(end-start));
			cl::Event event;
			cl_int err = cl::copy(get_GPU_queue(), start, end, buffer);
			if (err == CL_SUCCESS) {
				return buffer;
			} else throw "error creating GPU buffer from iterators";
		}
		template<typename T>
		cl::Buffer CPU_buffer(T* ptr, size_type size) {
			cl_int err = CL_SUCCESS;
			cl::Buffer buffer = cl::Buffer(get_CPU_context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(T)*size, ptr, &err);
			if (err == CL_SUCCESS) return buffer;
			else throw "error creating CPU buffer from pointer";
		}
		template<typename T>
		cl::Buffer GPU_buffer(T* ptr, size_type size) {
			cl_int err = CL_SUCCESS;
			cl::Buffer buffer = cl::Buffer(get_GPU_context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(T)*size, ptr, &err);
			if (err == CL_SUCCESS) return buffer;
			else throw "error creating GPU buffer from pointer";
		}
		
		template<typename T>
		void from_CPU_buffer(cl::Buffer buf, std::vector<T> & vec) {
			cl::copy(get_CPU_queue(), buf, vec.begin(), vec.end());
		}
		template<typename T>
		void from_GPU_buffer(cl::Buffer buf, std::vector<T> & vec) {
			cl::copy(get_GPU_queue(), buf, vec.begin(), vec.end());
		}
		
		cl::Context get_CPU_context() { return CPU_available ? CPU_context : GPU_context; }
		cl::Context get_GPU_context() { return GPU_available ? GPU_context : CPU_context; }
		cl::CommandQueue get_CPU_queue() { return CPU_available ? CPU_queue : GPU_queue; }
		cl::CommandQueue get_GPU_queue() { return GPU_available ? GPU_queue : CPU_queue; }
		
		private:
		bool initialized = false, CPU_available = false, GPU_available = false;
		cl::Context CPU_context, GPU_context;
		cl::CommandQueue CPU_queue, GPU_queue;
	};
	
	opencl_helper cl;
	
	template<typename T1, typename T2, typename T3>
	void parallel_compute(cl::Buffer aa, cl::Buffer bb, cl::Buffer cc, size_type size, enum operation op) {
		static const char* const starting_kernel_code = "__kernel void opencl_compute(global %s * aa, global %s * bb, global %s * cc) { const size_t i = get_global_id(0); const %s a = aa[i]; const %s b = bb[i]; %s c; %s cc[i] = c; }";
		
		static cl::Program program[num_ops];
		static cl::Kernel kernel[num_ops];
		static bool initialized[num_ops];
		cl_int err = CL_SUCCESS;
		if (!initialized[op]) {
			const char *T1_str = typeToStr<T1>();
			const char *T2_str = typeToStr<T2>();
			const char *T3_str = typeToStr<T3>();
			if (T1_str == NULL || T2_str == NULL || T3_str == NULL) throw "Unsupported type in computation";
			const char *op_str = op_to_str[op];
			//printf("%s\n", op_str);
			char kernel_code[600];
			sprintf(kernel_code, starting_kernel_code, T1_str, T2_str, T3_str, T1_str, T2_str, T3_str, op_str);
			try {
				cl::Program::Sources kernel_source(1, std::make_pair(kernel_code,strlen(kernel_code)));
				program[op] = cl::Program(cl.get_GPU_context(), std::string(kernel_code), false);
				program[op].build();
			} catch (cl::Error & err) {
				// debugging code from stackoverflow
				//char log[10000];
				//program[op].getBuildInfo(opencl_devices[1], CL_PROGRAM_BUILD_LOG, log);
				//cl::clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 10000, log, NULL);
				//printf("%s\n", log);

				throw "Error encountered during OpenCL compilation";
			}
			kernel[op] = cl::Kernel(program[op], "opencl_compute");
			initialized[op] = true;
		}
		
		//(cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(kernel))(cl::EnqueueArgs(size), aa, bb, cc);
		cl::CommandQueue queue = cl.get_GPU_queue();
		cl::Event event = (cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(kernel[op]))(cl::EnqueueArgs(queue, cl::NDRange(size)), aa, bb, cc);
		//cl::CommandQueue queue(opencl_context, opencl_devices[1], 0, NULL);
		//queue.enqueueNDRangeKernel((cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(kernel))(cl::EnqueueArgs(size), aa, bb, cc),
		//                           cl::NullRange, cl::NDRange(size), cl::NullRange, NULL, &event);
		event.wait();
	}
	
	template<typename T1, typename T2>
	void parallel_compute(cl::Buffer aa, cl::Buffer bb, size_type size, enum operation op) {
		static const char* const starting_kernel_code = "kernel void opencl_compute(global %s * aa, global %s * bb) { const size_t i = get_global_id(0); const %s a = aa[i]; %s b; %s bb[i] = b; }";
		
		static cl::Program program[num_ops];
		static cl::Kernel kernel[num_ops];
		static bool initialized[num_ops];
		if (!initialized[op]) {
			const char *T1_str = typeToStr<T1>();
			const char *T2_str = typeToStr<T2>();
			if (T1_str == NULL || T2_str == NULL) throw "Unsupported type in computation";
			const char *op_str = op_to_str[op];
			//printf("%s\n", op_str);
			char kernel_code[600];
			sprintf(kernel_code, starting_kernel_code, T1_str, T2_str, T1_str, T2_str, op_str);
			try {
				cl::Program::Sources kernel_source(1, std::make_pair(kernel_code,strlen(kernel_code)));
				program[op] = cl::Program(cl.get_GPU_context(), std::string(kernel_code), false);
				program[op].build();
			} catch (cl::Error & err) {
				// debugging code from stackoverflow
				//char log[10000];
				//program[op].getBuildInfo(opencl_devices[1], CL_PROGRAM_BUILD_LOG, log);
				//cl::clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 10000, log, NULL);
				//printf("%s\n", log);

				throw "Error encountered during OpenCL compilation";
			}
			kernel[op] = cl::Kernel(program[op], "opencl_compute");
			initialized[op] = true;
		}
		
		cl::CommandQueue queue = cl.get_GPU_queue();
		cl::Event event = (cl::make_kernel<cl::Buffer, cl::Buffer>(kernel[op]))(cl::EnqueueArgs(queue, cl::NDRange(size)), aa, bb);
		event.wait();
	}
	
	template<typename T1, typename T2, typename T3>
	void do_operation(std::vector<T1> & a, const std::vector<T2> & b, std::vector<T3> & c, size_type size, enum operation op) {
		if (a.size() != size || a.size() != b.size() || b.size() != c.size()) throw "ParallelVector size mismatch";
		cl.init();
		cl::Buffer a_buf = cl.GPU_buffer(a.begin(), a.end()), b_buf = cl.GPU_buffer(b.begin(), b.end()), c_buf = cl.GPU_buffer<T3>(a.size());
		parallel_compute<T1, T2, T3>(a_buf, b_buf, c_buf, a.size(), op);
		cl.from_GPU_buffer<T3>(c_buf, c);
	}
	
	template<typename T1, typename T2>
	void do_operation(std::vector<T1> & a, std::vector<T2> & b, size_type size, enum operation op) {
		if (a.size() != size || a.size() != b.size()) throw "ParallelVector size mismatch";
		cl.init();
		cl::Buffer a_buf = cl.GPU_buffer(a.begin(), a.end()), b_buf = cl.GPU_buffer<T2>(a.size());
		parallel_compute<T1, T2>(a_buf, b_buf, a.size(), op);
		cl.from_GPU_buffer<T2>(b_buf, b);
	}
	
	
	template<class T>
	class Vector {
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
				Vector<T> output(data.size());
				do_operation<T,T,T>(data, vec.data, output.data, data.size(), plus);
				return output;
			}
			// minus
			Vector operator- (const Vector& vec) {
				Vector<T> output(data.size());
				do_operation<T,T,T>(data, vec.data, output.data, data.size(), minus);
				return output;
			}
			// times
			Vector operator* (const Vector& vec) {
				Vector<T> output(data.size());
				do_operation<T,T,T>(data, vec.data, output.data, data.size(), times);
				return output;
			}
			// divide
			Vector operator/ (const Vector& vec) {
				Vector<T> output(data.size());
				do_operation<T,T,T>(data, vec.data, output.data, data.size(), divide);
				return output;
			}
			// mod
			Vector operator% (const Vector& vec) {
				Vector<T> output(data.size());
				do_operation<T,T,T>(data, vec.data, output.data, data.size(), mod);
				return output;
			}
			// negate
			Vector operator- () {
				Vector<T> output(data.size());
				do_operation<T,T>(data, output.data, data.size(), negate);
				return output;
			}
			// prefix increment
			Vector operator++ () {
				do_operation<T,T>(data, data, data.size(), increment);
				return (*this);
			}
			// postfix increment
			Vector operator++ (int value) {
				Vector<T> output(data);
				do_operation<T,T>(data, data, data.size(), increment);
				return output;
			}
			// prefix decrement
			Vector operator-- () {
				do_operation<T,T>(data, data, data.size(), decrement);
				return (*this);
			}
			// postfix decrement
			Vector operator-- (int value) {
				Vector<T> output(data);
				do_operation<T,T>(data, data, data.size(), decrement);
				return output;
			}
			// plus equals
			Vector operator+= (const Vector& vec) {
				do_operation<T,T,T>(data, vec.data, data, data.size(), plus);
				return (*this);
			}
			// minus equals
			Vector operator-= (const Vector& vec) {
				do_operation<T,T,T>(data, vec.data, data, data.size(), minus);
				return (*this);
			}
			
			//comparison operators
			// equals
			Vector<bool> operator== (const Vector& vec) {
				std::vector<bool> output(data.size());
				do_operation<T,T,bool>(data, vec.data, output, data.size(), equals);
				return Vector<bool>(std::move(output));
			}
			// not equals
			Vector<bool> operator!= (const Vector& vec) {
				std::vector<bool> output(data.size());
				do_operation<T,T,bool>(data, vec.data, output, data.size(), not_equals);
				return Vector<bool>(std::move(output));
			}
			// greater than
			Vector<bool> operator> (const Vector& vec) {
				std::vector<bool> output(data.size());
				do_operation<T,T,bool>(data, vec.data, output, data.size(), greater);
				return Vector<bool>(std::move(output));
			}
			// less than
			Vector<bool> operator< (const Vector& vec) {
				std::vector<bool> output(data.size());
				do_operation<T,T,bool>(data, vec.data, output, data.size(), lesser);
				return Vector<bool>(std::move(output));
			}
			// greater than or equal to
			Vector<bool> operator>= (const Vector& vec) {
				std::vector<bool> output(data.size());
				do_operation<T,T,bool>(data, vec.data, output, data.size(), greater_equal);
				return Vector<bool>(std::move(output));
			}
			// less than or equal to
			Vector<bool> operator<= (const Vector& vec) {
				std::vector<bool> output(data.size());
				do_operation<T,T,bool>(data, vec.data, output, data.size(), lesser_equal);
				return Vector<bool>(std::move(output));
			}
			
			//logical operators
			// and
			Vector<bool> operator&& (const Vector& vec) {
				Vector<bool> output(data.size());
				do_operation<T,T,T>(data, vec.data, output.data, data.size(), logical_and);
				return output;
			}
			// or
			Vector<bool> operator|| (const Vector& vec) {
				Vector<bool> output(data.size());
				do_operation<T,T,T>(data, vec.data, output.data, data.size(), logical_or);
				return output;
			}
			// not
			Vector<bool> operator! () {
				Vector<bool> output(data.size());
				do_operation<T,T>(data, output.data, data.size(), logical_not);
				return output;
			}
			
			// bitwise operators
			// and
			Vector operator& (const Vector& vec) {
				Vector<T> output(data.size());
				do_operation<T,T,T>(data, vec.data, output.data, data.size(), bitwise_and);
				return output;
			}
			// and equals
			Vector operator&= (const Vector& vec) {
				do_operation<T,T,T>(data, vec.data, data, data.size(), bitwise_and);
				return (*this);
			}
			// or
			Vector operator| (const Vector& vec) {
				Vector<T> output(data.size());
				do_operation<T,T,T>(data, vec.data, output.data, data.size(), bitwise_or);
				return output;
			}
			// or equals
			Vector operator|= (const Vector& vec) {
				do_operation<T,T,T>(data, vec.data, data, data.size(), bitwise_or);
				return (*this);
			}
			// xor
			Vector operator^ (const Vector& vec) {
				Vector<T> output(data.size());
				do_operation<T,T,T>(data, vec.data, output.data, data.size(), bitwise_xor);
				return output;
			}
			// xor equals
			Vector operator^= (const Vector& vec) {
				do_operation<T,T,T>(data, vec.data, data, data.size(), bitwise_xor);
				return (*this);
			}
			// not
			Vector operator~ () {
				Vector<T> output(data.size());
				do_operation<T,T>(data, output.data, data.size(), bitwise_not);
				return output;
			}
			// shift left
			Vector operator<< (const Vector& vec) {
				Vector<T> output(data.size());
				do_operation<T,T,T>(data, vec.data, output.data, data.size(), left_shift);
				return output;
			}
			// shift left equals
			Vector operator<<= (const Vector& vec) {
				do_operation<T,T,T>(data, vec.data, data, data.size(), left_shift);
				return (*this);
			}
			// shift right
			Vector operator>> (const Vector& vec) {
				Vector<T> output(data.size());
				do_operation<T,T,T>(data, vec.data, output.data, data.size(), right_shift);
				return output;
			}
			// shift right equals
			Vector operator>>= (const Vector& vec) {
				do_operation<T,T,T>(data, vec.data, data, data.size(), right_shift);
				return (*this);
			}
			
			// ternary operator will require hacks to work "natively"
			// may just use a method instead (e.g. ternary(A,B,C), choose(A,B,C), switch(A,B,C), pick(A,B,C), etc.)
			// another possibility would be something like if(A).then(B).else(C), A.isTrue(B).isFalse(C), or B.if(A).otherwise(C)
			// A.ifThenElse(B,C)
			// A.choose(B,C)   // best
			
			
			// OTHER METHODS
			// get size of vector
			size_type size() {
				return data.size();
			}
			// get max size of vector
			size_type max_size() {
				return 1 << 31;   // 4GB for now, this may change with implementation
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
			std::vector<T> data;
			//T * data;
			//size_type num_filled, num_allocated;
	};
}