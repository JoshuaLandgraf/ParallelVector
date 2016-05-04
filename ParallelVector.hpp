// ParallelVector DSL / header library
// Joshua Landgraf

// credit to the source below for lots of tricks that made this library possible
//    https://gist.github.com/9prady9/848e7774c86a03febe4a

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
		ternary,
		copy,
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
	                                 "c = a >> b;",
	                                 "d = a ? b : c;",
		                             "b = a;"};
	
	enum reduce_operation {reduce_plus, reduce_times, num_reduce_ops};
	const char* const reduce_op_to_str[] = {"+=", "*="};
	
	enum rotate_operation {rotate_left, rotate_right, num_rotate_ops};
	const char* const rotate_op_to_str[] = {"(i + size - 1) \% size", "(i + 1) \% size"};
	
	// catch supported data types and convert to opencl-supported string
	template<typename T> static const char* typeToStr() { throw "Unsupported PV::Vector datatype"; return nullptr; }
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
	//template<> const char* typeToStr<double>() { return "double"; }   // not supported by all devices
	
	// forward declare function(s)
	template<typename T1, typename T2>
	void parallel_compute(cl::Buffer & aa, cl::Buffer & bb, size_type size, enum operation op);
	
	class opencl_helper {
		public:
		opencl_helper() {
			cl_int err = CL_SUCCESS;
			CPU_available = true;
			CPU_context = cl::Context(CL_DEVICE_TYPE_CPU, nullptr, nullptr, &err);
			if (err != CL_SUCCESS) CPU_available = false;
			else CPU_queue = cl::CommandQueue(CPU_context, 0, &err);
			if (err != CL_SUCCESS) CPU_available = false;
			
			err = CL_SUCCESS;
			GPU_available = false;
			GPU_context = cl::Context(CL_DEVICE_TYPE_GPU, nullptr, nullptr, &err);
			if (err != CL_SUCCESS) GPU_available = false;
			else {
				std::vector<cl::Device> devices = GPU_context.getInfo<CL_CONTEXT_DEVICES>();
				GPU_context = cl::Context(devices.front(), nullptr, nullptr, nullptr, &err);
				if (err != CL_SUCCESS) GPU_available = false;
				else GPU_queue = cl::CommandQueue(GPU_context, 0, &err);
				if (err != CL_SUCCESS) GPU_available = false;
			}
		}
		template<typename T>
		cl::Buffer CPU_buffer(size_type size) {
			cl_int err = CL_SUCCESS;
			cl::Buffer buffer = cl::Buffer(get_CPU_context(), CL_MEM_READ_WRITE, sizeof(T)*size, nullptr, &err);
			if (err == CL_SUCCESS) return buffer;
			else throw "CPU buffer creation failed";
		}
		template<typename T>
		cl::Buffer GPU_buffer(size_type size) {
			cl_int err = CL_SUCCESS;
			cl::Buffer buffer = cl::Buffer(get_GPU_context(), CL_MEM_READ_WRITE, sizeof(T)*size, nullptr, &err);
			if (err == CL_SUCCESS) return buffer;
			else throw "GPU buffer creation failed";
		}
		template<typename T>
		cl::Buffer CPU_buffer(size_type size, T fill_value) {
			cl::Buffer buffer = CPU_buffer<T>(size);
			cl::Event event;
			cl_int err = get_CPU_queue().enqueueFillBuffer(buffer, fill_value, 0, size * sizeof(T), nullptr, &event);
			if (err == CL_SUCCESS) {
				event.wait();
				return buffer;
			} else throw "error filling CPU buffer";
		}
		template<typename T>
		cl::Buffer GPU_buffer(size_type size, T fill_value) {
			cl::Buffer buffer = GPU_buffer<T>(size);
			cl::Event event;
			cl_int err = get_GPU_queue().enqueueFillBuffer(buffer, fill_value, 0, size * sizeof(T), nullptr, &event);
			if (err == CL_SUCCESS) {
				event.wait();
				return buffer;
			} else throw "error filling CPU buffer";
		}
		template<class iterator_type>
		cl::Buffer CPU_buffer_iter(iterator_type begin, iterator_type end) {
			cl_int err = CL_SUCCESS;
			cl::Buffer buffer = cl::Buffer(get_CPU_queue(), begin, end, false, false, &err);
			if (err == CL_SUCCESS) {
				return buffer;
			} else throw "error creating GPU buffer from iterators";
		}
		template<class iterator_type>
		cl::Buffer GPU_buffer_iter(iterator_type begin, iterator_type end) {
			cl_int err = CL_SUCCESS;
			cl::Buffer buffer = cl::Buffer(get_GPU_queue(), begin, end, false, false, &err);
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
		cl::Buffer duplicate_buffer(cl::Buffer buf, size_type num_filled, size_type num_allocated) {
			cl::Buffer buffer = GPU_buffer<T>(num_allocated);
			parallel_compute<T, T>(buf, buffer, num_filled, copy);
			return buffer;
		}
		template<typename T>
		cl::Buffer move_buffer(cl::Buffer buf) {
			cl::Buffer buffer(std::move(buf));
			return buffer;
		}
		
		template<typename T>
		cl::Buffer get_sub_buffer(cl::Buffer & buf, size_type start, size_type size) {
			cl_buffer_region region;
			region.origin = start * sizeof(T);
			region.size = size * sizeof(T);
			cl_int err = CL_SUCCESS;
			cl::Buffer sub_buf = buf.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
			if (err != CL_SUCCESS) throw "error accessing GPU buffer";
			return sub_buf;
		}
		
		template<typename T>
		void from_CPU_buffer(cl::Buffer & buf, size_type start, std::vector<T> & vec) {
			cl::Buffer sub_buf = get_sub_buffer<T>(buf, start, vec.size());
			cl::copy(get_CPU_queue(), sub_buf, vec.begin(), vec.end());
		}
		template<typename T>
		void from_GPU_buffer(cl::Buffer & buf, size_type start, std::vector<T> & vec) {
			cl::Buffer sub_buf = get_sub_buffer<T>(buf, start, vec.size());
			cl::copy(get_GPU_queue(), sub_buf, vec.begin(), vec.end());
		}
		template<typename T>
		void from_CPU_buffer(cl::Buffer & buf, size_type start, T * data, size_type size) {
			cl::Buffer sub_buf = get_sub_buffer<T>(buf, start, size);
			cl::copy(get_CPU_queue(), sub_buf, data, data + size);
		}
		template<typename T>
		void from_GPU_buffer(cl::Buffer & buf, size_type start, T * data, size_type size) {
			cl::Buffer sub_buf = get_sub_buffer<T>(buf, start, size);
			cl::copy(get_GPU_queue(), sub_buf, data, data + size);
		}
		template<class iterator_type>
		void from_CPU_buffer(cl::Buffer & buf, size_type start, iterator_type begin, iterator_type end) {
			typedef typename std::iterator_traits<iterator_type>::value_type T;
			cl::Buffer sub_buf = get_sub_buffer<T>(buf, start, end-begin);
			cl::copy(get_CPU_queue(), sub_buf, begin, end);
		}
		template<class iterator_type>
		void from_GPU_buffer(cl::Buffer & buf, size_type start, iterator_type begin, iterator_type end) {
			typedef typename std::iterator_traits<iterator_type>::value_type T;
			cl::Buffer sub_buf = get_sub_buffer<T>(buf, start, end-begin);
			cl::copy(get_GPU_queue(), sub_buf, begin, end);
		}
		template<typename T>
		void to_CPU_buffer(cl::Buffer & buf, size_type start, std::vector<T> & vec) {
			cl::Buffer sub_buf = get_sub_buffer<T>(buf, start, vec.size());
			cl::copy(get_CPU_queue(), vec.begin(), vec.end(), buf);
		}
		template<typename T>
		void to_GPU_buffer(cl::Buffer & buf, size_type start, std::vector<T> & vec) {
			cl::Buffer sub_buf = get_sub_buffer<T>(buf, start, vec.size());
			cl::copy(get_GPU_queue(), vec.begin(), vec.end(), sub_buf);
		}
		template<typename T>
		void to_CPU_buffer(cl::Buffer & buf, size_type start, T * data, size_type size) {
			cl::Buffer sub_buf = get_sub_buffer<T>(buf, start, size);
			cl::copy(get_CPU_queue(), data, data + size, sub_buf);
		}
		template<typename T>
		void to_GPU_buffer(cl::Buffer & buf, size_type start, T * data, size_type size) {
			cl::Buffer sub_buf = get_sub_buffer<T>(buf, start, size);
			cl::copy(get_GPU_queue(), data, data + size, sub_buf);
		}
		template<class iterator_type>
		void to_CPU_buffer(cl::Buffer & buf, size_type start, iterator_type begin, iterator_type end) {
			typedef typename std::iterator_traits<iterator_type>::value_type T;
			cl::Buffer sub_buf = get_sub_buffer<T>(buf, start, end-begin);
			cl::copy(get_CPU_queue(), begin, end, sub_buf);
		}
		template<class iterator_type>
		void to_GPU_buffer(cl::Buffer & buf, size_type start, iterator_type begin, iterator_type end) {
			typedef typename std::iterator_traits<iterator_type>::value_type T;
			cl::Buffer sub_buf = get_sub_buffer<T>(buf, start, end-begin);
			cl::copy(get_GPU_queue(), begin, end, sub_buf);
		}
		
		template<typename T>
		void set_CPU_buffer_index(cl::Buffer & buf, size_type index, T val) {
			cl::Buffer sub_buf = get_sub_buffer<T>(buf, index, 1);
			cl::copy(get_CPU_queue(), &val, &val + 1, sub_buf);
		}
		template<typename T>
		void set_GPU_buffer_index(cl::Buffer & buf, size_type index, T val) {
			cl::Buffer sub_buf = get_sub_buffer<T>(buf, index, 1);
			cl::copy(get_GPU_queue(), &val, &val + 1, sub_buf);
		}
		template<typename T>
		T get_GPU_buffer_index(cl::Buffer & buf, size_type index) {
			cl::Buffer sub_buf = get_sub_buffer<T>(buf, index, 1);
			T val;
			cl::copy(get_GPU_queue(), sub_buf, &val, &val + 1);
			return val;
		}
		
		
		cl::Context get_CPU_context() { return CPU_available ? CPU_context : GPU_context; }
		cl::Context get_GPU_context() { return GPU_available ? GPU_context : CPU_context; }
		cl::CommandQueue get_CPU_queue() { return CPU_available ? CPU_queue : GPU_queue; }
		cl::CommandQueue get_GPU_queue() { return GPU_available ? GPU_queue : CPU_queue; }
		
		private:
		bool CPU_available, GPU_available;
		cl::Context CPU_context, GPU_context;
		cl::CommandQueue CPU_queue, GPU_queue;
	};
	
	opencl_helper cl;
	
	template<typename T1, typename T2>
	void parallel_compute(cl::Buffer & aa, cl::Buffer & bb, size_type size, enum operation op) {
		static const char* const starting_kernel_code =
			"__kernel void opencl_compute(global %s * aa, global %s * bb) \n"
			"{                                                            \n"
			"	const size_t i = get_global_id(0);                       \n"
			"	const %s a = aa[i];                                      \n"
			"	%s b;                                                    \n"
			"	%s                                                       \n"
			"	bb[i] = b;                                               \n"
			"}";
		
		static cl::Program program[num_ops];
		static cl::Kernel kernel[num_ops];
		static bool initialized[num_ops];
		if (!initialized[op]) {
			const char *T1_str = typeToStr<T1>();
			const char *T2_str = typeToStr<T2>();
			if (T1_str == nullptr || T2_str == nullptr) throw "Unsupported type in computation";
			const char *op_str = op_to_str[op];
			//printf("%s\n", op_str);
			char kernel_code[1500];
			sprintf(kernel_code, starting_kernel_code, T1_str, T2_str, T1_str, T2_str, op_str);
			try {
				cl::Program::Sources kernel_source(1, std::make_pair(kernel_code,strlen(kernel_code)));
				program[op] = cl::Program(cl.get_GPU_context(), std::string(kernel_code), true);
			} catch (cl::Error & err) {
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
	void parallel_compute(cl::Buffer & aa, const cl::Buffer & bb, cl::Buffer & cc, size_type size, enum operation op) {
		static const char* const starting_kernel_code =
			"__kernel void opencl_compute(global %s * aa, global %s * bb, global %s * cc) \n"
			"{                                      \n"
			"	const size_t i = get_global_id(0); \n"
			"	const %s a = aa[i];                \n"
			"	const %s b = bb[i];                \n"
			"	%s c;                              \n"
			"	%s                                 \n"
			"	cc[i] = c;                         \n"
			"}";
		
		static cl::Program program[num_ops];
		static cl::Kernel kernel[num_ops];
		static bool initialized[num_ops];
		cl_int err = CL_SUCCESS;
		if (!initialized[op]) {
			const char *T1_str = typeToStr<T1>();
			const char *T2_str = typeToStr<T2>();
			const char *T3_str = typeToStr<T3>();
			if (T1_str == nullptr || T2_str == nullptr || T3_str == nullptr) throw "Unsupported type in computation";
			const char *op_str = op_to_str[op];
			//printf("%s\n", op_str);
			char kernel_code[1500];
			sprintf(kernel_code, starting_kernel_code, T1_str, T2_str, T3_str, T1_str, T2_str, T3_str, op_str);
			try {
				cl::Program::Sources kernel_source(1, std::make_pair(kernel_code,strlen(kernel_code)));
				program[op] = cl::Program(cl.get_GPU_context(), std::string(kernel_code), true);
			} catch (cl::Error & err) {
				throw "Error encountered during OpenCL compilation";
			}
			kernel[op] = cl::Kernel(program[op], "opencl_compute");
			initialized[op] = true;
		}
		
		cl::CommandQueue queue = cl.get_GPU_queue();
		cl::Event event = (cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(kernel[op]))(cl::EnqueueArgs(queue, cl::NDRange(size)), aa, bb, cc);
		event.wait();
	}
	
	template<typename T1, typename T2, typename T3, typename T4>
	void parallel_compute(cl::Buffer & aa, const cl::Buffer & bb, const cl::Buffer & cc, cl::Buffer & dd, size_type size, enum operation op) {
		static const char* const starting_kernel_code =
			"__kernel void opencl_compute(global %s * aa, global %s * bb, global %s * cc, global %s * dd) \n"
			"{                                      \n"
			"	const size_t i = get_global_id(0); \n"
			"	const %s a = aa[i];                \n"
			"	const %s b = bb[i];                \n"
			"	const %s c = cc[i];                \n"
			"	%s d;                              \n"
			"	%s                                 \n"
			"	dd[i] = d;                         \n"
			"}";
		
		static cl::Program program[num_ops];
		static cl::Kernel kernel[num_ops];
		static bool initialized[num_ops];
		cl_int err = CL_SUCCESS;
		if (!initialized[op]) {
			const char *T1_str = typeToStr<T1>();
			const char *T2_str = typeToStr<T2>();
			const char *T3_str = typeToStr<T3>();
			const char *T4_str = typeToStr<T4>();
			if (T1_str == nullptr || T2_str == nullptr || T3_str == nullptr || T4_str == nullptr) throw "Unsupported type in computation";
			const char *op_str = op_to_str[op];
			//printf("%s\n", op_str);
			char kernel_code[1500];
			sprintf(kernel_code, starting_kernel_code, T1_str, T2_str, T3_str, T4_str, T1_str, T2_str, T3_str, T4_str, op_str);
			try {
				cl::Program::Sources kernel_source(1, std::make_pair(kernel_code,strlen(kernel_code)));
				program[op] = cl::Program(cl.get_GPU_context(), std::string(kernel_code), true);
			} catch (cl::Error & err) {
				throw "Error encountered during OpenCL compilation";
			}
			kernel[op] = cl::Kernel(program[op], "opencl_compute");
			initialized[op] = true;
		}
		
		cl::CommandQueue queue = cl.get_GPU_queue();
		cl::Event event = (cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(kernel[op]))(cl::EnqueueArgs(queue, cl::NDRange(size)), aa, bb, cc, dd);
		event.wait();
	}
	
	template<typename T>
	T parallel_reduce(cl::Buffer & aa, size_type size, enum reduce_operation op) {
		const unsigned reduction_amount = 64;
		static const char* const starting_kernel_code =
			"__kernel void opencl_reduce(__global %s *aa, __global %s *rr, __const size_t stride, __const size_t size) \n"
			"{                                                            \n"
			"	const size_t start = get_global_id(0);                   \n"
			"	%s accum = aa[start];                                    \n"
			"	for(size_t i = start + stride; i < size; i += stride) {  \n"
			"		accum %s aa[i];                                     \n"
			"	}                                                        \n"
			"	rr[start] = accum;                                       \n"
			"}";
		static cl::Program program[num_reduce_ops];
		static cl::Kernel kernel[num_reduce_ops];
		static bool initialized[num_reduce_ops];
		cl_int err = CL_SUCCESS;
		if (!initialized[op]) {
			const char *T_str = typeToStr<T>();
			if (T_str == nullptr) throw "Unsupported type in computation";
			const char* op_str = reduce_op_to_str[op];
			//printf("%s\n", op_str);
			char kernel_code[700];
			sprintf(kernel_code, starting_kernel_code, T_str, T_str, T_str, op_str);
			try {
				cl::Program::Sources kernel_source(1, std::make_pair(kernel_code,strlen(kernel_code)));
				program[op] = cl::Program(cl.get_GPU_context(), std::string(kernel_code), true);
			} catch (cl::Error & err) {
				throw "Error encountered during OpenCL compilation";
			}
			kernel[op] = cl::Kernel(program[op], "opencl_reduce");
			initialized[op] = true;
		}
		
		cl::CommandQueue queue = cl.get_GPU_queue();
		
		cl::Buffer in_buf = aa, out_buf;
		size_type reduced_size, last_size = size;
		for (reduced_size = size / reduction_amount; reduced_size > 10000; reduced_size /= reduction_amount) {
			out_buf = cl.GPU_buffer<T>(reduced_size);
			cl::Event event = (cl::make_kernel<cl::Buffer, cl::Buffer, const size_t, const size_t>(kernel[op]))(cl::EnqueueArgs(queue, cl::NDRange(reduced_size)), in_buf, out_buf, reduced_size, last_size);
			event.wait();
			in_buf = out_buf;
			last_size = reduced_size;
		}
		std::vector<T> partial_results(last_size);
		cl.from_GPU_buffer(in_buf, 0, partial_results);
		T result = partial_results[0];
		for (size_type i = 1; i < last_size; ++i) {
			if (op == reduce_plus) {result += partial_results[i];}
			else {result *= partial_results[i];}
		}
		return result;
	}
	
	template<typename T>
	size_type parallel_filter(cl::Buffer & nums, const cl::Buffer & bools, cl::Buffer & results, size_type size) {
		static const char* const starting_kernel_code =
			"__kernel void opencl_filter(__global %s *nums, __global bool *bools, __global %s *result, __global unsigned int *current_size) \n"
			"{                                                             \n"
			"	const size_t i = get_global_id(0);                        \n"
			"	if(bools[i]) {                                            \n"
			"		const unsigned int index = atomic_inc(current_size); \n"
			"		result[index] = nums[i];                             \n"
			"	}                                                         \n"
			"}";
		static cl::Program program;
		static cl::Kernel kernel;
		static bool initialized;
		cl_int err = CL_SUCCESS;
		if (!initialized) {
			const char *T_str = typeToStr<T>();
			if (T_str == nullptr) throw "Unsupported type in computation";
			char kernel_code[1000];
			sprintf(kernel_code, starting_kernel_code, T_str, T_str);
			try {
				cl::Program::Sources kernel_source(1, std::make_pair(kernel_code,strlen(kernel_code)));
				program = cl::Program(cl.get_GPU_context(), std::string(kernel_code), true);
			} catch (cl::Error & err) {
				throw "Error encountered during OpenCL compilation";
			}
			kernel = cl::Kernel(program, "opencl_filter");
			initialized = true;
		}
		
		cl::CommandQueue queue = cl.get_GPU_queue();
		cl::Buffer size_buffer = cl.GPU_buffer<unsigned int>(1, 0);
		cl::Event event = (cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(kernel))(cl::EnqueueArgs(queue, cl::NDRange(size)), nums, bools, results, size_buffer);
		event.wait();
		return cl.get_GPU_buffer_index<unsigned int>(size_buffer, 0);
	}
	
	template<typename T>
	void parallel_rotate(cl::Buffer & ins, const cl::Buffer & outs, long int rotation, size_type size) {
		static const char* const starting_kernel_code =
			"__kernel void opencl_rotate(__global %s *ins, __global %s *outs, __const long int rotation, __const size_t size) \n"
			"{                                        \n"
			"	const size_t i = get_global_id(0);   \n"
			"	outs[(i+rotation) %s size] = ins[i]; \n"
			"}";
		static cl::Program program;
		static cl::Kernel kernel;
		static bool initialized;
		cl_int err = CL_SUCCESS;
		if (!initialized) {
			const char *T_str = typeToStr<T>();
			if (T_str == nullptr) throw "Unsupported type in computation";
			char kernel_code[1000];
			sprintf(kernel_code, starting_kernel_code, T_str, T_str, "%");
			try {
				cl::Program::Sources kernel_source(1, std::make_pair(kernel_code,strlen(kernel_code)));
				program = cl::Program(cl.get_GPU_context(), std::string(kernel_code), true);
			} catch (cl::Error & err) {
				throw "Error encountered during OpenCL compilation";
			}
			kernel = cl::Kernel(program, "opencl_rotate");
			initialized = true;
		}
		
		cl::CommandQueue queue = cl.get_GPU_queue();
		cl::Event event = (cl::make_kernel<cl::Buffer, cl::Buffer, const long int, const size_t>(kernel))(cl::EnqueueArgs(queue, cl::NDRange(size)), ins, outs, rotation, size);
		event.wait();
	}
	
	template<typename T>
	void parallel_indices(cl::Buffer & buf, size_type size) {
		static const char* const starting_kernel_code =
			"__kernel void opencl_indices(__global %s *buf) \n"
			"{                                        \n"
			"	const size_t i = get_global_id(0);   \n"
			"	buf[i] = i;                          \n"
			"}";
		static cl::Program program;
		static cl::Kernel kernel;
		static bool initialized;
		cl_int err = CL_SUCCESS;
		if (!initialized) {
			const char *T_str = typeToStr<T>();
			if (T_str == nullptr) throw "Unsupported type in computation";
			char kernel_code[500];
			sprintf(kernel_code, starting_kernel_code, T_str);
			try {
				cl::Program::Sources kernel_source(1, std::make_pair(kernel_code,strlen(kernel_code)));
				program = cl::Program(cl.get_GPU_context(), std::string(kernel_code), true);
			} catch (cl::Error & err) {
				throw "Error encountered during OpenCL compilation";
			}
			kernel = cl::Kernel(program, "opencl_indices");
			initialized = true;
		}
		
		cl::CommandQueue queue = cl.get_GPU_queue();
		cl::Event event = (cl::make_kernel<cl::Buffer>(kernel))(cl::EnqueueArgs(queue, cl::NDRange(size)), buf);
		event.wait();
	}
	
	template<class T>
	class Vector {
		static_assert(std::is_same<T, bool>::value        || std::is_same<T, char>::value || 
		              std::is_same<T, signed char>::value || std::is_same<T, unsigned char>::value || 
		              std::is_same<T, short>::value       || std::is_same<T, uint16_t>::value || 
		              std::is_same<T, int>::value         || std::is_same<T, unsigned int>::value || 
		              std::is_same<T, long>::value        || std::is_same<T, unsigned long>::value || 
		              std::is_same<T, long long>::value   || std::is_same<T, unsigned long long>::value || 
		              std::is_same<T, float>::value, "Unsupported ParallelVector datatype");
		public:
			// CONSTRUCTORS
			// default constructor
			Vector() : num_filled(0), num_allocated(0), initialized(false) {};
			
			// fill constructors
			explicit Vector(size_type length) : data(cl.GPU_buffer<T>(length)), num_filled(length), num_allocated(length), initialized(true) {};
			explicit Vector(size_type length, T fill_value) : data(cl.GPU_buffer<T>(length, fill_value)), num_filled(length), num_allocated(length), initialized(true) {};
			
			// range constructors
			template<class input_iterator_type>
			//typedef typename std::iterator<std::input_iterator_tag, T> input_iterator_type;
			Vector(input_iterator_type begin, input_iterator_type end) : data(cl.GPU_buffer_iter(begin, end)), num_filled(end-begin), num_allocated(end-begin), initialized(true) {};
			Vector(T* data_in, size_type length) : data(cl.GPU_buffer(data_in, length)), num_filled(length), num_allocated(length), initialized(true) {};
			
			// copy constructors
			Vector(const Vector& vec) {
				initialized = vec.initialized;
				if (vec.initialized) {
					data = cl.duplicate_buffer<T>(vec.data, vec.num_filled, vec.num_allocated);
					num_filled = vec.num_filled;
					num_allocated = vec.num_allocated;
				}
			};
			Vector(const std::vector<T>& vec) : data(cl.GPU_buffer_iter(vec.begin(), vec.end())), num_filled(vec.size()), num_allocated(vec.size()), initialized(true) {};
			
			// move constructor
			Vector(const Vector&& vec) : data(cl.move_buffer<T>(vec.data)), num_filled(vec.num_filled), num_allocated(vec.num_allocated), initialized(vec.initialized) {};
			
			// copy assignment
			Vector<T> & operator=(const Vector<T>& vec) {
				initialized = vec.initialized;
				if (vec.initialized) {
					data = cl.duplicate_buffer<T>(vec.data, vec.num_filled, vec.num_allocated);
					num_filled = vec.num_filled;
					num_allocated = vec.num_allocated;
				}
				return get_this();
			};
			
			// OPERATORS
			// data accessor(s)
			// accessor
			T operator[] (size_type index) {
				if (!initialized) throw "Vector not initialized";
				return cl.get_GPU_buffer_index<T>(data, index);
			}
			// getters
			T get(size_type index) {
				if (!initialized) throw "Vector not initialized";
				if (index < num_filled) {
					return cl.get_GPU_buffer_index<T>(data, index);
				} else throw "index out of range";
			}
			void get(size_type start_index, T* data_in, size_type length) {
				if (!initialized) throw "Vector not initialized";
				if (start_index + length > num_filled) throw  "cannot get indices beyond end of Vector";
				cl.from_GPU_buffer(data, start_index, data_in, length);
			}
			void get(size_type start_index, std::vector<T> & vec) {
				if (!initialized) throw "Vector not initialized";
				if (start_index + vec.size() > num_filled) throw  "cannot get indices beyond end of Vector";
				cl.from_GPU_buffer(data, start_index, vec);
			}
			template<class iterator_type>
			void get(size_type start_index, iterator_type begin, iterator_type end) {
				if (!initialized) throw "Vector not initialized";
				if (start_index + (end-begin) > num_filled) throw  "cannot get indices beyond end of Vector";
				cl.from_GPU_buffer(data, start_index, begin, end);
			}
			// setters
			void set(size_type index, T val) {
				if (!initialized) throw "Vector not initialized";
				if (index < num_filled) {
					cl.set_GPU_buffer_index<T>(data, index, val);
				} else throw "index out of range";
			}
			void set(size_type start_index, T* data_in, size_type length) {
				if (!initialized) throw "Vector not initialized";
				if (start_index + length > num_filled) throw  "cannot set indices beyond end of Vector";
				cl.to_GPU_buffer(data, start_index, data_in, length);
			}
			void set(size_type start_index, std::vector<T> & vec) {
				if (!initialized) throw "Vector not initialized";
				if (start_index + vec.size() > num_filled) throw  "cannot set indices beyond end of Vector";
				cl.to_GPU_buffer(data, start_index, vec);
			}
			template<class iterator_type>
			void set(size_type start_index, iterator_type begin, iterator_type end) {
				if (!initialized) throw "Vector not initialized";
				if (start_index + (end-begin) > num_filled) throw  "cannot set indices beyond end of Vector";
				cl.to_GPU_buffer(data, start_index, begin, end);
			}
			
			
			// arithmetic operators
			// plus
			Vector operator+ (const Vector& vec) {
				Vector<T> output(size());
				do_operation<T,T,T>(get_this(), vec, output, plus);
				return output;
			}
			// minus
			Vector operator- (const Vector& vec) {
				Vector<T> output(size());
				do_operation<T,T,T>(get_this(), vec, output, minus);
				return output;
			}
			// times
			Vector operator* (const Vector& vec) {
				Vector<T> output(size());
				do_operation<T,T,T>(get_this(), vec, output, times);
				return output;
			}
			// divide
			Vector operator/ (const Vector& vec) {
				Vector<T> output(size());
				do_operation<T,T,T>(get_this(), vec, output, divide);
				return output;
			}
			// mod
			Vector operator% (const Vector& vec) {
				Vector<T> output(size());
				do_operation<T,T,T>(get_this(), vec, output, mod);
				return output;
			}
			// negate
			Vector operator- () {
				Vector<T> output(size());
				do_operation<T,T>(get_this(), output, negate);
				return output;
			}
			// prefix increment
			void operator++ () {
				do_operation<T,T>(get_this(), get_this(), increment);
			}
			// postfix increment
			Vector operator++ (int value) {
				Vector<T> output(get_this());
				do_operation<T,T>(get_this(), get_this(), increment);
				return output;
			}
			// prefix decrement
			void operator-- () {
				do_operation<T,T>(get_this(), get_this(), decrement);
			}
			// postfix decrement
			Vector operator-- (int value) {
				Vector<T> output(get_this());
				do_operation<T,T>(get_this(), get_this(), decrement);
				return output;
			}
			// plus equals
			void operator+= (const Vector& vec) {
				do_operation<T,T,T>(get_this(), vec, get_this(), plus);
			}
			// minus equals
			void operator-= (const Vector& vec) {
				do_operation<T,T,T>(get_this(), vec, get_this(), minus);
			}
			// times equals
			void operator*= (const Vector& vec) {
				do_operation<T,T,T>(get_this(), vec, get_this(), times);
			}
			// divide equals
			void operator/= (const Vector& vec) {
				do_operation<T,T,T>(get_this(), vec, get_this(), divide);
			}
			// mod equals
			void operator%= (const Vector& vec) {
				do_operation<T,T,T>(get_this(), vec, get_this(), mod);
		}
			
			//comparison operators
			// equals
			Vector<bool> operator== (const Vector& vec) {
				Vector<bool> output(size());
				do_operation<T,T,bool>(get_this(), vec, output, equals);
				return output;
			}
			// not equals
			Vector<bool> operator!= (const Vector& vec) {
				Vector<bool> output(size());
				do_operation<T,T,bool>(get_this(), vec, output, not_equals);
				return output;
			}
			// greater than
			Vector<bool> operator> (const Vector& vec) {
				Vector<bool> output(size());
				do_operation<T,T,bool>(get_this(), vec, output, greater);
				return output;
			}
			// less than
			Vector<bool> operator< (const Vector& vec) {
				Vector<bool> output(size());
				do_operation<T,T,bool>(get_this(), vec, output, lesser);
				return output;
			}
			// greater than or equal to
			Vector<bool> operator>= (const Vector& vec) {
				Vector<bool> output(size());
				do_operation<T,T,bool>(get_this(), vec, output, greater_equal);
				return output;
			}
			// less than or equal to
			Vector<bool> operator<= (const Vector& vec) {
				Vector<bool> output(size());
				do_operation<T,T,bool>(get_this(), vec, output, lesser_equal);
				return output;
			}
			
			//logical operators
			// and
			Vector<bool> operator&& (const Vector& vec) {
				Vector<bool> output(size());
				do_operation<T,T,T>(get_this(), vec, output, logical_and);
				return output;
			}
			// or
			Vector<bool> operator|| (const Vector& vec) {
				Vector<bool> output(size());
				do_operation<T,T,T>(get_this(), vec, output, logical_or);
				return output;
			}
			// not
			Vector<bool> operator! () {
				Vector<bool> output(size());
				do_operation<T,T>(get_this(), output, logical_not);
				return output;
			}
			
			// bitwise operators
			// and
			Vector operator& (const Vector& vec) {
				Vector<T> output(size());
				do_operation<T,T,T>(get_this(), vec, output, bitwise_and);
				return output;
			}
			// and equals
			Vector operator&= (const Vector& vec) {
				do_operation<T,T,T>(get_this(), vec, get_this(), bitwise_and);
				return get_this();
			}
			// or
			Vector operator| (const Vector& vec) {
				Vector<T> output(size());
				do_operation<T,T,T>(get_this(), vec, output, bitwise_or);
				return output;
			}
			// or equals
			Vector operator|= (const Vector& vec) {
				do_operation<T,T,T>(get_this(), vec, get_this(), bitwise_or);
				return get_this();
			}
			// xor
			Vector operator^ (const Vector& vec) {
				Vector<T> output(size());
				do_operation<T,T,T>(get_this(), vec, output, bitwise_xor);
				return output;
			}
			// xor equals
			Vector operator^= (const Vector& vec) {
				do_operation<T,T,T>(get_this(), vec, get_this(), bitwise_xor);
				return get_this();
			}
			// not
			Vector operator~ () {
				Vector<T> output(size());
				do_operation<T,T>(get_this(), output, bitwise_not);
				return output;
			}
			// shift left
			Vector operator<< (const Vector& vec) {
				Vector<T> output(size());
				do_operation<T,T,T>(get_this(), vec, output, left_shift);
				return output;
			}
			// shift left equals
			Vector operator<<= (const Vector& vec) {
				do_operation<T,T,T>(get_this(), vec, get_this(), left_shift);
				return get_this();
			}
			// shift right
			Vector operator>> (const Vector& vec) {
				Vector<T> output(size());
				do_operation<T,T,T>(get_this(), vec, output, right_shift);
				return output;
			}
			// shift right equals
			Vector operator>>= (const Vector& vec) {
				do_operation<T,T,T>(get_this(), vec, get_this(), right_shift);
				return get_this();
			}
			
			// ternary / choose operation
			template<typename U>
			Vector<U> choose(const Vector<U>& B, const Vector<U>& C) {
				//static_assert(std::is_same<T, bool>::value, "choose operator requires Vector<bool>");
				Vector<U> output(size());
				do_operation<T,U,U,U>(get_this(), B, C, output, ternary);
				return output;
			}
			
			// REDUCTIONS
			// sum
			T sum() {
				if (!initialized) throw "Vector not initialized";
				return parallel_reduce<T>(data, num_filled, reduce_plus);
			}
			
			// product
			T product() {
				if (!initialized) throw "Vector not initialized";
				return parallel_reduce<T>(data, num_filled, reduce_times);
			}
			
			Vector<T> filterBy(const Vector<bool> & vec) {
				if (!initialized || !vec.initialized) throw "Vector not initialized";
				if (size() != vec.size()) throw "Vector size mismatch";
				Vector<T> output(size());
				output.num_filled = parallel_filter<T>(data, vec.data, output.data, size());
				return output;
			}
			
			// ROTATIONS
			Vector<T> rotateBy(long int rotation) {
				long int final_rotation = rotation % (long int)size();
				// guarantee rotation is between 0 and size - 1
				if (final_rotation < 0) final_rotation += size();
				Vector<T> output(size());
				parallel_rotate<T>(data, output.data, final_rotation, size());
				return output;
			}
			
			// OTHER METHODS
			// get size of vector
			size_type size() const {
				return num_filled;
			}
			// resize vector
			void resize(size_type new_size) {
				if (!initialized) init();
				else if (new_size != num_filled && new_size <= num_allocated) {
					num_filled = new_size;
				}
				else {
					copy_resize_buffer(num_filled, new_size);
					num_filled = new_size;
				}
			}
			// guarantee the vector has space for the given number of elements
			void reserve(size_type reservation) {
				if (!initialized) init();
				else if (reservation > num_allocated) copy_resize_buffer(num_filled, reservation);
			}
			// first element in vector
			T front() {
				if (!initialized) throw "Vector not initialized";
				if (num_filled > 0) return cl.get_GPU_buffer_index<T>(data, 0);
				else throw "Cannot get front of empty Vector";
			}
			// last element in vector
			T back() {
				if (!initialized) throw "Vector not initialized";
				if (num_filled > 0) return cl.get_GPU_buffer_index<T>(data, num_filled-1);
				else throw "Cannot get back of empty Vector";
			}
			// push an element onto the vector
			void push_back(T val) {
				if (!initialized) init();
				else if (num_allocated <= num_filled) copy_resize_buffer(num_filled, num_filled * 2);
				cl.set_GPU_buffer_index<T>(data, num_filled, val);
				++num_filled;
			}
			// removes the last element from the vector
			void pop_back() {
				if (!initialized) throw "Vector not initialized";
				if (num_filled > 0) --num_filled;
			}
			
			
			
			cl::Buffer data;
		protected:
			// so we can access protected methods accross templates
			template<typename U>
			friend class Vector;
			
			template<typename T1, typename T2, typename T3, typename T4>
			void do_operation(Vector<T1> & a, const Vector<T2> & b, const Vector<T3> & c, Vector<T4> & d, enum operation op) {
				if (!a.initialized || !b.initialized || !c.initialized || !d.initialized) throw "Vector not initialized";
				if (a.size() != b.size() || b.size() != c.size() || c.size() != d.size()) throw "Vector size mismatch";
				parallel_compute<T1, T2, T3, T4>(a.data, b.data, c.data, d.data, a.size(), op);
			}
			
			template<typename T1, typename T2, typename T3>
			void do_operation(Vector<T1> & a, const Vector<T2> & b, Vector<T3> & c, enum operation op) {
				if (!a.initialized || !b.initialized || !c.initialized) throw "Vector not initialized";
				if (a.size() != b.size() || b.size() != c.size()) throw "Vector size mismatch";
				parallel_compute<T1, T2, T3>(a.data, b.data, c.data, a.size(), op);
			}
			
			template<typename T1, typename T2>
			void do_operation(Vector<T1> & a, Vector<T2> & b, enum operation op) {
				if (!a.initialized || !b.initialized) throw "Vector not initialized";
				if (a.size() != b.size()) throw "Vector size mismatch";
				parallel_compute<T1, T2>(a.data, b.data, a.size(), op);
			}
			
			Vector<T> & get_this() {
				return (*this);
			}
			
			void init() {
				data = cl.GPU_buffer<T>(init_size);
				num_allocated = init_size;
			}
			
			void copy_resize_buffer(size_type copy_size, size_type new_size) {
				cl::Buffer buf = cl.GPU_buffer<T>(new_size);
				parallel_compute<T, T>(data, buf, copy_size, copy);
				num_allocated = new_size;
				data = buf;
			}
			
			size_type num_filled, num_allocated;
			bool initialized;
			const size_type init_size = 8;
	};
	
	template<typename T>
	Vector<T> indices_Vector(size_type size) {
		Vector<T> output(size);
		parallel_indices<T>(output.data, size);
		return output;
	}
}