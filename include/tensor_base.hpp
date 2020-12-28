#ifndef MATHEMATICS_TENSOR_BASE_HPP
#define MATHEMATICS_TENSOR_BASE_HPP

#include <type_traits>
namespace mathsimd
{
	template<typename T, size_t R, size_t C>
	struct TensorBase
	{
		static_assert(std::is_arithmetic<T>::value);
		using numeric_type = T;
		static size_t constexpr rows() { return R; }
		static size_t constexpr cols() { return C; }
		static size_t constexpr length() { return R * C; };
		static size_t constexpr active_bytes() { return length() * sizeof(numeric_type); }
	};

}

#endif //MATHEMATICS_TENSOR_BASE_HPP
