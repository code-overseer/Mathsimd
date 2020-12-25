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
		static constexpr size_t rows = R;
		static constexpr size_t cols = C;
		static constexpr size_t length = R * C;
	};

}

#endif //MATHEMATICS_TENSOR_BASE_HPP
