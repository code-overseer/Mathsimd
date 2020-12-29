#ifndef MATHEMATICS_TENSOR_BASE_HPP
#define MATHEMATICS_TENSOR_BASE_HPP

#include <type_traits>
namespace mathsimd
{
	template<size_t R, size_t C>
	struct MatrixBase
	{;
		static size_t constexpr rows() { return R; }
		static size_t constexpr cols() { return C; }
	};

}

#endif //MATHEMATICS_TENSOR_BASE_HPP
