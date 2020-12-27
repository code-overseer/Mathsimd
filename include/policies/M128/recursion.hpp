#ifndef MATHEMATICS_RECURSION_HPP
#define MATHEMATICS_RECURSION_HPP

#include "../M128.hpp"
#include <tuple>

/// Only works with register_count <= 16, I don't know what will happen if you exceed this limit
template<typename T>
struct mathsimd::M128<T>::Recursion
{
private:

	// Compiler should optimize out the Register array[]
#define HELPER_FUNC(FUNC,OP) \
	template<size_t N,size_t... Idx> \
	static float FUNC(std::index_sequence<Idx...> seq, Register (&regs)[N]) \
	{ \
		if constexpr (N % 2) \
		{ \
			Register array[]{_mm_##OP##_ps(regs[2 * Idx], regs[2 * Idx + 1])..., regs[N - 1]}; \
			return FUNC(array); \
		} \
		else \
		{ \
			Register array[]{_mm_##OP##_ps(regs[2 * Idx], regs[2 * Idx + 1])...}; \
			return FUNC(array); \
		} \
	} \
	template<size_t N> \
	static float FUNC(Register (&regs)[N]) \
	{ \
		if constexpr (N != 1) \
		{ \
			return FUNC(std::make_index_sequence<(N / 2)>{}, regs); \
		} \
		else \
		{ \
			return FUNC(regs[0]); \
		} \
	} \
	static float FUNC(Register v) \
	{ \
		if constexpr (aligned_floats == 1) \
		{ \
			float f; \
			_mm_store_ss(&f, v); \
			return f; \
		} \
		else if constexpr (aligned_floats == 2) \
		{ \
			float f; \
			v = _mm_##OP##_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 1, 0, 1))); \
			_mm_store_ss(&f, v); \
			return f; \
		} \
		else if constexpr (aligned_floats == 3) \
		{ \
			float f; \
			v = _mm_##OP##_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 2, 0, 1))); \
			v = _mm_##OP##_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 0, 2, 2))); \
			_mm_store_ss(&f, v); \
			return f; \
		} \
		else \
		{ \
			float f; \
			v = _mm_##OP##_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 1, 0, 3))); \
			v = _mm_##OP##_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2))); \
			_mm_store_ss(&f, v); \
			return f; \
		} \
	} \
	template<size_t... Idx> \
	static float FUNC(std::index_sequence<Idx...>, T const& vector) \
	{ \
		if constexpr (register_count % 2) \
		{ \
			Register array[]{_mm_##OP##_ps(load<Idx * 2>(vector), load<Idx * 2 + 1>(vector))..., load<register_count - 1>(vector)}; \
			return FUNC(array); \
		} \
		else \
		{ \
			Register array[]{_mm_##OP##_ps(load<Idx * 2>(vector), load<Idx * 2 + 1>(vector))...}; \
			return FUNC(array); \
		} \
	} \
	static float FUNC(std::index_sequence<>, T const& vector) \
	{ \
		return FUNC(load<0>(vector)); \
	}

	HELPER_FUNC(minimum,min)
	HELPER_FUNC(maximum,max)
	HELPER_FUNC(sum,add)
	HELPER_FUNC(difference,sub)
#undef HELPER_FUNC
	template<size_t... Idx> 
	static float sum_product(std::index_sequence<Idx...>, T const& left, T const& right)
	{
		if constexpr (register_count % 2) 
		{
			Register rights[]{ _mm_mul_ps(load<Idx * 2 + 1>(left), load<Idx * 2 + 1>(right))...};
			Register array[]{_mm_fmadd_ps(load<Idx * 2>(left), load<Idx * 2>(right), rights[Idx])...,
					_mm_mul_ps(load<register_count - 1>(left), load<register_count - 1>(right))};
			return FUNC(array); 
		} 
		else 
		{
			Register rights[]{ _mm_mul_ps(load<Idx * 2 + 1>(left), load<Idx * 2 + 1>(right))...};
			Register array[]{_mm_fmadd_ps(load<Idx * 2>(left), load<Idx * 2>(right), rights[Idx])...};
			return FUNC(array); 
		} 
	} 
	static float sum_product(std::index_sequence<>, T const& left, T const& right)
	{ 
		return sum(_mm_mul_ps(load<0>(left), load<0>(right)));
	}
	template<size_t... Idx>
	static float diff_product(std::index_sequence<Idx...>, T const& left, T const& right)
	{
		if constexpr (register_count % 2)
		{
			Register rights[]{ _mm_mul_ps(load<Idx * 2 + 1>(left), load<Idx * 2 + 1>(right))...};
			Register array[]{_mm_fmsub_ps(load<Idx * 2>(left), load<Idx * 2>(right), rights[Idx])...,
							 _mm_mul_ps(load<register_count - 1>(left), load<register_count - 1>(right))};
			return FUNC(array);
		}
		else
		{
			Register rights[]{ _mm_mul_ps(load<Idx * 2 + 1>(left), load<Idx * 2 + 1>(right))...};
			Register array[]{_mm_fmadd_ps(load<Idx * 2>(left), load<Idx * 2>(right), rights[Idx])...};
			return FUNC(array);
		}
	}
	static float diff_product(std::index_sequence<>, T const& left, T const& right)
	{
		return difference(_mm_mul_ps(load<0>(left), load<0>(right)));
	}

public:
#define PUBLIC_UNARY_OP(FUNC) \
    static decltype(auto) FUNC(T const& vector) \
    { \
		return FUNC(std::make_index_sequence<(register_count / 2)>{}, vector); \
    }
	PUBLIC_UNARY_OP(minimum)
	PUBLIC_UNARY_OP(maximum)
	PUBLIC_UNARY_OP(sum)
	PUBLIC_UNARY_OP(difference)
#undef PUBLIC_UNARY_OP
};

#endif //MATHEMATICS_RECURSION_HPP
