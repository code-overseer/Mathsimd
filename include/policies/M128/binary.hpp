#ifndef MATHEMATICS_BINARY_HPP
#define MATHEMATICS_BINARY_HPP

#include "../M128.hpp"


template<typename T>
struct mathsimd::M128<T>::Binary
{
#define HELPER_FUNC(FUNC,OP) \
private: \
	template<size_t N> \
	static void FUNC(T& result, T const& left, T const& right) \
	{ \
		store<N>(result, _mm_##OP##_ps(load<N>(left), load<N>(right))); \
	} \
	template<size_t... Idx> \
	static T FUNC(T const& left, T const& right, std::index_sequence<Idx...>) \
	{ \
		T result; \
		(FUNC<Idx>(result, left, right),...); \
		return result; \
	} \
	template<size_t N> \
	static void FUNC(T const& result, T const& left, Register const& right) \
	{ \
		store<N>(result, _mm_##OP##_ps(load<N>(left), right)); \
	} \
	template<size_t N> \
	static void FUNC(T const& result, Register const& left, T const& right) \
	{ \
		store<N>(result, _mm_##OP##_ps(left, load<N>(right))); \
	} \
	template<size_t... Idx> \
	static T FUNC(T const& left, float const& right, std::index_sequence<Idx...>) \
	{ \
		T result; \
		(FUNC<Idx>(result, left, broadcast(right)),...); \
		return result; \
	} \
	template<size_t... Idx> \
	static T FUNC(float const& left, T const& right, std::index_sequence<Idx...>) \
	{ \
		T result; \
		(FUNC<Idx>(result, broadcast(left), right),...); \
		return result; \
	} \

	HELPER_FUNC(add, add)
	HELPER_FUNC(subtract, sub)
	HELPER_FUNC(multiply, mul)
	HELPER_FUNC(divide, div)
	HELPER_FUNC(minimum, min)
	HELPER_FUNC(maximum, max)
#undef HELPER_FUNC

};

#define BINARY_OPS(FUNC) \
template<typename T> \
decltype(auto) mathsimd::M128<T>::FUNC(T const& left, T const& right) \
{ \
	return Binary::FUNC(left, right, std::make_index_sequence<register_count>{}); \
} \
template<typename T> \
decltype(auto) mathsimd::M128<T>::FUNC(T const& left, float const& right) \
{ \
	return Binary::FUNC(left, right, std::make_index_sequence<register_count>{}); \
} \
template<typename T> \
decltype(auto) mathsimd::M128<T>::FUNC(float const& left, T const& right) \
{ \
	return Binary::FUNC(left, right, std::make_index_sequence<register_count>{}); \
}

BINARY_OPS(add)
BINARY_OPS(subtract)
BINARY_OPS(multiply)
BINARY_OPS(divide)
BINARY_OPS(maximum)
BINARY_OPS(minimum)

#undef BINARY_OPS


#endif //MATHEMATICS_BINARY_HPP
