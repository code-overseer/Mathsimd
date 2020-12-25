#ifndef MATHEMATICS_BINARY_HPP
#define MATHEMATICS_BINARY_HPP

#include "../M128.hpp"

template<typename T>
struct mathsimd::M128<T>::Binary
{
#define BINARY_OP(FUNC,OP) \
private: \
	template<size_t N> \
	static void FUNC(T const& result, T const& left, T const& right) \
	{ \
		store<N>(result, _mm_##OP##_ps(load<N>(left), load<N>(right))); \
	} \
	template<size_t... Idx> \
	static typename std::decay<T>::type FUNC(T const& left, T const& right, std::index_sequence<Idx...>) \
	{ \
		typename std::decay<T>::type result; \
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
	static typename std::decay<T>::type FUNC(T const& left, float const& right, std::index_sequence<Idx...>) \
	{ \
		typename std::decay<T>::type result; \
		(FUNC<Idx>(result, left, broadcast(right)),...); \
		return result; \
	} \
	template<size_t... Idx> \
	static typename std::decay<T>::type FUNC(float const& left, T const& right, std::index_sequence<Idx...>) \
	{ \
		typename std::decay<T>::type result; \
		(FUNC<Idx>(result, broadcast(left), right),...); \
		return result; \
	} \
public: \
	static decltype(auto) FUNC(T const& left, T const& right) \
	{ \
		return FUNC(left, right, std::make_index_sequence<register_count>{}); \
	} \
	static decltype(auto) FUNC(T const& left, float const& right) \
	{ \
		return FUNC(left, right, std::make_index_sequence<register_count>{}); \
	} \
	static decltype(auto) FUNC(float const& left, T const& right) \
	{ \
		return FUNC(left, right, std::make_index_sequence<register_count>{}); \
	}

	BINARY_OP(add, add)
	BINARY_OP(subtract, sub)
	BINARY_OP(multiply, mul)
	BINARY_OP(divide, div)
	BINARY_OP(minimum, min)
	BINARY_OP(maximum, max)
#undef BINARY_OP

};


#endif //MATHEMATICS_BINARY_HPP
