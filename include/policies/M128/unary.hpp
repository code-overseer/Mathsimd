#ifndef MATHEMATICS_UNARY_HPP
#define MATHEMATICS_UNARY_HPP
#include "../M128.hpp"

template<typename T>
struct mathsimd::M128<T>::Unary
{
	template<size_t N>
	static void reciprocal(T& result, T const& vector)
	{
		store<N>(result, _mm_rcp_ps(load<N>(vector)));
	}

	template<size_t N>
	static void sign(T& result, T const& vector)
	{
		auto reg = _mm_castps_si128(load<N>(vector));
		auto ones = _mm_castps_si128(broadcast(1.0f));
		store<N>(result, _mm_castsi128_ps(_mm_sign_epi32(ones, reg)));
	}

	template<size_t N>
	static void absolute(T& result, T const& vector)
	{
		store<N>(result, _mm_castsi128_ps(_mm_abs_epi32(_mm_castps_si128(load<N>(vector)))));
	}

	template<size_t N>
	static void rsqrt(T& result, T const& vector)
	{
		store<N>(result, _mm_rsqrt_ps(load<N>(vector)));
	}

	template<size_t N>
	static void sqrt(T& result, T const& vector)
	{
		store<N>(result, _mm_sqrt_ps(load<N>(vector)));
	}

	static void reciprocal(float& result, float const& scalar)
	{
		store(result, _mm_rcp_ps(load(scalar)));
	}

	static void sign(float& result, float const& scalar)
	{
		auto reg = _mm_castps_si128(load(scalar));
		auto ones = _mm_castps_si128(broadcast(1.0f));
		store(result, _mm_castsi128_ps(_mm_sign_epi32(ones, reg)));
	}

	static void absolute(float& result, float const& scalar)
	{
		store(result, _mm_castsi128_ps(_mm_abs_epi32(_mm_castps_si128(load(scalar)))));
	}

	static void rsqrt(float& result, float const& scalar)
	{
		store(result, _mm_rsqrt_ps(load(scalar)));
	}

	static void sqrt(float& result, float const& scalar)
	{
		store(result, _mm_sqrt_ps(load(scalar)));
	}

#define HELPER_FUNC(FUNC) \
		template<size_t... Idx> \
		static T FUNC(T const& vector, std::index_sequence<Idx...>) \
		{ \
			T result; \
			(FUNC<Idx>(result, vector),...); \
			return result; \
		}

	HELPER_FUNC(reciprocal)
	HELPER_FUNC(sign)
	HELPER_FUNC(absolute)
	HELPER_FUNC(rsqrt)
	HELPER_FUNC(sqrt)
#undef PRIVATE_UNARY_FORWARD
};

#define UNARY_OPS(FUNC) \
template<typename T> \
decltype(auto) mathsimd::M128<T>::FUNC(T const& vector) \
{ \
	return Unary::FUNC(vector, std::make_index_sequence<register_count>{}); \
}

UNARY_OPS(reciprocal)
UNARY_OPS(sign)
UNARY_OPS(absolute)
UNARY_OPS(rsqrt)
UNARY_OPS(sqrt)
#undef UNARY_OPS
#define UNARY_OPS(FUNC) \
template<typename T> \
decltype(auto) mathsimd::M128<T>::FUNC(float const& scalar) \
{ \
	return Unary::FUNC(scalar); \
}

UNARY_OPS(reciprocal)
UNARY_OPS(sign)
UNARY_OPS(absolute)
UNARY_OPS(rsqrt)
UNARY_OPS(sqrt)
#undef UNARY_OPS

#endif //MATHEMATICS_UNARY_HPP
