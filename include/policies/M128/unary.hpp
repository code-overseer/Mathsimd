#ifndef MATHEMATICS_UNARY_HPP
#define MATHEMATICS_UNARY_HPP
#include "../M128.hpp"

template<typename T>
struct mathsimd::M128<T>::Unary
{
private:
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

#define PRIVATE_UNARY_FORWARD(FUNC) \
		template<size_t... Idx> \
		static T FUNC(T const& vector, std::index_sequence<Idx...>) \
		{ \
			T result; \
			(FUNC<Idx>(result, vector),...); \
			return result; \
		}

	PRIVATE_UNARY_FORWARD(reciprocal)
	PRIVATE_UNARY_FORWARD(sign)
	PRIVATE_UNARY_FORWARD(absolute)
	PRIVATE_UNARY_FORWARD(rsqrt)
	PRIVATE_UNARY_FORWARD(sqrt)
#undef PRIVATE_UNARY_FORWARD

public:
#define PUBLIC_UNARY_OP(FUNC) \
    static decltype(auto) FUNC(T const& vector) \
    { \
        return FUNC(vector, std::make_index_sequence<register_count>{}); \
    }

	PUBLIC_UNARY_OP(reciprocal)
	PUBLIC_UNARY_OP(sign)
	PUBLIC_UNARY_OP(absolute)
	PUBLIC_UNARY_OP(rsqrt)
	PUBLIC_UNARY_OP(sqrt)
#undef PUBLIC_UNARY_OP
};

#endif //MATHEMATICS_UNARY_HPP
