#ifndef MATHEMATICS_RECURSION_HPP
#define MATHEMATICS_RECURSION_HPP

#include "../M128.hpp"
template<typename T>
struct mathsimd::M128<T>::Recursion
{
private:
	template<size_t... N>
	struct recursion_helper;

	template<> // todo deal with register_count > 16
	struct recursion_helper<> // Entry point
	{
		template<size_t... Idx>
		static float minimum(T const& vector, std::index_sequence<Idx...> seq)
		{
			static constexpr size_t pair_count = decltype(seq)::size();
			if constexpr (register_count == 1)
			{
				return recursion_helper<1>::minimum(load<0>(vector));
			}
			else if constexpr (register_count % 2)
			{
				return recursion_helper<pair_count + 1>::
				minimum(_mm_min_ps(load<Idx * 2>(vector), load<Idx * 2 + 1>(vector))..., load<register_count - 1>(vector));
			}
			else
			{
				return recursion_helper<pair_count>::
				minimum(_mm_min_ps(load<Idx * 2>(vector), load<Idx * 2 + 1>(vector))...);
			}
		}

		static float minimum(T const& vector)
		{
			return minimum(vector, std::make_index_sequence<(register_count / 2)>{});
		}
	};

	template<size_t N>
	struct recursion_helper<N> // Common case
	{
		template <int I, class... Ts>
		static decltype(auto) get(Ts&&... ts)
		{
			return std::get<I>(std::forward_as_tuple(ts...));
		}

		template<size_t... Idx, typename... Reg>
		static float minimum(std::index_sequence<Idx...> seq, Reg&&... regs)
		{
			static constexpr size_t pair_count = decltype(seq)::size();
			if constexpr (N % 2)
			{
				return recursion_helper<pair_count + 1>::
				minimum(_mm_min_ps(get<2 * Idx>(regs...), get<2 * Idx + 1>(regs...))..., std::get<N - 1>(regs...));
			}
			else
			{
				return recursion_helper<pair_count>::
				minimum(_mm_min_ps(get<2 * Idx>(regs...), get<2 * Idx + 1>(regs...))...);
			}
		}

		template<typename... Reg>
		static float minimum(Reg&&... regs)
		{
			return minimum(std::make_index_sequence<(N / 2)>{}, std::forward<Reg>(regs)...);
		}
	};

	template<>
	struct recursion_helper<1> // Termination
	{
		static float minimum(Register v)
		{
			return minimum<aligned_floats>(v);
		}

		template<size_t FloatCount>
		static float minimum(Register v);

		template<>
		static float minimum<1>(Register v)
		{
			float f;
			store(f, v);
			return f;
		}

		template<>
		static float minimum<2>(Register v)
		{
			float f;
			v = _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 1, 0, 1)));
			store(f, v);
			return f;
		}

		template<>
		static float minimum<3>(Register v)
		{
			float f;
			v = _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 2, 0, 1)));
			v = _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 0, 2, 2)));
			store(f,v);
			return f;
		}

		template<>
		static float minimum<4>(Register v)
		{
			float f;
			v = _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 1, 0, 3)));
			v = _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2)));
			store(f, v);
			return f;
		}
	};

public:
#define PUBLIC_UNARY_OP(FUNC) \
    static decltype(auto) FUNC(T const& vector) \
    { \
		return recursion_helper<>::FUNC(vector); \
    }

	PUBLIC_UNARY_OP(minimum)
//	PUBLIC_UNARY_OP(maximum)
//	PUBLIC_UNARY_OP(sum)
//	PUBLIC_UNARY_OP(difference)
#undef PUBLIC_UNARY_OP
};

#endif //MATHEMATICS_RECURSION_HPP
