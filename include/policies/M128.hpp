#ifndef MATHEMATICS_M128_HPP
#define MATHEMATICS_M128_HPP
#include <immintrin.h>
#include <type_traits>
#include <utility>
#include "../utility.hpp"

namespace mathsimd
{
	template<typename T>
	struct M128
	{
		using Register = __m128;
	private:
		static size_t constexpr ceil_div(size_t left, size_t right) { return 1 + ((left - 1) / right); }
		static bool constexpr is_pow2(size_t value) { return !(value & (value - 1)); }
		static float const* cast(T const& v) { return v; }
	public:
		static constexpr size_t min_alignment = 8;
		static constexpr size_t float_count = T::size();
		static constexpr size_t alignment = alignof(T);
		static constexpr size_t aligned_floats = T::aligned_floats() / (alignment / alignof(Register));
		static constexpr size_t floats_per_register = sizeof(Register) / sizeof(float);
		static constexpr size_t register_count = (float_count / aligned_floats);
		static_assert(std::conjunction<std::is_convertible<T, float const*>, // defined T::operator float const*()
		    std::bool_constant<is_pow2(alignment) && (alignment >= min_alignment)>, // alignment must be power of 2 and greater than 8
			std::bool_constant<(register_count > 0)>, //  at least 1 register
			std::bool_constant<(alignment > alignof(Register) && aligned_floats != 4)>>::value); // prevents alternating float count


		template<size_t N = 0>
		static Register load(T const& vector)
		{
			if constexpr (alignment > min_alignment)
			{
				return _mm_load_ps(cast(vector) + N * floats_per_register);
			}
			else
			{
				return _mm_castsi128_ps(_mm_loadu_si64(cast(vector) + N * floats_per_register));
			}
		}

		template<size_t N = 0>
		static void store(T& dst, Register const& src)
		{
			if constexpr (alignment > min_alignment)
			{
				_mm_store_ps(cast(dst) + N * floats_per_register, src);
			}
			else // float2 special-case
			{
				_mm_storeu_si64(cast(dst), _mm_castps_si128(src));
			}
		}

		static Register load(float const& value)
		{
			return _mm_load_ss(&value);
		}

		static Register broadcast(float const& value)
		{
			return _mm_broadcast_ss(&value);
		}

		static void store(float& dst, Register const& src)
		{
			_mm_store_ss(&dst, src);
		}

		struct Binary;
		template<typename BitField>
		struct Comparison;
		struct Unary;
		struct Recursion;
	};
}

#include "M128/binary.hpp"
#include "M128/comparison.hpp"
#include "M128/recursion.hpp"
#include "M128/unary.hpp"

#endif //MATHEMATICS_M128_HPP
