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
		static constexpr size_t ceil_div(size_t left, size_t right) { return 1 + ((left - 1) / right); }
		static constexpr bool pow_2(size_t value) { return !(value & (value - 1)); }
	public:
		static constexpr size_t min_alignment = 8;
		static constexpr size_t float_count = T::size();
		static constexpr size_t aligned_floats = T::aligned_floats();
		static constexpr size_t floats_per_register = sizeof(Register) / sizeof(float);
		static constexpr size_t register_count = (float_count / aligned_floats);
		static_assert(std::conjunction<std::is_convertible<T, float*>, // defined T::operator float*()
		    std::negation<std::is_pointer<T>>, // not a pointer
		    std::bool_constant<pow_2(alignof(T)) && (alignof(T) >= min_alignment)>, // alignment must be power of 2 and greater than 8
			std::bool_constant<register_count>>::value); //  at least 1 register

		template<size_t N = 0>
		static Register load(T const& vector)
		{
			if constexpr(alignof(T) > min_alignment)
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
			if constexpr (alignof(T) > min_alignment)
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

	private:
		template<size_t N,
			bool IsAligned = !(alignof(T) % alignof(Register)),
			bool IsExcess = float_count % floats_per_register>
		struct RegisterIO;
	};
}

#include "M128/binary.hpp"
#include "M128/comparison.hpp"
#include "M128/register_io.hpp"
#include "M128/recursion.hpp"
#include "M128/unary.hpp"

#endif //MATHEMATICS_M128_HPP
