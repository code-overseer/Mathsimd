#ifndef MATHEMATICS_SIMD_FLOAT2_HPP
#define MATHEMATICS_SIMD_FLOAT2_HPP

#include <immintrin.h>
#include "constants.hpp"
#include "bool.hpp"
#include "vector_operations.hpp"

namespace mathsimd 
{
    struct float2 : TensorBase<float, 2, 1>, private Operations<float2, SimdVectorPolicy<float>::M128>
    {
    protected:
        alignas(8) float _val[2]{0.f, 0.f};
    public:
        float2() = default;
        float2(float2&& other) noexcept = default;
        float2(float const &x, float const &y) : _val{x, y} {}
        float2(float2 const &other) { _mm_storeu_si64(_val, _mm_loadu_si64(other._val)); }
        float2(float const* other) { _mm_storeu_si64(_val, _mm_loadu_si64(other)); }
        float2(__m128 const &other) { _mm_storeu_si64(_val, _mm_castps_si128(other)); }
        float2(float const &x) { _mm_storeu_si64(_val, _mm_castps_si128(_mm_broadcast_ss(&x))); }
        inline operator float*() { return _val; }
        inline operator float const*() const { return _val; }
        inline operator __m128() const { return _mm_castsi128_ps(_mm_loadu_si64(_val)); }
        inline float2 &operator=(float2 const &other) = default;
		inline float2 &operator=(float2 &&other) noexcept = default;
        inline float2 &operator=(__m128 const &other) { _mm_storeu_si64(_val, _mm_castps_si128(other)); return *this; }
        inline float operator[](size_t const idx) const { return _val[idx]; }
        inline float& operator[](size_t const idx) { return _val[idx]; }
        float &x() { return _val[0]; }
        float &y() { return _val[1]; }
        [[nodiscard]] float x() const { return _val[0]; }
        [[nodiscard]] float y() const { return _val[1]; }

        #define FUNC(NAME,X,Y) \
        static inline float2 NAME () { return {X,Y}; }
        FUNC(up, 0,1)
        FUNC(down, 0,-1)
        FUNC(right, 1,0)
        FUNC(left, -1,0)
        FUNC(one, 1,1)
        FUNC(zero, 0,0)
        #undef FUNC
    };

    template<typename OperationPolicy>
    struct Float2 : float2, private Operations<Float2<OperationPolicy>, OperationPolicy>
	{
    	using base_type = float2;
    	using float2::float2;
		REBINDER(Float2, OperationPolicy)
	};

}
#endif