#ifndef MATHEMATICS_SIMD_FLOAT2_HPP
#define MATHEMATICS_SIMD_FLOAT2_HPP

#include <immintrin.h>
#include <array>
#include <cmath>
#include <iostream>
#include "constants.hpp"

namespace mathsimd {
    struct float2 {
    private:
        alignas(8) float _val[2]{0.f, 0.f};
    public:
        float2() = default;
        float2(float const &x, float const &y) : _val{x, y} {}
        float2(float const* other) { memcpy(_val, other, 2 * sizeof(float)); }
        inline operator __m128() const { return _mm_castsi128_ps(_mm_loadu_si64(_val)); }
        inline operator float const*() const { return _val; }
        float2(float2 const &other) { memcpy(_val, other._val, 2 * sizeof(float) );}
        float2(__m128 const &other) { _mm_storeu_si64(_val, _mm_castps_si128(other)); }
        inline float2 &operator=(float2 const &other) = default;
        inline float2 &operator=(__m128 const &other) { _mm_storeu_si64(_val, _mm_castps_si128(other)); return *this; }
        float &x() { return _val[0]; }
        float &y() { return _val[1]; }
        [[nodiscard]] float x() const { return _val[0]; }
        [[nodiscard]] float y() const { return _val[1]; }

        #define ARITHMETIC(OP) \
        friend float2 operator OP (float2 const &a, float2 const &b); \
        friend float2 operator OP (float const &a, float2 const &b); \
        friend float2 operator OP (float2 const &a, float const &b);
        ARITHMETIC(+)
        ARITHMETIC(-)
        ARITHMETIC(*)
        ARITHMETIC(/)
        #undef ARITHMETIC

        friend float dot(float2 const &a, float2 const &b);

        [[nodiscard]] inline float sqrMagnitude() const { return dot(*this, *this); }
        [[nodiscard]] inline float magnitude() const { 
            float f = sqrMagnitude(); 
            auto v = _mm_load_ss(&f);
            _mm_store_ss(&f, _mm_mul_ss(v, _mm_rsqrt_ss(v)));
            return f;
        }
        [[nodiscard]] inline float2 normalized() const {
            auto fl = sqrMagnitude();
            return _mm_mul_ps( *this, _mm_rsqrt_ps(_mm_load_ps1(&fl)) );
        }

        static inline float2 minimum(float2 const& l, float2 const & r) {
            auto c = _mm_cmplt_ps(l, r);
            return _mm_or_ps(_mm_and_ps(c,l), _mm_andnot_ps(c,r));
        }

        static inline float2 maximum(float2 const& l, float2 const & r) {
            auto c = _mm_cmpgt_ps(l, r);
            return _mm_or_ps(_mm_and_ps(c,l), _mm_andnot_ps(c,r));
        }

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

}
#endif