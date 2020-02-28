#ifndef MATHEMATICS_SIMD_FLOAT4_HPP
#define MATHEMATICS_SIMD_FLOAT4_HPP

#include <immintrin.h>
#include <utility>
#include <cmath>
#include <cstdio>
#include <iostream>
#include "constants.hpp"

namespace mathsimd {

    struct float4 {
    private:
        alignas(16) float _val[4]{0.f, 0.f, 0.f, 0.f};
    public:
        float4() = default;
        float4(float const &x, float const &y, float const &z, float const &w) : _val{x, y, z, w} {}
        float4(float4 const &other) { memcpy(_val, other._val, 4 * sizeof(float)); }
        float4(__m128 const &other) { _mm_store_ps(_val, other); }
        inline operator __m128() const { return _mm_load_ps(_val); }
        inline float4 &operator=(float4 const &other) = default;
        inline float4 &operator=(__m128 const &other) { _mm_store_ps(_val, other); return *this; }
        float &x() { return _val[0]; }
        float &y() { return _val[1]; }
        float &z() { return _val[2]; }
        float &w() { return _val[3]; }
        [[nodiscard]] float x() const { return _val[0]; }
        [[nodiscard]] float y() const { return _val[1]; }
        [[nodiscard]] float z() const { return _val[2]; }
        [[nodiscard]] float w() const { return _val[3]; }

        #define ARITHMETIC(OP) \
        friend float4 operator OP (float4 const &a, float4 const &b); \
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type *> \
        friend float4 operator OP (T const &a, float4 const &b); \
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type *> \
        friend float4 operator OP (float4 const &a, T const &b);
        ARITHMETIC(+)
        ARITHMETIC(-)
        ARITHMETIC(*)
        #undef ARITHMETIC
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type *>
        friend float4 operator / (float4 const &a, T const &b);

        static float dot(float4 const &a, float4 const &b);

        [[nodiscard]] inline float sqrMagnitude() const { return dot(*this, *this); }
        [[nodiscard]] inline float magnitude() const { 
            float f = sqrMagnitude(); 
            auto v = _mm_load_ss(&f);
            _mm_store_ss(&f, _mm_mul_ss(v, _mm_rsqrt_ss(v)));
            return f;
        }
        [[nodiscard]] inline float4 normalized() const {
            auto f = sqrMagnitude();
            return _mm_mul_ps( *this, _mm_rsqrt_ps(_mm_load_ps1(&f)) );
        }

        static inline float4 cross(float4 const &a, float4 const &b) {
            auto tmp0 = _mm_shuffle_ps(a,a,_MM_SHUFFLE(3,0,2,1));
            auto tmp1 = _mm_shuffle_ps(b,b,_MM_SHUFFLE(3,1,0,2));
            auto tmp2 = _mm_shuffle_ps(a,a,_MM_SHUFFLE(3,1,0,2));
            auto tmp3 = _mm_shuffle_ps(b,b,_MM_SHUFFLE(3,0,2,1));
            return _mm_sub_ps(_mm_mul_ps(tmp0,tmp1),_mm_mul_ps(tmp2,tmp3));
        }

        #define FUNC(NAME,X,Y,Z,W) \
        static inline float4 NAME () { return {X,Y,Z,W}; }
        FUNC(up, 0,1,0,0)
        FUNC(down, 0,-1,0,0)
        FUNC(right, 1,0,0,0)
        FUNC(left, -1,0,0,0)
        FUNC(forward, 0,0,1,0)
        FUNC(back, 0,0,-1,0)
        FUNC(in, 0,0,0,1)
        FUNC(out, 0,0,0,-1)
        FUNC(one, 1,1,1,1)
        FUNC(zero, 0,0,0,0)
        #undef FUNC
    };

    #define ARITHMETIC(OP) \
        inline float4 operator OP (float4 const &a, float4 const &b) { return static_cast<__m128>(a) OP static_cast<__m128>(b); } \
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr> \
        inline float4 operator OP (T const &a, float4 const &b) { return static_cast<float>(a) OP static_cast<__m128>(b); } \
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr> \
        inline float4 operator OP (float4 const &a, T const &b) { return static_cast<__m128>(a) OP static_cast<float>(b); }
    ARITHMETIC(+)
    ARITHMETIC(-)
    ARITHMETIC(*)
    #undef ARITHMETIC
    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    inline float4 operator / (float4 const &a, T const &b) { return static_cast<__m128>(a) / static_cast<float>(b); }

    inline bool operator==(float4 const &a, float4 const &b) {
        auto tmp = _mm_abs_ps(static_cast<__m128>(a) - static_cast<__m128>(b));
        return _mm_movemask_epi8(_mm_castps_si128(tmp < EPSILON_F)) == 0xffff;
    }

    inline bool operator!=(float4 const &a, float4 const &b) { return !(a == b); }

    inline std::ostream& operator << (std::ostream& stream, float4 const &input) {
        stream << "(" << input.x() << ", " << input.y() << ", " << input.z() << ", " << input.w() <<')';
        return stream;
    }

    inline float float4::dot(const mathsimd::float4 &a, const mathsimd::float4 &b) {
        float f;
        auto c = a * b;
        auto tmp = _mm_add_ps(c, _mm_permute_ps(c, 78));
        _mm_store_ss(&f, _mm_add_ss(tmp, _mm_permute_ps(tmp, 85)));
        return f;
    }
}
#endif //MATHEMATICS_SIMD_FLOAT4_HPP