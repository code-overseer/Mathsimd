#ifndef MATHEMATICS_SIMD_FLOAT3_HPP
#define MATHEMATICS_SIMD_FLOAT3_HPP

#include <immintrin.h>
#include <utility>
#include <cmath>
#include <cstdio>
#include <iostream>
#include "constants.hpp"

namespace mathsimd {

    struct float3 {
    private:
        alignas(16) float _val[3]{0, 0, 0};
    public:
        float3() = default;
        float3(float const &x, float const &y, float const &z) : _val{x, y, z} {}
        float3(float3 const &other) { memcpy(_val, other._val, 3 * sizeof(float)); }
        float3(__m128 const &other) { _mm_store_ps(_val, other); }
        inline operator __m128() const { return _mm_load_ps(_val); }
        inline float3 &operator=(float3 const &other) = default;
        inline float3 &operator=(__m128 const &other) { _mm_store_ps(_val, other); return *this; }
        float &x() { return _val[0]; }
        float &y() { return _val[1]; }
        float &z() { return _val[2]; }
        [[nodiscard]] float x() const { return _val[0]; }
        [[nodiscard]] float y() const { return _val[1]; }
        [[nodiscard]] float z() const { return _val[2]; }

        #define ARITHMETIC(OP) \
        friend float3 operator OP (float3 const &a, float3 const &b); \
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type *> \
        friend float3 operator OP (T const &a, float3 const &b); \
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type *> \
        friend float3 operator OP (float3 const &a, T const &b);
        ARITHMETIC(+)
        ARITHMETIC(-)
        ARITHMETIC(*)
        #undef ARITHMETIC
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type *>
        friend float3 operator / (float3 const &a, T const &b);

        static float dot(float3 const &a, float3 const &b) {
            auto c = a * b;
            return _mm_add_ss(_mm_add_ss(c, _mm_shuffle_ps(c,c,85)), _mm_unpackhi_ps(c,c))[0];
        }

        [[nodiscard]] inline float sqrMagnitude() const { return dot(*this, *this); }
        [[nodiscard]] inline float magnitude() const { 
            float f = sqrMagnitude(); 
            auto v = _mm_load_ss(&f);
            _mm_store_ss(&f, _mm_mul_ss(v, _mm_rsqrt_ss(v)));
            return f;
        }
        [[nodiscard]] inline float3 normalized() const {
            auto f = sqrMagnitude();
            return _mm_mul_ps( *this, _mm_rsqrt_ps(_mm_load_ps1(&f)) );
        }
        

        static inline float3 cross(float3 const &a, float3 const &b) {
            auto tmp0 = _mm_shuffle_ps(a,a,_MM_SHUFFLE(3,0,2,1));
            auto tmp1 = _mm_shuffle_ps(b,b,_MM_SHUFFLE(3,1,0,2));
            auto tmp2 = _mm_shuffle_ps(a,a,_MM_SHUFFLE(3,1,0,2));
            auto tmp3 = _mm_shuffle_ps(b,b,_MM_SHUFFLE(3,0,2,1));
            return _mm_sub_ps(_mm_mul_ps(tmp0,tmp1),_mm_mul_ps(tmp2,tmp3));
        }

        #define FUNC(NAME,X,Y,Z) \
        static inline float3 NAME () { return {X,Y,Z}; }
        FUNC(up, 0,1,0)
        FUNC(down, 0,-1,0)
        FUNC(right, 1,0,0)
        FUNC(left, -1,0,0)
        FUNC(forward, 0,0,1)
        FUNC(back, 0,0,-1)
        FUNC(one, 1,1,1)
        FUNC(zero, 0,0,0)
        #undef FUNC
    };

    #define ARITHMETIC(OP) \
        inline float3 operator OP (float3 const &a, float3 const &b) { return static_cast<__m128>(a) OP static_cast<__m128>(b); } \
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr> \
        inline float3 operator OP (T const &a, float3 const &b) { return static_cast<float>(a) OP static_cast<__m128>(b); } \
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr> \
        inline float3 operator OP (float3 const &a, T const &b) { return static_cast<__m128>(a) OP static_cast<float>(b); }
    ARITHMETIC(+)
    ARITHMETIC(-)
    ARITHMETIC(*)
    #undef ARITHMETIC
    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    inline float3 operator / (float3 const &a, T const &b) { return static_cast<__m128>(a) / static_cast<float>(b); }

    inline bool operator==(float3 const &a, float3 const &b) {
        auto tmp = _mm_abs_ps(static_cast<__m128>(a) - static_cast<__m128>(b));
        return _mm_movemask_epi8(_mm_castps_si128(tmp < EPSILON_F)) == 0xffff;
    }

    inline bool operator!=(float3 const &a, float3 const &b) { return !(a == b); }

    inline std::ostream& operator << (std::ostream& stream, float3 const &input) {
        stream << "(" << input.x() << ", " << input.y() << ", " << input.z() <<')';
        return stream;
    }

}
#endif //MATHEMATICS_SIMD_FLOAT3_HPP