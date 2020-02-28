#ifndef MATHEMATICS_SIMD_FLOAT4X4_HPP
#define MATHEMATICS_SIMD_FLOAT4X4_HPP

#include <immintrin.h>
#include <cmath>
#include <array>
#include <cstdlib>
#include "float4.hpp"
#include "constants.hpp"

namespace mathsimd {

	struct float4x4 {
	private:
	    union M4x4 {
	        float f[4][4];
	        __m128 cols[4]{__m128{0,0,0,0},__m128{0,0,0,0},__m128{0,0,0,0},__m128{0,0,0,0}};
            __m256 x2cols[2];
	        M4x4(__m256 const&a, __m256 const &b) : x2cols{a, b} {};
            M4x4(__m128 const&c0, __m128 const &c1, __m128 const &c2, __m128 const &c3) : cols{c0, c1, c2, c3} {};
            M4x4() = default;
            M4x4(M4x4 const& other) : x2cols{other.x2cols[0], other.x2cols[1]} {}
	    };
		M4x4 _val;
	public:
		float4x4() = default;
        float4x4(float4x4 const &other) : _val(other._val) {}
        float4x4(float4 const &c0, float4 const &c1, float4 const &c2, float4 const &c3) : _val(c0,c1,c2,c3) {}
        float4x4(__m128 const &c0, __m128 const &c1, __m128 const &c2, __m128 const &c3) : _val(c0,c1,c2,c3) {}
        float4x4(__m256 const &a, __m256 const &b) : _val(a,b) {}
        inline operator __m256 const*() const { return _val.x2cols; }
        inline float const* operator[](int i) const { return _val.f[i]; }
        inline operator float const*() const { return _val.f[0]; }

        __m128 &c0() { return _val.cols[0]; }
        __m128 &c1() { return _val.cols[1]; }
        __m128 &c2() { return _val.cols[2]; }
        __m128 &c3() { return _val.cols[3]; }
        float4 c0() const { return _val.cols[0]; }
        float4 c1() const { return _val.cols[1]; }
        float4 c2() const { return _val.cols[2]; }
        float4 c3() const { return _val.cols[3]; }

        #define ARITHMETIC(OP) \
        friend float4x4 operator OP (float4x4 const &a, float4x4 const &b); \
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type *> \
        friend float4x4 operator OP (T const &a, float4x4 const &b); \
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type *> \
        friend float4x4 operator OP (float4x4 const &a, T const &b);
        ARITHMETIC(+)
        ARITHMETIC(-)
        ARITHMETIC(*)
        #undef ARITHMETIC
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type *>
        friend float4x4 operator / (float4x4 const &a, T const &b);

        static float4x4 matmul(float4x4 const &a, float4x4 const &b) {
            __m128 const* l = a._val.cols;
            __m256 const* r = b._val.x2cols;
            __m256 out0;
            out0 = _mm256_mul_ps(_mm256_permute_ps(r[0], 0x00), _mm256_broadcast_ps(l));
            out0 = _mm256_add_ps(out0, _mm256_mul_ps(_mm256_permute_ps(r[0], 0x55), _mm256_broadcast_ps(l + 1)));
            out0 = _mm256_add_ps(out0, _mm256_mul_ps(_mm256_permute_ps(r[0], 0xaa), _mm256_broadcast_ps(l + 2)));
            out0 = _mm256_add_ps(out0, _mm256_mul_ps(_mm256_permute_ps(r[0], 0xff), _mm256_broadcast_ps(l + 3)));

            __m256 out1;
            out1 = _mm256_mul_ps(_mm256_permute_ps(r[1], 0x00), _mm256_broadcast_ps(l));
            out1 = _mm256_add_ps(out1, _mm256_mul_ps(_mm256_permute_ps(r[1], 0x55), _mm256_broadcast_ps(l + 1)));
            out1 = _mm256_add_ps(out1, _mm256_mul_ps(_mm256_permute_ps(r[1], 0xaa), _mm256_broadcast_ps(l + 2)));
            out1 = _mm256_add_ps(out1, _mm256_mul_ps(_mm256_permute_ps(r[1], 0xff), _mm256_broadcast_ps(l + 3)));

            return float4x4(out0, out1);
        }
        static inline float4x4 identity() { return {float4::right(),float4::up(),float4::forward(),float4::in()}; }

	};

	#define ARITHMETIC(OP) \
        inline float4x4 operator OP (float4x4 const &a, float4x4 const &b) { \
            return float4x4(a._val.x2cols[0] OP b._val.x2cols[0], a._val.x2cols[1] OP b._val.x2cols[1]); \
        } \
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr> \
        inline float4x4 operator OP (T const &a, float4x4 const &b) { \
            return float4x4(static_cast<float>(a) OP b._val.x2cols[0], static_cast<float>(a) OP b._val.x2cols[1]); \
        } \
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr> \
        inline float4x4 operator OP (float4x4 const &a, T const &b) { \
            return float4x4(a._val.x2cols[0] OP static_cast<float>(b), a._val.x2cols[1] OP static_cast<float>(b)); \
        }
    ARITHMETIC(+)
    ARITHMETIC(-)
    ARITHMETIC(*)
    #undef ARITHMETIC
    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type *>
    inline float4x4 operator / (float4x4 const &a, T const &b) {
        auto m = a._val.x2cols;
        return float4x4(m[0] / static_cast<float>(b), m[1] / static_cast<float>(b));
    }

}

#endif