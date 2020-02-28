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
        friend float4x4 operator OP (float const &a, float4x4 const &b); \
        friend float4x4 operator OP (float4x4 const &a, float const &b);
        ARITHMETIC(+)
        ARITHMETIC(-)
        ARITHMETIC(*)
        #undef ARITHMETIC
        friend float4x4 operator / (float4x4 const &a, float const &b);

        friend float4x4 matmul(float4x4 const &a, float4x4 const &b);
        friend float4 matmul(float4x4 const &a, float4 const &b);

        static inline float4x4 identity() { return {float4::right(),float4::up(),float4::forward(),float4::in()}; }

	};

}

#endif