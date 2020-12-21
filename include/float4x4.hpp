#ifndef MATHEMATICS_SIMD_FLOAT4X4_HPP
#define MATHEMATICS_SIMD_FLOAT4X4_HPP

#include <immintrin.h>
#include "float4.hpp"
#include "constants.hpp"

namespace mathsimd {

  struct float4x4 {
  private:
    alignas(32) float _val[16]{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,0.f, 0.f, 0.f, 0.f,0.f, 0.f, 0.f, 0.f};
  public:
    float4x4() = default;
    float4x4(float4x4 const &other) { 
      _mm256_store_ps(_val, _mm256_load_ps(other._val));
      _mm256_store_ps(_val + 8, _mm256_load_ps(other._val + 8));
    }
    float4x4(float4 const &c0, float4 const &c1, float4 const &c2, float4 const &c3) {
      _mm_store_ps(_val, _mm_load_ps(c0));
      _mm_store_ps(_val + 4, _mm_load_ps(c1));
      _mm_store_ps(_val + 8, _mm_load_ps(c2));
      _mm_store_ps(_val + 12, _mm_load_ps(c3));
    }
    float4x4(__m128 const &c0, __m128 const &c1, __m128 const &c2, __m128 const &c3) {
      _mm_store_ps(_val, c0);
      _mm_store_ps(_val + 4, c1);
      _mm_store_ps(_val + 8, c2);
      _mm_store_ps(_val + 12, c3);
    }
    float4x4(__m256 const &a, __m256 const &b) {
      _mm256_store_ps(_val, a);
      _mm256_store_ps(_val + 8, b);
    }
    inline float const* operator[](size_t i) const { return _val + 4 * i; }
    inline float* operator[](size_t i) { return _val + 4 * i; }
    inline operator float const*() const { return _val; }
    inline operator float*() { return _val; }

    float4 c0() const { return float4(_val); }
    float4 c1() const { return float4(_val + 4); }
    float4 c2() const { return float4(_val + 8); }
    float4 c3() const { return float4(_val + 12); }

#define ARITHMETIC(OP)							\
    friend float4x4 operator OP (float4x4 const &a, float4x4 const &b); \
    friend float4x4 operator OP (float const &a, float4x4 const &b);	\
    friend float4x4 operator OP (float4x4 const &a, float const &b);
    ARITHMETIC(+)
    ARITHMETIC(-)
    ARITHMETIC(*)
#undef ARITHMETIC
    friend float4x4 operator / (float4x4 const &a, float const &b);
    friend float4x4 fast_div(float4x4 const &a, float const &b);
    friend float4x4 reciprocal(float4x4 const &a);

    friend float4x4 matmul(float4x4 const &a, float4x4 const &b);
    friend float4 matmul(float4x4 const &a, float4 const &b);

    static inline float4x4 identity() { return {float4::right(),float4::up(),float4::forward(),float4::in()}; }

  };

}

#endif
