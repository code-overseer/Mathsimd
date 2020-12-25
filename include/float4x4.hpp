#ifndef MATHEMATICS_SIMD_FLOAT4X4_HPP
#define MATHEMATICS_SIMD_FLOAT4X4_HPP

#include <immintrin.h>
#include "vector_operations.hpp"
#include "float4.hpp"
#include "constants.hpp"

namespace mathsimd 
{
     struct float4x4 : TensorBase<float,4,4>
     {
     private:
         alignas(32) float _val[16]{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,0.f, 0.f, 0.f, 0.f,0.f, 0.f, 0.f, 0.f};
         using This = float4x4;
         float4x4(float (&other)[16]) { _mm_store_ps(_val, _mm_load_ps(other)); }
     public:
         float4x4() = default;
         float4x4(float4x4 const &other)
         {
             _mm256_store_ps(_val, _mm256_load_ps(other._val));
             _mm256_store_ps(_val + 8, _mm256_load_ps(other._val + 8));
         }
         float4x4(float4 const &c0, float4 const &c1, float4 const &c2, float4 const &c3)
         {
             _mm_store_ps(_val, _mm_load_ps(c0));
             _mm_store_ps(_val + 4, _mm_load_ps(c1));
             _mm_store_ps(_val + 8, _mm_load_ps(c2));
             _mm_store_ps(_val + 12, _mm_load_ps(c3));
         }
         float4x4(__m128 const &c0, __m128 const &c1, __m128 const &c2, __m128 const &c3)
         {
             _mm_store_ps(_val, c0);
             _mm_store_ps(_val + 4, c1);
             _mm_store_ps(_val + 8, c2);
             _mm_store_ps(_val + 12, c3);
         }
         float4x4(__m256 const &c01, __m256 const &c23)
         {
             _mm256_store_ps(_val, c01);
             _mm256_store_ps(_val + 8, c23);
         }
         inline float const* operator[](size_t const i) const { return _val + 4 * i; }
         inline float* operator[](size_t const i) { return _val + 4 * i; }
         inline operator float const*() const { return _val; }
         inline operator float*() { return _val; }
//     #define ARITHMETIC(OP, SYM) \
//         friend float4x4 operator SYM (float4x4 const &a, float4x4 const &b) \
//         { \
//             __m256 l[2]{_mm256_loadu_ps(a._val), _mm256_loadu_ps(a._val + 8)}; \
//             __m256 r[2]{_mm256_loadu_ps(b._val), _mm256_loadu_ps(b._val + 8)}; \
//             return float4x4(_mm256_ ## OP ## _ps(l[0],r[0]), _mm256_ ## OP ## _ps(l[1],r[1])); \
//         } \
//         friend float4x4 operator SYM (float const &a, float4x4 const &b) \
//         { \
//             __m256 r[2]{_mm256_loadu_ps(b._val), _mm256_loadu_ps(b._val + 8)}; \
//             __m256 l = _mm256_broadcast_ss(&a); \
//             return float4x4(_mm256_ ## OP ## _ps(l,r[0]), _mm256_ ## OP ## _ps(l,r[1])); \
//         } \
//         friend float4x4 operator SYM (float4x4 const &a, float const &b) \
//         { \
//             __m256 l[2]{_mm256_loadu_ps(a._val), _mm256_loadu_ps(a._val + 8)}; \
//             __m256 r = _mm256_broadcast_ss(&b); \
//             return float4x4(_mm256_ ## OP ## _ps(l[0],r), _mm256_ ## OP ## _ps(l[1],r)); \
//         }
//         ARITHMETIC(add,+)
//         ARITHMETIC(sub,-)
//         ARITHMETIC(mul,*)
//     #undef ARITHMETIC
//
//         friend float4x4 operator / (float4x4 const &a, float const &b)
//         {
//             __m256 m[2]{_mm256_loadu_ps(a._val), _mm256_loadu_ps(a._val + 8)};
//             __m256 mb = _mm256_broadcast_ss(&b);
//             return float4x4(_mm256_div_ps(m[0], mb), _mm256_div_ps(m[1], mb));
//         }
//
//         friend float4x4 fast_div(float4x4 const &a, float const &b)
//         {
//             __m256 m[2]{_mm256_loadu_ps(a._val), _mm256_loadu_ps(a._val + 8)};
//             __m256 mb = _mm256_rcp_ps(_mm256_broadcast_ss(&b));
//             return float4x4(_mm256_mul_ps(m[0], mb), _mm256_mul_ps(m[1], mb));
//         }
//
//         friend float4x4 reciprocal(float4x4 const &a)
//         {
//             __m256 m[2]{_mm256_loadu_ps(a._val), _mm256_loadu_ps(a._val + 8)};
//             return float4x4(_mm256_rcp_ps(m[0]), _mm256_rcp_ps(m[1]));
//         }
//
//         friend float4x4 matmul(float4x4 const &a, float4x4 const &b)
//         {
//             __m128 const l[4]{_mm_load_ps(a._val), _mm_load_ps(a._val + 4), _mm_load_ps(a._val + 8), _mm_load_ps(a._val + 12)};
//             __m256 tmp = _mm256_load_ps(b._val);
//             __m256 out0;
//             out0 = _mm256_mul_ps(_mm256_permute_ps(tmp, 0x00), _mm256_broadcast_ps(l));
//             out0 = _mm256_add_ps(out0, _mm256_mul_ps(_mm256_permute_ps(tmp, 0x55), _mm256_broadcast_ps(l + 1)));
//             out0 = _mm256_add_ps(out0, _mm256_mul_ps(_mm256_permute_ps(tmp, 0xaa), _mm256_broadcast_ps(l + 2)));
//             out0 = _mm256_add_ps(out0, _mm256_mul_ps(_mm256_permute_ps(tmp, 0xff), _mm256_broadcast_ps(l + 3)));
//
//             tmp = _mm256_load_ps(b._val + 8);
//             __m256 out1;
//             out1 = _mm256_mul_ps(_mm256_permute_ps(tmp, 0x00), _mm256_broadcast_ps(l));
//             out1 = _mm256_add_ps(out1, _mm256_mul_ps(_mm256_permute_ps(tmp, 0x55), _mm256_broadcast_ps(l + 1)));
//             out1 = _mm256_add_ps(out1, _mm256_mul_ps(_mm256_permute_ps(tmp, 0xaa), _mm256_broadcast_ps(l + 2)));
//             out1 = _mm256_add_ps(out1, _mm256_mul_ps(_mm256_permute_ps(tmp, 0xff), _mm256_broadcast_ps(l + 3)));
//
//             return float4x4(out0, out1);
//         }
//
//         friend float4 matmul(float4x4 const &a, float4 const &b)
//         {
//             __m256 l[2]{_mm256_loadu_ps(a._val), _mm256_loadu_ps(a._val + 8)};
//             __m128 tmp = static_cast<__m128>(b);
//             __m256 b0 = _mm256_broadcast_ps(&tmp);
//             __m256 b1 = _mm256_permutevar_ps(b0, __m256i{-6148914691236517206,-6148914691236517206,-1,-1});
//             b0 = _mm256_permutevar_ps(b0, __m256i{0,0,0x5555555555555555,0x5555555555555555});
//             b0 = _mm256_mul_ps(l[0], b0);
//             b1 = _mm256_mul_ps(l[1], b1);
//             b0 = _mm256_add_ps(b0,b1);
//             b1 = _mm256_permute2f128_ps(b0,b0,0x41);
//             b0 = _mm256_add_ps(b0,b1);
//             return _mm256_castps256_ps128(b0);
//         }

         static inline float4x4 identity() { return {float4::right(),float4::up(),float4::forward(),float4::in()}; }
     };

}

#endif
