#ifndef MATHEMATICS_OPERATIONS_HPP
#define MATHEMATICS_OPERATIONS_HPP
#include "float2.hpp"
#include "float3.hpp"
#include "float4.hpp"
#include "float4x4.hpp"
namespace mathsimd {
    inline __m128 _mm_abs_ps(__m128 fp_val) {
        static const __m128i NEG{0x7fffffff7fffffff,0x7fffffff7fffffff};
        auto tmp = _mm_and_si128(_mm_castps_si128(fp_val), NEG);
        return _mm_castsi128_ps(tmp);
    }
#define ARITHMETIC(TYPE, OP) \
    inline TYPE operator OP (TYPE const &a, TYPE const &b) { return static_cast<__m128>(a) OP static_cast<__m128>(b); } \
    inline TYPE operator OP (float const &a, TYPE const &b) { return a OP static_cast<__m128>(b); } \
    inline TYPE operator OP (TYPE const &a, float const &b) { return static_cast<__m128>(a) OP b; }

#define DIVISION(TYPE) \
    inline TYPE operator / (TYPE const &a, float const &b) { return static_cast<__m128>(a) * _mm_rcp_ps(_mm_load_ps1(&b)); } \
    inline TYPE operator / (float const &a, TYPE const &b) { return _mm_load_ps1(&a) * _mm_rcp_ps(static_cast<__m128>(b)); } \
    inline TYPE operator / (TYPE const &a, TYPE const &b) { return static_cast<__m128>(a) * _mm_rcp_ps(static_cast<__m128>(b)); }

#define EQUALITY_CHECK(TYPE) \
    inline bool operator==(TYPE const &a, TYPE const &b) { \
        auto tmp = _mm_abs_ps(static_cast<__m128>(a) - static_cast<__m128>(b)); \
        return _mm_movemask_epi8(_mm_castps_si128(tmp < EPSILON_F)) == 0xffff; \
    } \
    inline bool operator!=(TYPE const &a, TYPE const &b) { return !(a == b); }

#define SIMD_OPS(TYPE) \
ARITHMETIC(TYPE, +) \
ARITHMETIC(TYPE, -) \
ARITHMETIC(TYPE, *) \
DIVISION(TYPE) \
EQUALITY_CHECK(TYPE)

    SIMD_OPS(float2)
    SIMD_OPS(float3)
    SIMD_OPS(float4)

#undef SIMD_OPS
#undef EQUALITY_CHECK
#undef DIVISION
#undef ARITHMETIC

    inline std::ostream &operator<<(std::ostream &stream, mathsimd::float2 const &input) {
        stream << '(' << input.x() << ", " << input.y() << ')';
        return stream;
    }

    inline std::ostream &operator<<(std::ostream &stream, mathsimd::float3 const &input) {
        stream << '(' << input.x() << ", " << input.y() << ", " << input.z() << ')';
        return stream;
    }

    inline std::ostream &operator<<(std::ostream &stream, mathsimd::float4 const &input) {
        stream << '(' << input.x() << ", " << input.y() << ", " << input.z() << ", " << input.w() << ')';
        return stream;
    }


    inline float dot(float2 const &a, float2 const &b) {
        auto c = _mm_mul_ps(static_cast<__m128>(a), static_cast<__m128>(b));
        return c[0] + c[1];
    }

    inline float dot(float3 const &a, float3 const &b) {
        float f;
        auto c = _mm_mul_ps(static_cast<__m128>(a), static_cast<__m128>(b));
        _mm_store_ss(&f, _mm_add_ss(_mm_add_ss(c, _mm_shuffle_ps(c,c,85)), _mm_unpackhi_ps(c,c)));
        return f;
    }

    inline float dot(float4 const &a, float4 const &b) {
        float f;
        auto c = _mm_mul_ps(static_cast<__m128>(a), static_cast<__m128>(b));
        auto tmp = _mm_add_ps(c, _mm_permute_ps(c, 78));
        _mm_store_ss(&f, _mm_add_ss(tmp, _mm_permute_ps(tmp, 85)));
        return f;
    }

    inline float cross(float2 const &a, float2 const &b) {
        float f;
        constexpr int mask = _MM_SHUFFLE(3,2,0,1);
        auto ma = static_cast<__m128>(a);
        auto mb = static_cast<__m128>(b);
        auto tmp0 = _mm_permute_ps(mb,mask);
        tmp0 = _mm_mul_ps(ma,tmp0);
        _mm_store_ss(&f, _mm_sub_ss(tmp0, _mm_permute_ps(tmp0, mask)));
        return f;
    }

    inline float4 cross(float4 const &a, float4 const &b) {
        constexpr int mask0 = _MM_SHUFFLE(3,0,2,1);
        constexpr int mask1 = _MM_SHUFFLE(3,1,0,2);
        auto tmp0 = _mm_permute_ps(a,mask0);
        auto tmp1 = _mm_permute_ps(b,mask1);
        auto tmp2 = _mm_permute_ps(a,mask1);
        auto tmp3 = _mm_permute_ps(b,mask0);
        return _mm_sub_ps(_mm_mul_ps(tmp0,tmp1),_mm_mul_ps(tmp2,tmp3));
    }

    inline float3 cross(float3 const &a, float3 const &b) {
        constexpr int mask0 = _MM_SHUFFLE(3,0,2,1);
        constexpr int mask1 = _MM_SHUFFLE(3,1,0,2);
        auto tmp0 = _mm_permute_ps(a,mask0);
        auto tmp1 = _mm_permute_ps(b,mask1);
        auto tmp2 = _mm_permute_ps(a,mask1);
        auto tmp3 = _mm_permute_ps(b,mask0);
        return _mm_sub_ps(_mm_mul_ps(tmp0,tmp1),_mm_mul_ps(tmp2,tmp3));
    }

    /* float4x4 operations */
    #define ARITHMETIC(OP) \
        inline float4x4 operator OP (float4x4 const &a, float4x4 const &b) { \
            return float4x4(a._val.x2cols[0] OP b._val.x2cols[0], a._val.x2cols[1] OP b._val.x2cols[1]); \
        } \
        inline float4x4 operator OP (float const &a, float4x4 const &b) { \
            return float4x4(static_cast<float>(a) OP b._val.x2cols[0], static_cast<float>(a) OP b._val.x2cols[1]); \
        } \
        inline float4x4 operator OP (float4x4 const &a, float const &b) { \
            return float4x4(a._val.x2cols[0] OP static_cast<float>(b), a._val.x2cols[1] OP static_cast<float>(b)); \
        }
    ARITHMETIC(+)
    ARITHMETIC(-)
    ARITHMETIC(*)
    #undef ARITHMETIC
    inline float4x4 operator / (float4x4 const &a, float const &b) {
        auto const &m = a._val.x2cols;
        auto mb = _mm256_load_ps(&b);
        return float4x4(_mm256_mul_ps(m[0], _mm256_rcp_ps(mb)), _mm256_mul_ps(m[1], _mm256_rcp_ps(mb)));
    }

    inline float4x4 matmul(float4x4 const &a, float4x4 const &b) {
        auto const& l = a._val.cols;
        auto const& r = b._val.x2cols;
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

    inline float4 matmul(float4x4 const &a, float4 const &b) {
            __m128 tmp = static_cast<__m128>(b);
        __m256 b0 = _mm256_broadcast_ps(&tmp);
        __m256 b1 = _mm256_permutevar_ps(b0, __m256i{-6148914691236517206,-6148914691236517206,-1,-1});
        b0 = _mm256_permutevar_ps(b0, __m256i{0,0,0x5555555555555555,0x5555555555555555});
        b0 = _mm256_mul_ps(a._val.x2cols[0], b0);
        b1 = _mm256_mul_ps(a._val.x2cols[1], b1);
        b0 = _mm256_add_ps(b0,b1);
        b1 = _mm256_permute2f128_ps(b0,b0,0x41);
        b0 = _mm256_add_ps(b0,b1);
        return _mm256_castps256_ps128(b0);
    }

}
#endif //MATHEMATICS_OPERATIONS_HPP
