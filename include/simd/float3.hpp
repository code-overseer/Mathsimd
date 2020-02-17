#pragma clang diagnostic push
#pragma ide diagnostic ignored "portability-simd-intrinsics"
#pragma ide diagnostic ignored "hicpp-explicit-conversions"
#ifndef MATHEMATICS_SIMD_FLOAT3_HPP
#define MATHEMATICS_SIMD_FLOAT3_HPP

#include <immintrin.h>

namespace mathsimd {

    struct float3 {
        constexpr static float EPSILON = 1e-6f;
    private:
        __m128 _val{0, 0, 0};
    public:
        float3() = default;
        float3(float const &x, float const &y, float const &z) : _val{x, y, z} {}
        float3(float3 const &other) : _val(other) {}
        float3(__m128 const &other) : _val(other) {}
        operator __m128() const { return _val; }
        inline float3 &operator=(float3 const &other) {
            _val = other._val;
            return *this;
        }
        inline float3 &operator=(__m128 const &other) {
            _val = other;
            return *this;
        }
        float &x = *(float *) &_val;
        float &y = *((float *) &_val + sizeof(float));
        float &z = *((float *) &_val + 2 * sizeof(float));

        static float dot(float3 const &a, float3 const &b);

        static float3 cross(float3 const &a, float3 const &b);
    };

    inline float3 operator+(float3 const &a, float3 const &b) { return _mm_add_ps(a, b); }
    inline float3 operator-(float3 const &a, float3 const &b) { return _mm_sub_ps(a, b); }
    inline float3 operator*(float3 const &a, float3 const &b) { return _mm_mul_ps(a, b); }

    inline bool operator==(float3 const &a, float3 const &b) {
        return _mm_movemask_epi8(_mm_cmpeq_ps(a,b)) == 0xffff;
    }
    inline bool operator!=(float3 const &a, float3 const &b) { return !(a == b); }

    inline float3 operator * (float const & a, float3 const & b) { return _mm_mul_ps(b, _mm_load1_ps(&a)); }

    inline float3 operator * (float3 const & b, float const & a) { return _mm_mul_ps(b, _mm_load1_ps(&a)); }

    inline float3 operator / (float3 const & b, float const & a) { return _mm_div_ps(b, _mm_load1_ps(&a)); }

    inline __m128 _mm_abs_ps(__m128 fp_val) {
        static const __m128i NEG{0x7fffffff7fffffff,0x7fffffff7fffffff};
        auto tmp = _mm_and_si128(*reinterpret_cast<__m128i*>(&fp_val), NEG);
        return *reinterpret_cast<__m128*>(&tmp);
    }

}
#endif //MATHEMATICS_SIMD_FLOAT3_HPP

#pragma clang diagnostic pop