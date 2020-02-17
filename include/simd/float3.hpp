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
        auto tmp = _mm_cmpeq_ps(a,b);
        return tmp[0] != 0.f && tmp[1] != 0.f && tmp[2] != 0.f;
    }
    inline bool operator!=(float3 const &a, float3 const &b) { return !(a == b); }

}
#endif //MATHEMATICS_SIMD_FLOAT3_HPP

#pragma clang diagnostic pop