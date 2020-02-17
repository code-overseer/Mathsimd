#include "../../include/simd/float3.hpp"

float mathsimd::float3::dot(mathsimd::float3 const &a, mathsimd::float3 const &b) {
    auto c = _mm_mul_ps(a._val, b._val);
    c = _mm_hadd_ps(c,c);
    c = _mm_hadd_ps(c,c);
    return _mm_cvtss_f32(c);
}

mathsimd::float3 mathsimd::float3::cross(mathsimd::float3 const &a, mathsimd::float3 const &b) {
    auto tmp0 = _mm_shuffle_ps(a._val,a._val,_MM_SHUFFLE(3,0,2,1));
    auto tmp1 = _mm_shuffle_ps(b._val,b._val,_MM_SHUFFLE(3,1,0,2));
    auto tmp2 = _mm_shuffle_ps(a._val,a._val,_MM_SHUFFLE(3,1,0,2));
    auto tmp3 = _mm_shuffle_ps(b._val,b._val,_MM_SHUFFLE(3,0,2,1));
    return _mm_sub_ps(_mm_mul_ps(tmp0,tmp1),_mm_mul_ps(tmp2,tmp3));
}