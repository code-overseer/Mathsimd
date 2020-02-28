#ifndef MATHEMATICS_CONSTANTS_HPP
#define MATHEMATICS_CONSTANTS_HPP

namespace mathsimd {
    constexpr float EPSILON_F = 1e-6f;
    constexpr double EPSILON_D = 1e-13f;

    inline __m128 _mm_abs_ps(__m128 fp_val) {
        static const __m128i NEG{0x7fffffff7fffffff,0x7fffffff7fffffff};
        auto tmp = _mm_and_si128(_mm_castps_si128(fp_val), NEG);
        return _mm_castsi128_ps(tmp);
    }
}

#endif //MATHEMATICS_CONSTANTS_HPP
