#ifndef MATHEMATICS_SIMD_FLOAT2_HPP
#define MATHEMATICS_SIMD_FLOAT2_HPP

#include <immintrin.h>
#include <array>
#include <cmath>
#include <iostream>

namespace mathsimd {


    struct float2 {
    private:
        typedef float vec2 __attribute__((__vector_size__(2 * sizeof(float))));
        union F2 {
            float f[2];
            vec2 vec{0,0};
            F2(vec2 const& other) : vec{other} {}
            F2(float const &x, float const &y) : vec{x, y} {}
            F2() = default;
            F2(F2 const &other) :  vec(other.vec) {}
        };
        F2 _val{0.f, 0.f};
    public:
        float2() = default;
        float2(float const &x, float const &y) : _val{x, y} {}
        inline operator vec2() const { return _val.vec; }
        float2(float2 const &other) : _val(other._val) {}
        float2(vec2 const &other) : _val(other) {}
        inline float2 &operator=(float2 const &other) = default;
        inline float2 &operator=(vec2 const &other) { _val = other; return *this; }
        float &x() { return *_val.f; }
        float &y() { return *(_val.f + 1); }
        [[nodiscard]] float x() const { return _val.vec[0]; }
        [[nodiscard]] float y() const { return _val.vec[1]; }

        #define ARITHMETIC(OP) \
        friend float2 operator OP (float2 const &a, float2 const &b); \
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type *> \
        friend float2 operator OP (T const &a, float2 const &b); \
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type *> \
        friend float2 operator OP (float2 const &a, T const &b);
        ARITHMETIC(+)
        ARITHMETIC(-)
        ARITHMETIC(*)
        #undef ARITHMETIC
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type *>
        friend float2 operator / (float2 const &a, T const &b);

        static float dot(float2 const &a, float2 const &b) {
            auto c = a * b;
            return c._val.vec[0] + c._val.vec[1];
        }

        [[nodiscard]] inline float sqrMagnitude() const { return dot(*this, *this); }
        [[nodiscard]] inline float magnitude() const { return std::sqrt(sqrMagnitude()); }
        [[nodiscard]] inline float2 normalized() const { return _val.vec / magnitude(); }

        #define FUNC(NAME,X,Y) \
        static inline float2 NAME () { return {X,Y}; }
        FUNC(up, 0,1)
        FUNC(down, 0,-1)
        FUNC(right, 1,0)
        FUNC(left, -1,0)
        FUNC(one, 1,1)
        FUNC(zero, 0,0)
        #undef FUNC

        friend bool operator==(float2 const &a, float2 const &b);
    };

#define ARITHMETIC(OP) \
    inline float2 operator OP (float2 const &a, float2 const &b) { return a._val.vec OP b._val.vec; } \
    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr> \
    inline float2 operator OP (T const &a, float2 const &b) { return static_cast<float>(a) OP b._val.vec; } \
    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr> \
    inline float2 operator OP (float2 const &a, T const &b) { return a._val.vec OP static_cast<float>(b); }
    ARITHMETIC(+)
    ARITHMETIC(-)
    ARITHMETIC(*)
#undef ARITHMETIC


    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    inline float2 operator / (float2 const &a, T const &b) { return a._val.vec / static_cast<float>(b); }

    inline bool operator==(float2 const &a, float2 const &b) {
        auto tmp = (__m128{a._val.vec[0], a._val.vec[1]} - __m128{b._val.vec[0], b._val.vec[1]}) < EPSILON_F;
        return _mm_movemask_epi8(tmp) == 0xffff;
    }

    inline bool operator!=(float2 const &a, float2 const &b) { return !(a == b); }

    inline std::ostream& operator << (std::ostream& stream, float2 const &input) {
        stream << '(' << input.x() << ", " << input.y() <<')';
        return stream;
    }
}
#endif