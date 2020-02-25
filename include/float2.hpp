#ifndef FLOAT2_HPP
#define FLOAT2_HPP

#include <immintrin.h>
#include <array>
#include <cmath>
#include <iostream>

namespace mathsimd {
    typedef float f2 __attribute__((__vector_size__(2 * sizeof(float))));

    struct float2 {
    private:
        f2 _val{0.f, 0.f};
    public:
        float2() = default;
        float2(float const &x, float const &y) : _val{x, y} {}
        inline operator f2() const { return _val; }
        float2(float2 const &other) : _val(other) {}
        float2(f2 const &other) : _val(other) {}
        inline float2 &operator=(float2 const &other) = default;
        inline float2 &operator=(f2 const &other) { _val = other; return *this; }
        float &x() { return *reinterpret_cast<float*>(&_val); }
        float &y() { return *(reinterpret_cast<float*>(&_val) + 1); }
        [[nodiscard]] float x() const { return _val[0]; }
        [[nodiscard]] float y() const { return _val[1]; }

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
            __m128 tmp{c._val[0], c._val[1]};
            return _mm_add_ss(tmp, _mm_permute_ps(tmp,1))[0];
        }

        [[nodiscard]] inline float sqrMagnitude() const { return dot(*this, *this); }
        [[nodiscard]] inline float magnitude() const { return std::sqrt(sqrMagnitude()); }
        [[nodiscard]] inline float2 normalized() const { return this->_val / magnitude(); }

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
    inline float2 operator OP (float2 const &a, float2 const &b) { return a._val OP b._val; } \
    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr> \
    inline float2 operator OP (T const &a, float2 const &b) { return static_cast<float>(a) OP b._val; } \
    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr> \
    inline float2 operator OP (float2 const &a, T const &b) { return a._val OP static_cast<float>(b); }
    ARITHMETIC(+)
    ARITHMETIC(-)
    ARITHMETIC(*)
#undef ARITHMETIC


    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    inline float2 operator / (float2 const &a, T const &b) { return a._val / static_cast<float>(b); }

    inline bool operator==(float2 const &a, float2 const &b) {
        auto tmp = (__m128{a._val[0], a._val[1]} - __m128{b._val[0], b._val[1]}) < EPSILON_F;
        return _mm_movemask_epi8(tmp) == 0xffff;
    }

    inline bool operator!=(float2 const &a, float2 const &b) { return !(a == b); }

    inline std::ostream& operator << (std::ostream& stream, float2 const &input) {
        stream << '(' << input.x() << ", " << input.y() <<')';
        return stream;
    }
}
#endif