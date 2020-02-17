#pragma clang diagnostic push
#pragma ide diagnostic ignored "hicpp-explicit-conversions"
#ifndef MATHEMATICS_SEQ_FLOAT3_HPP
#define MATHEMATICS_SEQ_FLOAT3_HPP

#include <cmath>
#include <numeric>
#include <algorithm>
#include <array>
#include <limits>

namespace mathseq {

    struct float3 {
    private:
        std::array<float, 3> _val{0, 0, 0};
    public:
        float3() = default;
        float3(float const &x, float const &y, float const &z) : _val{x, y, z} {}
        float3(std::array<float, 3> const &other) : _val(other) {}
        float3(std::array<float, 3> &&other) : _val(other) {}
        float3(float3 const &other) : _val(other._val) {}
        float3(float3 &&other) noexcept : _val(other._val) {}
        operator std::array<float, 3>() const { return _val; }
        inline float3 &operator=(std::array<float, 3> const &other) {
            _val = other;
            return *this;
        }
        inline float3 &operator=(std::array<float, 3> &&other) {
            _val = other;
            return *this;
        }
        inline float3 &operator=(float3 const &other) {
            _val = other._val;
            return *this;
        }
        inline float3 &operator=(float3 &&other) noexcept {
            _val = other._val;
            return *this;
        }
        float &x = _val[0];
        float &y = _val[1];
        float &z = _val[2];

        static inline float dot(float3 const &a, float3 const &b) {
            return std::inner_product(a._val.begin(), a._val.end(), b._val.begin(), 0.f);
        }
        static inline float3 cross(float3 const &a, float3 const &b) {
            return float3{
                    a.y * b.z - b.y * a.z,
                    a.z * b.x - a.x * b.z,
                    a.x * b.y - a.y * b.x};
        }

        #define ARITHMETIC(OP) \
        friend float3 operator OP (float3 const & a, float3 const & b);

        ARITHMETIC(+)
        ARITHMETIC(-)
        ARITHMETIC(*)

        #undef ARITHMETIC
        friend bool operator==(float3 const & a, float3 const & b);
        friend bool operator!=(float3 const & a, float3 const & b);
        friend float3 operator * (float const & a, float3 const & b);
        friend float3 operator * (float3 const & a, float const & b);
        friend float3 operator / (float3 const & a, float const & b);

    };

    #define ARITHMETIC(OP) \
    inline float3 operator OP (float3 const & a, float3 const & b) { \
        constexpr auto operation = [](float l, float r) { return l OP r; }; \
        std::array<float,3> output{0,0,0}; \
        std::transform(a._val.begin(), a._val.end(), b._val.begin(), output.begin(), operation); \
        return output; \
    }
        ARITHMETIC(+)
        ARITHMETIC(-)
        ARITHMETIC(*)
    #undef ARITHMETIC

    inline float3 operator * (float const & a, float3 const & b) {
        return float3{a*b.x, a*b.y, a*b.z};
    }

    inline float3 operator * (float3 const & a, float const & b) {
        return float3{b * a.x, b * a.y, b * a.z};
    }

    inline float3 operator / (float3 const & a, float const & b) {
        return float3{b / a.x, b / a.y, b / a.z};
    }

    inline bool operator==(mathseq::float3 const & a, mathseq::float3 const & b) {
        auto output = true;

        for (auto i = 0; i < 3; ++i) {
            output &= a._val.at(i) == b._val.at(i);
        }
        return output;
    }

    inline bool operator!=(float3 const & a, float3 const & b) {
        return !(a == b);
    }
}
#endif //MATHEMATICS_SEQ_FLOAT3_HPP

#pragma clang diagnostic pop