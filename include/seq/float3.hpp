#pragma clang diagnostic push
#pragma ide diagnostic ignored "hicpp-explicit-conversions"
#ifndef MATHEMATICS_SEQ_FLOAT3_HPP
#define MATHEMATICS_SEQ_FLOAT3_HPP

#include <cmath>
#include <numeric>
#include <algorithm>
#include <array>

namespace mathseq {

    struct float3 {
        constexpr static float EPSILON = 1e-6f;
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

        static float dot(float3 const &a, float3 const &b);
        static float3 cross(float3 const &a, float3 const &b);

        #define ASSIGN(OP) \
        friend float3 operator OP (float3 const & a, float3 const & b);

        ASSIGN(+)
        ASSIGN(-)
        ASSIGN(*)

        #undef ASSIGN
        friend bool operator==(float3 const & a, float3 const & b);
        friend bool operator!=(float3 const & a, float3 const & b);

    };

#define ASSIGN(OP) \
inline float3 operator OP (float3 const & a, float3 const & b) { \
    constexpr auto operation = [](float l, float r) { return l OP r; }; \
    std::array<float,3> output; \
    std::transform(a._val.begin(), a._val.end(), b._val.begin(), output.begin(), operation); \
    return output; \
}

    ASSIGN(+)
    ASSIGN(-)
    ASSIGN(*)
#undef ASSIGN

inline bool operator==(float3 const & a, float3 const & b) {
    constexpr auto operation = [](float l, float r) { return std::fabs(l - r) < float3::EPSILON * (l+r); };
    constexpr auto true_check = [](bool l, bool r) { return l && r; };
    std::array<bool,3> output{false, false, false};
    std::transform(a._val.begin(), a._val.end(), b._val.begin(), output.begin(), operation);
    return std::accumulate(output.begin(), output.end(), true, true_check);
}
inline bool operator!=(float3 const & a, float3 const & b) {
    return !(a == b);
}
}
#endif //MATHEMATICS_SEQ_FLOAT3_HPP

#pragma clang diagnostic pop