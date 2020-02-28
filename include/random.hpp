#ifndef MATHEMATICS_RANDOM_HPP
#define MATHEMATICS_RANDOM_HPP
#include <algorithm>
#include <random>
namespace mathsimd {
    struct Random {
    private:
        std::default_random_engine _engine;
    public:
        explicit Random(int seed) : _engine(seed) {}
        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
        inline T rnd() {
            static std::uniform_real_distribution<T> distribution(static_cast<T>(0), static_cast<T>(1));
            return distribution(_engine);
        }

        template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
        inline T rnd(T lower, T upper) {
            auto w = std::max(upper,lower) - std::min(upper,lower);
            return w * rnd<T>() + lower;
        }

        template<typename T>
        T rnd_vec(bool zeroX = false, bool zeroY = false, bool zeroZ = false);

        template<>
        inline float3 rnd_vec<float3>(bool zeroX, bool zeroY, bool zeroZ) {
            return {rnd<float>() * !zeroX, rnd<float>() * !zeroY,rnd<float>() * !zeroZ};
        }

        template<>
        inline float2 rnd_vec<float2>(bool zeroX, bool zeroY, bool zeroZ) {
            return {rnd<float>() * !zeroX, rnd<float>() * !zeroY};
        }
    };
}

#endif //MATHEMATICS_RANDOM_HPP
