#ifndef MATHEMATICS_CONSTANTS_HPP
#define MATHEMATICS_CONSTANTS_HPP
#include <cfloat>
namespace mathsimd 
{
    constexpr float EPSILON_F = FLT_EPSILON;
    constexpr double EPSILON_D = DBL_EPSILON;
    constexpr float MIN_F = 0x00800000;
    constexpr float FAST_ERROR_F = 0x39800000;
}

#endif //MATHEMATICS_CONSTANTS_HPP
