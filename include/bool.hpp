#ifndef MATHEMATICS_SIMD_BOOL_HPP
#define MATHEMATICS_SIMD_BOOL_HPP
#include <cstring>

namespace mathsimd {
    
    template<size_t N>
    struct Bool
    {
    private:
        int _value{0};
        static_assert(N && N <= (sizeof(decltype(_value)) << 3));
        static constexpr int _all = (1 << N) - 1;
    public:
        explicit Bool(char const* val)
        {
            memcpy(&_value, val, sizeof(_value));
        }

        Bool(int val) : _value(val)
        {
            _value &= _all; 
        }

        inline operator int() const { return _value; }
        
        Bool(Bool const& other) = default;
        Bool(Bool&& other) noexcept = default;
        bool all_true() const { return _value == _all; }
        bool none_true() const { return !_value; }
        bool any_true() const { return _value & _all; }
        Bool operator!() { return Bool(~_value); }
        bool operator[](size_t idx) { return (1 << idx) & _value; }
        char* data() { return reinterpret_cast<char*>(&_value); }
        char const* data() const { return reinterpret_cast<char const*>(&_value); }
    };
    
}

#endif //MATHEMATICS_SIMD_BOOL_HPP
