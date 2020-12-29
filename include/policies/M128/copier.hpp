#ifndef MATHEMATICS_COPIER_HPP
#define MATHEMATICS_COPIER_HPP

#include "../M128.hpp"
template<typename T>
struct mathsimd::M128<T>::Copier
{
	template<size_t... Idx>
	void copy(T& dst, T const& src, std::index_sequence<Idx...>)
	{
		Register regs[]{load<Idx>(src)...};
		(store<Idx>(dst, regs[Idx]),...);
	}
};

template<typename T>
void mathsimd::M128<T>::copy(T& dst, T const& src)
{
	Copier::copy(dst, src, std::make_index_sequence<register_count>{});
}
#endif //MATHEMATICS_COPIER_HPP
