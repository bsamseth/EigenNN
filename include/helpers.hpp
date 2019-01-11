#pragma once

#include <tuple>
#include <utility>



// For-each loop for tuples
// Taken from https://stackoverflow.com/a/6894436/3377926

template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
    for_each(std::tuple<Tp...> &, FuncT)
{ }

template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type
    for_each(std::tuple<Tp...>& t, FuncT f)
{
    f(std::get<I>(t));
    for_each<I + 1, FuncT, Tp...>(t, f);
}


// Reverse iteration
template<typename... Tp, int I = sizeof...(Tp) - 1, typename FuncT>
inline typename std::enable_if<I < 0, void>::type
    for_each_reverse(std::tuple<Tp...> &, FuncT)
{ }

template<typename... Tp, int I = sizeof...(Tp) - 1, typename FuncT>
inline typename std::enable_if<I >= 0, void>::type
    for_each_reverse(std::tuple<Tp...>& t, FuncT f)
{
    f(std::get<I>(t));
    for_each<I - 1, FuncT, Tp...>(t, f);
}
