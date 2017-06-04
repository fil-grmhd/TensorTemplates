/*
 * =====================================================================================
 *
 *       Filename:  tuple_test.cc
 *
 *    Description:  Test empty tuple
 *
 *        Version:  1.0
 *        Created:  04/06/2017 09:05:08
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Elias Roland Most (ERM), emost@itp.uni-frankfurt.de
 *   Organization:  Goethe University Frankfurt
 *
 * =====================================================================================
 */

#include<tuple>
#include<iostream>
#include<utility>
#include<type_traits>

template<size_t... I>
inline constexpr decltype(auto) tuple_factory(std::index_sequence<I...>){
  return std::make_tuple(I...);
}

template<size_t N, typename I = std::make_index_sequence<N>>
inline constexpr decltype(auto) tuple_factory_call(){
  return tuple_factory(I{});
}

int main(){
  auto tp = tuple_factory_call<0>();

//  std::cout << std::get<0>(tp) <<std::endl;
  std::cout << std::tuple_size<decltype(tp)>::value;

  return 0;
}


