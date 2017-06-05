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

template<typename Tuple, int index, typename... Ts>
 struct print_tuple {
     void operator() (Tuple& t) {
         std::cout << std::get<index>(t) << " ";
         print_tuple<Tuple,index - 1, Ts...>{}(t);
     }
 };

 template<typename Tuple, typename... Ts>
 struct print_tuple<Tuple, 0, Ts...> {
     void operator() (Tuple& t) {
         std::cout << std::get<0>(t) << std::endl;
     }
 };

 template<typename Tuple, typename... Ts>
 void print(Tuple& t) {
     const auto size = std::tuple_size<Tuple>::value;
     print_tuple<Tuple,size - 1, Ts...>{}(t);
 }

int main(){

  auto tp = tuple_factory_call<10>();

  print(tp);
  std::cout << std::tuple_size<decltype(tp)>::value << std::endl;

  return 0;
}


