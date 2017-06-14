//  TensorTemplates: C++ tensor class templates
//  Copyright (C) 2016, Ludwig Jens Papenfort
//                      <papenfort@th.physik.uni-frankfurt.de>
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef TENSORS_UTILITIES_HH
#define TENSORS_UTILITIES_HH

#include <tuple>

namespace tensors {
namespace utilities {

template <size_t a, size_t b> struct static_pow {
  static constexpr size_t value = a * static_pow<a, b - 1>::value;
};
template <size_t a> struct static_pow<a, 0> {
  static constexpr size_t value = 1;
};

// Useful helper for checking conditions on template parameters
// https://stackoverflow.com/questions/28253399/check-traits-for-all-variadic-template-arguments/28253503#28253503
template <bool...> struct bool_pack;
template <bool... bs>
using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;

// printing for tuples (for debugging purposes)
template <typename Tuple, int index, typename... Ts> struct print_tuple_impl {
  void operator()(Tuple &t) {
    std::cout << std::get<std::tuple_size<Tuple>::value - 1 - index>(t) << " ";
    print_tuple_impl<Tuple, index - 1, Ts...>{}(t);
  }
};

template <typename Tuple, typename... Ts>
struct print_tuple_impl<Tuple, 0, Ts...> {
  void operator()(Tuple &t) { std::cout << std::get<std::tuple_size<Tuple>::value - 1>(t); }
};

template <typename Tuple, typename... Ts> void print_tuple(Tuple &t) {
  const auto size = std::tuple_size<Tuple>::value;
  print_tuple_impl<Tuple, size - 1, Ts...>{}(t);
}

//! Cumulative type check of index types
template <typename I1, typename I2, size_t... I>
constexpr bool compare_index_impl(std::index_sequence<I...>) {
  using namespace utilities;
  return all_true<
      std::is_same<std::remove_cv_t<typename std::tuple_element<I, I1>::type>,
                   std::remove_cv_t<typename std::tuple_element<I, I2>::type>>::
          value...>::value;
};
template <typename I1, typename I2, size_t ndim,
          typename Indices = std::make_index_sequence<ndim>>
constexpr bool compare_index() {
  return compare_index_impl<I1, I2>(Indices{});
}
//! Check if index combination is reducible
template<size_t i1, size_t i2, typename E1, typename E2>
struct is_reducible {
  static constexpr bool value = std::is_same<
                                  typename std::conditional<
                                    std::is_same<
                                      typename std::tuple_element<i1,typename E1::property_t::index_t>::type,
                                      lower_t
                                    >::value,
                                    lower_t,
                                    upper_t
                                  >::type,
                                  typename std::conditional<
                                    std::is_same<
                                      typename std::tuple_element<i2,typename E2::property_t::index_t>::type,
                                      upper_t
                                    >::value,
                                    lower_t,
                                    upper_t
                                  >::type
                                >::value;
};

} // namespace utilities
} // namespace tensors

#endif
