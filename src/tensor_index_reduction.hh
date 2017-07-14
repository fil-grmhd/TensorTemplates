//  TensorTemplates: C++ tensor class templates
//  Copyright (C) 2017, Ludwig Jens Papenfort
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

#ifndef TENSORS_INDEX_REDUCTION_HH
#define TENSORS_INDEX_REDUCTION_HH

namespace tensors {

//! Helper class to get around intel compiler "bug"
//  see https://software.intel.com/en-us/forums/intel-c-compiler/topic/710211
template <size_t i1, size_t i2, typename E1, typename E2>
class index_reduction_generator_t {
public:
  //! Get index type of a contraction expression
  // this is only here to deduce the index type, see below
  static inline __attribute__ ((always_inline)) constexpr decltype(auto) get_contraction_index_t() {
    using i1_t = typename E1::property_t::index_t;
    using i2_t = typename E2::property_t::index_t;

    constexpr size_t E1_size = E1::property_t::rank;
    constexpr size_t E2_size = E2::property_t::rank;

    // get subtuple types
    using E1_p1_t =
        decltype(get_subtuple<E1_size *(i1 < 1), (i1 - 1) * (i1 > 1)>(
            std::declval<i1_t>()));
    using E1_p2_t =
        decltype(get_subtuple<i1 + 1, E1_size - 1>(std::declval<i1_t>()));

    using E2_p1_t =
        decltype(get_subtuple<E2_size *(i2 < 1), (i2 - 1) * (i2 > 1)>(
            std::declval<i2_t>()));
    using E2_p2_t =
        decltype(get_subtuple<i2 + 1, E2_size - 1>(std::declval<i2_t>()));

    using index_t = decltype(
        std::tuple_cat(std::declval<E1_p1_t>(), std::declval<E1_p2_t>(),
                       std::declval<E2_p1_t>(), std::declval<E2_p2_t>()));

    return index_t{};
  }

  //! Get index type of a trace expression
  // this is only here to deduce the index type, see below
  static inline __attribute__ ((always_inline)) constexpr decltype(auto) get_trace_index_t() {
    using i1_t = typename E1::property_t::index_t;

    constexpr size_t E_size = E1::property_t::rank;

    // get subtuple types
    using E_p1_t = decltype(get_subtuple<E_size*(i1<1),(i1-1)*(i1>1)>(std::declval<i1_t>()));
    using E_p2_t = decltype(get_subtuple<i1+1,i2-1>(std::declval<i1_t>()));
    using E_p3_t = decltype(get_subtuple<i2+1,E_size-1>(std::declval<i1_t>()));

    using index_t = decltype(std::tuple_cat(std::declval<E_p1_t>(),
                                            std::declval<E_p2_t>(),
                                            std::declval<E_p3_t>()));

    return index_t{};
  }

  //! Get index type of a metric contraction expression
  // this is only here to deduce the index type, see below
  static inline __attribute__ ((always_inline)) constexpr decltype(auto) get_metric_contraction_index_t() {
    using i1_t = typename E1::property_t::index_t;
    using i2_t = typename E2::property_t::index_t;

    // create index tuples for both tensors
    constexpr i1_t E1_indices;
    constexpr i2_t E2_indices;

    constexpr size_t E2_size = E2::property_t::rank;

    // get subtuple types
    using E1_p1_t = std::tuple<typename std::tuple_element<0, i1_t>::type>;

    using E2_p1_t =
        decltype(get_subtuple<E2_size *(i2 < 1), (i2 - 1) * (i2 > 1)>(
            std::declval<i2_t>()));
    using E2_p2_t =
        decltype(get_subtuple<i2 + 1, E2_size - 1>(std::declval<i2_t>()));

    using index_t = decltype(std::tuple_cat(std::declval<E2_p1_t>(),
                                            std::declval<E1_p1_t>(),
                                            std::declval<E2_p2_t>()));

    return index_t{};
  }

  //! Get index type of a tensor product expression
  // this is only here to deduce the index type, see below
  static inline __attribute__ ((always_inline)) constexpr decltype(auto) get_concat_index_t() {
    using i1_t = typename E1::property_t::index_t;
    using i2_t = typename E2::property_t::index_t;

    using index_t = decltype(
        std::tuple_cat(std::declval<i1_t>(), std::declval<i2_t>()));

    return index_t{};
  };

};

} // namespace tensors

#endif
