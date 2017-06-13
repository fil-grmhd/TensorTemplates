//  TensorTemplates: C++ tensor class templates
//  Copyright (C) 2016, Ludwig Jens Papenfort
//                      <papenfort@th.physik.uni-frankfurt.de>
//  Copyright (C) 2017, Elias Roland Most
//                      <emost@th.physik.uni-frankfurt.de>
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

#ifndef TENSORS_TENSOR_HH
#define TENSORS_TENSOR_HH

#include <array>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <utility>

#include "tensor.hh"
#include "tensor_defs.hh"
#include "tensor_expressions.hh"
#include "utilities.hh"

namespace tensors {

// Only 4dim metric
template <typename T, size_t ndim = 4> class metric_t {
public:
  using metric_tensor_t =
      general_tensor_t<T, any_frame_t, 2, std::tuple<lower_t, lower_t>, ndim>;
  using invmetric_tensor_t =
      general_tensor_t<T, any_frame_t, 2, std::tuple<upper_t, upper_t>, ndim>;
  metric_tensor_t metric;
  invmetric_tensor_t invmetric;

  // Constructors...
  

  //! Raise index
  template <size_t i = 0, typename E>
  decltype(auto) const inline raise_index(E const &v) {
    return metric_contraction_t<i>(invmetric, v);
  };

  //! Lower index
  template <size_t i = 0, typename E>
  decltype(auto) const inline lower_index(E const &v) {
    return metric_contraction_t<i>(metric, v);
  };

  // FIXME maybe we should change the name here, since we also have a
  // metric_contraction_t
  //      for raising and lowering now!
  template <size_t i1, size_t i2, typename ind1, typename ind2, typename E1,
            typename E2>
  class metric_contraction {};

  template <size_t i1, size_t i2, typename E1, typename E2>
  class metric_contraction<lower_t, lower_t> {

    inline decltype(auto) const contract(metric_t const &m, E1 const &u,
                                         E2 const &v) {
      return ::contract<i1, i2>(u, m.raise_index<i2>(v));
    };
  };

  template <size_t i1, size_t i2, typename E1, typename E2>
  class metric_contraction<upper_t, lower_t> {

    inline decltype(auto) const contract(metric_t const &m, E1 const &u,
                                         E2 const &v) {
      return ::contract<i1, i2>(u, v);
    };
  };

  template <size_t i1, size_t i2, typename E1, typename E2>
  class metric_contraction<lower_t, upper_t> {

    inline decltype(auto) const contract(metric_t const &m, E1 const &u,
                                         E2 const &v) {
      return ::contract<i1, i2>(u, v);
    };
  };

  template <size_t i1, size_t i2, typename E1, typename E2>
  class metric_contraction<upper_t, upper_t> {

    inline decltype(auto) const contract(metric_t const &m, E1 const &u,
                                         E2 const &v) {
      return ::contract<i1, i2>(u, m.lower_index<i2>(v));
    };
  };

  template <size_t i1, size_t i2, typename E1, typename E2>
  inline decltype(auto) const contract(E1 const &u, E2 const &v) {
    return metric_contraction<
        i1, i2,
        typename std::tuple_element<i1, typename E1::property_t::index_t>::type,
        typename std::tuple_element<
            i1, typename E2::property_t::index_t>::type>::contract(*this, u, v);
  };
};

} // namespace tensors

#endif
