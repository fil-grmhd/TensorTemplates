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

#ifndef TENSORS_TRACE_HH
#define TENSORS_TRACE_HH

#include <type_traits>
#include <boost/type_traits.hpp>

#include "tensor_types.hh"

namespace tensors {

//! Traced tensor type, tensor of rank n-2
template<typename tensor_t>
  using traced_tensor_t = generic<
                            typename tensor_t::data_t,
                            tensor_t::ndim,
                            tensor_t::rank-2
                          >;

//! Trace return type, traced rank 2 tensors give a scalar
template<typename tensor_t>
  using trace_result_t = typename std::conditional<
                                    (tensor_t::rank == 2),
                                    typename tensor_t::data_t,
                                    traced_tensor_t<tensor_t>
                                  >::type;


//! Tracer type, defining a trace over two indices for a general tensor
/*!
 *
 *  This is needed to get a return type specialization, since
 *
 *  tensor<rank > 2> -> tensor<rank-2>
 *  tensor<rank == 2> -> scalar
 *
 *  and there is no function template partial specialization
 *  (an alternative would be to use std::enable_if).
 *
 *  The rank == 2 specialization can be found further down below.
 *
 *  \tparam t_index0 first traced index
 *  \tparam t_index1 second traced index
 *  \tparam data_t redundant parameter, needed for rank == 2 specialization
 *  \tparam ndim redundant parameter, needed for rank == 2 specialization
 *  \tparam tensor_t tensor type
 */

template<size_t t_index0, size_t t_index1, typename data_t, size_t ndim, typename tensor_t>
struct tracer_t {
  static inline traced_tensor_t<tensor_t>
  trace(tensor_t const & tensor) {

    // Resulting tensor
    traced_tensor_t<tensor_t> traced;

    // Get multiindex object, representing the natural indices
    auto tensor_mi = tensor.get_mi();

    // Loop over all indices of traced tensor
    for(auto traced_mi = traced.get_mi(); !traced_mi.end(); ++traced_mi) {
      // Distribute traced index on input tensor
      tensor_mi.distribute<t_index0,t_index1>(traced_mi);

      // Compute trace for fixed indices of resulting tensor
      // DO NOT REMOVE: the temporary seems to speed up the sum,
      //                probably due to less index transformation in total.
      data_t sum = 0;
      for(size_t i = 0; i<ndim; ++i) {
        // Fix traced indices
        tensor_mi[t_index0] = i;
        tensor_mi[t_index1] = i;

        // Add component to trace
        sum += tensor(tensor_mi);
      }
      traced(traced_mi) = sum;
    }
    return traced;
  }
};

//! Tracer type partial specializations for rank == 2 tensors
/*!
 *  \tparam t_index0 first traced index
 *  \tparam t_index1 second traced index
 *  \tparam data_t redundant parameter, needed for rank == 2 specialization
 *  \tparam ndim redundant parameter, needed for rank == 2 specialization
 */

template<size_t t_index0, size_t t_index1, typename data_t, size_t ndim>
struct tracer_t<
         t_index0,
         t_index1,
         data_t,
         ndim,
         symmetric2<data_t,ndim,2>> {

  static inline data_t trace(symmetric2<data_t,ndim,2> const & sym_tensor) {
    data_t result = 0;
    // Sum diagonal terms
    for(size_t i = 0; i<ndim; ++i) {
      result += sym_tensor(i,i);
    }
    return result;
  }
};

template<size_t t_index0, size_t t_index1, typename data_t, size_t ndim>
struct tracer_t<
         t_index0,
         t_index1,
         data_t,
         ndim,
         generic<data_t,ndim,2>> {

  static inline data_t trace(generic<data_t,ndim,2> const & tensor) {
    data_t result = 0;
    // Sum diagonal terms
    for(size_t i = 0; i<ndim; ++i) {
      result += tensor(i,i);
    }
    return result;
  }
};


//! Traces a general tensor over two indices and returns resulting tensor
/*!
 *
 *  Calls a (specialized) tracer type (s. above)
 *
 *  \tparam t_index0 first traced index
 *  \tparam t_index1 second traced index
 *  \tparam tensor_t tensor type, automatically deduced from argument
 */

template<size_t t_index0, size_t t_index1, typename tensor_t>
inline trace_result_t<tensor_t>
trace(tensor_t const & tensor) {

  // Consistency checks
  static_assert(tensor_t::rank >= 2,
                "utils::tensor::trace: "
                "Traced tensor must be of rank >= 2.");
  static_assert(tensor_t::rank > t_index0,
                "utils::tensor::trace: "
                "Traced index must be strictly smaller than rank of tensor.");
  static_assert(tensor_t::rank > t_index1,
                "utils::tensor::trace: "
                "Traced index must be strictly smaller than rank of tensor.");
  static_assert(t_index0 != t_index1,
                "utils::tensor::trace: "
                "Traced indinces must differ.");

  return tracer_t<
           t_index0,
           t_index1,
           typename tensor_t::data_t,
           tensor_t::ndim,
           tensor_t
         >::trace(tensor);
}

} // namespace tensors

#endif
