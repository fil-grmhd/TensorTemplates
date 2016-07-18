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

#ifndef TENSORS_CONTRACTION_HH
#define TENSORS_CONTRACTION_HH

#include <type_traits>
#include <boost/type_traits.hpp>

#include "tensor_types.hh"

namespace tensors {

// Some typedefs for better readability

// Contracted tensor type, tensor of rank n+m-2
template<typename tensor0_t, typename tensor1_t>
  using contracted_tensor_t = generic<
                                common_data_t<tensor0_t,tensor1_t>,
                                tensor0_t::ndim,
                                tensor0_t::rank+tensor1_t::rank-2
                              >;

// Contraction return type, contracted vectors give a scalar
template<typename tensor0_t, typename tensor1_t>
  using contraction_result_t = typename std::conditional<
                                          ((tensor0_t::rank == 1) &&
                                          (tensor1_t::rank == 1)),
                                          common_data_t<tensor0_t,tensor1_t>,
                                          contracted_tensor_t<tensor0_t,tensor1_t>
                                        >::type;

///////////////////////////////////////////////////////////////////////////////
// Tensor contractions
///////////////////////////////////////////////////////////////////////////////

// Contractor type, defining a general contraction over two indices
/*
 *
 *  This is needed to get a return type specialization, since
 *
 *  (tensor,tensor) -> tensor
 *  (vector,vector) -> scalar
 *
 *  and there is no function template partial specialization
 *  (an alternative would be to use std::enable_if).
 *
 *  The vector specialization can be found further down below.
 *
 *  \tparam c_index0 contracted index of first tensor
 *  \tparam c_index1 contracted index of second tensor
 *  \tparam data0_t redundant parameter, needed for vector specialization
 *  \tparam data1_t redundant parameter, needed for vector specialization
 *  \tparam ndim redundant parameter, needed for vector specialization
 *  \tparam tensor0_t tensor type
 *  \tparam tensor1_t tensor type
 */

template<size_t c_index0, size_t c_index1, typename data0_t, typename data1_t, size_t ndim, typename tensor0_t, typename tensor1_t>
struct contractor_t {

  static inline contracted_tensor_t<tensor0_t,tensor1_t>
  contract(tensor0_t const & tensor0, tensor1_t const & tensor1) {
    // Resulting tensor
    contracted_tensor_t<tensor0_t,tensor1_t> contracted;

    // Get multiindex objects, representing the natural indices
    auto tensor0_mi = tensor0.get_mi();
    auto tensor1_mi = tensor1.get_mi();

    // Loop over all indices of the resulting tensor
    for(auto c_mi = contracted.get_mi(); !c_mi.end(); ++c_mi) {
      // Distribute current index to tensor indices
      size_t offset_index = 0;
      tensor0_mi.distribute<c_index0>(c_mi,offset_index);
      tensor1_mi.distribute<c_index1>(c_mi,offset_index);

      // Compute contraction for fixed indices of resulting tensor
      // DO NOT REMOVE: the temporary seems to speed up the sum,
      //                probably due to less index transformation in total.
      common_data_t<tensor0_t,tensor1_t> sum = 0;
      for(size_t i = 0; i<tensor0_t::ndim; ++i) {
        // Fix contraction index
        tensor0_mi[c_index0] = i;
        tensor1_mi[c_index1] = i;

        // Add product of the components
        sum += tensor0(tensor0_mi)*tensor1(tensor1_mi);
      }
      contracted(c_mi) = sum;
    }
    return contracted;
  }
};

// Contractor type partial specialization for two vectors
/*
 *  \tparam c_index0 contracted index of first tensor
 *  \tparam c_index1 contracted index of second tensor
 *  \tparam data0_t redundant parameter, needed for vector specialization
 *  \tparam data1_t redundant parameter, needed for vector specialization
 *  \tparam ndim redundant parameter, needed for vector specialization
 */

template<size_t c_index0, size_t c_index1, typename data0_t, typename data1_t, size_t ndim>
struct contractor_t<
         c_index0,
         c_index1,
         data0_t,
         data1_t,
         ndim,
         generic<data0_t,ndim,1>,
         generic<data1_t,ndim,1>> {

  static inline common_data_t<generic<data0_t,ndim,1>,generic<data1_t,ndim,1>>
  contract(generic<data0_t,ndim,1> const & vec0, generic<data1_t,ndim,1> const & vec1) {

    // Resulting scalar
    common_data_t<
      generic<data0_t,ndim,1>,
      generic<data1_t,ndim,1>
    > contracted;

    contracted = 0;
    for(size_t i = 0; i<ndim; ++i) {
      contracted += vec0[i]*vec1[i];
    }
    return contracted;
  }
};

// Contracts two tensors over one index and returns resulting tensor
/*
 *
 *  Calls a (specialized) contractor type (s. above)
 *
 *  \tparam c_index0 contracted index of first tensor
 *  \tparam c_index1 contracted index of second tensor
 *  \tparam tensor0_t tensor type, automatically deduced from argument
 *  \tparam tensor1_t tensor type, automatically deduced from argument
 */
template<size_t c_index0, size_t c_index1, typename tensor0_t, typename tensor1_t>
inline contraction_result_t<tensor0_t,tensor1_t>
contract(tensor0_t const & tensor0, tensor1_t const & tensor1) {

  // Consistency checks
  static_assert(tensor0_t::ndim == tensor1_t::ndim,
                "utils::tensor::contract: "
                "Number of dimensions of contracted tensors do not match.");
  static_assert(tensor0_t::rank > c_index0,
            "utils::tensor::contract: "
            "Contracted index must be strictly smaller than rank of tensor.");
  static_assert(tensor1_t::rank > c_index1,
            "utils::tensor::contract: "
            "Contracted index must be strictly smaller than rank of tensor.");

  return contractor_t<
           c_index0,
           c_index1,
           typename tensor0_t::data_t,
           typename tensor1_t::data_t,
           tensor0_t::ndim,
           tensor0_t,
           tensor1_t
         >::contract(tensor0,tensor1);
}

} // namespace tensor

#endif
