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

#ifndef TENSORS_TYPES_HH
#define TENSORS_TYPES_HH

#include <cctk.h>

#include <type_traits>
#include <boost/type_traits.hpp>

namespace tensors {
// Forward declarations of tensor types

// General tensor type, see tensor.hh
template<typename T, size_t ndim, size_t rank, typename symmetry_t>
class general_tensor_t;

// Symmetry types, see symmetry_type.hh
class generic_symmetry_t;
class symmetric2_symmetry_t;

// old types, deprecated
template<typename T, size_t ndim, size_t rank>
  using generic = general_tensor_t<T,ndim,rank,generic_symmetry_t>;
template<typename T, size_t ndim, size_t rank>
  using symmetric2 = general_tensor_t<T,ndim,rank,symmetric2_symmetry_t>;
template<size_t ndim>
class metric;
template<size_t ndim>
class inv_metric;

// Tensor specializations
template<typename T, size_t ndim>
  using vector_t = general_tensor_t<T,ndim,1,generic_symmetry_t>;

template<typename T, size_t ndim, size_t rank>
  using generic_tensor_t = general_tensor_t<T,ndim,rank,generic_symmetry_t>;
template<typename T, size_t ndim, size_t rank>
  using symmetric_tensor_t = general_tensor_t<T,ndim,rank,symmetric2_symmetry_t>;

template<size_t ndim>
  using metric_t = metric<ndim>;
template<size_t ndim>
  using inv_metric_t = inv_metric<ndim>;


// Tensor fields, see tensor_field.hh
template<typename tensor_t>
class tensor_field_t;

template<typename data_t>
class scalar_field_t;

template<size_t ndim>
  using metric_field_t = tensor_field_t<metric_t<ndim>>;

template<size_t ndim>
  using inv_metric_field_t = tensor_field_t<inv_metric_t<ndim>>;

template<typename T, size_t ndim, size_t rank>
  using generic_field_t = tensor_field_t<generic_tensor_t<T,ndim,rank>>;

template<typename T, size_t ndim, size_t rank>
  using symmetric_field_t = tensor_field_t<symmetric_tensor_t<T,ndim,rank>>;

template<typename T, size_t ndim>
  using vector_field_t = tensor_field_t<vector_t<T,ndim>>;


// Common data type, both implicitly convertable to (e.g. int,float -> float)
// Used in trace/contraction routines to get a well defined result
template<typename tensor0_t, typename tensor1_t>
  using common_data_t = typename boost::common_type<
                                   typename tensor0_t::data_t,
                                   typename tensor1_t::data_t
                                 >::type;

// Common tensor type, using above common data type definition
template<typename tensor0_t, typename tensor1_t>
  using common_tensor_t = generic_tensor_t<
                            common_data_t<
                              tensor0_t,
                              tensor1_t
                            >,
                            tensor0_t::ndim,
                            tensor0_t::rank
                          >;

} // namespace tensors


#endif
