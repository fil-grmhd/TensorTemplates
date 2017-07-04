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

#ifndef TENSORS_VECTORTYPES_HH
#define TENSORS_VECTORTYPES_HH

namespace tensors {

// Vectorized typedefs, just for conveniece
// Tensor of rank index types, e.g. lower_t, upper_t, ...
template <typename T, size_t ndim_, typename frame_t_, typename... ranks>
using tensor_vt = tensor_t<Vc::Vector<T>, ndim_, frame_t_, ranks...>;

template <typename T, size_t ndim_, typename frame_t_, size_t i0, size_t i1, typename... ranks>
using sym2_tensor_vt = sym2_tensor_t<Vc::Vector<T>, ndim_, frame_t_, i0, i1, ranks...>;

template <typename T, size_t ndim_, typename frame_t_>
using vector_vt = tensor_vt<T, ndim_, frame_t_, upper_t>;

template <typename T>
using vector3_vt = tensor_vt<T, 3, eulerian_t, upper_t>;

template <typename T>
using covector3_vt = tensor_vt<T, 3, eulerian_t, lower_t>;

template <typename T, typename... ranks>
using tensor3_vt = tensor_vt<T, 3, eulerian_t, ranks...>;

template <typename T>
using vector4_vt = tensor_vt<T, 4, eulerian_t, upper_t>;

template <typename T>
using covector4_vt = tensor_vt<T, 4, eulerian_t, lower_t>;

template <typename T, typename... ranks>
using tensor4_vt = tensor_vt<T, 4, eulerian_t, ranks...>;

template <typename T>
using cm_vector3_vt = tensor_vt<T, 3, comoving_t, upper_t>;

template <typename T>
using cm_covector3_vt = tensor_vt<T, 3, comoving_t, lower_t>;

template <typename T, typename... ranks>
using cm_tensor3_vt = tensor_vt<T, 3, comoving_t, ranks...>;

template <typename T>
using cm_vector4_vt = tensor_vt<T, 4, comoving_t, upper_t>;

template <typename T>
using cm_covector4_vt = tensor_vt<T, 4, comoving_t, lower_t>;

template <typename T, typename... ranks>
using cm_tensor4_vt = tensor_vt<T, 4, comoving_t, ranks...>;

template <typename T, size_t i0 = 0, size_t i1 = 1, typename... ranks>
using sym_tensor3_vt = sym2_tensor_vt<T, 3, eulerian_t, i0, i1, ranks...>;
template <typename T, size_t i0 = 0, size_t i1 = 1, typename... ranks>
using sym_tensor4_vt = sym2_tensor_vt<T, 4, eulerian_t, i0, i1, ranks...>;


// metric types
template<typename T, size_t ndim>
using metric_tensor_vt = sym2_tensor_vt<T, ndim, any_frame_t, 0, 1, lower_t, lower_t>;
template<typename T, size_t ndim>
using invmetric_tensor_vt = sym2_tensor_vt<T, ndim, any_frame_t, 0, 1, upper_t, upper_t>;

} // namespace tensors

#endif
