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

#ifndef TENSORS_VECTORTYPES_HH
#define TENSORS_VECTORTYPES_HH

namespace tensors {
namespace vector {
// Vectorized tensor typedefs
// Tensor of rank index types, e.g. lower_t, upper_t, ...
template <typename T, size_t ndim_, typename frame_t_, typename... ranks>
using tensor_vt = general::tensor_t<Vc::Vector<T>, ndim_, frame_t_, ranks...>;

template <typename T, size_t ndim_, typename frame_t_, size_t i0, size_t i1, typename... ranks>
using sym2_tensor_vt = general::sym2_tensor_t<Vc::Vector<T>, ndim_, frame_t_, i0, i1, ranks...>;

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

// component type
template<typename T>
using comp_vt = Vc::Vector<T>;

// loop vector increment
template<typename T>
size_t loop_vinc = Vc::Vector<T>::Size;

// metric type
template<typename T>
using metric_tensor3_vt = general::metric_tensor3_t<Vc::Vector<T>>;

#ifdef TENSORS_AUTOVEC
// typdef vectorized version to default interface for vectorization agnostic code
template <typename T, size_t ndim_, typename frame_t_, typename... ranks>
using tensor_t = general::tensor_t<Vc::Vector<T>, ndim_, frame_t_, ranks...>;

template <typename T, size_t ndim_, typename frame_t_, size_t i0, size_t i1, typename... ranks>
using sym2_tensor_t = general::sym2_tensor_t<Vc::Vector<T>, ndim_, frame_t_, i0, i1, ranks...>;

template <typename T, size_t ndim_, typename frame_t_>
using vector_t = tensor_vt<T, ndim_, frame_t_, upper_t>;

template <typename T>
using vector3_t = tensor_vt<T, 3, eulerian_t, upper_t>;

template <typename T>
using covector3_t = tensor_vt<T, 3, eulerian_t, lower_t>;

template <typename T, typename... ranks>
using tensor3_t = tensor_vt<T, 3, eulerian_t, ranks...>;

template <typename T>
using vector4_t = tensor_vt<T, 4, eulerian_t, upper_t>;

template <typename T>
using covector4_t = tensor_vt<T, 4, eulerian_t, lower_t>;

template <typename T, typename... ranks>
using tensor4_t = tensor_vt<T, 4, eulerian_t, ranks...>;

template <typename T>
using cm_vector3_t = tensor_vt<T, 3, comoving_t, upper_t>;

template <typename T>
using cm_covector3_t = tensor_vt<T, 3, comoving_t, lower_t>;

template <typename T, typename... ranks>
using cm_tensor3_t = tensor_vt<T, 3, comoving_t, ranks...>;

template <typename T>
using cm_vector4_t = tensor_vt<T, 4, comoving_t, upper_t>;

template <typename T>
using cm_covector4_t = tensor_vt<T, 4, comoving_t, lower_t>;

template <typename T, typename... ranks>
using cm_tensor4_t = tensor_vt<T, 4, comoving_t, ranks...>;

template <typename T, size_t i0 = 0, size_t i1 = 1, typename... ranks>
using sym_tensor3_t = sym2_tensor_vt<T, 3, eulerian_t, i0, i1, ranks...>;
template <typename T, size_t i0 = 0, size_t i1 = 1, typename... ranks>
using sym_tensor4_t = sym2_tensor_vt<T, 4, eulerian_t, i0, i1, ranks...>;

// component type
template<typename T>
using comp_t = comp_vt<T>;

// loop vector increment
template<typename T>
size_t loop_inc = loop_vinc<T>;

// metric type
template<typename T>
using metric_tensor3_t = general::metric_tensor3_t<Vc::Vector<T>>;
#endif

} // namespace vector
} // namespace tensors

#endif
