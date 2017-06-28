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

#include <type_traits>

namespace tensors {

// forward decleration of general tensor class
template <typename T, typename frame_t_, typename symmetry_t_, size_t rank_, typename index_t_,
          size_t ndim_>
class general_tensor_t;

// Tensor of rank index types, e.g. lower_t, upper_t, ...
template <typename T, size_t ndim_, typename frame_t_, typename... ranks>
using tensor_t = general_tensor_t<T, frame_t_, generic_symmetry_t<ndim_,sizeof...(ranks)>,
                                  sizeof...(ranks),
                                  std::tuple<ranks...>, ndim_>;

// Symmetric tensor in two indices of rank index types, e.g. lower_t, upper_t, ...
template <typename T, size_t ndim_, typename frame_t_, size_t i0, size_t i1, typename... ranks>
using sym2_tensor_t = general_tensor_t<T, frame_t_, sym2_symmetry_t<ndim_,sizeof...(ranks),i0,i1>,
                                  sizeof...(ranks),
                                  std::tuple<ranks...>, ndim_>;

template <typename T, size_t ndim_, typename frame_t_>
using vector_t = tensor_t<T, ndim_, frame_t_, upper_t>;

template <typename T> using vector3_t = tensor_t<T, 3, eulerian_t, upper_t>;

template <typename T> using covector3_t = tensor_t<T, 3, eulerian_t, lower_t>;

template <typename T, typename... ranks>
using tensor3_t = tensor_t<T, 3, eulerian_t, ranks...>;

template <typename T> using vector4_t = tensor_t<T, 4, eulerian_t, upper_t>;

template <typename T> using covector4_t = tensor_t<T, 4, eulerian_t, lower_t>;

template <typename T, typename... ranks>
using tensor4_t = tensor_t<T, 4, eulerian_t, ranks...>;

template <typename T> using cm_vector3_t = tensor_t<T, 3, comoving_t, upper_t>;

template <typename T>
using cm_covector3_t = tensor_t<T, 3, comoving_t, lower_t>;

template <typename T, typename... ranks>
using cm_tensor3_t = tensor_t<T, 3, comoving_t, ranks...>;

template <typename T> using cm_vector4_t = tensor_t<T, 4, comoving_t, upper_t>;

template <typename T>
using cm_covector4_t = tensor_t<T, 4, comoving_t, lower_t>;

template <typename T, typename... ranks>
using cm_tensor4_t = tensor_t<T, 4, comoving_t, ranks...>;

template <typename T, size_t i0 = 0, size_t i1 = 1, typename... ranks>
using sym_tensor3_t = sym2_tensor_t<T, 3, eulerian_t, i0, i1, ranks...>;
template <typename T, size_t i0 = 0, size_t i1 = 1, typename... ranks>
using sym_tensor4_t = sym2_tensor_t<T, 4, eulerian_t, i0, i1, ranks...>;


//metric
// SYM: make this symmetric
template<typename data_t, size_t ndim>
using metric_tensor_t =
      general_tensor_t<data_t, any_frame_t, generic_symmetry_t<ndim,2>, 2, std::tuple<lower_t, lower_t>, ndim>;
template<typename data_t, size_t ndim>
using invmetric_tensor_t =
      general_tensor_t<data_t, any_frame_t, generic_symmetry_t<ndim,2>, 2, std::tuple<upper_t, upper_t>, ndim>;

} // namespace tensors

#endif
