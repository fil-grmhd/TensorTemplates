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

namespace tensors {
// forward decleration of general tensor class
template <typename T, typename frame_t_, typename symmetry_t_, size_t rank_, typename index_t_,
          size_t ndim_>
class general_tensor_t;

namespace general {
// forward decleration of metric types
template <typename T>
class metric_tensor3_t;

template <typename T>
class inv_metric_tensor3_t;


// general tensor typedefs

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

template <typename T, typename frame_t_,  size_t ndim_>
class kronecker_t;

template <typename T, typename frame_t_, typename... ranks>
class levi_civita_t;

template<typename T>
using kronecker3_t = general::kronecker_t<T,any_frame_t,3>;
template<typename T>
using kronecker4_t = general::kronecker_t<T,any_frame_t,4>;

template <typename T>
using levi_civita3_up_t = general::levi_civita_t<T,any_frame_t,upper_t,upper_t,upper_t>;
template <typename T>
using levi_civita4_up_t = general::levi_civita_t<T,any_frame_t,upper_t,upper_t,upper_t,upper_t>;


template <typename T>
using levi_civita3_down_t = general::levi_civita_t<T,any_frame_t,lower_t,lower_t,lower_t>;
template <typename T>
using levi_civita4_down_t = general::levi_civita_t<T,any_frame_t,lower_t,lower_t,lower_t,lower_t>;


// component type
template<typename T>
using comp_t = T;

// scalar loop inc
template<typename T>
constexpr size_t loop_inc = 1;
} // namespace general

#if !defined(TENSORS_VECTORIZED) || !defined(TENSORS_AUTOVEC)

// import general tensor typedefs to tensors namespace,
// if vectorization or autovec is disabled
using namespace general;

#endif // !TENSORS_VECTORIZED || !TENSORS_AUTOVEC

#ifdef TENSORS_VECTORIZED
// Vectorized tensor typedefs
#ifdef TENSORS_TSIMD
#ifdef __AVX512F__
 static constexpr int tsimd_width = 8;
#else
 static constexpr int tsimd_width = 4;
#endif

template<typename T>
using tensors_vector_t = tsimd::pack<T,tsimd_width>;
#else

template<typename T>
using tensors_vector_t = Vc::native_simd<T>;

#endif

// Tensor of rank index types, e.g. lower_t, upper_t, ...
template <typename T, size_t ndim_, typename frame_t_, typename... ranks>

using tensor_vt = general_tensor_t<tensors_vector_t<T>, frame_t_, generic_symmetry_t<ndim_,sizeof...(ranks)>,

                                  sizeof...(ranks),
                                  std::tuple<ranks...>, ndim_>;

// Symmetric tensor in two indices of rank index types, e.g. lower_t, upper_t, ...
template <typename T, size_t ndim_, typename frame_t_, size_t i0, size_t i1, typename... ranks>

using sym2_tensor_vt = general_tensor_t<tensors_vector_t<T>, frame_t_, sym2_symmetry_t<ndim_,sizeof...(ranks),i0,i1>,

                                  sizeof...(ranks),
                                  std::tuple<ranks...>, ndim_>;


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

template <typename T, typename frame_t_,  size_t ndim_>
using kronecker_vt = general::kronecker_t<tensors_vector_t<T>,frame_t_,ndim_>;

template <typename T, typename frame_t_, typename... ranks>
using levi_civita_vt = general::levi_civita_t<tensors_vector_t<T>,frame_t_,ranks...>;


template<typename T>
using kronecker3_vt = kronecker_vt<T,any_frame_t,3>;
template<typename T>
using kronecker4_vt = kronecker_vt<T,any_frame_t,4>;

template <typename T>
using levi_civita3_up_vt = levi_civita_vt<T,any_frame_t,upper_t,upper_t,upper_t>;
template <typename T>
using levi_civita4_up_vt = levi_civita_vt<T,any_frame_t,upper_t,upper_t,upper_t,upper_t>;

template <typename T>
using levi_civita3_down_vt = levi_civita_vt<T,any_frame_t,lower_t,lower_t,lower_t>;
template <typename T>
using levi_civita4_down_vt = levi_civita_vt<T,any_frame_t,lower_t,lower_t,lower_t,lower_t>;

// component type
template<typename T>
using comp_vt = tensors_vector_t<T>;

// loop vector increment
template<typename T>
#ifdef TENSORS_TSIMD
constexpr size_t loop_vinc = tensors_vector_t<T>::static_size;
#else
constexpr size_t loop_vinc = tensors_vector_t<T>::size();
#endif

// metric type
template<typename T>
using metric_tensor3_vt = general::metric_tensor3_t<tensors_vector_t<T>>;


// deliver vectorized types as default types if code is vectorization agnostic
#ifdef TENSORS_AUTOVEC

template <typename T, size_t ndim_, typename frame_t_, typename... ranks>
using tensor_t = tensor_vt<T, ndim_, frame_t_, ranks...>;

template <typename T, size_t ndim_, typename frame_t_, size_t i0, size_t i1, typename... ranks>
using sym2_tensor_t = sym2_tensor_vt<T, ndim_, frame_t_, i0, i1, ranks...>;

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

template <typename T, typename frame_t_,  size_t ndim_>
using kronecker_t = kronecker_vt<T,frame_t_,ndim_>;

template <typename T, typename frame_t_, typename... ranks>
using levi_civita_t = levi_civita_vt<T,frame_t_,ranks...>;

template<typename T>
using kronecker3_t = kronecker3_vt<T>;
template<typename T>
using kronecker4_t = kronecker4_vt<T>;

template <typename T>
using levi_civita3_up_t = levi_civita3_up_vt<T>;
template <typename T>
using levi_civita4_up_t = levi_civita4_up_vt<T>;

template <typename T>
using levi_civita3_down_t = levi_civita3_down_vt<T>;
template <typename T>
using levi_civita4_down_t = levi_civita4_down_vt<T>;

// component type
template<typename T>
using comp_t = comp_vt<T>;

// loop vector increment
template<typename T>
constexpr size_t loop_inc = loop_vinc<T>;

// metric type
template<typename T>
using metric_tensor3_t = metric_tensor3_vt<T>;

#endif // TENSORS_AUTOVEC

#endif // TENSORS_VECTORIZED

} // namespace tensors

#endif
