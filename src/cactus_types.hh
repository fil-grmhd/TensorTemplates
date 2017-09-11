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

#ifndef TENSORS_CACTUS_HH
#define TENSORS_CACTUS_HH


namespace tensors {
namespace fd {

#if !defined(TENSORS_VECTORIZED) || !defined(TENSORS_AUTOVEC)
//! Computes d'th derivative by a FD with order+1 terms on a Cactus GF using a node_t stencil
//  BE AWARE that order+1 is really just the number of terms and in general NOT the order of accuracy
template<size_t d_, size_t order_, typename node_t_>
struct cactus_diff {
  static constexpr size_t order = order_;
  static constexpr size_t d = d_;

  using node_t = node_t_;

  cGH const * const cctkGH;
  // one over dx^d, dy^d, dz^d
  std::array<double,3> const idx;
  // unit stride in x,y,z
  std::array<int,3> const stride;

  // the d'th derivative expects dx_i^d as prefactor
  cactus_diff(cGH const * const cctkGH_, double dxd, double dyd, double dzd)
             : cctkGH(cctkGH_),
               idx({1.0/dxd,1.0/dyd,1.0/dzd}),
               stride({CCTK_GFINDEX3D(cctkGH, 1, 0, 0) - CCTK_GFINDEX3D(cctkGH, 0, 0, 0),
                       CCTK_GFINDEX3D(cctkGH, 0, 1, 0) - CCTK_GFINDEX3D(cctkGH, 0, 0, 0),
                       CCTK_GFINDEX3D(cctkGH, 0, 0, 1) - CCTK_GFINDEX3D(cctkGH, 0, 0, 0)}) {}

  //! The actual FD routine called within a tensor field
  template<size_t dir, typename T>
  inline __attribute__ ((always_inline)) decltype(auto) diff(T const * const ptr, size_t const index) const {
    return idx[dir]*auto_diff<d,order,node_t>(ptr, index, stride[dir]);
  }
};
#endif

#ifdef TENSORS_VECTORIZED
//! Computes d'th derivative by a FD with order+1 terms on a Cactus GF using a node_t stencil
//  BE AWARE that order+1 is really just the number of terms and in general NOT the order of accuracy
template<size_t d_, size_t order_, typename node_t_>
struct cactus_diff_v {
  static constexpr size_t order = order_;
  static constexpr size_t d = d_;

  using node_t = node_t_;

  cGH const * const cctkGH;

  // one over dx^d, dy^d, dz^d
  std::array<double,3> const idx;

  // unit stride in x,y,z
  std::array<int,3> const stride;

  // the d'th derivative expects dx_i^d as prefactor
  cactus_diff_v(cGH const * const cctkGH_, double dxd, double dyd, double dzd)
             : cctkGH(cctkGH_),
               idx({1.0/dxd,1.0/dyd,1.0/dzd}),
               stride({CCTK_GFINDEX3D(cctkGH, 1, 0, 0) - CCTK_GFINDEX3D(cctkGH, 0, 0, 0),
                       CCTK_GFINDEX3D(cctkGH, 0, 1, 0) - CCTK_GFINDEX3D(cctkGH, 0, 0, 0),
                       CCTK_GFINDEX3D(cctkGH, 0, 0, 1) - CCTK_GFINDEX3D(cctkGH, 0, 0, 0)}) {}

  //! The actual FD routine called within a tensor field
  template<size_t dir, typename T>
  inline __attribute__ ((always_inline)) decltype(auto) diff(T const * const ptr, size_t const index) const {
    // in this case we want to get a vector register of derivatives
    Vc::Vector<T> vec_register;

    for(size_t i = 0; i < Vc::Vector<T>::Size; ++i) {
      vec_register[i] = idx[dir]*auto_diff<d,order,node_t>(ptr, index + i, stride[dir]);
    }
    return vec_register;
  }
};

// define default type, if vectorization agnostic code is used
#ifdef TENSORS_AUTOVEC
template<size_t d, size_t order, typename node_t>
using cactus_diff = cactus_diff_v<d,order,node_t>;
#endif

#endif

} // namespace fd
} // namespace tensors

#endif
