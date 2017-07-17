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

#ifndef TENSORS_CACTUS_VEC_HH
#define TENSORS_CACTUS_VEC_HH


namespace tensors {
namespace fd {

template<size_t order_>
struct cactus_cdiff_v {
  static constexpr size_t order = order_;

  cGH const * const cctkGH;

  // one over dx,dy,dz
  std::array<double,3> const idx;

  // unit stride in x,y,z
  std::array<int,3> const stride;

  cactus_cdiff_v(cGH const * const cctkGH_, double dx, double dy, double dz)
             : cctkGH(cctkGH_),
               idx({1.0/dx,1.0/dy,1.0/dz}),
               stride({CCTK_GFINDEX3D(cctkGH, 1, 0, 0) - CCTK_GFINDEX3D(cctkGH, 0, 0, 0),
                       CCTK_GFINDEX3D(cctkGH, 0, 1, 0) - CCTK_GFINDEX3D(cctkGH, 0, 0, 0),
                       CCTK_GFINDEX3D(cctkGH, 0, 0, 1) - CCTK_GFINDEX3D(cctkGH, 0, 0, 0)}) {}

  // compute central difference, uses general cdiff interface
  template<size_t dir, typename T>
  inline __attribute__ ((always_inline)) decltype(auto) diff(T const * const ptr, size_t const index) const {
    // this is sufficient, in principle, but generates less optimal code
    // due to repeated loads into vector registers for every point on the stencil
    //return idx[dir]*c_diff_v<dir,order>(ptr,index,stride[dir]);


    // in this case we want to get a vector register of derivatives
    Vc::Vector<T> vec_register;

    for(size_t i = 0; i < Vc::Vector<T>::Size; ++i) {
      vec_register[i] = idx[dir]*c_diff<dir,order>(ptr,index+i,stride[dir]);
    }
    return vec_register;
  }
};

} // namespace fd
} // namespace tensors

#endif
