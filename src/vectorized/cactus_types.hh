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

template<size_t order_>
struct cactus_cdiff_v {
  static constexpr size_t order = order_;

  cGH const * const cctkGH;
  // one over dx,dy,dz
  std::array<Vc::double_v,3> const idx;

  cactus_cdiff(cGH const * const cctkGH_, double dx, double dy, double dz)
             : cctkGH(cctkGH_), idx({Vc::double_v(1.0/dx),
                                     Vc::double_v(1.0/dy),
                                     Vc::double_v(1.0/dz)}) {}

  // get stride from cactus grid
  template<size_t dir>
  inline size_t stride() const {
    return (dir == 0)*(CCTK_GFINDEX3D(cctkGH, 1, 0, 0) - CCTK_GFINDEX3D(cctkGH, 0, 0, 0))
         + (dir == 1)*(CCTK_GFINDEX3D(cctkGH, 0, 1, 0) - CCTK_GFINDEX3D(cctkGH, 0, 0, 0))
         + (dir == 2)*(CCTK_GFINDEX3D(cctkGH, 0, 0, 1) - CCTK_GFINDEX3D(cctkGH, 0, 0, 0));
  }

  // compute central difference, uses general cdiff interface
  template<size_t dir, typename T>
  inline decltype(auto) diff(T const * const ptr, size_t const index) const {
    // in this case we want to get a vector register of derivatives
    Vc::Vector<T> vec_register;

    // all attempts to vectorize c_diff explicitly ended up in slower code...
    for(size_t i = 0; i < Vc::Vector<T>::Size; ++i) {
      vec_register[i] = idx[dir]*c_diff<dir,order>(ptr,index+i,this->stride<dir>);
    }
    return vec_register;
  }
};

} // namespace fd
} // namespace tensors

#endif
