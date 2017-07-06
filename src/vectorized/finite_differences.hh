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

#ifndef TENSORS_FINITEDIFF_VEC_HH
#define TENSORS_FINITEDIFF_VEC_HH

#include <iostream>

namespace tensors {
namespace fd {

///////////////////////////////////////////////////////////////////////////////
// Finite differences (based on Cactus Thorn FDCore by David Radice)
///////////////////////////////////////////////////////////////////////////////
template<int order_>
class fd_vt {
public:
  // Order of the fd
  static constexpr int order = order_;

  // Width of the stencil
  static constexpr int width = order + 1;

  // Weights of the stencil: so that stencil[p][i] is the weight of
  // at the ith point when the derivative is computed at the pth point
  // of the stencil, for example, if order == 6, stencil[3][i] gives
  // the weights for the centered finite-difference
//  static constexpr double stencil[order+1][order+1] = fd_stencils<order>::stencil;

  // template recursion to compute all terms of the fd
  //
  // testing showed that doing this more naturally in a reversed order N-dpoint -> N
  // leads to larger relative errors when compared to the original FDCore derivatives
  template<int N, int dpoint, typename T>
  struct stencil_sum {
    static inline decltype(auto) sum(int const stride,
                                     T const * const grid_ptr) {

      // grid values to be summed are at non-unit stride locations
      // this makes it inefficient to load them, so a simple loop is used
      Vc::Vector<T> weighted_value;
      for(int i = 0; i < Vc::Vector<T>::Size; ++i) {
        T const * const vector_index_ptr = grid_ptr + i;

        weighted_value[i] = vector_index_ptr[(order-N-dpoint)*stride]*fd_stencils<order>::stencil[dpoint][order-N];
      }

      return weighted_value
           + stencil_sum<N-1,dpoint,T>::sum(stride,grid_ptr);
    }
  };
  template<int dpoint, typename T>
  struct stencil_sum<0,dpoint,T> {
    static inline decltype(auto) sum(int const stride,
                                     T const * const grid_ptr) {
      // grid values to be summed are at non-unit stride locations
      // this makes it inefficient to load them, so a simple loop is used
      Vc::Vector<T> weighted_value;
      for(int i = 0; i < Vc::Vector<T>::Size; ++i) {
        T const * const vector_index_ptr = grid_ptr + i;
        weighted_value[i] = vector_index_ptr[(order-dpoint)*stride]*fd_stencils<order>::stencil[dpoint][order];
      }

      return weighted_value;
    }
  };

  // Static diff: compute the finite difference using the given stencil
  // dpoint: Point in the stencil in which to compute the derivative
  template<int dpoint, typename T>
  static inline decltype(auto) sdiff(
          // Grid function to differentiate at the diff. point
          T const * const grid_ptr,
          // Vector stride
          int const  stride) {

    // sum all terms from order...0
    return stencil_sum<order,dpoint,T>::sum(stride,grid_ptr);
  }
};

// "Vectorized" central fd interface
// Unfortunately slower than the non-vectorized version
template<int dir, int order, typename T>
inline decltype(auto) c_diff_v(T const * const grid_ptr, int const grid_index, int const stride) {

  constexpr int dpoint = order/2;
  T const * const index_ptr = grid_ptr + grid_index;

  return fd_vt<order>::sdiff<dpoint>(index_ptr, stride);
}

} // namespace fd
} // namespace tensors

#endif
