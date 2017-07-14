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

#ifndef TENSORS_FINITEDIFF_HH
#define TENSORS_FINITEDIFF_HH

namespace tensors {
namespace fd {
///////////////////////////////////////////////////////////////////////////////
// Finite difference stencils (taken from Cactus Thorn FDCore by David Radice)
///////////////////////////////////////////////////////////////////////////////

//! Template for stencils
template<int order>
struct fd_stencils {
  static_assert(order == order, "Can't find a stencil for the given fd order!");
};

//! Template specializations for every (known) stencil
template<>
struct fd_stencils<1> {
  static constexpr int width = 2;
  static constexpr double stencil[width][width] = {
    {-1.0, 1.0},
    {-1.0, 1.0}
  };
};

template<>
struct fd_stencils<2> {
  static constexpr int width = 3;
  static constexpr double stencil[width][width] = {
    {-1.5,  2.0, -0.5},
    {-0.5,  0,    0.5},
    { 0.5, -2.0,  1.5}
  };
};

template<>
struct fd_stencils<3> {
  static constexpr int width = 4;
  static constexpr double stencil[width][width] = {
    {-1.8333333333333333, 3., -1.5, 0.33333333333333333},
    {-0.33333333333333333, -0.5, 1.0, -0.16666666666666667},
    {0.16666666666666667, -1.0, 0.5, 0.33333333333333333},
    {-0.33333333333333333, 1.5, -3.0, 1.8333333333333333}
  };
};

template<>
struct fd_stencils<4> {
  static constexpr int width = 5;
  static constexpr double stencil[width][width] = {
    {-2.0833333333333335, 4, -3, 1.3333333333333333, -0.25},
    {-0.25,-0.8333333333333334,1.5,-0.5,0.08333333333333333},
    {0.08333333333333333,-0.6666666666666666,0, 0.6666666666666666,
        -0.08333333333333333},
    {-0.08333333333333333,0.5,-1.5,0.8333333333333334,0.25},
    {0.25,-1.3333333333333333,3,-4,2.0833333333333335}

  };
};

template<>
struct fd_stencils<5> {
  static constexpr int width = 6;
  static constexpr double stencil[width][width] = {
    {-2.2833333333333333, 5.0, -5.0, 3.3333333333333333, -1.25, 0.2},
    {-0.2, -1.0833333333333333, 2.0, -1.0, 0.33333333333333333, -0.05},
    {0.05, -0.5, -0.33333333333333333, 1.0, -0.25, 0.033333333333333333},
    {-0.033333333333333333, 0.25, -1.0, 0.33333333333333333, 0.5, -0.05},
    {0.05, -0.33333333333333333, 1.0, -2.0, 1.0833333333333333, 0.2},
    {-0.2, 1.25, -3.3333333333333333, 5.0, -5.0, 2.2833333333333333}
  };
};

template<>
struct fd_stencils<6> {
  static constexpr int width = 7;
  static constexpr double stencil[width][width] = {
    {-2.45,6,-7.5,6.666666666666667,-3.75,1.2, -0.16666666666666666},
    {-0.16666666666666666,-1.2833333333333334,2.5,-1.6666666666666667,
        0.8333333333333334,-0.25,0.03333333333333333},
    {0.03333333333333333,-0.4,-0.5833333333333334,1.3333333333333333,-0.5,
        0.13333333333333333,-0.016666666666666666},
    {-0.016666666666666666,0.15,-0.75,0,0.75,-0.15,0.016666666666666666},
    {0.016666666666666666,-0.13333333333333333,0.5,-1.3333333333333333,
        0.5833333333333334,0.4,-0.03333333333333333},
    {-0.03333333333333333,0.25,-0.8333333333333334,1.6666666666666667,-2.5,
        1.2833333333333334,0.16666666666666666},
    {0.16666666666666666,-1.2,3.75,-6.666666666666667,7.5,-6,2.45}
  };
};

template<>
struct fd_stencils<7> {
  static constexpr int width = 8;
  static constexpr double stencil[width][width] = {
    {-2.5928571428571429, 7.0, -10.5, 11.666666666666667, -8.75, 4.2,
        -1.1666666666666667, 0.14285714285714286},
    {-0.14285714285714286, -1.45, 3.0, -2.5, 1.6666666666666667, -0.75, 0.2,
        -0.023809523809523810},
    {0.023809523809523810, -0.33333333333333333, -0.78333333333333333,
        1.6666666666666667, -0.83333333333333333, 0.33333333333333333,
        -0.083333333333333333, 0.0095238095238095238},
    {-0.0095238095238095238, 0.1, -0.6, -0.25, 1.0, -0.3,
        0.066666666666666667, -0.0071428571428571429},
    {0.0071428571428571429, -0.066666666666666667, 0.3, -1.0, 0.25,
        0.6, -0.1, 0.0095238095238095238},
    {-0.0095238095238095238, 0.083333333333333333, -0.33333333333333333,
        0.83333333333333333, -1.6666666666666667, 0.78333333333333333,
        0.33333333333333333, -0.023809523809523810},
    {0.023809523809523810, -0.2, 0.75, -1.6666666666666667, 2.5, -3.0,
        1.45, 0.14285714285714286},
    {-0.14285714285714286, 1.1666666666666667, -4.2, 8.75,
        -11.666666666666667, 10.5, -7.0, 2.5928571428571429}
  };
};

template<>
struct fd_stencils<8> {
  static constexpr int width = 9;
  static constexpr double stencil[width][width] = {
    {-2.717857142857143,8,-14,18.666666666666668,-17.5,11.2,
        -4.666666666666667,1.1428571428571428,-0.125},
    {-0.125,-1.5928571428571427,3.5,-3.5,2.9166666666666665,-1.75,0.7,
        -0.16666666666666666,0.017857142857142856},
    {0.017857142857142856,-0.2857142857142857,-0.95,2,-1.25,
        0.6666666666666666,-0.25,0.05714285714285714,-0.005952380952380952},
    {-0.005952380952380952,0.07142857142857142,-0.5,-0.45,1.25,-0.5,
        0.16666666666666666,-0.03571428571428571,0.0035714285714285713},
    {0.0035714285714285713,-0.0380952380952381,0.2,-0.8,0,0.8,-0.2,
        0.0380952380952381,-0.0035714285714285713},
    {-0.0035714285714285713,0.03571428571428571,-0.16666666666666666,0.5,
        -1.25,0.45,0.5,-0.07142857142857142,0.005952380952380952},
    {0.005952380952380952,-0.05714285714285714,0.25,-0.6666666666666666,1.25,
        -2,0.95,0.2857142857142857,-0.017857142857142856},
    {-0.017857142857142856,0.16666666666666666,-0.7,1.75,-2.9166666666666665,
            3.5,-3.5,1.5928571428571427,0.125},
    {0.125,-1.1428571428571428,4.666666666666667,-11.2,17.5,
            -18.666666666666668,14,-8,2.717857142857143}
  };
};

// declare storage of static weight arrays
constexpr double fd_stencils<1>::stencil[2][2];
constexpr double fd_stencils<2>::stencil[3][3];
constexpr double fd_stencils<3>::stencil[4][4];
constexpr double fd_stencils<4>::stencil[5][5];
constexpr double fd_stencils<5>::stencil[6][6];
constexpr double fd_stencils<6>::stencil[7][7];
constexpr double fd_stencils<7>::stencil[8][8];
constexpr double fd_stencils<8>::stencil[9][9];

///////////////////////////////////////////////////////////////////////////////
// Finite differences (based on Cactus Thorn FDCore by David Radice)
///////////////////////////////////////////////////////////////////////////////
template<int order>
class fd {
public:
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
    static inline __attribute__ ((always_inline)) T sum(int const stride, T const * const grid_ptr) {
      return fd_stencils<order>::stencil[dpoint][order-N] * grid_ptr[(order-N-dpoint)*stride]
           + stencil_sum<N-1,dpoint,T>::sum(stride,grid_ptr);
    }
  };
  template<int dpoint, typename T>
  struct stencil_sum<0,dpoint,T> {
    static inline __attribute__ ((always_inline)) T sum(int const stride, T const * const grid_ptr) {
      return fd_stencils<order>::stencil[dpoint][order] * grid_ptr[(order-dpoint)*stride];
    }
  };

  // Static diff, compute the finite difference using the given stencil
  // dpoint: Point in the stencil in which to compute the derivative
  template<int dpoint, typename T>
  static inline __attribute__ ((always_inline)) T sdiff(
          // Grid function to differentiate at the diff. point
          T const * const grid_ptr,
          // Vector stride
          int const stride) {

    // sum all terms from order...0
    return stencil_sum<order,dpoint,T>::sum(stride,grid_ptr);
  }
};

template<int dir, int order, typename T>
inline __attribute__ ((always_inline)) T c_diff(T const * const ptr, int const index, int const stride) {

  constexpr int dpoint = order/2;
  T const * const index_ptr = ptr + index;

  return fd<order>::template sdiff<dpoint>(index_ptr, stride);
}

} // namespace fd
} // namespace tensors

#endif
