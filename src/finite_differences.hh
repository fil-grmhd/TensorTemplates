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
template<size_t order>
struct fd {
        // Width of the stencil
        enum {width = order + 1};

        // Weights of the stencil: so that stencil[p][i] is the weight of
        // at the ith point when the derivative is computed at the pth point
        // of the stencil, for example, if order == 6, stencil[3][i] gives
        // the weights for the centered finite-difference
        static double const stencil[width][width];

        // Static diff: compute the finite difference using the given stencil
        static double sdiff(
                // Grid function to differentiate at the diff. point
                double const * const grid_function_ijk,
                // Point in the stensil in which to compute the derivative
                int dpoint,
                // Vector stride
                int stride) {
            double d = 0;
            for(int p = 0; p < width; ++p) {
                d += stencil[dpoint][p] * grid_function_ijk[(p-dpoint)*stride];
            }
            return d;
        }
};

typedef double (*stencil_t)(double const * const, int, int);

template<size_t dir, size_t order, typename T>
inline T c_diff(T const * const ptr, size_t const index, size_t const stride) {
  constexpr size_t dpoint = order/2;

  T const * const index_ptr = ptr + index;

  stencil_t const dop[8] = {&fd<1>::sdiff, &fd<2>::sdiff, &fd<3>::sdiff,
      &fd<4>::sdiff, &fd<5>::sdiff, &fd<6>::sdiff, &fd<7>::sdiff,
      &fd<8>::sdiff};
  return (*dop[order-1])(index_ptr, dpoint, stride);
}

} // namespace fd
} // namespace tensors

#endif
