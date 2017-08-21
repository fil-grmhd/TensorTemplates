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

//Important shift needs to be int, not size_t, to not trigger a bug in the Intel compiler
//! Defines nodes for a onesided stencil, in positive or negative direction
template<bool flip, int shift = 0>
struct onesided_nodes {
  inline static constexpr int node(int const i) {
//    return  flip ?   i - shift
//                 : -(i - shift);
      return (i-shift)*(-1 + 2*flip);
  }
};

//! Defines nodes for a central stencil
struct central_nodes {
  inline static constexpr int node(int const i) {
    int half = i / 2;
    int node = half + (i % 2);
    return (i % 2) ?  node
                   : -node;
  }
};

//! Computes FD weights based on the given stencil / nodes
//  See https://www.researchgate.net/publication/242922236_Generation_of_Finite_Difference_Formulas_on_Arbitrarily_Spaced_Grids
//  for the algorithm, here x0 = 0 and equidistant nodes are assumed.
//  Could in principle be generalized, see
//  https://doi.org/10.1137/S0036144596322507
//
//  BE AWARE that order+1 is not really the order of accuracy of the FD
//           but the number of terms used to compute the FD
template<int deriv_, int order_, typename node_t_>
struct fd_weights {
  //! Some constants
  //! This includes weights up to deriv'th derivative
  static constexpr int deriv = deriv_;
  //! This includes weights for up to order terms
  static constexpr int order = order_;

  //! The nodes / stencil type
  using node_t = node_t_;

  //! Weights array
  //double ws[deriv+1][order+1][order+1];
  double ws[(deriv+1)*(order+1)*(order+1)];

//The Intel compiler likes Fortran indexing of 1d arrays instead of nested pointers...
  constexpr int index(int const i, int const j, int const k) const{
      return k + (order+1)*( j + (order+1)*i);
  };

  //! Returns weight of i'th node of deriv'th derivative with order+1 terms
  constexpr double weight(int const i) const {
    //return ws[deriv][order][i];
    return ws[index(deriv,order,i)];
  }

  //! Constructor filling the constexpr weights array
  constexpr fd_weights() : ws {} {

    // For more information on the algorithm see ref above
    ws[index(0,0,0)] = 1;
//    ws[0][0][0] = 1;
    double c1 = 1;

    for(int n = 1; n<=order; ++n) {
      double c2 = 1;
      for(int v = 0; v < n; ++v) {
        double c3 = node_t::node(n) - node_t::node(v);
        c2 *= c3;
        for(int m = 0; m <= std::min(n,deriv); ++m) {
//          double c4 = ws[m][n-1][v];
          double c4 = ws[index(m,n-1,v)];
          if(m-1 >= 0)
            ws[index(m,n,v)] = (node_t::node(n)*c4 - m*ws[index(m-1,n-1,v)])/c3;
//            ws[m][n][v] = (node_t::node(n)*c4 - m*ws[m-1][n-1][v])/c3;
          else
            ws[index(m,n,v)] = node_t::node(n)*c4/c3;
//            ws[m][n][v] = node_t::node(n)*c4/c3;
        }
      }
      for(int m = 0; m <= std::min(n,deriv); ++m) {
//        double c4 = ws[m][n-1][n-1];
        double c4 = ws[index(m,n-1,n-1)];
        double c5 = 1;
        if(m-1 >= 0)
          c5 = ws[index(m-1,n-1,n-1)];
//          c5 = ws[m-1][n-1][n-1];

        ws[index(m,n,n)] = c1/c2*(m*c5 - node_t::node(n-1) * c4);
//        ws[m][n][n] = c1/c2*(m*c5 - node_t::node(n-1) * c4);
      }
      c1 = c2;
    }
  }
};

//! Template recursion to compute FD given the weights
template<int N, typename weights_t, typename T>
struct fd_add_recursion {
  static inline __attribute__ ((always_inline)) T sum(T const * const grid_ptr, int const stride) {
    // Generate fd weights given the nodes / stencil type
    // we don't care regenerating this over and over again, because it is all computed at compile-time,
    // if compile-time is slow, check if computing this only once helps
    constexpr weights_t fd_w;
    constexpr double w = fd_w.weight(N);
    constexpr int node = weights_t::node_t::node(N);

    return w * grid_ptr[node * stride]
         + fd_add_recursion<N-1,weights_t,T>::sum(grid_ptr,stride);
  }
};
template<typename weights_t, typename T>
struct fd_add_recursion<0,weights_t,T> {
  static inline __attribute__ ((always_inline)) T sum(T const * const grid_ptr, int const stride) {
    // Generate fd weights given the nodes / stencil type
    // we don't care regenerating this over and over again, because it is all computed at compile-time,
    // if compile-time is slow, check if computing this only once helps
    constexpr weights_t fd_w;
    constexpr double w = fd_w.weight(0);
    constexpr int node = weights_t::node_t::node(0);

    return w * grid_ptr[node * stride];
  }
};


//! Computes deriv_order'th FD of given nodes / stencil type with order+1 terms
template<int deriv_order, int order, typename node_t, typename T>
inline __attribute__ ((always_inline)) T auto_diff(T const * const ptr, int const index, int const stride) {
  // Consistency check
  static_assert(deriv_order <= order,
                "n'th derivative needs at least n nodes (order >= deriv_order).");

  using weights_t = fd_weights<deriv_order,order,node_t>;
  // Shift grid pointer to x0
  T const * const x0_ptr = ptr + index;

  // Sum order+1 terms of FD to get deriv_order'th derivative
  return fd_add_recursion<order,weights_t,T>::sum(x0_ptr,stride);
}

#ifdef TENSORS_VECTORIZED
// THE FOLLOWING IS NOT USED
// the repeated loads into vector registers is not efficient

/*
///////////////////////////////////////////////////////////////////////////////
// Vectorized finite differences
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
    static inline __attribute__ ((always_inline)) decltype(auto)
      sum(int const stride, T const * const grid_ptr) {

      // grid values to be summed are at non-unit stride locations
      // this makes it inefficient to load them, so a simple loop is used
      Vc::Vector<T> weighted_value;
      for(int i = 0; i < Vc::Vector<T>::Size; ++i) {
        T const * const vector_index_ptr = grid_ptr + i;

        weighted_value[i] = vector_index_ptr[(order-N-dpoint)*stride];
      }

      return weighted_value * fd_stencils<order>::stencil[dpoint][order-N]
           + stencil_sum<N-1,dpoint,T>::sum(stride,grid_ptr);
    }
  };
  template<int dpoint, typename T>
  struct stencil_sum<0,dpoint,T> {
    static inline __attribute__ ((always_inline)) decltype(auto)
      sum(int const stride, T const * const grid_ptr) {
      // grid values to be summed are at non-unit stride locations
      // this makes it inefficient to load them, so a simple loop is used
      Vc::Vector<T> weighted_value;
      for(int i = 0; i < Vc::Vector<T>::Size; ++i) {
        T const * const vector_index_ptr = grid_ptr + i;

        weighted_value[i] = vector_index_ptr[(order-dpoint)*stride];
      }

      return weighted_value * fd_stencils<order>::stencil[dpoint][order];
    }
  };

  // Static diff: compute the finite difference using the given stencil
  // dpoint: Point in the stencil in which to compute the derivative
  template<int dpoint, typename T>
  static inline __attribute__ ((always_inline)) decltype(auto) sdiff(
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
inline __attribute__ ((always_inline)) decltype(auto) c_diff_v(T const * const grid_ptr, int const grid_index, int const stride) {

  constexpr int dpoint = order/2;
  T const * const index_ptr = grid_ptr + grid_index;

  return fd_vt<order>::template sdiff<dpoint>(index_ptr, stride);
}

#ifdef TENSORS_AUTOVEC
template<int dir, int order, typename T>
using c_diff = c_diff_v<dir,order,T>;
#endif
*/
#endif

} // namespace fd
} // namespace tensors

#endif
