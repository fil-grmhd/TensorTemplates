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

#ifndef TENSORS_SYMMETRY_HH
#define TENSORS_SYMMETRY_HH

#include "utilities.hh"
#include "multiindex.hh"

namespace tensors {

///////////////////////////////////////////////////////////////////////////////
//! Generic tensor symmetry type
///////////////////////////////////////////////////////////////////////////////

class generic_symmetry_t {
    public:
        //! Returns compile-time constant ndof
        template<size_t ndim, size_t rank>
        static inline constexpr size_t ndof() {
          return utilities::static_pow<ndim,rank>::result;
        }

        //! Computes the compressed index associated with the given indices
        /*!
         *  The data is stored in a row major format
         */
        template<size_t ndim, size_t rank>
        static inline size_t compress_indices(size_t const a) {
            return a;
        }
        template<size_t ndim, size_t rank, typename... indices_t>
        static inline size_t compress_indices(size_t const a, indices_t... indices) {
            return a*utilities::static_pow<ndim,sizeof...(indices)>::result
                    + compress_indices<ndim,rank>(indices...);
        }

        //! Computes the compressed index from the given multiindex object
        /*!
         *  The data is stored in a row major format
         */
        template<size_t ndim, size_t rank>
        static inline size_t compress_indices(multiindex_t<ndim,rank> const & mi) {
            size_t compressed_index = 0;
            for(size_t i = 0; i<rank; ++i) {
              size_t d_pow = 1;
              for(size_t j = i; j<rank-1; ++j) {
                d_pow *= ndim;
              }
              compressed_index += d_pow*mi[i];
            }
            return compressed_index;
        }
};

///////////////////////////////////////////////////////////////////////////////
//! Symmetric2 tensor symmetry type
///////////////////////////////////////////////////////////////////////////////

class symmetric2_symmetry_t {
    public:
        //! Returns compile-time constant ndof
        template<size_t ndim, size_t rank>
        static inline constexpr size_t ndof() {
          return utilities::static_pow<ndim,rank-1>::result*(ndim+1)/2;
        }

        //! Computes the compressed index associated with the given indices
        /*!
         *  The data is stored in a row major format.
         *  Symmetric components g(i,j) with i > j are not stored so that, for
         *  instance a symmetric2<2, 3> tensor is stored as  [Txxx, Txxy, Txyy,
         *  Tyxx, Tyxy, Tyyy]
         */
        template<size_t ndim, size_t rank>
        static inline size_t compress_indices(size_t const a, size_t const b) {
            if(b < a) {
                return b*ndim-b*(b+1)/2+a;
            }
            return a*ndim-a*(a+1)/2+b;
        }
        template<size_t ndim, size_t rank, typename... indices_t>
        static inline size_t compress_indices(size_t const a, size_t const b, indices_t... indices) {
            return a*ndof<ndim,rank>()
                   /utilities::static_pow<ndim,rank-sizeof...(indices)>::result
                   + compress_indices<ndim,rank>(b,indices...);
        }

        //! Computes the compressed index from the given multiindex object
        /*!
         *  The data is stored in a row major format
         */
        template<size_t ndim, size_t rank>
        static inline size_t compress_indices(multiindex_t<ndim,rank> const & mi) {
            size_t compressed_index = 0;
            for(size_t i = 0; i<rank-2; ++i) {
              size_t d_pow = 1;
              for(size_t j = i; j<rank-2; ++j) {
                d_pow *= ndim;
              }
              compressed_index += d_pow*(ndim+1)*mi[i]/2;
            }
            if(mi[rank-1] < mi[rank-2]) {
              return compressed_index
                     +mi[rank-1]*ndim
                     -mi[rank-1]*(mi[rank-1]+1)/2
                     +mi[rank-2];
            }
            return compressed_index
                   +mi[rank-2]*ndim
                   -mi[rank-2]*(mi[rank-2]+1)/2
                   +mi[rank-1];
        }
};

} // namespace tensors

#endif
