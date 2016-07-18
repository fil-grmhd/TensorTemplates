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

#ifndef TENSORS_MULTIINDEX_HH
#define TENSORS_MULTIINDEX_HH

namespace tensors {

///////////////////////////////////////////////////////////////////////////////
// Multiindex helper
///////////////////////////////////////////////////////////////////////////////
/*
 *  \tparam ndim_ number of dimensions
 *  \tparam rank_ number of indices
 */
template<size_t ndim_, size_t rank_>
class multiindex_t {
  static size_t constexpr rank = rank_;
  static size_t constexpr ndim = ndim_;

  // saves the current set of indices
  size_t current_indices[rank_];
  // reached last index?
  bool finished;

  public:

  // Member initialization
  multiindex_t() : current_indices { 0 }, finished(false) { };
  // Increment operator, to be used in for loops
  inline multiindex_t<ndim,rank>& operator++() {
    // loop through all indices
    for(size_t i = 0; i<rank; ++i) {
      // increment index up to ndim-1
      if(current_indices[i] < ndim-1) {
        // return on increment of a single index
        current_indices[i]++;
        return *this;
      }
      else {
      // Restart counting if ndim-1 is reached
        current_indices[i] = 0;
      }
    }
    // Return and end if all indices are equal ndim-1
    finished = true;
    return *this;
  }
  // Reset indices
  inline void reset() {
    for(size_t i = 0; i<rank; ++i) {
      current_indices[i] = 0;
    }
    finished = false;
  }
  // Index access operator
  inline size_t operator[](size_t const i) const {
    return current_indices[i];
  }
  inline size_t& operator[](size_t const i) {
    return current_indices[i];
  }
  // Termination criterion
  inline bool end() const {
    return finished;
  }
  // Distribute free indices in a contraction
  /*
   *  \tparam contracted contracted index
   *  \tparam ndim_ number of dimensions
   *  \tparam rank_ number of free indices
   */
  template<size_t contracted, size_t ndim__, size_t rank__>
  inline void distribute(multiindex_t<ndim__,rank__> const & mi, size_t& offset) {
    static_assert(contracted < rank,
                  "utils::tensor::multiindex_t: "
                  "contracted index must be smaller than rank.");
    static_assert(ndim == ndim__,"utils::tensor::multiindex_t: "
                  "Cannot distribute indices of more/fewer dimensions (yet).");

    for(size_t i = 0; i<rank; ++i) {
      // set only free indices
      if(i != contracted) {
        current_indices[i] = mi[offset];
        // increase index offset wrt free indices
        ++offset;
      }
    }
  }
  // Distribute free indices in a trace
  /*
   *  \tparam traced0 first traced index
   *  \tparam traced1 second traced index
   *  \tparam ndim_ number of dimensions
   *  \tparam rank_ number of free indices
   */
  template<size_t traced0, size_t traced1, size_t ndim__, size_t rank__>
  inline void distribute(multiindex_t<ndim__,rank__> const & mi) {
    static_assert((traced0 < rank) && (traced1 < rank),
                  "utils::tensor::multiindex_t: "
                  "Contracted index must be smaller than rank.");
    static_assert(ndim == ndim__,"utils::tensor::multiindex_t: "
                  "Cannot distribute indices of more/fewer dimensions (yet).");

    size_t offset = 0;
    for(size_t i = 0; i<rank; ++i) {
      // set only free indices
      if((i != traced0) && (i != traced1)) {
        current_indices[i] = mi[offset];
        // increase index offset wrt free indices
        ++offset;
      }
    }
  }
  // Comparison operators
  // Is any index greater or smaller than a
  inline bool operator>(size_t const a) {
    for(size_t i = 0; i<rank; ++i) {
      if(current_indices[i] <= a) {
        return false;
      }
    }
    return true;
  }
  inline bool operator<(size_t const a) {
    for(size_t i = 0; i<rank; ++i) {
      if(current_indices[i] >= a) {
        return false;
      }
    }
    return true;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Helper function templates
///////////////////////////////////////////////////////////////////////////////
// Output stream overload for multiindex type
template<size_t ndim, size_t rank, typename ostream_t>
ostream_t& operator<<(ostream_t& os, const multiindex_t<ndim,rank>& mi) {
  os << "(";
  for(size_t i = 0; i < rank; ++i) {
    os << mi[i];
    if(i < rank-1)
      os << ",";
  }
  os << ")";
  return os;
}

} // namespace tensors

#endif
