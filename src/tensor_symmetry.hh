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

#ifndef TENSOR_SYMMETRY_HH
#define TENSOR_SYMMETRY_HH

namespace tensors {

///////////////////////////////////////////////////////////////////////////////
//! Generic tensor symmetry type
//
//  Represents no symmetry
///////////////////////////////////////////////////////////////////////////////
template<size_t ndim, size_t rank>
struct generic_symmetry_t {
  //! Compile-time constant ndof
  static constexpr size_t ndof = utilities::static_pow<ndim,rank>::value;

  //! Computes the compressed index given template parameter indices
  template <size_t a, size_t... indices>
  struct compressed_index {
    static constexpr size_t value =
        compressed_index<indices...>::value * ndim + a;
  };
  // termination definition of recursion
  template <size_t a>
  struct compressed_index<a> {
    static constexpr size_t value = a;
  };

  //! Uncompresses single index of position index_pos from compressed index
  //! index
  template <size_t index_pos, size_t index>
  struct uncompress_index {
    static constexpr size_t value =
        static_cast<size_t>(index /
                            utilities::static_pow<ndim, index_pos>::value) %
        ndim;
  };

  //! Index transformation given an compressed index of this type
  //  Does nothing, index is already a generic compressed index
  template <size_t index>
  struct index_to_generic {
    static constexpr size_t value = index;
  };
  //! Index transformation given compressed index of generic type
  //  Does nothing, index is already a generic compressed index
  template <size_t index>
  struct index_from_generic {
    static constexpr size_t value = index;
  };

  //! Returns true if reduction of index i_red preserves this symmetry
  //  In this case there is no symmetry, thus it is always preserved
  template <size_t i_red>
  struct is_reduction_symmetric {
    static constexpr bool value = true;
  };
};

///////////////////////////////////////////////////////////////////////////////
//! Symmetric2 tensor symmetry type
//
//  Represents a symmetry of the two indices i0,i1
///////////////////////////////////////////////////////////////////////////////

template<size_t ndim, size_t rank, size_t i0, size_t i1>
struct sym2_symmetry_t {
  //! Compile-time constant ndof
  static constexpr size_t ndof = utilities::static_pow<ndim,rank-1>::value*(ndim+1)/2;

  static_assert((i0 < rank) && (i1 < rank),
                "One or both symmetry indices are out of bound, i.e. i0,i1 >= rank");
  static_assert(i0 < i1,
                "Make sure that symmetric indices are in ascending order and not equal");

  // Internal compressed index transformation, based on symmetric index positions
  // i0 and i1 are (internally) always shifted to be the first two indices
  template <size_t N_sym, size_t i0_val, size_t i1_val, size_t a, size_t... indices>
  struct compressed_index_impl {
    static constexpr size_t index_pos = rank - (sizeof...(indices) + 1);
    static constexpr size_t value = (i0 == index_pos)
                                   *(compressed_index_impl<N_sym+1,a,i1_val,indices...>::value)
                                  + (i1 == index_pos)
                                   *(compressed_index_impl<N_sym+1,i0_val,a,indices...>::value)
                                  + ((i0 != index_pos) && (i1 != index_pos))
                                   *(
                                      a*ndof/utilities::static_pow<ndim,sizeof...(indices)+N_sym-1>::value
                                    + compressed_index_impl<N_sym,i0_val,i1_val,indices...>::value
                                    );
  };
  template <size_t N_sym, size_t i0_val, size_t i1_val, size_t a>
  struct compressed_index_impl<N_sym,i0_val,i1_val,a> {
    static constexpr size_t value = (N_sym == 1)
                                   *(
                                      (a < i0_val)
                                     *(a*ndim-a*(a+1)/2+i0_val)
                                    + (a >= i0_val)
                                     *(i0_val*ndim-i0_val*(i0_val+1)/2+a)
                                    )
                                  + (N_sym == 2)
                                   *(
                                      a*ndof/utilities::static_pow<ndim,1>::value
                                    + (i1_val < i0_val)
                                     *(i1_val*ndim-i1_val*(i1_val+1)/2+i0_val)
                                    + (i1_val >= i0_val)
                                     *(i0_val*ndim-i0_val*(i0_val+1)/2+i1_val)
                                    );
  };







  // this is still producing wrong results
  template <size_t uncomp_index, size_t ind0, size_t ind1, size_t index_pos>
  struct compressed_index_impl_test {
  private:
    static constexpr bool asc = ind0 < ind1;
    static constexpr size_t c_ind0 = (!asc) * ind0 +   asc  * (ind0 * (ndim - (ind0 + 1) / 2));
    static constexpr size_t c_ind1 =   asc  * ind1 + (!asc) * (ind1 * (ndim - (ind1 + 1) / 2));
  public:
    static constexpr size_t value = (i0 == index_pos)
                                   *(c_ind0)
                                  + (i1 == index_pos)
                                   *(c_ind1)
                                  + ((i0 != index_pos) && (i1 != index_pos))
                                   *(uncomp_index * (ndim*(ndim+1)/2)
                                   * utilities::static_pow<ndim,index_pos - (index_pos > i0) - (index_pos > i1)>::value);
  };





  //! Computes the compressed index given template parameter indices
  template <size_t a, size_t... indices>
  struct compressed_index {
    static constexpr size_t value = compressed_index_impl<0,0,0,a,indices...>::value;
  };


  //! Template recursion to get smaller index of symmetric indices from compressed index
  template <size_t N, size_t index>
  struct uncompress_smaller_impl {
  private:
    static constexpr size_t tail = index / (ndim-N);
  public:
    static constexpr size_t value = (tail == 0) ? N : uncompress_smaller_impl<N+1,index-(ndim-N)>::value;
  };
  // Template recursion termination
  template <size_t index>
  struct uncompress_smaller_impl<ndim-1,index> {
    static constexpr size_t value = ndim-1;
  };

  //! Uncompress smaller symmetry index from compressed symmetric indices
  template <size_t index>
  struct uncompress_smaller {
    static constexpr size_t value = uncompress_smaller_impl<0,index>::value;
  };

  //! Uncompresses single index of position index_pos from compressed index
  //! index
  template <size_t index_pos, size_t index>
  struct uncompress_index {
  private:
    // divisor to get non-symmetric compressed index part
    static constexpr size_t div = utilities::static_pow<
                                    ndim,
                                    index_pos + 1 - (index_pos > i0) - (index_pos > i1)
                                  >::value
                                * (ndim + 1) / 2;
    // part of compressed index which belongs to the symmetric indices
    static constexpr size_t sym_index = index % (ndim * (ndim + 1) / 2);

    // decompress smaller index first, they are symmetric anyway
    static constexpr size_t ind0 = uncompress_smaller<sym_index>::value;
    // compute bigger index from smaller index
    static constexpr size_t ind1 = sym_index - ind0 * ndim + ind0 * (ind0 + 1) / 2;
  public:
    static constexpr size_t value =
        (index_pos == i0)
       *ind0
      + (index_pos == i1)
       *ind1
      + ((index_pos != i0) && (index_pos != i1))
       *((index / div) % ndim);
  };

  //! Template recursion to transform from symmetric compressed to generic compressed index
  template <size_t index_pos, size_t index>
  struct transform_sym2_impl {
    static constexpr size_t value = uncompress_index<index_pos,index>::value
                                   *utilities::static_pow<ndim,index_pos>::value
                                  + transform_sym2_impl<index_pos-1,index>::value;
  };
  template <size_t index>
  struct transform_sym2_impl<0,index> {
    static constexpr size_t value = uncompress_index<0,index>::value;
  };

  //! Template recursion to transform from symmetric compressed to generic compressed index
  template <size_t index_pos, size_t index>
  struct transform_gen_impl {
  private:
    static constexpr size_t ind0 = generic_symmetry_t<ndim,rank>
                                      ::template uncompress_index<i0,index>::value;
    static constexpr size_t ind1 = generic_symmetry_t<ndim,rank>
                                      ::template uncompress_index<i1,index>::value;
    static constexpr size_t uncomp_index = generic_symmetry_t<ndim,rank>
                                             ::template uncompress_index<index_pos,index>::value;
  public:
    static constexpr size_t value =  compressed_index_impl_test<uncomp_index,ind0,ind1,index_pos>::value
                                   + transform_gen_impl<index_pos-1,index>::value;
  };

  template <size_t index>
  struct transform_gen_impl<0,index> {
  private:
    static constexpr size_t ind0 = generic_symmetry_t<ndim,rank>
                                      ::template uncompress_index<i0,index>::value;
    static constexpr size_t ind1 = generic_symmetry_t<ndim,rank>
                                      ::template uncompress_index<i1,index>::value;
    static constexpr size_t uncomp_index = generic_symmetry_t<ndim,rank>
                                             ::template uncompress_index<0,index>::value;
  public:
    static constexpr size_t value = compressed_index_impl_test<uncomp_index,ind0,ind1,0>::value;
  };

  //! Index transformation given an compressed index of this type,
  //! resulting in a generic compressed index
  template <size_t index>
  struct index_to_generic {
    static constexpr size_t value = transform_sym2_impl<rank-1,index>::value;
  };
  //! Index transformation given compressed index of generic type,
  //! resulting in a sym2 compressed index
  template <size_t index>
  struct index_from_generic {
    static constexpr size_t value = transform_gen_impl<rank-1,index>::value;
  };

  //! Returns true if reduction of index i_red preserves this symmetry
  template <size_t i_red>
  struct is_reduction_symmetric {
    static constexpr bool value = (i_red != i0) && (i_red != i1);
  };
};

} // namespace tensors

#endif
