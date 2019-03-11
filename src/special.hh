//  TensorTemplates: C++ tensor class templates
//  Copyright (C) 2018, Ludwig Jens Papenfort
//                      <papenfort@th.physik.uni-frankfurt.de>
//  Copyright (C) 2018, Elias Roland Most
//                      <emost@th.physik.uni-frankfurt.de>
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

#ifndef TENSORS_SPECIAL_HH
#define TENSORS_SPECIAL_HH

namespace tensors {
namespace general {

template <typename T, typename frame_t_,  size_t ndim_>
class kronecker_t
    : public tensor_expression_t<kronecker_t<T, frame_t_, ndim_>> {

public:
  using this_tensor_t = general_tensor_t<T, frame_t_, generic_symmetry_t<ndim_,2>, 2, std::tuple<upper_t,lower_t>,ndim_>;
  using property_t = general_tensor_property_t<this_tensor_t>;

  kronecker_t() = default;

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline __attribute__ ((always_inline)) decltype(auto)
  operator[](size_t i) const = delete;

  template <size_t index>
  inline __attribute__ ((always_inline)) static constexpr T evaluate(){

    constexpr size_t i0 = generic_symmetry_t<ndim_,2>::template uncompress_index<0,index>::value;
    constexpr size_t i1 = generic_symmetry_t<ndim_,2>::template uncompress_index<1,index>::value;


    return (i0==i1);
  }
};


template <typename T, typename frame_t_, typename... ranks>
class levi_civita_t
    : public tensor_expression_t<levi_civita_t<T, frame_t_, ranks...>> {

public:
  // this tensor is totally antisymmetric, but we have no type for this yet
  static constexpr size_t ndim = sizeof...(ranks);
  static constexpr size_t rank = sizeof...(ranks);

  using this_tensor_t = general_tensor_t<T, frame_t_, generic_symmetry_t<ndim, rank>, rank, std::tuple<ranks...>, ndim>;
  using property_t = general_tensor_property_t<this_tensor_t>;

protected:

  // double constexpr loop constructing the product of signums
  // see https://en.wikipedia.org/wiki/Levi-Civita_symbol

  // this is the partial product for fixed i over j
  template<size_t sym_compressed_index, size_t current_index>
  struct signum_fixed_index : public index_recursion_constexpr_t<signum_fixed_index<sym_compressed_index,current_index>,current_index+1,rank-1> {
    template<int ind>
    inline static constexpr decltype(auto) call() {
      constexpr int current_val = property_t::symmetry_t::template uncompress_index<current_index,sym_compressed_index>::value;
      constexpr int index_val = property_t::symmetry_t::template uncompress_index<ind,sym_compressed_index>::value;

      constexpr int diff = index_val - current_val;
      return utilities::static_sign<diff>::value;
    }
  };
  // loop over i < j
  template<size_t sym_cind>
  struct signum_product : public index_recursion_constexpr_t<signum_product<sym_cind>,0,rank-2> {
    template<int ind>
    inline static decltype(auto) call() {
      return signum_fixed_index<sym_cind,ind>::template traverse<op_mult>();
    }
  };

public:

  levi_civita_t() = default;

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline __attribute__ ((always_inline)) decltype(auto)
  operator[](size_t i) const = delete;

  template <size_t cindex>
  inline __attribute__ ((always_inline)) static constexpr T evaluate(){
    constexpr int sym_cind = property_t::symmetry_t::template index_from_generic<cindex>::value;

    return signum_product<sym_cind>::template traverse<op_mult>();
  }
};


} // namespace general
} // namespace tensors

#endif
