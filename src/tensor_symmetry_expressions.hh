//  TensorTemplates: C++ tensor class templates
//  Copyright (C) 2017, Elias Roland Most (ERM)
//                      <emost@itp.uni-frankfurt.de>
//  Copyright (C) 2018, Ludwig Jens Papenfort
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


/*
 * =====================================================================================
 *
 *       Filename:  tensor_symmetry_expressions.hh
 *
 *    Description:  Tensor expression related to symmetric types
 *
 *        Version:  1.0
 *        Created:  26/06/2017 17:31:24
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Elias Roland Most (ERM), emost@itp.uni-frankfurt.de
 *   Organization:  Goethe University Frankfurt
 *
 * =====================================================================================
 */

#ifndef TENSOR_SYMMETRY_EXP_HH
#define TENSOR_SYMMETRY_EXP_HH

namespace tensors {

// Cast to different symmetry_t
//
template <typename E, typename symmetry_t_>
class symmetry_cast_t
    : public tensor_expression_t<symmetry_cast_t<E,symmetry_t_>> {

  using E_t = operant_t<E>;

  E_t _v;

public:
  using symmetry_t = symmetry_t_;

  using property_t = general_tensor_property_t<
    		     general_tensor_t<
		       typename E::property_t::data_t,
		       typename E::property_t::frame_t,
		       symmetry_t,
		       E::property_t::rank,
		       typename E::property_t::index_t,
		       E::property_t::ndim>
    		     >;

  symmetry_cast_t( E const &v) : _v(v){};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline __attribute__ ((always_inline)) decltype(auto)
  operator[](size_t i) const = delete;

  template <size_t index>
  inline __attribute__ ((always_inline)) decltype(auto) evaluate() const {
    // cast generic index to symmetric one and back
    // this makes sure, that always the same elements are accessed
    // i.e. sym2_cast(generic_tensor).evaluate<index>() with index = compress(2,1)
    // should always end up at index = compress(1,2)
    constexpr size_t sym_index = symmetry_t::template index_from_generic<index>::value;
    constexpr size_t gen_index = symmetry_t::template index_to_generic<sym_index>::value;

    return _v.template evaluate<gen_index>();
  };
};


template<size_t i0 = 0, size_t i1 = 1, typename E>
inline __attribute__ ((always_inline)) decltype(auto) sym2_cast(E const &v) {
    using symmetry_t = sym2_symmetry_t<
                         E::property_t::ndim,
                         E::property_t::rank,
                         i0,i1>;

    return symmetry_cast_t<E,symmetry_t>(v);
}

template<typename E>
inline __attribute__ ((always_inline)) decltype(auto) generic_cast(E const &v) {
    using symmetry_t = generic_symmetry_t<
                         E::property_t::ndim,
                         E::property_t::rank>;

    return symmetry_cast_t<E,symmetry_t>(v);
}

} // namespace tensors

#endif
