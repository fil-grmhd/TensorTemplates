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

//Symmetric2 cast to assign tensor expressions
//
template <typename E, size_t i0, size_t i1>
class tensor_sym2_cast_t
    : public tensor_expression_t<tensor_sym2_cast_t<E,i0,i1>> {
  E const &_v;

public:
  using property_t = general_tensor_property_t<
    		     general_tensor_t<
		       typename E::property_t::data_t,
		       typename E::property_t::frame_t,
		       sym2_symmetry_t<E::property_t::ndim,
			 	       E::property_t::rank,
		       		       i0,i1>,
		       E::property_t::rank,
		       typename E::property_t::index_t,
		       E::property_t::ndim>
    		     >;

  tensor_sym2_cast_t( E const &v) : _v(v){};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline decltype(auto)
  operator[](size_t i) const = delete;

  template <size_t index> 
  inline const typename E::property_t::data_t evaluate() const {
    return _v.template evaluate<index>();
  };
};

template<size_t i0, size_t i1, typename E>
inline tensor_sym2_cast_t<E,i0,i1> sym2_cast(E const &v){
  return tensor_sym2_cast_t<E,i0,i1>(v);
}



} // namespace tensors

#endif
