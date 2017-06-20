#ifndef TENSOR_REORDER_HH
#define TENSOR_REORDER_HH

#include <cassert>

namespace tensors {

//! Expression template for generic tensor contractions
template <class E1, size_t... Ind>
class tensor_reorder_index_t
    : public tensor_expression_t<tensor_reorder_index_t<E1>> {
  // references to both tensors
  E1 const &_u;

public:
  // Reordering the indices changes the tensor type, thus a special property class is
  // needed.
  // It gives all the relevant properties of a tensor resulting from a
  // contraction

  static_assert(sizeof...(Ind) == E1::property_t::rank, "You must specify as many indices as the rank!");
 
  using index_t = std::tuple<typename std::tuple_element<Ind,typename E1::property_t::index_t>::type...>;


  using property_t = general_tensor_property_t<general_tensor_t<
    		     typename E1::property_t::data_t,
    		     typename E1::property_t::frame_t,
    		     E1::property_t::rank,
		     index_t,
    		     E1::property_t::ndim>>;
    		     

  tensor_reorder_index_t(E1 const &u) : _u(u) {};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline decltype(auto)
  operator[](size_t i) const = delete;

  template <size_t c_index ,size_t ind0, size_t... Indices>
  struct compute_new_cindex{
    static constexpr size_t value = compute_new_cindex<c_index,Indices...>::value
      		  + uncompress_index_t<E1::property_t::ndim,
      				      ind0,
				      c_index>::value * 
		           (utilities::static_pow<E1::property_t::ndim, 
		            E1::property_t::rank -sizeof...(Indices)-1>::value);
  };

  template <size_t c_index, size_t ind0>
  struct compute_new_cindex<E1::property_t::ndim,c_index,ind0>{
    static constexpr size_t value = uncompress_index_t<E1::property_t::ndim, 
		     		      ind0,c_index>::value
				    *(utilities::static_pow<E1::property_t::ndim,E1::property_t::rank-1>::value);
  };


  template <size_t c_index> inline typename property_t::data_t const evaluate() const {

    return _u.template evaluate<compute_new_cindex<c_index,Ind...>::value>();
  }
};

//! Contraction "operator" for two tensor expressions
template <size_t... Indices>
tensor_reorder_index_t<E1, Indices...> const inline reorder_index(tensor_expression_t const &u){
  return tensor_reorder_index_t<E1, Indices...>(u);
};

}
#endif
