#ifndef TENSOR_REORDER_HH
#define TENSOR_REORDER_HH

namespace tensors {

//! Expression template for a index reordered tensor
template <class E, size_t... Ind>
class tensor_reorder_index_t
    : public tensor_expression_t<tensor_reorder_index_t<E,Ind...>> {
  // reference to reordered tensor
  E const &_u;

public:
  // Reordering the indices changes the tensor type, thus a special property class is
  // needed.
  static_assert(sizeof...(Ind) == E::property_t::rank, "You must specify as many indices as the rank!");

  using index_t = std::tuple<typename std::tuple_element<Ind,typename E::property_t::index_t>::type...>;

  using property_t = general_tensor_property_t<
                       general_tensor_t<
                         typename E::property_t::data_t,
    		                 typename E::property_t::frame_t,
    		                 E::property_t::rank,
		                     index_t,
    		                 E::property_t::ndim
    		               >
    		             >;

  tensor_reorder_index_t(E const &u) : _u(u) {};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline decltype(auto)
  operator[](size_t i) const = delete;


  // this needs some comments
  template <size_t c_index ,size_t ind0, size_t... Indices>
  struct compute_new_cindex{
    static constexpr size_t value = compute_new_cindex<c_index,Indices...>::value
      		  + generic_symmetry_t<property_t::ndim,property_t::rank>::template uncompress_index<ind0,
				      c_index>::value *
		           (utilities::static_pow<property_t::ndim,
		            property_t::rank -sizeof...(Indices)-1>::value);
  };
  template <size_t c_index, size_t ind0>
  struct compute_new_cindex<c_index,ind0>{
    static constexpr size_t value = generic_symmetry_t<property_t::ndim,property_t::rank>::template uncompress_index<ind0,c_index>::value
				    *(utilities::static_pow<property_t::ndim,property_t::rank-1>::value);
  };


  template <size_t index> inline typename property_t::data_t const evaluate() const {

    return _u.template evaluate<compute_new_cindex<index,Ind...>::value>();
  }
};

//! Reordering "operator" for a tensor expressions
template <size_t ...Indices, typename E>
tensor_reorder_index_t<E, Indices... > const inline reorder_index(E const &u){
  return tensor_reorder_index_t<E, Indices...>(u);
};

}
#endif
