#ifndef TENSOR_PRODUCT_HH
#define TENSOR_PRODUCT_HH

#include <utility>
#include "utilities.hh"

namespace tensors {

//! Expression template for a generic tensor product
template <typename E1, typename E2>
class tensor_product_t
    : public tensor_expression_t<tensor_product_t<E1, E2>> {
  // references to both tensors
  E1 const &_u;
  E2 const &_v;

  static constexpr bool using_E1 = E1::property_t::rank >= E2::property_t::rank;

public:
  // Concatination changes the tensor type, thus a special property class is
  // needed.
  using property_t = typename std::conditional< using_E1,
	typename E1::property_t, typename E2::property_t>::type;

  static_assert(E1::ndim == E2::ndim, "Dimensions don't match");

  template<std::size_t... I>
  tensor_product_t(E1 const &u, E2 const &v, std::index_sequence<I...>) : _u(u), _v(v){
    static_assert(utilities::all_true<(std::is_same< 
		    typename std::tuple_element<I, typename E1::property_t::index_t>::type,
		    typename std::tuple_element<I, typename E2::property_t::index_t>::type>::value
		   )...>::value, "Indices not consistent");
  };

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline __attribute__ ((always_inline)) decltype(auto)
  operator[](size_t i) const = delete;

  template <size_t index>
  inline __attribute__ ((always_inline)) decltype(auto) evaluate() const {


    constexpr size_t max_rank = using_E1 ? E2::property_t::rank : E1::property_t::rank;

    constexpr size_t ndim = E1::property_t::ndim;

    constexpr size_t index_part_E1 = using_E1 ? index :
        index % (utilities::static_pow<ndim, max_rank>::value);
    constexpr size_t index_part_E2 = using_E1 ? 
        index % (utilities::static_pow<ndim, max_rank>::value) : index;

    return _u.template evaluate<index_part_E1>()*_v.template evaluate<index_part_E2>();
  }
};

//! Tensor product "operator" for two tensor expressions
template <typename E1, typename E2, typename Indices = 
std::make_index_sequence<  E1::property_t::rank >= E2::property_t::rank ? E2::property_t::rank : E1::property_t::rank>>
tensor_product_t<E1, E2> const inline __attribute__ ((always_inline)) tensor_product(E1 const &u,
                                                           E2 const &v) {
  return tensor_product_t<E1, E2>(u, v, Indices{});
};

}
#endif
