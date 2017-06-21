#ifndef TENSOR_CONCAT_HH
#define TENSOR_CONCAT_HH

namespace tensors {

//! Expression template for a generic tensor product
template <typename E1, typename E2>
class tensor_concat_t
    : public tensor_expression_t<tensor_concat_t<E1, E2>> {
  // references to both tensors
  E1 const &_u;
  E2 const &_v;

public:
  // Concatination changes the tensor type, thus a special property class is
  // needed.
  using property_t = index_concat_property_t<E1, E2>;

  tensor_concat_t(E1 const &u, E2 const &v) : _u(u), _v(v){};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline decltype(auto)
  operator[](size_t i) const = delete;

  template <size_t c_index> inline typename property_t::data_t const evaluate() const {
    constexpr size_t max_pow_E1 = E1::property_t::rank;

    constexpr size_t ndim = E1::property_t::ndim;

    constexpr size_t index_part_E1 =
        c_index % (utilities::static_pow<ndim, max_pow_E1>::value);
    constexpr size_t index_part_E2 = c_index - index_part_E1;

    constexpr size_t index_part_E2_n =
        index_part_E2 / utilities::static_pow<ndim, max_pow_E1>::value;

    return _u.template evaluate<index_part_E1>()*_v.template evaluate<index_part_E2_n>();
  }
};

//! Tensor product "operator" for two tensor expressions
template <typename E1, typename E2>
tensor_concat_t<E1, E2> const inline tensor_cat(E1 const &u,
                                                           E2 const &v) {
  return tensor_concat_t<E1, E2>(u, v);
};

}
#endif
