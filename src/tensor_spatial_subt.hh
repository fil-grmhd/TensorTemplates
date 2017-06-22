#ifndef TENSOR_SPATIAL_HH
#define TENSOR_SPATIAL_HH

namespace tensors {

//! Expression template for a spatial subtensor expression
template <class E1>
class tensor_spatial_sub_t
    : public tensor_expression_t<tensor_spatial_sub_t<E1>> {
  // references to full tensor expression
  E1 const &_u;

public:
  using property_t = general_tensor_property_t<general_tensor_t<
      typename E1::property_t::data_t, typename E1::property_t::frame_t,
      E1::property_t::rank, typename E1::property_t::index_t,
      E1::property_t::ndim - 1>>;

  tensor_spatial_sub_t(E1 const &u) : _u(u){};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline decltype(auto)
  operator[](size_t i) const = delete;

  // Note: The spatial tensor has ndim -> ndim -1
  // So when we want to translate the index we need to bare in mind that
  // for this spatial part we have i_j*(ndim-1)^j with (i_j < ndim-1)
  // whereas the full tensor has k_j*(ndim)^j with (k_j < ndim)
  // So we need to a) account for the shift from i to k, i.e.
  // k_j = i_j +1 (since we always cut out the 0th component)
  // and b) that we uncompress with ndim - 1 but compress again
  // with ndim. 
  template <size_t c_index, size_t ind0, size_t... Indices>
  struct compute_new_cindex {
    static constexpr size_t value =
        compute_new_cindex<c_index, Indices...>::value
       +(uncompress_index_t<property_t::ndim, ind0, c_index>::value + 1)
       *(utilities::static_pow<E1::property_t::ndim, ind0>::value);
  };
  template <size_t c_index, size_t ind0>
  struct compute_new_cindex<c_index, ind0> {
    static constexpr size_t value =
        (uncompress_index_t<property_t::ndim, ind0, c_index>::value + 1)
       *(utilities::static_pow<E1::property_t::ndim, ind0>::value);
  };

  template <size_t c_index, size_t... Indices>
  static constexpr size_t
  compute_new_cindex_wrapper(std::index_sequence<Indices...>) {
    return compute_new_cindex<c_index, Indices...>::value;
  }

  template <size_t c_index,
            typename Indices = std::make_index_sequence<property_t::rank>>
  inline typename property_t::data_t const evaluate() const {

    return _u.template evaluate<
                         compute_new_cindex_wrapper<c_index>(Indices{})
                       >();
  }
};

template <typename E1>
tensor_spatial_sub_t<E1> const inline spatial_part(E1 const &u) {
  return tensor_spatial_sub_t<E1>(u);
};
}
#endif
