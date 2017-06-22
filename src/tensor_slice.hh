#ifndef TENSOR_SLICE_HH
#define TENSOR_SLICE_HH

namespace tensors {

//! Expression template for generic tensor expression slice
template <typename E, int... Ind>
class tensor_slice_t : public tensor_expression_t<tensor_slice_t<E, Ind...>> {
  // references to sliced expression
  E const &_u;

  // can only shift dimensions up to a two component (co)vector
  // a single component can be accessed directly...
  static constexpr size_t max_shift = E::property_t::ndim-2;

public:
  static_assert(utilities::all_true<
                  ((Ind >= -static_cast<int>(max_shift + 1)) &&
                   (Ind < static_cast<int>(E::property_t::ndim)))...
                >::value,
                "Indices out of range. Use 0..ndim-1 to fix indices "
                "and -1...-(ndim - 2)-1 to indicate a (shifted) free index.");

  static constexpr size_t rank = count_free_indices<Ind...>::value;
  static constexpr size_t ndim = E::property_t::ndim - free_index_shift<Ind...>::value;

  // The following needs some comments
  template <size_t N, size_t N_count, int i0, int... Indices>
  struct get_free_indices {
    static constexpr size_t value =
        (N + 1 > N_count) +
        get_free_indices<N, N_count + (i0 == -1), Indices...>::value;
  };

  template <size_t N, size_t N_count, int i0>
  struct get_free_indices<N, N_count, i0> {
    static constexpr size_t value = (N + 1 > N_count) - 1;
  };

  template <size_t N> struct get_index_t {
    using type =
        typename std::tuple_element<get_free_indices<N, 0, Ind...>::value,
                                    typename E::property_t::index_t>::type;
  };

  template <size_t... Ind_slice> struct slice_index_t {
    using type = std::tuple<typename get_index_t<Ind_slice>::type...>;
  };

  template <size_t... Ind_slice>
  static constexpr decltype(auto)
  slice_index_t_wrapper(std::index_sequence<Ind_slice...>) {
    return typename slice_index_t<Ind_slice...>::type{};
  };

  template <typename Ind_s_t =
                std::make_index_sequence<count_free_indices<Ind...>::value>>
  static constexpr decltype(auto) slice_index_t_wrapper_wrapper() {
    return slice_index_t_wrapper(Ind_s_t{});
  };

  using property_t = general_tensor_property_t<general_tensor_t<
      typename E::property_t::data_t, typename E::property_t::frame_t, rank,
      decltype(slice_index_t_wrapper_wrapper()), ndim>>;

  tensor_slice_t(E const &u) : _u(u){};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline decltype(auto)
  operator[](size_t i) const = delete;

  template <size_t c_index>
  inline typename property_t::data_t const evaluate() const {

    return _u.template evaluate<
                         compute_unsliced_cindex<
                           E,
                           E,
                           c_index,
                           0,
                           Ind...
                         >::value
                       >();
  }
};

template <int... Indices, typename E>
tensor_slice_t<E, Indices...> const inline slice(E const &u) {
  return tensor_slice_t<E, Indices...>(u);
};
}
#endif
