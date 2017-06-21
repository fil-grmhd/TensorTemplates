#ifndef TENSOR_SLICE_HH
#define TENSOR_SLICE_HH

#include <cassert>

namespace tensors {

//! Expression template for generic tensor contractions
template <typename E1, int... Ind>
class tensor_slice_t : public tensor_expression_t<tensor_slice_t<E1, Ind...>> {
  // references to both tensors
  E1 const &_u;

public:
  static_assert(utilities::all_true<
                    ((Ind >= -1) &&
                     Ind < static_cast<int>(E1::property_t::ndim))...>::value,
                "Indices out of range. Use 0..ndim -1 to fix indices for the "
                "slice and -1 to indicate a free index");

  // Count number of free indices
  template <int i0, int... Indices> struct count_free_indices {
    static constexpr size_t value =
        (i0 == -1) + count_free_indices<Indices...>::value;
  };

  template <int i0> struct count_free_indices<i0> {
    static constexpr size_t value = (i0 == -1);
  };

  static constexpr size_t rank = count_free_indices<Ind...>::value;

  // Count number of free indices
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
                                    typename E1::property_t::index_t>::type;
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
      typename E1::property_t::data_t, typename E1::property_t::frame_t, rank,
      decltype(slice_index_t_wrapper_wrapper()), E1::property_t::ndim>>;

  tensor_slice_t(E1 const &u) : _u(u){};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline decltype(auto)
  operator[](size_t i) const = delete;

  template <size_t c_index, size_t N_slice, int ind0, int... Indices>
  struct compute_new_cindex {
    static constexpr size_t value =
        (ind0 >= 0) * ind0 *
            utilities::static_pow<E1::property_t::ndim,
                                  E1::property_t::rank - sizeof...(Indices) -
                                      1>::value +
        (ind0 == -1) *
            (uncompress_index_t<property_t::ndim, N_slice, c_index>::value) *
            (utilities::static_pow<E1::property_t::ndim,
                                   E1::property_t::rank - sizeof...(Indices) -
                                       1>::value)
        // Recursion
        +
        compute_new_cindex<c_index, N_slice + (ind0 == -1), Indices...>::value;
  };

  template <size_t c_index, size_t N_slice, int ind0>
  struct compute_new_cindex<c_index, N_slice, ind0> {
    static constexpr size_t value =
        (ind0 >= 0) * ind0 *
            utilities::static_pow<E1::property_t::ndim,
                                  E1::property_t::rank - 1>::value +
        (ind0 == -1) *
            (uncompress_index_t<property_t::ndim, N_slice, c_index>::value) *
            (utilities::static_pow<E1::property_t::ndim,
                                   E1::property_t::rank - 1>::value);
  };

  template <size_t c_index>
  inline typename property_t::data_t const evaluate() const {

    return _u
        .template evaluate<compute_new_cindex<c_index, 0, Ind...>::value>();
  }
};

template <int... Indices, typename E1>
tensor_slice_t<E1, Indices...> const inline slice(E1 const &u) {
  return tensor_slice_t<E1, Indices...>(u);
};
}
#endif
