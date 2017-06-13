#ifndef TENSOR_CONTRACT_HH
#define TENSOR_CONTRACT_HH

#include <cassert>

namespace tensors {

//! Expression template for generic tensor contractions
template <size_t i1, size_t i2, typename E1, typename E2>
class tensor_contraction_t
    : public tensor_expression_t<tensor_contraction_t<i1, i2, E1, E2>> {
  // references to both tensors
  E1 const &_u;
  E2 const &_v;

public:
  // Contraction changes the tensor type, thus a special property class is
  // needed.
  // It gives all the relevant properties of a tensor resulting from a
  // contraction
  using property_t = index_reduction_property_t<i1, i2, E1, E2>;

  tensor_contraction_t(E1 const &u, E2 const &v) : _u(u), _v(v){};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline decltype(auto)
  operator[](size_t i) const = delete;

  //! Sum of contracted components for a specific index computed by a template
  //! recursion
  template <int N, int stride1, int stride2> struct recursive_contract {
    template <typename A, typename B>
    static inline decltype(auto) contract(A const &_u, B const &_v) {
      return recursive_contract<(N - 1), stride1, stride2>::contract(_u, _v) +
             _u.template evaluate<
                 stride1 +
                 N * utilities::static_pow<property_t::ndim, i1>::value>() *
                 _v.template evaluate<
                     stride2 +
                     N * utilities::static_pow<property_t::ndim, i2>::value>();
    };
  };
  template <int stride1, int stride2>
  struct recursive_contract<0, stride1, stride2> {
    template <typename A, typename B>
    static inline decltype(auto) contract(A const &_u, B const &_v) {
      return _u.template evaluate<stride1>() * _v.template evaluate<stride2>();
    };
  };

  template <size_t i, size_t index, size_t ndim>
  static constexpr size_t restore_index_and_compute_stride() {
    return (i == 0)
               ? index * ndim
               : (index - (index % utilities::static_pow<ndim, i>::value)) *
                         ndim +
                     (index % utilities::static_pow<ndim, i>::value);
  }

  template <size_t c_index> inline decltype(auto) evaluate() const {

    //      TUPLE FREE TENSOR CONTRACTION      //
    constexpr size_t max_pow_E1 = E1::property_t::rank - 2;

    constexpr size_t ndim = E1::property_t::ndim;

    constexpr size_t index_part_E1 =
        c_index % (utilities::static_pow<ndim, max_pow_E1 + 1>::value);
    constexpr size_t index_part_E2 = c_index - index_part_E1;

    constexpr size_t index_part_E2_n =
        index_part_E2 / utilities::static_pow<ndim, max_pow_E1 + 1>::value;

    constexpr size_t stride_1 =
        restore_index_and_compute_stride<i1, index_part_E1, ndim>();
    constexpr size_t stride_2 =
        restore_index_and_compute_stride<i2, index_part_E2_n, ndim>();

    static_assert(
        stride_1 >= 0,
        "contraction: stride is less than zero, this shouldn't happen");
    static_assert(
        stride_2 >= 0,
        "contraction: stride is less than zero, this shouldn't happen");

    // Compute sum over contracted index for index_r'th component by template
    // recursion
    return recursive_contract<property_t::ndim - 1, stride_1,
                              stride_2>::contract(_u, _v);
  }
};

//! Contraction "operator" for two tensor expressions
template <size_t i1, size_t i2, typename E1, typename E2>
tensor_contraction_t<i1, i2, E1, E2> const inline contract(E1 const &u,
                                                           E2 const &v) {
  return tensor_contraction_t<i1, i2, E1, E2>(u, v);
}

//! Expression template for generic tensor contractions
template <size_t i2, typename E1, typename E2>
class metric_contraction_t
    : public tensor_expression_t<metric_contraction_t<i2, E1, E2>> {
  // references to both tensors
  E1 const &_u;
  E2 const &_v;

public:
  // Contraction changes the tensor type, thus a special property class is
  // needed.
  // It gives all the relevant properties of a tensor resulting from a
  // contraction
  using property_t = index_metric_contraction_property_t<i2, E1, E2>;
  static constexpr size_t i1 = 1;

  metric_contraction_t(E1 const &u, E2 const &v) : _u(u), _v(v){};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline decltype(auto)
  operator[](size_t i) const = delete;

  //! Sum of contracted components for a specific index computed by a template
  //! recursion
  template <int N, int stride1, int stride2> struct recursive_contract {
    template <typename A, typename B>
    static inline decltype(auto) contract(A const &_u, B const &_v) {
      return recursive_contract<(N - 1), stride1, stride2>::contract(_u, _v) +
             _u.template evaluate<
                 stride1 +
                 N * utilities::static_pow<property_t::ndim, i1>::value>() *
                 _v.template evaluate<
                     stride2 +
                     N * utilities::static_pow<property_t::ndim, i2>::value>();
    };
  };
  template <int stride1, int stride2>
  struct recursive_contract<0, stride1, stride2> {
    template <typename A, typename B>
    static inline decltype(auto) contract(A const &_u, B const &_v) {
      return _u.template evaluate<stride1>() * _v.template evaluate<stride2>();
    };
  };

  template <size_t c_index> inline decltype(auto) evaluate() const {

    //      TUPLE FREE TENSOR CONTRACTION      //
    constexpr size_t max_pow_E1 = E1::property_t::rank - 1;

    constexpr size_t ndim = E1::property_t::ndim;

    constexpr size_t index_part_less_i2 =
        c_index % (utilities::static_pow<ndim, i2 + 1>::value);
    constexpr size_t index_part_less_i2_m1 =
        index_part_less_i2 % (utilities::static_pow<ndim, i2>::value);
    constexpr size_t index_part_metric =
        (index_part_less_i2 - index_part_less_i2_m1) /
        (utilities::static_pow<ndim, i2>::value);

    constexpr size_t stride_1 = index_part_metric * ndim;
    constexpr size_t stride_2 =
        c_index - (index_part_less_i2 - index_part_less_i2_m1);

    static_assert(
        stride_1 >= 0,
        "contraction: stride is less than zero, this shouldn't happen");
    static_assert(
        stride_2 >= 0,
        "contraction: stride is less than zero, this shouldn't happen");

    // Compute sum over contracted index for index_r'th component by template
    // recursion
    return recursive_contract<ndim - 1, stride_1, stride_2>::contract(_u, _v);
  }
};
}
#endif
