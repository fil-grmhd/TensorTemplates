#ifndef TENSOR_CONTRACT_HH
#define TENSOR_CONTRACT_HH

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
  using property_t = contraction_property_t<i1,i2,E1,E2>;

  tensor_contraction_t(E1 const &u, E2 const &v) : _u(u), _v(v){};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline decltype(auto)
  operator[](size_t i) const = delete;

  //! Sum of contracted components for a specific index computed by a template
  //! recursion
  template <int N, int stride1, int stride2> struct recursive_contract {
    template <typename A, typename B>
    static inline decltype(auto) contract(A const &_u, B const &_v) {
      return recursive_contract<(N - 1), stride1, stride2>::contract(_u, _v)
           + _u.template evaluate<
                 stride1 +
                 N * utilities::static_pow<property_t::ndim, i1>::value>()
           * _v.template evaluate<
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
    // if contracted index is the first one, just shift index by one power of ndim
    return (i == 0)
               ? index * ndim
    // if not, shift all indices above by one power of ndim and keep the rest
               : (index - (index % utilities::static_pow<ndim, i>::value)) *
                         ndim +
                     (index % utilities::static_pow<ndim, i>::value);
  }

  template <size_t index> inline typename property_t::data_t const evaluate() const {

    // TUPLE FREE TENSOR CONTRACTION, fix for problems with intel compiler
    constexpr size_t max_pow_E1 = E1::property_t::rank - 1;

    constexpr size_t ndim = E1::property_t::ndim;

    // index is compressed with respect to E1::rank + E2::rank - 2 indices
    // get part of index which belongs to E1 (without the contracted index)
    constexpr size_t index_part_E1 =
        index % (utilities::static_pow<ndim, max_pow_E1>::value);

    // get part of index which belongs to E2 (without the contracted index)
    constexpr size_t index_part_E2 = index - index_part_E1;
    // normalize, i.e. remove excess powers of ndim coming from E1
    constexpr size_t index_part_E2_n =
        index_part_E2 / utilities::static_pow<ndim, max_pow_E1>::value;

    // restore indices and compute the corresponding strides
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

    // Compute sum over contracted index for index'th component by template
    // recursion
    return recursive_contract<property_t::ndim - 1, stride_1,
                              stride_2>::contract(_u, _v);
  }
};

//! Helper structure to compute contraction of two vectors by a template recursion
template <typename E1, typename E2, size_t N>
struct scalar_contraction_recursion {
  static inline decltype(auto) contract(E1 const &u, E2 const &v) {
    return scalar_contraction_recursion<E1,E2,N-1>::contract(u,v)
         + u.template evaluate<N>() * v.template evaluate<N>();
  }
};
// Termination definition of recursion
template<typename E1, typename E2>
struct scalar_contraction_recursion<E1,E2,0> {
  static inline decltype(auto) contract(E1 const &u, E2 const &v) {
    return u.template evaluate<0>() * v.template evaluate<0>();
  }
};

//! Wrapper type for contraction with specialization for two vectors
//  General contraction expression result
template<typename E1, typename E2, size_t contracted_rank, size_t i1, size_t i2>
struct contractor_t {
  static inline decltype(auto) contract(E1 const &u, E2 const &v) {
    // CHECK: should we check here for any index order?
    return tensor_contraction_t<i1, i2, E1, E2>(u, v);
  }
};
// Scalar contraction result (for E1,2::rank == 1)
template<typename E1, typename E2, size_t i1, size_t i2>
struct contractor_t<E1,E2,0,i1,i2> {
  static_assert(utilities::is_reducible<i1,i2,E1,E2>::value, "Can only contract covariant with contravariant indices!");

  static inline decltype(auto) contract(E1 const &u, E2 const &v) {
    return scalar_contraction_recursion<E1,E2,E1::property_t::ndim-1>::contract(u,v);
  }
};

//! Contraction "operator" for two tensor expressions
//  Returns a tensor_contraction_t or a scalar (if E1,2::rank == 1)
template <size_t i1 = 0, size_t i2 = 0, typename E1, typename E2>
decltype(auto) inline contract(E1 const &u, E2 const &v) {
  return contractor_t<E1,
                      E2,
                      E1::property_t::rank + E2::property_t::rank - 2,
                      i1,
                      i2
                      >::contract(u, v);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////

// CHECK: can we inherit from general tensor_contraction_t above?
//! Expression template for raising and lowering indices
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
  using property_t = metric_contraction_property_t<i2, E1, E2>;
  static constexpr size_t i1 = 0;

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

  template <size_t index> inline typename property_t::data_t const evaluate() const {
    // TUPLE FREE TENSOR CONTRACTION, fix for problems with intel compiler
    constexpr size_t max_pow_E1 = E1::property_t::rank - 1;

    constexpr size_t ndim = E1::property_t::ndim;

    // CHECK: please add some comments
    constexpr size_t index_part_less_i2 =
        index % (utilities::static_pow<ndim, i2 + 1>::value);
    constexpr size_t index_part_less_i2_m1 =
        index % (utilities::static_pow<ndim, i2>::value);
    constexpr size_t index_part_metric =
        (index_part_less_i2 - index_part_less_i2_m1) /
        (utilities::static_pow<ndim, i2>::value);

    constexpr size_t stride_1 = index_part_metric * ndim;
    constexpr size_t stride_2 =
        index - (index_part_less_i2 - index_part_less_i2_m1);

    static_assert(
        stride_1 >= 0,
        "contraction: stride is less than zero, this shouldn't happen");
    static_assert(
        stride_2 >= 0,
        "contraction: stride is less than zero, this shouldn't happen");

    return recursive_contract<ndim - 1, stride_1, stride_2>::contract(_u, _v);
  }
};
}
#endif
