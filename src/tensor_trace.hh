#ifndef TENSOR_TRACE_HH
#define TENSOR_TRACE_HH

namespace tensors {

//! Expression template for generic tensor trace
template<size_t i1, size_t i2, typename E>
class tensor_trace_t : public tensor_expression_t<tensor_trace_t<i1,i2,E> > {
    // references to traced tensor
    E const& _u;

  public:
    // Trace changes the tensor type, thus a special property class is needed.
    // It gives all the relevant properties of a tensor resulting from a trace
    using property_t = trace_property_t<i1,i2,E>;

    tensor_trace_t(E const& u) : _u(u) {};

    [[deprecated("Do not access the tensor expression via the [] operator, this is UNDEFINED!")]]
    inline decltype(auto) operator[](size_t i) const = delete;

    //! Sum of traced components for a specific index computed by a template recursion
    template<size_t N, size_t stride>
    struct recursive_component_trace {
      template<typename A>
      static inline decltype(auto) trace(A const & _u) {
        return recursive_component_trace<(N-1),stride>::trace(_u)
             + _u.template evaluate<
                             stride
                           + N*utilities::static_pow<property_t::ndim,i1>::value
                           + N*utilities::static_pow<property_t::ndim,i2>::value
                           >();
      }
    };
    template<size_t stride>
    struct recursive_component_trace<0,stride> {
      template<typename A>
      static inline decltype(auto) trace(A const & _u) {
        return _u.template evaluate<stride>();
      }
    };

    template<size_t index>
    inline typename property_t::data_t const evaluate() const {
      // tuple free stride computation
      constexpr size_t ndim = E::property_t::ndim;

      // get unaffected part of index
      constexpr size_t stride_1 = index % utilities::static_pow<ndim,i1>::value;
      // get intermediate part of index which gets shifted by ndim
      constexpr size_t stride_2 = (index % utilities::static_pow<ndim,i2-1>::value) - stride_1;
      // get trailing part of index which gets shifted by ndim^2
      constexpr size_t stride_3 = index - stride_1 - stride_2;

      // compute the actual stride
      constexpr size_t stride = stride_1 + stride_2*ndim + stride_3*ndim*ndim;

      static_assert(stride >= 0,
                    "trace: stride is less than zero, this shouldn't happen");

      // Compute sum over traced indices for index in a template recursion
      return recursive_component_trace<property_t::ndim-1,stride>::trace(_u);
    }
};

//! Helper structure to compute trace of rank 2 tensor by a template recursion
template <typename E, size_t N>
struct scalar_trace_recursion {
  static inline decltype(auto) trace(E const &u) {
    return scalar_trace_recursion<E,N-1>::trace(u)
         + u.template evaluate<E::property_t
                                ::symmetry_t
                                ::template compressed_index<N,N>::value>();
  }
};
// Termination definition of recursion
template<typename E>
struct scalar_trace_recursion<E,0> {
  static inline decltype(auto) trace(E const &u) {
    return u.template evaluate<0>();
  }
};

//! Wrapper type for trace with specialization for rank 2 tensor
//  General trace expression result
template<typename E, size_t rank, size_t i1, size_t i2>
struct tracer_t {
  static inline decltype(auto) trace(E const &u) {
    // automatically change indices if they are not in ascending order
    return tensor_trace_t<(i1 < i2) ? i1 : i2,(i1 > i2) ? i1 : i2,E>(u);
  }
};
// Scalar trace result (for E::rank == 2)
template<typename E, size_t i1, size_t i2>
struct tracer_t<E,2,i1,i2> {
  static_assert(is_reducible<i1,i2,E,E>::value, "Can only contract covariant with contravariant indices!");

  static inline decltype(auto) trace(E const &u) {
    return scalar_trace_recursion<E,E::property_t::ndim-1>::trace(u);
  }
};

//! Trace "operator" for a tensor expression
//  Returns a tensor_trace_t or a scalar (if E::rank == 2)
template<size_t i1 = 0, size_t i2 = 1, typename E>
decltype(auto)
inline trace(E const &u) {
  return tracer_t<E,E::property_t::rank,i1,i2>::trace(u);
}

}
#endif
