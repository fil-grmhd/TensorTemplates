#ifndef TENSOR_TRACE_HH
#define TENSOR_TRACE_HH

namespace tensors {

//! Expression template for generic tensor trace
template<size_t i1, size_t i2, typename E>
class tensor_trace_t : public tensor_expression_t<tensor_trace_t<i1,i2,E> > {
    // references to traced tensor
    E const& _u;

  public:
    // check if someone tries to trace an index with itself
    static_assert(i1 != i2, "You cannot trace an index with itself, i.e. make sure that i1 != i2");
    // this can only trigger if tensor_trace_t is created manually, see below in trace "operator"
    static_assert(i1 < i2, "Please make sure that traced indices are in ascending order, i.e. i1 < i2");

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
    inline decltype(auto) evaluate() const {
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

/* debugging code
      constexpr auto ids = uncompress_index<ndim,property_t::rank,index>();

      std::cout << "Computing traced component ";
      utilities::print_tuple(ids);
      std::cout << std::endl;
      std::cout << "stride_1=" << stride_1 << ", stride_2=" << stride_2 << ", stride_3=" << stride_3<< ", stride=" << stride << std::endl;
*/

      // Compute sum over traced indices for index in a template recursion
      return recursive_component_trace<property_t::ndim-1,stride>::trace(_u);
    }
};

//! Trace "operator" for a tensor expression
template<size_t i1, size_t i2, typename E>
decltype(auto)
inline trace(E const &u) {
  // automatically change indices if they are not in ascending order
  return tensor_trace_t<(i1 < i2) ? i1 : i2,(i1 > i2) ? i1 : i2,E>(u);
}

}
#endif
