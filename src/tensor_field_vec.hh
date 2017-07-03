#ifndef TENSOR_FIELD_VEC_HH
#define TENSOR_FIELD_VEC_HH

namespace tensors {

//! Template for generic vectorized tensor field expression
//! This represents a vector of tensors at Vc::Vector<T>::Size grid points
template<typename T>
class tensor_field_expression_vt : public tensor_expression_t<tensor_field_expression_vt<T>> {
  public:
    // Get properties of underlying tensor type
    using property_t = typename T::property_t;
    using data_t = typename property_t::data_t;
    static constexpr size_t ndof = property_t::ndof;

  private:
    //! Reference to pointer array of underlying tensor field
    std::array<data_t * __restrict__ const,ndof> const & ptr_array;

    //! Internal (pointer) index
    size_t const ptr_index;

    // Template recursion to set components, fastest for chained expressions
    template<size_t N, typename E>
    struct setter_t {
      static inline void set(size_t const i, E const& e, decltype(ptr_array) const & arr) {
        // N goes from ndof to zero of this tensor type
        // one has to cast to generic index before one can call evaluate
        constexpr size_t gen_index = E::property_t::symmetry_t::template index_to_generic<N>::value;

        // gets a vector register and let it store it Vc::Vector<T>::Size elements to memory starting at i
        (e.template evaluate<gen_index>()).store(&arr[N][i]);
        setter_t<N-1,E>::set(i,e,arr);
      }
    };
    template<typename E>
    struct setter_t<0,E> {
      static inline void set(size_t const i, E const& e, decltype(ptr_array) const & arr) {
        (e.template evaluate<0>()).store(&arr[0][i]);
      }
    };

  public:
    //! Contructor (called from a tensor field)
    tensor_field_expression_vt(decltype(ptr_array) const & arr, size_t const index)
        : ptr_array(arr), ptr_index(index) {}


    [[deprecated("Do not access the tensor expression via the [] operator, this is UNDEFINED!")]]
    inline decltype(auto) operator[](size_t i) const = delete;

    template<size_t index>
    inline data_t const & evaluate() const {
      constexpr size_t converted_index = property_t::symmetry_t::template index_from_generic<index>::value;

      // reads Vc::Vector<T>::Size values from ptr_index on into vector register
      Vc::Vector<T> vec_register(&(ptr_array[converted_index][ptr_index]));

      return vec_register;
    }

    template<typename E>
    inline void operator=(E const &e) {
      // this only a check of compatibility of T and E
      using property_check = arithmetic_expression_property_t<T,E>;
      // evaluate expression for every component
      // and set GFs at index i to that value

      static_assert(std::is_same<typename T::property_t::symmetry_t, typename E::property_t::symmetry_t>::value,
                    "Please make sure that tensor expression and tensor field have the same symmetry.");


      setter_t<E::property_t::ndof-1,E>::set(ptr_index,e,ptr_array);
    }
};

//! Template for generic tensor field
//! A tensor field is not a tensor expression,
//! but delivers tensor field expressions at different grid points.
template<typename T>
using tensor_field_vt = tensor_field_t<T, tensor_field_expression_vt<T>>;

}
#endif
