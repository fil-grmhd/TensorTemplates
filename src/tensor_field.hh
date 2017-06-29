#ifndef TENSOR_FIELD_HH
#define TENSOR_FIELD_HH

namespace tensors {

// forward declaration
template<typename T>
class tensor_field_t;


//! Template for generic tensor field expression
//! This represents a tensor at a specific grid position
template<typename T>
class tensor_field_expression_t : public tensor_expression_t<tensor_field_expression_t<T>> {
  public:
    // Get properties of underlying tensor type
    using property_t = typename T::property_t;
    using data_t = typename property_t::data_t;
    static constexpr size_t ndof = property_t::ndof;

  private:
    //! Reference to pointer array of underlying tensor field
//    std::array<data_t * const,ndof> const & ptr_array;
    std::array<data_t * __restrict__ const,ndof> const & ptr_array;

    //! Internal (pointer) index
    size_t const ptr_index;

  public:
    //! Contructor (called from a tensor field)
    tensor_field_expression_t(decltype(ptr_array) const & arr, size_t const index)
        : ptr_array(arr), ptr_index(index) {}


    [[deprecated("Do not access the tensor expression via the [] operator, this is UNDEFINED!")]]
    inline decltype(auto) operator[](size_t i) const = delete;

    template<size_t index>
    inline data_t const & evaluate() const {
      constexpr size_t converted_index = property_t::symmetry_t::template index_from_generic<index>::value;
      return ptr_array[converted_index][ptr_index];
    }

    template<typename E>
    inline void operator=(E const &e) {
      // this only a check of compatibility of T and E
      using property_check = arithmetic_expression_property_t<T,E>;
      // evaluate expression for every component
      // and set GFs at index i to that value

      static_assert(std::is_same<typename T::property_t::symmetry_t, typename E::property_t::symmetry_t>::value,
                    "Please make sure that tensor expression and tensor field have the same symmetry.");
      tensor_field_t<T>::template setter_t<E::property_t::ndof-1,E>::set(ptr_index,e,ptr_array);
    }
};

//! Template for generic tensor field
//! A tensor field is not a tensor expression,
//! but delivers tensor field expressions at different grid points.
template<typename T>
class tensor_field_t {
  public:
    // Get properties of underlying tensor type
    using property_t = typename T::property_t;
    using data_t = typename property_t::data_t;
    static constexpr size_t ndof = property_t::ndof;

  private:
    //! Storage for ndof pointers
//    const std::array<data_t * const,ndof> ptr_array;
    const std::array<data_t * __restrict__ const,ndof> ptr_array;

  public:
    // Template recursion to set components, fastest for chained expressions
    template<size_t N, typename E>
    struct setter_t {
      static inline void set(size_t const i, E const& e, decltype(ptr_array) const & arr) {
        arr[N][i] = e.template evaluate<N>();
        setter_t<N-1,E>::set(i,e,arr);
      }
    };
    template<typename E>
    struct setter_t<0,E> {
      static inline void set(size_t const i, E const& e, decltype(ptr_array) const & arr) {
        arr[0][i] = e.template evaluate<0>();
      }
    };
/*
    // works but shouldn't be used (much slower)
    template<size_t... I, typename E>
    inline void set_component_impl(size_t const i, E const &e, std::index_sequence<I...>) {
      // dirty trick to get unpacking and assignment working
      // temporary array should be compiled away
      // see https://stackoverflow.com/questions/25680461/variadic-template-pack-expansion
      using expander = int[];
      (void) expander { 0, ((ptr_array[I][i] = e.template evaluate<I>()), 0)... };
    }
*/
    //! Constructor from pointer parameters
    template <typename... TArgs>
    tensor_field_t(data_t * __restrict__ const first_elem, TArgs... elem)
//    tensor_field_t(data_t * const first_elem, TArgs... elem)
        : ptr_array({first_elem, elem...}) {
      static_assert(sizeof...(TArgs)==ndof-1, "You need to specify exactly ndof arguments!");
    };

    //! Returns a tensor field expression at (pointer) index i
    inline decltype(auto) operator[](size_t const i) const {
      return tensor_field_expression_t<T>(ptr_array,i);
    }

    //! Set the tensor field components at (pointer) index i
    template<typename E>
    inline void set_components(size_t const i, E const &e) {
      // this is only a check of compatibility of T and E
      using property_check = arithmetic_expression_property_t<T,E>;
      // evaluate expression for every component ndof
      // and set GFs at index i to that value

      static_assert(std::is_same<typename T::property_t::symmetry_t, typename E::property_t::symmetry_t>::value,
                    "Please make sure that tensor expression and tensor field have the same symmetry.");
      setter_t<E::property_t::ndof-1,E>::set(i,e,ptr_array);
    }
};

}
#endif
