#ifndef TENSOR_FIELD_HH
#define TENSOR_FIELD_HH

namespace tensors {

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
    std::array<data_t*,ndof> const & ptr_array;

    //! Internal (pointer) index
    size_t const ptr_index;

  public:
    //! Contructor (called from a tensor field)
    tensor_field_expression_t(std::array<data_t*,ndof> const & arr, size_t const index)
        : ptr_array(arr), ptr_index(index) {}


    [[deprecated("Do not access the tensor expression via the [] operator, this is UNDEFINED!")]]
    inline decltype(auto) operator[](size_t i) const = delete;
/*
    inline decltype(auto) operator[](size_t i) const {
      return ptr_array[i][ptr_index];
    }
*/
    template<size_t index>
    inline data_t const & evaluate() const {
      return ptr_array[index][ptr_index];
    }

    template<typename E>
    inline void operator=(E const &e);
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
    std::array<data_t*,ndof> ptr_array;

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
    tensor_field_t(data_t * first_elem, TArgs... elem)
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
      // evaluate expression for every component
      // and set GFs at index i to that value
      setter_t<property_check::ndof-1,E>::set(i,e,ptr_array);

/*      // calling this is slightly slower
      set_component_impl(i,e,Indices{});
*/

/*      // slower than recursion for chained expressions, for whatever reason

      // Create a tensor, triggers evaluation of any expression and does type checks
      T t(e);
      for(size_t k = 0; k<ndof; ++k)
        ptr_array[k][i] = t[k];
*/
    }
};


template<typename T>
template<typename E>
inline void tensor_field_expression_t<T>::operator=(E const &e){
      // this only a check of compatibility of T and E
      using property_check = arithmetic_expression_property_t<T,E>;
      // evaluate expression for every component
      // and set GFs at index i to that value
      tensor_field_t<T>::template setter_t<property_check::ndof-1,E>::set(ptr_index,e,ptr_array);
};


}
#endif
