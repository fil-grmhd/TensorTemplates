#ifndef TENSORS_TENSOR_FIELD_HH
#define TENSORS_TENSOR_FIELD_HH

namespace tensors {

#if !defined(TENSORS_VECTORIZED) || !defined(TENSORS_AUTOVEC)
//! Template for generic tensor field expression
//! This represents a tensor at a specific grid position
template<typename T, typename ptr_array_t>
class tensor_field_expression_t : public tensor_expression_t<tensor_field_expression_t<T,ptr_array_t>> {
  public:
    // Get properties of underlying tensor type
    using property_t = typename T::property_t;
    using data_t = typename property_t::data_t;
    static constexpr size_t ndof = property_t::ndof;

  protected:
    //! Reference to grid pointer array of underlying tensor field
    ptr_array_t const & ptr_array;

    //! Internal (grid pointer) index
    size_t const grid_index;

    // Template recursion to set components, fastest for chained expressions
    template<size_t N, typename E>
    struct setter_t {
      static inline __attribute__ ((always_inline)) void set(size_t const i, E const& e, ptr_array_t const & arr) {
        // N goes from ndof to zero of this tensor type
        // one has to cast to generic index before one can call evaluate
        constexpr size_t gen_index = E::property_t::symmetry_t::template index_to_generic<N>::value;

        arr[N][i] = e.template evaluate<gen_index>();
        setter_t<N-1,E>::set(i,e,arr);
      }
    };
    template<typename E>
    struct setter_t<0,E> {
      static inline __attribute__ ((always_inline)) void set(size_t const i, E const& e, ptr_array_t const & arr) {
        arr[0][i] = e.template evaluate<0>();
      }
    };

  public:
    //! Contructor (called from a tensor field)
    tensor_field_expression_t(ptr_array_t const & arr, size_t const grid_index_)
        : ptr_array(arr), grid_index(grid_index_) {}


    [[deprecated("Do not access the tensor expression via the [] operator, this is UNDEFINED!")]]
    inline __attribute__ ((always_inline)) decltype(auto) operator[](size_t i) const = delete;

    template<size_t index>
    inline __attribute__ ((always_inline)) decltype(auto) evaluate() const {
      constexpr size_t converted_index = property_t::symmetry_t::template index_from_generic<index>::value;
      return ptr_array[converted_index][grid_index];
    }

    //! Returns a partial derivative of this tensor field
    template<typename fd_t>
    inline __attribute__ ((always_inline)) decltype(auto) finite_diff(fd_t const & fd) const {
      return tensor_partial_derivative_t<T,ptr_array_t,fd_t>(grid_index,ptr_array,fd);
    }

    //! Returns the upwind derivative of this tensor field
    template<typename fd_u_t,typename fd_d_t,typename beta_t>
    inline __attribute__ ((always_inline)) decltype(auto) upwind_finite_diff(fd_u_t const & fdu, fd_d_t const & fdd, beta_t const & beta) const {
      return tensor_advective_derivative_t<T,ptr_array_t,beta_t,fd_u_t,fd_d_t>(grid_index,ptr_array, beta, fdu, fdd);
    }

    template<typename E>
    inline __attribute__ ((always_inline)) void operator=(E const &e) {
      // this only a check of compatibility of T and E
      using property_check = arithmetic_expression_property_t<T,E>;
      // evaluate expression for every component
      // and set GFs at index i to that value

      static_assert(std::is_same<typename T::property_t::symmetry_t, typename E::property_t::symmetry_t>::value,
                    "Please make sure that tensor expression and tensor field have the same symmetry.");
      setter_t<E::property_t::ndof-1,E>::set(grid_index,e,ptr_array);
    }
};

//! Template for generic tensor field
//! A tensor field is not a tensor expression,
//! but delivers tensor field expressions at different grid points.
//  The tensor field expression is itself a template parameter,
//  which implements the load/store operations.
template<typename T>
class tensor_field_t {
  public:
    // Get properties of underlying tensor type
    using property_t = typename T::property_t;
    using data_t = typename property_t::data_t;
    static constexpr size_t ndof = property_t::ndof;

  protected:
    //! Storage for ndof grid pointers
    const std::array<data_t * __restrict__ const,ndof> ptr_array;

  public:
    //! Constructor from grid pointer parameters
    template <typename... TArgs>
    tensor_field_t(data_t * __restrict__ const first_elem, TArgs... elem)
        : ptr_array({first_elem, elem...}) {
      static_assert(sizeof...(TArgs)==ndof-1, "You need to specify exactly ndof arguments!");
    };

    //! Returns a tensor field expression at (grid pointer) index i
    inline __attribute__ ((always_inline)) decltype(auto) operator[](size_t const i) const {
      return tensor_field_expression_t<T,decltype(ptr_array)>(ptr_array,i);
    }
};

#endif

#ifdef TENSORS_VECTORIZED
//! Template for a generic vectorized tensor field expression
//! This represents a vector register of tensors at Vc::Vector<T>::Size successive grid points
template<typename T, typename ptr_array_t>
class tensor_field_expression_vt : public tensor_expression_t<tensor_field_expression_vt<T,ptr_array_t>> {
  public:
    // Get properties of underlying tensor type
    using property_t = typename T::property_t;
    // The data type is in this case a vector register type
    using vec_t = typename property_t::data_t;
    // The actual data type
    using data_t = typename vec_t::EntryType;

    static constexpr size_t ndof = property_t::ndof;

  protected:
    //! Reference to grid pointer array of underlying tensor field
    ptr_array_t const & ptr_array;

    //! Internal (grid pointer) index
    size_t const grid_index;

    // Template recursion to set components, fastest for chained expressions
    template<size_t N, typename E>
    struct setter_t {
      static inline __attribute__ ((always_inline)) void set(size_t const i, E const& e, ptr_array_t const & arr) {
        // N goes from ndof to zero of this tensor type
        // one has to cast to generic index before one can call evaluate
        constexpr size_t gen_index = E::property_t::symmetry_t::template index_to_generic<N>::value;

        // gets a vector register and let it store Vc::Vector<T>::Size elements to memory starting at i
        (e.template evaluate<gen_index>()).store(&arr[N][i]);
        setter_t<N-1,E>::set(i,e,arr);
      }
    };
    template<typename E>
    struct setter_t<0,E> {
      static inline __attribute__ ((always_inline)) void set(size_t const i, E const& e, ptr_array_t const & arr) {
        (e.template evaluate<0>()).store(&arr[0][i]);
      }
    };

  public:
    //! Contructor (called from a tensor field)
    tensor_field_expression_vt(ptr_array_t const & arr, size_t const index)
        : ptr_array(arr), grid_index(index) {}


    [[deprecated("Do not access the tensor expression via the [] operator, this is UNDEFINED!")]]
    inline __attribute__ ((always_inline)) decltype(auto) operator[](size_t i) const = delete;

    template<size_t index>
    inline __attribute__ ((always_inline)) decltype(auto) evaluate() const {
      constexpr size_t converted_index = property_t::symmetry_t::template index_from_generic<index>::value;

      // reads Vc::Vector<data_t>::Size values from grid_index on into vector register of data_t
      vec_t vec_register(&(ptr_array[converted_index][grid_index]));

      return vec_register;
    }

    //! Returns a partial derivative of this tensor field
    template<typename fd_t>
    inline __attribute__ ((always_inline)) decltype(auto) finite_diff(fd_t const & fd) const {
      return tensor_partial_derivative_t<T,ptr_array_t,fd_t>(grid_index,ptr_array,fd);
    }

    //! Returns the upwind derivative of this tensor field
    template<typename fd_u_t,typename fd_d_t,typename beta_t>
    inline __attribute__ ((always_inline)) decltype(auto) upwind_finite_diff(fd_u_t const & fdu, fd_d_t const & fdd, beta_t const & beta) const {
      return tensor_advective_derivative_t<T,ptr_array_t,beta_t,fd_u_t,fd_d_t>(grid_index,ptr_array, beta, fdu, fdd);
    }


    template<typename E>
    inline __attribute__ ((always_inline)) void operator=(E const &e) {
      // this only a check of compatibility of T and E
      using property_check = arithmetic_expression_property_t<T,E>;
      // evaluate expression for every component
      // and set GFs at index i to that value

      static_assert(std::is_same<typename T::property_t::symmetry_t, typename E::property_t::symmetry_t>::value,
                    "Please make sure that tensor expression and tensor field have the same symmetry.");


      setter_t<E::property_t::ndof-1,E>::set(grid_index,e,ptr_array);
    }
};

//! Template for generic vectorized tensor field
//! A tensor field is not a tensor expression,
//! but delivers tensor field expressions at different grid points.
template<typename T>
class tensor_field_vt {
  public:
    // Get properties of underlying tensor type
    using property_t = typename T::property_t;
    // The data type is in this case a vector register type
    using vec_t = typename property_t::data_t;
    // The actual data type
    using data_t = typename vec_t::EntryType;
    static constexpr size_t ndof = property_t::ndof;

  protected:
    //! Storage for ndof grid pointers
    const std::array<data_t * __restrict__ const,ndof> ptr_array;

  public:
    //! Constructor from grid pointer parameters
    template <typename... TArgs>
    tensor_field_vt(data_t * __restrict__ const first_elem, TArgs... elem)
        : ptr_array({first_elem, elem...}) {
      static_assert(sizeof...(TArgs)==ndof-1, "You need to specify exactly ndof arguments!");
    };

    //! Returns a tensor field expression at (grid pointer) index i
    inline __attribute__ ((always_inline)) decltype(auto) operator[](size_t const i) const {
      return tensor_field_expression_vt<T,decltype(ptr_array)>(ptr_array,i);
    }
};

#ifdef TENSORS_AUTOVEC
template<typename T>
using tensor_field_t = tensor_field_vt<T>;
#endif

#endif

} // namespace tensors
#endif
