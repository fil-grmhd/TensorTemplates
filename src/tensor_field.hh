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

        arr[N][i] = e.template evaluate<gen_index>();
        setter_t<N-1,E>::set(i,e,arr);
      }
    };
    template<typename E>
    struct setter_t<0,E> {
      static inline void set(size_t const i, E const& e, decltype(ptr_array) const & arr) {
        arr[0][i] = e.template evaluate<0>();
      }
    };

  public:
    //! Contructor (called from a tensor field)
    tensor_field_expression_t(decltype(ptr_array) const & arr, size_t const index)
        : ptr_array(arr), ptr_index(index) {}


    [[deprecated("Do not access the tensor expression via the [] operator, this is UNDEFINED!")]]
    inline decltype(auto) operator[](size_t i) const = delete;

    template<size_t index>
    inline decltype(auto) evaluate() const {
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
      setter_t<E::property_t::ndof-1,E>::set(ptr_index,e,ptr_array);
    }
};

//! Template for generic tensor field
//! A tensor field is not a tensor expression,
//! but delivers tensor field expressions at different grid points.
//  The tensor field expression is itself a template parameter,
//  which implements the load/store operations.
template<typename T, typename tf_expression_t = tensor_field_expression_t<T>>
class tensor_field_t {
  public:
    // Get properties of underlying tensor type
    using property_t = typename T::property_t;
    using data_t = typename property_t::data_t;
    static constexpr size_t ndof = property_t::ndof;

  private:
    //! Storage for ndof pointers
    const std::array<data_t * __restrict__ const,ndof> ptr_array;

    // Template recursion to compute derivatives of components
    template<size_t order, size_t N, typename dT, typename fd_t>
    struct fds_gen_t {
      static inline void set(size_t const i, dT& dt, decltype(ptr_array) const & arr, fd_t const & fd) {
        // N goes from ndof to zero of this tensor type
        // one has to cast to generic index before one can call evaluate
        constexpr size_t gen_index = dT::property_t::symmetry_t::template index_to_generic<N>::value;
        constexpr size_t rank = dT::property_t::rank;
        constexpr size_t ndim = dT::property_t::ndim;

        // Remove híghest power of ndim -> remove last index -> T index
        constexpr size_t t_gen_index = gen_index % utilities::static_pow<ndim,rank-1>::value;
        // Transform to sym index of T
        constexpr size_t t_sym_index = T::property_t::symmetry_t::template index_from_generic<t_gen_index>::value;
        // Get last index (derivative index)
        constexpr size_t deriv_index = dT::property_t::symmetry_t::template uncompress_index<rank-1,gen_index>::value;

        dt.template compressed_c<gen_index>() = fd.template diff<deriv_index,order>(arr[t_sym_index],i);

        fds_gen_t<order,N-1,dT,fd_t>::set(i,dt,arr,fd);
      }
    };
    template<size_t order, typename dT, typename fd_t>
    struct fds_gen_t<order,0,dT,fd_t> {
      static inline void set(size_t const i, dT& dt, decltype(ptr_array) const & arr, fd_t const & fd) {
        dt.template compressed_c<0>() = fd.template diff<0,order>(arr[0],i);
      }
    };

  public:
    //! Constructor from pointer parameters
    template <typename... TArgs>
    tensor_field_t(data_t * __restrict__ const first_elem, TArgs... elem)
        : ptr_array({first_elem, elem...}) {
      static_assert(sizeof...(TArgs)==ndof-1, "You need to specify exactly ndof arguments!");
    };

    //! Returns a tensor field expression at (pointer) index i
    inline decltype(auto) operator[](size_t const i) const {
      return tf_expression_t(ptr_array,i);
    }

    //! Returns a partial derivative of this tensor field at (pointer) index i
    template<size_t order, typename fd_t>
    inline decltype(auto) diff(size_t const i, fd_t const & fd) const {
      // derivative adds a new (lower) index to this tensor
      using new_index_t = decltype(std::tuple_cat(std::declval<T::property_t::index_t>(),
                                                  std::declval<std::tuple<lower_t>>()));

      // partial derivative of tensor
      constexpr size_t ndim = T::property_t::ndim;
      constexpr size_t rank = T::property_t::rank + 1;
      using dT = general_tensor_t<typename T::property_t::data_t,
                   typename T::property_t::frame_t,
// doesn't work, one would need to changed rank in symmetry_t
//                   typename T::property_t::symmetry_t,
                   generic_symmetry_t<ndim,rank>,
                   rank,
                   new_index_t,
                   ndim>;
      dT dt;
      // fill with finite differences
      fds_gen_t<order,dT::property_t::ndof-1,dT,fd_t>::set(i,dt,ptr_array,fd);

      return dt;
    }
};

}
#endif
