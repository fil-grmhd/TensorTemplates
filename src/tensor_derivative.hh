#ifndef TENSORS_DERIVATIVE_HH
#define TENSORS_DERIVATIVE_HH

namespace tensors {

//! Expression template for a partial derivative of a tensor
template <typename E, typename array_t, typename fd_t>
class tensor_partial_derivative_t
    : public tensor_expression_t<tensor_partial_derivative_t<E,array_t,fd_t>> {

protected:
  //! Object defining a finite difference operation on a pointer
  //  Must implement a diff<direction>(pointer,grid_index) member function
  //  returning the finite difference at the point represented by index
  fd_t const & fd;

  //! Array to the raw tensor field component pointers
  array_t const & ptr_array;
  //! Index representing a position on the grid
  size_t const grid_index;

public:
  // A partial derivative adds a new lower index (to the right),
  // thus a special property class is needed.

  // adds a lower index
  using index_t = decltype(std::tuple_cat(std::declval<typename E::property_t::index_t>(),
                                          std::declval<std::tuple<lower_t>>()));

  static constexpr size_t ndim = E::property_t::ndim;
  // adds an index
  static constexpr size_t rank = E::property_t::rank + 1;

  // no symmetry reconstruction here, please cast expression to given symmetry
  using property_t = general_tensor_property_t<
                       general_tensor_t<
                         typename E::property_t::data_t,
                         typename E::property_t::frame_t,
                         generic_symmetry_t<ndim,rank>,
                         E::property_t::rank + 1,
                         index_t,
                         E::property_t::ndim
                       >
                     >;


  tensor_partial_derivative_t(size_t const grid_index_, array_t const & ptr_array_, fd_t const & fd_)
                            : grid_index(grid_index_), ptr_array(ptr_array_), fd(fd_) {};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline __attribute__ ((always_inline)) decltype(auto)
  operator[](size_t i) const = delete;


  template <size_t index> inline __attribute__ ((always_inline)) decltype(auto) evaluate() const {
    // get index without partial derivative index part
    constexpr size_t tensor_gen_index = index % utilities::static_pow<ndim,rank-1>::value;
    // transform to underlying symmetry index (since array has only ndof elements)
    constexpr size_t tensor_sym_index = E::property_t::symmetry_t
                                          ::template index_from_generic<tensor_gen_index>::value;
    // get last index, which is the derivative index and defines the direction of the fd
    constexpr size_t deriv_index = property_t::symmetry_t::template uncompress_index<rank-1,index>::value;

    // compute fd at point grid_index of tensor_sym_index' component
    return fd.template diff<deriv_index>(ptr_array[tensor_sym_index],grid_index);
  }
};




//! Expression template for an advective derivative of a tensor, e.g. beta^i \partial_i tensor
template <typename E, typename array_t, typename beta_t ,typename fd_u_t, typename fd_d_t>
class tensor_advective_derivative_t
    : public tensor_expression_t<tensor_advective_derivative_t<E,array_t, beta_t ,fd_u_t, fd_d_t>> {

protected:
  //! Objects defining the (up/down) finite difference operation on a pointer
  //  Must implement a diff<direction>(pointer,grid_index) member function
  //  returning the finite difference at the point represented by index
  fd_u_t const & fdu;
  fd_d_t const & fdd;

  //! Array to the raw tensor field component pointers
  array_t const & ptr_array;
  //! Array to the beta vector field component pointers
  beta_t const & beta;
  //! Index representing a position on the grid
  size_t const grid_index;

public:
  // A partial derivative adds a new lower index (to the right),
  // thus a special property class is needed.

  // adds a lower index
  using index_t = typename E::property_t::index_t;

  static constexpr size_t ndim = E::property_t::ndim;
  // adds an index
  static constexpr size_t rank = E::property_t::rank;

  static_assert(beta_t::property_t::ndim == E::property_t::ndim,
                "Dimensions must match!");

  static_assert(beta_t::property_t::rank == 1,
                "Characteristic vector needs to have rank 1!");

  static_assert(std::is_same<
                  typename std::tuple_element<0, typename beta_t::property_t::index_t>::type,
                  lower_t
                >::value,
      	 	      "Characteristic vector needs to be contravariant!");

  // no symmetry reconstruction here, please cast expression to given symmetry
  // CHECK: construct the right property_t
  using property_t = typename E::property_t;

  tensor_advective_derivative_t(size_t const grid_index_, array_t const & ptr_array_, beta_t const& beta_,
                   			        fd_u_t const & fdu_, fd_d_t const & fdd_)
    : grid_index(grid_index_), ptr_array(ptr_array_), beta(beta_), fdu(fdu_), fdd(fdd_) {};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline __attribute__ ((always_inline)) decltype(auto)
  operator[](size_t i) const = delete;


  template<size_t tensor_sym_index, size_t index>
  struct beta_dE{
    public:
      inline __attribute__ ((always_inline)) decltype(auto) value(beta_t const & beta, fd_u_t const & fdu, fd_d_t const & fdd,
	                                                                array_t const & ptr_array, size_t const grid_index) {
       return beta_dE<tensor_sym_index,index-1>::value(beta,fdu,fdd,ptr_array,grid_index)
            + beta.template evaluate<index>() * ((beta.template evaluate<index>() > 0)
            ? fdu.template diff<index>(ptr_array[tensor_sym_index],grid_index)
  	        : fdd.template diff<index>(ptr_array[tensor_sym_index],grid_index));
    }
  };

  template<size_t tensor_sym_index>
  struct beta_dE<tensor_sym_index,0>{
    public:
      inline __attribute__ ((always_inline)) decltype(auto) value(beta_t const & beta, fd_u_t const & fdu, fd_d_t const & fdd,
	                                                                array_t const & ptr_array, size_t const grid_index) {
       return beta.template evaluate<0>() * ((beta.template evaluate<0>() > 0)
            ? fdu.template diff<0>(ptr_array[tensor_sym_index],grid_index)
  	        : fdd.template diff<0>(ptr_array[tensor_sym_index],grid_index) );
    }
  };


  template <size_t index> inline __attribute__ ((always_inline)) decltype(auto) evaluate() const {
    // transform to underlying symmetry index (since array has only ndof elements)
    constexpr size_t tensor_sym_index = E::property_t::symmetry_t
                                          ::template index_from_generic<index>::value;
    // compute fd at point grid_index of tensor_sym_index' component
    return beta_dE<tensor_sym_index,ndim-1>::value(beta,fdu,fdd,ptr_array,grid_index);
  }
};

}
#endif
