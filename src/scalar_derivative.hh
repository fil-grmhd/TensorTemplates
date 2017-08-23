#ifndef TENSORS_SCALAR_DERIVATIVE_HH
#define TENSORS_SCALAR_DERIVATIVE_HH

namespace tensors {

//! Expression template for a partial derivative of a scalar
//  T is the data_t
template <typename T, typename ptr_t, typename fd_t>
class scalar_partial_derivative_t
    : public tensor_expression_t<scalar_partial_derivative_t<T,ptr_t,fd_t>> {

protected:
  //! Object defining a finite difference operation on a pointer
  //  Must implement a diff<direction>(pointer,grid_index) member function
  //  returning the finite difference at the point represented by index
  fd_t const & fd;

  //! Array to the raw tensor field component pointers
  ptr_t const grid_ptr;
  //! Index representing a position on the grid
  size_t const grid_index;

  
  template<size_t d=1, bool _ =true>
  struct get_index_t{
      using index_t = decltype(std::tuple_cat(
	                     std::declval<typename get_index_t<d-1>::index_t>(),
                                          std::declval<std::tuple<lower_t>>()));
  };
  template<bool _>
  struct get_index_t<0,_>{
      using index_t = std::tuple<>;
  };


public:
  using data_t = T;

  // A partial derivative adds a new lower index (to the right),
  // thus a special property class is needed.

  // adds a lower index
//FIXME!!
  using index_t = typename get_index_t<1>::index_t;

  // up to now, only considering patial FDs here
  static constexpr size_t ndim = 3;
  // adds an index
  static constexpr size_t rank = 1; //FIXME //fd_t::d;

  // no symmetry reconstruction here, please cast expression to given symmetry
  using property_t = general_tensor_property_t<
                       general_tensor_t<
                         data_t,
                         any_frame_t,
                         generic_symmetry_t<ndim,rank>,
                         rank,
                         index_t,
                         ndim
                       >
                     >;


  scalar_partial_derivative_t(size_t const grid_index_, ptr_t const grid_ptr_, fd_t const & fd_)
                            : grid_index(grid_index_), grid_ptr(grid_ptr_), fd(fd_) {};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline __attribute__ ((always_inline)) decltype(auto)
  operator[](size_t i) const = delete;


  template <size_t index> inline __attribute__ ((always_inline)) decltype(auto) evaluate() const {
    // compute fd at point grid_index in index direction
    return fd.template diff<index>(grid_ptr,grid_index);
  }
};




//! Expression template for an advective derivative of a scalar, e.g. beta^i \partial_i scalar
template <typename T, typename ptr_t, typename beta_t, typename fd_u_t, typename fd_d_t>
class scalar_advective_derivative_t {

protected:
  //! Objects defining the (up/down) finite difference operation on a pointer
  //  Must implement a diff<direction>(pointer,grid_index) member function
  //  returning the finite difference at the point represented by index
  fd_u_t const & fdu;
  fd_d_t const & fdd;

  //! Array to the raw tensor field component pointers
  ptr_t const  grid_ptr;
  //! Array to the beta vector field component pointers
  beta_t const & beta;
  //! Index representing a position on the grid
  size_t const grid_index;

public:
  // A partial derivative adds a new lower index (to the right),
  // thus a special property class is needed.


  // up to now, only considering patial FDs here
  // CHECK: could we generalize to include time derivatives?
  static constexpr size_t ndim = 3;

  static_assert(beta_t::property_t::rank == 1,
      	 	      "Characteristic vector needs to have rank 1!");

  static_assert(std::is_same<
                  typename std::tuple_element<0, typename beta_t::property_t::index_t>::type,
                  upper_t
                >::value,
      	 	      "Characteristic vector needs to be contravariant!");

    struct property_t {
      using this_tensor_t = T;
    };


  // no symmetry reconstruction here, please cast expression to given symmetry
  // no symmetry reconstruction here, please cast expression to given symmetry

  scalar_advective_derivative_t(size_t const grid_index_, ptr_t const grid_ptr_, beta_t const & beta_,
     			                      fd_u_t const & fdu_, fd_d_t const & fdd_)
    : grid_index(grid_index_), grid_ptr(grid_ptr_), beta(beta_), fdu(fdu_),fdd(fdd_) {};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline __attribute__ ((always_inline)) decltype(auto)
  operator[](size_t i) const = delete;

  // Template recursion to compute the contracted advective derivative
  // The bool is only a placeholder, since explicit specialization is forbidden in class scope
  template<bool _, size_t index>
  struct beta_dE {
    public:
      static inline __attribute__ ((always_inline)) decltype(auto) value(beta_t const & beta, fd_u_t const & fdu, fd_d_t const & fdd,
      	                                                          ptr_t grid_ptr, size_t const grid_index) {
#if !defined(TENSORS_VECTORIZED) || !defined(TENSORS_AUTOVEC)
       return beta_dE<_,index-1>::value(beta,fdu,fdd,grid_ptr,grid_index)
            + beta.template evaluate<index>()
            * ((beta.template evaluate<index>() > 0)
            ? fdu.template diff<index>(grid_ptr,grid_index)
      	    : fdd.template diff<index>(grid_ptr,grid_index));
#else
       T tmp;
       where(beta.template evaluate<index>() > 0) | tmp = fdu.template diff<index>(grid_ptr,grid_index);
       where(beta.template evaluate<index>() < 0) | tmp = fdd.template diff<index>(grid_ptr,grid_index);
       return beta_dE<_,index-1>::value(beta,fdu,fdd,grid_ptr,grid_index)
	      +beta.template evaluate<index>() *tmp;
#endif
    }
  };
  template<bool _>
  struct beta_dE<_,0> {
    public:
      static inline __attribute__ ((always_inline)) decltype(auto) value(beta_t const & beta , fd_u_t const & fdu, fd_d_t const & fdd,
	                                                                ptr_t const grid_ptr, size_t const grid_index){
#if !defined(TENSORS_VECTORIZED) || !defined(TENSORS_AUTOVEC)
       return beta.template evaluate<0>()
            * ((beta.template evaluate<0>() > 0)
            ? fdu.template diff<0>(grid_ptr,grid_index)
  	        : fdd.template diff<0>(grid_ptr,grid_index));
#else
       T tmp;
       where(beta.template evaluate<0>() > 0) | tmp = fdu.template diff<0>(grid_ptr,grid_index);
       where(beta.template evaluate<0>() < 0) | tmp = fdd.template diff<0>(grid_ptr,grid_index);
       return beta.template evaluate<0>() *tmp;
#endif
    }
  };


inline operator T const () const {
    // compute advective fd at point grid_index
    return T(beta_dE<true,ndim-1>::value(beta,fdu,fdd, grid_ptr, grid_index));
  }
};

}
#endif
