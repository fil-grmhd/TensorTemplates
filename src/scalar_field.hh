#ifndef TENSORS_SCALAR_FIELD_HH
#define TENSORS_SCALAR_FIELD_HH

namespace tensors {

#if !defined(TENSORS_VECTORIZED) || !defined(TENSORS_AUTOVEC)

//! A thin wrapper around a scalar, implementing FD and set routines
//  T is the data type, e.g. a double
template<typename T, typename ptr_t>
class scalar_wrapper_t {
  protected:
    //! Storage for the grid pointer
    ptr_t const grid_ptr;
    //! Internal (grid pointer) index
    size_t const grid_index;

  public:

    struct property_t {
      using this_tensor_t = T;
      static constexpr bool is_persistent = true;
    };


    //! Constructor (called from a scalar field)
    scalar_wrapper_t(ptr_t const grid_ptr_, size_t const grid_index_)
        : grid_ptr(grid_ptr_), grid_index(grid_index_) {}

    //! Returns a partial derivative of this scalar field
    template<typename fd_t>
    inline __attribute__ ((always_inline)) decltype(auto) finite_diff(fd_t const & fd) const {
      return scalar_partial_derivative_t<T,ptr_t,fd_t>(grid_index,grid_ptr,fd);
    }

    //! Returns the upwind derivative of this tensor field
    template<typename fd_u_t,typename fd_d_t,typename beta_t>
    inline __attribute__ ((always_inline)) decltype(auto) upwind_finite_diff(fd_u_t const & fdu, fd_d_t const & fdd, beta_t const & beta) const {
      return scalar_advective_derivative_t<T,ptr_t,beta_t,fd_u_t,fd_d_t>(grid_index, grid_ptr, beta, fdu, fdd);
    }

    //! Sets the scalar at grid_index to data
    inline __attribute__ ((always_inline)) void operator=(T const data) {
      grid_ptr[grid_index] = data;
    }

    //! Add the scalar at grid_index to data
    inline __attribute__ ((always_inline)) void operator+=(T const data) {
      grid_ptr[grid_index] += data;
    }

    //! Substract the scalar at grid_index by data
    inline __attribute__ ((always_inline)) void operator-=(T const data) {
      grid_ptr[grid_index] -= data;
    }

    //! Multiply the scalar at grid_index by data
    inline __attribute__ ((always_inline)) void operator*=(T const data) {
      grid_ptr[grid_index] *= data;
    }

    //! Divide the scalar at grid_index by data
    inline __attribute__ ((always_inline)) void operator/=(T const data) {
      grid_ptr[grid_index] /= data;
    }

    //! Conversion operators, so that it is usable as a simple POD scalar
    operator T & () {
      return grid_ptr[grid_index];
    }
    operator T const & () const {
      return grid_ptr[grid_index];
    }

};


//! Template for a scalar field
// A scalar field is not a tensor expression,
// but delivers scalar wrappers at different grid points.
// This is a thin-wrapper around a grid pointer,
// which is needed to write vectorization-agnostic code.
template<typename T>
class scalar_field_t {
  protected:
    //! Storage for the grid pointer
    T * __restrict__ const grid_ptr;

  public:
    //! Constructor from grid pointer parameter
    scalar_field_t(T * __restrict__ const grid_ptr_)
        : grid_ptr(grid_ptr_) {};

    //! Returns value at (grid pointer) index i
    inline __attribute__ ((always_inline)) decltype(auto) operator[](size_t const i) const {
      return scalar_wrapper_t<T,decltype(grid_ptr)>(grid_ptr,i);
    }
};
#endif

#ifdef TENSORS_VECTORIZED
//! A thin wrapper around a scalar, implementing FD and set routines
//  T is the data type, e.g. a double
template<typename T, typename ptr_t>
class scalar_wrapper_vt {
  public:
    // Data is stored as vector register
    using vec_t = Vc::native_simd<T>;
    // The actual data type
    using data_t = T;

  protected:
    //! Storage for the grid pointer
    ptr_t const grid_ptr;
    //! Internal (grid pointer) index
    size_t const grid_index;

  public:
    struct property_t {
      using this_tensor_t = vec_t;
      static constexpr bool is_persistent = true;
    };

    //! Constructor (called from a scalar field)
    scalar_wrapper_vt(ptr_t const grid_ptr_, size_t const grid_index_)
        : grid_ptr(grid_ptr_), grid_index(grid_index_) {}

    //! Returns a partial derivative of this scalar field
    template<typename fd_t>
    inline __attribute__ ((always_inline)) decltype(auto) finite_diff(fd_t const & fd) const {
      return scalar_partial_derivative_t<vec_t,ptr_t,fd_t>(grid_index,grid_ptr,fd);
    }

    //! Returns the upwind derivative of this tensor field
    template<typename fd_u_t,typename fd_d_t,typename beta_t>
    inline __attribute__ ((always_inline)) decltype(auto) upwind_finite_diff(fd_u_t const & fdu, fd_d_t const & fdd, beta_t const & beta) const {
      return scalar_advective_derivative_t<vec_t,ptr_t,beta_t,fd_u_t,fd_d_t>(grid_index, grid_ptr, beta, fdu, fdd);
    }

    //! Sets vec_t::size() scalar field values beginning at grid_index to data
    inline __attribute__ ((always_inline)) void operator=(vec_t const & data) {
      data.copy_to(&grid_ptr[grid_index], Vc::vector_aligned);
    }

    //! Add the scalar at grid_index to data
    inline __attribute__ ((always_inline)) void operator+=(vec_t const & data) {
      // reads Vc::native_simd<data_t>::size() values from grid_index on into vector register
      vec_t vec_register(&grid_ptr[grid_index], Vc::vector_aligned);
      (*this) = vec_register + data;
    }

    //! Substract the scalar at grid_index by data
    inline __attribute__ ((always_inline)) void operator-=(vec_t const & data) {
      // reads Vc::native_simd<data_t>::size() values from grid_index on into vector register
      vec_t vec_register(&grid_ptr[grid_index], Vc::vector_aligned);
      (*this) = vec_register - data;
    }

    //! Multiply the scalar at grid_index by data
    inline __attribute__ ((always_inline)) void operator*=(vec_t const & data) {
      // reads Vc::native_simd<data_t>::size() values from grid_index on into vector register
      vec_t vec_register(&grid_ptr[grid_index], Vc::vector_aligned);
      (*this) = vec_register * data;
    }

    //! Divide the scalar at grid_index by data
    inline __attribute__ ((always_inline)) void operator/=(vec_t const & data) {
      // reads Vc::native_simd<data_t>::size() values from grid_index on into vector register
      vec_t vec_register(&grid_ptr[grid_index], Vc::vector_aligned);
      (*this) = vec_register / data;
    }

    //! Conversion operator, so that it is usable as plain vector register (Vc::native_simd)
    operator vec_t const () const {
      // reads Vc::native_simd<data_t>::size() values from grid_index on into vector register
      vec_t vec_register(&grid_ptr[grid_index], Vc::vector_aligned);

      return vec_register;
    }
};

//! Template for a vectorized scalar field
// A scalar field is not a tensor expression,
// but delivers scalar values at different grid points.
template<typename T>
class scalar_field_vt {
  public:
    // Data is stored as vector register
    using vec_t = Vc::native_simd<T>;
    // The actual data type
    using data_t = T;

  protected:
    //! Storage for grid pointer
    data_t * __restrict__ const grid_ptr;

  public:
    //! Constructor from grid pointer parameter
    scalar_field_vt(data_t * __restrict__ const grid_ptr_)
        : grid_ptr(grid_ptr_) {};

    //! Returns a vector register from (grid pointer) index i on
    inline __attribute__ ((always_inline)) decltype(auto) operator[](size_t const i) const {
      return scalar_wrapper_vt<T,decltype(grid_ptr)>(grid_ptr,i);
    }
};

#ifdef TENSORS_AUTOVEC
template<typename T>
using scalar_field_t = scalar_field_vt<T>;
#endif
#endif

} // namespace tensors
#endif
