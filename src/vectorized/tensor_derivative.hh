#ifndef TENSORS_DERIVATIVE_VEC_HH
#define TENSORS_DERIVATIVE_VEC_HH

namespace tensors {

//! Expression template for a vectorized partial derivative of a tensor
template <typename E, typename array_t, typename fd_t>
class tensor_partial_derivative_vt
    : public tensor_expression_t<tensor_partial_derivative_vt<E,array_t,fd_t>> {

private:
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
  using index_t = decltype(std::tuple_cat(std::declval<E::property_t::index_t>(),
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

  // The data type is in this case a vector register type
  using vec_t = typename property_t::data_t;
  // The actual data type
  using data_t = typename vec_t::EntryType;



  tensor_partial_derivative_vt(size_t const grid_index_, array_t const & ptr_array_, fd_t const & fd_)
                            : grid_index(grid_index_), ptr_array(ptr_array_), fd(fd_) {};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline decltype(auto)
  operator[](size_t i) const = delete;


  template <size_t index>
  inline decltype(auto) evaluate() const {
    // get index without partial derivative index part
    constexpr size_t tensor_gen_index = index % utilities::static_pow<ndim,rank-1>::value;
    // transform to underlying symmetry index (since array has only ndof elements)
    constexpr size_t tensor_sym_index = E::property_t::symmetry_t
                                          ::template index_from_generic<tensor_gen_index>::value;
    // get last index, which is the derivative index and defines the direction of the fd
    constexpr size_t deriv_index = property_t::symmetry_t::template uncompress_index<rank-1,index>::value;

    // vector register of data_t with Vc::Vector<data_t>::Size elemets
    vec_t vec_register;

    // compute fd at points grid_index...gird_index + Vc::Vector<data_t>::Size of tensor_sym_index' component
    for(size_t i = 0; i < vec_t::Size; ++i) {
      vec_register[i] = fd.template diff<deriv_index>(ptr_array[tensor_sym_index],grid_index+i);
    }

    return vec_register;
  }
};

}
#endif
