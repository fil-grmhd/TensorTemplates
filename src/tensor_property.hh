//  TensorTemplates: C++ tensor class templates
//  Copyright (C) 2017, Ludwig Jens Papenfort
//                      <papenfort@th.physik.uni-frankfurt.de>
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef TENSORS_PROPERTY_HH
#define TENSORS_PROPERTY_HH
#include "tensor_index_reduction.hh"

namespace tensors {

// SYM: check if property classes could forward / preserve symmetry

//! Property class holding data and types defining a specific tensor
template <typename E, bool is_persistent_ = false> class general_tensor_property_t {
public:
  //! Data type of components
  using data_t = typename E::data_t;
  //! Frame in which this tensor is defined
  using frame_t = typename E::frame_t;
  //! Symmetry type of this tensor
  using symmetry_t = typename E::symmetry_t;
  //! Number of dimensions
  static constexpr size_t ndim = E::ndim;
  //! Rank of the tensor (i.e. number of indices)
  static constexpr size_t rank = E::rank;
  //! Number of degrees of freedom
  static constexpr size_t ndof = symmetry_t::ndof;
  //! Number of components (disregarding any symmetries)
  static constexpr size_t ncomp = utilities::static_pow<ndim,rank>::value;

  //! Type of tensor indices (e.g. tuple of upper/lower_t)
  using index_t = typename E::index_t;
  //! Actual type of tensor with these properties
  using this_tensor_t = typename E::this_tensor_t;

  //! Is this a persistent object or could it be optimized away?
  //  i.e. does it have storage <=> is it an instantiated tensor
  static constexpr bool is_persistent = is_persistent_;
};

//! Property class holding data and types defining arithmetic combination of two
//! tensors of same type
//  These operations don't change the tensor properties, but one has to check
//  for compatibility
template <typename E1, typename E2>
class arithmetic_expression_property_t
    : public general_tensor_property_t<typename E1::property_t::this_tensor_t> {
public:
  using symmetry_t = typename std::conditional<
                                std::is_same<
                                  typename E1::property_t::symmetry_t,
                                  typename E2::property_t::symmetry_t
                                >::value,
                                typename E1::property_t::symmetry_t,
		                            generic_symmetry_t<
                                  E1::property_t::ndim,
                                  E1::property_t::rank>
                              >::type;

  using this_tensor_t = general_tensor_t<typename E1::property_t::data_t,
                                         typename E1::property_t::frame_t,
                                         symmetry_t,
                                         E1::property_t::rank,
                                         typename E1::property_t::index_t,
                                         E1::property_t::ndim>;

  static_assert(
      std::is_same<typename E1::property_t::frame_t,
                   typename E2::property_t::frame_t>::value ||
          std::is_same<typename E1::property_t::frame_t, any_frame_t>::value ||
          std::is_same<typename E2::property_t::frame_t, any_frame_t>::value,
      "Frame types don't match!");

  static_assert(E1::property_t::ndim == E2::property_t::ndim,
                "Dimensions don't match!");

  static_assert(E1::property_t::rank == E2::property_t::rank,
                "Ranks don't match!");

  static_assert(std::is_same<typename E1::property_t::data_t,
                             typename E2::property_t::data_t>::value,
                "Data types don't match!");

  // CHECK: is this really useful in tensor arithmetic expressions?
  static_assert(compare_index_types<typename E1::property_t::index_t,
                                    typename E2::property_t::index_t,
                                    E1::property_t::rank>(),
                "Index types do not match (e.g. lower_t != upper_t)!");
};

//! Property class holding data and types defining a scalar operation on one
//! tensor
//  These operations don't change the tensor properties, but one could check
//  scalar data type
template <typename E>
class scalar_expression_property_t
    : public general_tensor_property_t<typename E::property_t::this_tensor_t> {
  /*
        static_assert(std::is_same<typename E::property_t::data_t,
     scalar_data_t>::value,
                      "Data types don't match!");
  */
};

//! Property class holding data and types defining a tensor expression resulting
//! from a contraction
template <size_t i1, size_t i2, typename E1, typename E2>
class contraction_property_t {
public:
  using data_t = typename E1::property_t::data_t;
  using frame_t = typename E1::property_t::frame_t;

  static constexpr size_t ndim = E1::property_t::ndim;
  // two indices are removed by this expression
  static constexpr size_t rank =
      E1::property_t::rank + E2::property_t::rank - 2;

  //! Symmetry type of this tensor
  using symmetry_t = generic_symmetry_t<ndim,rank>;
  //! Number of degrees of freedom
  static constexpr size_t ndof = symmetry_t::ndof;
  //! Number of components (disregarding any symmetries)
  static constexpr size_t ncomp = utilities::static_pow<ndim,rank>::value;

  //! Is this a persistent object or could it be optimized away?
  //  i.e. does it have storage <=> is it an instantiated tensor
  //  this is an expression, so NO
  static constexpr bool is_persistent = false;

  // static compile-time routine to get index_t, doesn't work, see generator above
  // static inline __attribute__ ((always_inline)) constexpr decltype(auto) get_index_t(){}

  using index_t =
      decltype(index_reduction_generator_t<i1, i2, E1, E2>::get_contraction_index_t());

  using this_tensor_t = general_tensor_t<data_t, frame_t, symmetry_t, rank, index_t, ndim>;


  // Do some compile time checks of the expression properties

  static_assert((i1 < E1::property_t::rank) && (i2 < E2::property_t::rank),
                "Contracted indices are out of bound, i.e. i1,i2 >= E1,E2::rank.");

  static_assert(
      std::is_same<typename E1::property_t::frame_t,
                   typename E2::property_t::frame_t>::value ||
          std::is_same<typename E1::property_t::frame_t, any_frame_t>::value ||
          std::is_same<typename E2::property_t::frame_t, any_frame_t>::value,
      "Frame types don't match!");

  static_assert(E1::property_t::ndim == E2::property_t::ndim,
                "Dimensions don't match!");

  static_assert(std::is_same<typename E1::property_t::data_t,
                             typename E2::property_t::data_t>::value,
                "Data types don't match!");

  static_assert(is_reducible<i1,i2,E1,E2>::value,"Can only contract covariant with contravariant indices!");

  static_assert(rank == std::tuple_size<index_t>::value,
                "Index tuple size != rank, this should not happen");
};

//! Property class holding data and types defining a tensor resulting from a trace
template<size_t i1, size_t i2, typename E>
class trace_property_t {
public:

  using data_t = typename E::property_t::data_t;
  using frame_t = typename E::property_t::frame_t;
  static constexpr size_t ndim = E::property_t::ndim;
  // two indices are removed by a trace
  static constexpr size_t rank = E::property_t::rank - 2;

  //! Symmetry type of this tensor
  using symmetry_t = generic_symmetry_t<ndim,rank>;
  //! Number of degrees of freedom
  static constexpr size_t ndof = symmetry_t::ndof;
  //! Number of components (disregarding any symmetries)
  static constexpr size_t ncomp = utilities::static_pow<ndim,rank>::value;

  //! Is this a persistent object or could it be optimized away?
  //  i.e. does it have storage <=> is it an instantiated tensor
  //  this is an expression, so NO
  static constexpr bool is_persistent = false;

  // static compile-time routine to get index_t, doesn't work, see above
  //static inline __attribute__ ((always_inline)) constexpr decltype(auto) get_index_t(){}

  using index_t = decltype(index_reduction_generator_t<i1,i2,E,E>::get_trace_index_t());

  using this_tensor_t = general_tensor_t<data_t,frame_t,symmetry_t,rank,index_t,ndim>;

  // Do some compile time checks of the expression properties

  // check if someone tries to trace a vector
  static_assert(E::property_t::rank > 1, "You cannot trace a vector, this is undefined!");
  // check if trace indices bounds are violated
  static_assert((i1 < E::property_t::rank) && (i2 < E::property_t::rank),
                "Traced indices are out of bound, i.e. i1,i2 >= rank.");
  // check if someone tries to trace an index with itself
  static_assert(i1 != i2, "You cannot trace an index with itself, i.e. make sure that i1 != i2");
  // this can only trigger if tensor_trace_t is created manually, see below in trace "operator"
  static_assert(i1 < i2, "Please make sure that traced indices are in ascending order, i.e. i1 < i2");

  static_assert(is_reducible<i1,i2,E,E>::value,"Can only contract covariant with contravariant indices!");
};


//! Property class holding data and types defining a tensor expression which
//! reduces two indices
template <size_t i2, typename E1, typename E2>
class metric_contraction_property_t {
public:
  using data_t = typename E2::property_t::data_t;
  using frame_t = typename E2::property_t::frame_t;
  static constexpr size_t ndim = E2::property_t::ndim;
  // two indices are removed by this expression
  static constexpr size_t rank = E2::property_t::rank;

  //! Symmetry type of this tensor
  using symmetry_t = generic_symmetry_t<ndim,rank>;
  //! Number of degrees of freedom
  static constexpr size_t ndof = symmetry_t::ndof;
  //! Number of components (disregarding any symmetries)
  static constexpr size_t ncomp = utilities::static_pow<ndim,rank>::value;

  //! Is this a persistent object or could it be optimized away?
  //  i.e. does it have storage <=> is it an instantiated tensor
  //  this is an expression, so NO
  static constexpr bool is_persistent = false;

  // static compile-time routine to get index_t
  // static inline __attribute__ ((always_inline)) constexpr decltype(auto) get_index_t(){}

  using index_t =
      decltype(index_reduction_generator_t<i2,i2,E1,E2>::get_metric_contraction_index_t());

  using this_tensor_t = general_tensor_t<data_t, frame_t, symmetry_t, rank, index_t, ndim>;

  //    static_assert(std::is_same<typename E1::property_t::frame_t, typename
  //    E2::property_t::frame_t>::value,
  //                  "Frame types don't match!");
  static_assert(
      E1::property_t::rank == 2 &&
          std::is_same<typename std::tuple_element<
                           0, typename E1::property_t::index_t>::type,
                       typename std::tuple_element<
                           0, typename E1::property_t::index_t>::type>::value &&
          std::is_same<typename E1::property_t::frame_t, any_frame_t>::value,
      "The first tensor has to be a metric_t!");

  static_assert(E1::property_t::ndim == E2::property_t::ndim,
                "Dimensions don't match!");

  static_assert(std::is_same<typename E1::property_t::data_t,
                             typename E2::property_t::data_t>::value,
                "Data types don't match!");

  static_assert(
      std::is_same<
          typename std::conditional<
              std::is_same<typename std::tuple_element<
                               0, typename E1::property_t::index_t>::type,
                           lower_t>::value,
              lower_t, upper_t>::type,
          typename std::conditional<
              std::is_same<typename std::tuple_element<
                               i2, typename E2::property_t::index_t>::type,
                           upper_t>::value,
              lower_t, upper_t>::type>::value,
      "Can only contract covariant with contravariant indices!");

  static_assert(rank == std::tuple_size<index_t>::value,
                "Index tuple size != rank, this should not happen");
};

//! Property class holding data and types defining a tensor expression
//! which is the tensor product of two tensors
template <typename E1, typename E2>
class index_concat_property_t {
public:
  using data_t = typename E2::property_t::data_t;
  using frame_t = typename E2::property_t::frame_t; //Might need to be changed to any_frame_t
  static constexpr size_t ndim = E2::property_t::ndim;
  // two indices are "added" by this expression
  static constexpr size_t rank = E2::property_t::rank + E1::property_t::rank;

  //! Symmetry type of this tensor
  using symmetry_t = generic_symmetry_t<ndim,rank>;
  //! Number of degrees of freedom
  static constexpr size_t ndof = symmetry_t::ndof;
  //! Number of components (disregarding any symmetries)
  static constexpr size_t ncomp = utilities::static_pow<ndim,rank>::value;

  //! Is this a persistent object or could it be optimized away?
  //  i.e. does it have storage <=> is it an instantiated tensor
  //  this is an expression, so NO
  static constexpr bool is_persistent = false;

  // static compile-time routine to get index_t
  // static inline __attribute__ ((always_inline)) constexpr decltype(auto) get_index_t(){}

  using index_t =
      decltype(index_reduction_generator_t<0,0,E1,E2>::get_concat_index_t());

  using this_tensor_t = general_tensor_t<data_t, frame_t, symmetry_t, rank, index_t, ndim>;

  static_assert(
      std::is_same<typename E1::property_t::frame_t,
                   typename E2::property_t::frame_t>::value ||
          std::is_same<typename E1::property_t::frame_t, any_frame_t>::value ||
          std::is_same<typename E2::property_t::frame_t, any_frame_t>::value,
      "Frame types don't match!");

  static_assert(E1::property_t::ndim == E2::property_t::ndim,
                "Dimensions don't match!");

  static_assert(std::is_same<typename E1::property_t::data_t,
                             typename E2::property_t::data_t>::value,
                "Data types don't match!");

  static_assert(rank == std::tuple_size<index_t>::value,
                "Index tuple size != rank, this should not happen");
};


} // namespace tensors

#endif
