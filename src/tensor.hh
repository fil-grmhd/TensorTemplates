//  TensorTemplates: C++ tensor class templates
//  Copyright (C) 2016, Ludwig Jens Papenfort
//                      <papenfort@th.physik.uni-frankfurt.de>
//  Copyright (C) 2017, Elias Roland Most
//                      <emost@th.physik.uni-frankfurt.de>
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

#ifndef TENSORS_TENSOR_HH
#define TENSORS_TENSOR_HH

#include <array>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <utility>

#include "tensor_defs.hh"
#include "tensor_expressions.hh"
#include "utilities.hh"

namespace tensors {

template <typename T, typename frame_t_, size_t rank_, typename index_t_,
          size_t ndim_>
class general_tensor_t
    : public tensor_expression_t<
          general_tensor_t<T, frame_t_, rank_, index_t_, ndim_>> {
public:
  //! Data type
  using data_t = T;
  //! tensor indices encoded in a std::tuple type
  //! In fact we only care about the type here, which should be of size rank
  using index_t = index_t_;
  //! frame type
  using frame_t = frame_t_;

  //! Rank of the tensor
  static size_t constexpr rank = rank_; // std::tuple_size<index_t>::value;
  //! Number of dimensions
  static size_t constexpr ndim = ndim_;
  //! Number of degrees of freedom
  static size_t constexpr ndof = utilities::static_pow<ndim, rank>::value;

  //! This tensor type
  using this_tensor_t = general_tensor_t<T, frame_t_, rank_, index_t, ndim_>;

  //! Fill property type with constexpr and types above (used in expressions)
  using property_t = general_tensor_property_t<this_tensor_t>;

private:
  //! Data storage for ndof elements
  std::array<T, ndof> m_data{};

public:
  //! Constructor from tensor expression given a index sequence
  //! Generates components from arbitrary tensor expression type (e.g. chained
  //! ones)

  // This is not a copy constructor, leading to (hopefully optimal) default copy
  // and move constructors
  // C++14 way to completely get around the for loop to initialize component
  // array
  template <typename E, std::size_t... I>
  general_tensor_t(tensor_expression_t<E> const &tensor_expression,
                   std::index_sequence<I...>)
      : m_data({tensor_expression.template evaluate<I>()...}) {

    static_assert(std::is_same<frame_t, typename E::property_t::frame_t>::value,
                  "Frame types don't match!");

    static_assert(ndim == E::property_t::ndim, "Dimensions don't match!");

    static_assert(rank == E::property_t::rank, "Ranks don't match!");

    static_assert(std::is_same<data_t, typename E::property_t::data_t>::value,
                  "Data types don't match!");

    static_assert(
        utilities::compare_index<index_t, typename E::property_t::index_t,
                                 rank>(),
        "Index types do not match (e.g. lower_t != upper_t)!");
  };

  //! Constructor from tensor expression
  //! Calls constructor above with index sequence Indices
  template <typename E, typename Indices = std::make_index_sequence<ndof>>
  general_tensor_t(tensor_expression_t<E> const &tensor_expression)
      : this_tensor_t(tensor_expression, Indices{}){};

  //! Constructor from parameters
  template <typename... TArgs>
  general_tensor_t(data_t first_elem, TArgs... elem)
      : m_data({first_elem, static_cast<data_t>(elem)...}) {
	  static_assert(sizeof...(TArgs)==ndof-1, "You need to specify exactly ndof arguments!");
	  static_assert(utilities::all_true<(std::is_convertible<data_t,TArgs>::value)...>::value, "The data_types are incompatible!");
  };

  general_tensor_t() : m_data({0}){};

  //! Computes the compressed index associated with the given indices
  //! one would need to cusomize this if one wants to implement symmetries
  template <size_t a, size_t... indices>
  static inline constexpr size_t compressed_index() {
    static_assert(rank == sizeof...(indices) + 1,
                  "Number of indices must match rank!");
    static_assert(utilities::all_true<(indices < ndim)...>::value,
                  "Trying to access index > ndim!");

    return compressed_index_t<ndim, a, indices...>::value;
  }

  template <typename tuple_t>
  static inline constexpr size_t compressed_index(tuple_t t) {
    return compressed_index_tuple<ndim>(t);
  }

  //! Access the components of a tensor using a compressed index
  /*!
   *  The data is stored in a column major format
   */
  inline T &operator[](size_t const a) { return m_data[a]; }
  //! Access the components of a tensor using a compressed index
  /*!
   *  The data is stored in a column major format
   */
  inline T const &operator[](size_t const a) const { return m_data[a]; }

  //! Evaluation routine for expression templates
  template <size_t index> inline T const & evaluate() const {
    return m_data[index];
  }

  // CHECK: element access is only possible with compile-time constants
  //        we should add (i,j,k,...) operator also to the class

  //! Access the components of a tensor using the natural indices
  template <size_t a, size_t... indices> inline T &operator()() {
    static_assert(sizeof...(indices) + 1 == rank,
                  "Number of indices must match rank.");
	return m_data[compressed_index<a,indices...>()];
  }

  //! Access the components of a tensor using the natural indices
  template <size_t a, size_t... indices> inline T const &operator()() {
    static_assert(sizeof...(indices) + 1 == rank,
                  "Number of indices must match rank.");

    return evaluate<compressed_index<a, indices...>()>();
  }

  //! Sets tensor to zero for all components
  inline void zero() {
    for (size_t i = 0; i < ndof; ++i) {
      m_data[i] = 0;
    }
  }

  //! Comparison routine to a tensor of the same kind
  template<int exponent>
  inline bool compare_components(this_tensor_t const& t) {
    double eps = 1.0/utilities::static_pow<10,exponent>::value;

    for(size_t i = 0; i<ndof; ++i) {
      if(2*std::abs(m_data[i]- t[i])
         /(std::abs(m_data[i])+std::abs(t[i])) > eps) {
        return false;
      }
    }
    return true;
  }

  //! Easy print to out stream, e.g. std::out
  //  This stream operator is automatically called for any tensor expression,
  //  since they are implicitly convertible (this_tensor_t constructor from an expression).
  friend std::ostream& operator<< (std::ostream& stream, const this_tensor_t& t) {
    stream << "[";
    for(size_t i = 0; i<ndof-1; ++i)
      stream << t[i] << " ";
    stream << t[ndof-1] << "]";
    return stream;
  }

};

} // namespace tensors

#endif
