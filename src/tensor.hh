//  TensorTemplates: C++ tensor class templates
//  Copyright (C) 2017, Ludwig Jens Papenfort
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
#include <iostream>

namespace tensors {

template <typename T, typename frame_t_, typename symmetry_t_, size_t rank_, typename index_t_,
          size_t ndim_>
class general_tensor_t
    : public tensor_expression_t<
          general_tensor_t<T, frame_t_, symmetry_t_, rank_, index_t_, ndim_>> {
public:
  //! Data type
  using data_t = T;
  //! tensor indices encoded in a std::tuple type
  //! In fact we only care about the type here, which should be of size rank
  using index_t = index_t_;
  //! frame type
  using frame_t = frame_t_;

  //! Rank of the tensor
  static constexpr size_t rank = rank_; // std::tuple_size<index_t>::value;
  //! Number of dimensions
  static constexpr size_t ndim = ndim_;

  //! This tensor has no symmetry
  using symmetry_t = symmetry_t_;

  //! Number of degrees of freedom
  static constexpr size_t ndof = symmetry_t::ndof;
  //! Number of components (disregarding any symmetries)
  static constexpr size_t ncomp = utilities::static_pow<ndim,rank>::value;

  //! This tensor type
  using this_tensor_t = general_tensor_t<T, frame_t_, symmetry_t_, rank_, index_t, ndim_>;

  //! Fill property type with constexpr and types above (used in expressions)
  using property_t = general_tensor_property_t<this_tensor_t>;

protected:
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
      : m_data({tensor_expression.template evaluate<
                  symmetry_t::template index_to_generic<I>::value>()...}) {

//    static_assert(std::is_same<frame_t, typename E::property_t::frame_t>::value,
//                  "Frame types don't match!");

    static_assert(ndim == E::property_t::ndim, "Dimensions don't match!");

    static_assert(rank == E::property_t::rank, "Ranks don't match!");

    static_assert(std::is_same<data_t, typename E::property_t::data_t>::value,
                  "Data types don't match!");

    static_assert(
        compare_index_types<index_t, typename E::property_t::index_t,
                                 rank>(),
        "Index types do not match (e.g. lower_t != upper_t)!");
  };

  //! Constructor from tensor expression
  //! Calls constructor above with index sequence Indices
  template <typename E, typename Indices = std::make_index_sequence<ndof>>
  general_tensor_t(tensor_expression_t<E> const &tensor_expression)
      : this_tensor_t(tensor_expression, Indices{}){};

  //! Constructor from parameters
  // SYM: this is in weird order for sym2 and rank > 2, first sym indices than rest
  template <typename... TArgs>
  general_tensor_t(data_t first_elem, TArgs... elem) : m_data({first_elem, static_cast<data_t>(elem)...}) {
	  static_assert(sizeof...(TArgs)==ndof-1, "You need to specify exactly ndof arguments!");
	  static_assert(utilities::all_true<(std::is_convertible<data_t,TArgs>::value)...>::value, "The data_types are incompatible!");
  };

  general_tensor_t() : m_data({0}){};

  //! Computes the GENERIC compressed index associated with the given indices
  //  All members are expecting a generic compressed index,
  //  which is then transformed internally.
  //  This makes it easier to write generic tensor expressions.
  //  The symmetry dependence of ndof makes sure that only ndof expressions
  //  are evaluated, when a tensor is instantiated.
  template <size_t a, size_t... indices>
  static inline __attribute__ ((always_inline)) constexpr size_t compressed_index() {
    static_assert(rank == sizeof...(indices) + 1,
                  "Number of indices must match rank!");
    static_assert(utilities::all_true<(indices < ndim)...>::value,
                  "Trying to access index > ndim!");

    return generic_symmetry_t<ndim,rank>::template compressed_index<a, indices...>::value;
  }

  //! Access the components of a tensor using (compile-time) natural indices
  //  This gives back a non-const reference, since this is a evaluated expression
  template <size_t... Ind>
  inline __attribute__ ((always_inline)) data_t & c() {
    return this->cc<compressed_index<Ind...>()>();
  }
  //  This gives back a const reference, for const tensors
  template <size_t... Ind>
  inline __attribute__ ((always_inline)) data_t c() const {
    return this->cc<compressed_index<Ind...>()>();
  }

  //! Get component reference at (generic) compressed index position
  //  Needed if one wants to assign something to a specific element,
  //  given a generic compressed index.
  template<size_t index>
  inline __attribute__ ((always_inline)) data_t & cc() {
    return m_data[symmetry_t::template index_from_generic<index>::value];
  }
  template<size_t index>
  inline __attribute__ ((always_inline)) data_t cc() const {
    return m_data[symmetry_t::template index_from_generic<index>::value];
  }

// CHECK: OBSOLETE but still used in expressions test, we should convert that and remove this
// in that case we also need to remove it in expression base class
  //! Access the components of a tensor using a (generic) compressed index
  inline __attribute__ ((always_inline)) T & operator[](size_t const a) {
    return m_data[a];
  }
  //! Access the components of a tensor using a (generic) compressed index
  inline __attribute__ ((always_inline)) T const & operator[](size_t const a) const {
    return m_data[a];
  }

  //! Evaluation routine for expression templates
  //  This is ALWAYS expecting index to be a GENERIC compressed index,
  //  which is then transformed to the respective symmetry compressed index
  template <size_t index> inline __attribute__ ((always_inline)) T const & evaluate() const {
    return m_data[symmetry_t::template index_from_generic<index>::value];
  }

  //! Set (part) of the tensor to the given expression
  //  Make sure to set a symmetric tensor to a symmetric tensor (cast), to remove redundant assignements
  template<int... Ind, typename E>
  inline __attribute__ ((always_inline)) void set(E const &e) {
    // count free indices
    constexpr size_t num_free_indices = count_free_indices<Ind...>::value;
    constexpr size_t max_shift = (property_t::ndim - E::property_t::ndim);

    static_assert(sizeof...(Ind) == property_t::rank, "You need to specify rank indices.");
    static_assert(num_free_indices == E::property_t::rank, "You need to specify rank free indices.");
    static_assert(utilities::all_true<
                    ((Ind >= -static_cast<int>(max_shift + 1)) &&
                     (Ind < static_cast<int>(property_t::ndim)))...
                  >::value,
                  "Indices out of range. Use 0..ndim-1 to fix indices "
                  "and -1...-(ndim - ndim_in)-1 to indicate a (shifted) free index.");
    static_assert(E::property_t::ndim <= property_t::ndim, "Can't set a lower dim tensor to a higher dim.");

    // recursion over all components of e, sets all matching components of *this
    setter_t<E,this_tensor_t,E::property_t::ndof-1,Ind...>::set(e,*this);
  }

  //! Sets tensor to zero for all components
  inline __attribute__ ((always_inline)) void zero() {
    for (size_t i = 0; i < ndof; ++i) {
      m_data[i] = 0;
    }
  }

  //! Compute the Frobenius norm
  inline __attribute__ ((always_inline)) data_t norm() {
    data_t squared_sum = sum_squares<this_tensor_t,ncomp-1>::sum(*this);
    return std::sqrt(squared_sum);
  }

  //! Comparison routine to a tensor of the same kind
  //  This is not optimally optimized, please change if used in non-debugging environment
  //  Any expression passed as argument is converted to this_tensor_t automagically (if possible)
  template<int exponent>
  inline __attribute__ ((always_inline)) decltype(auto) compare_components(this_tensor_t const& t) {
    double eps = 1.0/utilities::static_pow<10,exponent>::value;

    double max_rel_err = 0;
    for(size_t i = 0; i<ndof; ++i) {
      auto rel_err_ = 2*std::abs(m_data[i] - t[i])
                      /(std::abs(m_data[i])+std::abs(t[i]));
      #ifdef TENSORS_VECTORIZED
      double rel_err = rel_err_.max();
      #else
      double rel_err = rel_err_;
      #endif
      if(rel_err > max_rel_err) {
        max_rel_err = rel_err;
      }
      if(rel_err > eps) {
        return std::pair<bool,decltype(rel_err)>(false,rel_err);
      }
    }
    return std::pair<bool,decltype(max_rel_err)>(true,max_rel_err);
  }

  //! Easy print to out stream, e.g. std::out
  //  This stream operator is automatically called for any tensor expression,
  //  since they are implicitly convertible (this_tensor_t constructor from an expression).
  friend std::ostream& operator<< (std::ostream& stream, const this_tensor_t& t) {
    stream << "[";
    for(size_t i = 0; i<ndof-1; ++i) {
      stream << t[i] << " ";
    }
    stream << t[ndof-1] << "]";
    return stream;
  }


  //Add expression to tensor
  template <typename E>
  inline __attribute__ ((always_inline)) void operator+=(tensor_expression_t<E> const &tensor_expression){
//    static_assert(std::is_same<frame_t, typename E::property_t::frame_t>::value,
//                  "Frame types don't match!");

    static_assert(ndim == E::property_t::ndim, "Dimensions don't match!");

    static_assert(rank == E::property_t::rank, "Ranks don't match!");

    static_assert(std::is_same<data_t, typename E::property_t::data_t>::value,
                  "Data types don't match!");

    static_assert(
        compare_index_types<index_t, typename E::property_t::index_t,
                                 rank>(),
        "Index types do not match (e.g. lower_t != upper_t)!");

   add_to_tensor_t<E,this_tensor_t,ndof-1>::add_to_tensor(tensor_expression,*this);

  };

  //Subtract expression from tensor
  template <typename E>
  inline __attribute__ ((always_inline)) void operator-=(tensor_expression_t<E> const &tensor_expression){
//    static_assert(std::is_same<frame_t, typename E::property_t::frame_t>::value,
//                  "Frame types don't match!");

    static_assert(ndim == E::property_t::ndim, "Dimensions don't match!");

    static_assert(rank == E::property_t::rank, "Ranks don't match!");

    static_assert(std::is_same<data_t, typename E::property_t::data_t>::value,
                  "Data types don't match!");

    static_assert(
        compare_index_types<index_t, typename E::property_t::index_t,
                                 rank>(),
        "Index types do not match (e.g. lower_t != upper_t)!");

   subtract_from_tensor_t<E,this_tensor_t,ndof-1>::subtract_from_tensor(tensor_expression,*this);

  };


  //Multiply tensor with scalar
  inline __attribute__ ((always_inline)) void operator*=(data_t const & lambda){

   multiply_tensor_with_t<this_tensor_t,ndof-1>::multiply_tensor_with(lambda,*this);

  };

  //Divide tensor by scalar
  inline __attribute__ ((always_inline)) void operator/=(data_t const & lambda){
  
   multiply_tensor_with_t<this_tensor_t,ndof-1>::multiply_tensor_with(1./lambda,*this);
 
  };

};

template<typename E>
typename E::property_t::this_tensor_t evaluate(E const & u){
   return typename E::property_t::this_tensor_t(u);
}

/**
 * Bugfix to allow expressions such as evaluate(contract(...) + 42);
 **/
inline double evaluate(double const & u){
   return u;
}


} // namespace tensors

#endif
