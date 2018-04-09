//  TensorTemplates: C++ tensor class templates
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

#ifndef TENSORS_EXPRESSIONS_HH
#define TENSORS_EXPRESSIONS_HH

#include <array>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <utility>

#include "tensor_index.hh"

namespace tensors {

//! Base class for tensor expressions, including actual tensors
template <typename E> class tensor_expression_t {
public:

  //! Index access of tensor expression E
  inline __attribute__ ((always_inline)) decltype(auto) operator[](size_t i) const {
    return static_cast<E const &>(*this)[i];
  };

  //! Evaluation routine, triggering actual computation
  template <size_t index> inline __attribute__ ((always_inline)) decltype(auto) evaluate() const {
    return static_cast<E const &>(*this).template evaluate<index>();
  };

  //! (generic) compressed index access to components of tensor expression E
  //  Calls evaluation, which can be expensive, if the expression is not evaluated yet.
  template<size_t index>
  inline __attribute__ ((always_inline)) decltype(auto) cc() const {
    return static_cast<E const &>(*this).template evaluate<index>();
  }

  //! Natural index access to components of tensor expression E
  //  Calls evaluation, which can be expensive, if the expression is not evaluated yet.
  template <size_t... Ind>
  inline __attribute__ ((always_inline)) decltype(auto) c() const {
    constexpr size_t index = generic_symmetry_t<
                               E::property_t::ndim,
                               E::property_t::rank
                             >::template compressed_index<Ind...>::value;

    return static_cast<E const &>(*this).template evaluate<index>();
  };

  //! Conversion operator to reference of tensor expression E
  operator E &() { return static_cast<E &>(*this); };
  //! Conversion operator to constant reference of tensor expression E
  operator const E &() const { return static_cast<const E &>(*this); };
};

template <typename E1, typename E2>
class tensor_sum_t : public tensor_expression_t<tensor_sum_t<E1, E2>> {
  E1 const &_u;
  E2 const &_v;

public:
  // This operation doesn't change the tensor properties, but one has to check
  // for compatibility
  using property_t = arithmetic_expression_property_t<E1, E2>;

  tensor_sum_t(E1 const &u, E2 const &v) : _u(u), _v(v){};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline __attribute__ ((always_inline)) decltype(auto)
  operator[](size_t i) const = delete;
  //    inline __attribute__ ((always_inline)) decltype(auto) operator[](size_t i) const { return _u[i] + _v[i];
  //    };

  template <size_t index> inline __attribute__ ((always_inline)) decltype(auto) evaluate() const {
    return _u.template evaluate<index>() + _v.template evaluate<index>();
  };
};

template <typename E1, typename E2>
class tensor_sub_t : public tensor_expression_t<tensor_sub_t<E1, E2>> {
  E1 const &_u;
  E2 const &_v;

public:
  // This operation doesn't change the tensor properties, but one has to check
  // for compatibility
  using property_t = arithmetic_expression_property_t<E1, E2>;

  tensor_sub_t(E1 const &u, E2 const &v) : _u(u), _v(v){};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline __attribute__ ((always_inline)) decltype(auto)
  operator[](size_t i) const = delete;
  //    inline __attribute__ ((always_inline)) decltype(auto) operator[](size_t i) const { return _u[i] - _v[i];
  //    };

  template <size_t index> inline __attribute__ ((always_inline)) decltype(auto) evaluate() const {
    return _u.template evaluate<index>() - _v.template evaluate<index>();
  };
};

template <typename E>
class tensor_scalar_mult_t
    : public tensor_expression_t<tensor_scalar_mult_t<E>> {
  typename E::property_t::data_t const &_u;
  E const &_v;

public:
  //! Scalar expression doesn't change tensor properties
  using property_t = scalar_expression_property_t<E>;

  tensor_scalar_mult_t(typename property_t::data_t const &u, E const &v)
      : _u(u), _v(v){};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline __attribute__ ((always_inline)) decltype(auto)
  operator[](size_t i) const = delete;
  //    inline __attribute__ ((always_inline)) decltype(auto) operator[](size_t i) const { return _u * _v[i]; };

  template <size_t index> inline __attribute__ ((always_inline)) decltype(auto) evaluate() const {
    return _v.template evaluate<index>() * _u;
  };
};

template <typename E>
class tensor_scalar_div_t : public tensor_expression_t<tensor_scalar_div_t<E>> {
  typename E::property_t::data_t const &_v;
  E const &_u;

public:
  //! Scalar expression doesn't change tensor properties
  using property_t = scalar_expression_property_t<E>;

  tensor_scalar_div_t(E const &u, typename property_t::data_t const &v)
      : _u(u), _v(v){};

  [[deprecated("Do not access the tensor expression via the [] operator, this "
               "is UNDEFINED!")]] inline __attribute__ ((always_inline)) decltype(auto)
  operator[](size_t i) const = delete;
  //    inline __attribute__ ((always_inline)) decltype(auto) operator[](size_t i) const { return _u[i]/_v; };

  template <size_t index> inline __attribute__ ((always_inline)) decltype(auto) evaluate() const {
    return _u.template evaluate<index>() / _v;
  };
};

template <typename E1, typename E2>
tensor_sum_t<E1, E2> const inline __attribute__ ((always_inline)) operator+(E1 const &u, E2 const &v) {
  return tensor_sum_t<E1, E2>(u, v);
}

template <typename E1, typename E2>
tensor_sub_t<E1, E2> const inline __attribute__ ((always_inline)) operator-(E1 const &u, E2 const &v) {
  return tensor_sub_t<E1, E2>(u, v);
}

template <typename E>
tensor_scalar_mult_t<E> const inline __attribute__ ((always_inline))
operator*(typename E::property_t::data_t const &u, E const &v) {
  // Should we add a data_t check here??
  return tensor_scalar_mult_t<E>(u, v);
}

template <typename E>
tensor_scalar_mult_t<E> const inline __attribute__ ((always_inline))
operator*(E const &u, typename E::property_t::data_t const &v) {
  // Should we add a data_t check here??
  return v * u;
}

template <typename E>
tensor_scalar_div_t<E> const inline __attribute__ ((always_inline))
operator/(E const &u, typename E::property_t::data_t const &v) {
  // Should we add a data_t check here??
  return tensor_scalar_div_t<E>(u, v);
}

template <typename E>
tensor_scalar_mult_t<E> const inline __attribute__ ((always_inline)) operator-(E const &u) {
  // Attention: This does not work properly and most likely spills out wrong numbers.
  // I don't yet know why.
  typename E::property_t::data_t minus(-1.0); // This must be a double, not an integer.
  return tensor_scalar_mult_t<E>(minus, u);
}

} // namespace tensors

#endif
