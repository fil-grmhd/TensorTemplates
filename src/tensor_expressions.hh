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

#include <cmath>
#include <type_traits>
#include <tuple>
#include <array>
#include <utility>

#include "tensor_index.hh"


namespace tensors {

template <typename E>
class tensor_expression_t {
  public:
    inline decltype(auto) operator[](size_t i) const { return static_cast<E const&>(*this)[i];     };
    template<size_t index>
    inline decltype(auto) evaluate() const { return static_cast<E const&>(*this).template evaluate<index>();     };
      
    operator E& () { return static_cast<E&>(*this); };
    operator const E& () const { return static_cast<const E&>(*this); };

  };

template <typename E1, typename E2>
class tensor_sum_t : public tensor_expression_t<tensor_sum_t<E1, E2> > {
   E1 const& _u;
   E2 const& _v;
    
public:

    using data_t = typename E1::data_t;
    using frame_t = typename E1::frame_t;

   using this_tensor_t = typename E1::this_tensor_t;

    tensor_sum_t(E1 const& u, E2 const& v) : _u(u), _v(v) {
            static_assert(std::is_same<typename E1::this_tensor_t, typename E2::this_tensor_t>::value,
		"Tensor types don't match!");
    };
    
    inline decltype(auto) operator[](size_t i) const { return _u[i] + _v[i]; };
    template<size_t index>
    inline decltype(auto) evaluate() const { return _u.template evaluate<index>() + _v.template evaluate<index>(); };
};

//Q: Is it better to define this separately or to just use a - b = a +(-1)*b ??

template <typename E1, typename E2>
class tensor_sub_t : public tensor_expression_t<tensor_sub_t<E1, E2> > {
   E1 const& _u;
   E2 const& _v;
    
public:

   using this_tensor_t = typename E1::this_tensor_t;
    using data_t = typename E1::data_t;
    using frame_t = typename E1::frame_t;


    tensor_sub_t(E1 const& u, E2 const& v) : _u(u), _v(v) {
            static_assert(std::is_same<typename E1::this_tensor_t, typename E2::this_tensor_t>::value,
		"Tensor types don't match!");
    };
    
    inline decltype(auto) operator[](size_t i) const { return _u[i] - _v[i]; };
    template<size_t index>
    inline decltype(auto) evaluate() const { return _u.template evaluate<index>() - _v.template evaluate<index>(); };
};

template <typename E2>
class tensor_scalar_mult_t : public tensor_expression_t<tensor_scalar_mult_t< E2> > {
   typename E2::data_t const&  _u;
   E2 const& _v;
    
public:

   using this_tensor_t = typename E2::this_tensor_t;
    using data_t = typename E2::data_t;
    using frame_t = typename E2::frame_t;

    tensor_scalar_mult_t(data_t const& u, E2 const& v) : _u(u), _v(v) {};
    
    inline decltype(auto) operator[](size_t i) const { return _u * _v[i]; };
    template<size_t index>
    inline decltype(auto) evaluate() const { return _v.template evaluate<index>() *_u;};
};

template <typename E2>
class tensor_scalar_div_t : public tensor_expression_t<tensor_scalar_div_t< E2> > {
   typename E2::data_t const&  _v;
   E2 const& _u;
    
public:

   using this_tensor_t = typename E2::this_tensor_t;
    using data_t = typename E2::data_t;
    using frame_t = typename E2::frame_t;

    tensor_scalar_div_t( E2 const& u, data_t const & v) : _u(u), _v(v) {};
    
    inline decltype(auto) operator[](size_t i) const { return _u[i]/_v; };
    template<size_t index>
    inline decltype(auto) evaluate() const { return _u.template evaluate<index>()/_v;};
};


template <typename E1, typename E2>
tensor_sum_t<E1,E2> const
inline operator+(E1 const& u, E2 const& v) {
   return tensor_sum_t<E1, E2>(u, v);
}

template <typename E1, typename E2>
tensor_sub_t<E1,E2> const
inline operator-(E1 const& u, E2 const& v) {
   return tensor_sub_t<E1, E2>(u, v);
}


template <typename E2>
tensor_scalar_mult_t<E2> const
inline operator*(typename E2::data_t const& u, E2 const& v) {
  //Should we add a data_t check here??
   return tensor_scalar_mult_t<E2>(u, v);
}

template < typename E2>
tensor_scalar_mult_t<E2> const
inline operator*(E2 const& u, typename E2::data_t const& v) {
  //Should we add a data_t check here??
   return v*u;
}

template < typename E2>
tensor_scalar_div_t<E2> const
inline operator/(E2 const& u, typename E2::data_t const& v) {
  //Should we add a data_t check here??
   return tensor_scalar_div_t<E2>(u,v);
}


} // namespace tensors

#endif
