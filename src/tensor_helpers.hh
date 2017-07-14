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

#ifndef TENSORS_HELPERS_HH
#define TENSORS_HELPERS_HH

namespace tensors {

// Template recursion to set components, fastest for chained expressions
template<typename E, typename T, size_t N, int... Ind>
struct setter_t {
  static inline __attribute__ ((always_inline)) void set(E const& e, T & t) {
    // convert 1,...,ndof-1 to a generic index
    constexpr size_t gen_index = E::property_t::symmetry_t::template index_to_generic<N>::value;

    // computes (shifted) compressed index of t,
    // corresponding to (generic) compressed index of e
    constexpr size_t c_index = compute_unsliced_cindex<
                                 T,
                                 E,
                                 gen_index,
                                 0,
                                 Ind...
                               >::value;

    // compressed element of generic index, index is transformed internally
    // this is necessary, because evaluate gives a const ref which is not assignable
    t.template cc<c_index>() = e.template evaluate<gen_index>();
    setter_t<E,T,N-1,Ind...>::set(e,t);
  }
};
template<typename E, typename T, int... Ind>
struct setter_t<E,T,0,Ind...> {
  static inline __attribute__ ((always_inline)) void set(E const& e, T & t) {
    constexpr size_t c_index = compute_unsliced_cindex<
                                 T,
                                 E,
                                 0,
                                 0,
                                 Ind...
                               >::value;

    t.template cc<c_index>() = e.template evaluate<0>();
  }
};

template<typename E, typename T, size_t N>
struct add_to_tensor_t{
  static inline __attribute__ ((always_inline)) void add_to_tensor( E const &e, T &t){
    constexpr size_t gen_index = T::property_t::symmetry_t::template index_to_generic<N>::value;
    t.template cc<gen_index>() += e.template evaluate<gen_index>();
    add_to_tensor_t<E,T,N-1>::add_to_tensor(e,t);
  };
};

template<typename E, typename T>
struct add_to_tensor_t<E,T,0>{
  static inline __attribute__ ((always_inline)) void add_to_tensor( E const &e, T &t){
    constexpr size_t gen_index = T::property_t::symmetry_t::template index_to_generic<0>::value;
    t.template cc<gen_index>() += e.template evaluate<gen_index>();
  };
};

template<typename E, typename T, size_t N>
struct subtract_from_tensor_t{
  static inline __attribute__ ((always_inline)) void subtract_from_tensor( E const &e, T &t){
    constexpr size_t gen_index = T::property_t::symmetry_t::template index_to_generic<N>::value;
    t.template cc<gen_index>() -= e.template evaluate<gen_index>();
    subtract_from_tensor_t<E,T,N-1>::subtract_from_tensor(e,t);
  };
};

template<typename E, typename T>
struct subtract_from_tensor_t<E,T,0>{
  static inline __attribute__ ((always_inline)) void subtract_from_tensor( E const &e, T &t){
    constexpr size_t gen_index = T::property_t::symmetry_t::template index_to_generic<0>::value;
    t.template cc<gen_index>() -= e.template evaluate<gen_index>();
  };
};

template<typename T, size_t N>
struct multiply_tensor_with_t{
  static inline __attribute__ ((always_inline)) void multiply_tensor_with( typename T::property_t::data_t const &e, T &t){
    constexpr size_t gen_index = T::property_t::symmetry_t::template index_to_generic<N>::value;
    t.template cc<gen_index>() *= e;
    multiply_tensor_with_t<T,N-1>::multiply_tensor_with(e,t);
  };
};

template<typename T>
struct multiply_tensor_with_t<T,0>{
  static inline __attribute__ ((always_inline)) void multiply_tensor_with( typename T::property_t::data_t const &e, T &t){
    constexpr size_t gen_index = T::property_t::symmetry_t::template index_to_generic<0>::value;
    t.template cc<gen_index>() *= e;
  };
};





// Template recursion to compute sum of squares of components
// T should be a explicit tensor, because of a lot of evaluations
template<typename T, size_t N>
struct sum_squares {
  static inline __attribute__ ((always_inline)) typename T::data_t sum(T const &t) {
    constexpr size_t gen_index = T::symmetry_t::template index_to_generic<N>::value;
    return t.template evaluate<gen_index>()
         * t.template evaluate<gen_index>()
         + sum_squares<T,N-1>::sum(t);
  }
};
template<typename T>
struct sum_squares<T,0> {
  static inline __attribute__ ((always_inline)) typename T::data_t sum(T const &t) {
    return t.template evaluate<0>() * t.template evaluate<0>();
  }
};

} // namespace tensors

#endif
