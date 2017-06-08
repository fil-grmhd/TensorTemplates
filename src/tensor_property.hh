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

namespace tensors {

//! Property class holding data and types defining a specific tensor
template<typename E>
class general_tensor_property_t {
  public:

    using data_t = typename E::data_t;
    using frame_t = typename E::frame_t;
    static constexpr size_t ndim = E::ndim;
    static constexpr size_t rank = E::rank;

    using index_t = typename E::index_t;
    using this_tensor_t = typename E::this_tensor_t;
};

//! Property class holding data and types defining arithmetic combination of two tensors of same type
//  These operations don't change the tensor properties, but one has to check for compatibility
template<typename E1, typename E2>
class arithmetic_expression_property_t : public general_tensor_property_t<E1> {
      static_assert(std::is_same<typename E1::property_t::frame_t, typename E2::property_t::frame_t>::value,
                    "Frame types don't match!");

      static_assert(E1::property_t::ndim == E2::property_t::ndim,
                    "Dimensions don't match!");

      static_assert(E1::property_t::rank == E2::property_t::rank,
                    "Ranks don't match!");

      static_assert(std::is_same<typename E1::property_t::data_t, typename E2::property_t::data_t>::value,
                    "Data types don't match!");

      static_assert(utilities::compare_index<typename E1::property_t::index_t, typename E2::property_t::index_t, E1::property_t::rank>(),
                    "Indices do not match!");
};

//! Property class holding data and types defining a scalar operation on one tensor
//  These operations don't change the tensor properties, but one could check scalar data type
template<typename E>
class scalar_expression_property_t : public general_tensor_property_t<E> {
/*
      static_assert(std::is_same<typename E::property_t::data_t, scalar_data_t>::value,
                    "Data types don't match!");
*/
};

//! Property class holding data and types defining a tensor contraction result
template<size_t i1, size_t i2, typename E1, typename E2>
class contraction_property_t {
  public:

    using data_t = typename E1::property_t::data_t;
    using frame_t = typename E1::property_t::frame_t;
    static constexpr size_t ndim = E1::property_t::ndim;
    // two indices are removed by contraction
    static constexpr size_t rank = E1::property_t::rank + E2::property_t::rank -2;

    // static compile-time routine to get index_t
    static inline constexpr decltype(auto) get_index_t(){
      // create index tuples for both tensors
      constexpr auto E1_indices = typename E1::property_t::index_t();
      constexpr auto E2_indices = typename E2::property_t::index_t();

      constexpr size_t E1_size = E1::property_t::rank;
      constexpr size_t E2_size = E2::property_t::rank;

      // create subtuples
      constexpr auto E1_p1 = get_subtuple<E1_size*(i1<1),(i1-1)*(i1>1)>(E1_indices);
      constexpr auto E1_p2 = get_subtuple<i1+1,E1_size-1>(E1_indices);

      constexpr auto E2_p1 = get_subtuple<E2_size*(i2<1),(i2-1)*(i2>1)>(E2_indices);
      constexpr auto E2_p2 = get_subtuple<i2+1,E2_size-1>(E2_indices);

      return std::tuple_cat(E1_p1,E1_p2,E2_p1,E2_p2);
    }

    using index_t = decltype(get_index_t());

    using this_tensor_t = general_tensor_t<data_t,frame_t,rank,index_t,ndim>;

    static_assert(std::is_same<typename E1::property_t::frame_t, typename E2::property_t::frame_t>::value,
                  "Frame types don't match!");

    static_assert(E1::property_t::ndim == E2::property_t::ndim,
                  "Dimensions don't match!");

    static_assert(std::is_same<typename E1::property_t::data_t, typename E2::property_t::data_t>::value,
                  "Data types don't match!");


    static_assert(std::is_same<
                    typename std::conditional<
                      std::is_same<
                        typename std::tuple_element<i1,typename E1::property_t::index_t>::type,
                        lower_t
                      >::value,
                      lower_t,
                      upper_t
                    >::type,
                    typename std::conditional<
                      std::is_same<
                        typename std::tuple_element<i2,typename E2::property_t::index_t>::type,
                        upper_t
                      >::value,
                      lower_t,
                      upper_t
                    >::type
                  >::value,
                  "Ranks don't match! Can only contract covariant with contravariant indices!");

    static_assert(rank == std::tuple_size<index_t>::value ,
                  "Index tuple size != rank, this should not happen");

};


} // namespace tensors

#endif
