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

#ifndef TENSORS_METRIC_HH
#define TENSORS_METRIC_HH

#include <array>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <utility>

#include "tensor.hh"
#include "tensor_defs.hh"
#include "tensor_expressions.hh"
#include "utilities.hh"

namespace tensors {

template <typename data_t, size_t ndim > class metric_t {
public:
  using metric_tensor_t =
      general_tensor_t<data_t, any_frame_t, 2, std::tuple<lower_t, lower_t>, ndim>;
  using invmetric_tensor_t =
      general_tensor_t<data_t, any_frame_t, 2, std::tuple<upper_t, upper_t>, ndim>;
  using shift_t = general_tensor_t<data_t, eulerian_t, 1, std::tuple<upper_t>, 3>; //Note that we are assume a 3+1 split where dim(shift) = 3 always!


  metric_tensor_t metric;
  data_t lapse;
  shift_t shift;
 //The order here matters!
 // Note that variables are initialised in the constructor according to the
 // order of their declaration!
  data_t sqrtdet; //Always sqrtdet3 
  invmetric_tensor_t invmetric;

private:
  // Constructors...
  inline void compute_inverse_metric(){
      static_assert(false, "Not implemented for this dimension. Only 3 and 4 metric
	supported for now");
  };

  //NOTE: This has to be stored in sqrtdet! And the derivative has to be taken separately
  inline void compute_det(){
      static_assert(false, "Not implemented for this dimension. Only 3 and 4 metric
	supported for now");
  };
  
  static constexpr data_t SQ(data_t& x) {return x*x};


public:
  

  //! Move constructors
  metric_t(data_t lapse_, shift_t&&  shift_, metric_tensor_t&& metric_, invmetric_tensor_t&& invmetric_,
           data_t sqrtdet3) : lapse(lapse_), metric(std::move(metric_)),
  	   invmetric(std::move(invmetric_)), sqrtdet(sqrtdet3), shift(std::move(shift_)) {};

  metric_t(data_t lapse_, shift_t&&  shift_, metric_tensor_t&& metric_ ) : lapse(lapse_), metric(std::move(metric_)),
  	   shift(std::move(shift_)) {
	  
        compute_det(); //sqrtdet now stores det!!
	compute_inverse_metric();
        //Note also that sqrt(g) = lapse * sqrt(gamma)!
	//But for consistency we store only sqrt(gamma) here!
	sqrtdet=sqrt(sqrtdet); //Now we fix this.
};
  
  //TODO Do we need more constructors?

  //! Raise index
  template <size_t i = 0, typename E>
  decltype(auto) const inline raise_index(E const &v) {
    return metric_contraction_t<i>(invmetric, v);
  };

  //! Lower index
  template <size_t i = 0, typename E>
  decltype(auto) const inline lower_index(E const &v) {
    return metric_contraction_t<i>(metric, v);
  };

  // FIXME maybe we should change the name here, since we also have a
  // metric_contraction_t
  //      for raising and lowering now!
  template <size_t i1, size_t i2, typename ind1, typename ind2, typename E1,
            typename E2>
  class metric_contraction {};

  template <size_t i1, size_t i2, typename E1, typename E2>
  class metric_contraction<lower_t, lower_t> {

    inline decltype(auto) const contract(metric_t const &m, E1 const &u,
                                         E2 const &v) {
      return ::contract<i1, i2>(u, m.raise_index<i2>(v));
    };
  };

  template <size_t i1, size_t i2, typename E1, typename E2>
  class metric_contraction<upper_t, lower_t> {

    inline decltype(auto) const contract(metric_t const &m, E1 const &u,
                                         E2 const &v) {
      return ::contract<i1, i2>(u, v);
    };
  };

  template <size_t i1, size_t i2, typename E1, typename E2>
  class metric_contraction<lower_t, upper_t> {

    inline decltype(auto) const contract(metric_t const &m, E1 const &u,
                                         E2 const &v) {
      return ::contract<i1, i2>(u, v);
    };
  };

  template <size_t i1, size_t i2, typename E1, typename E2>
  class metric_contraction<upper_t, upper_t> {

    inline decltype(auto) const contract(metric_t const &m, E1 const &u,
                                         E2 const &v) {
      return ::contract<i1, i2>(u, m.lower_index<i2>(v));
    };
  };

  template <size_t i1, size_t i2, typename E1, typename E2>
  inline decltype(auto) const contract(E1 const &u, E2 const &v) {
    return metric_contraction<
        i1, i2,
        typename std::tuple_element<i1, typename E1::property_t::index_t>::type,
        typename std::tuple_element<
            i1, typename E2::property_t::index_t>::type>::contract(*this, u, v);
  };
};




//////////////////////////////////////////////////////////////////////////////////////////////////////
//                    
//                Specialization to three and four dimensions!
//
/////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename data_t>
inline void metric<data_t,3>::compute_det(){

  constexpr size_t GXX = metric_t::compressed_index<0,0>();
  constexpr size_t GXY = metric_t::compressed_index<0,1>();
  constexpr size_t GXZ = metric_t::compressed_index<0,2>();
  constexpr size_t GYY = metric_t::compressed_index<1,1>();
  constexpr size_t GYZ = metric_t::compressed_index<1,2>();
  constexpr size_t GZZ = metric_t::compressed_index<2,2>();


    //   We are deliberately storing det in sqrtdet and take the square-root later in the initialisation
    sqrtdet= -SQ(metric.evaluate<GXZ>()) * metric.evaluate<GYY>() +
           2.0 * metric.evaluate<GXY>() * metric.evaluate<GXZ>() * metric.evaluate<GYZ>() -
           metric.evaluate<GXX>() * SQ(metric.evaluate<GYZ>()) - SQ(metric.evaluate<GXY>()) * metric.evaluate<GZZ>() +
           metric.evaluate<GXX>() * metric.evaluate<GYY>() * metric.evaluate<GZZ>();
  };

template<typename data_t>
inline void metric_t<data_t,4>::compute_det(){

  constexpr size_t GXX = metric_t::compressed_index<1,1>();
  constexpr size_t GXY = metric_t::compressed_index<1,2>();
  constexpr size_t GXZ = metric_t::compressed_index<1,3>();
  constexpr size_t GYY = metric_t::compressed_index<2,2>();
  constexpr size_t GYZ = metric_t::compressed_index<2,3>();
  constexpr size_t GZZ = metric_t::compressed_index<3,3>();



    //   We are deliberately storing det in sqrtdet and take the square-root later in the initialisation
    sqrtdet= ( -SQ(metric.evaluate<GXZ>()) * metric.evaluate<GYY>() +
           2.0 * metric.evaluate<GXY>() * metric.evaluate<GXZ>() * metric.evaluate<GYZ>() -
           metric.evaluate<GXX>() * SQ(metric.evaluate<GYZ>()) - SQ(metric.evaluate<GXY>()) * metric.evaluate<GZZ>() +
           metric.evaluate<GXX>() * metric.evaluate<GYY>() * metric.evaluate<GZZ>());
  };


template<typename data_t>
inline void metric_t<data_t,3>::compute_inv_metric(){

  constexpr size_t GXX = metric_t::compressed_index<0,0>();
  constexpr size_t GXY = metric_t::compressed_index<0,1>();
  constexpr size_t GXZ = metric_t::compressed_index<0,2>();
  constexpr size_t GYX = metric_t::compressed_index<1,0>();
  constexpr size_t GYY = metric_t::compressed_index<1,1>();
  constexpr size_t GYZ = metric_t::compressed_index<1,2>();
  constexpr size_t GZX = metric_t::compressed_index<2,0>();
  constexpr size_t GZY = metric_t::compressed_index<2,1>();
  constexpr size_t GZZ = metric_t::compressed_index<2,2>();

    //IMPORTANT:   We are deliberately storing det in sqrtdet and take the square-root later in the initialisation
    invmetric[GXX] = (-SQ(metric.evaluate<GYZ>()) + metric.evaluate<GYY>() * metric.evaluate<GZZ>()) / sqrtdet;
    invmetric[GXY] = ((metric.evaluate<GYZ>() * metric.evaluate<GXZ>()) - metric.evaluate<GXY>() * metric.evaluate<GZZ>()) / sqrtdet;
    invmetric[GYY] = (-SQ(metric.evaluate<GXZ>()) + metric.evaluate<GXX>() * metric.evaluate<GZZ>()) / sqrtdet;
    invmetric[GXZ] =
        (-(metric.evaluate<GXZ>() * metric.evaluate<GYY>()) + metric.evaluate<GXY>() * metric.evaluate<GYZ>()) / sqrtdet;
    invmetric[GYZ] = ((metric.evaluate<GXY>() * metric.evaluate<GXZ>()) - metric.evaluate<GXX>() * metric.evaluate<GYZ>()) / sqrtdet;
    invmetric[GZZ] = (-SQ(metric.evaluate<GXY>()) + metric.evaluate<GXX>() * metric.evaluate<GYY>()) / sqrtdet;

    //Symmetrize
    invmetric[GYX] = invmetric.evaluate<GXY>();
    invmetric[GZX] = invmetric.evaluate<GXZ>();
    invmetric[GZY] = invmetric.evaluate<GYZ>();

};

template<typename data_t>
inline void metric_t<data_t,3>::compute_inv_metric(){

  constexpr size_t GTT = metric_t::compressed_index<0,0>();
  constexpr size_t GTX = metric_t::compressed_index<0,1>();
  constexpr size_t GTY = metric_t::compressed_index<0,2>();
  constexpr size_t GTZ = metric_t::compressed_index<0,3>();
  constexpr size_t GXT = metric_t::compressed_index<1,0>();
  constexpr size_t GYT = metric_t::compressed_index<2,0>();
  constexpr size_t GZT = metric_t::compressed_index<3,0>();

  constexpr size_t GXX = metric_t::compressed_index<1,1>();
  constexpr size_t GXY = metric_t::compressed_index<1,2>();
  constexpr size_t GXZ = metric_t::compressed_index<1,3>();
  constexpr size_t GYX = metric_t::compressed_index<2,1>();
  constexpr size_t GYY = metric_t::compressed_index<2,2>();
  constexpr size_t GYZ = metric_t::compressed_index<2,3>();
  constexpr size_t GZX = metric_t::compressed_index<3,1>();
  constexpr size_t GZY = metric_t::compressed_index<3,2>();
  constexpr size_t GZZ = metric_t::compressed_index<3,3>();

 
    invmetric[GTT] = -1./SQ(lapse);
    invmetric[GTX] = -invmetric.evaluate<GTT>()*shift.evaluate<0>();
    invmetric[GTY] = -invmetric.evaluate<GTT>()*shift.evaluate<1>();
    invmetric[GTZ] = -invmetric.evaluate<GTT>()*shift.evaluate<2>();

    //IMPORTANT:   We are deliberately storing det in sqrtdet and take the square-root later in the initialisation
    invmetric[GXX] = (-SQ(metric.evaluate<GYZ>()) + metric.evaluate<GYY>() * metric.evaluate<GZZ>()) / sqrtdet;
    invmetric[GXY] = ((metric.evaluate<GYZ>() * metric.evaluate<GXZ>()) - metric.evaluate<GXY>() * metric.evaluate<GZZ>()) / sqrtdet;
    invmetric[GYY] = (-SQ(metric.evaluate<GXZ>()) + metric.evaluate<GXX>() * metric.evaluate<GZZ>()) / sqrtdet;
    invmetric[GXZ] =
        (-(metric.evaluate<GXZ>() * metric.evaluate<GYY>()) + metric.evaluate<GXY>() * metric.evaluate<GYZ>()) / sqrtdet;
    invmetric[GYZ] = ((metric.evaluate<GXY>() * metric.evaluate<GXZ>()) - metric.evaluate<GXX>() * metric.evaluate<GYZ>()) / sqrtdet;
    invmetric[GZZ] = (-SQ(metric.evaluate<GXY>()) + metric.evaluate<GXX>() * metric.evaluate<GYY>()) / sqrtdet;


    invmetric[GXX] += invmetric[GTT]*shift.evaluate<0>()*shift.evaluate<0>();
    invmetric[GXY] += invmetric[GTT]*shift.evaluate<0>()*shift.evaluate<1>();
    invmetric[GXZ] += invmetric[GTT]*shift.evaluate<0>()*shift.evaluate<2>();
    invmetric[GYY] += invmetric[GTT]*shift.evaluate<1>()*shift.evaluate<1>();
    invmetric[GYZ] += invmetric[GTT]*shift.evaluate<1>()*shift.evaluate<2>();
    invmetric[GZZ] += invmetric[GTT]*shift.evaluate<2>()*shift.evaluate<2>();



    //Symmetrize

    invmetric[GXT] = invmetric.evaluate<GTX>();
    invmetric[GYT] = invmetric.evaluate<GTY>();
    invmetric[GZT] = invmetric.evaluate<GTZ>();

    invmetric[GYX] = invmetric.evaluate<GXY>();
    invmetric[GZX] = invmetric.evaluate<GXZ>();
    invmetric[GZY] = invmetric.evaluate<GYZ>();


};

} // namespace tensors

#endif
