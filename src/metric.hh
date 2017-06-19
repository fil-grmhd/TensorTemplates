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

//! Raise index
template <size_t i, typename Tmetric, typename E>
decltype(auto) inline raise_index(Tmetric const& metric_, E const &v) {
  return metric_contraction_t<i,typename Tmetric::invmetric_tensor_t, E>(metric_.invmetric, v);
};

//! Lower index
template <size_t i, typename Tmetric, typename E>
decltype(auto) inline lower_index(Tmetric const& metric_, E const &v) {
  return metric_contraction_t<i,typename Tmetric::metric_tensor_t, E>(metric_.metric, v);
};

//! Contract indices
template <size_t i1, size_t i2, typename Tmetric, typename E1, typename E2>
inline decltype(auto) contract( Tmetric const & metric_, E1 const &u, E2 const &v) {
    return Tmetric::template metric_contraction<
        i1, i2,
        typename std::tuple_element<i1, typename E1::property_t::index_t>::type,
        typename std::tuple_element<
            i1, typename E2::property_t::index_t>::type,
	E1,E2>::contract(metric_, u, v);
};



template <typename data_t, size_t ndim, typename dim_specialization_t> class metric_t {
public:
  using metric_tensor_t =
      general_tensor_t<data_t, any_frame_t, 2, std::tuple<lower_t, lower_t>, ndim>;
  using invmetric_tensor_t =
      general_tensor_t<data_t, any_frame_t, 2, std::tuple<upper_t, upper_t>, ndim>;
  using shift_t = general_tensor_t<data_t, eulerian_t, 1, std::tuple<upper_t>, 3>; //Note that we are assume a 3+1 split where dim(shift) = 3 always!


  metric_tensor_t metric;
  data_t lapse;
  shift_t shift;
  data_t sqrtdet;
  invmetric_tensor_t invmetric;

protected:
  static constexpr data_t SQ(data_t const & x) {return x*x;};

private:
  inline void compute_inverse_metric(){ return static_cast<dim_specialization_t*>(this)->compute_inverse_metric();};
  //NOTE: This has to be stored in sqrtdet! And the sqrt has to be taken separately
  inline data_t compute_det() { return static_cast<dim_specialization_t*>(this)->compute_det();};



public:
  metric_t(data_t lapse_, shift_t&& shift_) : lapse(lapse_), shift(std::move(shift_)) {}
  //! Move constructors
  metric_t(data_t lapse_, shift_t&&  shift_, metric_tensor_t&& metric_, invmetric_tensor_t&& invmetric_,
           data_t sqrtdet3) : lapse(lapse_), metric(std::move(metric_)),
  	   invmetric(std::move(invmetric_)), sqrtdet(sqrtdet3), shift(std::move(shift_)) {};

  metric_t(data_t lapse_, shift_t&&  shift_, metric_tensor_t&& metric_ ) : lapse(lapse_), metric(std::move(metric_)),
  	   shift(std::move(shift_)) {

    sqrtdet = compute_det(); //sqrtdet now stores det!!
	  compute_inverse_metric();
    //Note also that sqrt(g) = lapse * sqrt(gamma)!
	  //But for consistency we store only sqrt(gamma) here!
	  sqrtdet=sqrt(sqrtdet); //Now we fix this.
  };

  // FIXME maybe we should change the name here, since we also have a
  // metric_contraction_t
  //      for raising and lowering now!
  template <size_t i1, size_t i2, typename ind1, typename ind2, typename E1,
            typename E2>
  class metric_contraction {};

  template <size_t i1, size_t i2, typename E1, typename E2>
  class metric_contraction<i1,i2, lower_t, lower_t,E1,E2> {

   public:
    static inline decltype(auto) contract(metric_t const &m, E1 const &u,
                                         E2 const &v) {
      return tensors::template contract<i1, i2>(u, raise_index<i2>(m,v));
    };
  };

  template <size_t i1, size_t i2, typename E1, typename E2>
  class metric_contraction<i1,i2,upper_t, lower_t,E1,E2> {

   public:
    static inline decltype(auto) contract(metric_t const &m, E1 const &u,
                                         E2 const &v) {
      return tensors::template contract<i1, i2>(u, v);
    };
  };

  template <size_t i1, size_t i2, typename E1, typename E2>
  class metric_contraction<i1,i2, lower_t, upper_t,E1,E2> {

   public:
    static inline decltype(auto) contract(metric_t const &m, E1 const &u,
                                         E2 const &v) {
      return tensors::template contract<i1, i2>(u, v);
    };
  };

  template <size_t i1, size_t i2, typename E1, typename E2>
  class metric_contraction<i1,i2,upper_t, upper_t,E1,E2> {

   public:
    static inline decltype(auto) contract(metric_t const &m, E1 const &u,
                                         E2 const &v) {
      return tensors::template contract<i1, i2>(u, lower_index<i2>(m,v));
    };
  };
};



//////////////////////////////////////////////////////////////////////////////////////////////////////
//
//                Specialization to three and four dimensions!
//
/////////////////////////////////////////////////////////////////////////////////////////////////////


//It is not possible to partially specialise a member function, instead we need to partially specialise the class

template<typename data_t>
class metric3_t : public metric_t<data_t,3,metric3_t<data_t>>{

public:
  using metric_t<data_t,3,metric3_t<data_t>>::metric_t; //Inherit constructors

  using invmetric_tensor_t = typename metric_t<data_t,3,metric3_t<data_t>>::invmetric_tensor_t;
  using metric_tensor_t = typename metric_t<data_t,3,metric3_t<data_t>>::metric_tensor_t;
  using shift_t = typename metric_t<data_t,3,metric3_t<data_t>>::shift_t;

  using super = metric_t<data_t,3,metric3_t<data_t>>;

  inline void compute_inverse_metric();
  //NOTE: This has to be stored in sqrtdet! And the sqrt has to be taken separately
  inline data_t compute_det();

/*
  metric3_t(data_t lapse_, shift_t&&  shift_, metric_tensor_t&& metric_ ) : super::lapse(lapse_), super::metric(std::move(metric_)),
  	   super::shift(std::move(shift_)) {

    super::sqrtdet = compute_det(); //sqrtdet now stores det!!
	  compute_inverse_metric();
    //Note also that sqrt(g) = lapse * sqrt(gamma)!
	  //But for consistency we store only sqrt(gamma) here!
	  super::sqrtdet=sqrt(super::sqrtdet); //Now we fix this.
  };
*/
};

template<typename data_t>
class metric4_t : public metric_t<data_t,4,metric4_t<data_t>>{
public:
  using metric_tensor_t = typename metric_t<data_t,4,metric4_t<data_t>>::metric_tensor_t;
  using invmetric_tensor_t = typename metric_t<data_t,4,metric4_t<data_t>>::invmetric_tensor_t;
  using shift_t = typename metric_t<data_t,4,metric4_t<data_t>>::shift_t;

  using metric_tensor3_t =
      general_tensor_t<data_t, any_frame_t, 2, std::tuple<lower_t, lower_t>, 3>;
  using invmetric_tensor3_t =
      general_tensor_t<data_t, any_frame_t, 2, std::tuple<upper_t, upper_t>, 3>;

  using metric_t<data_t,4,metric4_t<data_t>>::metric_t; //Inherit constructors
  using super = metric_t<data_t,4,metric4_t<data_t>>;

  //Additional constructor
//  metric4_t(data_t lapse_, shift_t&&  shift_, metric_tensor3_t&& metric_);

  inline void compute_metric4_from3(metric_tensor3_t& metric3);
  inline void compute_inverse_metric();
  //NOTE: This has to be stored in sqrtdet! And the derivative has to be taken separately
  inline data_t compute_det();

  metric4_t(data_t lapse_, shift_t&&  shift_, metric_tensor3_t&& metric_) : super(lapse_,std::move(shift_)) {
	  compute_metric4_from3(metric_);
    compute_det(); //sqrtdet now stores det!!
	  compute_inverse_metric();
    //Note also that sqrt(g) = lapse * sqrt(gamma)!
	  //But for consistency we store only sqrt(gamma) here!
	  super::sqrtdet=sqrt(super::sqrtdet); //Now we fix this.
};

};



template<typename data_t>
inline data_t metric3_t<data_t>::compute_det(){

  metric_tensor_t& metric = super::metric;
  data_t& lapse = super::lapse;
  shift_t& shift = super::shift;
  data_t& sqrtdet =super::sqrtdet;
  invmetric_tensor_t& invmetric= super::invmetric;

  constexpr size_t GXX = metric_tensor_t::template compressed_index<0,0>();
  constexpr size_t GXY = metric_tensor_t::template compressed_index<0,1>();
  constexpr size_t GXZ = metric_tensor_t::template compressed_index<0,2>();
  constexpr size_t GYY = metric_tensor_t::template compressed_index<1,1>();
  constexpr size_t GYZ = metric_tensor_t::template compressed_index<1,2>();
  constexpr size_t GZZ = metric_tensor_t::template compressed_index<2,2>();


    //   We are deliberately storing det in sqrtdet and take the square-root later in the initialisation
    return -this->SQ(metric.template evaluate<GXZ>()) * metric.template evaluate<GYY>() +
           2.0 * metric.template evaluate<GXY>() * metric.template evaluate<GXZ>() * metric.template evaluate<GYZ>() -
           metric.template evaluate<GXX>() * this->SQ(metric.template evaluate<GYZ>()) - this->SQ(metric.template evaluate<GXY>()) * metric.template evaluate<GZZ>() +
           metric.template evaluate<GXX>() * metric.template evaluate<GYY>() * metric.template evaluate<GZZ>();
  };

template<typename data_t>
inline data_t metric4_t<data_t>::compute_det(){
  metric_tensor_t& metric = super::metric;
  data_t& lapse = super::lapse;
  shift_t& shift = super::shift;
  data_t& sqrtdet =super::sqrtdet;
  invmetric_tensor_t& invmetric= super::invmetric;



  constexpr size_t GXX = metric_tensor_t::template compressed_index<1,1>();
  constexpr size_t GXY = metric_tensor_t::template compressed_index<1,2>();
  constexpr size_t GXZ = metric_tensor_t::template compressed_index<1,3>();
  constexpr size_t GYY = metric_tensor_t::template compressed_index<2,2>();
  constexpr size_t GYZ = metric_tensor_t::template compressed_index<2,3>();
  constexpr size_t GZZ = metric_tensor_t::template compressed_index<3,3>();


    //   We are deliberately storing det in sqrtdet and take the square-root later in the initialisation
    return ( -this->SQ(metric.template evaluate<GXZ>()) * metric.template evaluate<GYY>() +
           2.0 * metric.template evaluate<GXY>() * metric.template evaluate<GXZ>() * metric.template evaluate<GYZ>() -
           metric.template evaluate<GXX>() * this->SQ(metric.template evaluate<GYZ>()) - this->SQ(metric.template evaluate<GXY>()) * metric.template evaluate<GZZ>() +
           metric.template evaluate<GXX>() * metric.template evaluate<GYY>() * metric.template evaluate<GZZ>());
  };


template<typename data_t>
inline void metric3_t<data_t>::compute_inverse_metric(){
  metric_tensor_t& metric = super::metric;
  data_t& lapse = super::lapse;
  shift_t& shift = super::shift;
  data_t& sqrtdet =super::sqrtdet;
  invmetric_tensor_t& invmetric= super::invmetric;

  constexpr size_t GXX = metric_tensor_t::template compressed_index<0,0>();
  constexpr size_t GXY = metric_tensor_t::template compressed_index<0,1>();
  constexpr size_t GXZ = metric_tensor_t::template compressed_index<0,2>();
  constexpr size_t GYX = metric_tensor_t::template compressed_index<1,0>();
  constexpr size_t GYY = metric_tensor_t::template compressed_index<1,1>();
  constexpr size_t GYZ = metric_tensor_t::template compressed_index<1,2>();
  constexpr size_t GZX = metric_tensor_t::template compressed_index<2,0>();
  constexpr size_t GZY = metric_tensor_t::template compressed_index<2,1>();
  constexpr size_t GZZ = metric_tensor_t::template compressed_index<2,2>();

    //IMPORTANT:   We are deliberately storing det in sqrtdet and take the square-root later in the initialisation
    invmetric[GXX] = (-this->SQ(metric.template evaluate<GYZ>()) + metric.template evaluate<GYY>() * metric.template evaluate<GZZ>()) / sqrtdet;
    invmetric[GXY] = ((metric.template evaluate<GYZ>() * metric.template evaluate<GXZ>()) - metric.template evaluate<GXY>() * metric.template evaluate<GZZ>()) / sqrtdet;
    invmetric[GYY] = (-this->SQ(metric.template evaluate<GXZ>()) + metric.template evaluate<GXX>() * metric.template evaluate<GZZ>()) / sqrtdet;
    invmetric[GXZ] =
        (-(metric.template evaluate<GXZ>() * metric.template evaluate<GYY>()) + metric.template evaluate<GXY>() * metric.template evaluate<GYZ>()) / sqrtdet;
    invmetric[GYZ] = ((metric.template evaluate<GXY>() * metric.template evaluate<GXZ>()) - metric.template evaluate<GXX>() * metric.template evaluate<GYZ>()) / sqrtdet;
    invmetric[GZZ] = (-this->SQ(metric.template evaluate<GXY>()) + metric.template evaluate<GXX>() * metric.template evaluate<GYY>()) / sqrtdet;

    //Symmetrize
    invmetric[GYX] = invmetric.template evaluate<GXY>();
    invmetric[GZX] = invmetric.template evaluate<GXZ>();
    invmetric[GZY] = invmetric.template evaluate<GYZ>();

};

// CHECK: formula especially time components
template<typename data_t>
inline void metric4_t<data_t>::compute_inverse_metric(){
  metric_tensor_t& metric = super::metric;
  data_t& lapse = super::lapse;
  shift_t& shift = super::shift;
  data_t& sqrtdet =super::sqrtdet;
  invmetric_tensor_t& invmetric= super::invmetric;

  constexpr size_t GTT = metric_tensor_t::template compressed_index<0,0>();
  constexpr size_t GTX = metric_tensor_t::template compressed_index<0,1>();
  constexpr size_t GTY = metric_tensor_t::template compressed_index<0,2>();
  constexpr size_t GTZ = metric_tensor_t::template compressed_index<0,3>();
  constexpr size_t GXT = metric_tensor_t::template compressed_index<1,0>();
  constexpr size_t GYT = metric_tensor_t::template compressed_index<2,0>();
  constexpr size_t GZT = metric_tensor_t::template compressed_index<3,0>();

  constexpr size_t GXX = metric_tensor_t::template compressed_index<1,1>();
  constexpr size_t GXY = metric_tensor_t::template compressed_index<1,2>();
  constexpr size_t GXZ = metric_tensor_t::template compressed_index<1,3>();
  constexpr size_t GYX = metric_tensor_t::template compressed_index<2,1>();
  constexpr size_t GYY = metric_tensor_t::template compressed_index<2,2>();
  constexpr size_t GYZ = metric_tensor_t::template compressed_index<2,3>();
  constexpr size_t GZX = metric_tensor_t::template compressed_index<3,1>();
  constexpr size_t GZY = metric_tensor_t::template compressed_index<3,2>();
  constexpr size_t GZZ = metric_tensor_t::template compressed_index<3,3>();


  invmetric[GTT] = -1./this->SQ(lapse);
  invmetric[GTX] = -invmetric.template evaluate<GTT>()*shift.template evaluate<0>();
  invmetric[GTY] = -invmetric.template evaluate<GTT>()*shift.template evaluate<1>();
  invmetric[GTZ] = -invmetric.template evaluate<GTT>()*shift.template evaluate<2>();

  //IMPORTANT:   We are deliberately storing det in sqrtdet and take the square-root later in the initialisation
  invmetric[GXX] = (-this->SQ(metric.template evaluate<GYZ>()) + metric.template evaluate<GYY>() * metric.template evaluate<GZZ>()) / sqrtdet;
  invmetric[GXY] = ((metric.template evaluate<GYZ>() * metric.template evaluate<GXZ>()) - metric.template evaluate<GXY>() * metric.template evaluate<GZZ>()) / sqrtdet;
  invmetric[GYY] = (-this->SQ(metric.template evaluate<GXZ>()) + metric.template evaluate<GXX>() * metric.template evaluate<GZZ>()) / sqrtdet;
  invmetric[GXZ] =
      (-(metric.template evaluate<GXZ>() * metric.template evaluate<GYY>()) + metric.template evaluate<GXY>() * metric.template evaluate<GYZ>()) / sqrtdet;
  invmetric[GYZ] = ((metric.template evaluate<GXY>() * metric.template evaluate<GXZ>()) - metric.template evaluate<GXX>() * metric.template evaluate<GYZ>()) / sqrtdet;
  invmetric[GZZ] = (-this->SQ(metric.template evaluate<GXY>()) + metric.template evaluate<GXX>() * metric.template evaluate<GYY>()) / sqrtdet;


  invmetric[GXX] += invmetric[GTT]*shift.template evaluate<0>()*shift.template evaluate<0>();
  invmetric[GXY] += invmetric[GTT]*shift.template evaluate<0>()*shift.template evaluate<1>();
  invmetric[GXZ] += invmetric[GTT]*shift.template evaluate<0>()*shift.template evaluate<2>();
  invmetric[GYY] += invmetric[GTT]*shift.template evaluate<1>()*shift.template evaluate<1>();
  invmetric[GYZ] += invmetric[GTT]*shift.template evaluate<1>()*shift.template evaluate<2>();
  invmetric[GZZ] += invmetric[GTT]*shift.template evaluate<2>()*shift.template evaluate<2>();



  //Symmetrize

  invmetric[GXT] = invmetric.template evaluate<GTX>();
  invmetric[GYT] = invmetric.template evaluate<GTY>();
  invmetric[GZT] = invmetric.template evaluate<GTZ>();

  invmetric[GYX] = invmetric.template evaluate<GXY>();
  invmetric[GZX] = invmetric.template evaluate<GXZ>();
  invmetric[GZY] = invmetric.template evaluate<GYZ>();
};

// CHECK: formula
template<typename data_t>
inline void metric4_t<data_t>::compute_metric4_from3(metric_tensor3_t& metric3){

  metric_tensor_t& metric = super::metric;
  data_t& lapse = super::lapse;
  shift_t& shift = super::shift;
  data_t& sqrtdet =super::sqrtdet;
  invmetric_tensor_t& invmetric= super::invmetric;



  constexpr size_t GTT = metric_tensor_t::template compressed_index<0,0>();
  constexpr size_t GTX = metric_tensor_t::template compressed_index<0,1>();
  constexpr size_t GTY = metric_tensor_t::template compressed_index<0,2>();
  constexpr size_t GTZ = metric_tensor_t::template compressed_index<0,3>();
  constexpr size_t GXT = metric_tensor_t::template compressed_index<1,0>();
  constexpr size_t GYT = metric_tensor_t::template compressed_index<2,0>();
  constexpr size_t GZT = metric_tensor_t::template compressed_index<3,0>();

  constexpr size_t GXX = metric_tensor_t::template compressed_index<1,1>();
  constexpr size_t GXY = metric_tensor_t::template compressed_index<1,2>();
  constexpr size_t GXZ = metric_tensor_t::template compressed_index<1,3>();
  constexpr size_t GYX = metric_tensor_t::template compressed_index<2,1>();
  constexpr size_t GYY = metric_tensor_t::template compressed_index<2,2>();
  constexpr size_t GYZ = metric_tensor_t::template compressed_index<2,3>();
  constexpr size_t GZX = metric_tensor_t::template compressed_index<3,1>();
  constexpr size_t GZY = metric_tensor_t::template compressed_index<3,2>();
  constexpr size_t GZZ = metric_tensor_t::template compressed_index<3,3>();

  //indices of the three metric
  constexpr size_t G3XX = metric_tensor3_t::template compressed_index<0,0>();
  constexpr size_t G3XY = metric_tensor3_t::template compressed_index<0,1>();
  constexpr size_t G3XZ = metric_tensor3_t::template compressed_index<0,2>();
  constexpr size_t G3YX = metric_tensor3_t::template compressed_index<1,0>();
  constexpr size_t G3YY = metric_tensor3_t::template compressed_index<1,1>();
  constexpr size_t G3YZ = metric_tensor3_t::template compressed_index<1,2>();
  constexpr size_t G3ZX = metric_tensor3_t::template compressed_index<2,0>();
  constexpr size_t G3ZY = metric_tensor3_t::template compressed_index<2,1>();
  constexpr size_t G3ZZ = metric_tensor3_t::template compressed_index<2,2>();


  //beta low
  std::array<double,3> beta_low;

  metric[GTX] = shift.template evaluate<0>()*metric3.template evaluate<G3XX>() + shift.template evaluate<1>()*metric3.template evaluate<G3XY>() + shift.template evaluate<2>()*metric3.template evaluate<G3XZ>();
  metric[GTY] = shift.template evaluate<0>()*metric3.template evaluate<G3XY>() + shift.template evaluate<1>()*metric3.template evaluate<G3YY>() + shift.template evaluate<2>()*metric3.template evaluate<G3YZ>();
  metric[GTZ] = shift.template evaluate<0>()*metric3.template evaluate<G3XZ>() + shift.template evaluate<1>()*metric3.template evaluate<G3YZ>() + shift.template evaluate<2>()*metric3.template evaluate<G3ZZ>();

  metric[GTT] = -this->SQ(lapse) + metric.template evaluate<GTX>()*shift.template evaluate<0>() + metric.template evaluate<GTY>()*shift.template evaluate<1>() + metric.template evaluate<GTZ>()*shift.template evaluate<2>();

  metric[GXX] = metric3.template evaluate<G3XX>();
  metric[GXY] = metric3.template evaluate<G3XY>();
  metric[GXZ] = metric3.template evaluate<G3XZ>();
  metric[GYY] = metric3.template evaluate<G3YY>();
  metric[GYZ] = metric3.template evaluate<G3YZ>();
  metric[GZZ] = metric3.template evaluate<G3ZZ>();

  //Symmetrize

  metric[GXT] = metric.template evaluate<GTX>();
  metric[GYT] = metric.template evaluate<GTY>();
  metric[GZT] = metric.template evaluate<GTZ>();

  metric[GYX] = metric.template evaluate<GXY>();
  metric[GZX] = metric.template evaluate<GXZ>();
  metric[GZY] = metric.template evaluate<GYZ>();
};

} // namespace tensors

#endif
