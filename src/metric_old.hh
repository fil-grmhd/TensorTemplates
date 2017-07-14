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

namespace tensors {

//! Raise index
template <size_t i, typename Tmetric, typename E>
decltype(auto) inline __attribute__ ((always_inline)) raise_index(Tmetric const& metric_, E const &v) {
  return metric_contraction_t<i,typename Tmetric::internal_im_t, E>(metric_.invmetric, v);
};

//! Lower index
template <size_t i, typename Tmetric, typename E>
decltype(auto) inline __attribute__ ((always_inline)) lower_index(Tmetric const& metric_, E const &v) {
  return metric_contraction_t<i,typename Tmetric::internal_m_t, E>(metric_.metric, v);
};

//! Contract indices
template <size_t i1, size_t i2, typename Tmetric, typename E1, typename E2>
inline __attribute__ ((always_inline)) decltype(auto) contract( Tmetric const & metric_, E1 const &u, E2 const &v) {
    return Tmetric::template metric_contraction<
        i1, i2,
        typename std::tuple_element<i1, typename E1::property_t::index_t>::type,
        typename std::tuple_element<i1, typename E2::property_t::index_t>::type,
	      E1,E2>::contract(metric_, u, v);
};



template <typename m_tensor_t,typename shift_t,  typename data_t, size_t ndim, typename dim_specialization_t> class metric_t {
public:
  using internal_im_t = invmetric_tensor_t<data_t,ndim>;
  using internal_m_t = m_tensor_t;

  m_tensor_t metric;
  data_t lapse;
  shift_t shift;
  data_t sqrtdet;
  internal_im_t invmetric;

  static_assert( std::is_same<typename m_tensor_t::property_t,
      		              typename ::tensors::metric_tensor_t<data_t,ndim>::property_t
			      > ::value, "You can only use metric_types!");

  static_assert( std::is_same<typename shift_t::property_t,
      		              typename vector3_t<data_t>::property_t
			      > ::value, "You can only use a vector3 type for the shift!");


protected:
  static constexpr data_t SQ(data_t const & x) {return x*x;};

private:
  inline __attribute__ ((always_inline)) void compute_inverse_metric(){ return static_cast<dim_specialization_t*>(this)->compute_inverse_metric();};
  //NOTE: This has to be stored in sqrtdet! And the sqrt has to be taken separately
  inline __attribute__ ((always_inline)) data_t compute_det() { return static_cast<dim_specialization_t*>(this)->compute_det();};



public:
  metric_t(data_t lapse_, shift_t&& shift_) : lapse(lapse_), shift(std::move(shift_)) {}
  //! Move constructors
  metric_t(data_t lapse_, shift_t&&  shift_, m_tensor_t&& metric_, internal_im_t&& invmetric_,
           data_t sqrtdet3) : lapse(lapse_), metric(std::move(metric_)),
  	   invmetric(std::move(invmetric_)), sqrtdet(sqrtdet3), shift(std::move(shift_)) {};

 metric_t() = default;

/*
  metric_t(data_t lapse_, shift_t&&  shift_, internal_m_t&& metric_ ) : lapse(lapse_), metric(std::move(metric_)),
  	   shift(std::move(shift_)) {

    sqrtdet = compute_det(); //sqrtdet now stores det!!
	  compute_inverse_metric();
    //Note also that sqrt(g) = lapse * sqrt(gamma)!
	  //But for consistency we store only sqrt(gamma) here!
	  sqrtdet=sqrt(sqrtdet); //Now we fix this.
  };
*/
  // FIXME maybe we should change the name here, since we also have a
  // metric_contraction_t
  //      for raising and lowering now!
  template <size_t i1, size_t i2, typename ind1, typename ind2, typename E1,
            typename E2>
  class metric_contraction {};

  template <size_t i1, size_t i2, typename E1, typename E2>
  class metric_contraction<i1,i2, lower_t, lower_t,E1,E2> {

   public:
    static inline __attribute__ ((always_inline)) decltype(auto) contract(metric_t const &m, E1 const &u,
                                         E2 const &v) {
      return tensors::template contract<i1, i2>(u, raise_index<i2>(m,v));
    };
  };

  template <size_t i1, size_t i2, typename E1, typename E2>
  class metric_contraction<i1,i2,upper_t, lower_t,E1,E2> {

   public:
    static inline __attribute__ ((always_inline)) decltype(auto) contract(metric_t const &m, E1 const &u,
                                         E2 const &v) {
      return tensors::template contract<i1, i2>(u, v);
    };
  };

  template <size_t i1, size_t i2, typename E1, typename E2>
  class metric_contraction<i1,i2, lower_t, upper_t,E1,E2> {

   public:
    static inline __attribute__ ((always_inline)) decltype(auto) contract(metric_t const &m, E1 const &u,
                                         E2 const &v) {
      return tensors::template contract<i1, i2>(u, v);
    };
  };

  template <size_t i1, size_t i2, typename E1, typename E2>
  class metric_contraction<i1,i2,upper_t, upper_t,E1,E2> {

   public:
    static inline __attribute__ ((always_inline)) decltype(auto) contract(metric_t const &m, E1 const &u,
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

template<typename data_t, typename m_tensor_t,typename shift_t>
class metric3_t : public metric_t<m_tensor_t,shift_t,data_t,3,metric3_t<data_t,m_tensor_t, shift_t>>{

public:
  using super = metric_t<m_tensor_t, shift_t, data_t,3,metric3_t<data_t,m_tensor_t, shift_t>>;

  using metric_t<m_tensor_t,shift_t,data_t,3,metric3_t<data_t,m_tensor_t, shift_t>>::metric_t; //Inherit constructors

  using internal_m_t = m_tensor_t;
  using internal_im_t = typename super::internal_im_t;


  metric3_t(data_t lapse_, shift_t&&  shift_, m_tensor_t&& metric_ ) {

    super::lapse = lapse_;
    super::shift = std::move(shift_);
    super::metric = std::move(metric_);
    super::sqrtdet = compute_det(); //sqrtdet now stores det!!
	  compute_inverse_metric();
    //Note also that sqrt(g) = lapse * sqrt(gamma)!
	  //But for consistency we store only sqrt(gamma) here!
	  super::sqrtdet=sqrt(super::sqrtdet); //Now we fix this.
  };


  inline __attribute__ ((always_inline)) void compute_inverse_metric();
  //NOTE: This has to be stored in sqrtdet! And the sqrt has to be taken separately
  inline __attribute__ ((always_inline)) data_t compute_det();
};

template<typename data_t, typename m_tensor_t,typename shift_t>
class metric4_t : public metric_t<metric_tensor_t<data_t,4>,shift_t, data_t,4,metric4_t<data_t, m_tensor_t, shift_t>>{
public:
  using super = metric_t<metric_tensor_t<data_t,4>,shift_t,data_t,4,metric4_t<data_t, m_tensor_t, shift_t>>;
  using internal_im_t = typename super::internal_im_t;
  //Inherit constructors
  using metric_t<metric_tensor_t<data_t,4>,shift_t, data_t,4,metric4_t<data_t, m_tensor_t, shift_t>>::metric_t;

  using internal_m_t = metric_tensor_t<data_t,4>;

  using metric_tensor3_t = m_tensor_t;
  using invmetric_tensor3_t = typename super::internal_im_t;


  decltype(sym2_cast(slice<-2,-2>(super::metric))) metric3;
  decltype(sym2_cast(slice<-2,-2>(super::invmetric) - tensor_cat(super::shift,super::shift))) invmetric3;
// CHECK: This doesnt compile, why?! we are only interested in the tensor type here, so...
//  decltype(sym2_cast(slice<-2,-2>(super::invmetric))) invmetric3;

  inline __attribute__ ((always_inline)) void compute_metric4_from3(metric_tensor3_t& metric3);
  inline __attribute__ ((always_inline)) void compute_inverse_metric();
  //NOTE: This has to be stored in sqrtdet! And the derivative has to be taken separately
  inline __attribute__ ((always_inline)) data_t compute_det();


  static_assert(std::is_same<typename m_tensor_t::property_t,
      		                   typename  ::tensors::metric_tensor_t<data_t,3>::property_t
			      >::value, "You can only use metric3_types!");


  metric4_t(data_t lapse_, shift_t&&  shift_, m_tensor_t&& metric_) :
      super(lapse_,std::move(shift_)), metric3(sym2_cast(slice<-2,-2>(super::metric)))
      , invmetric3(sym2_cast(slice<-2,-2>(super::invmetric) - tensor_cat(super::shift,super::shift))) {
	  compute_metric4_from3(metric_);
    super::sqrtdet=compute_det(); //sqrtdet now stores det!!
	  compute_inverse_metric();
    //Note also that sqrt(g) = lapse * sqrt(gamma)!
	  //But for consistency we store only sqrt(gamma) here!
	  super::sqrtdet=sqrt(super::sqrtdet); //Now we fix this.

  };

  //Declare metric_3 for raising and lowering


};



template<typename data_t, typename m_tensor_t,typename shift_t>
inline __attribute__ ((always_inline)) data_t metric3_t<data_t, m_tensor_t,shift_t>::compute_det(){

  m_tensor_t& metric = super::metric;
  data_t& lapse = super::lapse;
  data_t& sqrtdet =super::sqrtdet;
  internal_im_t& invmetric= super::invmetric;

  // this must be a generic compressed index, because evaluate expects one
  // symmetry transformation is done internally
  constexpr size_t GXX = internal_m_t::template compressed_index<0,0>();
  constexpr size_t GXY = internal_m_t::template compressed_index<0,1>();
  constexpr size_t GXZ = internal_m_t::template compressed_index<0,2>();
  constexpr size_t GYY = internal_m_t::template compressed_index<1,1>();
  constexpr size_t GYZ = internal_m_t::template compressed_index<1,2>();
  constexpr size_t GZZ = internal_m_t::template compressed_index<2,2>();


    //   We are deliberately storing det in sqrtdet and take the square-root later in the initialisation
    return -this->SQ(metric.template evaluate<GXZ>()) * metric.template evaluate<GYY>() +
           2.0 * metric.template evaluate<GXY>() * metric.template evaluate<GXZ>() * metric.template evaluate<GYZ>() -
           metric.template evaluate<GXX>() * this->SQ(metric.template evaluate<GYZ>()) - this->SQ(metric.template evaluate<GXY>()) * metric.template evaluate<GZZ>() +
           metric.template evaluate<GXX>() * metric.template evaluate<GYY>() * metric.template evaluate<GZZ>();
  };

template<typename data_t, typename m_tensor_t,typename shift_t>
inline __attribute__ ((always_inline)) data_t metric4_t<data_t, m_tensor_t,shift_t>::compute_det(){
  internal_m_t& metric = super::metric;
  data_t& lapse = super::lapse;
  data_t& sqrtdet =super::sqrtdet;
  internal_im_t& invmetric= super::invmetric;



  constexpr size_t GXX = internal_m_t::template compressed_index<1,1>();
  constexpr size_t GXY = internal_m_t::template compressed_index<1,2>();
  constexpr size_t GXZ = internal_m_t::template compressed_index<1,3>();
  constexpr size_t GYY = internal_m_t::template compressed_index<2,2>();
  constexpr size_t GYZ = internal_m_t::template compressed_index<2,3>();
  constexpr size_t GZZ = internal_m_t::template compressed_index<3,3>();


    //   We are deliberately storing det in sqrtdet and take the square-root later in the initialisation
    return ( -this->SQ(metric.template evaluate<GXZ>()) * metric.template evaluate<GYY>() +
           2.0 * metric.template evaluate<GXY>() * metric.template evaluate<GXZ>() * metric.template evaluate<GYZ>() -
           metric.template evaluate<GXX>() * this->SQ(metric.template evaluate<GYZ>()) - this->SQ(metric.template evaluate<GXY>()) * metric.template evaluate<GZZ>() +
           metric.template evaluate<GXX>() * metric.template evaluate<GYY>() * metric.template evaluate<GZZ>());
  };


template<typename data_t, typename m_tensor_t,typename shift_t>
inline __attribute__ ((always_inline)) void metric3_t<data_t, m_tensor_t, shift_t >::compute_inverse_metric(){
  internal_m_t& metric = super::metric;
  data_t& lapse = super::lapse;
  data_t& sqrtdet =super::sqrtdet;
  internal_im_t& invmetric= super::invmetric;

  // this must be a generic compressed index, because evaluate expects one
  // symmetry transformation is done internally
  constexpr size_t GXX = m_tensor_t::template compressed_index<0,0>();
  constexpr size_t GXY = m_tensor_t::template compressed_index<0,1>();
  constexpr size_t GXZ = m_tensor_t::template compressed_index<0,2>();
  constexpr size_t GYY = m_tensor_t::template compressed_index<1,1>();
  constexpr size_t GYZ = m_tensor_t::template compressed_index<1,2>();
  constexpr size_t GZZ = m_tensor_t::template compressed_index<2,2>();

    //IMPORTANT:   We are deliberately storing det in sqrtdet and take the square-root later in the initialisation
    invmetric.template cc<GXX>() = (-this->SQ(metric.template evaluate<GYZ>()) + metric.template evaluate<GYY>() * metric.template evaluate<GZZ>()) / sqrtdet;
    invmetric.template cc<GXY>() = ((metric.template evaluate<GYZ>() * metric.template evaluate<GXZ>()) - metric.template evaluate<GXY>() * metric.template evaluate<GZZ>()) / sqrtdet;
    invmetric.template cc<GYY>() = (-this->SQ(metric.template evaluate<GXZ>()) + metric.template evaluate<GXX>() * metric.template evaluate<GZZ>()) / sqrtdet;
    invmetric.template cc<GXZ>() =
        (-(metric.template evaluate<GXZ>() * metric.template evaluate<GYY>()) + metric.template evaluate<GXY>() * metric.template evaluate<GYZ>()) / sqrtdet;
    invmetric.template cc<GYZ>() = ((metric.template evaluate<GXY>() * metric.template evaluate<GXZ>()) - metric.template evaluate<GXX>() * metric.template evaluate<GYZ>()) / sqrtdet;
    invmetric.template cc<GZZ>() = (-this->SQ(metric.template evaluate<GXY>()) + metric.template evaluate<GXX>() * metric.template evaluate<GYY>()) / sqrtdet;

};

// CHECK: formula especially time components
template<typename data_t, typename m_tensor_t,typename shift_t>
inline __attribute__ ((always_inline)) void metric4_t<data_t, m_tensor_t, shift_t>::compute_inverse_metric(){
  internal_m_t& metric = super::metric;
  data_t& lapse = super::lapse;
  data_t& sqrtdet =super::sqrtdet;
  internal_im_t& invmetric= super::invmetric;
  shift_t& shift = super::shift;

  // this must be a generic compressed index, because evaluate expects one
  // symmetry transformation is done internally
  constexpr size_t GTT = internal_m_t::template compressed_index<0,0>();
  constexpr size_t GTX = internal_m_t::template compressed_index<0,1>();
  constexpr size_t GTY = internal_m_t::template compressed_index<0,2>();
  constexpr size_t GTZ = internal_m_t::template compressed_index<0,3>();

  constexpr size_t GXX = internal_m_t::template compressed_index<1,1>();
  constexpr size_t GXY = internal_m_t::template compressed_index<1,2>();
  constexpr size_t GXZ = internal_m_t::template compressed_index<1,3>();
  constexpr size_t GYY = internal_m_t::template compressed_index<2,2>();
  constexpr size_t GYZ = internal_m_t::template compressed_index<2,3>();
  constexpr size_t GZZ = internal_m_t::template compressed_index<3,3>();


  invmetric.template c<0,0>() = -1./this->SQ(lapse);
  invmetric.template set<0,-2>((-invmetric.template c<0,0>())*shift);


  //IMPORTANT:   We are deliberately storing det in sqrtdet and take the square-root later in the initialisation
  invmetric.template cc<GXX>() = (-this->SQ(metric.template evaluate<GYZ>()) + metric.template evaluate<GYY>() * metric.template evaluate<GZZ>()) / sqrtdet;
  invmetric.template cc<GXY>() = ((metric.template evaluate<GYZ>() * metric.template evaluate<GXZ>()) - metric.template evaluate<GXY>() * metric.template evaluate<GZZ>()) / sqrtdet;
  invmetric.template cc<GYY>() = (-this->SQ(metric.template evaluate<GXZ>()) + metric.template evaluate<GXX>() * metric.template evaluate<GZZ>()) / sqrtdet;
  invmetric.template cc<GXZ>() =
      (-(metric.template evaluate<GXZ>() * metric.template evaluate<GYY>()) + metric.template evaluate<GXY>() * metric.template evaluate<GYZ>()) / sqrtdet;
  invmetric.template cc<GYZ>() = ((metric.template evaluate<GXY>() * metric.template evaluate<GXZ>()) - metric.template evaluate<GXX>() * metric.template evaluate<GYZ>()) / sqrtdet;
  invmetric.template cc<GZZ>() = (-this->SQ(metric.template evaluate<GXY>()) + metric.template evaluate<GXX>() * metric.template evaluate<GYY>()) / sqrtdet;

  // FIXME Doesn't work yet...
  //assign_slice<-2,-2>(invmetric) += invmetric.template c<0,0>()*sym2_cast(tensor_cat(shift,shift));


  invmetric.template cc<GXX>() += invmetric.template cc<GTT>()*shift.template evaluate<0>()*shift.template evaluate<0>();
  invmetric.template cc<GXY>() += invmetric.template cc<GTT>()*shift.template evaluate<0>()*shift.template evaluate<1>();
  invmetric.template cc<GXZ>() += invmetric.template cc<GTT>()*shift.template evaluate<0>()*shift.template evaluate<2>();
  invmetric.template cc<GYY>() += invmetric.template cc<GTT>()*shift.template evaluate<1>()*shift.template evaluate<1>();
  invmetric.template cc<GYZ>() += invmetric.template cc<GTT>()*shift.template evaluate<1>()*shift.template evaluate<2>();
  invmetric.template cc<GZZ>() += invmetric.template cc<GTT>()*shift.template evaluate<2>()*shift.template evaluate<2>();


};

template<typename data_t, typename m_tensor_t,typename shift_t>
inline __attribute__ ((always_inline)) void metric4_t<data_t, m_tensor_t,shift_t>::compute_metric4_from3(metric_tensor3_t& metric3){

  internal_m_t& metric = super::metric;
  data_t& lapse = super::lapse;
  data_t& sqrtdet =super::sqrtdet;
  internal_im_t& invmetric= super::invmetric;

  //beta low
  metric.template set<0,-2>( contract(metric3,super::shift));
  // - alp^2 + beta_i beta^i
  metric.template c<0,0>() = - this->SQ(lapse) + contract(slice<0,-2>(metric), super::shift); 

  metric.template set<-2,-2>(metric3);

};

////////////////////////////////////////
//
//   METRIC FACTORIES
//
/////////////////////////////////////////

template<typename data_t, typename shift_t, typename m_tensor_t>
decltype(auto) make_metric3(data_t lapse_, shift_t&&  shift_, m_tensor_t&& metric_ ) {
  	return metric3_t<data_t,m_tensor_t,shift_t>(lapse_,std::move(shift_), std::move(metric_));
}

template<typename data_t, typename shift_t, typename m_tensor_t>
decltype(auto) make_metric4(data_t lapse_, shift_t&&  shift_, m_tensor_t&& metric_ ) {
  	return metric4_t<data_t,m_tensor_t,shift_t>(lapse_,std::move(shift_), std::move(metric_));
}

} // namespace tensors

#endif
