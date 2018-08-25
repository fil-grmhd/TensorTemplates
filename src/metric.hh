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

#ifndef TENSORS_METRIC_HH
#define TENSORS_METRIC_HH

namespace tensors {
namespace general {

template <typename T>
class metric_tensor3_t
    // this is a symmetric three dimensional tensor of rank 2 of lower indices
    : public sym2_tensor_t<T, 3, any_frame_t, 0, 1, lower_t, lower_t> {
public:
  //! Data type
  using data_t = T;

  //! This tensor type, symmetric in both indices
  using this_tensor_t = sym2_tensor_t<T, 3, any_frame_t, 0, 1, lower_t, lower_t>;

  static constexpr size_t ndof = this_tensor_t::symmetry_t::ndof;

  using property_t = typename this_tensor_t::property_t;

  //! Constructor from tensor expression given a index sequence
  //! Generates components from arbitrary tensor expression type (e.g. chained
  //! ones)

  using this_tensor_t::this_tensor_t;

  template<size_t ndim_, typename... ranks>
  using general_metric_t = sym2_tensor_t<T, ndim_, any_frame_t, 0, 1, ranks...>;

  //! All important metric types
  using metric4_t = general_metric_t<4, lower_t, lower_t>;
  using inv_metric4_t = general_metric_t<4, upper_t, upper_t>;

  inline __attribute__ ((always_inline)) static constexpr data_t SQ(data_t const & x) {return x*x;};

  //! Computes the determinant of this 3-metric
  inline __attribute__ ((always_inline)) data_t det() const {

    constexpr size_t GXX = this_tensor_t::template compressed_index<0,0>();
    constexpr size_t GXY = this_tensor_t::template compressed_index<0,1>();
    constexpr size_t GXZ = this_tensor_t::template compressed_index<0,2>();
    constexpr size_t GYY = this_tensor_t::template compressed_index<1,1>();
    constexpr size_t GYZ = this_tensor_t::template compressed_index<1,2>();
    constexpr size_t GZZ = this_tensor_t::template compressed_index<2,2>();

    return - this->SQ(this->template evaluate<GXZ>()) * this->template evaluate<GYY>()
           + 2.0 * this->template evaluate<GXY>() * this->template evaluate<GXZ>() * this->template evaluate<GYZ>()
           - this->template evaluate<GXX>() * this->SQ(this->template evaluate<GYZ>())
           - this->SQ(this->template evaluate<GXY>()) * this->template evaluate<GZZ>()
           + this->template evaluate<GXX>() * this->template evaluate<GYY>() * this->template evaluate<GZZ>();
  }

  inline __attribute__ ((always_inline)) data_t sqrt_det() const {
    return std::sqrt(this->det());
  }

  //! Computes the inverse of this 3-metric
  inline __attribute__ ((always_inline)) decltype(auto) inverse(data_t const det) const {
    constexpr size_t GXX = this_tensor_t::template compressed_index<0,0>();
    constexpr size_t GXY = this_tensor_t::template compressed_index<0,1>();
    constexpr size_t GXZ = this_tensor_t::template compressed_index<0,2>();
    constexpr size_t GYY = this_tensor_t::template compressed_index<1,1>();
    constexpr size_t GYZ = this_tensor_t::template compressed_index<1,2>();
    constexpr size_t GZZ = this_tensor_t::template compressed_index<2,2>();

    // inverse has upper indices
    inv_metric_tensor3_t<T> inverse_metric;

    inverse_metric.template cc<GXX>() = (-this->SQ(this->template evaluate<GYZ>())
                                      + this->template evaluate<GYY>() * this->template evaluate<GZZ>()) / det;
    inverse_metric.template cc<GXY>() = ((this->template evaluate<GYZ>() * this->template evaluate<GXZ>())
                                      - this->template evaluate<GXY>() * this->template evaluate<GZZ>()) / det;
    inverse_metric.template cc<GYY>() = (-this->SQ(this->template evaluate<GXZ>())
                                      + this->template evaluate<GXX>() * this->template evaluate<GZZ>()) / det;
    inverse_metric.template cc<GXZ>() = (-(this->template evaluate<GXZ>() * this->template evaluate<GYY>())
                                      + this->template evaluate<GXY>() * this->template evaluate<GYZ>()) / det;
    inverse_metric.template cc<GYZ>() = ((this->template evaluate<GXY>() * this->template evaluate<GXZ>())
                                      - this->template evaluate<GXX>() * this->template evaluate<GYZ>()) / det;
    inverse_metric.template cc<GZZ>() = (-this->SQ(this->template evaluate<GXY>())
                                      + this->template evaluate<GXX>() * this->template evaluate<GYY>()) / det;

    return inverse_metric;
  }

  //! Computes the inverse of this 3-metric
  inline __attribute__ ((always_inline)) decltype(auto) inverse() const {
    return this->inverse(this->det());
  }

  //! Computes the 4-metric from this 3-metric
  inline __attribute__ ((always_inline)) decltype(auto)
  spacetime_metric(data_t const alpha,
                   vector3_t<data_t> const & beta,
                   covector3_t<data_t> const & beta_lower) const {
    // 4-metric
    metric4_t g;

    //beta low
    g.template set<0,-2>(beta_lower);
    // - alp^2 + beta_i beta^i
    g.template c<0,0>() = - this->SQ(alpha) + contract(beta_lower, beta);

    // g_ij = gamma_ij
    g.template set<-2,-2>(*this);

    return g;
  }

  //! Computes the 4-metric from this 3-metric
  inline __attribute__ ((always_inline)) decltype(auto)
  spacetime_metric(data_t const alpha,
                   vector3_t<data_t> const & beta) const {
    // lower index of beta
    auto beta_lower = contract(*this,beta);
    // pass both to general routine
    return this->spacetime_metric(alpha,beta,beta_lower);
  }

  //! Computes the inverse 4-metric from lapse, shift, the 3-metric and det(gamma)
  inline __attribute__ ((always_inline)) decltype(auto)
  inverse_spacetime_metric(data_t const alpha,
                           vector3_t<data_t> const & beta,
                            data_t const det) const {

    // inverse 4-metric
    inv_metric4_t inv_g;

    // indices wrt to four dimensions
    constexpr size_t GTT = inv_metric4_t::template compressed_index<0,0>();
    constexpr size_t GTX = inv_metric4_t::template compressed_index<0,1>();
    constexpr size_t GTY = inv_metric4_t::template compressed_index<0,2>();
    constexpr size_t GTZ = inv_metric4_t::template compressed_index<0,3>();

    constexpr size_t GXX = inv_metric4_t::template compressed_index<1,1>();
    constexpr size_t GXY = inv_metric4_t::template compressed_index<1,2>();
    constexpr size_t GXZ = inv_metric4_t::template compressed_index<1,3>();
    constexpr size_t GYY = inv_metric4_t::template compressed_index<2,2>();
    constexpr size_t GYZ = inv_metric4_t::template compressed_index<2,3>();
    constexpr size_t GZZ = inv_metric4_t::template compressed_index<3,3>();

    // indices wrt to three dimensions
    constexpr size_t XX3 = this_tensor_t::template compressed_index<0,0>();
    constexpr size_t XY3 = this_tensor_t::template compressed_index<0,1>();
    constexpr size_t XZ3 = this_tensor_t::template compressed_index<0,2>();
    constexpr size_t YY3 = this_tensor_t::template compressed_index<1,1>();
    constexpr size_t YZ3 = this_tensor_t::template compressed_index<1,2>();
    constexpr size_t ZZ3 = this_tensor_t::template compressed_index<2,2>();


    inv_g.template cc<GTT>() = -1./this->SQ(alpha);
    inv_g.template set<0,-2>( - inv_g.template cc<GTT>() * beta);


    inv_g.template cc<GXX>() = (-this->SQ(this->template evaluate<YZ3>())
                             + this->template evaluate<YY3>() * this->template evaluate<ZZ3>()) / det;
    inv_g.template cc<GXY>() = ((this->template evaluate<YZ3>() * this->template evaluate<XZ3>())
                             - this->template evaluate<XY3>() * this->template evaluate<ZZ3>()) / det;
    inv_g.template cc<GYY>() = (-this->SQ(this->template evaluate<XZ3>())
                             + this->template evaluate<XX3>() * this->template evaluate<ZZ3>()) / det;
    inv_g.template cc<GXZ>() = (-(this->template evaluate<XZ3>() * this->template evaluate<YY3>())
                             + this->template evaluate<XY3>() * this->template evaluate<YZ3>()) / det;
    inv_g.template cc<GYZ>() = ((this->template evaluate<XY3>() * this->template evaluate<XZ3>())
                             - this->template evaluate<XX3>() * this->template evaluate<YZ3>()) / det;
    inv_g.template cc<GZZ>() = (-this->SQ(this->template evaluate<XY3>())
                             + this->template evaluate<XX3>() * this->template evaluate<YY3>()) / det;

  // FIXME Doesn't work yet...
  //assign_slice<-2,-2>(inv_g) += inv_g.template c<0,0>()*sym2_cast(tensor_cat(shift,shift));


    inv_g.template cc<GXX>() += inv_g.template cc<GTT>()
                              * beta.template evaluate<0>()
                              * beta.template evaluate<0>();
    inv_g.template cc<GXY>() += inv_g.template cc<GTT>()
                              * beta.template evaluate<0>()
                              * beta.template evaluate<1>();
    inv_g.template cc<GXZ>() += inv_g.template cc<GTT>()
                              * beta.template evaluate<0>()
                              * beta.template evaluate<2>();
    inv_g.template cc<GYY>() += inv_g.template cc<GTT>()
                              * beta.template evaluate<1>()
                              * beta.template evaluate<1>();
    inv_g.template cc<GYZ>() += inv_g.template cc<GTT>()
                              * beta.template evaluate<1>()
                              * beta.template evaluate<2>();
    inv_g.template cc<GZZ>() += inv_g.template cc<GTT>()
                              * beta.template evaluate<2>()
                              * beta.template evaluate<2>();

    return inv_g;
  }

  //! Computes the inverse 4-metric from lapse, shift, the 3-metric and det(gamma)
  inline __attribute__ ((always_inline)) decltype(auto)
  inverse_spacetime_metric(data_t const alpha,
                           vector3_t<data_t> const & beta) const {
    return this->inverse_spacetime_metric(alpha,beta,this->det());
  }

  //! Computes tensor with lowered index
  template<size_t index, typename E>
  inline __attribute__ ((always_inline)) decltype(auto) lower(E const & u) const {
    return metric_contraction_t<index,this_tensor_t,E>(*this, u);
  }
};

template <typename T>
class inv_metric_tensor3_t
    // this is a symmetric three dimensional tensor of rank 2 of upper indices
    : public sym2_tensor_t<T, 3, any_frame_t, 0, 1, upper_t, upper_t> {
public:
  //! Data type
  using data_t = T;

  //! This tensor type, symmetric in both indices
  using this_tensor_t = sym2_tensor_t<T, 3, any_frame_t, 0, 1, upper_t, upper_t>;

  static constexpr size_t ndof = this_tensor_t::symmetry_t::ndof;

  using property_t = typename this_tensor_t::property_t;

  //! Constructor from tensor expression given a index sequence
  //! Generates components from arbitrary tensor expression type (e.g. chained
  //! ones)

  using this_tensor_t::this_tensor_t;

  template<size_t ndim_, typename... ranks>
  using general_metric_t = sym2_tensor_t<T, ndim_, any_frame_t, 0, 1, ranks...>;

  //! Computes tensor with raised index
  template<size_t index, typename E>
  inline __attribute__ ((always_inline)) decltype(auto) raise(E const & u) const {
    return metric_contraction_t<index,this_tensor_t,E>(*this, u);
  }
};

} // namespace general
} // namespace tensors

#endif
