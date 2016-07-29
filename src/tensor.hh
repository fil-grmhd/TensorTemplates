//  TensorTemplates: C++ tensor class templates
//  Copyright (C) 2016, Ludwig Jens Papenfort
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


#ifndef TENSORS_TENSOR_HH
#define TENSORS_TENSOR_HH

#include <cmath>
#include <cctk.h>

#include "utilities.hh"
#include "multiindex.hh"
#include "symmetry_types.hh"
#include "tensor_types.hh"

#include <tensor_contraction.hh>
#include <tensor_trace.hh>
#include <utils.hh>

namespace tensors {

///////////////////////////////////////////////////////////////////////////////
//! Tensor class
///////////////////////////////////////////////////////////////////////////////
/*!
 *  \tparam T data type
 *  \tparam ndim_ number of dimensions
 *  \tparam rank_ tensor rank
 *  \tparam symmetry_t_ tensor symmetry type
 */
template<typename T, size_t ndim_, size_t rank_, typename symmetry_t_>
class general_tensor_t {
    public:
        //! Data type
        typedef T data_t;
        //! This tensor type
        typedef symmetry_t_ symmetry_t;


        //! Rank of the tensor
        static size_t constexpr rank = rank_;
        //! Number of dimensions
        static size_t constexpr ndim = ndim_;
        //! Number of degrees of freedom
        static size_t constexpr ndof = symmetry_t::template ndof<ndim,rank>();


        //! This tensor type
        typedef general_tensor_t<T,ndim,rank,symmetry_t> this_tensor_t;



        //! Computes the compressed index associated with the given indices
        //! Delegates the call to symmetry type
        template<typename... indices_t>
        static inline size_t compress_indices(indices_t... indices) {
          return symmetry_t::template compress_indices<ndim,rank>(indices...);
        }

        //! Computes the compressed index from the given multiindex object
        //! Delegates the call to symmetry type
        static inline size_t compress_indices(multiindex_t<ndim,rank> const & mi) {
          return symmetry_t::template compress_indices<ndim,rank>(mi);
        }

        //! Create a suitable multiindex
        static inline multiindex_t<ndim,rank> get_mi() {
          multiindex_t<ndim,rank> mi;
          return mi;
        }

        //! Create an identity tensor
        static inline this_tensor_t get_id() {
          this_tensor_t id;
          id.set_to_zero();
          auto id_mi = id.get_mi();

          for(size_t i = 0; i<ndim; ++i) {
            for(size_t j = 0; j<rank; ++j) {
              id_mi[j] = i;
            }
            id(id_mi) = 1;
          }
          return id;
        }

        //! Sets tensor to zero for all components
        inline void set_to_zero() {
          for(size_t i = 0; i<ndof; ++i) {
            m_data[i] = 0;
          }
        }

        //! Contract with another tensor and return resulting tensor
        /*!
         *
         *  Implementation can be found in tensor_contraction.hh
         *
         *  \tparam c_index0 contracted index of this tensor
         *  \tparam c_index1 contracted index of the other tensor
         *  \tparam tensor_t tensor type, automatically deduced from argument
         */
        template<size_t c_index0, size_t c_index1, typename tensor_t>
        inline contraction_result_t<this_tensor_t,tensor_t>
               contract(tensor_t const & tensor) const {
            return tensors::contract<c_index0,c_index1>(*this,tensor);
        }

        //! Trace over the two given indices and return resulting tensor
        /*!
         *
         *  Implementation can be found in tensor_trace.hh
         *
         *  \tparam t_index0 first index to trace
         *  \tparam t_index1 second index to trace
         */
        template<size_t t_index0, size_t t_index1>
        inline trace_result_t<this_tensor_t> trace() const {
          return tensors::trace<t_index0,t_index1>(*this);
        }

        //! Returns a lower dimensional sub-tensor
        /*!
         * \tparam max_dim number of dimensions of subtensor
         */
        template<size_t max_dim>
        inline general_tensor_t<T,max_dim,rank,symmetry_t> subtensor() {
          general_tensor_t<T,max_dim,rank,symmetry_t> subtensor;
          auto subtensor_mi = subtensor.get_mi();

          // compute dimensional offset
          size_t index_diff = ndim-max_dim;
          for(auto mi = get_mi(); !mi.end(); ++mi) {
            // copy value to subtensor if all indices are greater
            if(mi > index_diff-1) {
              for(size_t i = 0; i<rank; ++i) {
                subtensor_mi[i] = mi[i]-index_diff;
              }
              subtensor(subtensor_mi) = this->operator()(mi);
            }
          }
          return subtensor;
        }

        //! Compute component-wise difference with tensor of same type
        /*!
         *  Only defined for same tensor type, for now
         */
        inline this_tensor_t operator-(const this_tensor_t& tensor) {
            this_tensor_t diff;
            for(size_t i = 0; i<ndof; ++i) {
              diff[i] = this->operator[](i)-tensor[i];
            }
            return diff;
        }

        //! Compute component-wise sum with tensor of same type
        /*!
         *  Only defined for same tensor type, for now
         */
        inline this_tensor_t operator+(const this_tensor_t& tensor) {
            this_tensor_t sum;
            for(size_t i = 0; i<ndof; ++i) {
              sum[i] = this->operator[](i)+tensor[i];
            }
            return sum;
        }

        //! Compute component-wise scalar multiplication
        template<typename scalar_t_>
        inline this_tensor_t operator*(const scalar_t_ scalar) {
          this_tensor_t result;
          for(auto mi = get_mi(); !mi.end(); ++mi) {
            result(mi) = scalar*this->operator()(mi);
          }
        }

        //! Compute the Euclidean or Frobenius norm
        inline data_t norm() {
            data_t squared_sum = 0;
            for(auto mi = get_mi(); !mi.end(); ++mi) {
              squared_sum += this->operator()(mi)*this->operator()(mi);
            }
            return std::sqrt(squared_sum);
        }

        //! Access the components of a tensor using a compressed index
        /*!
         *  The data is stored in a row major format
         */
        inline T & operator[](size_t const a) {
            return m_data[a];
        }
        //! Access the components of a tensor using a compressed index
        /*!
         *  The data is stored in a row major format
         */
        inline T const & operator[](size_t const a) const {
            return m_data[a];
        }

        //! Access the components of a tensor using the natural indices
        template<typename... indices_t>
        inline T & operator()(size_t const a, indices_t... indices) {
            static_assert(sizeof...(indices)+1 == rank,
                "utils::tensor::tensor: Number of indices must match rank.");

            return this->operator[](compress_indices(a,indices...));
        }
        //! Access the components of a tensor using the natural indices
        template<typename... indices_t>
        inline T const & operator()(size_t const a, indices_t... indices) const {
            static_assert(sizeof...(indices)+1 == rank,
                "utils::tensor::tensor: Number of indices must match rank.");

            return this->operator[](compress_indices(a,indices...));
        }

        //! Access the components of a tenser using multiindex objects
        inline T const & operator()(multiindex_t<ndim,rank> const & mi) const {
            return this->operator[](compress_indices(mi));
        }
        //! Access the components of a tenser using multiindex objects
        inline T & operator()(multiindex_t<ndim,rank> const & mi) {
            return this->operator[](compress_indices(mi));
        }
        //! Comparison routine
        template<int exponent,typename tensor_t>
        inline bool compare_components(tensor_t const & tensor) {
          static_assert(rank == tensor_t::rank,
                        "utils::tensor::tensor: "
                        "Tensor ranks have to match.");
          static_assert(ndim == tensor_t::ndim,
                        "utils::tensor::compare_tensors: "
                        "Tensor dimensions have to match.");
          common_data_t<this_tensor_t,tensor_t> eps = 1.0/utilities::static_pow<10,exponent>::result;

          for(auto mi = get_mi(); !mi.end(); ++mi) {
            if(std::abs(this->operator()(mi) - tensor(mi)) > eps) {
              return false;
            }
          }
          return true;
        }
    private:
        T m_data[ndof];

};


///////////////////////////////////////////////////////////////////////////////
//! Metric tensor
///////////////////////////////////////////////////////////////////////////////
template<size_t ndim>
class metric;

template<size_t ndim>
class inv_metric;

//! Spatial metric
template<>
class metric<3>: public symmetric2<CCTK_REAL, 3, 2> {
    public:
        //! Computes the metric determinant at the given point
        inline CCTK_REAL det() const {
            return utils::metric::spatial_det(
                    this->operator[](0),
                    this->operator[](1),
                    this->operator[](2),
                    this->operator[](3),
                    this->operator[](4),
                    this->operator[](5));
        }
};

//! Inverse of the spatial metric
template<>
class inv_metric<3>: public symmetric2<CCTK_REAL, 3, 2> {
    public:
        //! Constructs the inverse metric from the metric
        inline void from_metric(
                CCTK_REAL const gxx,
                CCTK_REAL const gxy,
                CCTK_REAL const gxz,
                CCTK_REAL const gyy,
                CCTK_REAL const gyz,
                CCTK_REAL const gzz) {
            CCTK_REAL const det = utils::metric::spatial_det(
                    gxx, gxy, gxz, gyy, gyz, gzz);
            CCTK_REAL uxx, uxy, uxz, uyy, uyz, uzz;
            utils::metric::spatial_inv(det,
                    gxx, gxy, gxz, gyy, gyz, gzz,
                    &uxx, &uxy, &uxz, &uyy, &uyz, &uzz);
            this->operator[](0) = uxx;
            this->operator[](1) = uxy;
            this->operator[](2) = uxz;
            this->operator[](3) = uyy;
            this->operator[](4) = uyz;
            this->operator[](5) = uzz;
        }
        //! Constructs the inverse metric from the metric and the spatial det
        inline void from_metric(
                CCTK_REAL const gxx,
                CCTK_REAL const gxy,
                CCTK_REAL const gxz,
                CCTK_REAL const gyy,
                CCTK_REAL const gyz,
                CCTK_REAL const gzz,
                CCTK_REAL const det) {
            CCTK_REAL uxx, uxy, uxz, uyy, uyz, uzz;
            utils::metric::spatial_inv(det,
                    gxx, gxy, gxz, gyy, gyz, gzz,
                    &uxx, &uxy, &uxz, &uyy, &uyz, &uzz);
            this->operator[](0) = uxx;
            this->operator[](1) = uxy;
            this->operator[](2) = uxz;
            this->operator[](3) = uyy;
            this->operator[](4) = uyz;
            this->operator[](5) = uzz;
        }
        //! Constructs the inverse metric from the metric
        inline void from_metric(metric<3> const & g) {
            CCTK_REAL const det = g.det();
            CCTK_REAL uxx, uxy, uxz, uyy, uyz, uzz;
            utils::metric::spatial_inv(det,
                    g[0], g[1], g[2], g[3], g[4], g[5],
                    &uxx, &uxy, &uxz, &uyy, &uyz, &uzz);
            this->operator[](0) = uxx;
            this->operator[](1) = uxy;
            this->operator[](2) = uxz;
            this->operator[](3) = uyy;
            this->operator[](4) = uyz;
            this->operator[](5) = uzz;
        }
        //! Constructs the inverse metric from the metric and the spatial det
        inline void from_metric_det(metric<3> const & g, CCTK_REAL const det) {
            CCTK_REAL uxx, uxy, uxz, uyy, uyz, uzz;
            utils::metric::spatial_inv(det,
                    g[0], g[1], g[2], g[3], g[4], g[5],
                    &uxx, &uxy, &uxz, &uyy, &uyz, &uzz);
            this->operator[](0) = uxx;
            this->operator[](1) = uxy;
            this->operator[](2) = uxz;
            this->operator[](3) = uyy;
            this->operator[](4) = uyz;
            this->operator[](5) = uzz;
        }
};

//! Spacetime metric
template<>
class metric<4>: public symmetric2<CCTK_REAL, 4, 2> {
    public:
        //! Construct the spacetime metric from the ADM quantities
        inline void from_adm(
                CCTK_REAL const alp,
                CCTK_REAL const betax,
                CCTK_REAL const betay,
                CCTK_REAL const betaz,
                CCTK_REAL const gxx,
                CCTK_REAL const gxy,
                CCTK_REAL const gxz,
                CCTK_REAL const gyy,
                CCTK_REAL const gyz,
                CCTK_REAL const gzz) {
            CCTK_REAL g[16];
            utils::metric::spacetime(alp, betax, betay, betaz, gxx, gxy, gxz,
                    gyy, gyz, gzz, &g[0]);
            for(int a = 0; a < 4; ++a)
            for(int b = a; b < 4; ++b) {
                this->operator()(a, b) = g[4*a + b];
            }
        }
        inline metric<3> spatial_metric() const {
            metric<3> s_metric;
            for(int a = 1; a < 4; ++a)
            for(int b = 1; b < 4; ++b) {
                s_metric(a-1,b-1) = this->operator()(a,b);
            }
            return s_metric;
        }
};

//! Spacetime inverse metric
template<>
class inv_metric<4>: public symmetric2<CCTK_REAL, 4, 2> {
    public:
        //! Construct the spacetime metric from the ADM quantities
        inline void from_adm(
                CCTK_REAL const alp,
                CCTK_REAL const betax,
                CCTK_REAL const betay,
                CCTK_REAL const betaz,
                CCTK_REAL const gxx,
                CCTK_REAL const gxy,
                CCTK_REAL const gxz,
                CCTK_REAL const gyy,
                CCTK_REAL const gyz,
                CCTK_REAL const gzz) {
            CCTK_REAL u[16];
            utils::metric::spacetime_upper(alp, betax, betay,
                    betaz, gxx, gxy, gxz, gyy, gyz, gzz, &u[0]);
            for(int a = 0; a < 4; ++a)
            for(int b = a; b < 4; ++b) {
                this->operator()(a, b) = u[4*a + b];
            }
        }
        inline inv_metric<3> spatial_metric() const {
            inv_metric<3> s_metric;
            for(int a = 1; a < 4; ++a)
            for(int b = 1; b < 4; ++b) {
                s_metric(a-1,b-1) = this->operator()(a,b);
            }
            return s_metric;
        }

};


} // namespace tensors

#endif
