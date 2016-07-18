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

#ifndef TENSORS_TENSOR_FIELD_HH
#define TENSORS_TENSOR_FIELD_HH

#include "utilities.hh"
#include "multiindex.hh"

#include <tensor.hh>
#include <finite_difference.h>

namespace tensors {

  /*  Tensor field class template
   *
   *  \tparam tensor_t underlying tensor type
   */
  template<typename tensor_t>
  class tensor_field_t {
    private:
      // Tensor data type
      typedef typename tensor_t::data_t data_t;
      // Symmetry type
      typedef typename tensor_t::symmetry_t symmetry_t;
      // Multiindex type
      typedef multiindex_t<tensor_t::ndim,
                           tensor_t::rank> mi_t;

      // Rank of the tensor
      static size_t constexpr rank = tensor_t::rank;
      // Number of dimensions
      static size_t constexpr ndim = tensor_t::ndim;
      // Number of degrees of freedom
      static size_t constexpr ndof = tensor_t::ndof;

      //! Tensor type consistency check
      static_assert(utilities::static_pow<ndim,rank>::result
                    >= ndof,
                    "utils::tensor_field::tensor_field_t: "
                    "ndof cannot be greater than dim^rank.");

      // Array of GF pointers
      data_t* gf_pointers[ndof];

      /*  Registers the GF pointers
       *
       *  The data is stored in a row major format
       */
      void registerGFPointers(data_t* p);

      template<typename... pointers_t>
      void registerGFPointers(data_t* p,
                              pointers_t... pointers);

    public:
      /*  Constructor taking the GF pointers
       *
       *  Passing arguments to registerGFPointers
       */
      template<typename... pointers_t>
      tensor_field_t(pointers_t... pointers);

      // Access the GF pointers using the natural indices of the tensor
      template<typename... indices_t>
      inline data_t* operator()(indices_t... indices) const;

      // Access the GF pointers using a multiindex object
      inline data_t* operator()(const mi_t& mi) const;

      // Returns a tensor instantiation at the given point
      inline tensor_t operator[](const size_t ijk) const;

      //! Sets tensor components at the given point
      inline void set_components(const size_t ijk, const tensor_t& tensor) const;

      /*  Compute (FD) spatial partial derivative of tensor at the given point
       *
       *  \tparam fd_order finite differencing order
       */
      template<int fd_order>
      inline general_tensor_t<data_t,
                              ndim,
                              rank+1,
                              symmetry_t>
      partial_derivative(CCTK_POINTER_TO_CONST cctkGH,
                         const size_t i,
                         const size_t j,
                         const size_t k,
                         const vector_t<CCTK_REAL,3>& one_over_dx) const;
  };

  template<typename data_t>
  class scalar_field_t {
    private:
      data_t* gf_pointer;

    public:
      scalar_field_t(data_t* gf_pointer) {
        this->gf_pointer = gf_pointer;
      }

      //! Access the GF pointer
      inline data_t* operator()() const {
        return gf_pointer;
      }
      //! Returns scalar value at given point
      inline const data_t operator[](const size_t ijk) {
        return gf_pointer[ijk];
      }
      //! Returns partial derivative (FD) of scalar field at the given point
      template<int fd_order>
      inline vector_t<data_t,3>
      partial_derivative(CCTK_POINTER_TO_CONST cctkGH,
                         const size_t i,
                         const size_t j,
                         const size_t k,
                         const vector_t<CCTK_REAL,3>& one_over_dx) {


        //! Resulting vector of finite differenced scalar field
        vector_t<data_t,3> fd_vector;

        //! Loop over its indices
        for(size_t partial_index = 0; partial_index<3; ++partial_index) {
          //! Compute partial derivative in i-direction
          fd_vector(partial_index) = one_over_dx(partial_index)
                                     *adiff_1(cctkGH,
                                              gf_pointer,
                                              i,j,k,
                                              partial_index,
                                              fd_order);
        }
        return fd_vector;
      }

    };

  // Implementations

  template<typename tensor_t>
  template<typename... pointers_t>
  tensor_field_t<tensor_t>::tensor_field_t(pointers_t... pointers) {
    static_assert(sizeof...(pointers) == ndof,
                  "utils::tensor_field::tensor_field_t: "
                  "Wrong number of GF pointers passed (!= ndof)");
    registerGFPointers(pointers...);

  }

  template<typename tensor_t>
  void tensor_field_t<tensor_t>::registerGFPointers(data_t* p) {
    gf_pointers[ndof-1] = p;
  }

  template<typename tensor_t>
  template<typename... pointers_t>
  void tensor_field_t<tensor_t>::registerGFPointers(data_t* p, pointers_t... pointers) {
    gf_pointers[ndof-(sizeof...(pointers)+1)] = p;
    registerGFPointers(pointers...);
  }

  template<typename tensor_t>
  template<typename... indices_t>
  inline typename tensor_t::data_t*
  tensor_field_t<tensor_t>::operator()(indices_t... indices) const {
    static_assert(sizeof...(indices) == tensor_t::rank,
                  "utils::tensor_field::tensor_field_t: "
                  "Number of indices must match rank of tensor.");
    return gf_pointers[tensor_t::compress_indices(indices...)];
  }

  template<typename tensor_t>
  inline typename tensor_t::data_t*
  tensor_field_t<tensor_t>::operator()(const mi_t& mi) const {
    return gf_pointers[tensor_t::compress_indices(mi)];
  }

  template<typename tensor_t>
  inline tensor_t tensor_field_t<tensor_t>::operator[](const size_t ijk) const {
    tensor_t tensor;

    for(size_t i = 0; i<ndof; ++i) {
      tensor[i] = gf_pointers[i][ijk];
    }
    return tensor;

  }

  template<typename tensor_t>
  inline void tensor_field_t<tensor_t>::set_components(const size_t ijk, const tensor_t& tensor) const {
    for(size_t i = 0; i<ndof; ++i) {
      gf_pointers[i][ijk] = tensor[i];
    }
  }

  template<typename tensor_t>
  template<int fd_order>
/*
  inline general_tensor_t<typename tensor_t::data_t,
                          tensor_t::ndim,
                          tensor_t::rank+1,
                          typename tensor_t::symmetry_t>
*/
  inline general_tensor_t<typename tensor_field_t<tensor_t>::data_t,
                          tensor_field_t<tensor_t>::ndim,
                          tensor_field_t<tensor_t>::rank+1,
                          typename tensor_field_t<tensor_t>::symmetry_t>
  tensor_field_t<tensor_t>::partial_derivative(CCTK_POINTER_TO_CONST cctkGH,
                                               const size_t i,
                                               const size_t j,
                                               const size_t k,
                                               const vector_t<CCTK_REAL,3>& one_over_dx) const {
    static_assert(ndim == 3,
                  "utils::tensor_field::tensor_field_t: "
                  "Partial derivative only defined for ndim = 3.");

    // Resulting tensor of finite differenced tensor components
    // First index is defined to be index of partial derivative
    general_tensor_t<data_t,ndim,rank+1,symmetry_t> fd_tensor;

    // Loop over its indices
    for(auto mi = fd_tensor.get_mi(); !mi.end(); ++mi) {
      // First index is defined to be index of partial derivative
      size_t partial_index = mi[0];

      // Distribute remaining free indices
      auto free_mi = tensor_t::get_mi();
      for(size_t l = 1; l<rank+1; ++l) {
        free_mi[l-1] = mi[l];
      }

      //! Compute partial derivative for this component
      fd_tensor(mi) = one_over_dx(partial_index)
                      *adiff_1(cctkGH,
                               this->operator()(free_mi),
                               i,j,k,
                               partial_index,
                               fd_order);
    }
    return fd_tensor;
  }
} // namespace tensors

#endif
