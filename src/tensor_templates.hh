//  TensorTemplates: C++ tensor class templates
//  Copyright (C) 2016, Ludwig Jens Papenfort <papenfort@th.physik.uni-frankfurt.de>
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


#ifndef TENSORS_HH
#define TENSORS_HH


#ifdef TENSORS_VECTORIZED
  #pragma message ("TensorTemplates: Vectorization support enabled.")
  #include <Vc/Vc>
#else
  #pragma message ("TensorTemplates: Vectorization support disabled.")
#endif

#ifdef TENSORS_CACTUS
  #pragma message ("TensorTemplates: Cactus support enabled.")
  #include "cctk.h"
#endif

// some helpers
#include "utilities.hh"

// core typedefs, like index types
#include "tensor_core_types.hh"

// symmetry types, implementing index transformations and compessed indices
#include "tensor_symmetry.hh"

// general tensor typedefs
#include "tensor_types.hh"

// index helpers
#include "tensor_index.hh"

// property types of tensors and tensor expressions
#include "tensor_property.hh"

// general tensor expression and simple expressions
#include "tensor_expressions.hh"

// some tensor expression helpers
#include "tensor_helpers.hh"

// THE general tensor class
#include "tensor.hh"

// slice expressions, cutting tensors into pieces
#include "tensor_slice.hh"

// a general finite difference expression, i.e. the partial derivative of a tensor
#include "tensor_derivative.hh"

// scalar finite differences, i.e. the partial derivative of a scalar
#include "scalar_derivative.hh"

// symmetry casts of tensor expressions, useful to make expressions explicitly symmetric
#include "tensor_symmetry_expressions.hh"

// general contraction expression of two tensor expressions
#include "tensor_contraction.hh"

// general trace expression of a tensor expression
#include "tensor_trace.hh"

// general tensor product expression
#include "tensor_concat.hh"

// general hadamard like product
#include "tensor_product.hh"

// general index reordering expression
#include "tensor_index_reordering.hh"

// metric tensor implementing useful things, like det, inverse etc.
#include "metric.hh"

// special tensors, like the identity and levi civita
#include "special.hh"

// tensor field and tensor field expressions, loading / storing tensors from / to memory, get FDs
#include "tensor_field.hh"

// scalar field and scalar wrapper, loading / storing scalars from / to memory, get FDs
#include "scalar_field.hh"

// general finite difference routines
#include "finite_differences.hh"

#ifdef TENSORS_CACTUS
  // specialized Cactus FD routines
  #include "cactus_types.hh"
#endif

//! main namespace
namespace tensors {
}

#endif
