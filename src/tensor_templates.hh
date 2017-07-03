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
#pragma message ("TensorTemplates: Vectorization support activated.")
#include <Vc/Vc>
#endif

#include "utilities.hh"
#include "tensor_core_types.hh"
#include "tensor_symmetry.hh"
#include "tensor_types.hh"
#include "tensor_index.hh"
#include "tensor_property.hh"
#include "tensor_expressions.hh"
#include "tensor_helpers.hh"
#include "tensor.hh"
#include "tensor_slice.hh"
#include "tensor_field.hh"
#include "tensor_symmetry_expressions.hh"
#include "tensor_contraction.hh"
#include "tensor_trace.hh"
#include "tensor_concat.hh"
#include "tensor_index_reordering.hh"
#include "metric.hh"

#ifdef TENSORS_VECTORIZED
#include "tensor_field_vectorized.hh"
#endif

//! main namespace
namespace tensors {
}

#endif
