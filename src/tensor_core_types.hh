//  TensorTemplates: C++ tensor class templates
//  Copyright (C) 2017, Elias Roland Most (ERM)
//                      <emost@itp.uni-frankfurt.de>
//  Copyright (C) 2018, Ludwig Jens Papenfort
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

#ifndef TENSOR_DEFS_HH
#define TENSOR_DEFS_HH

#include <type_traits>

namespace tensors {

// frame types
struct comoving_t;
struct eulerian_t;
struct any_frame_t;

// rank types
class rank_t {};
class upper_t : public rank_t {};
class lower_t : public rank_t {};

// trait to decide if one wants to store by ref or value
// storing by ref is very bad for disappearing expressions,
// i.e. temporaries and stuff which gets optimized away
template<typename E>
using operant_t = typename std::conditional<E::property_t::is_persistent, E const &, E>::type;

} // namespace tensors

#endif
