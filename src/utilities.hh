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


#ifndef TENSORS_UTILITIES_HH
#define TENSORS_UTILITIES_HH

namespace tensors {
namespace utilities {

template<size_t a, size_t b>
struct static_pow {
    static constexpr size_t value = a*static_pow<a,b-1>::value;
};
template<size_t a>
struct static_pow<a,0> {
    static constexpr size_t value = 1;
};


//Useful helper for checking conditions on template parameters
//https://stackoverflow.com/questions/28253399/check-traits-for-all-variadic-template-arguments/28253503#28253503
template<bool...> struct bool_pack;
template<bool... bs> 
using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;



} // namespace utilities
} // namespace tensors

#endif
