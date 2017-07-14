//  TensorTemplates: C++ tensor class templates
//  Copyright (C) 2017, Ludwig Jens Papenfort
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

#include "tensor_templates.hh"

#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"

#include "utils.hh"


extern "C" void THC_GRSource_comparison(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTS
    DECLARE_CCTK_PARAMETERS

    using namespace tensors;

    constexpr size_t exp = 16;
    constexpr double eps_err = 1.0/utilities::static_pow<10,exp>::value;

    int const gsiz = UTILS_GFSIZE(cctkGH);

    tensor_field_t<covector3_t<CCTK_REAL>> r_scon(&rhs_scon_temp[0*gsiz],
                                                  &rhs_scon_temp[1*gsiz],
                                                  &rhs_scon_temp[2*gsiz]);

    tensor_field_t<covector3_t<CCTK_REAL>> r_scon_orig(&rhs_scon_orig[0*gsiz],
                                                       &rhs_scon_orig[1*gsiz],
                                                       &rhs_scon_orig[2*gsiz]);

    #pragma omp parallel for
    for(int k = cctk_nghostzones[2]; k < cctk_lsh[2]-cctk_nghostzones[2]; ++k)
    for(int j = cctk_nghostzones[1]; j < cctk_lsh[1]-cctk_nghostzones[1]; ++j)
    for(int i = cctk_nghostzones[0]; i < cctk_lsh[0]-cctk_nghostzones[0]; ++i) {

        int const ijk = CCTK_GFINDEX3D(cctkGH, i, j, k);
        covector3_t<CCTK_REAL> rhs_scon_temp(r_scon[ijk]);
        covector3_t<CCTK_REAL> rhs_scon_orig = r_scon_orig[ijk];

        auto compare_scon = rhs_scon_temp.compare_components<exp>(rhs_scon_orig);

        rhs_scon_reldiff[ijk] = compare_scon.second;

        double rel_err_tau = 2*std::abs(rhs_tau_temp[ijk] - rhs_tau_orig[ijk])
                             /(std::abs(rhs_tau_temp[ijk]) + std::abs(rhs_tau_orig[ijk]));
        rhs_tau_reldiff[ijk] = rel_err_tau;
    }
}

extern "C" void THC_GRSource_comp_test(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTS
    DECLARE_CCTK_PARAMETERS

    CCTK_VInfo(CCTK_THORNSTRING,"Local patch: [%i,%i,%i]",cctk_lsh[2],cctk_lsh[1],cctk_lsh[0]);
    CCTK_VInfo(CCTK_THORNSTRING,"Ghostzones: [%i,%i,%i]",cctk_nghostzones[2],cctk_nghostzones[1],cctk_nghostzones[0]);

    for(int k = cctk_nghostzones[2]; k < cctk_lsh[2]-cctk_nghostzones[2]; ++k)
    for(int j = cctk_nghostzones[1]; j < cctk_lsh[1]-cctk_nghostzones[1]; ++j)
    for(int i = cctk_nghostzones[0]; i < cctk_lsh[0]-cctk_nghostzones[0]; ++i) {
      int const ijk = CCTK_GFINDEX3D(cctkGH, i, j, k);
      CCTK_VInfo(CCTK_THORNSTRING,"(%i,%i,%i) -> %i",i,j,k,ijk);
    }
}

