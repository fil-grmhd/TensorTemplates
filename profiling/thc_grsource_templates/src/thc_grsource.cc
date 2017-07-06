//  Templated Hydrodynamics Code: an hydro code built on top of HRSCCore
//  Copyright (C) 2015, David Radice <dradice@caltech.edu>
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


#include <cmath>

#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"

// doesn't work yet
// activates vectorized types and implementations
#define TENSORS_VECTORIZED

// needed for cactus fd interface
#define TENSORS_CACTUS
#include "tensor_templates.hh"



extern "C" void THC_GRSource_temp(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTS
    DECLARE_CCTK_PARAMETERS

    constexpr int fd_order = 4;

    int const gsiz = cctkGH->cctk_ash[0]*cctkGH->cctk_ash[1]*cctkGH->cctk_ash[2];

    CCTK_REAL * velx = &vel[0*gsiz];
    CCTK_REAL * vely = &vel[1*gsiz];
    CCTK_REAL * velz = &vel[2*gsiz];

    CCTK_REAL * rhs_sconx  = &rhs_scon_temp[0*gsiz];
    CCTK_REAL * rhs_scony  = &rhs_scon_temp[1*gsiz];
    CCTK_REAL * rhs_sconz  = &rhs_scon_temp[2*gsiz];


    using namespace tensors;

    tensor_field_t<vector3_t<CCTK_REAL>> beta(betax,
                                              betay,
                                              betaz);

    tensor_field_t<covector3_t<CCTK_REAL>> r_scon(rhs_sconx,
                                                  rhs_scony,
                                                  rhs_sconz);

    tensor_field_t<sym_tensor3_t<CCTK_REAL,0,1,lower_t,lower_t>> gamma(gxx,gxy,gxz,gyy,gyz,gzz);

    tensor_field_t<sym_tensor3_t<CCTK_REAL,0,1,lower_t,lower_t>> K(kxx,kxy,kxz,kyy,kyz,kzz);

    CCTK_REAL const dx  = CCTK_DELTA_SPACE(0);
    CCTK_REAL const dy  = CCTK_DELTA_SPACE(1);
    CCTK_REAL const dz  = CCTK_DELTA_SPACE(2);

    fd::cactus_cdiff<fd_order> cdiff(cctkGH,dx,dy,dz);

    #pragma omp parallel for
    for(int k = cctk_nghostzones[2]; k < cctk_lsh[2]-cctk_nghostzones[2]; ++k)
    for(int j = cctk_nghostzones[1]; j < cctk_lsh[1]-cctk_nghostzones[1]; ++j)
    #pragma forceinline recursive
    for(int i = cctk_nghostzones[0]; i < cctk_lsh[0]-cctk_nghostzones[0]; ++i) {
            int const ijk = CCTK_GFINDEX3D(cctkGH, i, j, k);

            // construct four metric and inverse
            // this is the only way to make it work, passing gamma[ijk] or evaluate(gamma[ijk]) fails
            auto metric4d = make_metric4(alp[ijk],beta[ijk],metric_tensor_t<CCTK_REAL,3>(gamma[ijk]));

            // Derivatives of the lapse, metric and shift
            // this could be simplified by a specialized scalar field type
            covector3_t<CCTK_REAL> dalp(cdiff.diff<0>(alp, ijk),
                                        cdiff.diff<1>(alp, ijk),
                                        cdiff.diff<2>(alp, ijk));

            tensor3_t<CCTK_REAL,upper_t,lower_t> dbeta = beta[ijk].finite_diff(cdiff);

            sym_tensor3_t<CCTK_REAL,0,1,lower_t,lower_t,lower_t> dgamma = gamma[ijk].finite_diff(cdiff);

            // helpers to construct four metric derivative
            auto dg00i = - 2*alp[ijk]*dalp
                         + 2*contract(dbeta,contract(gamma[ijk],beta[ijk]))
                         + contract(beta[ijk],contract(beta[ijk],dgamma));

            auto dg0ji = contract(gamma[ijk],dbeta)
                       + contract(dgamma,beta[ijk]);

            // gj0i is the same due to symmetry
            // gjki is dgamma

            // initialized to zero
            sym_tensor4_t<CCTK_REAL,0,1,lower_t,lower_t,lower_t> dg;

            // g00,i
            dg.set<0,0,-2>(dg00i);

            // g0j,i
            dg.set<0,-2,-2>(dg0ji);

            // gjk,i
            dg.set<-2,-2,-2>(dgamma);

            // four velocity u^mu
            const auto u0 =  w_lorentz[ijk]/alp[ijk];
            auto u = vector4_t<CCTK_REAL> (u0,
                                          (alp[ijk]*velx[ijk] - betax[ijk])*u0,
                                          (alp[ijk]*vely[ijk] - betay[ijk])*u0,
                                          (alp[ijk]*velz[ijk] - betaz[ijk])*u0);
            // construct tensor product of u^mu u^nu
            auto uu = sym2_cast(tensor_cat(u,u));

            // construct em tensor
            sym_tensor4_t<CCTK_REAL,0,1,upper_t,upper_t> T = (rho[ijk]*(1+eps[ijk])+press[ijk])*uu
                                                           +  press[ijk]*metric4d.invmetric;

            // slice expression to contain only the spatial components, which is rhs S conserved
            r_scon[ijk] = slice<-2>(0.5*alp[ijk]*metric4d.sqrtdet
                                         *(trace(contract(T,dg))));

            // slice em tensor in different ways
            auto T0i = slice<0,-2>(T);

            auto Tij = sym2_cast(slice<-2,-2>(T));

            // construct tensor products
            auto bb = sym2_cast(tensor_cat(beta[ijk],beta[ijk]));
            auto T0ib = tensor_cat(T0i,beta[ijk]);

            rhs_tau_temp[ijk] = alp[ijk] * metric4d.sqrtdet
                              * (trace(
                                   contract(
                                     T.c<0,0>()*bb + 2 * T0ib + Tij,
                                     K[ijk]
                                   )
                                 )
                               - contract(
                                   T.c<0,0>()*beta[ijk] + T0i,
                                   dalp
                                 )
                                );

    }
}
