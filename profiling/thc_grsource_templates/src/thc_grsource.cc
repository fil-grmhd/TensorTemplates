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

#include "finite_difference.h"
#include "utils.hh"
#include "tensor_templates.hh"

#define SQ(X) ((X)*(X))

extern "C" void THC_GRSource_temp(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTS
    DECLARE_CCTK_PARAMETERS

    constexpr int fd_order = 4;

    int const gsiz = UTILS_GFSIZE(cctkGH);

    CCTK_REAL * velx = &vel[0*gsiz];
    CCTK_REAL * vely = &vel[1*gsiz];
    CCTK_REAL * velz = &vel[2*gsiz];

/* tensor field is not compatible with const pointers (yet)
    // should use const tensor_field_t instead
    CCTK_REAL const * velx = &vel[0*gsiz];
    CCTK_REAL const * vely = &vel[1*gsiz];
    CCTK_REAL const * velz = &vel[2*gsiz];
*/
    CCTK_REAL * rhs_sconx  = &rhs_scon[0*gsiz];
    CCTK_REAL * rhs_scony  = &rhs_scon[1*gsiz];
    CCTK_REAL * rhs_sconz  = &rhs_scon[2*gsiz];


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
    CCTK_REAL const idx = 1.0/dx;
    CCTK_REAL const idy = 1.0/dy;
    CCTK_REAL const idz = 1.0/dz;

#pragma omp parallel
    {
        UTILS_LOOP3(thc_grsource_templates,
                k, cctk_nghostzones[2], cctk_lsh[2]-cctk_nghostzones[2],
                j, cctk_nghostzones[1], cctk_lsh[1]-cctk_nghostzones[1],
                i, cctk_nghostzones[0], cctk_lsh[0]-cctk_nghostzones[0]) {
            int const ijk = CCTK_GFINDEX3D(cctkGH, i, j, k);


            // contruct four metric and inverse
// doesn't work (yet)
//            metric4_t<CCTK_REAL> metric4d(alp[ijk],beta[ijk],gamma[ijk]);
            metric4_t<CCTK_REAL> metric4d(alp[ijk],beta[ijk],evaluate(gamma[ijk]));

            // constructor is choking if gamma is passed directly
//            metric4_t<CCTK_REAL> metric4d(alp[ijk],beta[ijk],metric_tensor_t<double,3>(gamma[ijk]));


            // Derivatives of the lapse, metric and shift
            covector3_t<CCTK_REAL> dalp(idx*cdiff_x(cctkGH, alp, i, j, k, fd_order),
                                        idy*cdiff_y(cctkGH, alp, i, j, k, fd_order),
                                        idz*cdiff_z(cctkGH, alp, i, j, k, fd_order));

// maybe remove zero inits
            tensor3_t<CCTK_REAL,upper_t,lower_t> dbeta;

            dbeta.c<0,0>() = idx*cdiff_x(cctkGH, betax, i, j, k, fd_order);
            dbeta.c<1,0>() = idx*cdiff_x(cctkGH, betay, i, j, k, fd_order);
            dbeta.c<2,0>() = idx*cdiff_x(cctkGH, betaz, i, j, k, fd_order);
            dbeta.c<0,1>() = idy*cdiff_y(cctkGH, betax, i, j, k, fd_order);
            dbeta.c<1,1>() = idy*cdiff_y(cctkGH, betay, i, j, k, fd_order);
            dbeta.c<2,1>() = idy*cdiff_y(cctkGH, betaz, i, j, k, fd_order);
            dbeta.c<0,2>() = idz*cdiff_z(cctkGH, betax, i, j, k, fd_order);
            dbeta.c<1,2>() = idz*cdiff_z(cctkGH, betay, i, j, k, fd_order);
            dbeta.c<2,2>() = idz*cdiff_z(cctkGH, betaz, i, j, k, fd_order);

            sym_tensor3_t<CCTK_REAL,0,1,lower_t,lower_t,lower_t> dgamma;

            // we need a derivative expression...
            dgamma.c<0,0,0>() = idx*cdiff_x(cctkGH, gxx, i, j, k, fd_order);
            dgamma.c<0,1,0>() = idx*cdiff_x(cctkGH, gxy, i, j, k, fd_order);
            dgamma.c<0,2,0>() = idx*cdiff_x(cctkGH, gxz, i, j, k, fd_order);
            dgamma.c<1,1,0>() = idx*cdiff_x(cctkGH, gyy, i, j, k, fd_order);
            dgamma.c<1,2,0>() = idx*cdiff_x(cctkGH, gyz, i, j, k, fd_order);
            dgamma.c<2,2,0>() = idx*cdiff_x(cctkGH, gzz, i, j, k, fd_order);
            dgamma.c<0,0,1>() = idy*cdiff_y(cctkGH, gxx, i, j, k, fd_order);
            dgamma.c<0,1,1>() = idy*cdiff_y(cctkGH, gxy, i, j, k, fd_order);
            dgamma.c<0,2,1>() = idy*cdiff_y(cctkGH, gxz, i, j, k, fd_order);
            dgamma.c<1,1,1>() = idy*cdiff_y(cctkGH, gyy, i, j, k, fd_order);
            dgamma.c<1,2,1>() = idy*cdiff_y(cctkGH, gyz, i, j, k, fd_order);
            dgamma.c<2,2,1>() = idy*cdiff_y(cctkGH, gzz, i, j, k, fd_order);
            dgamma.c<0,0,2>() = idz*cdiff_z(cctkGH, gxx, i, j, k, fd_order);
            dgamma.c<0,1,2>() = idz*cdiff_z(cctkGH, gxy, i, j, k, fd_order);
            dgamma.c<0,2,2>() = idz*cdiff_z(cctkGH, gxz, i, j, k, fd_order);
            dgamma.c<1,1,2>() = idz*cdiff_z(cctkGH, gyy, i, j, k, fd_order);
            dgamma.c<1,2,2>() = idz*cdiff_z(cctkGH, gyz, i, j, k, fd_order);
            dgamma.c<2,2,2>() = idz*cdiff_z(cctkGH, gzz, i, j, k, fd_order);

            // helpers to construct four metric derivative
            auto dg00i = - 2*dalp
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

            // four velocity
            const auto u0 =  w_lorentz[ijk]/alp[ijk];
            auto u = vector4_t<CCTK_REAL> (u0,
                                          (alp[ijk]*velx[ijk] - betax[ijk])*u0,
                                          (alp[ijk]*vely[ijk] - betay[ijk])*u0,
                                          (alp[ijk]*velz[ijk] - betaz[ijk])*u0);
            // construct tensor product
            auto uu = sym2_cast(tensor_cat(u,u));
            // inv metric still not symmetric
//            auto invmetric = sym2_cast(metric4d.invmetric);

            // construct em tensor
            sym_tensor4_t<CCTK_REAL,0,1,upper_t,upper_t> T = (rho[ijk]*(1+eps[ijk])+press[ijk])*uu
                                                           +  press[ijk]*metric4d.invmetric;

            // slice expression to contain only the spatial components
            auto rhs_scon_vec = slice<-2>(0.5*alp[ijk]*metric4d.sqrtdet
                                         *(trace(contract(T,dg))));

            r_scon[ijk] = rhs_scon_vec;

            // slice em tensor
            auto T0i = slice<0,-2>(T);

            auto Tij = sym2_cast(slice<-2,-2>(T));

            auto bb = sym2_cast(tensor_cat(beta[ijk],beta[ijk]));
            auto T0ib = tensor_cat(T0i,beta[ijk]);

            rhs_tau[ijk] = alp[ijk] * metric4d.sqrtdet
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

        } UTILS_ENDLOOP3(thc_grsource_templates);
    }
}
