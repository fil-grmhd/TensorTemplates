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

extern "C" void THC_GRSource(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTS
    DECLARE_CCTK_PARAMETERS

    CCTK_INFO("THC_GRSource_templates");

    constexpr int fd_order = 4;
    constexpr size_t exp = 4;

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
    //tensor_field_t<vector3_t<CCTK_REAL>> v(velx,vely,velz);
    tensor_field_t<vector3_t<CCTK_REAL>> beta(betax,betay,betaz);

    // dont know type yet
    tensor_field_t<covector3_t<CCTK_REAL>> r_scon(rhs_sconx,rhs_scony,rhs_sconz);

    tensor_field_t<metric_tensor_t<double,3>> gamma(gxx,gxy,gxz,
                                                    gxy,gyy,gyz,
                                                    gxz,gyz,gzz);

    tensor_field_t<tensor3_t<CCTK_REAL,lower_t,lower_t>> K(kxx,kxy,kxz,
                                                           kxy,kyy,kyz,
                                                           kxz,kyz,kzz);

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

            // Derivatives of the lapse, metric and shift
            covector3_t<CCTK_REAL> dalp;

            dalp(0) = idx*cdiff_x(cctkGH, alp, i, j, k, fd_order);
            dalp(1) = idy*cdiff_y(cctkGH, alp, i, j, k, fd_order);
            dalp(2) = idz*cdiff_z(cctkGH, alp, i, j, k, fd_order);

            tensor3_t<CCTK_REAL, upper_t, lower_t> dbeta;

            dbeta(0,0) = idx*cdiff_x(cctkGH, betax, i, j, k, fd_order);
            dbeta(1,0) = idx*cdiff_x(cctkGH, betay, i, j, k, fd_order);
            dbeta(2,0) = idx*cdiff_x(cctkGH, betaz, i, j, k, fd_order);
            dbeta(0,1) = idy*cdiff_y(cctkGH, betax, i, j, k, fd_order);
            dbeta(1,1) = idy*cdiff_y(cctkGH, betay, i, j, k, fd_order);
            dbeta(2,1) = idy*cdiff_y(cctkGH, betaz, i, j, k, fd_order);
            dbeta(0,2) = idz*cdiff_z(cctkGH, betax, i, j, k, fd_order);
            dbeta(1,2) = idz*cdiff_z(cctkGH, betay, i, j, k, fd_order);
            dbeta(2,2) = idz*cdiff_z(cctkGH, betaz, i, j, k, fd_order);

            tensor3_t<CCTK_REAL, lower_t, lower_t, lower_t> dgamma;

            // we need a derivative expression...
            dgamma(0,0,0) = idx*cdiff_x(cctkGH, gxx, i, j, k, fd_order);
            dgamma(0,1,0) = idx*cdiff_x(cctkGH, gxy, i, j, k, fd_order);
            dgamma(0,2,0) = idx*cdiff_x(cctkGH, gxz, i, j, k, fd_order);
            dgamma(1,1,0) = idx*cdiff_x(cctkGH, gyy, i, j, k, fd_order);
            dgamma(1,2,0) = idx*cdiff_x(cctkGH, gyz, i, j, k, fd_order);
            dgamma(2,2,0) = idx*cdiff_x(cctkGH, gzz, i, j, k, fd_order);
            dgamma(0,0,1) = idy*cdiff_y(cctkGH, gxx, i, j, k, fd_order);
            dgamma(0,1,1) = idy*cdiff_y(cctkGH, gxy, i, j, k, fd_order);
            dgamma(0,2,1) = idy*cdiff_y(cctkGH, gxz, i, j, k, fd_order);
            dgamma(1,1,1) = idy*cdiff_y(cctkGH, gyy, i, j, k, fd_order);
            dgamma(1,2,1) = idy*cdiff_y(cctkGH, gyz, i, j, k, fd_order);
            dgamma(2,2,1) = idy*cdiff_y(cctkGH, gzz, i, j, k, fd_order);
            dgamma(0,0,2) = idz*cdiff_z(cctkGH, gxx, i, j, k, fd_order);
            dgamma(0,1,2) = idz*cdiff_z(cctkGH, gxy, i, j, k, fd_order);
            dgamma(0,2,2) = idz*cdiff_z(cctkGH, gxz, i, j, k, fd_order);
            dgamma(1,1,2) = idz*cdiff_z(cctkGH, gyy, i, j, k, fd_order);
            dgamma(1,2,2) = idz*cdiff_z(cctkGH, gyz, i, j, k, fd_order);
            dgamma(2,2,2) = idz*cdiff_z(cctkGH, gzz, i, j, k, fd_order);

            // we need a symmetrizer / symmetric type
            dgamma(1,0,0) = dgamma(0,1,0);
            dgamma(2,0,0) = dgamma(0,2,0);
            dgamma(2,1,0) = dgamma(1,2,0);

            dgamma(1,0,1) = dgamma(0,1,1);
            dgamma(2,0,1) = dgamma(0,2,1);
            dgamma(2,1,1) = dgamma(1,2,1);

            dgamma(1,0,2) = dgamma(0,1,2);
            dgamma(2,0,2) = dgamma(0,2,2);
            dgamma(2,1,2) = dgamma(1,2,2);

            // helpers to construct four metric derivative
            covector3_t<CCTK_REAL> g00i = - 2*dalp
                                          + 2*contract<0,0>(dbeta,contract<0,0>(gamma[ijk],beta[ijk]))
                                          + contract<0,0>(beta[ijk],contract<0,0>(beta[ijk],dgamma));

            tensor3_t<CCTK_REAL, lower_t, lower_t> g0ji = contract<0,0>(gamma[ijk],dbeta)
                                                        + contract<0,0>(dgamma,beta[ijk]);
            // gj0i is the same due to symmetry
            // gjki is dgamma

            // four metric derivative, here some type of collective component set would be useful
            // initialized to zero
            tensor4_t<CCTK_REAL, lower_t, lower_t, lower_t> dmetric;

            // g00,i
            dmetric(0,0,1) = g00i(0);
            dmetric(0,0,2) = g00i(1);
            dmetric(0,0,3) = g00i(2);

            // g0j,i
            dmetric(0,1,1) = g0ji(0,0);
            dmetric(0,1,2) = g0ji(0,1);
            dmetric(0,1,3) = g0ji(0,2);

            dmetric(0,2,1) = g0ji(1,0);
            dmetric(0,2,2) = g0ji(1,1);
            dmetric(0,2,3) = g0ji(1,2);

            dmetric(0,3,1) = g0ji(2,0);
            dmetric(0,3,2) = g0ji(2,1);
            dmetric(0,3,3) = g0ji(2,2);

            // gj0,i
            dmetric(1,0,1) = g0ji(0,0);
            dmetric(1,0,2) = g0ji(0,1);
            dmetric(1,0,3) = g0ji(0,2);

            dmetric(2,0,1) = g0ji(1,0);
            dmetric(2,0,2) = g0ji(1,1);
            dmetric(2,0,3) = g0ji(1,2);

            dmetric(3,0,1) = g0ji(2,0);
            dmetric(3,0,2) = g0ji(2,1);
            dmetric(3,0,3) = g0ji(2,2);

            // gjk,i
            dmetric(1,1,1) = dgamma(0,0,0);
            dmetric(1,1,2) = dgamma(0,0,1);
            dmetric(1,1,3) = dgamma(0,0,2);

            dmetric(1,2,1) = dgamma(0,1,0);
            dmetric(1,2,2) = dgamma(0,1,1);
            dmetric(1,2,3) = dgamma(0,1,2);

            dmetric(1,3,1) = dgamma(0,2,0);
            dmetric(1,3,2) = dgamma(0,2,1);
            dmetric(1,3,3) = dgamma(0,2,2);

            dmetric(2,1,1) = dgamma(1,0,0);
            dmetric(2,1,2) = dgamma(1,0,1);
            dmetric(2,1,3) = dgamma(1,0,2);

            dmetric(2,2,1) = dgamma(1,1,0);
            dmetric(2,2,2) = dgamma(1,1,1);
            dmetric(2,2,3) = dgamma(1,1,2);

            dmetric(2,3,1) = dgamma(1,2,0);
            dmetric(2,3,2) = dgamma(1,2,1);
            dmetric(2,3,3) = dgamma(1,2,2);

            dmetric(3,1,1) = dgamma(2,0,0);
            dmetric(3,1,2) = dgamma(2,0,1);
            dmetric(3,1,3) = dgamma(2,0,2);

            dmetric(3,2,1) = dgamma(2,1,0);
            dmetric(3,2,2) = dgamma(2,1,1);
            dmetric(3,2,3) = dgamma(2,1,2);

            dmetric(3,3,1) = dgamma(2,2,0);
            dmetric(3,3,2) = dgamma(2,2,1);
            dmetric(3,3,3) = dgamma(2,2,2);


            // four metric
            metric4_t<CCTK_REAL> metric4d(alp[ijk],beta[ijk],metric_tensor_t<double,3>(gamma[i]));

            // four velocity
            const auto u0 =  w_lorentz[ijk]/alp[ijk];
            auto u = vector4_t<CCTK_REAL> (u0,
                                          (alp[ijk]*velx[ijk] - betax[ijk])*u0,
                                          (alp[ijk]*vely[ijk] - betay[ijk])*u0,
                                          (alp[ijk]*velz[ijk] - betaz[ijk])*u0);

            // rank-2 tensor through tensor product
            auto uu = tensor_cat(u,u);

            // construct em tensor
            tensor4_t<CCTK_REAL, upper_t, upper_t> em_tensor = (rho[ijk]*(1+eps[ijk])+press[ijk])*uu
                                                             +  press[ijk]*metric4d.invmetric;

            // metric4d.sqrtdet is actually sqrt(det(gamma))
            covector4_t<CCTK_REAL> rhs_scon_vec = 0.5*alp[ijk]*metric4d.sqrtdet*trace<0,1>(contract<0,0>(em_tensor,dmetric));
            //auto rhs_scon_vec = 0.5*alp[ijk]*metric4d.sqrtdet*trace<0,1>(contract<0,0>(em_tensor,dmetric));

            // can't do this, because there is no subtensor expression
            // r_scon[ijk] = rhs_scon_vec;

            rhs_sconx[ijk] = rhs_scon_vec(1);
            rhs_scony[ijk] = rhs_scon_vec(2);
            rhs_sconz[ijk] = rhs_scon_vec(3);

        } UTILS_ENDLOOP3(thc_grsource_templates);
    }
}
