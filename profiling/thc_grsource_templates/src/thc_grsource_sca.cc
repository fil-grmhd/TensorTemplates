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


// this has to be included first, before the cctk stuff
// if not, Vc headers fail to compile, not sure why yet
// activates vectorized types and implementations
#define TENSORS_VECTORIZED
// needed for cactus fd interface
#define TENSORS_CACTUS
#include "tensor_templates.hh"

#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"

#define COMPUTE_DERIVATIVES

extern "C" void THC_GRSource_temp(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTS
    DECLARE_CCTK_PARAMETERS

    constexpr int fd_order = 4;

    // get grid size to know position of GF array component pointers
    int const gsiz = cctkGH->cctk_ash[0]*cctkGH->cctk_ash[1]*cctkGH->cctk_ash[2];

    using namespace tensors;

    // init tensor fields with GF pointers

    // these are only here to make it symmetric to the vectorized version
    scalar_field_t<CCTK_REAL> alpha_field(alp);
    scalar_field_t<CCTK_REAL> w_lorentz_field(w_lorentz);
    scalar_field_t<CCTK_REAL> rho_field(rho);
    scalar_field_t<CCTK_REAL> eps_field(eps);
    scalar_field_t<CCTK_REAL> press_field(press);

    tensor_field_t<vector3_t<CCTK_REAL>> beta_field(betax,
                                                    betay,
                                                    betaz);

    tensor_field_t<vector3_t<CCTK_REAL>> vel_field(&vel[0*gsiz],
                                                   &vel[1*gsiz],
                                                   &vel[2*gsiz]);

    tensor_field_t<covector3_t<CCTK_REAL>> rhs_sources(&rhs_scon_temp[0*gsiz],
                                                       &rhs_scon_temp[1*gsiz],
                                                       &rhs_scon_temp[2*gsiz]);

    tensor_field_t<sym_tensor3_t<CCTK_REAL,0,1,lower_t,lower_t>> gamma_field(gxx,gxy,gxz,gyy,gyz,gzz);

    tensor_field_t<sym_tensor3_t<CCTK_REAL,0,1,lower_t,lower_t>> K_field(kxx,kxy,kxz,kyy,kyz,kzz);

    using data_t = CCTK_REAL;

    // get grid spacing
    CCTK_REAL const dx  = CCTK_DELTA_SPACE(0);
    CCTK_REAL const dy  = CCTK_DELTA_SPACE(1);
    CCTK_REAL const dz  = CCTK_DELTA_SPACE(2);

    // create a (specialized) cactus central differentiator
    fd::cactus_cdiff<fd_order> cdiff(cctkGH,dx,dy,dz);

    // loop over local grid
    #pragma omp parallel for
    for(int k = cctk_nghostzones[2]; k < cctk_lsh[2]-cctk_nghostzones[2]; ++k)
    for(int j = cctk_nghostzones[1]; j < cctk_lsh[1]-cctk_nghostzones[1]; ++j)
    // force inlining helps Intel compiler to generate optimal code (substituting all expressions)
    #pragma forceinline recursive
    for(int i = cctk_nghostzones[0]; i < cctk_lsh[0]-cctk_nghostzones[0]; ++i) {
      int const ijk = CCTK_GFINDEX3D(cctkGH, i, j, k);

      // 3-metric and its determinant
      metric_tensor3_t<CCTK_REAL> gamma(gamma_field[ijk]);
      data_t det = gamma.det();
      data_t sqrt_det = std::sqrt(det);

      auto beta = beta_field[ijk];
      auto alpha = alpha_field[ijk];

      // inverse 4-metric, needed for energy-momentum tensor
      auto inv_g = gamma.inverse_spacetime_metric(alpha,beta,det);

      // Derivatives of the lapse, metric and shift
      // this could be simplified by a specialized scalar field type
      #ifdef COMPUTE_DERIVATIVES
      covector3_t<CCTK_REAL> dalp(cdiff.diff<0>(alp, ijk),
                                  cdiff.diff<1>(alp, ijk),
                                  cdiff.diff<2>(alp, ijk));
      #else
      covector3_t<CCTK_REAL> dalp;
      #endif

      // get central finite difference of beta components, defined by cdiff
      #ifdef COMPUTE_DERIVATIVES
      tensor3_t<CCTK_REAL,upper_t,lower_t> dbeta = beta_field[ijk].finite_diff(cdiff);
      #else
      tensor3_t<CCTK_REAL,upper_t,lower_t> dbeta;
      #endif

      #ifdef COMPUTE_DERIVATIVES
      sym_tensor3_t<CCTK_REAL,0,1,lower_t,lower_t,lower_t> dgamma = gamma_field[ijk].finite_diff(cdiff);
      #else
      sym_tensor3_t<CCTK_REAL,0,1,lower_t,lower_t,lower_t> dgamma;
      #endif

      // helpers to construct four metric derivative
      auto dg00i = - 2*alpha*dalp
                   + 2*contract(dbeta,contract(gamma,beta))
                   + contract(beta,contract(beta,dgamma));

      auto dg0ji = contract(gamma,dbeta)
                 + contract(dgamma,beta);

      // gj0i is the same due to symmetry
      // gjki is dgamma

      // initialized to zero, first two indices are symmetric
      sym_tensor4_t<CCTK_REAL,0,1,lower_t,lower_t,lower_t> dg;

      // g00,i
      dg.set<0,0,-2>(dg00i);

      // g0j,i
      dg.set<0,-2,-2>(dg0ji);

      // gjk,i
      dg.set<-2,-2,-2>(dgamma);

      // helper for four velocity u
      const double u0 =  w_lorentz_field[ijk]/alpha;
      auto ui = u0*(alpha*vel_field[ijk] - beta);

      // four velocity u^mu
      vector4_t<CCTK_REAL> u;
      u.c<0>() = u0;
      u.set<-2>(ui);

      // construct tensor product of u^mu u^nu
      auto uu = sym2_cast(tensor_cat(u,u));

      // construct energy-momentum tensor
      sym_tensor4_t<CCTK_REAL,0,1,upper_t,upper_t> T = (rho_field[ijk]*(1+eps_field[ijk])+press_field[ijk])*uu
                                                     +  press_field[ijk]*inv_g;

      // slice expression to contain only the spatial components
      rhs_sources[ijk] = slice<-2>(0.5*alpha * sqrt_det
                                   *(trace(contract(T,dg))));

      // slice em tensor in different ways
      auto T0i = slice<0,-2>(T);

      auto Tij = sym2_cast(slice<-2,-2>(T));

      // construct tensor products
      auto bb = sym2_cast(tensor_cat(beta,beta));
      auto T0ib = tensor_cat(T0i,beta);

      rhs_tau_temp[ijk] = alpha * sqrt_det
                        * (trace(
                             contract(
                               T.c<0,0>()*bb + 2 * T0ib + Tij,
                               K_field[ijk]
                             )
                           )
                         - contract(
                             T.c<0,0>()*beta + T0i,
                             dalp
                           )
                          );
    }
}
