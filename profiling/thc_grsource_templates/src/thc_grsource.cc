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
// default types to vectorized types if vectorization is active
#define TENSORS_AUTOVEC
// activates cactus fd interface
#define TENSORS_CACTUS
#include "tensor_templates.hh"

#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"

// turn off derivatives for profiling non-FD code
#define COMPUTE_DERIVATIVES

extern "C" void THC_GRSource_temp(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTS
    DECLARE_CCTK_PARAMETERS

    constexpr int fd_order = 4;

    // get grid size to know position of GF array component pointers
    int const gsiz = cctkGH->cctk_ash[0]*cctkGH->cctk_ash[1]*cctkGH->cctk_ash[2];

    using namespace tensors;

    // either one or Vc::double_v::Size
    size_t inc = loop_inc<CCTK_REAL>;

    // get the used types into this namespace

    // this is either a double or a Vc::double_v, "tensor component type"
    using scalar_t = comp_t<CCTK_REAL>;

    // init tensor fields with GF pointers
    scalar_field_t<CCTK_REAL> const alpha_field(alp);
    scalar_field_t<CCTK_REAL> const w_lorentz_field(w_lorentz);
    scalar_field_t<CCTK_REAL> const rho_field(rho);
    scalar_field_t<CCTK_REAL> const eps_field(eps);
    scalar_field_t<CCTK_REAL> const press_field(press);

    scalar_field_t<CCTK_REAL> rhs_tau_field(rhs_tau_temp);

    tensor_field_t<vector3_t<CCTK_REAL>>
      const beta_field(betax,betay,betaz);

    tensor_field_t<vector3_t<CCTK_REAL>>
      const vel_field(&vel[0*gsiz],
                      &vel[1*gsiz],
                      &vel[2*gsiz]);

    tensor_field_t<covector3_t<CCTK_REAL>>
      rhs_sources(&rhs_scon_temp[0*gsiz],
                  &rhs_scon_temp[1*gsiz],
                  &rhs_scon_temp[2*gsiz]);

    tensor_field_t<metric_tensor3_t<CCTK_REAL>>
      const gamma_field(gxx,gxy,gxz,
                            gyy,gyz,
                                gzz);

    tensor_field_t<sym_tensor3_t<CCTK_REAL,0,1,lower_t,lower_t>>
      const K_field(kxx,kxy,kxz,
                        kyy,kyz,
                            kzz);
    // get grid spacing
    CCTK_REAL const dx  = CCTK_DELTA_SPACE(0);
    CCTK_REAL const dy  = CCTK_DELTA_SPACE(1);
    CCTK_REAL const dz  = CCTK_DELTA_SPACE(2);

    // create a (specialized) cactus central differentiator, in this case a central FD for the 1st derivative
    fd::cactus_diff<1,fd_order,fd::central_nodes> cdiff(cctkGH,dx,dy,dz);

    // loop over local grid
    // CHECK: define a macro for this loop, order and increment are important
    #pragma omp parallel for
    for(int k = cctk_nghostzones[2]; k < cctk_lsh[2]-cctk_nghostzones[2]; ++k)
    for(int j = cctk_nghostzones[1]; j < cctk_lsh[1]-cctk_nghostzones[1]; ++j)
    // force inlining helps Intel compiler to generate optimal code (substituting all expressions)
    #pragma forceinline recursive
    // CHECK: this loop shoots over into the ghost zones for vectorized code,
    //        one should check that loop_inc < cctk_nghostzones[0] + 1
    for(int i = cctk_nghostzones[0]; i < cctk_lsh[0]-cctk_nghostzones[0]; i += inc) {
      // get compressed GF index
      int const ijk = CCTK_GFINDEX3D(cctkGH, i, j, k);

      // 3-metric and its determinant
      // CHECK: evaluate doesn't work here (yet), thus the explicit tensor type is given here
      metric_tensor3_t<CCTK_REAL> const gamma = gamma_field[ijk];
      scalar_t const det = gamma.det();
      scalar_t const sqrt_det = std::sqrt(det);

      // if vectorization is enabled, we want to load from memory only once
      auto const beta = evaluate(beta_field[ijk]);
      auto const alpha = evaluate(alpha_field[ijk]);

      // inverse 4-metric, needed for energy-momentum tensor
      auto const inv_g = gamma.inverse_spacetime_metric(alpha,beta,det);

      // derivatives of the lapse, metric and shift
      #ifdef COMPUTE_DERIVATIVES
      covector3_t<CCTK_REAL> const dalp = alpha_field[ijk].finite_diff(cdiff);
      #else
      covector3_t<CCTK_REAL> const dalp;
      #endif

      // get central finite difference of beta components, defined by cdiff
      #ifdef COMPUTE_DERIVATIVES
      tensor3_t<CCTK_REAL,upper_t,lower_t> const dbeta = beta_field[ijk].finite_diff(cdiff);
      #else
      tensor3_t<CCTK_REAL,upper_t,lower_t> const dbeta;
      #endif

      #ifdef COMPUTE_DERIVATIVES
      sym_tensor3_t<CCTK_REAL,0,1,lower_t,lower_t,lower_t> const dgamma = gamma_field[ijk].finite_diff(cdiff);
      #else
      sym_tensor3_t<CCTK_REAL,0,1,lower_t,lower_t,lower_t> const dgamma;
      #endif

      // helpers to construct four metric derivative
      auto const dg00i = - 2*alpha*dalp
                         + 2*contract(dbeta,contract(gamma,beta))
                         + contract(beta,contract(beta,dgamma));

      auto const dg0ji = contract(gamma,dbeta)
                       + contract(dgamma,beta);

      // gj0i is the same due to symmetry
      // gjki is dgamma

      // four metric derivative
      // initialized to zero, first two indices are symmetric
      sym_tensor4_t<CCTK_REAL,0,1,lower_t,lower_t,lower_t> dg;

      // g00,i
      dg.set<0,0,-2>(dg00i);

      // g0j,i
      dg.set<0,-2,-2>(dg0ji);

      // gjk,i
      dg.set<-2,-2,-2>(dgamma);

      // helper for four velocity u
      auto const u0 =  evaluate(w_lorentz_field[ijk])/alpha;
      auto const ui = u0 * (alpha * vel_field[ijk] - beta);

      // four velocity
      vector4_t<CCTK_REAL> u;
      u.c<0>() = u0;
      u.set<-2>(ui);

      // construct tensor product of u^mu u^nu
      auto const uu = sym2_cast(tensor_cat(u,u));

      // implicit conversion doesn't work, have to cast explicitly
      auto const pressure = evaluate(press_field[ijk]);
      auto const density = evaluate(rho_field[ijk]);
      auto const int_energy = evaluate(eps_field[ijk]);

      // construct energy-momentum tensor
      sym_tensor4_t<CCTK_REAL,0,1,upper_t,upper_t> const
        T = (density * (1 + int_energy) + pressure) * uu
          + pressure * inv_g;

      // slice expression to contain only the spatial components
      // hence the result is a covector3_t
      rhs_sources[ijk] = slice<-2>(0.5*alpha * sqrt_det
                                   *(trace(contract(T,dg))));

      // slice em tensor in different ways
      // this is a vector3_t
      auto const T0i = slice<0,-2>(T);

      // this is a tensor3_t of upper_t,upper_t indices
      auto const Tij = sym2_cast(slice<-2,-2>(T));

      // construct tensor products
      auto const bb = sym2_cast(tensor_cat(beta,beta));
      auto const T0ib = tensor_cat(T0i,beta);

      // the operations result in a scalar or vector register
      rhs_tau_field[ijk] = alpha * sqrt_det
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
