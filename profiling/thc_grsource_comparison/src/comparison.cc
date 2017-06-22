#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"

#include "tensor_templates.hh"
#include "utils.hh"


extern "C" void THC_GRSource_comparison(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTS
    DECLARE_CCTK_PARAMETERS

    using namespace tensors;

    constexpr size_t exp = 16;
    constexpr double eps_err = 1.0/utilities::static_pow<10,exp>::value;
    size_t N = 0;

    int const gsiz = UTILS_GFSIZE(cctkGH);

    CCTK_REAL * rhs_sconx  = &rhs_scon[0*gsiz];
    CCTK_REAL * rhs_scony  = &rhs_scon[1*gsiz];
    CCTK_REAL * rhs_sconz  = &rhs_scon[2*gsiz];

    tensor_field_t<covector3_t<CCTK_REAL>> r_scon(rhs_sconx,
                                                  rhs_scony,
                                                  rhs_sconz);

    CCTK_REAL * rhs_sconx_orig  = &rhs_scon_orig[0*gsiz];
    CCTK_REAL * rhs_scony_orig  = &rhs_scon_orig[1*gsiz];
    CCTK_REAL * rhs_sconz_orig  = &rhs_scon_orig[2*gsiz];

    tensor_field_t<covector3_t<CCTK_REAL>> r_scon_orig(rhs_sconx_orig,
                                                       rhs_scony_orig,
                                                       rhs_sconz_orig);

        #pragma omp parallel for reduction(+:N)
        for(int k = cctk_nghostzones[2]; k < cctk_lsh[2]-cctk_nghostzones[2]; ++k)
        for(int j = cctk_nghostzones[1]; j < cctk_lsh[1]-cctk_nghostzones[1]; ++j)
        for(int i = cctk_nghostzones[0]; i < cctk_lsh[0]-cctk_nghostzones[0]; ++i) {

            int const ijk = CCTK_GFINDEX3D(cctkGH, i, j, k);
            covector3_t<CCTK_REAL> rhs_scon_temp(r_scon[ijk]);
            covector3_t<CCTK_REAL> rhs_scon_orig = r_scon_orig[ijk];

            if(rhs_scon_temp.norm() < cut)
              continue;

            auto compare_scon = rhs_scon_temp.compare_components<exp>(rhs_scon_orig);
            if(!compare_scon.first)
              CCTK_VInfo(CCTK_THORNSTRING,"rhs scon differ at (%i,%e,%e,%e): [%e,%e,%e] != [%e,%e,%e]",
                                           ijk, x[ijk], y[ijk], z[ijk],
                                           rhs_scon_temp.c<0>(),
                                           rhs_scon_temp.c<1>(),
                                           rhs_scon_temp.c<2>(),
                                           rhs_scon_orig.c<0>(),
                                           rhs_scon_orig.c<1>(),
                                           rhs_scon_orig.c<2>());

            N += 1;

            if(rhs_tau[ijk] < cut)
              continue;

            double rel_err_tau = 2*std::abs(rhs_tau[ijk]- rhs_tau_orig[ijk])
                                 /(std::abs(rhs_tau[ijk])+std::abs(rhs_tau_orig[ijk]));
            if(rel_err_tau > eps_err)
              CCTK_VInfo(CCTK_THORNSTRING,"rhs tau differ at (%i,%e,%e,%e): [%e] != [%e]",
                                           ijk, x[ijk], y[ijk], z[ijk],
                                           rhs_tau[ijk],
                                           rhs_tau_orig[ijk]);

        }
        CCTK_VInfo(CCTK_THORNSTRING,"Found %lu rscon vectors > %e",N,cut);
}
