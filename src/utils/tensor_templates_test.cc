#include <cassert>
#include <cmath>
#include <iostream>

//#ifndef TENSORS_DEBUG
//#define TENSORS_DEBUG
#include <tensor_templates.hh>
#include <utils.hh>
//#endif

#define DELTA(a,b) ((a)==(b) ? 1 : 0)
#define TEST_EPSILON 1e-14
//#define TEST_VERBOSE

int main(void) {
    std::cout << "Testing symmetric2 tensor..." << std::endl;
    // Test rank 2 symmetric tensor
    tensors::symmetric2<CCTK_REAL, 4, 2> A;
    for(int i = 0; i < tensors::symmetric2<CCTK_REAL, 4, 2>::ndof; ++i) {
        A[i] = i;
    }
    for(int a = 0; a < 4; ++a)
    for(int b = 0; b < 4; ++b) {
        assert(A(a,b) == A(b,a));
    }
#ifdef TEST_VERBOSE
    std::cout << std::endl << "symmetric2<CCTK_REAL, 4, 2>" << std::endl;
    for(int a = 0; a < 4; ++a) {
        for(int b = 0; b < 4; ++b) {
            int const i = static_cast<int>(&A(a,b) - &A[0]);
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << "symmetric2<CCTK_REAL, 3, 2>" << std::endl;
    tensors::symmetric2<3, 2> B;
    for(int a = 0; a < 3; ++a) {
        for(int b = 0; b < 3; ++b) {
            int const i = static_cast<int>(&B(a,b) - &B[0]);
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }
#endif

    std::cout << "Testing rank 3 symmetric2 tensor..." << std::endl;
    // Test rank 3 symmetric tensor
    tensors::symmetric2<CCTK_REAL, 4, 3> Gamma;
    for(int i = 0; i < tensors::symmetric2<CCTK_REAL, 4, 3>::ndof; ++i) {
        Gamma[i] = i;
    }
    for(int a = 0; a < 4; ++a)
    for(int b = 0; b < 4; ++b)
    for(int c = 0; c < 4; ++c) {
        assert(Gamma(a,b,c) == Gamma(a,c,b));
    }
#ifdef TEST_VERBOSE
    std::cout << std::endl << "symmetric2<CCTK_REAL, 4, 3>" << std::endl;
    for(int a = 0; a < 4; ++a) {
        for(int b = 0; b < 4; ++b) {
            for(int c = 0; c < 4; ++c) {
                int const i = static_cast<int>(&Gamma(a,b,c) - &Gamma[0]);
                std::cout << i << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
#endif

    std::cout << "Test spacetime metric and inverse..." << std::endl;

    // Test spacetime metric and inverse
    CCTK_REAL alp = 0.8;
    CCTK_REAL betax[2] = {0.1, -0.1};
    CCTK_REAL betay[2] = {0.0, 0.1};
    CCTK_REAL betaz[2] = {-0.1, 0.0};
    CCTK_REAL gxx[2] = {0.9, 1.2};
    CCTK_REAL gxy[2] = {0.6, 0.3};
    CCTK_REAL gxz[2] = {0.0, 0.7};
    CCTK_REAL gyy[2] = {1.0, 2.0};
    CCTK_REAL gyz[2] = {0.1, 1.4};
    CCTK_REAL gzz[2] = {1.0, 2.8};

    int index = 0;
    tensors::metric<4> g;
    g.from_adm(alp, betax[index], betay[index], betaz[index],
               gxx[index], gxy[index], gxz[index],
               gyy[index], gyz[index], gzz[index]);
    tensors::inv_metric<4> u;
    u.from_adm(alp, betax[index], betay[index], betaz[index],
               gxx[index], gxy[index], gxz[index],
               gyy[index], gyz[index], gzz[index]);

    tensors::symmetric2<CCTK_REAL, 4, 2> id;
    for(int a = 0; a < 4; ++a) {
        for(int b = a; b < 4; ++b) {
            id(a, b) = 0.0;
            for(int c = 0; c < 4; ++c) {
                id(a, b) += g(a, c) * u(c, b);
            }
            assert(std::abs(id(a,b) - DELTA(a,b)) < TEST_EPSILON);
        }
    }

    std::cout << "Testing tensor field, contraction and trace..." << std::endl;

    // Test tensor field, contraction and trace

    std::cout << "Testing metric contractions..." << std::endl;
    auto id_0 = g.contract<0,0>(u);
    auto id_1 = g.contract<0,1>(u);
    auto id_2 = g.contract<1,0>(u);
    auto id_3 = g.contract<1,1>(u);

    auto kron = tensors::symmetric2<CCTK_REAL,4,2>::get_id();
    for(int a = 0; a < 4; ++a) {
        for(int b = 0; b < 4; ++b) {
            assert(std::abs(id_0(a,b) - kron(a,b)) < TEST_EPSILON);
            assert(std::abs(id_1(a,b) - kron(a,b)) < TEST_EPSILON);
            assert(std::abs(id_2(a,b) - kron(a,b)) < TEST_EPSILON);
            assert(std::abs(id_3(a,b) - kron(a,b)) < TEST_EPSILON);
        }
    }

    std::cout << "Testing id trace..." << std::endl;

    CCTK_REAL trace_id = id_0.trace<0,1>();
    assert((trace_id - 4.0) < TEST_EPSILON);

    std::cout << "Testing tensor fields..." << std::endl;

    tensors::metric_field_t<3> metric_field(gxx,gxy,gxz,gyy,gyz,gzz);
    tensors::metric<3> spatial_g = metric_field[index];
    tensors::inv_metric<3> spatial_u;
    spatial_u.from_metric(spatial_g);

    std::cout << "Testing spatial metric contraction..." << std::endl;
    auto id_spatial = spatial_g.contract<0,0>(spatial_u);
    for(int a = 0; a < 3; ++a) {
        for(int b = 0; b < 3; ++b) {
            assert(std::abs(id_spatial(a,b) - kron(a,b)) < TEST_EPSILON);
        }
    }
    std::cout << "Testing spatial metric trace..." << std::endl;
    CCTK_REAL trace_id_spatial = id_spatial.trace<0,1>();
    assert((trace_id_spatial - 3.0) < TEST_EPSILON);

    std::cout << "Testing sub tensor metric..." << std::endl;
    auto sub_metric = g.subtensor<3>();

    for(auto mi = spatial_g.get_mi(); !mi.end(); ++mi) {
        assert(spatial_g(mi) == sub_metric(mi));
    }

    std::cout << "Testing metric and vector contractions..." << std::endl;
    const int exp = 14;
    assert(spatial_g.compare_components<exp>(sub_metric));

    tensors::vector_field_t<CCTK_REAL,3> beta_field(betax,betay,betaz);
    tensors::vector_t<CCTK_REAL,3> beta = beta_field[index];

    for(auto mi = spatial_u.get_mi(); !mi.end(); ++mi) {
        assert(std::abs(spatial_u(mi) - (u(mi[0]+1,mi[1]+1)
                  +beta(mi[0])*beta(mi[1])/(alp*alp))) < TEST_EPSILON);
    }

    tensors::vector_t<CCTK_REAL,3> up_beta;
    up_beta.zero();

    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
        up_beta(a) += spatial_u(a,b)*beta(b);
    }
    auto up_beta_c = beta.contract<0,1>(spatial_u);

    for(int a = 0; a < 3; ++a) {
        assert(std::abs(up_beta(a) - up_beta_c(a)) < TEST_EPSILON);
    }
    assert(up_beta.compare_components<exp>(up_beta_c));
}
