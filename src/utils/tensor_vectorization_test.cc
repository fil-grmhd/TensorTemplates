#include <Vc/Vc>
#include <iostream>
#include <array>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <limits>

#define TENSORS_VECTORIZED
#include "../tensor_templates.hh"

#define SCA
#define VEC
#define COMPUTE_DERIVATIVES

using namespace tensors;

// some artificial diff classes
// scalar version
template<typename node_t, size_t order>
struct test_diff {
  const double idx;
  const int stride[3];

  test_diff() : idx(0.1), stride {1,2,3} {}

  template<size_t dir, typename T>
  inline __attribute__ ((always_inline)) T diff(T const * const ptr, size_t const index) const {
    return idx*fd::auto_diff<1,order,node_t>(ptr, index, stride[dir]);
  }
};
// version how it is implemented atm, without vectorization of the FD terms
template<typename node_t, size_t order>
struct test_diff_v {
  double idx;
  const int stride[3];

  test_diff_v() : idx(0.1), stride {1,2,3} {}

  template<size_t dir, typename T>
  inline __attribute__ ((always_inline)) decltype(auto) diff(T const * const ptr, size_t const index) const {
    Vc::Vector<T> vec_register;
    for(size_t i = 0; i < Vc::Vector<T>::Size; ++i) {
      vec_register[i] = fd::auto_diff<1,order,node_t>(ptr, index + i, stride[dir]);
    }
    return idx*vec_register;
  }
};


int main() {
  std::cout.setf(std::ios_base::scientific, std::ios_base::floatfield);
  std::cout.precision(std::numeric_limits<double>::max_digits10);

  using sca_type = double;
//  using sca_type = float;

  using vec_type = Vc::Vector<sca_type>;
  using node_t = fd::central_nodes;
  constexpr size_t fd_order = 8;

//  constexpr size_t N = 80000000;
  constexpr size_t N = 800000;
  constexpr size_t vec_size = vec_type::Size;
  constexpr size_t num_data = N*vec_size;

  // some GF
  std::vector<sca_type> data_0(num_data);
  std::vector<sca_type> data_1(num_data);
  std::vector<sca_type> data_2(num_data);

  std::vector<sca_type> data_000(num_data);
  std::vector<sca_type> data_010(num_data);
  std::vector<sca_type> data_020(num_data);
  std::vector<sca_type> data_110(num_data);
  std::vector<sca_type> data_120(num_data);
  std::vector<sca_type> data_220(num_data);

  std::vector<sca_type> data_001(num_data);
  std::vector<sca_type> data_011(num_data);
  std::vector<sca_type> data_021(num_data);
  std::vector<sca_type> data_111(num_data);
  std::vector<sca_type> data_121(num_data);
  std::vector<sca_type> data_221(num_data);

  // some result GF for scalar and vector version
  std::vector<sca_type> result0(num_data);
  std::vector<sca_type> result1(num_data);
  std::vector<sca_type> result2(num_data);
  std::vector<sca_type> result3(num_data);

  std::vector<sca_type> result_vec0(num_data);
  std::vector<sca_type> result_vec1(num_data);
  std::vector<sca_type> result_vec2(num_data);
  std::vector<sca_type> result_vec3(num_data);

  // fill GFs with random data
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_real_distribution<> dist_uni(0,1);

  for(int i = 0; i < num_data; ++i) {
    data_0[i] = dist_uni(gen);
    data_1[i] = dist_uni(gen);
    data_2[i] = dist_uni(gen);

    data_000[i] = dist_uni(gen);
    data_010[i] = dist_uni(gen);
    data_020[i] = dist_uni(gen);
    data_110[i] = dist_uni(gen);
    data_120[i] = dist_uni(gen);
    data_220[i] = dist_uni(gen);

    data_000[i] = dist_uni(gen);
    data_010[i] = dist_uni(gen);
    data_020[i] = dist_uni(gen);
    data_110[i] = dist_uni(gen);
    data_120[i] = dist_uni(gen);
    data_220[i] = dist_uni(gen);
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  auto t1 = std::chrono::high_resolution_clock::now();

  #ifdef SCA
  std::cout << "Starting scalar computations..." << std::endl;
  // scalar version
  t0 = std::chrono::high_resolution_clock::now();

{
  test_diff<node_t,fd_order> diff;

  scalar_field_t<sca_type> alpha_field(&data_0[0]);

  tensor_field_t<vector3_t<sca_type>> beta(&data_0[0],&data_1[0],&data_2[0]);

  tensor_field_t<covector3_t<sca_type>> rhs_scon(&result0[0],&result1[0],&result2[0]);

  tensor_field_t<metric_tensor3_t<sca_type>> gamma(&data_000[0],&data_010[0],&data_020[0],
                                                                &data_110[0],&data_120[0],
                                                                             &data_220[0]);

  tensor_field_t<metric_tensor3_t<sca_type>> K(&data_001[0],&data_011[0],&data_021[0],
                                                            &data_111[0],&data_121[0],
                                                                         &data_221[0]);


  #pragma forceinline recursive
  for(int i = 0; i < num_data; ++i) {
    sca_type alp = alpha_field[i];

    // this is only because we have not special scalar type
    covector3_t<double> dalp = alpha_field[i].finite_diff(diff);

/*
    covector3_t<double> dalp(diff.diff<0>(&data_0[0],i),
                             diff.diff<1>(&data_0[0],i),
                             diff.diff<2>(&data_0[0],i));
*/
    tensor3_t<double,upper_t,lower_t> dbeta = beta[i].finite_diff(diff);

    sym_tensor3_t<double,0,1,lower_t,lower_t,lower_t> dgamma = gamma[i].finite_diff(diff);

// this fails with gcc
    auto dg00i = - 2* alp * dalp
                 + 2*contract(dbeta,contract(gamma[i],beta[i]))
                 + contract(beta[i],contract(beta[i],dgamma));
    auto dg0ji = contract(gamma[i],dbeta)
               + contract(dgamma,beta[i]);

    sym_tensor4_t<double,0,1,lower_t,lower_t,lower_t> dg;

    // g00,i
    dg.set<0,0,-2>(dg00i);

    // g0j,i
    dg.set<0,-2,-2>(dg0ji);

    // gjk,i
    dg.set<-2,-2,-2>(dgamma);

    vector4_t<double> u(1,2,3,4);

// this fails with gcc
    auto uu = sym2_cast(tensor_cat(u,u));

    sym_tensor4_t<double,0,1,upper_t,upper_t> invmetric(1,1,1,1,
                                                          2,2,2,
                                                            3,3,
                                                              4);

    sym_tensor4_t<double,0,1,upper_t,upper_t> T = uu*1337 + 42.0*invmetric;

// this fails with gcc
    rhs_scon[i] = slice<-2>(0.5*alp*(trace(contract(T,dg))));

    auto T0i = slice<0,-2>(T);

    auto Tij = sym2_cast(slice<-2,-2>(T));

    auto bb = sym2_cast(tensor_cat(beta[i],beta[i]));
    auto T0ib = tensor_cat(T0i,beta[i]);

    result3[i] = alp
              * (trace(
                   contract(
                     T.c<0,0>()*bb + 2 * T0ib + Tij,
                     K[i]
                   )
                 )
               - contract(
                   T.c<0,0>()*beta[i] + T0i,
                   dalp
                 )
                );
  }
}
  t1 = std::chrono::high_resolution_clock::now();

  auto t_scalar = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
  std::cout << "Scalar: " << t_scalar << " ms" << std::endl;
  #endif

  #ifdef VEC
  std::cout << "Starting vector computations..." << std::endl;
  // vectorized version
  t0 = std::chrono::high_resolution_clock::now();

{
  test_diff_v<node_t,fd_order> diff;

  scalar_field_vt<sca_type> alpha_field(&data_0[0]);

  tensor_field_vt<vector3_vt<sca_type>> beta(&data_0[0],&data_1[0],&data_2[0]);

  tensor_field_vt<covector3_vt<sca_type>> rhs_scon(&result_vec0[0],&result_vec1[0],&result_vec2[0]);

  tensor_field_vt<metric_tensor3_vt<sca_type>> gamma(&data_000[0],&data_010[0],&data_020[0],
                                                                  &data_110[0],&data_120[0],
                                                                               &data_220[0]);

  tensor_field_vt<metric_tensor3_vt<sca_type>> K(&data_001[0],&data_011[0],&data_021[0],
                                                              &data_111[0],&data_121[0],
                                                                           &data_221[0]);

  #pragma forceinline recursive
  for(int i = 0; i < num_data; i += vec_size) {
    vec_type alp = alpha_field[i];

    covector3_vt<sca_type> dalp = alpha_field[i].finite_diff(diff);

/*
    covector3_vt<sca_type> dalp(diff.diff<0>(&data_0[0],i),
                                diff.diff<1>(&data_0[0],i),
                                diff.diff<2>(&data_0[0],i));
*/
    tensor3_vt<sca_type,upper_t,lower_t> dbeta = beta[i].finite_diff(diff);

    sym_tensor3_vt<sca_type,0,1,lower_t,lower_t,lower_t> dgamma = gamma[i].finite_diff(diff);

    auto dg00i = - 2* alp * dalp
                 + 2*contract(dbeta,contract(gamma[i],beta[i]))
                 + contract(beta[i],contract(beta[i],dgamma));
    auto dg0ji = contract(gamma[i],dbeta)
               + contract(dgamma,beta[i]);


    sym_tensor4_vt<sca_type,0,1,lower_t,lower_t,lower_t> dg;

    // g00,i
    dg.set<0,0,-2>(dg00i);

    // g0j,i
    dg.set<0,-2,-2>(dg0ji);

    // gjk,i
    dg.set<-2,-2,-2>(dgamma);

    vector4_vt<sca_type> u(vec_type(1),vec_type(2),vec_type(3),vec_type(4));

    auto uu = sym2_cast(tensor_cat(u,u));

    sym_tensor4_vt<sca_type,0,1,upper_t,upper_t> invmetric(vec_type(1),vec_type(1),vec_type(1),vec_type(1),
                                                                       vec_type(2),vec_type(2),vec_type(2),
                                                                                   vec_type(3),vec_type(3),
                                                                                               vec_type(4));

    sym_tensor4_vt<sca_type,0,1,upper_t,upper_t> T = uu*1337 + 42.0*invmetric;

    rhs_scon[i] = slice<-2>(0.5*alp*(trace(contract(T,dg))));

    auto T0i = slice<0,-2>(T);

    auto Tij = sym2_cast(slice<-2,-2>(T));

    auto bb = sym2_cast(tensor_cat(beta[i],beta[i]));
    auto T0ib = tensor_cat(T0i,beta[i]);

    vec_type reg_result3 = alp
              * (trace(
                   contract(
                     T.c<0,0>()*bb + 2 * T0ib + Tij,
                     K[i]
                   )
                 )
               - contract(
                   T.c<0,0>()*beta[i] + T0i,
                   dalp
                 )
                );

    reg_result3.store(&result_vec3[i]);
  }
}
  t1 = std::chrono::high_resolution_clock::now();

  auto t_vector = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
  std::cout << "Vector: " << t_vector << " ms" << std::endl;
  #endif

  #ifdef SCA
  #ifdef VEC
  std::cout << "t_vector / t_scalar = " << (double) t_vector/t_scalar << std::endl;
  #endif
  #endif

  for(int i = 0; i < num_data; i+=num_data / (dist_uni(gen) * 10 + 5)) {
    std::cout << "s(" << i << "): " << result0[i] << " " << result1[i] << " " << result2[i] << " " << result3[i] << std::endl;
    std::cout << "v(" << i << "): " << result_vec0[i] << " " << result_vec1[i] << " " << result_vec2[i] << " " << result_vec3[i] << std::endl;
  }
  std::cin.ignore();
  return 0;
}
