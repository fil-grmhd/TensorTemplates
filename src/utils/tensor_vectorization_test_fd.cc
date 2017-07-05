#include <Vc/Vc>
#include <iostream>
#include <array>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

#define TENSORS_VECTORIZED
#include "../tensor_templates.hh"

#define SCA
#define VEC

using namespace tensors;

// some artificial diff class
struct test_diff {
  template<size_t dir>
  inline size_t stride() const {
    return dir+1;
  }

  template<size_t dir, typename T>
  inline T diff(T const * const ptr, size_t const index) const {
    constexpr size_t idx[3] = {1,1,1};
    return idx[dir]*fd::c_diff<dir,4>(ptr, index, this->stride<dir>());
  }
};


int main() {

  using sca_type = double;
//  using sca_type = float;

  using vec_type = Vc::Vector<sca_type>;

  test_diff diff;

//  constexpr size_t N = 80000000;
  constexpr size_t N = 80000;
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
/*
    data_0[i] = 1;
    data_1[i] = 2;
    data_2[i] = 3;

    data_000[i] = 4;
    data_010[i] = 5;
    data_020[i] = 6;
    data_110[i] = 7;
    data_120[i] = 8;
    data_220[i] = 9;

    data_000[i] = 10;
    data_010[i] = 11;
    data_020[i] = 12;
    data_110[i] = 13;
    data_120[i] = 14;
    data_220[i] = 15;
*/
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  auto t1 = std::chrono::high_resolution_clock::now();

  #ifdef SCA
  std::cout << "Starting scalar computations..." << std::endl;
  // scalar version
  t0 = std::chrono::high_resolution_clock::now();

{
  tensor_field_t<vector3_t<sca_type>> beta(&data_0[0],&data_1[0],&data_2[0]);

  tensor_field_t<covector3_t<sca_type>> rhs_scon(&result0[0],&result1[0],&result2[0]);

  tensor_field_t<metric_tensor_t<sca_type,3>> gamma(&data_000[0],&data_010[0],&data_020[0],
                                                                 &data_110[0],&data_120[0],
                                                                              &data_220[0]);

  tensor_field_t<metric_tensor_t<sca_type,3>> K(&data_001[0],&data_011[0],&data_021[0],
                                                             &data_111[0],&data_121[0],
                                                                          &data_221[0]);


  #pragma forceinline recursive
  for(int i = 0; i < num_data; ++i) {
    sca_type alp = data_0[i];

    // this is only because we have not special scalar type
    covector3_t<double> dalp(diff.diff<0>(&data_0[0],i),
                             diff.diff<1>(&data_0[0],i),
                             diff.diff<2>(&data_0[0],i));

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
  tensor_field_vt<vector3_vt<sca_type>> beta(&data_0[0],&data_1[0],&data_2[0]);

  tensor_field_vt<covector3_vt<sca_type>> rhs_scon(&result_vec0[0],&result_vec1[0],&result_vec2[0]);

  tensor_field_vt<metric_tensor_vt<sca_type,3>> gamma(&data_000[0],&data_010[0],&data_020[0],
                                                                   &data_110[0],&data_120[0],
                                                                                &data_220[0]);

  tensor_field_vt<metric_tensor_vt<sca_type,3>> K(&data_001[0],&data_011[0],&data_021[0],
                                                               &data_111[0],&data_121[0],
                                                                            &data_221[0]);

  #pragma forceinline recursive
  for(int i = 0; i < num_data; i += vec_size) {
    vec_type alp(&data_0[i]);

    // this is needed because we have no scalar field type
    vec_type dalp_vec0;
    vec_type dalp_vec1;
    vec_type dalp_vec2;

    for(size_t k = 0; k < vec_type::Size; ++k) {
      dalp_vec0[k] = diff.diff<0>(&data_0[0],i+k);
      dalp_vec1[k] = diff.diff<1>(&data_0[0],i+k);
      dalp_vec2[k] = diff.diff<2>(&data_0[0],i+k);
    }

    covector3_vt<sca_type> dalp(dalp_vec0,
                                dalp_vec1,
                                dalp_vec2);

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
