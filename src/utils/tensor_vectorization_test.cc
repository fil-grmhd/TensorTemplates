
#include <Vc/Vc>
#include <iostream>
#include <array>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

#include "../tensor_templates.hh"


int main() {
  using namespace tensors;

  using vec_type = Vc::double_v;
  using sca_type = double;
//  using vec_type = Vc::float_v;
//  using sca_type = float;

  constexpr size_t N = 3000;
  constexpr size_t vec_size = vec_type::Size;
  constexpr size_t num_data = N*vec_size;

  // some GF
  std::vector<sca_type> data_0(num_data);
  std::vector<sca_type> data_1(num_data);
  std::vector<sca_type> data_2(num_data);
  std::vector<sca_type> data_3(num_data);

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
    data_3[i] = dist_uni(gen);
  }

  // scalar version
  auto t0 = std::chrono::high_resolution_clock::now();
  #pragma forceinline recursive
  for(int i = 0; i < num_data; ++i) {
    sca_type alp = data_0[i];
    vector3_t<sca_type> beta(data_0[i],data_1[i],data_2[i]);
    metric_tensor_t<double,3> gamma(data_0[i],data_1[i],data_2[i],
                                              data_1[i],data_2[i],
                                                        data_2[i]);
    metric_tensor_t<double,3> K(data_0[i],data_1[i],data_2[i],
                                          data_1[i],data_2[i],
                                                    data_2[i]);

    covector3_t<double> dalp(data_0[i],data_1[i],data_2[i]);
    tensor3_t<double,upper_t,lower_t> dbeta(data_0[i],data_1[i],data_2[i],
                                            data_0[i],data_1[i],data_2[i],
                                            data_0[i],data_1[i],data_2[i]);

    sym_tensor3_t<double,0,1,lower_t,lower_t,lower_t> dgamma;

    dgamma.c<0,0,0>() = 1;
    dgamma.c<0,1,0>() = 1;
    dgamma.c<0,2,0>() = 1;
    dgamma.c<1,1,0>() = 1;
    dgamma.c<1,2,0>() = 1;
    dgamma.c<2,2,0>() = 1;
    dgamma.c<0,0,1>() = 1;
    dgamma.c<0,1,1>() = 1;
    dgamma.c<0,2,1>() = 1;
    dgamma.c<1,1,1>() = 1;
    dgamma.c<1,2,1>() = 1;
    dgamma.c<2,2,1>() = 1;
    dgamma.c<0,0,2>() = 1;
    dgamma.c<0,1,2>() = 1;
    dgamma.c<0,2,2>() = 1;
    dgamma.c<1,1,2>() = 1;
    dgamma.c<1,2,2>() = 1;
    dgamma.c<2,2,2>() = 1;

    auto dg00i = - 2* alp * dalp
                 + 2*contract(dbeta,contract(gamma,beta))
                 + contract(beta,contract(beta,dgamma));
    auto dg0ji = contract(gamma,dbeta)
               + contract(dgamma,beta);


    sym_tensor4_t<double,0,1,lower_t,lower_t,lower_t> dg;

    // g00,i
    dg.set<0,0,-2>(dg00i);

    // g0j,i
    dg.set<0,-2,-2>(dg0ji);

    // gjk,i
    dg.set<-2,-2,-2>(dgamma);

    vector4_t<double> u(data_0[i],data_1[i],data_2[i],data_3[i]);

    auto uu = sym2_cast(tensor_cat(u,u));

    sym_tensor4_t<double,0,1,upper_t,upper_t> invmetric(data_0[i],data_1[i],data_2[i],data_3[i],
                                                                  data_1[i],data_2[i],data_3[i],
                                                                            data_2[i],data_3[i],
                                                                                      data_3[i]);

    sym_tensor4_t<double,0,1,upper_t,upper_t> T = uu*1337 + 42.0*invmetric;

    auto rhs_scon = slice<-2>(0.5*alp*(trace(contract(T,dg))));

    result0[i] = rhs_scon.c<1>();
    result1[i] = rhs_scon.c<2>();
    result2[i] = rhs_scon.c<3>();

    auto T0i = slice<0,-2>(T);

    auto Tij = sym2_cast(slice<-2,-2>(T));

    auto bb = sym2_cast(tensor_cat(beta,beta));
    auto T0ib = tensor_cat(T0i,beta);

    result3[i] = alp
              * (trace(
                   contract(
                     T.c<0,0>()*bb + 2 * T0ib + Tij,
                     K
                   )
                 )
               - contract(
                   T.c<0,0>()*beta + T0i,
                   dalp
                 )
                );
  }
  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout << "Scalar: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << " ms" << std::endl;

  // vectorized version
  t0 = std::chrono::high_resolution_clock::now();
  #pragma forceinline recursive
  for(int i = 0; i < num_data; i += vec_size) {
    vec_type reg00(&data_0[i]);
    vec_type reg01(&data_1[i]);
    vec_type reg02(&data_2[i]);
    vec_type reg03(&data_3[i]);

    vec_type reg10(&data_0[i]);
    vec_type reg11(&data_1[i]);
    vec_type reg12(&data_2[i]);
    vec_type reg13(&data_3[i]);

    vec_type reg20(&data_0[i]);
    vec_type reg21(&data_1[i]);
    vec_type reg22(&data_2[i]);
    vec_type reg23(&data_3[i]);

    vec_type reg30(&data_0[i]);
    vec_type reg31(&data_1[i]);
    vec_type reg32(&data_2[i]);
    vec_type reg33(&data_3[i]);


    vec_type alp = reg00;
    vector3_t<vec_type> beta(reg00,reg01,reg02);
    metric_tensor_t<vec_type,3> gamma(reg00,reg01,reg02,
                                            reg11,reg12,
                                                  reg22);
    metric_tensor_t<vec_type,3> K(reg00,reg01,reg02,
                                        reg11,reg12,
                                              reg22);

    covector3_t<vec_type> dalp(reg00,reg01,reg02);

    tensor3_t<vec_type,upper_t,lower_t> dbeta(reg00,reg01,reg02,
                                              reg10,reg11,reg12,
                                              reg20,reg21,reg22);

    sym_tensor3_t<vec_type,0,1,lower_t,lower_t,lower_t> dgamma;

    dgamma.c<0,0,0>() = vec_type(1);
    dgamma.c<0,1,0>() = vec_type(1);
    dgamma.c<0,2,0>() = vec_type(1);
    dgamma.c<1,1,0>() = vec_type(1);
    dgamma.c<1,2,0>() = vec_type(1);
    dgamma.c<2,2,0>() = vec_type(1);
    dgamma.c<0,0,1>() = vec_type(1);
    dgamma.c<0,1,1>() = vec_type(1);
    dgamma.c<0,2,1>() = vec_type(1);
    dgamma.c<1,1,1>() = vec_type(1);
    dgamma.c<1,2,1>() = vec_type(1);
    dgamma.c<2,2,1>() = vec_type(1);
    dgamma.c<0,0,2>() = vec_type(1);
    dgamma.c<0,1,2>() = vec_type(1);
    dgamma.c<0,2,2>() = vec_type(1);
    dgamma.c<1,1,2>() = vec_type(1);
    dgamma.c<1,2,2>() = vec_type(1);
    dgamma.c<2,2,2>() = vec_type(1);

    auto dg00i = - 2* alp * dalp
                 + 2*contract(dbeta,contract(gamma,beta))
                 + contract(beta,contract(beta,dgamma));
    auto dg0ji = contract(gamma,dbeta)
               + contract(dgamma,beta);


    sym_tensor4_t<vec_type,0,1,lower_t,lower_t,lower_t> dg;

    // g00,i
    dg.set<0,0,-2>(dg00i);

    // g0j,i
    dg.set<0,-2,-2>(dg0ji);

    // gjk,i
    dg.set<-2,-2,-2>(dgamma);

    vector4_t<vec_type> u(reg00,reg01,reg02,reg03);

    auto uu = sym2_cast(tensor_cat(u,u));

    sym_tensor4_t<vec_type,0,1,upper_t,upper_t> invmetric(reg00,reg01,reg02,reg03,
                                                                reg11,reg12,reg13,
                                                                      reg22,reg23,
                                                                            reg33);

    sym_tensor4_t<vec_type,0,1,upper_t,upper_t> T = uu*1337 + 42.0*invmetric;

    auto rhs_scon = slice<-2>(0.5*alp*(trace(contract(T,dg))));

    vec_type reg_result0 = rhs_scon.c<1>();
    vec_type reg_result1 = rhs_scon.c<2>();
    vec_type reg_result2 = rhs_scon.c<3>();

    auto T0i = slice<0,-2>(T);

    auto Tij = sym2_cast(slice<-2,-2>(T));

    auto bb = sym2_cast(tensor_cat(beta,beta));
    auto T0ib = tensor_cat(T0i,beta);

    vec_type reg_result3 = alp
              * (trace(
                   contract(
                     T.c<0,0>()*bb + 2 * T0ib + Tij,
                     K
                   )
                 )
               - contract(
                   T.c<0,0>()*beta + T0i,
                   dalp
                 )
                );

    reg_result0.store(&result_vec0[i]);
    reg_result1.store(&result_vec1[i]);
    reg_result2.store(&result_vec2[i]);
    reg_result3.store(&result_vec3[i]);
  }
  t1 = std::chrono::high_resolution_clock::now();

  std::cout << "Vector: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << " ms" << std::endl;

  for(int i = 0; i < num_data; i+=num_data/10) {
    std::cout << "s: " << result0[i] << " " << result1[i] << " " << result2[i] << " " << result3[i] << std::endl;
    std::cout << "v: " << result_vec0[i] << " " << result_vec1[i] << " " << result_vec2[i] << " " << result_vec3[i] << std::endl;
  }

}
