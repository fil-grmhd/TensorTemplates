/*
 * =====================================================================================
 *
 *       Filename:  expression_templates_example.cc
 *
 *    Description:  Sample code from wikipedia for expression templates
 *    		    updated to C++14 and supporting also scalar multiplication
 *
 *        Version:  1.0
 *        Created:  01/06/2017 21:32:32
 *       Revision:  none
 *       Compiler:  clang++ 4.0
 *
 *         Author:  Wikipedia and Elias R. Most
 *
 * =====================================================================================
 */

#include<iostream>
#include<typeinfo>

// nothing vectorized yet
//#define TENSORS_VECTORIZED
#include "../src/tensor_templates.hh"


int main(){

  using namespace tensors;

  using gen_t = generic_symmetry_t<5,4>;
  using sym_t = sym2_symmetry_t<5,4,1,3>;

  constexpr size_t gen_ndof = gen_t::ndof;
  constexpr size_t sym_ndof = sym_t::ndof;

  std::cout << "2,4,1,0" << std::endl;
  constexpr size_t gen_index = gen_t::compressed_index<2,4,1,0>::value;
  constexpr size_t gen_index_flip = gen_t::compressed_index<2,0,1,4>::value;
  constexpr size_t sym_index = sym_t::compressed_index<2,4,1,0>::value;

  constexpr size_t gen_trans_gen = gen_t::index_from_generic<gen_index>::value;
  constexpr size_t gen_trans_sym = gen_t::index_to_generic<gen_index>::value;

  constexpr size_t sym_trans_gen = sym_t::index_from_generic<gen_index>::value;
  constexpr size_t sym_trans_sym = sym_t::index_to_generic<sym_index>::value;

  constexpr size_t gen0 = gen_t::uncompress_index<0,gen_index>::value;
  constexpr size_t gen1 = gen_t::uncompress_index<1,gen_index>::value;
  constexpr size_t gen2 = gen_t::uncompress_index<2,gen_index>::value;
  constexpr size_t gen3 = gen_t::uncompress_index<3,gen_index>::value;

  constexpr size_t sym0 = sym_t::uncompress_index<0,sym_index>::value;
  constexpr size_t sym1 = sym_t::uncompress_index<1,sym_index>::value;
  constexpr size_t sym2 = sym_t::uncompress_index<2,sym_index>::value;
  constexpr size_t sym3 = sym_t::uncompress_index<3,sym_index>::value;

  std::cout << "Ndof: " << gen_ndof << " , " << sym_ndof << std::endl;

  std::cout << "Compressed: " << gen_index << " , " << gen_index_flip << " , " << sym_index << std::endl;

  std::cout << "Generic trafo: " << gen_trans_gen << " , " << gen_trans_sym << std::endl;
  std::cout << "Symmetric trafo: " << sym_trans_gen << " , " << sym_trans_sym << std::endl;

  std::cout << "Generic uncompressed: " << gen0 << " , " << gen1 << " , " << gen2 << " , " << gen3 << std::endl;
  std::cout << "Symmetric uncompressed: " << sym0 << " , " << sym1 << " , " << sym2 << " , " << sym3 << std::endl;


  vector3_t<double> a; a[0] =1; a[1] =2; a[2]=3;
  vector3_t<double> b; b[0] =3; b[1]= 2; b[2] =1;

  vector3_t<double> c = a+b;
  vector3_t<double> d = 2*a+ b*2.0 - c/1;

  std::cout << "c :" << c[0] <<" , "<< c[1] << " , " << c[2] <<std::endl;
  std::cout << "d :" << d[0] <<" , "<< d[1] << " , " << d[2] <<std::endl;


  tensor3_t<double, upper_t, lower_t> M;

  int nn=1;
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      M[i+3*j] = nn++;
  }

  std::cout<< std::tuple_size<typename decltype(M)::index_t>::value << " " << std::tuple_size<typename decltype(a)::index_t>::value <<std::endl;

  vector3_t<double> e = contract<1,0>(M,a);

  std::cout << "M = " << std::endl;
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      std::cout << " "<< M[i+3*j] << " ";
    std::cout<<std::endl;
  }

  std::cout << "a :" << a[0] <<" , "<< a[1] << " , " << a[2] <<std::endl;
  std::cout << "e = M*a :" << e[0] <<" , "<< e[1] << " , " << e[2] <<std::endl;



  vector3_t<double> f = contract<1,0>(M,a) + c - 3.15*a +b/2.;

  std::cout << "f = M*a +c - 3.15*a + b/2" <<std::endl;
  std::cout << "f :" << f[0] <<" , "<< f[1] << " , " << f[2] <<std::endl;
  std::cout<<"According to Matlab:\n" << "f : 16.35 , 30.7 , 45.05" <<std::endl;


  tensor3_t<double, upper_t, lower_t> A,B;

  A[0] = 1; A[1] = 14.5; A[2] = 0.1278;
  A[3] = -12;  A[4] = 7834; A[5] = 0.002;
  A[6] = -8; A[7] = 1.e-4; A[8] = 919;

  B[0] = 0; B[1] = 1.5; B[2] = 34;
  B[3] = -3;  B[4] = 7834; B[5] = 5404;
  B[6] = -3459; B[7] = 2300; B[8] = 93;


  std::cout << "A = " << std::endl;
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      std::cout << " "<< A[i+3*j] << " ";
    std::cout<<std::endl;
  }

  std::cout << "B = " << std::endl;
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      std::cout << " "<< B[i+3*j] << " ";
    std::cout<<std::endl;
  }

  tensor3_t<double, upper_t, lower_t> C=contract<1,0>(A,A);
  tensor3_t<double, upper_t, lower_t> D= contract<1,0>(A,B);

  std::cout << "C = " << std::endl;
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      std::cout << " "<< C[i+3*j] << " ";
    std::cout<<std::endl;
  }

  std::cout << "D = " << std::endl;
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      std::cout << " "<< D[i+3*j] << " ";
    std::cout<<std::endl;
  }

  D = tensor3_t<double, upper_t, lower_t>(11,21,31,
                                          12,22,32,
                                          13,23,33);
  std::cout << "D = " << std::endl;
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      std::cout << " "<< D[i+3*j] << " ";
    std::cout<<std::endl;
  }

  sym_tensor3_t<double,0,1,upper_t,lower_t> D_sym(D);

  std::cout << "D_sym = " << D_sym << std::endl;

  auto D_spatial = evaluate(slice<-2,-2>(D));

  std::cout << "D_sub = " << std::endl;
  for(int i=0; i<2; ++i){
    for(int j=0; j<2; ++j)
      std::cout << " "<< D_spatial[i+2*j] << " ";
    std::cout<<std::endl;
  }

  auto D_sym_spatial = evaluate(generic_cast(evaluate(slice<-2,-2>(D_sym))));

  std::cout << "D_sym_sub = " << std::endl;
  for(int i=0; i<2; ++i){
    for(int j=0; j<2; ++j)
      std::cout << " "<< D_sym_spatial[i+2*j] << " ";
    std::cout<<std::endl;
  }


  std::cout<<std::endl;
  tensor3_t<double,lower_t,upper_t> D_T = reorder_index<1,0>(D);
  std::cout << "D " << D <<std::endl;
  std::cout<<std::endl;
  std::cout << "D reordered" << D_T <<std::endl;

  std::cout<<std::endl;
  std::cout << "D_sym -1 0 slice" << evaluate(slice<-1,0>(D_sym)) << std::endl;
  std::cout<<std::endl;

  std::cout<<std::endl;
  std::cout << "D_sym 0 -1 slice" << evaluate(slice<0,-1>(D_sym)) << std::endl;
  std::cout<<std::endl;

  std::cout<<std::endl;
  std::cout << "D_sym 1 -1 slice" << evaluate(slice<1,-1>(D_sym)) << std::endl;
  std::cout<<std::endl;

  std::cout<<std::endl;
  std::cout << "D_sym 2 -1 slice" << evaluate(slice<2,-1>(D_sym)) << std::endl;
  std::cout<<std::endl;

  std::cout<<std::endl;
  std::cout << "D -1 0 slice" << evaluate(slice<-1,0>(D)) << std::endl;
  std::cout<<std::endl;

  std::cout<<std::endl;
  std::cout << "D -1 1 slice" << evaluate(slice<-1,1>(D)) << std::endl;
  std::cout<<std::endl;

  std::cout<<std::endl;
  std::cout << "D -1 -1 slice" << evaluate(slice<-1,-1>(D)) << std::endl;
  std::cout<<std::endl;

  std::cout<<std::endl;
  std::cout << "D -1 2 slice" << evaluate(slice<-1,2>(D)) << std::endl;
  std::cout<<std::endl;

  std::cout<<std::endl;
  std::cout << "D -2 0 slice" << evaluate(slice<-2,0>(D)) << std::endl;
  std::cout<<std::endl;

  std::cout << "D 2 -1 slice" << evaluate(slice<2,-1>(D)) << std::endl;
  std::cout<<std::endl;

  std::cout << "D 0 -2 slice" << evaluate(slice<0,-2>(D)) << std::endl;
  std::cout<<std::endl;

  std::cout << "D -2 -2 slice" << evaluate(slice<-2,-2>(D)) << std::endl;
  std::cout<<std::endl;

  tensor3_t<double, upper_t, lower_t> E(D);
  E.set<0,-1>(a);

  std::cout << "E = " << std::endl;
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      std::cout << " "<< E[i+3*j] << " ";
    std::cout<<std::endl;
  }

  E.set<-1,2>(a);

  std::cout << "E = " << std::endl;
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      std::cout << " "<< E[i+3*j] << " ";
    std::cout<<std::endl;
  }

  tensor4_t<double, upper_t, lower_t> F;

  std::cout << "F = " << std::endl;
  for(int i=0; i<4; ++i){
    for(int j=0; j<4; ++j)
      std::cout << " "<< F[i+4*j] << " ";
    std::cout<<std::endl;
  }

//  F.set<-1,-1>(E);
//  F.set<-2,-2>(E);
  F.set<-1,-2>(E);

  std::cout << "F = " << std::endl;
  for(int i=0; i<4; ++i){
    for(int j=0; j<4; ++j)
      std::cout << " "<< F[i+4*j] << " ";
    std::cout<<std::endl;
  }

  std::cout << "F(1,2) = " << F.c<1,2>() << std::endl;

  auto G = evaluate(slice<-1,1>(F));
  std::cout << "G(1) = " << G.c<1>() << std::endl;
  G.c<1>() = 1337;
  std::cout << "G(1) = " << G.c<1>() << std::endl;

  tensor3_t<double,upper_t,upper_t> ab = tensor_cat(a,b);
  std::cout << "Concatenating a and b: " << ab << std::endl;

/////////////////////////////////////////////////////////////////////
//
//            METRIC TESTS
//
/////////////////////////////////////////////////////////////////////
/* old interface, needs to be updated to the new metric type

//Flat space-time test
double lapse = 1.;
vector3_t<double> shift {1,2,3};

metric_tensor_t<double,3> mt{1, 0, -0.3,
                                1,  0,
                                    1};

//metric3_t metric(lapse, std::move(shift), std::move(mt));
auto metric = make_metric3(lapse, std::move(shift), std::move(mt));

std::cout << "Inverse metric" << metric.invmetric <<std::endl;
std::cout << "metric" << metric.metric <<std::endl;
std::cout << "sqrt(gamma)" << metric.sqrtdet <<std::endl;

vector3_t<double> am1 {11.,12.,13.};
covector3_t<double> bm1 = lower_index<0>(metric, am1);
vector3_t<double> cm1 = raise_index<0>(metric, bm1);

std::cout << "am1 upper_t" << am1 <<std::endl;
std::cout << "am1 lower_t" << bm1 <<std::endl;
std::cout << "raised again" << cm1 <<std::endl;

double norm2_am1 = contract<0,0>(metric,am1,am1);

std::cout << "norm2 am1 :" << norm2_am1 <<std::endl;

// 4-dim metric

//metric4_t metric4(lapse, std::move(shift), std::move(mt));
auto metric4 = make_metric4(lapse, std::move(shift), std::move(mt));

std::cout << "Inverse metric4" << metric4.invmetric <<std::endl;
std::cout << "metric4" << metric4.metric <<std::endl;
std::cout << "sqrt(gamma)" << metric4.sqrtdet <<std::endl;

vector4_t<double> am14 {-11.,12.,13.,14.};
covector4_t<double> bm14 = lower_index<0>(metric4, am14);
vector4_t<double> cm14 = raise_index<0>(metric4, bm14);

std::cout << "am14 upper_t" << am14 <<std::endl;
std::cout << "am14 lower_t" << bm14 <<std::endl;
std::cout << "raised again" << cm14 <<std::endl;

double norm2_am14 = contract<0,0>(metric4,am14,am14);

std::cout << "norm2 am14 :" << norm2_am14 <<std::endl;

*/


// Symmetry of two indices
sym_tensor3_t<double,0,1,upper_t,lower_t> sym_tensor3;
sym_tensor4_t<double,0,1,upper_t,lower_t> sym_tensor4;

sym_tensor4_t<double,0,1,upper_t,lower_t,lower_t> sym_tensor4_3;

for(int i = 0; i<sym2_symmetry_t<3,2,0,1>::ndof; ++i)
  sym_tensor3[i] = i+1;
for(int i = 0; i<sym2_symmetry_t<4,2,0,1>::ndof; ++i)
  sym_tensor4[i] = i+1;
for(int i = 0; i<sym2_symmetry_t<4,3,0,1>::ndof; ++i)
  sym_tensor4_3[i] = i+1;

std::cout << "sym3: " << sym_tensor3 << std::endl;
std::cout << "sym4: " << sym_tensor4 << std::endl;
std::cout << "sym4_3: " << sym_tensor4_3 << std::endl;

std::cout << "Symmetric tensor 3:" << std::endl;
std::cout << sym_tensor3.c<0,0>() << " " << sym_tensor3.c<0,1>() << " " << sym_tensor3.c<0,2>() << std::endl;
std::cout << sym_tensor3.c<1,0>() << " " << sym_tensor3.c<1,1>() << " " << sym_tensor3.c<1,2>() << std::endl;
std::cout << sym_tensor3.c<2,0>() << " " << sym_tensor3.c<2,1>() << " " << sym_tensor3.c<2,2>() << std::endl;

std::cout << "Symmetric tensor 4:" << std::endl;
std::cout << sym_tensor4.c<0,0>() << " " << sym_tensor4.c<0,1>() << " " << sym_tensor4.c<0,2>() << " " << sym_tensor4.c<0,3>() << std::endl;
std::cout << sym_tensor4.c<1,0>() << " " << sym_tensor4.c<1,1>() << " " << sym_tensor4.c<1,2>() << " " << sym_tensor4.c<1,3>() << std::endl;
std::cout << sym_tensor4.c<2,0>() << " " << sym_tensor4.c<2,1>() << " " << sym_tensor4.c<2,2>() << " " << sym_tensor4.c<2,3>() << std::endl;
std::cout << sym_tensor4.c<3,0>() << " " << sym_tensor4.c<3,1>() << " " << sym_tensor4.c<3,2>() << " " << sym_tensor4.c<3,3>() << std::endl;

//tensor3_t<double,upper_t,lower_t> from_sym3(sym_tensor3);
//tensor4_t<double,upper_t,lower_t> from_sym4(sym_tensor4);

//tensor3_t<double,upper_t,lower_t> from_sym3;
//tensor4_t<double,upper_t,lower_t> from_sym4;

//from_sym3.set<-1,-1>(sym_tensor3);
//from_sym4.set<-1,-1>(sym_tensor4);

auto from_sym3 = evaluate(generic_cast(sym_tensor3));
auto from_sym4 = evaluate(generic_cast(sym_tensor4));

std::cout << "Generic from symmetric tensor 3:" << std::endl;
std::cout << from_sym3.c<0,0>() << " " << from_sym3.c<0,1>() << " " << from_sym3.c<0,2>() << std::endl;
std::cout << from_sym3.c<1,0>() << " " << from_sym3.c<1,1>() << " " << from_sym3.c<1,2>() << std::endl;
std::cout << from_sym3.c<2,0>() << " " << from_sym3.c<2,1>() << " " << from_sym3.c<2,2>() << std::endl;

std::cout << "Generic from symmetric tensor 4:" << std::endl;
std::cout << from_sym4.c<0,0>() << " " << from_sym4.c<0,1>() << " " << from_sym4.c<0,2>() << " " << from_sym4.c<0,3>() << std::endl;
std::cout << from_sym4.c<1,0>() << " " << from_sym4.c<1,1>() << " " << from_sym4.c<1,2>() << " " << from_sym4.c<1,3>() << std::endl;
std::cout << from_sym4.c<2,0>() << " " << from_sym4.c<2,1>() << " " << from_sym4.c<2,2>() << " " << from_sym4.c<2,3>() << std::endl;
std::cout << from_sym4.c<3,0>() << " " << from_sym4.c<3,1>() << " " << from_sym4.c<3,2>() << " " << from_sym4.c<3,3>() << std::endl;

// here generic -> symmetric tensor is automatically called (through constructor)
std::cout << "Generic and symmetric are the same: " << sym_tensor3.compare_components<14>(from_sym3).first << " "
                                                    << sym_tensor4.compare_components<14>(from_sym4).first << std::endl;
// here sym2_cast_t -> symmetric tensor is automatically called (through constructor)
std::cout << "Generic and symmetric are the same (cast): "
          << sym_tensor3.compare_components<14>(sym2_cast<0,1>(from_sym3)).first << " "
          << sym_tensor4.compare_components<14>(sym2_cast<0,1>(from_sym4)).first << std::endl;


sym_tensor3_t<double,0,1,upper_t,lower_t> from_generic3(E);
from_generic3.set<-1,0>(a);

std::cout << "Symmetric from gemeric (E) tensor 3:" << std::endl;
std::cout << from_generic3.c<0,0>() << " " << from_generic3.c<0,1>() << " " << from_generic3.c<0,2>() << std::endl;
std::cout << from_generic3.c<1,0>() << " " << from_generic3.c<1,1>() << " " << from_generic3.c<1,2>() << std::endl;
std::cout << from_generic3.c<2,0>() << " " << from_generic3.c<2,1>() << " " << from_generic3.c<2,2>() << std::endl;

sym_tensor4_t<double,0,1,upper_t,lower_t> sym_tensor4_set(sym_tensor4);
//sym_tensor4_set.set<-2,-2>(E);
sym_tensor4_set.set<-2,-2>(from_generic3);

vector4_t<double> vec4(1337,1338,42,43);
sym_tensor4_set.set<-1,0>(vec4);

std::cout << "Symmetric tensor 4 with symmetric 3 (E):" << std::endl;
std::cout << sym_tensor4_set.c<0,0>() << " " << sym_tensor4_set.c<0,1>() << " " << sym_tensor4_set.c<0,2>() << " " << sym_tensor4_set.c<0,3>() << std::endl;
std::cout << sym_tensor4_set.c<1,0>() << " " << sym_tensor4_set.c<1,1>() << " " << sym_tensor4_set.c<1,2>() << " " << sym_tensor4_set.c<1,3>() << std::endl;
std::cout << sym_tensor4_set.c<2,0>() << " " << sym_tensor4_set.c<2,1>() << " " << sym_tensor4_set.c<2,2>() << " " << sym_tensor4_set.c<2,3>() << std::endl;
std::cout << sym_tensor4_set.c<3,0>() << " " << sym_tensor4_set.c<3,1>() << " " << sym_tensor4_set.c<3,2>() << " " << sym_tensor4_set.c<3,3>() << std::endl;

// some contraction tests
auto sym_contract0 = evaluate(contract<0,1>(sym_tensor3,from_sym3));
auto sym_contract1 = evaluate(contract<0,1>(sym_tensor4,from_sym4));

auto sym_contract2 = evaluate(contract<0,1>(sym_tensor3,sym_tensor3));
auto sym_contract3 = evaluate(contract<0,1>(sym_tensor4,sym_tensor4));

auto gen_contract0 = evaluate(contract<0,1>(from_sym3,from_sym3));
auto gen_contract1 = evaluate(contract<0,1>(from_sym4,from_sym4));

std::cout << "Mixed generic and symmetric contraction are the same: "
          << sym_contract0.compare_components<14>(sym_contract2).first << " "
          << sym_contract1.compare_components<14>(sym_contract3).first << std::endl;

std::cout << "Generic and symmetric contraction are the same: "
          << sym_contract2.compare_components<14>(gen_contract0).first << " "
          << sym_contract3.compare_components<14>(gen_contract1).first << std::endl;

double trace_sym = trace(sym_tensor3);
double trace_gen = trace(from_sym3);

std::cout << "Trace of sym and gen tensors: " << trace_sym << " " << trace_gen << std::endl;
std::cout << "Ndof: " << decltype(sym_tensor3)::ndof << " " << decltype(from_sym3)::ndof << std::endl;


/////////////////////////////////////////////////////////////
//
// Subtensor assignment
//
/////////////////////////////////////////////////////////////



  std::cout << "A = " << std::endl;
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      std::cout << " "<< A[i+3*j] << " ";
    std::cout<<std::endl;
  }


  std::cout << "D_sub = " << std::endl;
  for(int i=0; i<2; ++i){
    for(int j=0; j<2; ++j)
      std::cout << " "<< D_spatial[i+2*j] << " ";
    std::cout<<std::endl;
  }

  assign_slice<-2,-2>(A) += D_spatial;
  std::cout << "A<-2,-2> + D_spatial = " << std::endl;
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      std::cout << " "<< A[i+3*j] << " ";
    std::cout<<std::endl;
  }

  assign_slice<-2,-2>(A) -= 0.5*D_spatial;
  std::cout << "A<-2,-2> - 0.5*D_spatial = " << std::endl;
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      std::cout << " "<< A[i+3*j] << " ";
    std::cout<<std::endl;
  }

  assign_slice<-2,-2>(A) *= 10;
  std::cout << "A<-2,-2>*10 = " << std::endl;
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      std::cout << " "<< A[i+3*j] << " ";
    std::cout<<std::endl;
  }


  std::cout << "Kronecker test" << std::endl;

  std::cout << kronecker3_t<double>::template evaluate<0>()<< std::endl;
  std::cout << kronecker3_t<double>::template evaluate<1>()<< std::endl;
  std::cout << kronecker3_t<double>::template evaluate<2>()<< std::endl;
  std::cout << kronecker3_t<double>::template evaluate<3>()<< std::endl;
  std::cout << kronecker3_t<double>::template evaluate<4>()<< std::endl;
  std::cout << kronecker3_t<double>::template evaluate<5>()<< std::endl;
  std::cout << kronecker3_t<double>::template evaluate<6>()<< std::endl;
  std::cout << kronecker3_t<double>::template evaluate<7>()<< std::endl;
  std::cout << kronecker3_t<double>::template evaluate<8>()<< std::endl;

  std::cout << "Levi Civita test" << std::endl;

  auto lc3 = evaluate(levi_civita3_up_t<double>());

  int count = 0;
  for(int i=0; i<3; ++i) {
    for(int j=0; j<3; ++j)
      for(int k=0; k<3; ++k) {
        if(lc3.access(i,j,k) != 0) {
          std::cout << "(" << i  << "," << j << "," << k << ") = " << lc3.access(i,j,k) << " ";
          ++count;
        }
      }
    std::cout<<std::endl;
  }
  std::cout << "Found " << count << " non-zero elements" << std::endl;

  auto lc4 = evaluate(levi_civita4_up_t<double>());

  count = 0;
  for(int i=0; i<4; ++i) {
    for(int j=0; j<4; ++j)
      for(int k=0; k<4; ++k)
        for(int l=0; l<4; ++l) {
          if(lc4.access(i,j,k,l) != 0) {
            std::cout << "(" << i  << "," << j << "," << k << "," << l << ") = " << lc4.access(i,j,k,l) << " ";
            ++count;
          }
        }
    std::cout<<std::endl;
  }
  std::cout << "Found " << count << " non-zero elements" << std::endl;

  levi_civita4_up_t<double> lc4_up;
  levi_civita4_down_t<double> lc4_down;
  levi_civita3_up_t<double> lc3_up;
  levi_civita3_down_t<double> lc3_down;

  auto lc_fac = trace<0,1>(
                trace<0,2>(
                trace<0,3>(
                contract<0,0>(lc4_up,lc4_down))));

  std::cout << "4! = " << lc_fac << std::endl;

  auto kron = evaluate(0.5*
              trace<1,3>(
              contract<1,1>(lc3_down,lc3_up)));

  std::cout << "kronecker = " << std::endl;
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      std::cout << " "<< kron.access(i,j) << " ";
    std::cout << std::endl;
  }

  auto H = evaluate(-A+B);

  std::cout << "A = " << std::endl;
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      std::cout << " "<< A.access(i,j) << " ";
    std::cout<<std::endl;
  }

  std::cout << "B = " << std::endl;
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      std::cout << " "<< B.access(i,j) << " ";
    std::cout<<std::endl;
  }

  std::cout << "H = " << std::endl;
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      std::cout << " "<< H.access(i,j) << " ";
    std::cout<<std::endl;
  }
}



