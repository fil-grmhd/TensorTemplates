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
#include "../tensor_templates.hh"


int main(){

  using namespace tensors;

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

  auto D_spatial = evaluate(spatial_part(D));

  std::cout << "D_sub = " << std::endl;
  for(int i=0; i<2; ++i){
    for(int j=0; j<2; ++j)
      std::cout << " "<< D_spatial[i+2*j] << " ";
    std::cout<<std::endl;
  }


  tensor3_t<double,lower_t,upper_t> D_T = reorder_index<1,0>(D);
  std::cout << "D " << D <<std::endl;
  std::cout << "D reordered" << D_T <<std::endl;

  tensor3_t<double,upper_t,upper_t> ab = tensor_cat(a,b);
  std::cout << "Concatenating a and b: " << ab << std::endl;

/////////////////////////////////////////////////////////////////////
//
//            METRIC TESTS
//
/////////////////////////////////////////////////////////////////////


//Flat space-time test
double lapse = 1.;
vector3_t<double> shift {0,0,0};

metric_tensor_t<double,3> mt {1,0,-0.3,
			      0,1,0,
		      	     -0.3,0,1};

metric3_t<double> metric(lapse, std::move(shift), std::move(mt));

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


}
