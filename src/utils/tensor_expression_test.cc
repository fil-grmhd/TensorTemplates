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

  std::cout << "f :" << f[0] <<" , "<< f[1] << " , " << f[2] <<std::endl;


}



