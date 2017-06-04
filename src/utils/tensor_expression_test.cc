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
#include "../tensor_templates.hh"


int main(){

  using namespace tensors;

  vector3_t<double> a; a[0] =1; a[1] =2; a[2]=3;
  vector3_t<double> b; b[0] =3; b[1]= 2; b[2] =1;

  vector3_t<double> c = a+b;
  vector3_t<double> d = 2.*a+ b*2.0 - c/1.;

  std::cout << "c :" << c[0] <<" , "<< c[1] << " , " << c[2] <<std::endl;
  std::cout << "d :" << d[0] <<" , "<< d[1] << " , " << d[2] <<std::endl;


  tensor3_t<double, upper_t, lower_t> M;
  
  int nn=1;
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      M[i+3*j] = nn++;
  }

  vector3_t<double> e = contract<1,0>(M,a);
  
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      std::cout << " "<< M[i+3*j] << " ";
    std::cout<<std::endl;
  }


}



