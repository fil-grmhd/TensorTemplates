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

#include<array>
#include<cmath>
#include<utility>
#include<iostream>
#include<type_traits>


template <typename E>
class VecExpression {
  public:
    inline decltype(auto) operator[](size_t i) const { return static_cast<E const&>(*this)[i];     };
    inline constexpr size_t size() { return E::size();};
      
    // The following overload conversions to E, the template argument type;
    // e.g., for VecExpression<VecSum>, this is a conversion to VecSum.
    operator E& () { return static_cast<E&>(*this); };
    operator const E& () const { return static_cast<const E&>(*this); };

  };

template<typename T,size_t dim>
class Vec : public VecExpression<Vec<T,dim>> {

private:
    std::array<T,dim> elems;
    
public:

    using type = T;

    static constexpr size_t size() {return dim;}
    inline T operator[](size_t i) const { return elems[i]; };
    inline T &operator[](size_t i)      { return elems[i]; };

    Vec() = default;
    
    // Old C++03 way from wikipedia
 /*  
    template <typename E>
    Vec(VecExpression<E> const& vec) {
       static_assert(E::size() == dim);
       
	#pragma unroll
       for(int i=0;i<dim; ++i)
	 elems[i] = vec[i];
    };
*/

    //C++14 way to completely get around the for loop
    template<typename E, std::size_t... I>
    Vec(VecExpression<E> const& vec, std::index_sequence<I...>) : elems({vec[I]...}) {};
     
    template<typename E, typename Indices = std::make_index_sequence<dim>>
    Vec(VecExpression<E> const& vec) : Vec(Vec(vec,Indices{})) {};
    
};

template <typename E1, typename E2>
class VecSum : public VecExpression<VecSum<E1, E2> > {
   E1 const& _u;
   E2 const& _v;
    
public:

   using type = typename E1::type;

   static constexpr size_t size() {return E1::size();};

    VecSum(E1 const& u, E2 const& v) : _u(u), _v(v) {
            static_assert(E1::size() == E2::size(), "Vectors need to have the same dimension.");
            static_assert(std::is_same< typename E1::type, typename E2::type>::value, "Vectors need to have the same type.");
    };
    
    inline decltype(auto) operator[](size_t i) const { return _u[i] + _v[i]; };
};

//Q: Is it better to define this separately or to just use a - b = a +(-1)*b ??

template <typename E1, typename E2>
class VecSub : public VecExpression<VecSub<E1, E2> > {
   E1 const& _u;
   E2 const& _v;
    
public:

   using type = typename E1::type;

   static constexpr size_t size() {return E1::size();};

    VecSub(E1 const& u, E2 const& v) : _u(u), _v(v) {
            static_assert(E1::size() == E2::size(), "Vectors need to have the same dimension.");
            static_assert(std::is_same< typename E1::type, typename E2::type>::value, "Vectors need to have the same type.");
    };
    
    inline decltype(auto) operator[](size_t i) const { return _u[i] - _v[i]; };
};

template <typename E2>
class VecScalarMult : public VecExpression<VecScalarMult< E2> > {
   typename E2::type const&  _u;
   E2 const& _v;
    
public:

   using type = typename E2::type;

   static constexpr size_t size() {return E2::size();};

    VecScalarMult(type const& u, E2 const& v) : _u(u), _v(v) {};
    
    inline decltype(auto) operator[](size_t i) const { return _u*_v[i]; };
};


template <typename E1, typename E2>
VecSum<E1,E2> const
inline operator+(E1 const& u, E2 const& v) {
   static_assert(E1::size() == E2::size(), "Vectors need to have the same dimension.");
   static_assert(std::is_same< typename E1::type, typename E2::type>::value, "Vectors need to have the same type.");

   return VecSum<E1, E2>(u, v);
}

template <typename E1, typename E2>
VecSub<E1,E2> const
inline operator-(E1 const& u, E2 const& v) {
   static_assert(E1::size() == E2::size(), "Vectors need to have the same dimension.");
   static_assert(std::is_same< typename E1::type, typename E2::type>::value, "Vectors need to have the same type.");

   return VecSub<E1, E2>(u, v);
}


template <typename E2>
VecScalarMult<E2> const
inline operator*(typename E2::type const& u, E2 const& v) {
  //Should we add a type check here??
   return VecScalarMult<E2>(u, v);
}

template < typename E2>
VecScalarMult<E2> const
inline operator*(E2 const& u, typename E2::type const& v) {
  //Should we add a type check here??
   return v*u;
}

template < typename E2>
VecScalarMult<E2> const
inline operator/(E2 const& u, typename E2::type const& v) {
  //Should we add a type check here??
   return (1./v)*u;
}

int main(){

  Vec<double,3> a; a[0] =1; a[1] =2; a[2]=3;
  Vec<double,3> b; b[0] =3; b[1]= 2; b[2] =1;

  Vec<double,3> c = a+b;
  Vec<double,3> d = 2*a+ b*2.0 - c/1;

  std::cout << "c :" << c[0] <<" , "<< c[1] << " , " << c[2] <<std::endl;
  std::cout << "d :" << d[0] <<" , "<< d[1] << " , " << d[2] <<std::endl;

}



