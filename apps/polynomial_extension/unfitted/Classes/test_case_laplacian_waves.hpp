
#ifndef test_case_laplacian_waves_hpp
#define test_case_laplacian_waves_hpp


///// test_case_laplacian_waves
// exact solution : t*t*sin(\pi x) sin(\pi y)               in \Omega_1
//                  t*t*sin(\pi x) sin(\pi y)               in \Omega_2
// (\kappa_1,\rho_1) = (\kappa_2,\rho_2) = (1,1)
template<typename T, typename Function, typename Mesh>
class test_case_laplacian_waves: public test_case_laplacian<T, Function, Mesh>
{
   public:
       
   test_case_laplacian_waves(T t,Function level_set__)
       : test_case_laplacian<T, Function, Mesh>
       (level_set__, params<T>(),
        [level_set__,t](const typename Mesh::point_type& pt) -> T { /* sol */
           if(level_set__(pt) > 0)
               return t*t*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
           else return t*t*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());},
        [level_set__,t](const typename Mesh::point_type& pt) -> T { /* rhs */
            if(level_set__(pt) > 0)
                return 2.0*(1.0 + M_PI*M_PI*t*t)*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
           else return 2.0*(1.0 + M_PI*M_PI*t*t)*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());},
        [level_set__,t](const typename Mesh::point_type& pt) -> T { // bcs
            if(level_set__(pt) > 0)
               return t*t*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
           else return t*t*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());},
        [level_set__,t](const typename Mesh::point_type& pt) -> auto { // grad
            Matrix<T, 1, 2> ret;
            if(level_set__(pt) > 0)
            {
                ret(0) = M_PI*t*t*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
                ret(1) = M_PI*t*t*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
                return ret;
            }
            else {
                ret(0) = M_PI*t*t*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
                ret(1) = M_PI*t*t*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
                return ret;}},
        [](const typename Mesh::point_type& pt) -> T {/* Null Dir */
            return 0;},
        [level_set__](const typename Mesh::point_type& pt) -> T {/* Null Neu */
            return 0;})
       {}
    
//     test_case_laplacian_waves(T t,Function level_set__)
//         : test_case_laplacian<T, Function, Mesh>
//         (level_set__, params<T>(),
//          [level_set__,t](const typename Mesh::point_type& pt) -> T { /* sol */
//             if(level_set__(pt) > 0)
//                 return (1 - pt.x())*pt.x()*(1 - pt.y())*pt.y()*std::sin(std::sqrt(2.0)*M_PI*t);
//             else return (1 - pt.x())*pt.x()*(1 - pt.y())*pt.y()*std::sin(std::sqrt(2.0)*M_PI*t);},
//          [level_set__,t](const typename Mesh::point_type& pt) -> T { /* rhs */
//             T x,y;
//             x = pt.x();
//             y = pt.y();
//              if(level_set__(pt) > 0)
//                  return 2*(x - x*x + y - M_PI*M_PI*(-1 + x)*x*(-1 + y)*y - y*y)*std::sin(std::sqrt(2.0)*M_PI*t);
//             else return 2*(x - x*x + y - M_PI*M_PI*(-1 + x)*x*(-1 + y)*y - y*y)*std::sin(std::sqrt(2.0)*M_PI*t);},
//          [level_set__,t](const typename Mesh::point_type& pt) -> T { // bcs
//             T x,y;
//             x = pt.x();
//             y = pt.y();
//              if(level_set__(pt) > 0)
//                 return (1 - pt.x())*pt.x()*(1 - pt.y())*pt.y()*std::sin(std::sqrt(2.0)*M_PI*t);
//             else return (1 - pt.x())*pt.x()*(1 - pt.y())*pt.y()*std::sin(std::sqrt(2.0)*M_PI*t);},
//          [level_set__,t](const typename Mesh::point_type& pt) -> auto { // grad
//              Matrix<T, 1, 2> ret;
//             T x,y;
//             x = pt.x();
//             y = pt.y();
//              if(level_set__(pt) > 0)
//              {
//                  ret(0) = (1 - x)*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t) - x*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t);
//                  ret(1) = (1 - x)*x*(1 - y)*std::sin(std::sqrt(2.0)*M_PI*t) - (1 - x)*x*y*std::sin(std::sqrt(2.0)*M_PI*t);
//                  return ret;
//              }
//              else {
//                  ret(0) = (1 - x)*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t) - x*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t);
//                  ret(1) = (1 - x)*x*(1 - y)*std::sin(std::sqrt(2.0)*M_PI*t) - (1 - x)*x*y*std::sin(std::sqrt(2.0)*M_PI*t);
//                  return ret;}},
//          [](const typename Mesh::point_type& pt) -> T {/* Null Dir */
//              return 0;},
//          [level_set__](const typename Mesh::point_type& pt) -> T {/* Null Neu */
//              return 0;})
//         {}
    
//    test_case_laplacian_waves(T t,Function level_set__)
//    : test_case_laplacian<T, Function, Mesh>
//    (level_set__, params<T>(),
//     [level_set__,t](const typename Mesh::point_type& pt) -> T { /* sol */
//        if(level_set__(pt) > 0)
//            return (1.0/(std::sqrt(2.0)*M_PI))*std::sin(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
//        else return (1.0/(std::sqrt(2.0)*M_PI))*std::sin(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());},
//     [level_set__,t](const typename Mesh::point_type& pt) -> T { /* rhs */
//         if(level_set__(pt) > 0)
//             return 0;
//        else return 0;},
//     [level_set__,t](const typename Mesh::point_type& pt) -> T { // bcs
//         if(level_set__(pt) > 0)
//            return (1.0/(std::sqrt(2.0)*M_PI))*std::sin(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
//        else return (1.0/(std::sqrt(2.0)*M_PI))*std::sin(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());},
//     [level_set__,t](const typename Mesh::point_type& pt) -> auto { // grad
//         Matrix<T, 1, 2> ret;
//        T x,y;
//        x = pt.x();
//        y = pt.y();
//         if(level_set__(pt) > 0)
//         {
//             ret(0) = (std::sin(std::sqrt(2)*M_PI*t)*std::cos(M_PI*x)*std::sin(M_PI*y))/std::sqrt(2.0);
//             ret(1) = (std::sin(std::sqrt(2)*M_PI*t)*std::sin(M_PI*x)*std::cos(M_PI*y))/std::sqrt(2.0);
//             return ret;
//         }
//         else {
//             ret(0) = (std::sin(std::sqrt(2)*M_PI*t)*std::cos(M_PI*x)*std::sin(M_PI*y))/std::sqrt(2.0);
//             ret(1) = (std::sin(std::sqrt(2)*M_PI*t)*std::sin(M_PI*x)*std::cos(M_PI*y))/std::sqrt(2.0);
//             return ret;}},
//     [](const typename Mesh::point_type& pt) -> T {/* Null Dir */
//         return 0;},
//     [level_set__](const typename Mesh::point_type& pt) -> T {/* Null Neu */
//         return 0;})
//    {}

};

template<typename Mesh, typename Function>
auto make_test_case_laplacian_waves(double t, const Mesh& msh, Function level_set_function)
{
    return test_case_laplacian_waves<typename Mesh::coordinate_type, Function, Mesh>(t,level_set_function);
}

#endif


