
#ifndef test_cases_hpp
#define test_cases_hpp

///// test_case_laplacian_conv
template<typename T, typename Function, typename Mesh>
class test_case_laplacian_conv: public test_case_laplacian<T, Function, Mesh>
{
   public:
    
    test_case_laplacian_conv(Function level_set__)
    : test_case_laplacian<T, Function, Mesh>
    (level_set__, params<T>(),
     [level_set__](const typename Mesh::point_type& pt) -> T { /* sol */
        if(level_set__(pt) > 0)
            return std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
        else return std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());},
     [level_set__](const typename Mesh::point_type& pt) -> T { /* rhs */
         if(level_set__(pt) > 0)
             return 2.0*(M_PI*M_PI)*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
        else return 2.0*(M_PI*M_PI)*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());},
     [level_set__](const typename Mesh::point_type& pt) -> T { // bcs
         if(level_set__(pt) > 0)
            return std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
        else return std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());},
     [level_set__](const typename Mesh::point_type& pt) -> auto { // grad
         Matrix<T, 1, 2> ret;
         if(level_set__(pt) > 0)
         {
             ret(0) = M_PI*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
             ret(1) = M_PI*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
             return ret;
         }
         else {
             ret(0) = M_PI*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
             ret(1) = M_PI*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
             return ret;}},
     [](const typename Mesh::point_type& pt) -> T {/* Null Dir */
         return 0;},
     [level_set__](const typename Mesh::point_type& pt) -> T {/* Null Neu */
         return 0;})
    {}
    
//    test_case_laplacian_conv(Function level_set__)
//    : test_case_laplacian<T, Function, Mesh>
//    (level_set__, params<T>(),
//     [level_set__](const typename Mesh::point_type& pt) -> T { /* sol */
//        if(level_set__(pt) > 0)
//            return (1.0-pt.x())*pt.x() * (1.0-pt.y())*pt.y();
//        else return (1.0-pt.x())*pt.x() * (1.0-pt.y())*pt.y();},
//     [level_set__](const typename Mesh::point_type& pt) -> T { /* rhs */
//        T x,y;
//        x = pt.x();
//        y = pt.y();
//         if(level_set__(pt) > 0)
//             return -2.0*((x - 1)*x + (y - 1)*y);
//        else return -2.0*((x - 1)*x + (y - 1)*y);},
//     [level_set__](const typename Mesh::point_type& pt) -> T { // bcs
//         if(level_set__(pt) > 0)
//            return (1.0-pt.x())*pt.x() * (1.0-pt.y())*pt.y();
//        else return (1.0-pt.x())*pt.x() * (1.0-pt.y())*pt.y();},
//     [level_set__](const typename Mesh::point_type& pt) -> auto { // grad
//         Matrix<T, 1, 2> ret;
//        T x,y;
//        x = pt.x();
//        y = pt.y();
//         if(level_set__(pt) > 0)
//         {
//             ret(0) = (1 - x)*(1 - y)*y - x*(1 - y)*y;
//             ret(1) = (1 - x)*x*(1 - y) - (1 - x)*x*y;
//             return ret;
//         }
//         else {
//             ret(0) = (1 - x)*(1 - y)*y - x*(1 - y)*y;
//             ret(1) = (1 - x)*x*(1 - y) - (1 - x)*x*y;
//             return ret;}},
//     [](const typename Mesh::point_type& pt) -> T {/* Null Dir */
//         return 0;},
//     [level_set__](const typename Mesh::point_type& pt) -> T {/* Null Neu */
//         return 0;})
//    {}
    
};

template<typename Mesh, typename Function>
auto make_test_case_laplacian_conv(const Mesh& msh, Function level_set_function)
{
    return test_case_laplacian_conv<typename Mesh::coordinate_type, Function, Mesh>(level_set_function);
}



#endif
