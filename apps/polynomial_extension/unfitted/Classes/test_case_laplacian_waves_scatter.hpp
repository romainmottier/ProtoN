
#ifndef test_case_laplacian_waves_scatter_hpp
#define test_case_laplacian_waves_scatter_hpp

///// test_case_laplacian_waves_scatter
// (\kappa_1,\rho_1) = (2,1)
// (\kappa_2,\rho_2) = (1,1)
template<typename T, typename Function, typename Mesh>
class test_case_laplacian_waves_scatter: public test_case_laplacian<T, Function, Mesh>
{
   public:

test_case_laplacian_waves_scatter(T t,Function level_set__)
    : test_case_laplacian<T, Function, Mesh>
    (level_set__, params<T>(),
     [level_set__,t](const typename Mesh::point_type& pt) -> T { /* sol */
        if(level_set__(pt) > 0)
        {
            T x,y,xc,yc,r,wave,vx,vy,v,c,lp,factor;
            x = pt.x();
            y = pt.y();
            xc = 0.0;
            yc = 0.0;//2.0/3.0;
            c = 10.0;
            lp = std::sqrt(9.0)/c;
            r = std::sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc));
            wave = (c)/(std::exp((1.0/(lp*lp))*r*r*M_PI*M_PI));
            factor = (lp*lp/(2.0*M_PI*M_PI));
            return factor*wave;
        }
        else {
//            T u,r,r0,dx,dy;
//            r0 = 0.1;
//            dx = pt.x() -0.5;
//            dy = pt.y() -2.0/3.0;
//            r = std::sqrt(dx*dx+dy*dy);
//            if(r < r0){
//                u = 1.0 + std::cos(M_PI*r/r0);
//            }else{
//                u = 0.0;
//            }
//            return u;
            T x,y,xc,yc,r,wave,vx,vy,v,c,lp,factor;
            x = pt.x();
            y = pt.y();
            xc = 0.0;
            yc = 0.0;//2.0/3.0;
            c = 10.0;
            lp = std::sqrt(9.0)/c;
            r = std::sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc));
            wave = (c)/(std::exp((1.0/(lp*lp))*r*r*M_PI*M_PI));
            factor = (lp*lp/(2.0*M_PI*M_PI));
            return factor*wave;
        }},
     [level_set__,t](const typename Mesh::point_type& pt) -> T { /* rhs */
         if(level_set__(pt) > 0)
             return 0.0;
        else return 0.0;},
     [level_set__,t](const typename Mesh::point_type& pt) -> T { // bcs
         if(level_set__(pt) > 0)
            return 0.0;
        else return 0.0;},
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
    
};

template<typename Mesh, typename Function>
auto make_test_case_laplacian_waves_scatter(double t, const Mesh& msh, Function level_set_function)
{
    return test_case_laplacian_waves_scatter<typename Mesh::coordinate_type, Function, Mesh>(t,level_set_function);
}

#endif
