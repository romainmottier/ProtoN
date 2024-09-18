/*
 *       /\        Guillaume Delay 2018,2019
 *      /__\       guillaume.delay@enpc.fr
 *     /_\/_\      École Nationale des Ponts et Chaussées - CERMICS
 *    /\    /\
 *   /__\  /__\    This is ProtoN, a library for fast Prototyping of
 *  /_\/_\/_\/_\   Numerical methods.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * If you use this code or parts of it for scientific publications, you
 * are required to cite it as following:
 *
 * Implementation of Discontinuous Skeletal methods on arbitrary-dimensional,
 * polytopal meshes using generic programming.
 * M. Cicuttin, D. A. Di Pietro, A. Ern.
 * Journal of Computational and Applied Mathematics.
 * DOI: 10.1016/j.cam.2017.09.017
 */

#pragma once
#include"../../methods/cuthho"

template<typename T, typename Function, typename Mesh>
class test_case {
public:
    Function level_set_;
    std::function<T(const typename Mesh::point_type &)> sol_fun;
    std::function<T(const typename Mesh::point_type &)> rhs_fun;
    std::function<T(const typename Mesh::point_type &)> bcs_fun;

    test_case() {}

    test_case(Function level_set__,
              std::function<T(const typename Mesh::point_type &)> sol_fun_,
              std::function<T(const typename Mesh::point_type &)> rhs_fun_,
              std::function<T(const typename Mesh::point_type &)> bcs_fun_)
            : level_set_(level_set__), sol_fun(sol_fun_), rhs_fun(rhs_fun_), bcs_fun(bcs_fun_) {}


};

template<typename T, typename Function, typename Mesh>
class test_case_LS_discrete {
public:
    Function level_set_;
    std::function<T(const typename Mesh::point_type &)> sol_fun;
    std::function<T(const typename Mesh::point_type &)> rhs_fun;
    std::function<T(const typename Mesh::point_type &)> bcs_fun;
    typedef typename Mesh::cell_type cell_type;
    //cell_type agglocl;

    test_case_LS_discrete() {}

    test_case_LS_discrete(Function level_set__,
                          std::function<T(const typename Mesh::point_type &)> sol_fun_,
                          std::function<T(const typename Mesh::point_type &)> rhs_fun_,
                          std::function<T(const typename Mesh::point_type &)> bcs_fun_)
            : level_set_(level_set__), sol_fun(sol_fun_), rhs_fun(rhs_fun_), bcs_fun(bcs_fun_) {
        std::cout << "Im mother class 2  " << std::endl;
    }

    /*
    void test_case_cell_assignment(const cell_type& cl)
    {
        agglocl = cl;
        //level_set_.cell_assignment(agglocl);
            std::cout<<"Sono qua in assignment? in test_case_LS_discrete.."<<std::endl;
             std::cout<<"The size of agglocl is "<<agglocl.user_data.offset_subcells.size()<<std::endl;
         std::cout<<"The size of LS_agglocl is "<<level_set_.agglo_LS_cl.user_data.offset_subcells.size()<<std::endl;

    }
     */
    /*
    typename Mesh::cell_type get_cell()
    {
         std::cout<<"Im get_cell del figlio-figlio class 1  "<<std::endl;
        return agglocl;
    }
    */
};

/////////////////////////////  TESTS CASES FOR LAPLACIAN  ////////////////////////////
template<typename T>
struct params
{
    T kappa_1, kappa_2;
    T c_1, c_2;

    params() : kappa_1(1.0), kappa_2(1.0), c_1(1.0), c_2(1.0) {}
    params(T kap1, T kap2) : kappa_1(kap1), kappa_2(kap2) {
        c_1 = 1.0;
        c_2 = 1.0;
    }
};


template<typename T, typename Function, typename Mesh>
class test_case_laplacian : public test_case<T, Function, Mesh> {
public:
    std::function<Eigen::Matrix<T, 1, 2>(const typename Mesh::point_type &)> sol_grad;
    std::function<T(const typename Mesh::point_type &)> dirichlet_jump;
    std::function<T(const typename Mesh::point_type &)> neumann_jump;

    struct params<T> parms;

    test_case_laplacian() {}

    test_case_laplacian(Function level_set__, params<T> parms_,
                        std::function<T(const typename Mesh::point_type &)> sol_fun_,
                        std::function<T(const typename Mesh::point_type &)> rhs_fun_,
                        std::function<T(const typename Mesh::point_type &)> bcs_fun_,
                        std::function<Eigen::Matrix<T, 1, 2>
                                (const typename Mesh::point_type &)> sol_grad_,
                        std::function<T(const typename Mesh::point_type &)> dirichlet_jump_,
                        std::function<T(const typename Mesh::point_type &)> neumann_jump_)
            : test_case<T, Function, Mesh>(level_set__, sol_fun_, rhs_fun_, bcs_fun_),
              parms(parms_), sol_grad(sol_grad_), dirichlet_jump(dirichlet_jump_),
              neumann_jump(neumann_jump_) {}
};


template<typename T, typename Function, typename Mesh>
class test_case_laplacian_LS_discrete : public test_case_LS_discrete<T, Function, Mesh> {
public:
    std::function<Eigen::Matrix<T, 1, 2>(const typename Mesh::point_type &)> sol_grad;
    std::function<T(const typename Mesh::point_type &)> dirichlet_jump;
    std::function<T(const typename Mesh::point_type &)> neumann_jump;

    struct params<T> parms;

    test_case_laplacian_LS_discrete() {}

    test_case_laplacian_LS_discrete(Function level_set__, params<T> parms_,
                                    std::function<T(const typename Mesh::point_type &)> sol_fun_,
                                    std::function<T(const typename Mesh::point_type &)> rhs_fun_,
                                    std::function<T(const typename Mesh::point_type &)> bcs_fun_,
                                    std::function<Eigen::Matrix<T, 1, 2>
                                            (const typename Mesh::point_type &)> sol_grad_,
                                    std::function<T(const typename Mesh::point_type &)> dirichlet_jump_,
                                    std::function<T(const typename Mesh::point_type &)> neumann_jump_)
            : test_case_LS_discrete<T, Function, Mesh>(level_set__, sol_fun_, rhs_fun_, bcs_fun_),
              parms(parms_), sol_grad(sol_grad_), dirichlet_jump(dirichlet_jump_),
              neumann_jump(neumann_jump_) {
        std::cout << "Im son class 2  " << std::endl;
    }
};


///// test_case_laplacian_sin_sin
// exact solution : sin(\pi x) * sin(\pi y) in the whole domain
// \kappa_1 = \kappa_2 = 1
template<typename T, typename Function, typename Mesh>
class test_case_laplacian_sin_sin : public test_case_laplacian<T, Function, Mesh> {
public:
    test_case_laplacian_sin_sin(Function level_set__)
            : test_case_laplacian<T, Function, Mesh>
                      (level_set__, params<T>(),
                       [](const typename Mesh::point_type &pt) -> T { // sol
                           return std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                       },
                       [](const typename Mesh::point_type &pt) -> T { // rhs
                           return 2.0 * M_PI * M_PI * std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                       },
                       [&](const typename Mesh::point_type &pt) -> T { // bcs
                           return std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                       },
                       [](const typename Mesh::point_type &pt) -> auto { // grad
                           Matrix<T, 1, 2> ret;
                           ret(0) = M_PI * std::cos(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                           ret(1) = M_PI * std::sin(M_PI * pt.x()) * std::cos(M_PI * pt.y());
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> T {/* Dir */ return 0.0; },
                       [](const typename Mesh::point_type &pt) -> T {/* Neu */ return 0.0; }) {}
};

template<typename Mesh, typename Function>
auto make_test_case_laplacian_sin_sin(const Mesh &msh, Function level_set_function) {
    return test_case_laplacian_sin_sin<typename Mesh::coordinate_type, Function, Mesh>(level_set_function);
}


///// test_case_laplacian_sin_sin_bis
// exact solution : 1 + sin(\pi x) * sin(\pi y) in the whole domain
// \kappa_1 = \kappa_2 = 1
template<typename T, typename Function, typename Mesh>
class test_case_laplacian_sin_sin_bis : public test_case_laplacian<T, Function, Mesh> {
public:
    test_case_laplacian_sin_sin_bis(Function level_set__)
            : test_case_laplacian<T, Function, Mesh>
                      (level_set__, params<T>(),
                       [](const typename Mesh::point_type &pt) -> T { // sol
                           //return 1 + pt.x() + pt.y() + std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());},
                           return 1 + std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                       },
                       [](const typename Mesh::point_type &pt) -> T { // rhs
                           return 2.0 * M_PI * M_PI * std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                       },
                       [](const typename Mesh::point_type &pt) -> T { // bcs
                           // return 1 + pt.x() + pt.y() + std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());},
                           return 1 + std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                       },
                       [](const typename Mesh::point_type &pt) -> auto { // grad
                           Matrix<T, 1, 2> ret;
                           // ret(0) = 1 + M_PI * std::cos(M_PI*pt.x()) * std::sin(M_PI*pt.y());
                           // ret(1) = 1 + M_PI * std::sin(M_PI*pt.x()) * std::cos(M_PI*pt.y());
                           ret(0) = M_PI * std::cos(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                           ret(1) = M_PI * std::sin(M_PI * pt.x()) * std::cos(M_PI * pt.y());
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> T {/* Dir */ return 0.0; },
                       [](const typename Mesh::point_type &pt) -> T {/* Neu */ return 0.0; }) {}
};

template<typename Mesh, typename Function>
auto make_test_case_laplacian_sin_sin_bis(const Mesh &msh, Function level_set_function) {
    return test_case_laplacian_sin_sin_bis<typename Mesh::coordinate_type, Function, Mesh>(level_set_function);
}


///// test_case_laplacian_sin_sin_gen
// exact solution : sin(\pi (x-a)/(b-a)) * sin(\pi (y-c)/(d-c)) in the whole domain
// \kappa_1 = \kappa_2 = 1
template<typename T, typename Function, typename Mesh>
class test_case_laplacian_sin_sin_gen : public test_case_laplacian<T, Function, Mesh> {
public:
    test_case_laplacian_sin_sin_gen(Function level_set__, T a, T b, T c, T d)
            : test_case_laplacian<T, Function, Mesh>
                      (level_set__, params<T>(),
                       [a, b, c, d](const typename Mesh::point_type &pt) -> T { // sol
                           return std::sin(M_PI * (pt.x() - a) / (b - a)) * std::sin(M_PI * (pt.y() - c) / (d - c));
                       },
                       [a, b, c, d](const typename Mesh::point_type &pt) -> T { // rhs
                           return (1.0 / ((b - a) * (b - a)) + 1.0 / ((d - c) * (d - c))) * M_PI * M_PI
                                  * std::sin(M_PI * (pt.x() - a) / (b - a)) * std::sin(M_PI * (pt.y() - c) / (d - c));
                       },
                       [a, b, c, d](const typename Mesh::point_type &pt) -> T { // bcs
                           return std::sin(M_PI * (pt.x() - a) / (b - a)) * std::sin(M_PI * (pt.y() - c) / (d - c));
                       },
                       [a, b, c, d](const typename Mesh::point_type &pt) -> auto { // grad
                           Matrix<T, 1, 2> ret;
                           ret(0) = (M_PI / (b - a))
                                    * std::cos(M_PI * (pt.x() - a) / (b - a)) * std::sin(M_PI * (pt.y() - c) / (d - c));
                           ret(1) = (M_PI / (d - c))
                                    * std::sin(M_PI * (pt.x() - a) / (b - a)) * std::cos(M_PI * (pt.y() - c) / (d - c));
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> T {/* Dir */ return 0.0; },
                       [](const typename Mesh::point_type &pt) -> T {/* Neu */ return 0.0; }),
              a_(a), b_(b), c_(c), d_(d) {}

    T a_, b_, c_, d_;
};

template<typename Mesh, typename Function, typename T>
auto make_test_case_laplacian_sin_sin_gen(const Mesh &msh, Function level_set_function,
                                          T a, T b, T c, T d) {
    return test_case_laplacian_sin_sin_gen<T, Function, Mesh>(level_set_function, a, b, c, d);
}


///// test_case_laplacian_exp_cos
// exact solution : exp(x) * cos(y) in the whole domain
// \kappa_1 = \kappa_2 = 1
template<typename T, typename Function, typename Mesh>
class test_case_laplacian_exp_cos : public test_case_laplacian<T, Function, Mesh> {
public:
    test_case_laplacian_exp_cos(Function level_set__)
            : test_case_laplacian<T, Function, Mesh>
                      (level_set__, params<T>(),
                       [](const typename Mesh::point_type &pt) -> T { /* sol */
                           return exp(pt.x()) * std::cos(pt.y());
                       },
                       [](const typename Mesh::point_type &pt) -> T { /* rhs */ return 0.0; },
                       [&](const typename Mesh::point_type &pt) -> T { // bcs
                           return exp(pt.x()) * std::cos(pt.y());
                       },
                       [](const typename Mesh::point_type &pt) -> auto { // grad
                           Matrix<T, 1, 2> ret;
                           ret(0) = exp(pt.x()) * std::cos(pt.y());
                           ret(1) = -exp(pt.x()) * std::sin(pt.y());
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> T {/* Dir */ return 0.0; },
                       [](const typename Mesh::point_type &pt) -> T {/* Neu */ return 0.0; }) {}
};

template<typename Mesh, typename Function>
auto make_test_case_laplacian_exp_cos(const Mesh &msh, Function level_set_function) {
    return test_case_laplacian_exp_cos<typename Mesh::coordinate_type, Function, Mesh>(level_set_function);
}

///// test_case_laplacian_jumps_1
// exact solution : sin(\pi x) sin(\pi y)  in \Omega_1
//                  exp(x) * cos(y)        in \Omega_2
// \kappa_1 = \kappa_2 = 1
template<typename T, typename Function, typename Mesh>
class test_case_laplacian_jumps_1 : public test_case_laplacian<T, Function, Mesh> {
public:
    test_case_laplacian_jumps_1(Function level_set__)
            : test_case_laplacian<T, Function, Mesh>
                      (level_set__, params<T>(),
                       [level_set__](const typename Mesh::point_type &pt) -> T { /* sol */
                           if (level_set__(pt) < 0)
                               return std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                           else return exp(pt.x()) * std::cos(pt.y());
                       },
                       [level_set__](const typename Mesh::point_type &pt) -> T { /* rhs */
                           if (level_set__(pt) < 0)
                               return 2.0 * M_PI * M_PI * std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                           else return 0.0;
                       },
                       [level_set__](const typename Mesh::point_type &pt) -> T { // bcs
                           if (level_set__(pt) < 0)
                               return std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                           else return exp(pt.x()) * std::cos(pt.y());
                       },
                       [level_set__](const typename Mesh::point_type &pt) -> auto { // grad
                           Matrix<T, 1, 2> ret;
                           if (level_set__(pt) < 0) {
                               ret(0) = M_PI * std::cos(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                               ret(1) = M_PI * std::sin(M_PI * pt.x()) * std::cos(M_PI * pt.y());
                               return ret;
                           } else {
                               ret(0) = exp(pt.x()) * std::cos(pt.y());
                               ret(1) = -exp(pt.x()) * std::sin(pt.y());
                               return ret;
                           }
                       },
                       [](const typename Mesh::point_type &pt) -> T {/* Dir */
                           return std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y()) - exp(pt.x()) * std::cos(pt.y());
                       },
                       [level_set__](const typename Mesh::point_type &pt) -> T {/* Neu */
                           Matrix<T, 1, 2> normal = level_set__.normal(pt);
                           return (M_PI * std::cos(M_PI * pt.x()) * std::sin(M_PI * pt.y()) -
                                   exp(pt.x()) * std::cos(pt.y())) * normal(0) +
                                  (M_PI * std::sin(M_PI * pt.x()) * std::cos(M_PI * pt.y()) +
                                   exp(pt.x()) * std::sin(pt.y())) * normal(1);
                       }) {}
};

template<typename Mesh, typename Function>
auto make_test_case_laplacian_jumps_1(const Mesh &msh, Function level_set_function) {
    return test_case_laplacian_jumps_1<typename Mesh::coordinate_type, Function, Mesh>(level_set_function);
}


///// test_case_laplacian_jumps_2
// exact solution : exp(x) * cos(y)        in \Omega_1
//                  sin(\pi x) sin(\pi y)  in \Omega_2
// \kappa_1 = \kappa_2 = 1
template<typename T, typename Function, typename Mesh>
class test_case_laplacian_jumps_2 : public test_case_laplacian<T, Function, Mesh> {
public:
    test_case_laplacian_jumps_2(Function level_set__)
            : test_case_laplacian<T, Function, Mesh>
                      (level_set__, params<T>(),
                       [level_set__](const typename Mesh::point_type &pt) -> T { /* sol */
                           if (level_set__(pt) > 0)
                               return std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                           else return exp(pt.x()) * std::cos(pt.y());
                       },
                       [level_set__](const typename Mesh::point_type &pt) -> T { /* rhs */
                           if (level_set__(pt) > 0)
                               return 2.0 * M_PI * M_PI * std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                           else return 0.0;
                       },
                       [level_set__](const typename Mesh::point_type &pt) -> T { // bcs
                           if (level_set__(pt) > 0)
                               return std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                           else return exp(pt.x()) * std::cos(pt.y());
                       },
                       [level_set__](const typename Mesh::point_type &pt) -> auto { // grad
                           Matrix<T, 1, 2> ret;
                           if (level_set__(pt) > 0) {
                               ret(0) = M_PI * std::cos(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                               ret(1) = M_PI * std::sin(M_PI * pt.x()) * std::cos(M_PI * pt.y());
                               return ret;
                           } else {
                               ret(0) = exp(pt.x()) * std::cos(pt.y());
                               ret(1) = -exp(pt.x()) * std::sin(pt.y());
                               return ret;
                           }
                       },
                       [](const typename Mesh::point_type &pt) -> T {/* Dir */
                           return -std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y()) + exp(pt.x()) * std::cos(pt.y());
                       },
                       [level_set__](const typename Mesh::point_type &pt) -> T {/* Neu */
                           Matrix<T, 1, 2> normal = level_set__.normal(pt);
                           return -(M_PI * std::cos(M_PI * pt.x()) * std::sin(M_PI * pt.y()) -
                                    exp(pt.x()) * std::cos(pt.y())) * normal(0) -
                                  (M_PI * std::sin(M_PI * pt.x()) * std::cos(M_PI * pt.y()) +
                                   exp(pt.x()) * std::sin(pt.y())) * normal(1);
                       }) {}
};

template<typename Mesh, typename Function>
auto make_test_case_laplacian_jumps_2(const Mesh &msh, Function level_set_function) {
    return test_case_laplacian_jumps_2<typename Mesh::coordinate_type, Function, Mesh>(level_set_function);
}


///// test_case_laplacian_jumps_3
// exact solution : sin(\pi x) sin(\pi y)               in \Omega_1
//                  sin(\pi x) sin(\pi y) + 2 + x^3 * y^3   in \Omega_2
// \kappa_1 = \kappa_2 = 1
template<typename T, typename Function, typename Mesh>
class test_case_laplacian_jumps_3 : public test_case_laplacian<T, Function, Mesh> {
public:
    test_case_laplacian_jumps_3(Function level_set__)
            : test_case_laplacian<T, Function, Mesh>
                      (level_set__, params<T>(),
                       [level_set__](const typename Mesh::point_type &pt) -> T { /* sol */
                           if (level_set__(pt) > 0)
                               return std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y())
                                      + 2. + pt.x() * pt.x() * pt.x() * pt.y() * pt.y() * pt.y();
                           else return std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                       },
                       [level_set__](const typename Mesh::point_type &pt) -> T { /* rhs */
                           if (level_set__(pt) > 0)
                               return 2.0 * M_PI * M_PI * std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y()) -
                                      6 * pt.x() * pt.y() * (pt.x() * pt.x() + pt.y() * pt.y());
                           else return 2.0 * M_PI * M_PI * std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                       },
                       [level_set__](const typename Mesh::point_type &pt) -> T { // bcs
                           if (level_set__(pt) > 0)
                               return std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y()) + 2.
                                      + pt.x() * pt.x() * pt.x() * pt.y() * pt.y() * pt.y();
                           else return std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                       },
                       [level_set__](const typename Mesh::point_type &pt) -> auto { // grad
                           Matrix<T, 1, 2> ret;
                           if (level_set__(pt) > 0) {
                               ret(0) = M_PI * std::cos(M_PI * pt.x()) * std::sin(M_PI * pt.y()) +
                                        3 * pt.x() * pt.x() * pt.y() * pt.y() * pt.y();
                               ret(1) = M_PI * std::sin(M_PI * pt.x()) * std::cos(M_PI * pt.y()) +
                                        3 * pt.x() * pt.x() * pt.x() * pt.y() * pt.y();
                               return ret;
                           } else {
                               ret(0) = M_PI * std::cos(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                               ret(1) = M_PI * std::sin(M_PI * pt.x()) * std::cos(M_PI * pt.y());
                               return ret;
                           }
                       },
                       [](const typename Mesh::point_type &pt) -> T {/* Dir */
                           return -2. - pt.x() * pt.x() * pt.x() * pt.y() * pt.y() * pt.y();
                       },
                       [level_set__](const typename Mesh::point_type &pt) -> T {/* Neu */
                           Matrix<T, 1, 2> normal = level_set__.normal(pt);
                           return -3 * pt.x() * pt.x() * pt.y() * pt.y() * (pt.y() * normal(0) + pt.x() * normal(1));
                       }) {}
};

template<typename Mesh, typename Function>
auto make_test_case_laplacian_jumps_3(const Mesh &msh, Function level_set_function) {
    return test_case_laplacian_jumps_3<typename Mesh::coordinate_type, Function, Mesh>(level_set_function);
}


///// test_case_laplacian_jumps_3bis FOR DISCRETE LEVEL_SET
// exact solution : sin(\pi x) sin(\pi y)               in \Omega_1
//                  sin(\pi x) sin(\pi y) + 2 + x^3 * y^3   in \Omega_2
// \kappa_1 = \kappa_2 = 1
template<typename T, typename Function, typename Mesh, typename Curr_Cell>
class test_case_laplacian_jumps_3_LS_discrete : public test_case_laplacian_LS_discrete<T, Function, Mesh> {
    typedef typename Mesh::cell_type cell_type;

    Function ls_cl;
public:
    cell_type cl_uploaded;
    Curr_Cell current_cell;

    test_case_laplacian_jumps_3_LS_discrete(Function level_set__, Curr_Cell current_cell)
            : ls_cl(level_set__), current_cell(current_cell), test_case_laplacian_LS_discrete<T, Function, Mesh>
            (level_set__, params<T>(),
             [=](const typename Mesh::point_type &pt) -> T { /* sol */
                 Function ls_cl_bis = ls_cl;
                 ls_cl_bis.cell_assignment(current_cell.get_cell());

                 if (ls_cl_bis(pt) > 0)
                     return std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y())
                            + 2. + pt.x() * pt.x() * pt.x() * pt.y() * pt.y() * pt.y();
                 else return std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
             },
             [=](const typename Mesh::point_type &pt) mutable -> T { /* rhs */
                 std::cout << "Here in RHS..NON DEVO ESSERE QUI" << std::endl;
                 Function ls_cl_bis = ls_cl;
                 ls_cl_bis.cell_assignment(current_cell.get_cell());
                 //  std::cout<<"The size of current_cell1 is "<<current_cell.crr_cl.user_data.offset_subcells.size()<<std::endl;
                 //  std::cout<<"The size of current_cell2 is "<<ls_cl_bis.agglo_LS_cl.user_data.offset_subcells.size()<<std::endl;

                 if (ls_cl_bis(pt) > 0)
                     return 2.0 * M_PI * M_PI * std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y()) -
                            6 * pt.x() * pt.y() * (pt.x() * pt.x() + pt.y() * pt.y());
                 else return 2.0 * M_PI * M_PI * std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
             },
             [this, level_set__](const typename Mesh::point_type &pt) -> T { // bcs
                 if (level_set__(pt) > 0)
                     return std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y()) + 2.
                            + pt.x() * pt.x() * pt.x() * pt.y() * pt.y() * pt.y();
                 else return std::sin(M_PI * pt.x()) * std::sin(M_PI * pt.y());
             },
             [this, level_set__](const typename Mesh::point_type &pt) -> auto { // grad
                 Matrix<T, 1, 2> ret;
                 if (level_set__(pt) > 0) {
                     ret(0) = M_PI * std::cos(M_PI * pt.x()) * std::sin(M_PI * pt.y()) +
                              3 * pt.x() * pt.x() * pt.y() * pt.y() * pt.y();
                     ret(1) = M_PI * std::sin(M_PI * pt.x()) * std::cos(M_PI * pt.y()) +
                              3 * pt.x() * pt.x() * pt.x() * pt.y() * pt.y();
                     return ret;
                 } else {
                     ret(0) = M_PI * std::cos(M_PI * pt.x()) * std::sin(M_PI * pt.y());
                     ret(1) = M_PI * std::sin(M_PI * pt.x()) * std::cos(M_PI * pt.y());
                     return ret;
                 }
             },
             [](const typename Mesh::point_type &pt) -> T {/* Dir */
                 return -2. - pt.x() * pt.x() * pt.x() * pt.y() * pt.y() * pt.y();
             },
             [this, level_set__](const typename Mesh::point_type &pt) -> T {/* Neu */
                 Matrix<T, 1, 2> normal = level_set__.normal(pt);
                 return -3 * pt.x() * pt.x() * pt.y() * pt.y() * (pt.y() * normal(0) + pt.x() * normal(1));
             }) {
        std::cout << "Im figlio del figlio class 1. NON DEVO ESSERE QUI " << std::endl;
    }


    void test_case_cell_assignment(const typename Mesh::cell_type &cl) {
        cl_uploaded = cl;
        // ls_cl.cell_assignment(agglocl);
        //    std::cout<<"Sono qua in assignment CLASSE FIGLIO FIGLIO? in RHS.."<<std::endl;
        //    std::cout<<"The size of cl_uploaded is "<<cl_uploaded.user_data.offset_subcells.size()<<std::endl;
    }


    //typename Mesh::cell_type get_cell()
    //{
    //      std::cout<<"Im get_cell del figlio-figlio class 1  "<<std::endl;
    //    return agglocl;
    // }

};

template<typename Mesh, typename Function, typename Curr_Cell>
auto make_test_case_laplacian_jumps_3bis(const Mesh &msh, Function level_set_function, Curr_Cell current_cell) {
    return test_case_laplacian_jumps_3_LS_discrete<typename Mesh::coordinate_type, Function, Mesh, Curr_Cell>(
            level_set_function, current_cell);
}















////////////////////  TESTS CASES FOR CIRCLES   ////////////////////////////

///// test_case_laplacian_contrast_2
// !! available for circle_level_set only !!
// circle : radius = R, center = (a,b)
// exact solution : r^2 / \kappa_1 in \Omega_1
//                  (r^2 - R^2) / \kappa_2 + R^2 / \kappa_1 in \Omega_2
// \kappa_1 and \kappa_2 : parameters to choose (in parms_)
template<typename T, typename Mesh>
class test_case_laplacian_contrast_2 : public test_case_laplacian<T, circle_level_set < T>, Mesh

>
{
public:

test_case_laplacian_contrast_2(T R, T a, T b, params<T> parms_)
        : test_case_laplacian<T, circle_level_set < T>, Mesh

>
(
circle_level_set<T>(R, a, b
), parms_,
[R, a, b, parms_](
const typename Mesh::point_type &pt
) -> T { /* sol */
T r2 = (pt.x() - a) * (pt.x() - a) + (pt.y() - b) * (pt.y() - b);
if(r2<R *R) {
return r2 / parms_.
kappa_1;
}
else {
return (r2 -
R *R
) / parms_.kappa_2
+
R *R
/ parms_.
kappa_1;
}},
[](
const typename Mesh::point_type &pt
) -> T { /* rhs */ return -4.0;},
[R, a, b, parms_](
const typename Mesh::point_type &pt
) -> T { // bcs
T r2 = (pt.x() - a) * (pt.x() - a) + (pt.y() - b) * (pt.y() - b);
if (r2<R *R) {
return r2 / parms_.
kappa_1;
}
else {
return (r2 -
R *R
) / parms_.kappa_2
+
R *R
/ parms_.
kappa_1;
}},
[R, a, b, parms_](
const typename Mesh::point_type &pt
) -> auto { // grad
Matrix<T, 1, 2> ret;
T r2 = (pt.x() - a) * (pt.x() - a) + (pt.y() - b) * (pt.y() - b);
if (r2<R *R)
{
ret(0) = 2 * ( pt.

x()

- a ) / parms_.
kappa_1;
ret(1) = 2 * ( pt.

y()

- b ) / parms_.
kappa_1;
}
else
{
ret(0) = 2 * ( pt.

x()

- a ) / parms_.
kappa_2;
ret(1) = 2 * ( pt.

y()

- b ) / parms_.
kappa_2;
}
return
ret;
},
[](
const typename Mesh::point_type &pt
) -> T {/* Dir */ return 0.0;},
[](
const typename Mesh::point_type &pt
) -> T {/* Neu */ return 0.0;})
{
}
};


template<typename Mesh>
auto make_test_case_laplacian_contrast_2(const Mesh &msh, circle_level_set<typename Mesh::coordinate_type> LS,
                                         params<typename Mesh::coordinate_type> parms) {
    return test_case_laplacian_contrast_2 < typename Mesh::coordinate_type, Mesh >
                                                                            (LS.radius, LS.alpha, LS.beta, parms);
}

///// test_case_laplacian_contrast_6
// !! available for circle_level_set only !!
// circle : radius = R, center = (a,b)
// exact solution : r^6 / \kappa_1 in \Omega_1
//                  (r^6 - R^6) / \kappa_2 + R^6 / \kappa_1 in \Omega_2
// \kappa_1 and \kappa_2 : parameters to choose (in parms_)
template<typename T, typename Mesh>
class test_case_laplacian_contrast_6 : public test_case_laplacian<T, circle_level_set < T>, Mesh

>
{
public:

test_case_laplacian_contrast_6(T R, T a, T b, params<T> parms_)
        : test_case_laplacian<T, circle_level_set < T>, Mesh

>
(
circle_level_set<T>(R, a, b
), parms_,
[R, a, b, parms_](
const typename Mesh::point_type &pt
) -> T { /* sol */
T r2 = (pt.x() - a) * (pt.x() - a) + (pt.y() - b) * (pt.y() - b);
if(r2<R *R) {
return
r2 *r2
*r2 / parms_.
kappa_1;
}
else {
return (
r2 *r2
*r2) / parms_.kappa_2
+ (
R *R
*
R *R
*
R *R
) * (1.0 / parms_.kappa_1 - 1.0 / parms_.kappa_2);}},
[a, b](
const typename Mesh::point_type &pt
) -> T { /* rhs */
T r2 = (pt.x() - a) * (pt.x() - a) + (pt.y() - b) * (pt.y() - b);
return -36.0 *
r2 *r2;
},
[R, a, b, parms_](
const typename Mesh::point_type &pt
) -> T { // bcs
T r2 = (pt.x() - a) * (pt.x() - a) + (pt.y() - b) * (pt.y() - b);
if(r2<R *R) {
return
r2 *r2
*r2 / parms_.
kappa_1;
}
else {
return (
r2 *r2
*r2) / parms_.kappa_2
+ (
R *R
*
R *R
*
R *R
) * (1.0 / parms_.kappa_1 - 1.0 / parms_.kappa_2);}},
[R, a, b, parms_](
const typename Mesh::point_type &pt
) -> auto { // grad
Matrix<T, 1, 2> ret;
T r2 = (pt.x() - a) * (pt.x() - a) + (pt.y() - b) * (pt.y() - b);
if (r2<R *R)
{
ret(0) = 6 *
r2 *r2
* ( pt.

x()

- a ) / parms_.
kappa_1;
ret(1) = 6 *
r2 *r2
* ( pt.

y()

- b ) / parms_.
kappa_1;
}
else
{
ret(0) = 6 *
r2 *r2
* ( pt.

x()

- a ) / parms_.
kappa_2;
ret(1) = 6 *
r2 *r2
* ( pt.

y()

- b ) / parms_.
kappa_2;
}
return
ret;
},
[](
const typename Mesh::point_type &pt
) -> T {/* Dir */ return 0.0;},
[](
const typename Mesh::point_type &pt
) -> T {/* Neu */ return 0.0;})
{
}
};


template<typename Mesh>
auto make_test_case_laplacian_contrast_6(const Mesh &msh, circle_level_set<typename Mesh::coordinate_type> LS,
                                         params<typename Mesh::coordinate_type> parms) {
    return test_case_laplacian_contrast_6 < typename Mesh::coordinate_type, Mesh >
                                                                            (LS.radius, LS.alpha, LS.beta, parms);
}


///// test_case_laplacian_circle_hom
// !! available for circle_level_set only !!
// circle : radius = R, center = (a,b)
// exact solution : (R - r) * r^n in the whole domain
// coded here for n = 6
// \kappa_1 = \kappa_2 = 1
template<typename T, typename Mesh>
class test_case_laplacian_circle_hom : public test_case_laplacian<T, circle_level_set < T>, Mesh

>
{
public:

test_case_laplacian_circle_hom(T R, T a, T b)
        : test_case_laplacian<T, circle_level_set < T>, Mesh

>
(
circle_level_set<T>(R, a, b
),

params<T>(),

[R, a, b](
const typename Mesh::point_type &pt
) -> T { /* sol */
T x1 = pt.x() - a;
T x2 = x1 * x1;
T y1 = pt.y() - b;
T y2 = y1 * y1;
T r2 = x2 + y2;
T r1 = std::sqrt(r2);
return (R - r1) *
r2 *r2
*
r2;
},
[R, a, b](
const typename Mesh::point_type &pt
) -> T { /* rhs */
T x1 = pt.x() - a;
T x2 = x1 * x1;
T y1 = pt.y() - b;
T y2 = y1 * y1;
T r2 = x2 + y2;
T r1 = std::sqrt(r2);
return (2 * 6 + 1) *
r2 *r2
* r1 - 6*6 * (R-r1) *
r2 *r2;
},
[R, a, b](
const typename Mesh::point_type &pt
) -> T { // bcs
T x1 = pt.x() - a;
T x2 = x1 * x1;
T y1 = pt.y() - b;
T y2 = y1 * y1;
T r2 = x2 + y2;
T r1 = std::sqrt(r2);
return (R - r1) *
r2 *r2
*
r2;
},
[R, a, b](
const typename Mesh::point_type &pt
) -> auto { // grad
Matrix<T, 1, 2> ret;
T x1 = pt.x() - a;
T x2 = x1 * x1;
T y1 = pt.y() - b;
T y2 = y1 * y1;
T r2 = x2 + y2;
T r1 = std::sqrt(r2);
T B = 6 * (R - r1) * r2 * r2 - r2 * r2 * r1;
ret(0) =
B *x1;
ret(1) =
B *y1;
return
ret;
},
[](
const typename Mesh::point_type &pt
) -> T {/* Dir */ return 0.0;},
[](
const typename Mesh::point_type &pt
) -> T {/* Neu */ return 0.0;})
{
}
};


template<typename Mesh>
auto make_test_case_laplacian_circle_hom(const Mesh &msh, circle_level_set<typename Mesh::coordinate_type> LS) {
    return test_case_laplacian_circle_hom < typename Mesh::coordinate_type, Mesh > (LS.radius, LS.alpha, LS.beta);
}


/******************************************************************************************/
/*******************                                               ************************/
/*******************                 STOKES  PROBLEM               ************************/
/*******************                                               ************************/
/******************************************************************************************/

template<typename T, typename Function, typename Mesh>
class test_case_stokes {
public:
    Function level_set_;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> sol_vel;
    std::function<T(const typename Mesh::point_type &)> sol_p;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> rhs_fun;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> bcs_vel;
    std::function<Eigen::Matrix<T, 2, 2>(const typename Mesh::point_type &)> vel_grad;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> dirichlet_jump;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> neumann_jump;

    struct params<T> parms;

    test_case_stokes(const test_case_stokes &other) {
        level_set_ = other.level_set_;
        sol_vel = other.sol_vel;
        sol_p = other.sol_p;
        rhs_fun = other.rhs_fun;
        bcs_vel = other.bcs_vel;
        vel_grad = other.vel_grad;
        dirichlet_jump = other.dirichlet_jump;
        neumann_jump = other.neumann_jump;

    }

    test_case_stokes() {}

    test_case_stokes
            (Function &level_set__, params<T> parms_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> sol_vel_,
             std::function<T(const typename Mesh::point_type &)> sol_p_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> rhs_fun_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> bcs_vel_,
             std::function<Eigen::Matrix<T, 2, 2>(const typename Mesh::point_type &)> vel_grad_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> dirichlet_jump_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> neumann_jump_)
            : level_set_(level_set__), sol_vel(sol_vel_), sol_p(sol_p_), rhs_fun(rhs_fun_),
              bcs_vel(bcs_vel_), parms(parms_), vel_grad(vel_grad_), dirichlet_jump(dirichlet_jump_),
              neumann_jump(neumann_jump_) {}
};


template<typename T, typename Function, typename Mesh, typename Para_Interface>
class test_case_stokes_eps {
public:
    Function level_set_;
    Para_Interface parametric_interface;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> sol_vel;
    std::function<T(const typename Mesh::point_type &)> sol_p;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> rhs_fun;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> bcs_vel;
    std::function<Eigen::Matrix<T, 2, 2>(const typename Mesh::point_type &)> vel_grad;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> dirichlet_jump;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> neumann_jump;

    struct params<T> parms;

    test_case_stokes_eps(const test_case_stokes_eps &other) {
        level_set_ = other.level_set_;
        parametric_interface = other.parametric_interface;
        sol_vel = other.sol_vel;
        sol_p = other.sol_p;
        rhs_fun = other.rhs_fun;
        bcs_vel = other.bcs_vel;
        vel_grad = other.vel_grad;
        dirichlet_jump = other.dirichlet_jump;
        neumann_jump = other.neumann_jump;
        parms = other.parms;
    }

    test_case_stokes_eps(test_case_stokes_eps &other) {
        level_set_ = other.level_set_;
        parametric_interface = other.parametric_interface;
        sol_vel = other.sol_vel;
        sol_p = other.sol_p;
        rhs_fun = other.rhs_fun;
        bcs_vel = other.bcs_vel;
        vel_grad = other.vel_grad;
        dirichlet_jump = other.dirichlet_jump;
        neumann_jump = other.neumann_jump;
        parms = other.parms;

    }

    test_case_stokes_eps() {}

    test_case_stokes_eps(Function &level_set__, Para_Interface &para_interface, params<T> parms_,
                         std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> sol_vel_,
                         std::function<T(const typename Mesh::point_type &)> sol_p_,
                         std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> rhs_fun_,
                         std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> bcs_vel_,
                         std::function<Eigen::Matrix<T, 2, 2>(const typename Mesh::point_type &)> vel_grad_,
                         std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> dirichlet_jump_,
                         std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> neumann_jump_)
            : level_set_(level_set__), parametric_interface(para_interface), sol_vel(sol_vel_), sol_p(sol_p_),
              rhs_fun(rhs_fun_),
              bcs_vel(bcs_vel_), parms(parms_), vel_grad(vel_grad_), dirichlet_jump(dirichlet_jump_),
              neumann_jump(neumann_jump_) {}
};


template<typename T, typename Function, typename Mesh>
class test_case_stokes_ref_pts {
public:
    Function level_set_;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> sol_vel;
    std::function<T(const typename Mesh::point_type &)> sol_p;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> rhs_fun;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> bcs_vel;
    std::function<Eigen::Matrix<T, 2, 2>(const typename Mesh::point_type &)> vel_grad;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> dirichlet_jump;
    std::function<Eigen::Matrix<T, 2, 1>(const T &, const std::vector<typename Mesh::point_type> &)> neumann_jump;

    struct params<T> parms;

    test_case_stokes_ref_pts(const test_case_stokes_ref_pts &other) {
        level_set_ = other.level_set_;
        sol_vel = other.sol_vel;
        sol_p = other.sol_p;
        rhs_fun = other.rhs_fun;
        bcs_vel = other.bcs_vel;
        vel_grad = other.vel_grad;
        dirichlet_jump = other.dirichlet_jump;
        neumann_jump = other.neumann_jump;

    }

    test_case_stokes_ref_pts() {}

    test_case_stokes_ref_pts
            (Function &level_set__, params<T> parms_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> sol_vel_,
             std::function<T(const typename Mesh::point_type &)> sol_p_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> rhs_fun_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> bcs_vel_,
             std::function<Eigen::Matrix<T, 2, 2>(const typename Mesh::point_type &)> vel_grad_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> dirichlet_jump_,
             std::function<Eigen::Matrix<T, 2, 1>(const T &,
                                                  const std::vector<typename Mesh::point_type> &)> neumann_jump_)
            : level_set_(level_set__), sol_vel(sol_vel_), sol_p(sol_p_), rhs_fun(rhs_fun_),
              bcs_vel(bcs_vel_), parms(parms_), vel_grad(vel_grad_), dirichlet_jump(dirichlet_jump_),
              neumann_jump(neumann_jump_) {}
};


template<typename T, typename Function, typename Mesh, typename Para_Interface>
class test_case_stokes_ref_pts_cont {
public:
    Function level_set_;
    Para_Interface parametric_interface;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> sol_vel;
    std::function<T(const typename Mesh::point_type &)> sol_p;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> rhs_fun;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> bcs_vel;
    std::function<Eigen::Matrix<T, 2, 2>(const typename Mesh::point_type &)> vel_grad;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> dirichlet_jump;
    std::function<Eigen::Matrix<T, 2, 1>(const T &, const size_t &)> neumann_jump;

    struct params<T> parms;

    test_case_stokes_ref_pts_cont(const test_case_stokes_ref_pts_cont &other) {
        level_set_ = other.level_set_;
        parametric_interface = other.parametric_interface;
        sol_vel = other.sol_vel;
        sol_p = other.sol_p;
        rhs_fun = other.rhs_fun;
        bcs_vel = other.bcs_vel;
        vel_grad = other.vel_grad;
        dirichlet_jump = other.dirichlet_jump;
        neumann_jump = other.neumann_jump;

    }

    test_case_stokes_ref_pts_cont() {}

    test_case_stokes_ref_pts_cont
            (Function &level_set__, Para_Interface &para_interface, params<T> parms_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> sol_vel_,
             std::function<T(const typename Mesh::point_type &)> sol_p_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> rhs_fun_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> bcs_vel_,
             std::function<Eigen::Matrix<T, 2, 2>(const typename Mesh::point_type &)> vel_grad_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> dirichlet_jump_,
             std::function<Eigen::Matrix<T, 2, 1>(const T &, const size_t &)> neumann_jump_)
            : level_set_(level_set__), parametric_interface(para_interface), sol_vel(sol_vel_), sol_p(sol_p_),
              rhs_fun(rhs_fun_),
              bcs_vel(bcs_vel_), parms(parms_), vel_grad(vel_grad_), dirichlet_jump(dirichlet_jump_),
              neumann_jump(neumann_jump_) {}
};


template<typename T, typename Mesh, typename Para_Interface>
class test_case_stokes_parametric {
public:
    Para_Interface parametric_interface;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> sol_vel;
    std::function<T(const typename Mesh::point_type &)> sol_p;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> rhs_fun;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> bcs_vel;
    std::function<Eigen::Matrix<T, 2, 2>(const typename Mesh::point_type &)> vel_grad;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> dirichlet_jump;
    std::function<Eigen::Matrix<T, 2, 1>(const T &, const size_t &)> neumann_jump;

    struct params<T> parms;

    test_case_stokes_parametric(const test_case_stokes_parametric &other) {
        parametric_interface = other.parametric_interface;
        sol_vel = other.sol_vel;
        sol_p = other.sol_p;
        rhs_fun = other.rhs_fun;
        bcs_vel = other.bcs_vel;
        vel_grad = other.vel_grad;
        dirichlet_jump = other.dirichlet_jump;
        neumann_jump = other.neumann_jump;

    }

    test_case_stokes_parametric() {}

    test_case_stokes_parametric
            (Para_Interface &para_interface, params<T> parms_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> sol_vel_,
             std::function<T(const typename Mesh::point_type &)> sol_p_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> rhs_fun_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> bcs_vel_,
             std::function<Eigen::Matrix<T, 2, 2>(const typename Mesh::point_type &)> vel_grad_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> dirichlet_jump_,
             std::function<Eigen::Matrix<T, 2, 1>(const T &, const size_t &)> neumann_jump_)
            : parametric_interface(para_interface), sol_vel(sol_vel_), sol_p(sol_p_), rhs_fun(rhs_fun_),
              bcs_vel(bcs_vel_), parms(parms_), vel_grad(vel_grad_), dirichlet_jump(dirichlet_jump_),
              neumann_jump(neumann_jump_) {}
};


///// test_case_stokes_1
// exact solution : X^2 (X^2 - 2X + 1) Y (4Y^2 - 6Y + 2) in the whole domain for vel_component 1
//                 -Y^2 (Y^2 - 2Y + 1) X (4X^2 - 6X + 2) in the whole domain for vel_component 2
//                  X^5 + Y^5                            in the whole domain for p
// where X = x - 0.5, Y = y - 0.5
// \kappa_1 = \kappa_2 = 1
template<typename T, typename Function, typename Mesh>
class test_case_stokes_1 : public test_case_stokes<T, Function, Mesh> {
public:
    test_case_stokes_1(Function &level_set__)
            : test_case_stokes<T, Function, Mesh>
                      (level_set__, params < T > (),
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // sol_vel
                           Matrix<T, 2, 1> ret;
                           T x1 = pt.x() - 0.5;
                           T x2 = x1 * x1;
                           T y1 = pt.y() - 0.5;
                           T y2 = y1 * y1;
                           ret(0) = x2 * (x2 - 2. * x1 + 1.) * y1 * (4. * y2 - 6. * y1 + 2.);
                           ret(1) = -y2 * (y2 - 2. * y1 + 1.) * x1 * (4. * x2 - 6. * x1 + 2.);
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> T { // p
                           T x1 = pt.x() - 0.5;
                           T y1 = pt.y() - 0.5;
                           T x2 = x1 * x1;
                           T y2 = y1 * y1;
                           return x2 * x2 * x1 + y2 * y2 * y1;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                           Matrix<T, 2, 1> ret;
                           T x1 = pt.x() - 0.5;
                           T x2 = x1 * x1;
                           T y1 = pt.y() - 0.5;
                           T y2 = y1 * y1;

                           T ax = x2 * (x2 - 2. * x1 + 1.);
                           T ay = y2 * (y2 - 2. * y1 + 1.);
                           T bx = x1 * (4. * x2 - 6. * x1 + 2.);
                           T by = y1 * (4. * y2 - 6. * y1 + 2.);
                           T cx = 12. * x2 - 12. * x1 + 2.;
                           T cy = 12. * y2 - 12. * y1 + 2.;
                           T dx = 24. * x1 - 12.;
                           T dy = 24. * y1 - 12.;

                           ret(0) = -cx * by - ax * dy + 5. * x2 * x2;
                           ret(1) = +cy * bx + ay * dx + 5. * y2 * y2;

                           return ret;
                       },
                       [&](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                           Matrix<T, 2, 1> ret;
                           T x1 = pt.x() - 0.5;
                           T x2 = x1 * x1;
                           T y1 = pt.y() - 0.5;
                           T y2 = y1 * y1;
                           ret(0) = x2 * (x2 - 2. * x1 + 1.) * y1 * (4. * y2 - 6. * y1 + 2.);
                           ret(1) = -y2 * (y2 - 2. * y1 + 1.) * x1 * (4. * x2 - 6. * x1 + 2.);
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> auto { // grad
                           Matrix<T, 2, 2> ret;
                           T x1 = pt.x() - 0.5;
                           T x2 = x1 * x1;
                           T y1 = pt.y() - 0.5;
                           T y2 = y1 * y1;

                           ret(0, 0) = x1 * (4. * x2 - 6. * x1 + 2.) * y1 * (4. * y2 - 6. * y1 + 2.);
                           ret(0, 1) = x2 * (x2 - 2. * x1 + 1.) * (12. * y2 - 12. * y1 + 2.);
                           ret(1, 0) = -y2 * (y2 - 2. * y1 + 1.) * (12. * x2 - 12. * x1 + 2.);
                           ret(1, 1) = -ret(0, 0);
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Neu */
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       }) {}
};

template<typename Mesh, typename Function>
auto make_test_case_stokes_1(const Mesh &msh, Function &level_set_function) {
    return test_case_stokes_1<typename Mesh::coordinate_type, Function, Mesh>(level_set_function);
}

///// test_case_stokes_2
// exact solution : X^2 (X^2 - 2X + 1) Y (4Y^2 - 6Y + 2) in the whole domain for vel_component 1
//                 -Y^2 (Y^2 - 2Y + 1) X (4X^2 - 6X + 2) in the whole domain for vel_component 2
//                   sin(X + Y)                          in the whole domain for p
// where X = x - 0.5, Y = y - 0.5
// \kappa_1 = \kappa_2 = 1
template<typename T, typename Function, typename Mesh>
class test_case_stokes_2 : public test_case_stokes<T, Function, Mesh> {
public:
    test_case_stokes_2(Function &level_set__)
            : test_case_stokes<T, Function, Mesh>
                      (level_set__, params < T > (),
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // sol_vel
                           Matrix<T, 2, 1> ret;
                           T x1 = pt.x() - 0.5;
                           T x2 = x1 * x1;
                           T y1 = pt.y() - 0.5;
                           T y2 = y1 * y1;
                           ret(0) = x2 * (x2 - 2. * x1 + 1.) * y1 * (4. * y2 - 6. * y1 + 2.);
                           ret(1) = -y2 * (y2 - 2. * y1 + 1.) * x1 * (4. * x2 - 6. * x1 + 2.);
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> T { // p
                           T x1 = pt.x() - 0.5;
                           T y1 = pt.y() - 0.5;
                           return std::sin(x1 + y1);
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                           Matrix<T, 2, 1> ret;
                           T x1 = pt.x() - 0.5;
                           T x2 = x1 * x1;
                           T y1 = pt.y() - 0.5;
                           T y2 = y1 * y1;

                           T ax = x2 * (x2 - 2. * x1 + 1.);
                           T ay = y2 * (y2 - 2. * y1 + 1.);
                           T bx = x1 * (4. * x2 - 6. * x1 + 2.);
                           T by = y1 * (4. * y2 - 6. * y1 + 2.);
                           T cx = 12. * x2 - 12. * x1 + 2.;
                           T cy = 12. * y2 - 12. * y1 + 2.;
                           T dx = 24. * x1 - 12.;
                           T dy = 24. * y1 - 12.;

                           ret(0) = -cx * by - ax * dy + std::cos(x1 + y1);
                           ret(1) = +cy * bx + ay * dx + std::cos(x1 + y1);

                           return ret;
                       },
                       [&](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                           Matrix<T, 2, 1> ret;
                           T x1 = pt.x() - 0.5;
                           T x2 = x1 * x1;
                           T y1 = pt.y() - 0.5;
                           T y2 = y1 * y1;
                           ret(0) = x2 * (x2 - 2. * x1 + 1.) * y1 * (4. * y2 - 6. * y1 + 2.);
                           ret(1) = -y2 * (y2 - 2. * y1 + 1.) * x1 * (4. * x2 - 6. * x1 + 2.);
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> auto { // grad
                           Matrix<T, 2, 2> ret;
                           T x1 = pt.x() - 0.5;
                           T x2 = x1 * x1;
                           T y1 = pt.y() - 0.5;
                           T y2 = y1 * y1;

                           ret(0, 0) = x1 * (4. * x2 - 6. * x1 + 2.) * y1 * (4. * y2 - 6. * y1 + 2.);
                           ret(0, 1) = x2 * (x2 - 2. * x1 + 1.) * (12. * y2 - 12. * y1 + 2.);
                           ret(1, 0) = -y2 * (y2 - 2. * y1 + 1.) * (12. * x2 - 12. * x1 + 2.);
                           ret(1, 1) = -ret(0, 0);
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Neu */
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       }) {}
};

template<typename Mesh, typename Function>
auto make_test_case_stokes_2(const Mesh &msh, Function &level_set_function) {
    return test_case_stokes_2<typename Mesh::coordinate_type, Function, Mesh>(level_set_function);
}


///// test_case_static_bubble ANALYTICAL LEVEL SET
// !! available for circle_level_set only !!
// exact solution : 0 in the whole domain for vel_component 1
//                  0 in the whole domain for vel_component 2
//                  k/R - \pi k R         in Omega_1 for p
//                  - \pi k R             in Omega_2 for p
// \kappa_1 = \kappa_2 = 1

// I PUT COMMENTED SINCE THERE IS A PROBLEM WITH THE CONSTRUCTOR: IF I DECLERE THE circle_level_set DIRECTLY HERE IT CANNOT BE PASSED AS A REFERENCE. IW WOULD BE LOST.
template<typename T, typename Mesh, typename Function>
class test_case_static_bubble : public test_case_stokes<T, Function, Mesh> {
//<<<<<<< HEAD
//   public:
//    test_case_static_bubble(T R, T a, T b, T k)
//        : test_case_stokes<T, circle_level_set<T>, Mesh>
//        (circle_level_set<T>(R, a, b), params<T>(),
//         [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> { // sol_vel
//=======
public:
    test_case_static_bubble(T R, T a, T b, T k, Function &level_set_)
            : test_case_stokes<T, Function, Mesh>
                      (level_set_, params < T > (),
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // sol_vel
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [a, b, R, k](const typename Mesh::point_type &pt) -> T { // p
                           T x1 = pt.x() - a;
                           T y1 = pt.y() - b;
                           if (x1 * x1 + y1 * y1 < R * R)
                               return k / R - M_PI * R * k;
                           else
                               return -M_PI * R * k;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
//>>>>>>> 928e4b8 (Analytical Velocity. LS evolution.)
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
//<<<<<<< HEAD
//         [a,b,R,k](const typename Mesh::point_type& pt) -> T { // p
//             T x1 = pt.x() - a;
//             T y1 = pt.y() - b;
//             if( x1 * x1 + y1 * y1 < R*R)
//                 return k / R - M_PI * R * k;
//             else
//                 return -M_PI * R * k;},
//         [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> { // rhs
//             Matrix<T, 2, 1> ret;
//             ret(0) = 0.0;
//             ret(1) = 0.0;
//             return ret;},
//         [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> { // bcs
//=======
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> auto { // grad
                           Matrix<T, 2, 2> ret;
                           ret(0, 0) = 0.0;
                           ret(0, 1) = 0.0;
                           ret(1, 0) = 0.0;
                           ret(1, 1) = 0.0;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {// Dir //
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
//<<<<<<< HEAD
//         [](const typename Mesh::point_type& pt) -> auto { // grad
//             Matrix<T, 2, 2> ret;
//             ret(0,0) = 0.0;
//             ret(0,1) = 0.0;
//             ret(1,0) = 0.0;
//             ret(1,1) = 0.0;
//             return ret;},
//         [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
//             Matrix<T, 2, 1> ret;
//             ret(0) = 0.0;
//             ret(1) = 0.0;
//             return ret;},
//         [this, k, R](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> {/* Neu */
//             Matrix<T, 2, 1> ret;
//             Matrix<T, 1, 2> normal = this->level_set_.normal(pt);
//             ret(0) = - k / R * normal(0);
//             ret(1) = - k / R * normal(1);
//             return ret;})
//        {}
//=======
                       [this, k, R](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {// Neu //
                           Matrix<T, 2, 1> ret;
                           Matrix<T, 1, 2> normal = this->level_set_.normal(pt);
                           ret(0) = -k / R * normal(0);
                           ret(1) = -k / R * normal(1);
                           return ret;
                       }) {}
//>>>>>>> 928e4b8 (Analytical Velocity. LS evolution.)
};

template<typename Mesh, typename T, typename Function>
auto make_test_case_static_bubble(const Mesh &msh, T R, T a, T b, T k, Function &level_set_) {
    return test_case_static_bubble<typename Mesh::coordinate_type, Mesh, Function>(R, a, b, k, level_set_);
}


template<typename T, typename Mesh, typename Function>
class test_case_M_shaped : public test_case_stokes<T, Function, Mesh> {
public:
    test_case_M_shaped(T k, Function &level_set_)
            : test_case_stokes<T, Function, Mesh>
                      (level_set_, params < T > (),
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // sol_vel
                           Matrix<T, 2, 1> ret;
                           T x = pt.x();
                           T y = pt.y();
                           ret(0) = x * x * (1.0 - x) * (1.0 - x) * (2. * y - 6. * y * y + 4. * y * y * y);
                           ret(1) = -y * y * (1. - y) * (1. - y) * (2. * x - 6. * x * x + 4. * x * x * x);
                           return ret;

                       },
                       [](const typename Mesh::point_type &pt) -> T { // p
                           T x = pt.x();
                           T y = pt.y();
                           return x * (1.0 - x);

                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                           Matrix<T, 2, 1> ret;
                           T x = pt.x();
                           T y = pt.y();
//            ret(0) =    1.0-2.0*x - (x*x+ x*x*x*x -2.0*x*x*x)*(-12.0+24.0*y);
//            ret(1) =   - (-y*y- y*y*y*y + 2.0*y*y*y)*(-12.0+24.0*x);
                           ret(0) = 1.0 - 2.0 * x - (4.0 * (-1.0 + 2.0 * y) * (3.0 * (-1.0 + x) * (-1.0 + x) * x * x +
                                                                               (1.0 + 6.0 * (-1.0 + x) * x) *
                                                                               (-1.0 + y) * y));
                           ret(1) = -(4.0 * (-1.0 + 2.0 * x) * (-3.0 * (-1.0 + y) * (-1.0 + y) * y * y -
                                                                (-1.0 + x) * x * (1.0 + 6.0 * (-1.0 + y) * y)));
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                           Matrix<T, 2, 1> ret;
                           T x = pt.x();
                           T y = pt.y();
                           ret(0) = x * x * (1.0 - x) * (1.0 - x) * (2. * y - 6. * y * y + 4. * y * y * y);
                           ret(1) = -y * y * (1. - y) * (1. - y) * (2. * x - 6. * x * x + 4. * x * x * x);
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> auto { // grad
                           Matrix<T, 2, 2> ret;
                           T x = pt.x();
                           T y = pt.y();
                           ret(0, 0) = 2.0 * (1.0 - x) * (1.0 - x) * x * (2.0 * y - 6.0 * y * y + 4.0 * y * y * y) -
                                       2.0 * (1.0 - x) * x * x * (2.0 * y - 6.0 * y * y + 4.0 * y * y * y);
                           ret(0, 1) = (1.0 - x) * (1.0 - x) * x * x * (2.0 - 12.0 * y + 12.0 * y * y);
                           ret(1, 0) = -((2.0 - 12.0 * x + 12.0 * x * x) * (1.0 - y) * (1.0 - y) * y * y);
                           ret(1, 1) = -2.0 * (2.0 * x - 6.0 * x * x + 4.0 * x * x * x) * (1.0 - y) * (1.0 - y) * y +
                                       2.0 * (2.0 * x - 6.0 * x * x + 4.0 * x * x * x) * (1.0 - y) * y * y;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {// Dir //
                           Matrix<T, 2, 1> ret;
                           T x = pt.x();
                           T y = pt.y();
                           ret(0) = 0.; // x*x*(1-x)*(1-x)*(2*y-6*y*y+4*y*y*y);
                           ret(1) = 0.; // -y*y*(1-y)*(1-y)*(2*x-6*x*x+4*x*x*x);
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {// Neu
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       }) {}
};

template<typename Mesh, typename T, typename Function>
auto make_test_case_M_shaped(const Mesh &msh, T k, Function &level_set_) {
    return test_case_M_shaped<typename Mesh::coordinate_type, Mesh, Function>(k, level_set_);
}


template<typename T, typename Function, typename Mesh>
class test_case_stokes_old {
public:
    Function level_set_;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> sol_vel;
    std::function<T(const typename Mesh::point_type &)> sol_p;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> rhs_fun;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> bcs_vel;
    std::function<Eigen::Matrix<T, 2, 2>(const typename Mesh::point_type &)> vel_grad;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> dirichlet_jump;
    std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> neumann_jump;

    struct params<T> parms;

    test_case_stokes_old() {}

    test_case_stokes_old
            (Function level_set__, params<T> parms_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> sol_vel_,
             std::function<T(const typename Mesh::point_type &)> sol_p_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> rhs_fun_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> bcs_vel_,
             std::function<Eigen::Matrix<T, 2, 2>(const typename Mesh::point_type &)> vel_grad_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> dirichlet_jump_,
             std::function<Eigen::Matrix<T, 2, 1>(const typename Mesh::point_type &)> neumann_jump_)
            : level_set_(level_set__), sol_vel(sol_vel_), sol_p(sol_p_), rhs_fun(rhs_fun_),
              bcs_vel(bcs_vel_), parms(parms_), vel_grad(vel_grad_), dirichlet_jump(dirichlet_jump_),
              neumann_jump(neumann_jump_) {}
};


///// test_case_static_bubble
// !! available for circle_level_set only !!
// exact solution : 0 in the whole domain for vel_component 1
//                  0 in the whole domain for vel_component 2
//                  k/R - \pi k R         in Omega_1 for p
//                  - \pi k R             in Omega_2 for p
// \kappa_1 = \kappa_2 = 1
template<typename T, typename Mesh>
class test_case_static_bubble_old : public test_case_stokes_old<T, circle_level_set < T>, Mesh

>
{
public:

test_case_static_bubble_old(T R, T a, T b, T k)
        : test_case_stokes_old<T, circle_level_set < T>, Mesh

>
(
circle_level_set<T>(R, a, b
),

params<T>(),

[](
const typename Mesh::point_type &pt
) -> Eigen::Matrix<T, 2, 1> { // sol_vel
Matrix<T, 2, 1> ret;
ret(0) = 0.0;
ret(1) = 0.0;
return
ret;
},
[a,b,R,k](
const typename Mesh::point_type &pt
) -> T { // p
T x1 = pt.x() - a;
T y1 = pt.y() - b;
if(
x1 *x1
+
y1 *y1<R *R
)
return k / R -
M_PI *R
*
k;
else
return -
M_PI *R
*
k;
},
[](
const typename Mesh::point_type &pt
) -> Eigen::Matrix<T, 2, 1> { // rhs
Matrix<T, 2, 1> ret;
ret(0) = 0.0;
ret(1) = 0.0;
return
ret;
},
[](
const typename Mesh::point_type &pt
) -> Eigen::Matrix<T, 2, 1> { // bcs
Matrix<T, 2, 1> ret;
ret(0) = 0.0;
ret(1) = 0.0;
return
ret;
},
[](
const typename Mesh::point_type &pt
) -> auto { // grad
Matrix<T, 2, 2> ret;
ret(0,0) = 0.0;
ret(0,1) = 0.0;
ret(1,0) = 0.0;
ret(1,1) = 0.0;
return
ret;
},
[](
const typename Mesh::point_type &pt
) -> Eigen::Matrix<T, 2, 1> {/* Dir */
Matrix<T, 2, 1> ret;
ret(0) = 0.0;
ret(1) = 0.0;
return
ret;
},
[this, k, R](
const typename Mesh::point_type &pt
) -> Eigen::Matrix<T, 2, 1> {/* Neu */
Matrix<T, 2, 1> ret;
Matrix<T, 1, 2> normal = this->level_set_.normal(pt);
ret(0) = - k /
R *normal(0);
ret(1) = - k /
R *normal(1);
return
ret;
})
{
}
};

template<typename Mesh, typename T>
auto make_test_case_static_bubble_old(const Mesh &msh, T R, T a, T b, T k) {
    return test_case_static_bubble_old < typename Mesh::coordinate_type, Mesh > (R, a, b, k);
}


///// test_case_static_bubble NUMERICAL LEVEL SET
// !! available for circle_level_set only !!
// exact solution : 0 in the whole domain for vel_component 1
//                  0 in the whole domain for vel_component 2
//                  k/R - \pi k R         in Omega_1 for p
//                  - \pi k R             in Omega_2 for p
// \kappa_1 = \kappa_2 = 1
template<typename T, typename Function, typename Mesh>
class test_case_static_bubble_numerical_ls : public test_case_stokes<T, Function, Mesh> {
    Mesh m_msh;
    typename Mesh::cell_type m_cl;

public:
    explicit test_case_static_bubble_numerical_ls(Function &level_set__, params<T> parms_)
            : test_case_stokes<T, Function, Mesh>
                      (level_set__, parms_,
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // sol_vel
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [level_set__, this](const typename Mesh::point_type &pt) mutable -> T { // p
                           T k = 0.05;
                           level_set__.cell_assignment(m_cl);
                           T R = level_set__.radius;
                           if (level_set__(pt) < 0)
                               return k / R - M_PI * R * k;
                           else
                               return -M_PI * R * k;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> auto { // grad
                           Matrix<T, 2, 2> ret;
                           ret(0, 0) = 0.0;
                           ret(0, 1) = 0.0;
                           ret(1, 0) = 0.0;
                           ret(1, 1) = 0.0;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [this, level_set__](
                               const typename Mesh::point_type &pt) mutable -> Eigen::Matrix<T, 2, 1> {/* Neu */
                           Matrix<T, 2, 1> ret;
                           T k = 0.05;
                           level_set__.cell_assignment(m_cl);
                           T R = level_set__.radius;
                           Matrix<T, 1, 2> normal = level_set__.normal(pt);
                           ret(0) = -k / R * normal(0);
                           ret(1) = -k / R * normal(1);
                           return ret;
                       }) {}

    test_case_static_bubble_numerical_ls(const test_case_static_bubble_numerical_ls &other)
            : test_case_stokes<T, Function, Mesh>(other) {
        m_msh = other.m_msh;
        m_cl = other.m_cl;
    }

    void test_case_cell_assignment(const typename Mesh::cell_type &cl_new) {
        m_cl = cl_new;
    }

    void refresh_lambdas(Function &level_set__, params<T> parms_, bool sym_grad) {


        this->neumann_jump = [this, level_set__](
                const typename Mesh::point_type &pt) mutable -> Eigen::Matrix<T, 2, 1> {/* Neu */
            Matrix<T, 2, 1> ret;
            T k = 0.05;
            T R = level_set__.radius;
            level_set__.cell_assignment(m_cl);
            Matrix<T, 1, 2> normal = level_set__.normal(pt);
            ret(0) = -k / R * normal(0);
            ret(1) = -k / R * normal(1);
            return ret;
        };


        this->sol_p = [level_set__, this](const typename Mesh::point_type &pt) mutable -> T {/* Pressure */
            T k = 0.05;
            level_set__.cell_assignment(m_cl);
            T R = level_set__.radius;
            //---> ATTENTION, THE LEVEL SET GAMMA COULD BE DIFFERENT FROM ZERO!
            if (level_set__(pt) < 0)
                return k / R - M_PI * R * k;
            else
                return -M_PI * R * k;
        };

    }

    void test_case_mesh_assignment(const Mesh &msh_new) {

        std::cout << "-----> test_case_mesh_assignment " << std::endl;
        m_msh = msh_new;

    }

    typename Mesh::cell_type &upload_cl() {
        return m_cl;
    }


};

template<typename Mesh, typename T, typename Function>
auto make_test_case_static_bubble_numerical_ls(const Mesh &msh, Function &level_set_function, params<T> parms_) {
    return test_case_static_bubble_numerical_ls<typename Mesh::coordinate_type, Function, Mesh>(level_set_function,
                                                                                                parms_);
}


/// FATTA DA GUILLAUME !!!!

///// test_case_kink_velocity
// !! available for circle_level_set only !!
// exact solution : u(r) sin(theta) in the whole domain for vel_component 1
//                 -u(r) cos(theta) in the whole domain for vel_component 2
//         with u(r) = r^6 / kappa_1 in Omega_1
//              u(r) = (r^6 - R^6)/kappa_2 + R^6/kappa_1 in Omega_2


//<<<<<<< HEAD


//                 p(r) = r^4 - 7/180    in the whole domain for p
// \kappa_1 , \kappa_2 given
//template<typename T, typename Mesh>
//class test_case_kink_velocity: public test_case_stokes<T, circle_level_set<T>, Mesh>
//{
//   public:
//    test_case_kink_velocity(T R, T a, T b, params<T> parms_, bool sym_grad)
//        : test_case_stokes<T, circle_level_set<T>, Mesh>
//        (circle_level_set<T>(R, a, b), parms_,
//         [R,a,b,parms_](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> { // sol_vel
//=======
//                   sin(x+y)     in the whole domain for p
//              p(r) = r^4 - 7/180    in the whole domain for p
// \kappa_1 , \kappa_2 given

template<typename T, typename Mesh, typename Function>
class test_case_kink_velocity : public test_case_stokes<T, Function, Mesh> {
public:
    test_case_kink_velocity(T R, T a, T b, params<T> parms_, bool sym_grad, Function &level_set_)
            : test_case_stokes<T, Function, Mesh>
                      (level_set_, parms_,
                       [R, a, b, parms_](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // sol_vel
                           Matrix<T, 2, 1> ret;
                           T x1 = pt.x() - a;
                           T y1 = pt.y() - b;
                           T r2 = x1 * x1 + y1 * y1;
                           T r = std::sqrt(r2);
                           T R2 = R * R;
                           T ur;
                           if (r2 < R2)
                               ur = r2 * r2 * r2 / parms_.kappa_1;
                           else
                               ur = (r2 * r2 * r2 - R2 * R2 * R2) / parms_.kappa_2
                                    + R2 * R2 * R2 / parms_.kappa_1;
                           ret(0) = ur * y1 / r;
                           ret(1) = -ur * x1 / r;
                           return ret;
                       },
                       [a, b](const typename Mesh::point_type &pt) -> T { // p
                           T x1 = pt.x() - a;
                           T y1 = pt.y() - b;
                           T r2 = x1 * x1 + y1 * y1;
                           return r2 * r2 - 7.0 / 180.0;
                       },
                       [R, a, b, parms_](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                           T x1 = pt.x() - a;
                           T y1 = pt.y() - b;
                           T r2 = x1 * x1 + y1 * y1;
                           T r = std::sqrt(r2);
                           T R2 = R * R;
                           T kappa_Delta_ur = 35.0 * r2 * r2;
                           if (r2 > R2)
                               kappa_Delta_ur = kappa_Delta_ur
                                                + (1.0 - parms_.kappa_2 / parms_.kappa_1) * R2 * R2 * R2 / r2;

                           Matrix<T, 2, 1> ret;
                           ret(0) = -y1 * kappa_Delta_ur / r + 4.0 * x1 * r2;
                           ret(1) = x1 * kappa_Delta_ur / r + 4.0 * y1 * r2;
                           return ret;
                       },
                       [R, a, b, parms_](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                           Matrix<T, 2, 1> ret;
                           T x1 = pt.x() - a;
                           T y1 = pt.y() - b;
                           T r2 = x1 * x1 + y1 * y1;
                           T r = std::sqrt(r2);
                           T R2 = R * R;
                           T ur;
                           if (r2 < R2)
                               ur = r2 * r2 * r2 / parms_.kappa_1;
                           else
                               ur = (r2 * r2 * r2 - R2 * R2 * R2) / parms_.kappa_2
                                    + R2 * R2 * R2 / parms_.kappa_1;
                           ret(0) = ur * y1 / r;
                           ret(1) = -ur * x1 / r;
                           return ret;
                       },
                       [R, a, b, parms_](const typename Mesh::point_type &pt) -> auto { // grad
                           T x1 = (pt.x() - a);
                           T y1 = (pt.y() - b);
                           T r2 = x1 * x1 + y1 * y1;
                           T r = std::sqrt(r2);
                           T R2 = R * R;
                           T A = 5.0 * r * r2 * x1 * y1;
                           T B = 5.0 * r * r2 * y1 * y1 + r * r2 * r2;
                           T C = -5.0 * r * r2 * x1 * x1 - r * r2 * r2;
                           T D = -5.0 * r * r2 * x1 * y1;

                           Matrix<T, 2, 2> ret;
                           if (r2 < R2) {
                               ret(0, 0) = A / parms_.kappa_1;
                               ret(0, 1) = B / parms_.kappa_1;
                               ret(1, 0) = C / parms_.kappa_1;
                               ret(1, 1) = D / parms_.kappa_1;
                           } else {
                               ret(0, 0) = A / parms_.kappa_2
                                           + (1.0 / parms_.kappa_2 - 1.0 / parms_.kappa_1) * R2 * R2 * R2 * x1 * y1 /
                                             (r * r2);
                               ret(0, 1) = B / parms_.kappa_2
                                           - (1.0 / parms_.kappa_2 - 1.0 / parms_.kappa_1) * R2 * R2 * R2 * x1 * x1 /
                                             (r * r2);
                               ret(1, 0) = C / parms_.kappa_2
                                           + (1.0 / parms_.kappa_2 - 1.0 / parms_.kappa_1) * R2 * R2 * R2 * y1 * y1 /
                                             (r * r2);
                               ret(1, 1) = D / parms_.kappa_2
                                           - (1.0 / parms_.kappa_2 - 1.0 / parms_.kappa_1) * R2 * R2 * R2 * x1 * y1 /
                                             (r * r2);
                           }
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [R, a, b, parms_, sym_grad](
                               const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Neu */
                           Matrix<T, 2, 1> ret;
                           if (sym_grad) {
                               T x1 = (pt.x() - a);
                               T y1 = (pt.y() - b);
                               T r2 = x1 * x1 + y1 * y1;
                               ret(0) = -(1.0 - parms_.kappa_2 / parms_.kappa_1) * r2 * r2 * y1;
                               ret(1) = (1.0 - parms_.kappa_2 / parms_.kappa_1) * r2 * r2 * x1;
                           } else {
                               ret(0) = 0.0;
                               ret(1) = 0.0;
                           }
                           return ret;
                       }) {}
};

template<typename Mesh, typename T, typename Function>
auto
make_test_case_kink_velocity(const Mesh &msh, T R, T a, T b, params<T> parms_, bool sym_grad, Function &level_set_) {
    return test_case_kink_velocity<typename Mesh::coordinate_type, Mesh, Function>(R, a, b, parms_, sym_grad,
                                                                                   level_set_);
}


///// test_case_kink_velocity
// !! available for circle_level_set only !!
// exact solution : u(r) sin(theta) in the whole domain for vel_component 1
//                 -u(r) cos(theta) in the whole domain for vel_component 2
//         with u(r) = r^6 / kappa_1 in Omega_1
//              u(r) = (r^6 - R^6)/kappa_2 + R^6/kappa_1 in Omega_2
//                 p(r) = r^4 - 7/180    in the whole domain for p
// \kappa_1 , \kappa_2 given
template<typename T, typename Mesh>
class test_case_kink_velocity_old : public test_case_stokes_old<T, circle_level_set < T>, Mesh

>
{
public:

test_case_kink_velocity_old(T R, T a, T b, params<T> parms_, bool sym_grad)
        : test_case_stokes_old<T, circle_level_set < T>, Mesh

>
(
circle_level_set<T>(R, a, b
), parms_,
[R,a,b,parms_](
const typename Mesh::point_type &pt
) -> Eigen::Matrix<T, 2, 1> { // sol_vel
Matrix<T, 2, 1> ret;
T x1 = pt.x() - a;
T y1 = pt.y() - b;
T r2 = x1 * x1 + y1 * y1;
T r = std::sqrt(r2);
T R2 = R * R;
T ur;
if( r2<R2 )
ur = r2 * r2 * r2 / parms_.kappa_1;
else
ur = (r2 * r2 * r2 - R2 * R2 * R2) / parms_.kappa_2 + R2 * R2 * R2 / parms_.kappa_1;
ret(0) =
ur *y1
/
r;
ret(1) = -
ur *x1
/
r;
return
ret;
},
[a,b](
const typename Mesh::point_type &pt
) -> T { // p
T x1 = pt.x() - a;
T y1 = pt.y() - b;
T r2 = x1 * x1 + y1 * y1;
return
r2 *r2
- 7.0/180.0;},
[R,a,b,parms_](
const typename Mesh::point_type &pt
) -> Eigen::Matrix<T, 2, 1> { // rhs
T x1 = pt.x() - a;
T y1 = pt.y() - b;
T r2 = x1 * x1 + y1 * y1;
T r = std::sqrt(r2);
T R2 = R * R;
T kappa_Delta_ur = 35.0 * r2 * r2;
if(r2 > R2)
kappa_Delta_ur = kappa_Delta_ur + (1.0 - parms_.kappa_2 / parms_.kappa_1) * R2 * R2 * R2 / r2;

Matrix<T, 2, 1> ret;
ret(0) = -
y1 *kappa_Delta_ur
/ r + 4.0*
x1 *r2;
ret(1) =
x1 *kappa_Delta_ur
/ r + 4.0*
y1 *r2;
return
ret;
},
[R,a,b,parms_](
const typename Mesh::point_type &pt
) -> Eigen::Matrix<T, 2, 1> { // bcs
Matrix<T, 2, 1> ret;
T x1 = pt.x() - a;
T y1 = pt.y() - b;
T r2 = x1 * x1 + y1 * y1;
T r = std::sqrt(r2);
T R2 = R * R;
T ur;
if( r2<R2 )
ur = r2 * r2 * r2 / parms_.kappa_1;
else
ur = (r2 * r2 * r2 - R2 * R2 * R2) / parms_.kappa_2
     + R2 * R2 * R2 / parms_.kappa_1;
ret(0) =
ur *y1
/
r;
ret(1) = -
ur *x1
/
r;
return
ret;
},
[R,a,b,parms_](
const typename Mesh::point_type &pt
) -> auto { // grad
T x1 = (pt.x() - a);
T y1 = (pt.y() - b);
T r2 = x1 * x1 + y1 * y1;
T r = std::sqrt(r2);
T R2 = R * R;
T A = 5.0 * r * r2 * x1 * y1;
T B = 5.0 * r * r2 * y1 * y1 + r * r2 * r2;
T C = -5.0 * r * r2 * x1 * x1 - r * r2 * r2;
T D = -5.0 * r * r2 * x1 * y1;

Matrix<T, 2, 2> ret;
if(r2<R2)
{
ret(0,0) = A / parms_.
kappa_1;
ret(0,1) = B / parms_.
kappa_1;
ret(1,0) = C / parms_.
kappa_1;
ret(1,1) = D / parms_.
kappa_1;
}
else
{
ret(0,0) = A / parms_.kappa_2 + (1.0/parms_.kappa_2 - 1.0/parms_.kappa_1) *
R2 *R2
*
R2 *x1
* y1 / (
r *r2
);
ret(0,1) = B / parms_.kappa_2 - (1.0/parms_.kappa_2 - 1.0/parms_.kappa_1) *
R2 *R2
*
R2 *x1
* x1 / (
r *r2
);
ret(1,0) = C / parms_.kappa_2 + (1.0/parms_.kappa_2 - 1.0/parms_.kappa_1) *
R2 *R2
*
R2 *y1
* y1 / (
r *r2
);
ret(1,1) = D / parms_.kappa_2 - (1.0/parms_.kappa_2 - 1.0/parms_.kappa_1) *
R2 *R2
*
R2 *x1
* y1 / (
r *r2
);
}
return
ret;
},
[](
const typename Mesh::point_type &pt
) -> Eigen::Matrix<T, 2, 1> {/* Dir */
Matrix<T, 2, 1> ret;
ret(0) = 0.0;
ret(1) = 0.0;
return
ret;
},
[R,a,b,parms_,sym_grad](
const typename Mesh::point_type &pt
) -> Eigen::Matrix<T, 2, 1> {/* Neu */
Matrix<T, 2, 1> ret;
if(sym_grad)
{
T x1 = (pt.x() - a);
T y1 = (pt.y() - b);
T r2 = x1 * x1 + y1 * y1;
ret(0) = - (1.0 - parms_.kappa_2 / parms_.kappa_1) *
r2 *r2
*
y1;
ret(1) = (1.0 - parms_.kappa_2 / parms_.kappa_1) *
r2 *r2
*
x1;
}
else
{
ret(0) = 0.0;
ret(1) = 0.0;
}
return
ret;
})
{
}
};

template<typename Mesh, typename T>
auto make_test_case_kink_velocity_old(const Mesh &msh, T R, T a, T b, params<T> parms_, bool sym_grad) {
    return test_case_kink_velocity_old < typename Mesh::coordinate_type, Mesh > (R, a, b, parms_, sym_grad);
}


// ADDED BY STEFANO: Stokes interface problem

///// test_case_kink_velocity --> DISCRETE LEVEL SET
// !! available for circle_level_set only !!
// exact solution : u(r) sin(theta) in the whole domain for vel_component 1
//                 -u(r) cos(theta) in the whole domain for vel_component 2
//         with u(r) = r^6 / kappa_1 in Omega_1
//              u(r) = (r^6 - R^6)/kappa_2 + R^6/kappa_1 in Omega_2
//                   sin(x+y)     in the whole domain for p
// \kappa_1 , \kappa_2 given
template<typename T, typename Mesh, typename Function>
class test_case_eshelby :
        public test_case_stokes<T, Function, Mesh> {
public:
    test_case_eshelby(const Function &level_set__, params<T> parms_, bool sym_grad)
            : test_case_stokes<T, Function, Mesh>
                      (level_set__, parms_,
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // sol_vel
                           Matrix<T, 2, 1> ret;
                           T C = 2.0;
                           T mid_point = 1.0 / 2.0;
                           T x1 = pt.x() - mid_point;
                           T y1 = mid_point - pt.y();
                           ret(0) = C * x1;
                           ret(1) = C * y1;


                           return ret;
                       },
                       [level_set__](const typename Mesh::point_type &pt) -> T { // p
//return 1.0;

                           T k = 1.0;
                           T R = 1.0 / 3.0;
                           if (level_set__(pt) < 0)
                               return k / R - M_PI * R * k;
                           else
                               return -M_PI * R * k;

                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                           Matrix<T, 2, 1> ret;
                           T C = 2.0;
                           T mid_point = 1.0 / 2.0;
                           T x1 = pt.x() - mid_point;
                           T y1 = mid_point - pt.y();
                           ret(0) = C * x1;
                           ret(1) = C * y1;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> auto { // grad

                           Matrix<T, 2, 2> ret;
                           T C = 2.0;
                           ret(0, 0) = C * 1.0;
                           ret(0, 1) = C * 0.0;
                           ret(1, 0) = C * 0.0;
                           ret(1, 1) = C * (-1.0);
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [level_set__, parms_, sym_grad](
                               const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Neu */
                           Matrix<T, 2, 1> ret;
                           if (sym_grad) {
                               T gamma = 1.0;
//T k = 1.0;
//T R = 1.0/3.0;

//T H = level_set__.normal(pt)
                               ret(0) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                               ret(1) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);

//ret(0) = - k / R * level_set__.normal(pt)(0);
//ret(1) = - k / R * level_set__.normal(pt)(1);

                           } else {
                               ret(0) = 0.0;
                               ret(1) = 0.0;
                           }
                           return ret;
                       }) {
    }
};

template<typename Mesh, typename T, typename Function>
auto make_test_case_eshelby(const Mesh &msh, const Function &level_set_function, params<T> parms_, bool sym_grad) {
    return test_case_eshelby<typename Mesh::coordinate_type, Mesh, Function>(level_set_function, parms_, sym_grad);
}

/// FATTO BY STEFANO
// Starting from an elliptic level set -> final circle equilibrium
// exact solution : u = 0
//                  constant = \kappa    in the whole domain for p
// \kappa  given

template<typename T, typename Mesh, typename Function>
class test_case_eshelby_2 :
        public test_case_stokes<T, Function, Mesh> {
public:
    test_case_eshelby_2(const Function &level_set__, params<T> parms_, bool sym_grad)
            : test_case_stokes<T, Function, Mesh>
                      (level_set__, parms_,
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {
// sol_vel
                           Matrix<T, 2, 1> ret;

                           ret(0) = 0.0;
                           ret(1) = 0.0;

                           return ret;
                       },
                       [level_set__](const typename Mesh::point_type &pt) -> T { // p

                           return 0.0;

                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                           Matrix<T, 2, 1> ret;

                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> auto { // grad

                           Matrix<T, 2, 2> ret;
                           ret(0, 0) = 0.0;
                           ret(0, 1) = 0.0;
                           ret(1, 0) = 0.0;
                           ret(1, 1) = (0.0);
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [level_set__, sym_grad](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Neu */
                           Matrix<T, 2, 1> ret;
                           if (sym_grad) {
                               T gamma = 1.0;
//T k = 1.0;
//T R = 1.0/3.0;

//T H = level_set__.normal(pt)

                               ret(0) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                               ret(1) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);


//ret(0) = - k / R * level_set__.normal(pt)(0);
//ret(1) = - k / R * level_set__.normal(pt)(1);

                           } else {
                               T gamma = 1.0;

                               ret(0) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                               ret(1) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);
                           }
                           return ret;
                       }) {
    }
};

template<typename Mesh, typename T, typename Function>
auto make_test_case_eshelby_2(const Mesh &msh, const Function &level_set_function, params<T> parms_, bool sym_grad) {
    return test_case_eshelby_2<typename Mesh::coordinate_type, Mesh, Function>(level_set_function, parms_, sym_grad);
}


template<typename T, typename Mesh, typename Function>
class test_case_eshelby_2_prova :
        public test_case_stokes<T, Function, Mesh> {

public:

    Mesh m_msh;
    typename Mesh::cell_type m_cl;
//    Mesh* msh_pt ;
//    typename Mesh::cell_type* cl_pt ;
//    size_t i = 2;
//std::shared_ptr<typename Mesh::cell_type> cl_pointer;

    explicit test_case_eshelby_2_prova(Function &level_set__, params<T> parms_, bool sym_grad)
            : test_case_stokes<T, Function, Mesh>
                      (level_set__, parms_,
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {
// sol_vel
                           Matrix<T, 2, 1> ret;

                           ret(0) = 0.0;
                           ret(1) = 0.0;

                           return ret;
                       },
                       [level_set__](const typename Mesh::point_type &pt) -> T { // p



                           return 0.0;

                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                           Matrix<T, 2, 1> ret;

                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> auto { // grad

                           Matrix<T, 2, 2> ret;
                           ret(0, 0) = 0.0;
                           ret(0, 1) = 0.0;
                           ret(1, 0) = 0.0;
                           ret(1, 1) = (0.0);
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [level_set__, sym_grad, this](
                               const typename Mesh::point_type &pt) mutable -> Eigen::Matrix<T, 2, 1> {/* Neu */
                           Matrix<T, 2, 1> ret;
                           if (sym_grad) {
                               T gamma = 1.0;
//T k = 1.0;
//T R = 1.0/3.0;
//cl = cl_pointer.get() ;
//std::cout<<"SONO  IN NEUMANN CONDITION."<<std::endl;
//                std::cout<<"i = "<<i<<std::endl;
//auto cl_uploaded = this->cl ;
//                i++;
//                std::cout<<"i = "<<i<<std::endl;
//                Mesh msh_prova = *msh_pt ;
//                typename Mesh::cell_type cl_prova = *cl_pt ;


//                std::cout<<"-----> test_case_eshelby_2_prova : CELL PROVA = "<<offset(msh_prova,cl_prova)<<std::endl;
//if(i==3)
//   this->cl = msh.cells[227];
//std::cout<<"-----> test_case_eshelby_2_prova : CELL = "<<offset(m_msh,m_cl)<<std::endl;
/*
                auto cl_new = this->cl ;
                std::cout<<"-----> test_case_eshelby_2_prova : CELL = "<<offset(msh,cl_new)<<std::endl;

                auto cl_new2 = upload_cl();
                std::cout<<"-----> test_case_eshelby_2_prova : CELL = "<<offset(msh,cl_new2)<<std::endl;

                auto cl_new3 = upload_cl2();
                std::cout<<"-----> test_case_eshelby_2_prova : CELL = "<<offset(msh,cl_new3)<<std::endl;
                */
                               level_set__.cell_assignment(m_cl);
//T H = level_set__.normal(pt)

                               ret(0) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                               ret(1) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);


//ret(0) = - k / R * level_set__.normal(pt)(0);
//ret(1) = - k / R * level_set__.normal(pt)(1);

                           } else {
                               T gamma = 1.0;
                               level_set__.cell_assignment(m_cl);
                               ret(0) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                               ret(1) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);
                           }
                           return ret;
                       }) {
    }

    test_case_eshelby_2_prova(const test_case_eshelby_2_prova &other) : test_case_stokes<T, Function, Mesh>(other) {
        m_msh = other.m_msh;
        m_cl = other.m_cl;
    }

//    void test_case_cell_assignment(const Mesh& msh , const typename Mesh::cell_type& cl_new )
//    {
//        std::cout<<"sono qua 0"<<std::endl;
//        std::cout<<"-----> test_case_cell_assignment : CELL OLD = "<<offset(msh,cl)<<std::endl;
//        std::cout<<"sono qua 1"<<std::endl;
//        std::cout<<"-----> test_case_cell_assignment : CELL NEW = "<<offset(msh,cl_new)<<std::endl;
//        cl = cl_new ;
//        cl_pt = & cl;
//        //msh = msh ;
//        std::cout<<"----------------> test_case_cell_assignment : CELL NEW = "<<offset(msh,cl_new)<< " and CELL UPLOADED = "<<offset(msh,cl)<<std::endl;
//        std::cout<<"----------------> test_case_cell_assignment : CELL NEW PUNTATORE = "<<offset(msh,*cl_pt)<<std::endl;
//        i+=8;
//        //cl_pointer = std::make_shared<typename Mesh::cell_type>(cl_new);
//        //std::cout<<"----------------> test_case_cell_assignment : CELL POINTER = "<<offset(msh, cl_pointer.get())<< " and CELL UPLOADED = "<<offset(msh,cl)<<std::endl;
//        //level_set__.cell_assignment(cl_new);
//    }

    void test_case_cell_assignment(const typename Mesh::cell_type &cl_new) {
        m_cl = cl_new;
    }

    void refresh_lambdas(Function &level_set__, params<T> parms_, bool sym_grad) {

        this->neumann_jump = [level_set__, sym_grad, this](
                const typename Mesh::point_type &pt) mutable -> Eigen::Matrix<T, 2, 1> {/* Neu */
            Matrix<T, 2, 1> ret;
            if (sym_grad) {
                T gamma = 1.0;
//T k = 1.0;
//T R = 1.0/3.0;
//cl = cl_pointer.get() ;
//std::cout<<"SONO  IN NEUMANN CONDITION."<<std::endl;
//                std::cout<<"i = "<<i<<std::endl;
//auto cl_uploaded = this->cl ;
//                i++;
//                std::cout<<"i = "<<i<<std::endl;
//                Mesh msh_prova = *msh_pt ;
//                typename Mesh::cell_type cl_prova = *cl_pt ;


//                std::cout<<"-----> test_case_eshelby_2_prova : CELL PROVA = "<<offset(msh_prova,cl_prova)<<std::endl;
//if(i==3)
//   this->cl = msh.cells[227];
//std::cout<<"-----> test_case_eshelby_2_prova : CELL = "<<offset(m_msh,m_cl)<<std::endl;
/*
                        auto cl_new = this->cl ;
                        std::cout<<"-----> test_case_eshelby_2_prova : CELL = "<<offset(msh,cl_new)<<std::endl;

                        auto cl_new2 = upload_cl();
                        std::cout<<"-----> test_case_eshelby_2_prova : CELL = "<<offset(msh,cl_new2)<<std::endl;

                        auto cl_new3 = upload_cl2();
                        std::cout<<"-----> test_case_eshelby_2_prova : CELL = "<<offset(msh,cl_new3)<<std::endl;
                        */
                level_set__.cell_assignment(m_cl);
//T H = level_set__.normal(pt)

                ret(0) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                ret(1) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);


//ret(0) = - k / R * level_set__.normal(pt)(0);
//ret(1) = - k / R * level_set__.normal(pt)(1);

            } else {
                T gamma = 1.0;
                level_set__.cell_assignment(m_cl);
                ret(0) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                ret(1) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);
            }
            return ret;
        };

    }

    void test_case_mesh_assignment(const Mesh &msh_new) {

//std::cout<<"-----> test_case_mesh_assignment "<<std::endl;
        m_msh = msh_new;
//        msh_pt = & msh ;

    }

    typename Mesh::cell_type &upload_cl() {
        return m_cl;
    }
/*
    typename Mesh::cell_type upload_cl2()
    {
        return m_cl ;
    }
    */


};

template<typename Mesh, typename T, typename Function>
auto make_test_case_eshelby_2_prova(const Mesh &msh, Function &level_set_function, params<T> parms_, bool sym_grad) {
    return test_case_eshelby_2_prova<typename Mesh::coordinate_type, Mesh, Function>(level_set_function, parms_,
                                                                                     sym_grad);
}


template<typename T, typename Mesh, typename Function>
class test_case_eshelby_analytic :
        public test_case_stokes<T, Function, Mesh> {
public:
    test_case_eshelby_analytic(const Function &level_set__, params<T> parms_, bool sym_grad, T radius)
            : test_case_stokes<T, Function, Mesh>
                      (level_set__, parms_,
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {
// sol_vel
                           Matrix<T, 2, 1> ret;

                           ret(0) = 0.0;
                           ret(1) = 0.0;

                           return ret;
                       },
                       [level_set__](const typename Mesh::point_type &pt) -> T { // p

                           return 0.0;

                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                           Matrix<T, 2, 1> ret;

                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> auto { // grad

                           Matrix<T, 2, 2> ret;
                           ret(0, 0) = 0.0;
                           ret(0, 1) = 0.0;
                           ret(1, 0) = 0.0;
                           ret(1, 1) = (0.0);
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [level_set__, parms_, sym_grad, radius](
                               const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Neu */
                           Matrix<T, 2, 1> ret;
                           if (sym_grad) {
                               T gamma = 1.0;
//T k = 1.0;
//T R = 1.0/3.0;

//T H = level_set__.normal(pt)
//ret(0) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
//ret(1) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);

                               ret(0) = gamma * radius * level_set__.normal(pt)(0);

                               ret(1) = gamma * radius * level_set__.normal(pt)(1);

//ret(0) = - k / R * level_set__.normal(pt)(0);
//ret(1) = - k / R * level_set__.normal(pt)(1);

                           } else {
                               T gamma = 1.0;

                               ret(0) = gamma * radius * level_set__.normal(pt)(0);
                               ret(1) = gamma * radius * level_set__.normal(pt)(1);
                           }
                           return ret;
                       }) {
    }
};


template<typename Mesh, typename T, typename Function>
auto
make_test_case_eshelby_analytic(const Mesh &msh, const Function &level_set_function, params<T> parms_, bool sym_grad,
                                T radius) {
    return test_case_eshelby_analytic<typename Mesh::coordinate_type, Mesh, Function>(level_set_function, parms_,
                                                                                      sym_grad, radius);
}




// --> TEST CASE (ESHELBY PB WITH mu_1 = mu_2): it should be correct! W.r.t. test_case_eshelby_2_prova changes the pressure solution (now it's correct, before no)

template<typename T, typename Mesh, typename Function>
class test_case_eshelby_correct :
        public test_case_stokes<T, Function, Mesh> {

public:

    Mesh m_msh;
    typename Mesh::cell_type m_cl;
    T gamma = 1.0;

    explicit test_case_eshelby_correct(Function &level_set__, params<T> parms_, bool sym_grad, T gamma)
            : gamma(gamma), test_case_stokes<T, Function, Mesh>
            (level_set__, parms_,
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {
// sol_vel
                 Matrix<T, 2, 1> ret;

                 ret(0) = 0.0;
                 ret(1) = 0.0;

                 return ret;
             },
             [level_set__, gamma, this](const typename Mesh::point_type &pt) mutable -> T { // p

//T gamma = 0.05 ; // 1.0 ;
//T gamma = this->gamma;
                 level_set__.cell_assignment(m_cl);
                 T R = level_set__.radius;
//std::cout<<"pressure, cl= "<<offset(m_msh,m_cl) <<std::endl;
//std::cout<<"The radius = "<<R<<" , the divergence = "<<level_set__.divergence(pt)<<std::endl;


                 if (level_set__(pt) < 0)
                     return gamma / R - M_PI * R * gamma;
                 else
                     return -M_PI * R * gamma;


             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                 Matrix<T, 2, 1> ret;
                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                 Matrix<T, 2, 1> ret;

                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [](const typename Mesh::point_type &pt) -> auto { // grad

                 Matrix<T, 2, 2> ret;
                 ret(0, 0) = 0.0;
                 ret(0, 1) = 0.0;
                 ret(1, 0) = 0.0;
                 ret(1, 1) = (0.0);
                 return ret;
             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                 Matrix<T, 2, 1> ret;
                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [level_set__, sym_grad, gamma, this](
                     const typename Mesh::point_type &pt) mutable -> Eigen::Matrix<T, 2, 1> {/* Neu */
                 Matrix<T, 2, 1> ret;
//T gamma = this->gamma;
                 if (sym_grad) {
//T gamma = 0.05 ; // 1.0 ;

                     level_set__.cell_assignment(m_cl);
//T H = level_set__.normal(pt)
//std::cout<<"neumann cond, cl= "<<offset(m_msh,m_cl) <<std::endl;

// NEW NEUMANN COND WITHOUT 2*
                     ret(0) = gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                     ret(1) = gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);

// TENTATIVO PER CAPIRE DOVE NASCE L'ERRORE
//                T R = level_set__.radius ;
//                ret(0) = - gamma / R * level_set__.normal(pt)(0);
//                ret(1) = - gamma / R * level_set__.normal(pt)(1);

                 } else {
//T gamma = 0.05 ; // 1.0 ;
                     level_set__.cell_assignment(m_cl);
//std::cout<<"neumann cond, cl= "<<offset(m_msh,m_cl) ;


// NEW NEUMANN COND WITHOUT 2* (DA  RIMETTERE)
                     ret(0) = gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                     ret(1) = gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);

// TENTATIVO PER CAPIRE DOVE NASCE L'ERRORE
//                T R = level_set__.radius ;
//                ret(0) = - gamma / R * level_set__.normal(pt)(0);
//                ret(1) = - gamma / R * level_set__.normal(pt)(1);


                 }
                 return ret;
             }) {
    }

    test_case_eshelby_correct(const test_case_eshelby_correct &other) : test_case_stokes<T, Function, Mesh>(other) {
        m_msh = other.m_msh;
        m_cl = other.m_cl;
        gamma = other.gamma;
    }


    void test_case_cell_assignment(const typename Mesh::cell_type &cl_new) {
        m_cl = cl_new;
    }

/*
    void refresh_lambdas(Function & level_set__, params<T> parms_, bool sym_grad){

       this->neumann_jump = [level_set__,sym_grad,this](const typename Mesh::point_type& pt) mutable -> Eigen::Matrix<T, 2, 1> {// Neu //

           //T gamma = this->gamma;
           Matrix<T, 2, 1> ret;
           if(sym_grad)
           {
               //T gamma = 0.05 ; // 1.0 ;


               level_set__.cell_assignment(m_cl);
               //T H = level_set__.normal(pt)
               //std::cout<<"gamma = "<<gamma<<std::endl;
               ret(0) = gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
               ret(1) = gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);

               // TENTATIVO PER CAPIRE DOVE NASCE L'ERRORE

               //T R = level_set__.radius ;
               //ret(0) = - gamma / R * level_set__.normal(pt)(0);
               //ret(1) = - gamma / R * level_set__.normal(pt)(1);
               //T k_curv = level_set__.divergence(pt) ;
               //T curvature_err = std::abs(std::abs(k_curv) -1.0/R ) ;

               //std::cout<<"R = "<<R<<" , level_set__.divergence(pt) = "<<k_curv<<std::endl;
               //std::cout<<"interface_point = "<<pt<<" , curvature = "<<k_curv<<" , 1/R = "<<-1.0/R<<" , diff = "<< curvature_err <<std::endl;
            }
            else
            {
                //T gamma = 0.05 ; // 1.0 ;
                level_set__.cell_assignment(m_cl);
                ret(0) =  gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                ret(1) =  gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);
                // TENTATIVO PER CAPIRE DOVE NASCE L'ERRORE
                //T R = level_set__.radius ;
                //ret(0) = - gamma / R * level_set__.normal(pt)(0);
                //ret(1) = - gamma / R * level_set__.normal(pt)(1);
            }

           return ret;
       };

        this->sol_p = [level_set__,this](const typename Mesh::point_type& pt) mutable -> T {// Pressure //

            //T gamma = this->gamma;
            //T gamma = 0.05 ; // 1.0 ;
            level_set__.cell_assignment(m_cl);
            T R = level_set__.radius ;
            //std::cout<<"gamma = "<<gamma<<std::endl;

            //std::cout<<"The radius = "<<R<<" , the divergence = "<<level_set__.divergence(pt)<<std::endl;

            if( level_set__(pt) < 0 )
                return gamma / R - M_PI * R * gamma;
            else
                return -M_PI * R * gamma;


        };

    }
    */
    void test_case_mesh_assignment(const Mesh &msh_new) {

//std::cout<<"-----> test_case_mesh_assignment "<<std::endl;
        m_msh = msh_new;
//        msh_pt = & msh ;

    }

/*
    void test_case_gamma_setting(T gamma_n )
    {

        std::cout<<"-----> setting of gamma "<<std::endl;
        gamma = gamma_n ;

    }
    */
    typename Mesh::cell_type &upload_cl() {
        return m_cl;
    }
/*
    typename Mesh::cell_type upload_cl2()
    {
        return m_cl ;
    }
    */


};

template<typename Mesh, typename T, typename Function>
auto make_test_case_eshelby_correct(const Mesh &msh, Function &level_set_function, params<T> parms_, bool sym_grad,
                                    T gamma) {
    return test_case_eshelby_correct<typename Mesh::coordinate_type, Mesh, Function>(level_set_function, parms_,
                                                                                     sym_grad, gamma);
}


// --> TEST CASE (ESHELBY PB WITH mu_1 = mu_2): it should be correct! W.r.t. test_case_eshelby_2_prova changes the pressure solution (now it's correct, before no)

template<typename T, typename Mesh, typename Function>
class test_case_eshelby_correct_parametric :
        public test_case_stokes_ref_pts<T, Function, Mesh> {

public:

    Mesh m_msh;
    typename Mesh::cell_type m_cl;
    T gamma = 1.0;

    explicit test_case_eshelby_correct_parametric(Function &level_set__, params<T> parms_, bool sym_grad, T gamma)
            : gamma(gamma), test_case_stokes_ref_pts<T, Function, Mesh>
            (level_set__, parms_,
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {
// sol_vel
                 Matrix<T, 2, 1> ret;

                 ret(0) = 0.0;
                 ret(1) = 0.0;

                 return ret;
             },
             [level_set__, gamma, this](const typename Mesh::point_type &pt) mutable -> T { // p

//T gamma = 0.05 ; // 1.0 ;
//T gamma = this->gamma;
                 level_set__.cell_assignment(m_cl);
                 T R = level_set__.radius;
//std::cout<<"pressure, cl= "<<offset(m_msh,m_cl) <<std::endl;
//std::cout<<"The radius = "<<R<<" , the divergence = "<<level_set__.divergence(pt)<<std::endl;


                 if (level_set__(pt) < 0)
                     return gamma / R - M_PI * R * gamma;
                 else
                     return -M_PI * R * gamma;


             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                 Matrix<T, 2, 1> ret;
                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                 Matrix<T, 2, 1> ret;

                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [](const typename Mesh::point_type &pt) -> auto { // grad

                 Matrix<T, 2, 2> ret;
                 ret(0, 0) = 0.0;
                 ret(0, 1) = 0.0;
                 ret(1, 0) = 0.0;
                 ret(1, 1) = (0.0);
                 return ret;
             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                 Matrix<T, 2, 1> ret;
                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [level_set__, sym_grad, gamma, this](const T &pt,
                                                  const std::vector<typename Mesh::point_type> &pts) mutable -> Eigen::Matrix<T, 2, 1> {/* Neu */
//           , const std::vector<typename Mesh::point_type>& pts
//           auto pts =points(m_cl.user_data.integration_msh,m_cl.user_data.integration_msh.cells[0]);
                 Matrix<T, 2, 1> ret;
//T gamma = this->gamma;
                 if (sym_grad) {
                     level_set__.cell_assignment(m_cl);

                     size_t degree = m_cl.user_data.integration_msh.degree_curve;
                     Interface_parametrisation_mesh1d curve(degree);

                     ret(0) = gamma * curve.curvature(pt, pts, degree) * curve.normal(pt, pts, degree)(0);
                     ret(1) = gamma * curve.curvature(pt, pts, degree) * curve.normal(pt, pts, degree)(1);


                 } else {
                     level_set__.cell_assignment(m_cl);

                     size_t degree = m_cl.user_data.integration_msh.degree_curve;
                     Interface_parametrisation_mesh1d curve(degree);
                     ret(0) = gamma * curve.curvature(pt, pts, degree) * curve.normal(pt, pts, degree)(0);
                     ret(1) = gamma * curve.curvature(pt, pts, degree) * curve.normal(pt, pts, degree)(1);


                 }
                 return ret;
             }) {
    }

    test_case_eshelby_correct_parametric(const test_case_eshelby_correct_parametric &other)
            : test_case_stokes_ref_pts<T, Function, Mesh>(other) {
        m_msh = other.m_msh;
        m_cl = other.m_cl;
        gamma = other.gamma;
    }


    void test_case_cell_assignment(const typename Mesh::cell_type &cl_new) {
        m_cl = cl_new;
    }

    void test_case_mesh_assignment(const Mesh &msh_new) {

//std::cout<<"-----> test_case_mesh_assignment "<<std::endl;
        m_msh = msh_new;
//        msh_pt = & msh ;

    }

    typename Mesh::cell_type &upload_cl() {
        return m_cl;
    }


};

template<typename Mesh, typename T, typename Function>
auto make_test_case_eshelby_correct_parametric(const Mesh &msh, Function &level_set_function, params<T> parms_,
                                               bool sym_grad, T gamma) {
    return test_case_eshelby_correct_parametric<typename Mesh::coordinate_type, Mesh, Function>(level_set_function,
                                                                                                parms_, sym_grad,
                                                                                                gamma);
}


// --> TEST CASE (ESHELBY PB WITH mu_1 = mu_2): it should be correct! W.r.t. test_case_eshelby_2_prova changes the pressure solution (now it's correct, before no)

template<typename T, typename Mesh, typename Function, typename Para_Interface>
class test_case_eshelby_correct_parametric_cont :
        public test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface> {

public:

    Mesh m_msh;
    typename Mesh::cell_type m_cl;
    T gamma = 1.0;

    explicit test_case_eshelby_correct_parametric_cont(Function &level_set__, Para_Interface &parametric_curve_cont,
                                                       params<T> parms_, bool sym_grad, T gamma)
            : gamma(gamma), test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface>
            (level_set__, parametric_curve_cont, parms_,
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {
// sol_vel
                 Matrix<T, 2, 1> ret;

                 ret(0) = 0.0;
                 ret(1) = 0.0;

                 return ret;
             },
             [level_set__, gamma, this](const typename Mesh::point_type &pt) mutable -> T { // p

                 level_set__.cell_assignment(m_cl);
                 T R = level_set__.radius;

                 if (level_set__(pt) < 0)
                     return gamma / R - M_PI * R * gamma;
                 else
                     return -M_PI * R * gamma;


             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                 Matrix<T, 2, 1> ret;
                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                 Matrix<T, 2, 1> ret;

                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [](const typename Mesh::point_type &pt) -> auto { // grad

                 Matrix<T, 2, 2> ret;
                 ret(0, 0) = 0.0;
                 ret(0, 1) = 0.0;
                 ret(1, 0) = 0.0;
                 ret(1, 1) = (0.0);
                 return ret;
             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                 Matrix<T, 2, 1> ret;
                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [level_set__, parametric_curve_cont, sym_grad, gamma, this](const T &pt,
                                                                         const size_t &global_cl_i) mutable -> Eigen::Matrix<T, 2, 1> {/* Neu */

                 Matrix<T, 2, 1> ret;

                 if (sym_grad) {

                     ret(0) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(0);
                     ret(1) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(1);


                 } else {
                     ret(0) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(0);
                     ret(1) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(1);


                 }
                 return ret;
             }) {
    }

    test_case_eshelby_correct_parametric_cont(const test_case_eshelby_correct_parametric_cont &other)
            : test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface>(other) {
        m_msh = other.m_msh;
        m_cl = other.m_cl;
        gamma = other.gamma;
    }


    void test_case_cell_assignment(const typename Mesh::cell_type &cl_new) {
        m_cl = cl_new;
    }

    void test_case_mesh_assignment(const Mesh &msh_new) {

        m_msh = msh_new;

    }

    typename Mesh::cell_type &upload_cl() {
        return m_cl;
    }


};

template<typename Mesh, typename T, typename Function, typename Para_Interface>
auto make_test_case_eshelby_correct_parametric_cont(const Mesh &msh, Function &level_set_function,
                                                    Para_Interface &parametric_curve_cont, params<T> parms_,
                                                    bool sym_grad, T gamma) {
    return test_case_eshelby_correct_parametric_cont<typename Mesh::coordinate_type, Mesh, Function, Para_Interface>(
            level_set_function, parametric_curve_cont, parms_, sym_grad, gamma);
}


template<typename T, typename Mesh, typename Function, typename Para_Interface>
class test_case_eshelby_correct_parametric_cont_TGV_source :
        public test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface> {

public:

    Mesh m_msh;
    typename Mesh::cell_type m_cl;
    T gamma = 1.0;

    explicit test_case_eshelby_correct_parametric_cont_TGV_source(Function &level_set__,
                                                                  Para_Interface &parametric_curve_cont,
                                                                  params<T> parms_, bool sym_grad, T gamma)
            : gamma(gamma), test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface>
            (level_set__, parametric_curve_cont, parms_,
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {
// sol_vel
                 Matrix<T, 2, 1> ret;

                 ret(0) = std::sin(2 * M_PI * pt.x()) * std::cos(2 * M_PI * pt.y());
                 ret(1) = -std::cos(2 * M_PI * pt.x()) * std::sin(2 * M_PI * pt.y());

                 return ret;
             },
             [level_set__, gamma, this](const typename Mesh::point_type &pt) mutable -> T { // p

                 level_set__.cell_assignment(m_cl);
                 T R = level_set__.radius;

                 if (level_set__(pt) < 0)
                     return gamma / R - M_PI * R * gamma;
                 else
                     return -M_PI * R * gamma;


             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                 Matrix<T, 2, 1> ret;
                 ret(0) = std::sin(2 * M_PI * pt.x()) * std::cos(2 * M_PI * pt.y());
                 ret(1) = -std::cos(2 * M_PI * pt.x()) * std::sin(2 * M_PI * pt.y());
                 return ret;
             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                 Matrix<T, 2, 1> ret;

                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [](const typename Mesh::point_type &pt) -> auto { // grad

                 Matrix<T, 2, 2> ret;
                 ret(0, 0) = 2 * M_PI * std::cos(2 * M_PI * pt.x()) * std::cos(2 * M_PI * pt.y());
                 ret(0, 1) = -2 * M_PI * std::sin(2 * M_PI * pt.x()) * std::sin(2 * M_PI * pt.y());
                 ret(1, 0) = 2 * M_PI * std::sin(2 * M_PI * pt.x()) * std::sin(2 * M_PI * pt.y());
                 ret(1, 1) = -2 * M_PI * std::cos(2 * M_PI * pt.x()) * std::cos(2 * M_PI * pt.y());
                 return ret;
             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                 Matrix<T, 2, 1> ret;
                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [level_set__, parametric_curve_cont, sym_grad, gamma, this](const T &pt,
                                                                         const size_t &global_cl_i) mutable -> Eigen::Matrix<T, 2, 1> {/* Neu */

                 Matrix<T, 2, 1> ret;

                 if (sym_grad) {

                     ret(0) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(0);
                     ret(1) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(1);


                 } else {
                     ret(0) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(0);
                     ret(1) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(1);


                 }
                 return ret;
             }) {
    }

    test_case_eshelby_correct_parametric_cont_TGV_source(
            const test_case_eshelby_correct_parametric_cont_TGV_source &other)
            : test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface>(other) {
        m_msh = other.m_msh;
        m_cl = other.m_cl;
        gamma = other.gamma;
    }


    void test_case_cell_assignment(const typename Mesh::cell_type &cl_new) {
        m_cl = cl_new;
    }

    void test_case_mesh_assignment(const Mesh &msh_new) {

        m_msh = msh_new;

    }

    typename Mesh::cell_type &upload_cl() {
        return m_cl;
    }


};

template<typename Mesh, typename T, typename Function, typename Para_Interface>
auto make_test_case_eshelby_correct_parametric_cont_TGV_source(const Mesh &msh, Function &level_set_function,
                                                               Para_Interface &parametric_curve_cont, params<T> parms_,
                                                               bool sym_grad, T gamma) {
    return test_case_eshelby_correct_parametric_cont_TGV_source<typename Mesh::coordinate_type, Mesh, Function, Para_Interface>(
            level_set_function, parametric_curve_cont, parms_, sym_grad, gamma);
}


template<typename T, typename Mesh, typename Function, typename Para_Interface>
class test_case_eshelby_correct_parametric_cont_DIRICHLET_eps :
        public test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface> {

public:

    Mesh m_msh;
    typename Mesh::cell_type m_cl;
    T gamma = 1.0;
    T eps = 1.0;

    explicit test_case_eshelby_correct_parametric_cont_DIRICHLET_eps(Function &level_set__,
                                                                     Para_Interface &parametric_curve_cont,
                                                                     params<T> parms_, bool sym_grad, T gamma, T eps)
            : gamma(gamma), eps(eps), test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface>
            (level_set__, parametric_curve_cont, parms_,
             [eps](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {
// sol_vel
                 Matrix<T, 2, 1> ret;
                 T px = pt.x();
                 T py = pt.y();
                 ret(0) = eps * (2.0 * px - 1.0);
                 ret(1) = eps * (-2.0 * py + 1.0);

                 return ret;
             },
             [level_set__, gamma, this](const typename Mesh::point_type &pt) mutable -> T { // p

                 level_set__.cell_assignment(m_cl);
                 T R = level_set__.radius;

                 if (level_set__(pt) < 0)
                     return gamma / R - M_PI * R * gamma;
                 else
                     return -M_PI * R * gamma;


             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                 Matrix<T, 2, 1> ret;
                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [eps](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                 Matrix<T, 2, 1> ret;

                 T px = pt.x();
                 T py = pt.y();
                 ret(0) = eps * (2.0 * px - 1.0);
                 ret(1) = eps * (-2.0 * py + 1.0);
                 return ret;
             },
             [eps](const typename Mesh::point_type &pt) -> auto { // grad

                 Matrix<T, 2, 2> ret;
//           T px = pt.x();
//           T py = pt.y();

                 ret(0, 0) = eps * 2.0;
                 ret(0, 1) = 0.0;
                 ret(1, 0) = 0.0;
                 ret(1, 1) = -eps * 2.0;
                 return ret;
             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                 Matrix<T, 2, 1> ret;
                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [level_set__, parametric_curve_cont, sym_grad, gamma, this](const T &pt,
                                                                         const size_t &global_cl_i) mutable -> Eigen::Matrix<T, 2, 1> {/* Neu */

                 Matrix<T, 2, 1> ret;

                 if (sym_grad) {

                     ret(0) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(0);
                     ret(1) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(1);


                 } else {
                     ret(0) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(0);
                     ret(1) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(1);


                 }
                 return ret;
             }) {
    }

    test_case_eshelby_correct_parametric_cont_DIRICHLET_eps(
            const test_case_eshelby_correct_parametric_cont_DIRICHLET_eps &other)
            : test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface>(other) {
        m_msh = other.m_msh;
        m_cl = other.m_cl;
        gamma = other.gamma;
        eps = other.eps;
    }


    void test_case_cell_assignment(const typename Mesh::cell_type &cl_new) {
        m_cl = cl_new;
    }

    void test_case_mesh_assignment(const Mesh &msh_new) {

        m_msh = msh_new;

    }

    typename Mesh::cell_type &upload_cl() {
        return m_cl;
    }


};

template<typename Mesh, typename T, typename Function, typename Para_Interface>
auto make_test_case_eshelby_correct_parametric_cont_DIRICHLET_eps(const Mesh &msh, Function &level_set_function,
                                                                  Para_Interface &parametric_curve_cont,
                                                                  params<T> parms_, bool sym_grad, T gamma, T eps) {
    return test_case_eshelby_correct_parametric_cont_DIRICHLET_eps<typename Mesh::coordinate_type, Mesh, Function, Para_Interface>(
            level_set_function, parametric_curve_cont, parms_, sym_grad, gamma, eps);
}


// shear flow - curve evaluation domain (-a,a)^2
template<typename T, typename Mesh, typename Function, typename Para_Interface>
class test_case_eshelby_parametric_cont_eps_DIR_domSym :
        public test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface> {

public:

    Mesh m_msh;
    typename Mesh::cell_type m_cl;
    T gamma = 1.0;
    T eps = 1.0;

    explicit test_case_eshelby_parametric_cont_eps_DIR_domSym(Function &level_set__,
                                                              Para_Interface &parametric_curve_cont, params<T> parms_,
                                                              bool sym_grad, T gamma, T eps)
            : gamma(gamma), eps(eps), test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface>
            (level_set__, parametric_curve_cont, parms_,
             [eps](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {
// sol_vel
                 Matrix<T, 2, 1> ret;

                 ret(0) = eps * pt.x();
                 ret(1) = -eps * pt.y();

                 return ret;
             },
             [level_set__, gamma, this](const typename Mesh::point_type &pt) mutable -> T { // p

                 level_set__.cell_assignment(m_cl);
                 T R = level_set__.radius;

                 if (level_set__(pt) < 0)
                     return gamma / R - M_PI * R * gamma;
                 else
                     return -M_PI * R * gamma;


             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                 Matrix<T, 2, 1> ret;
                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [eps](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                 Matrix<T, 2, 1> ret;

                 ret(0) = eps * pt.x();
                 ret(1) = -eps * pt.y();

                 return ret;
             },
             [eps](const typename Mesh::point_type &pt) -> auto { // grad

                 Matrix<T, 2, 2> ret;

                 ret(0, 0) = eps;
                 ret(0, 1) = 0.0;
                 ret(1, 0) = 0.0;
                 ret(1, 1) = -eps;
                 return ret;
             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                 Matrix<T, 2, 1> ret;
                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [level_set__, parametric_curve_cont, sym_grad, gamma, this](const T &pt,
                                                                         const size_t &global_cl_i) mutable -> Eigen::Matrix<T, 2, 1> {/* Neu */

                 Matrix<T, 2, 1> ret;

                 if (sym_grad) {

                     ret(0) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(0);
                     ret(1) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(1);


                 } else {
                     ret(0) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(0);
                     ret(1) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(1);


                 }
                 return ret;
             }) {
    }

    test_case_eshelby_parametric_cont_eps_DIR_domSym(const test_case_eshelby_parametric_cont_eps_DIR_domSym &other)
            : test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface>(other) {
        m_msh = other.m_msh;
        m_cl = other.m_cl;
        gamma = other.gamma;
        eps = other.eps;
    }


    void test_case_cell_assignment(const typename Mesh::cell_type &cl_new) {
        m_cl = cl_new;
    }

    void test_case_mesh_assignment(const Mesh &msh_new) {

        m_msh = msh_new;

    }

    typename Mesh::cell_type &upload_cl() {
        return m_cl;
    }


};

template<typename Mesh, typename T, typename Function, typename Para_Interface>
auto make_test_case_eshelby_parametric_cont_eps_DIR_domSym(const Mesh &msh, Function &level_set_function,
                                                           Para_Interface &parametric_curve_cont, params<T> parms_,
                                                           bool sym_grad, T gamma, T eps) {
    return test_case_eshelby_parametric_cont_eps_DIR_domSym<typename Mesh::coordinate_type, Mesh, Function, Para_Interface>(
            level_set_function, parametric_curve_cont, parms_, sym_grad, gamma, eps);
}


// shear flow - curve evaluation domain (-a,a)^2
template<typename T, typename Mesh, typename Function, typename Para_Interface>
class test_case_eshelby_parametric_cont_eps_perturbated_DIR_domSym :
        public test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface> {

public:

    Mesh m_msh;
    typename Mesh::cell_type m_cl;
    T gamma = 1.0;
    T eps = 1.0;
    T perturbation = 0.1;

    explicit test_case_eshelby_parametric_cont_eps_perturbated_DIR_domSym(Function &level_set__,
                                                                          Para_Interface &parametric_curve_cont,
                                                                          params<T> parms_, bool sym_grad, T gamma,
                                                                          T eps, T perturbation)
            : gamma(gamma), eps(eps), perturbation(perturbation),
              test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface>
                      (level_set__, parametric_curve_cont, parms_,
                       [eps, perturbation](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {
// sol_vel
                           Matrix<T, 2, 1> ret;
                           T px = pt.x();
                           T py = pt.y();
                           ret(0) = eps * (px + perturbation * std::sin(M_PI * py));
                           ret(1) = eps * (-py + perturbation * std::sin(M_PI * px));

                           return ret;
                       },
                       [level_set__, gamma, this](const typename Mesh::point_type &pt) mutable -> T { // p

                           level_set__.cell_assignment(m_cl);
                           T R = level_set__.radius;

                           if (level_set__(pt) < 0)
                               return gamma / R - M_PI * R * gamma;
                           else
                               return -M_PI * R * gamma;


                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [eps, perturbation](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                           Matrix<T, 2, 1> ret;

                           T px = pt.x();
                           T py = pt.y();
                           ret(0) = eps * (px + perturbation * std::sin(M_PI * py));
                           ret(1) = eps * (-py + perturbation * std::sin(M_PI * px));

                           return ret;
                       },
                       [eps, perturbation](const typename Mesh::point_type &pt) -> auto { // grad

                           Matrix<T, 2, 2> ret;

                           ret(0, 0) = eps;
                           ret(0, 1) = eps * M_PI * perturbation * std::cos(M_PI * pt.y());
                           ret(1, 0) = eps * M_PI * perturbation * std::cos(M_PI * pt.x());
                           ret(1, 1) = -eps;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [level_set__, parametric_curve_cont, sym_grad, gamma, this](const T &pt,
                                                                                   const size_t &global_cl_i) mutable -> Eigen::Matrix<T, 2, 1> {/* Neu */

                           Matrix<T, 2, 1> ret;

                           if (sym_grad) {

                               ret(0) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                                        parametric_curve_cont.normal_cont(pt, global_cl_i)(0);
                               ret(1) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                                        parametric_curve_cont.normal_cont(pt, global_cl_i)(1);


                           } else {
                               ret(0) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                                        parametric_curve_cont.normal_cont(pt, global_cl_i)(0);
                               ret(1) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                                        parametric_curve_cont.normal_cont(pt, global_cl_i)(1);


                           }
                           return ret;
                       }) {
    }

    test_case_eshelby_parametric_cont_eps_perturbated_DIR_domSym(
            const test_case_eshelby_parametric_cont_eps_perturbated_DIR_domSym &other)
            : test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface>(other) {
        m_msh = other.m_msh;
        m_cl = other.m_cl;
        gamma = other.gamma;
        eps = other.eps;
        perturbation = other.perturbation;
    }


    void test_case_cell_assignment(const typename Mesh::cell_type &cl_new) {
        m_cl = cl_new;
    }

    void test_case_mesh_assignment(const Mesh &msh_new) {

        m_msh = msh_new;

    }

    typename Mesh::cell_type &upload_cl() {
        return m_cl;
    }


};

template<typename Mesh, typename T, typename Function, typename Para_Interface>
auto make_test_case_eshelby_parametric_cont_eps_perturbated_DIR_domSym(const Mesh &msh, Function &level_set_function,
                                                                       Para_Interface &parametric_curve_cont,
                                                                       params<T> parms_, bool sym_grad, T gamma, T eps,
                                                                       T perturbation) {
    return test_case_eshelby_parametric_cont_eps_perturbated_DIR_domSym<typename Mesh::coordinate_type, Mesh, Function, Para_Interface>(
            level_set_function, parametric_curve_cont, parms_, sym_grad, gamma, eps, perturbation);
}


// Perturbeted case
template<typename T, typename Mesh, typename Function, typename Para_Interface>
class test_case_shear_flow_perturbated :
        public test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface> {

public:

    Mesh m_msh;
    typename Mesh::cell_type m_cl;
    T gamma = 1.0;
    T eps = 1.0;
    T perturbation = 0.1;

    explicit test_case_shear_flow_perturbated(Function &level_set__, Para_Interface &parametric_curve_cont,
                                              params<T> parms_, bool sym_grad, T gamma, T eps, T perturbation)
            : gamma(gamma), eps(eps), perturbation(perturbation),
              test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface>
                      (level_set__, parametric_curve_cont, parms_,
                       [eps, perturbation](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {
// sol_vel
                           Matrix<T, 2, 1> ret;
                           T px = pt.x();
                           T py = pt.y();
                           ret(0) = eps * (2.0 * px - 1.0 + perturbation * std::sin(2.0 * M_PI * py));
                           ret(1) = eps * (-2.0 * py + 1.0 + perturbation * std::sin(2.0 * M_PI * px));

                           return ret;
                       },
                       [level_set__, gamma, this](const typename Mesh::point_type &pt) mutable -> T { // p

                           level_set__.cell_assignment(m_cl);
                           T R = level_set__.radius;

                           if (level_set__(pt) < 0)
                               return gamma / R - M_PI * R * gamma;
                           else
                               return -M_PI * R * gamma;


                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [eps, perturbation](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                           Matrix<T, 2, 1> ret;

                           T px = pt.x();
                           T py = pt.y();
                           ret(0) = eps * (2.0 * px - 1.0 + perturbation * std::sin(2.0 * M_PI * py));
                           ret(1) = eps * (-2.0 * py + 1.0 + perturbation * std::sin(2.0 * M_PI * px));
                           return ret;
                       },
                       [eps, perturbation](const typename Mesh::point_type &pt) -> auto { // grad

                           Matrix<T, 2, 2> ret;
                           T px = pt.x();
                           T py = pt.y();

                           ret(0, 0) = eps * 2.0;
                           ret(0, 1) = 2.0 * M_PI * perturbation * std::cos(2.0 * M_PI * py);
                           ret(1, 0) = -2.0 * M_PI * perturbation * std::cos(2.0 * M_PI * px);
                           ret(1, 1) = -eps * 2.0;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [level_set__, parametric_curve_cont, sym_grad, gamma, this](const T &pt,
                                                                                   const size_t &global_cl_i) mutable -> Eigen::Matrix<T, 2, 1> {/* Neu */

                           Matrix<T, 2, 1> ret;

                           if (sym_grad) {

                               ret(0) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                                        parametric_curve_cont.normal_cont(pt, global_cl_i)(0);
                               ret(1) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                                        parametric_curve_cont.normal_cont(pt, global_cl_i)(1);


                           } else {
                               ret(0) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                                        parametric_curve_cont.normal_cont(pt, global_cl_i)(0);
                               ret(1) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                                        parametric_curve_cont.normal_cont(pt, global_cl_i)(1);


                           }
                           return ret;
                       }) {
    }

    test_case_shear_flow_perturbated(const test_case_shear_flow_perturbated &other)
            : test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface>(other) {
        m_msh = other.m_msh;
        m_cl = other.m_cl;
        gamma = other.gamma;
        eps = other.eps;
        perturbation = other.perturbation;
    }


    void test_case_cell_assignment(const typename Mesh::cell_type &cl_new) {
        m_cl = cl_new;
    }

    void test_case_mesh_assignment(const Mesh &msh_new) {

        m_msh = msh_new;

    }

    typename Mesh::cell_type &upload_cl() {
        return m_cl;
    }


};

template<typename Mesh, typename T, typename Function, typename Para_Interface>
auto make_test_case_shear_flow_perturbated(const Mesh &msh, Function &level_set_function,
                                           Para_Interface &parametric_curve_cont, params<T> parms_, bool sym_grad,
                                           T gamma, T eps, T perturbation) {
    return test_case_shear_flow_perturbated<typename Mesh::coordinate_type, Mesh, Function, Para_Interface>(
            level_set_function, parametric_curve_cont, parms_, sym_grad, gamma, eps, perturbation);
}


// TGV case -> Stokes flow FP
template<typename T, typename Mesh, typename Function, typename Para_Interface>
class test_case_TGV_FPscheme :
        public test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface> {

public:

    Mesh m_msh;
    typename Mesh::cell_type m_cl;
    T gamma = 1.0;
    T eps = 1.0;

    explicit test_case_TGV_FPscheme(Function &level_set__, Para_Interface &parametric_curve_cont, params<T> parms_,
                                    bool sym_grad, T gamma, T eps)
            : gamma(gamma), eps(eps), test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface>
            (level_set__, parametric_curve_cont, parms_,
             [eps](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {
// sol_vel
                 Matrix<T, 2, 1> ret;
                 T px = pt.x();
                 T py = pt.y();
                 ret(0) = eps * (std::sin(2.0 * M_PI * px) * std::cos(2.0 * M_PI * py));
                 ret(1) = -eps * (std::sin(2.0 * M_PI * py) * std::cos(2.0 * M_PI * px));

                 return ret;
             },
             [level_set__, gamma, this](const typename Mesh::point_type &pt) mutable -> T { // p

                 level_set__.cell_assignment(m_cl);
                 T R = level_set__.radius;

                 if (level_set__(pt) < 0)
                     return gamma / R - M_PI * R * gamma;
                 else
                     return -M_PI * R * gamma;


             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                 Matrix<T, 2, 1> ret;
                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [eps](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                 Matrix<T, 2, 1> ret;

                 T px = pt.x();
                 T py = pt.y();
                 ret(0) = eps * (std::sin(2.0 * M_PI * px) * std::cos(2.0 * M_PI * py));
                 ret(1) = -eps * (std::sin(2.0 * M_PI * py) * std::cos(2.0 * M_PI * px));
                 return ret;
             },
             [eps](const typename Mesh::point_type &pt) -> auto { // grad

                 Matrix<T, 2, 2> ret;
                 T px = pt.x();
                 T py = pt.y();

                 ret(0, 0) = eps * 2.0 * M_PI * (std::cos(2.0 * M_PI * px) * std::cos(2.0 * M_PI * py));
                 ret(0, 1) = -eps * 2.0 * M_PI * (std::sin(2.0 * M_PI * px) * std::sin(2.0 * M_PI * py));
                 ret(1, 0) = eps * 2.0 * M_PI * (std::sin(2.0 * M_PI * px) * std::sin(2.0 * M_PI * py));
                 ret(1, 1) = -eps * 2.0 * M_PI * (std::cos(2.0 * M_PI * px) * std::cos(2.0 * M_PI * py));
                 return ret;
             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                 Matrix<T, 2, 1> ret;
                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [level_set__, parametric_curve_cont, sym_grad, gamma, this](const T &pt,
                                                                         const size_t &global_cl_i) mutable -> Eigen::Matrix<T, 2, 1> {/* Neu */

                 Matrix<T, 2, 1> ret;

                 if (sym_grad) {

                     ret(0) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(0);
                     ret(1) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(1);


                 } else {
                     ret(0) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(0);
                     ret(1) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                              parametric_curve_cont.normal_cont(pt, global_cl_i)(1);


                 }
                 return ret;
             }) {
    }

    test_case_TGV_FPscheme(const test_case_TGV_FPscheme &other)
            : test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface>(other) {
        m_msh = other.m_msh;
        m_cl = other.m_cl;
        gamma = other.gamma;
        eps = other.eps;
    }


    void test_case_cell_assignment(const typename Mesh::cell_type &cl_new) {
        m_cl = cl_new;
    }

    void test_case_mesh_assignment(const Mesh &msh_new) {

        m_msh = msh_new;

    }

    typename Mesh::cell_type &upload_cl() {
        return m_cl;
    }


};

template<typename Mesh, typename T, typename Function, typename Para_Interface>
auto make_test_case_TGV_FPscheme(const Mesh &msh, Function &level_set_function, Para_Interface &parametric_curve_cont,
                                 params<T> parms_, bool sym_grad, T gamma, T eps) {
    return test_case_TGV_FPscheme<typename Mesh::coordinate_type, Mesh, Function, Para_Interface>(level_set_function,
                                                                                                  parametric_curve_cont,
                                                                                                  parms_, sym_grad,
                                                                                                  gamma, eps);
}


// Perturbeted case
template<typename T, typename Mesh, typename Function, typename Para_Interface>
class test_case_shear_y :
        public test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface> {

public:

    Mesh m_msh;
    typename Mesh::cell_type m_cl;
    T gamma = 1.0;
    T eps = 1.0;
    T perturbation = 0.1;

    explicit test_case_shear_y(Function &level_set__, Para_Interface &parametric_curve_cont, params<T> parms_,
                               bool sym_grad, T gamma, T eps, T perturbation)
            : gamma(gamma), eps(eps), perturbation(perturbation),
              test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface>
                      (level_set__, parametric_curve_cont, parms_,
                       [eps, perturbation](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {
// sol_vel
                           Matrix<T, 2, 1> ret;
                           T px = pt.x();
                           T py = pt.y();
                           ret(0) = eps * (py - 0.5);
                           ret(1) = 0;

                           return ret;
                       },
                       [level_set__, gamma, this](const typename Mesh::point_type &pt) mutable -> T { // p

                           level_set__.cell_assignment(m_cl);
                           T R = level_set__.radius;

                           if (level_set__(pt) < 0)
                               return gamma / R - M_PI * R * gamma;
                           else
                               return -M_PI * R * gamma;


                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [eps, perturbation](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                           Matrix<T, 2, 1> ret;

                           T px = pt.x();
                           T py = pt.y();
                           ret(0) = eps * (py - 0.5);
                           ret(1) = 0;
                           return ret;
                       },
                       [eps, perturbation](const typename Mesh::point_type &pt) -> auto { // grad

                           Matrix<T, 2, 2> ret;
                           T px = pt.x();
                           T py = pt.y();

                           ret(0, 0) = 0.0;
                           ret(0, 1) = eps;
                           ret(1, 0) = 0.0;
                           ret(1, 1) = 0.0;
                           return ret;
                       },
                       [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                           Matrix<T, 2, 1> ret;
                           ret(0) = 0.0;
                           ret(1) = 0.0;
                           return ret;
                       },
                       [level_set__, parametric_curve_cont, sym_grad, gamma, this](const T &pt,
                                                                                   const size_t &global_cl_i) mutable -> Eigen::Matrix<T, 2, 1> {/* Neu */

                           Matrix<T, 2, 1> ret;

                           if (sym_grad) {

                               ret(0) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                                        parametric_curve_cont.normal_cont(pt, global_cl_i)(0);
                               ret(1) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                                        parametric_curve_cont.normal_cont(pt, global_cl_i)(1);


                           } else {
                               ret(0) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                                        parametric_curve_cont.normal_cont(pt, global_cl_i)(0);
                               ret(1) = gamma * parametric_curve_cont.curvature_cont(pt, global_cl_i) *
                                        parametric_curve_cont.normal_cont(pt, global_cl_i)(1);


                           }
                           return ret;
                       }) {
    }

    test_case_shear_y(const test_case_shear_y &other)
            : test_case_stokes_ref_pts_cont<T, Function, Mesh, Para_Interface>(other) {
        m_msh = other.m_msh;
        m_cl = other.m_cl;
        gamma = other.gamma;
        eps = other.eps;
        perturbation = other.perturbation;
    }


    void test_case_cell_assignment(const typename Mesh::cell_type &cl_new) {
        m_cl = cl_new;
    }

    void test_case_mesh_assignment(const Mesh &msh_new) {

        m_msh = msh_new;

    }

    typename Mesh::cell_type &upload_cl() {
        return m_cl;
    }


};

template<typename Mesh, typename T, typename Function, typename Para_Interface>
auto make_test_case_shear_y(const Mesh &msh, Function &level_set_function, Para_Interface &parametric_curve_cont,
                            params<T> parms_, bool sym_grad, T gamma, T eps, T perturbation) {
    return test_case_shear_y<typename Mesh::coordinate_type, Mesh, Function, Para_Interface>(level_set_function,
                                                                                             parametric_curve_cont,
                                                                                             parms_, sym_grad, gamma,
                                                                                             eps, perturbation);
}


template<typename T, typename Mesh, typename Function, typename Para_Interface>
class test_case_eshelby_LS_eps_DIR :
        public test_case_stokes_eps<T, Function, Mesh, Para_Interface> {

public:

    Mesh m_msh;
    typename Mesh::cell_type m_cl;
    T gamma = 1.0;
    T eps = 1.0;

    explicit test_case_eshelby_LS_eps_DIR(Function &level_set__, Para_Interface &curve, params<T> parms_,
                                          bool sym_grad, T gamma, T eps)
            : gamma(gamma), eps(eps), test_case_stokes_eps<T, Function, Mesh, Para_Interface>
            (level_set__, curve, parms_,
             [eps](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {
// sol_vel
                 Matrix<T, 2, 1> ret;

                 ret(0) = eps * (2.0 * pt.x() - 1.0);
                 ret(1) = eps * (-2.0 * pt.y() + 1.0);

                 return ret;
             },
             [level_set__, gamma, this](const typename Mesh::point_type &pt) mutable -> T { // p


                 level_set__.cell_assignment(m_cl);
                 T R = level_set__.radius;


                 if (level_set__(pt) < 0)
                     return gamma / R - M_PI * R * gamma;
                 else
                     return -M_PI * R * gamma;


             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                 Matrix<T, 2, 1> ret;
                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [eps](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                 Matrix<T, 2, 1> ret;

                 ret(0) = eps * (2.0 * pt.x() - 1.0);
                 ret(1) = eps * (-2.0 * pt.y() + 1.0);
                 return ret;
             },
             [eps](const typename Mesh::point_type &pt) -> auto { // grad

                 Matrix<T, 2, 2> ret;
                 ret(0, 0) = eps * 2.0;
                 ret(0, 1) = 0.0;
                 ret(1, 0) = 0.0;
                 ret(1, 1) = -eps * 2.0;
                 return ret;
             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                 Matrix<T, 2, 1> ret;
                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [level_set__, sym_grad, gamma, this](
                     const typename Mesh::point_type &pt) mutable -> Eigen::Matrix<T, 2, 1> {/* Neu */
                 Matrix<T, 2, 1> ret;

                 level_set__.cell_assignment(m_cl);


                 ret(0) = gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                 ret(1) = gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);

                 return ret;
             }) {
    }

    test_case_eshelby_LS_eps_DIR(const test_case_eshelby_LS_eps_DIR &other)
            : test_case_stokes_eps<T, Function, Mesh, Para_Interface>(other) {
        m_msh = other.m_msh;
        m_cl = other.m_cl;
        gamma = other.gamma;
        eps = other.eps;
    }


    test_case_eshelby_LS_eps_DIR(test_case_eshelby_LS_eps_DIR &other)
            : test_case_stokes_eps<T, Function, Mesh, Para_Interface>(other) {
        m_msh = other.m_msh;
        m_cl = other.m_cl;
        gamma = other.gamma;
        eps = other.eps;
    }

    void test_case_cell_assignment(const typename Mesh::cell_type &cl_new) {
        m_cl = cl_new;
    }

    void test_case_mesh_assignment(const Mesh &msh_new) {


        m_msh = msh_new;


    }

    typename Mesh::cell_type &upload_cl() {
        return m_cl;
    }


};

template<typename Mesh, typename T, typename Function, typename Para_Interface>
auto make_test_case_eshelby_LS_eps_DIR(const Mesh &msh, Function &level_set_function, Para_Interface &curve,
                                       params<T> parms_, bool sym_grad, T gamma, T eps) {
    return test_case_eshelby_LS_eps_DIR<T, Mesh, Function, Para_Interface>(level_set_function, curve, parms_, sym_grad,
                                                                           gamma, eps);
}


template<typename T, typename Mesh, typename Function, typename Para_Interface>
class test_case_eshelby_LS_eps_DIR_domSym :
        public test_case_stokes_eps<T, Function, Mesh, Para_Interface> {

public:

    Mesh m_msh;
    typename Mesh::cell_type m_cl;
    T gamma = 1.0;
    T eps = 1.0;
    T sizeBox = 1;

    explicit test_case_eshelby_LS_eps_DIR_domSym(Function &level_set__, Para_Interface &curve, params<T> parms_,
                                                 bool sym_grad, T gamma, T eps, T sizeBox)
            : gamma(gamma), eps(eps), test_case_stokes_eps<T, Function, Mesh, Para_Interface>
            (level_set__, curve, parms_,
             [eps](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {
// sol_vel
                 Matrix<T, 2, 1> ret;

                 ret(0) = eps * pt.x();
                 ret(1) = -eps * pt.y();

                 return ret;
             },
             [level_set__, gamma, this](const typename Mesh::point_type &pt) mutable -> T { // p


                 level_set__.cell_assignment(m_cl);
                 T R = level_set__.radius;


                 if (level_set__(pt) < 0)
                     return gamma / R - M_PI * R * gamma;
                 else
                     return -M_PI * R * gamma;


             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // rhs
                 Matrix<T, 2, 1> ret;
                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [eps](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> { // bcs
                 Matrix<T, 2, 1> ret;

                 ret(0) = eps * pt.x();
                 ret(1) = -eps * pt.y();
                 return ret;
             },
             [eps](const typename Mesh::point_type &pt) -> auto { // grad

                 Matrix<T, 2, 2> ret;
                 ret(0, 0) = eps;
                 ret(0, 1) = 0.0;
                 ret(1, 0) = 0.0;
                 ret(1, 1) = -eps;
                 return ret;
             },
             [](const typename Mesh::point_type &pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
                 Matrix<T, 2, 1> ret;
                 ret(0) = 0.0;
                 ret(1) = 0.0;
                 return ret;
             },
             [level_set__, sym_grad, gamma, this](
                     const typename Mesh::point_type &pt) mutable -> Eigen::Matrix<T, 2, 1> {/* Neu */
                 Matrix<T, 2, 1> ret;

                 level_set__.cell_assignment(m_cl);


                 ret(0) = gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                 ret(1) = gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);

                 return ret;
             }) {
    }

    test_case_eshelby_LS_eps_DIR_domSym(const test_case_eshelby_LS_eps_DIR_domSym &other)
            : test_case_stokes_eps<T, Function, Mesh, Para_Interface>(other) {
        m_msh = other.m_msh;
        m_cl = other.m_cl;
        gamma = other.gamma;
        eps = other.eps;
        sizeBox = other.sizeBox;
    }


    test_case_eshelby_LS_eps_DIR_domSym(test_case_eshelby_LS_eps_DIR_domSym &other)
            : test_case_stokes_eps<T, Function, Mesh, Para_Interface>(other) {
        m_msh = other.m_msh;
        m_cl = other.m_cl;
        gamma = other.gamma;
        eps = other.eps;
        sizeBox = other.sizeBox;
    }

    void test_case_cell_assignment(const typename Mesh::cell_type &cl_new) {
        m_cl = cl_new;
    }

    void test_case_mesh_assignment(const Mesh &msh_new) {


        m_msh = msh_new;


    }

    typename Mesh::cell_type &upload_cl() {
        return m_cl;
    }


};


template<typename Mesh, typename T, typename Function, typename Para_Interface>
auto make_test_case_eshelby_LS_eps_DIR_domSym(const Mesh &msh, Function &level_set_function, Para_Interface &curve,
                                              params<T> parms_, bool sym_grad, T gamma, T eps, T sizeBox) {
    return test_case_eshelby_LS_eps_DIR_domSym<T, Mesh, Function, Para_Interface>(level_set_function, curve, parms_,
                                                                                  sym_grad, gamma, eps, sizeBox);
}

