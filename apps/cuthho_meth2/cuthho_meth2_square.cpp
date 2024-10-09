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

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cmath>
#include <memory>
#include <sstream>
#include <list>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/SparseExtra>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

using namespace Eigen;

#include "core/core"
#include "core/solvers"
#include "dataio/silo_io.hpp"

#include "methods/hho"
#include "methods/cuthho"

//#include "sol2/sol.hpp"


/*****************************************************************************
 *   Test stuff
 *****************************************************************************/



template<typename Mesh, typename Function>
void
test_quadrature_bis(const Mesh& msh, const Function& level_set_function, size_t degree)
{
    using T = typename Mesh::coordinate_type;

    // x^k
    auto ref_x = [](const typename cuthho_poly_mesh<T>::point_type& pt, const size_t k) -> T {
        return iexp_pow(pt.x(), k);
    };

    // y^k
    auto ref_y = [](const typename cuthho_poly_mesh<T>::point_type& pt, const size_t k) -> T {
        return iexp_pow(pt.y(), k);
    };

    T integral_x = 0.0;
    T integral_y = 0.0;

    auto cbs = cell_basis<Mesh, T>::size(degree);
    for (auto cl : msh.cells)
    {
        // basis functions
        cell_basis<Mesh, T> cb(msh, cl, degree);

        if( location(msh, cl) == element_location::ON_INTERFACE )
        {
            // local mass matrix
            Matrix<T, Dynamic, Dynamic> mass1 = Matrix<T, Dynamic, Dynamic>::Zero(cbs,cbs);
            Matrix<T, Dynamic, Dynamic> mass2 = Matrix<T, Dynamic, Dynamic>::Zero(cbs,cbs);

            // projection : right-hand side
            Matrix<T, Dynamic, 1> RHS_1x = Matrix<T, Dynamic, 1>::Zero(cbs);
            Matrix<T, Dynamic, 1> RHS_2x = Matrix<T, Dynamic, 1>::Zero(cbs);
            Matrix<T, Dynamic, 1> RHS_1y = Matrix<T, Dynamic, 1>::Zero(cbs);
            Matrix<T, Dynamic, 1> RHS_2y = Matrix<T, Dynamic, 1>::Zero(cbs);

            auto qps_n = integrate(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                auto phi = cb.eval_basis(qp.first);
                RHS_1x += qp.second * ref_x(qp.first, degree) * phi;
                RHS_1y += qp.second * ref_y(qp.first, degree) * phi;

                mass1 += qp.second * phi * phi.transpose();
            }
            auto qps_p = integrate(msh, cl, 2*degree, element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                auto phi = cb.eval_basis(qp.first);
                RHS_2x += qp.second * ref_x(qp.first, degree) * phi;
                RHS_2y += qp.second * ref_y(qp.first, degree) * phi;

                mass2 += qp.second * phi * phi.transpose();
            }
            // computation of degrees of projection coefficients
            auto M1_ldlt = mass1.ldlt();
            auto M2_ldlt = mass2.ldlt();
            Matrix<T, Dynamic, 1> U_1x = M1_ldlt.solve(RHS_1x);
            Matrix<T, Dynamic, 1> U_1y = M1_ldlt.solve(RHS_1y);
            Matrix<T, Dynamic, 1> U_2x = M2_ldlt.solve(RHS_2x);
            Matrix<T, Dynamic, 1> U_2y = M2_ldlt.solve(RHS_2y);

            // computation of the integral value
            for (auto& qp : qps_n)
            {
                auto t_phi = cb.eval_basis( qp.first );
                integral_x += qp.second * U_1x.dot(t_phi);
                integral_y += qp.second * U_1y.dot(t_phi);
            }
            for (auto& qp : qps_p)
            {
                auto t_phi = cb.eval_basis( qp.first );
                integral_x += qp.second * U_2x.dot(t_phi);
                integral_y += qp.second * U_2y.dot(t_phi);
            }
        }
        else
        {
            // local mass matrix
            Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, cl, degree);

            // projection : right-hand side
            Matrix<T, Dynamic, 1> RHS_x = Matrix<T, Dynamic, 1>::Zero(cbs);
            Matrix<T, Dynamic, 1> RHS_y = Matrix<T, Dynamic, 1>::Zero(cbs);
            auto qps = integrate(msh, cl, degree);
            for (auto& qp : qps)
            {
                auto phi = cb.eval_basis(qp.first);
                RHS_x += qp.second * ref_x(qp.first, degree) * phi;
                RHS_y += qp.second * ref_y(qp.first, degree) * phi;
            }

            // computation of degrees of projection coefficients
            auto M_ldlt = mass.ldlt();
            Matrix<T, Dynamic, 1> U_x = M_ldlt.solve(RHS_x);
            Matrix<T, Dynamic, 1> U_y = M_ldlt.solve(RHS_y);

            // computation of the integral value
            for (auto& qp : qps)
            {
                auto t_phi = cb.eval_basis( qp.first );
                integral_x += qp.second * U_x.dot(t_phi);
                integral_y += qp.second * U_y.dot(t_phi);
            }
        }
    }

    // final error
    std::cout << "error x^k = " << 1.0/(degree+1) - integral_x << std::endl;
    std::cout << "error y^k = " << 1.0/(degree+1) - integral_y << std::endl;
}




template<typename Mesh, typename Function>
void test_projection(const Mesh& msh, const Function& level_set_function, size_t degree)
{
    using T = typename Mesh::coordinate_type;

    // reference function (to be projected)
    auto ref_fun = [](const typename cuthho_poly_mesh<T>::point_type& pt) -> T {
        return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
    };

    auto grad_ref_fun = [](const typename cuthho_poly_mesh<T>::point_type& pt) -> auto {
        Matrix<T, 1, 2> ret;

        ret(0) = M_PI * std::cos(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        ret(1) = M_PI * std::sin(M_PI*pt.x()) * std::cos(M_PI*pt.y());

        return ret;
    };

    T H1_error_uncut = 0.0;
    T H1_error_cut = 0.0;
    T L2_error_uncut = 0.0;
    T L2_error_cut = 0.0;
    T interface_L2_error = 0.0;

    size_t proj_degree = degree + 1;
    auto cbs = cell_basis<Mesh, T>::size(proj_degree);
    for (auto cl : msh.cells)
    {
        // basis functions
        cell_basis<Mesh, T> cb(msh, cl, proj_degree);

        // local mass matrix
        Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, cl, proj_degree);

        // projection : right-hand side
        Matrix<T, Dynamic, 1> RHS = Matrix<T, Dynamic, 1>::Zero(cbs);
        auto qps = integrate(msh, cl, 2*proj_degree);
        for (auto& qp : qps)
        {
            auto phi = cb.eval_basis(qp.first);
            RHS += qp.second * ref_fun(qp.first) * phi;
        }

        // computation of projection coefficients
        auto M_ldlt = mass.ldlt();
        Matrix<T, Dynamic, 1> U = M_ldlt.solve(RHS);

        // computation of H1 and L2 errors
        if( location(msh, cl) == element_location::ON_INTERFACE )
        {
            auto qps_n = integrate(msh, cl, 2*proj_degree, element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                // H1_error
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<T, 1, 2> grad = Matrix<T, 1, 2>::Zero();
                for (size_t i = 1; i < cbs; i++ )
                    grad += U(i) * t_dphi.block(i, 0, 1, 2);

                Matrix<T, Dynamic, 1> delta_grad = grad_ref_fun(qp.first) - grad;
                H1_error_cut += qp.second * delta_grad.dot(delta_grad);

                // L2_error
                auto t_phi = cb.eval_basis( qp.first );
                auto v = U.dot(t_phi);
                L2_error_cut += qp.second * (ref_fun(qp.first) - v) * (ref_fun(qp.first) - v);
            }

            auto qps_p = integrate(msh, cl, 2*proj_degree, element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                // H1_error
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<T, 1, 2> grad = Matrix<T, 1, 2>::Zero();
                for (size_t i = 1; i < cbs; i++ )
                    grad += U(i) * t_dphi.block(i, 0, 1, 2);

                Matrix<T, Dynamic, 1> delta_grad = grad_ref_fun(qp.first) - grad;
                H1_error_cut += qp.second * delta_grad.dot(delta_grad);

                // L2_error
                auto t_phi = cb.eval_basis( qp.first );
                auto v = U.dot(t_phi);
                L2_error_cut += qp.second * (ref_fun(qp.first) - v) * (ref_fun(qp.first) - v);
            }

            auto qps_int = integrate_interface(msh, cl, 2*proj_degree,
                                               element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_int)
            {
                // L2_error
                auto t_phi = cb.eval_basis( qp.first );
                auto v = U.dot(t_phi);
                interface_L2_error += qp.second * (ref_fun(qp.first) - v) * (ref_fun(qp.first) - v);
            }
        }
        else
        {
            auto qps = integrate(msh, cl, 2*proj_degree);
            for (auto& qp : qps)
            {
                // H1_error
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<T, 1, 2> grad = Matrix<T, 1, 2>::Zero();
                for (size_t i = 1; i < cbs; i++ )
                    grad += U(i) * t_dphi.block(i, 0, 1, 2);

                Matrix<T, Dynamic, 1> delta_grad = grad_ref_fun(qp.first) - grad;
                H1_error_uncut += qp.second * delta_grad.dot(delta_grad);

                // L2_error
                auto t_phi = cb.eval_basis( qp.first );
                auto v = U.dot(t_phi);
                L2_error_uncut += qp.second * (ref_fun(qp.first) - v) * (ref_fun(qp.first) - v);
            }
        }
    }

    // write the results
    std::cout << bold << green << "UNcut cells: " << std::endl;
    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error_uncut) << std::endl;
    std::cout << bold << green << "L2-norm absolute error:           " << std::sqrt(L2_error_uncut) << std::endl;

    std::cout << bold << green << "CUT cells: " << std::endl;
    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error_cut) << std::endl;
    std::cout << bold << green << "L2-norm absolute error:           " << std::sqrt(L2_error_cut) << std::endl;
    std::cout << bold << red << "L2-norm absolute error on interface :           " << std::sqrt(interface_L2_error) << std::endl;
}




void tests_stabilization()
{
    using T = double;
    using Mesh = cuthho_poly_mesh<T>;

    // reference function (to be projected)
    auto ref_fun = [](const typename cuthho_poly_mesh<T>::point_type& pt) -> T {
        return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
    };


    std::vector<size_t> mesh_sizes, pol_orders;

    // meshes
    mesh_sizes.push_back(8);
    mesh_sizes.push_back(16);
    mesh_sizes.push_back(32);
    mesh_sizes.push_back(64);
    mesh_sizes.push_back(128);

    // polynomial orders
    pol_orders.push_back(0);
    pol_orders.push_back(1);
    pol_orders.push_back(2);
    pol_orders.push_back(3);

    for (std::vector<size_t>::iterator it = pol_orders.begin(); it != pol_orders.end(); it++)
    {
        size_t k = *it;

        std::cout << bold << blue << "!!!!!!!! start tests for k = " << k << std::endl;

        for (std::vector<size_t>::iterator it_msh = mesh_sizes.begin();
             it_msh != mesh_sizes.end(); it_msh++)
        {
            // mesh
            size_t N = *it_msh;
            mesh_init_params<T> mip;
            mip.Nx = N;
            mip.Ny = N;
            cuthho_poly_mesh<T> msh(mip);


            T error_T = 0.0;
            T error_F = 0.0;
            T error_stab = 0.0;

            auto cbs = cell_basis<Mesh, T>::size(k+1);
            auto fbs = face_basis<Mesh, T>::size(k);
            for (auto cl : msh.cells)
            {
                T hT = diameter(msh, cl);

                /////////  CELL PROJECTION
                // basis functions
                cell_basis<Mesh, T> cb(msh, cl, k+1);

                // local mass matrix
                Matrix<T, Dynamic, Dynamic> mass_T = make_mass_matrix(msh, cl, k+1);

                // projection : right-hand side
                Matrix<T, Dynamic, 1> RHS_T = Matrix<T, Dynamic, 1>::Zero(cbs);
                auto qps_T = integrate(msh, cl, 2*(k+1));
                for (auto& qp : qps_T)
                {
                    auto phi = cb.eval_basis(qp.first);
                    RHS_T += qp.second * ref_fun(qp.first) * phi;
                }

                // computation of projection coefficients (cell)
                auto M_ldlt = mass_T.ldlt();
                Matrix<T, Dynamic, 1> U_T = M_ldlt.solve(RHS_T);

                // computation of cell projection error
                for (auto& qp : qps_T)
                {
                    auto phi = cb.eval_basis(qp.first);
                    auto delta = ref_fun(qp.first) - phi.dot(U_T);
                    error_T += qp.second * delta * delta;
                }

                // stabilization matrix
                hho_degree_info hdi(k+1, k);
                auto hho_stab = make_hho_naive_stabilization(msh, cl, hdi);

                auto fcs = faces(msh, cl);
                auto num_faces = fcs.size();
                Matrix<T, Dynamic, 1> loc_vect
                    = Matrix<T, Dynamic, 1>::Zero( cbs + num_faces * fbs );
                loc_vect.head(cbs) = U_T;

                ////////// FACE PROJECTION
                for (size_t i = 0; i < fcs.size(); i++)
                {
                    auto fc = fcs[i];
                    face_basis<Mesh, T> fb(msh, fc, k);

                    Matrix<T, Dynamic, Dynamic> mass_F = make_mass_matrix(msh, fc, k);
                    Matrix<T, Dynamic, 1> RHS_F = Matrix<T, Dynamic, 1>::Zero(fbs);
                    auto qps_F = integrate(msh, fc, 2*k);
                    for (auto& qp : qps_F)
                    {
                        auto phi_F = fb.eval_basis(qp.first);
                        RHS_F += qp.second * ref_fun(qp.first) * phi_F;
                    }

                    // computation of projection coefficients (face)
                    auto M_ldlt_F = mass_F.ldlt();
                    Matrix<T, Dynamic, 1> U_F = M_ldlt_F.solve(RHS_F);


                    ///////// Computation of errors
                    // computation of face projection error
                    auto qps_F_bis = integrate(msh, fc, 2*(k+1));
                    for (auto& qp : qps_F_bis)
                    {
                        auto phi_F = fb.eval_basis(qp.first);
                        auto delta = ref_fun(qp.first) - phi_F.dot(U_F);
                        error_F += hT * qp.second * delta * delta;
                    }
                    // computation of the stabilization error

                    loc_vect.block(cbs+i*fbs, 0, fbs, 1) = U_F;
                }
                auto loc_error = (hho_stab * loc_vect).dot(loc_vect);
                if( loc_error < 0)
                {
                    // std::cout << bold << green << "!!!!! loc_error < 0 !!!!! : "
                    //           << loc_error << std::endl;
                    error_stab -= loc_error;
                }
                else
                    error_stab += loc_error;
            }

            std::cout << bold << red
                      << "mesh_size = " << 1.0/N << std::endl;
            std::cout << bold << yellow
                      << "Errors : proj_cells = " << sqrt(error_T) << std::endl;
            std::cout << "         proj_faces = " << sqrt(error_F) << std::endl;
            std::cout << "         stab_error = " << sqrt(error_stab) << std::endl;
        }
    }
}


//////////////////////////   END  TESTS   /////////////////////////////


///////////////////////   FICTITIOUS DOMAIN METHODS  ///////////////////////////

template<typename T, size_t ET, typename testType>
class fictdom_method
{
    using Mat  = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

protected:
    fictdom_method(){}

    virtual std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType test_case, const hho_degree_info hdi,
                     const element_location where = element_location::IN_NEGATIVE_SIDE,
                     const params<T>& parms = params<T>()) = 0;

public:
    std::pair<Mat, Vect>
    make_contrib_uncut(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType test_case)
    {
        auto gr = make_hho_gradrec_vector(msh, cl, hdi);
        Mat stab = make_hho_naive_stabilization(msh, cl, hdi);
        Mat lc = gr.second + stab;
        Mat f = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);
        return std::make_pair(lc, f);
    }


    std::pair<Mat, Vect>
    make_contrib(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const testType test_case, const hho_degree_info hdi,
                 const element_location where = element_location::IN_NEGATIVE_SIDE,
                 const params<T>& parms = params<T>())
    {
        if( location(msh, cl) == where )
            return make_contrib_uncut(msh, cl, hdi, test_case);
        else if( location(msh, cl) != element_location::ON_INTERFACE )
        {
            Mat lc;
            Vect f;
            return std::make_pair(lc, f);
        }
        else // on interface
            return make_contrib_cut(msh, cl, test_case, hdi, where, parms);
    }
};

/////////////////////////  GRADREC_FICTITIOUS_METHOD

template<typename T, size_t ET, typename testType>
class gradrec_fictdom_method : public fictdom_method<T, ET, testType>
{
    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:
    T eta;

    gradrec_fictdom_method(T eta_)
        : fictdom_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType test_case, const hho_degree_info hdi,
                     const element_location where = element_location::IN_NEGATIVE_SIDE,
                     const params<T>& parms = params<T>())
    {
        // LHS
        auto gr = make_hho_gradrec_vector(msh, cl, test_case.level_set_, hdi, where, 1.0);
        Mat stab = make_hho_cut_stabilization(msh, cl, hdi, where)
            + make_hho_cut_interface_penalty(msh, cl, hdi, eta);
        Mat lc = gr.second + stab;


        // RHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        Vect f = Vect::Zero(lc.rows());
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, where);
        f.block(0, 0, cbs, 1) += make_rhs_penalty(msh, cl, celdeg, test_case.bcs_fun, eta);
        f += make_GR_rhs(msh, cl, celdeg, test_case.bcs_fun, test_case.level_set_, gr.first);

        return std::make_pair(lc, f);
    }
};



template<typename T, size_t ET, typename testType>
auto make_gradrec_fictdom_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                 const testType test_case)
{
    return gradrec_fictdom_method<T, ET, testType>(eta_);
}

////////////////////////  NITSCHE_FICTITIOUS_METHOD


template<typename T, size_t ET, typename testType>
class Nitsche_fictdom_method : public fictdom_method<T, ET, testType>
{
    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:
    T eta;

    Nitsche_fictdom_method(T eta_)
        : fictdom_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType test_case, const hho_degree_info hdi,
                     const element_location where = element_location::IN_NEGATIVE_SIDE,
                     const params<T>& parms = params<T>())
    {

        // LHS
        auto gr = make_hho_gradrec_vector(msh, cl, test_case.level_set_, hdi, where, 0.0);
        Mat stab = make_hho_cut_stabilization(msh, cl, hdi, where)
            + make_hho_cut_interface_penalty(msh, cl, hdi, eta);
        Mat Nitsche = make_Nitsche(msh, cl, test_case.level_set_, hdi);
        Mat lc = gr.second + stab + Nitsche;


        // RHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        Vect f = Vect::Zero(lc.rows());
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, where);
        f.block(0, 0, cbs, 1) += make_rhs_penalty(msh, cl, celdeg, test_case.bcs_fun, eta);

        cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);

        auto qpsi = integrate_interface(msh, cl, 2*celdeg - 1, element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : qpsi)
        {
            const auto n = test_case.level_set_.normal(qp.first);
            const auto c_dphi  = cb.eval_gradients(qp.first);
            const auto c_dphi_n  = c_dphi * n;

            f.block(0, 0, cbs, 1) -= qp.second * test_case.bcs_fun(qp.first) * c_dphi_n;
        }

        return std::make_pair(lc, f);
    }
};

template<typename T, size_t ET, typename testType>
auto make_Nitsche_fictdom_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                 testType test_case)
{
    return Nitsche_fictdom_method<T, ET, testType>(eta_);
}

/////////////////////////////////

template<typename Mesh, typename testType>
test_info<typename Mesh::coordinate_type>
run_cuthho_fictdom(const Mesh& msh, size_t degree, testType test_case)
{
    using RealType = typename Mesh::coordinate_type;

    auto level_set_function = test_case.level_set_;

    auto rhs_fun = test_case.rhs_fun;    
    auto sol_fun = test_case.sol_fun;
    auto sol_grad = test_case.sol_grad;
    auto bcs_fun = test_case.bcs_fun;
    
    
    /************** OPEN SILO DATABASE **************/
    silo_database silo;
    silo.create("cuthho_fictdom.silo");
    silo.add_mesh(msh, "mesh");

    /************** MAKE A SILO VARIABLE FOR CELL POSITIONING **************/
    std::vector<RealType> cut_cell_markers;
    for (auto& cl : msh.cells)
    {
        if ( location(msh, cl) == element_location::IN_POSITIVE_SIDE )
            cut_cell_markers.push_back(1.0);
        else if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
            cut_cell_markers.push_back(-1.0);
        else if ( location(msh, cl) == element_location::ON_INTERFACE )
            cut_cell_markers.push_back(0.0);
        else
            throw std::logic_error("shouldn't have arrived here...");
    }
    silo.add_variable("mesh", "cut_cells", cut_cell_markers.data(), cut_cell_markers.size(), zonal_variable_t);

    /************** MAKE A SILO VARIABLE FOR LEVEL SET FUNCTION **************/
    std::vector<RealType> level_set_vals;
    for (auto& pt : msh.points)
        level_set_vals.push_back( level_set_function(pt) );
    silo.add_variable("mesh", "level_set", level_set_vals.data(), level_set_vals.size(), nodal_variable_t);

    /************** MAKE A SILO VARIABLE FOR NODE POSITIONING **************/
    std::vector<RealType> node_pos;
    for (auto& n : msh.nodes)
        node_pos.push_back( location(msh, n) == element_location::IN_POSITIVE_SIDE ? +1.0 : -1.0 );
    silo.add_variable("mesh", "node_pos", node_pos.data(), node_pos.size(), nodal_variable_t);
    
    
    timecounter tc;

    bool sc = true; // static condensation

    /************** ASSEMBLE PROBLEM **************/
    hho_degree_info hdi(degree+1, degree);

    element_location where = element_location::IN_NEGATIVE_SIDE;

    tc.tic();
    auto assembler = make_fict_assembler(msh, bcs_fun, hdi, where);
    auto assembler_sc = make_fict_condensed_assembler(msh, bcs_fun, hdi, where);


    // method with gradient reconstruction (penalty-free)
    auto class_meth = make_gradrec_fictdom_method(msh, 1.0, test_case);
    // Nitsche's method
    // auto class_meth = make_Nitsche_fictdom_method(msh, 1.0, test_case);

    for (auto& cl : msh.cells)
    {
        auto contrib = class_meth.make_contrib(msh, cl, test_case, hdi,
                                               element_location::IN_NEGATIVE_SIDE);
        auto lc = contrib.first;
        auto f = contrib.second;

        if( sc )
            assembler_sc.assemble(msh, cl, lc, f);
        else
            assembler.assemble(msh, cl, lc, f);
    }

    if( sc )
        assembler_sc.finalize();
    else
        assembler.finalize();

    tc.toc();
    std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

    if( sc )
        std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
    else
        std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::cout << "Faces: " << msh.faces.size() << std::endl;


    /************** SOLVE **************/
    tc.tic();
#if 0
    SparseLU<SparseMatrix<RealType>>  solver;
    Matrix<RealType, Dynamic, 1> sol;

    if( sc )
    {
        solver.analyzePattern(assembler_sc.LHS);
        solver.factorize(assembler_sc.LHS);
        sol = solver.solve(assembler_sc.RHS);
    }
    else
    {
        solver.analyzePattern(assembler.LHS);
        solver.factorize(assembler.LHS);
        sol = solver.solve(assembler.RHS);
    }
#endif
#if 1
    Matrix<RealType, Dynamic, 1> sol;
    cg_params<RealType> cgp;
    cgp.histfile = "cuthho_cg_hist.dat";
    cgp.verbose = true;
    cgp.apply_preconditioner = true;
    if( sc )
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
        cgp.max_iter = assembler_sc.LHS.rows();
        conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);
    }
    else
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
        cgp.max_iter = assembler.LHS.rows();
        conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
    }
#endif
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    /************** POSTPROCESS **************/



    postprocess_output<RealType>  postoutput;

    auto uT_gp  = std::make_shared< gnuplot_output_object<RealType> >("fictdom_uT.dat");
    auto Ru_gp  = std::make_shared< gnuplot_output_object<RealType> >("fictdom_Ru.dat");
    auto int_gp  = std::make_shared< gnuplot_output_object<RealType> >("ficdom_int.dat");
    auto diff_gp  = std::make_shared< gnuplot_output_object<RealType> >("fictdom_diff.dat");


    std::vector<RealType>   solution_uT, eigval_data;

    tc.tic();
    RealType    H1_error = 0.0;
    RealType    L2_error = 0.0;
    size_t      cell_i   = 0;
    for (auto& cl : msh.cells)
    {
        bool hide_fict_dom = true; // hide the fictitious domain in the gnuplot outputs
        if (hide_fict_dom && location(msh,cl) == element_location::IN_POSITIVE_SIDE)
            continue;
        
        cell_basis<cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
        auto cbs = cb.size();

        Matrix<RealType, Dynamic, 1> locdata;
        if( sc )
            locdata = assembler_sc.take_local_data(msh, cl, sol);
        else
            locdata = assembler.take_local_data(msh, cl, sol);
        
        Matrix<RealType, Dynamic, 1> cell_dofs = locdata.head(cbs);
        
        auto bar = barycenter(msh, cl, element_location::IN_NEGATIVE_SIDE);
        
        Matrix<RealType, Dynamic, 1> c_phi = cb.eval_basis(bar);
        auto c_val = cell_dofs.dot( c_phi );
        solution_uT.push_back(c_val);
        

        if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE ||
             location(msh, cl) == element_location::ON_INTERFACE )
        {
            Matrix<RealType, 1, 2> real_grad_int = Matrix<RealType, 1, 2>::Zero();
            Matrix<RealType, 1, 2> comp_grad_int = Matrix<RealType, 1, 2>::Zero();
            auto qps = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad(qp.first) - grad).dot(sol_grad(qp.first) - grad);



                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs.dot(t_phi);
                L2_error += qp.second * (sol_fun(qp.first) - v) * (sol_fun(qp.first) - v);

                int_gp->add_data( qp.first, 1.0 );
                uT_gp->add_data( qp.first, cell_dofs.dot(t_phi) );
            }
        }

        cell_i++;
    }

    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;

    postoutput.add_object(uT_gp);
    postoutput.add_object(Ru_gp);
    postoutput.add_object(diff_gp);
    postoutput.add_object(int_gp);
    postoutput.write();

    silo.add_variable("mesh", "uT", solution_uT.data(), solution_uT.size(), zonal_variable_t);

    test_info<RealType> TI;
    TI.H1 = std::sqrt(H1_error);
    TI.L2 = std::sqrt(L2_error);

    tc.toc();
    std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;

    return TI;
}



//////////////////////////////  INTERFACE METHODS  ///////////////////////////

template<typename T, size_t ET, typename testType>
class interface_method
{
    using Mat  = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

protected:
    interface_method(){}

    virtual std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType test_case, const hho_degree_info hdi) = 0;

public:
    std::pair<Mat, Vect>
    make_contrib_uncut(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType test_case)
    {
        T kappa;
        if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
            kappa = test_case.parms.kappa_1;
        else
            kappa = test_case.parms.kappa_2;

        auto gr = make_hho_gradrec_vector(msh, cl, hdi);
        Mat stab = make_hho_naive_stabilization(msh, cl, hdi);
        Mat lc = kappa * (gr.second + stab);
        Mat f = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);
        return std::make_pair(lc, f);
    }


    std::pair<Mat, Vect>
    make_contrib(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const testType test_case, const hho_degree_info hdi)
    {
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return make_contrib_uncut(msh, cl, hdi, test_case);
        else // on interface
            return make_contrib_cut(msh, cl, test_case, hdi);
    }
};

////////////////////////  NITSCHE INTERFACE METHOD


template<typename T, size_t ET, typename testType>
class Nitsche_interface_method : public interface_method<T, ET, testType>
{
    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:
    T eta;

    Nitsche_interface_method(T eta_)
        : interface_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType test_case, const hho_degree_info hdi)
    {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        ////////   LHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        // GR
        auto gr_n = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_NEGATIVE_SIDE, 0.0);
        auto gr_p = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_POSITIVE_SIDE, 0.0);

        // stab
        Mat stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, parms);

        Mat penalty = make_hho_cut_interface_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
        stab.block(0, 0, cbs, cbs) += parms.kappa_1 * penalty;
        stab.block(0, cbs, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, 0, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, cbs, cbs, cbs) += parms.kappa_1 * penalty;

        Mat Nitsche = make_NS_Nitsche(msh, cl, level_set_function, hdi).block(0, 0, cbs, cbs);
        stab.block(0, 0, cbs, cbs) -= parms.kappa_1 * Nitsche;
        stab.block(0, 0, cbs, cbs) -= parms.kappa_1 * Nitsche.transpose();
        stab.block(cbs, 0, cbs, cbs) += parms.kappa_1 * Nitsche;
        stab.block(0, cbs, cbs, cbs) += parms.kappa_1 * Nitsche.transpose();

        Mat lc = stab + parms.kappa_1 * gr_n.second + parms.kappa_2 * gr_p.second;

        ////////    RHS
        Vect f = Vect::Zero(lc.rows());
        // neg part
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                          element_location::IN_NEGATIVE_SIDE);
        f.head(cbs) += parms.kappa_1 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_NEGATIVE_SIDE,
                                level_set_function, dir_jump, eta);
        f.head(cbs) += make_flux_jump(msh, cl, celdeg, element_location::IN_NEGATIVE_SIDE,
                                      test_case.neumann_jump);

        // pos part
        f.block(cbs, 0, cbs, 1) = make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                           element_location::IN_POSITIVE_SIDE);
        f.block(cbs, 0, cbs, 1) += parms.kappa_1 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);

        return std::make_pair(lc, f);
    }
};

template<typename T, size_t ET, typename testType>
auto make_Nitsche_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                   testType test_case)
{
    return Nitsche_interface_method<T, ET, testType>(eta_);
}


////////////////////////  SYMMETRIC GRADREC INTERFACE METHOD


template<typename T, size_t ET, typename testType>
class Sym_gradrec_interface_method : public interface_method<T, ET, testType>
{
    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:
    T eta;

    Sym_gradrec_interface_method(T eta_)
        : interface_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType test_case, const hho_degree_info hdi)
    {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        ///////////////   LHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        // GR
        auto gr_n = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_NEGATIVE_SIDE, 0.5);
        auto gr_p = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_POSITIVE_SIDE, 0.5);

        // stab
        Mat stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, parms);

        Mat penalty = make_hho_cut_interface_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
        stab.block(0, 0, cbs, cbs) += parms.kappa_2 * penalty;
        stab.block(0, cbs, cbs, cbs) -= parms.kappa_2 * penalty;
        stab.block(cbs, 0, cbs, cbs) -= parms.kappa_2 * penalty;
        stab.block(cbs, cbs, cbs, cbs) += parms.kappa_2 * penalty;

        Mat lc = stab + parms.kappa_1 * gr_n.second + parms.kappa_2 * gr_p.second;

        ////////////////    RHS

        Vect f = Vect::Zero(lc.rows());
        // neg part
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                          element_location::IN_NEGATIVE_SIDE);
        // we use element_location::IN_POSITIVE_SIDE to get rid of the Nitsche term
        // (see definition of make_Dirichlet_jump)
        f.head(cbs) -= parms.kappa_2 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);
        f.head(cbs) += 0.5*make_flux_jump(msh, cl, celdeg, element_location::IN_NEGATIVE_SIDE,
                                      test_case.neumann_jump);

        // pos part
        f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                           element_location::IN_POSITIVE_SIDE);
        f.block(cbs, 0, cbs, 1) += parms.kappa_2 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);
        f.block(cbs, 0, cbs, 1)
            += 0.5 * make_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                    test_case.neumann_jump);


        // rhs term with GR
        auto gbs = vector_cell_basis<cuthho_poly_mesh<T>,T>::size(hdi.grad_degree());
        vector_cell_basis<cuthho_poly_mesh<T>, T> gb( msh, cl, hdi.grad_degree() );
        Matrix<T, Dynamic, 1> F_bis = Matrix<T, Dynamic, 1>::Zero( gbs );
        auto iqps = integrate_interface(msh, cl, 2*hdi.grad_degree(),
                                        element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : iqps)
        {
            const auto g_phi    = gb.eval_basis(qp.first);
            const Matrix<T,2,1> n      = level_set_function.normal(qp.first);

            F_bis += qp.second * dir_jump(qp.first) * g_phi * n;
        }
        f -= 0.5 * F_bis.transpose() * (parms.kappa_1 * gr_n.first + parms.kappa_2 * gr_p.first);

        return std::make_pair(lc, f);
    }
};

template<typename T, size_t ET, typename testType>
auto make_sym_gradrec_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                   testType test_case)
{
    return Sym_gradrec_interface_method<T, ET, testType>(eta_);
}


////////////////////////  GRADREC INTERFACE METHOD (method used in the article)


template<typename T, size_t ET, typename testType>
class gradrec_interface_method : public interface_method<T, ET, testType>
{
    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:
    T eta;

    gradrec_interface_method(T eta_)
        : interface_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType test_case, const hho_degree_info hdi)
    {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        ///////////////    LHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        // GR
        auto gr_n = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_NEGATIVE_SIDE, 1.0);
        auto gr_p = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_POSITIVE_SIDE, 0.0);

        // stab
        Mat stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, parms);

        Mat penalty = make_hho_cut_interface_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
        stab.block(0, 0, cbs, cbs) += parms.kappa_1 * penalty;
        stab.block(0, cbs, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, 0, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, cbs, cbs, cbs) += parms.kappa_1 * penalty;

        Mat lc = stab + parms.kappa_1 * gr_n.second + parms.kappa_2 * gr_p.second;

        ///////////////    RHS
        Vect f = Vect::Zero(lc.rows());
        // neg part
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                          element_location::IN_NEGATIVE_SIDE);
        // we use element_location::IN_POSITIVE_SIDE to get rid of the Nitsche term
        // (see definition of make_Dirichlet_jump)
        f.head(cbs) -= parms.kappa_1 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);

        // pos part
        f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                           element_location::IN_POSITIVE_SIDE);
        f.block(cbs, 0, cbs, 1) += parms.kappa_1 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);
        f.block(cbs, 0, cbs, 1)
            += make_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                    test_case.neumann_jump);


        // rhs term with GR
        auto gbs = vector_cell_basis<cuthho_poly_mesh<T>,T>::size(hdi.grad_degree());
        vector_cell_basis<cuthho_poly_mesh<T>, T> gb( msh, cl, hdi.grad_degree() );
        Matrix<T, Dynamic, 1> F_bis = Matrix<T, Dynamic, 1>::Zero( gbs );
        auto iqps = integrate_interface(msh, cl, 2*hdi.grad_degree(),
                                        element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : iqps)
        {
            const auto g_phi    = gb.eval_basis(qp.first);
            const Matrix<T,2,1> n      = level_set_function.normal(qp.first);

            F_bis += qp.second * dir_jump(qp.first) * g_phi * n;
        }
        f -= F_bis.transpose() * (parms.kappa_1 * gr_n.first );

        return std::make_pair(lc, f);
    }
};

template<typename T, size_t ET, typename testType>
auto make_gradrec_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                   testType test_case)
{
    return gradrec_interface_method<T, ET, testType>(eta_);
}


////////////////////////  NITSCHE INTERFACE METHOD 2 (hat gradrec - hat gradrec)


template<typename T, size_t ET, typename testType>
class Nitsche_interface_method_2 : public interface_method<T, ET, testType>
{
    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:
    T eta;

    Nitsche_interface_method_2(T eta_)
        : interface_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType test_case, const hho_degree_info hdi)
    {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        ///////////////////    LHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        // GR
        auto gr_n = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_NEGATIVE_SIDE, 1.0);
        auto gr_p = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_POSITIVE_SIDE, 1.0);

        // stab
        Mat stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, parms);

        Mat penalty = make_hho_cut_interface_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
        stab.block(0, 0, cbs, cbs) += parms.kappa_1 * penalty;
        stab.block(0, cbs, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, 0, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, cbs, cbs, cbs) += parms.kappa_1 * penalty;

        Mat Nitsche = make_NS_Nitsche(msh, cl, level_set_function, hdi).block(0, 0, cbs, cbs);
        stab.block(0, cbs, cbs, cbs) += parms.kappa_2 * Nitsche;
        stab.block(cbs, 0, cbs, cbs) += parms.kappa_2 * Nitsche.transpose();
        stab.block(cbs, cbs, cbs, cbs) -= parms.kappa_2 * Nitsche;
        stab.block(cbs, cbs, cbs, cbs) -= parms.kappa_2 * Nitsche.transpose();

        Mat lc = stab + parms.kappa_1 * gr_n.second + parms.kappa_2 * gr_p.second;

        //////////////////////    RHS
        Vect f = Vect::Zero(lc.rows());
        // neg part
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                          element_location::IN_NEGATIVE_SIDE);
        // we use element_location::IN_POSITIVE_SIDE to get rid of the Nitsche term
        // (see definition of make_Dirichlet_jump)
        f.head(cbs) -= parms.kappa_1 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);

        // pos part
        f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                           element_location::IN_POSITIVE_SIDE);
        f.block(cbs, 0, cbs, 1) -= parms.kappa_2 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_NEGATIVE_SIDE,
                                level_set_function, dir_jump, eta);
        f.block(cbs, 0, cbs, 1) -= (parms.kappa_1-parms.kappa_2) *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);

        f.block(cbs, 0, cbs, 1)
            += make_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                    test_case.neumann_jump);


        // rhs term with GR
        auto gbs = vector_cell_basis<cuthho_poly_mesh<T>,T>::size(hdi.grad_degree());
        vector_cell_basis<cuthho_poly_mesh<T>, T> gb( msh, cl, hdi.grad_degree() );
        Matrix<T, Dynamic, 1> F_bis = Matrix<T, Dynamic, 1>::Zero( gbs );
        auto iqps = integrate_interface(msh, cl, 2*hdi.grad_degree(),
                                        element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : iqps)
        {
            const auto g_phi    = gb.eval_basis(qp.first);
            const Matrix<T,2,1> n      = level_set_function.normal(qp.first);

            F_bis += qp.second * dir_jump(qp.first) * g_phi * n;
        }
        f -= F_bis.transpose() * (parms.kappa_1 * gr_n.first + parms.kappa_2 * gr_p.first);

        return std::make_pair(lc, f);
    }
};

template<typename T, size_t ET, typename testType>
auto make_Nitsche_interface_method_2(const cuthho_mesh<T, ET>& msh, const T eta_,
                                     testType test_case)
{
    return Nitsche_interface_method_2<T, ET, testType>(eta_);
}

/////////////////////////////////

template<typename Mesh, typename testType, typename meth>
test_info<typename Mesh::coordinate_type>
run_cuthho_interface(const Mesh& msh, size_t degree, meth method, testType test_case)
{
    using RealType = typename Mesh::coordinate_type;

    auto level_set_function = test_case.level_set_;

    auto rhs_fun = test_case.rhs_fun;
    auto sol_fun = test_case.sol_fun;
    auto sol_grad = test_case.sol_grad;
    auto bcs_fun = test_case.bcs_fun;
    auto dirichlet_jump = test_case.dirichlet_jump;
    auto neumann_jump = test_case.neumann_jump;
    struct params<RealType> parms = test_case.parms;

    timecounter tc;

    bool sc = true; // static condensation


    /************** ASSEMBLE PROBLEM **************/
    hho_degree_info hdi(degree+1, degree);

    tc.tic();
    auto assembler = make_interface_assembler(msh, bcs_fun, hdi);
    auto assembler_sc = make_interface_condensed_assembler(msh, bcs_fun, hdi);
    for (auto& cl : msh.cells)
    {
        auto contrib = method.make_contrib(msh, cl, test_case, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;

        if( sc )
            assembler_sc.assemble(msh, cl, lc, f);
        else
            assembler.assemble(msh, cl, lc, f);
    }

    if( sc )
        assembler_sc.finalize();
    else
        assembler.finalize();

    tc.toc();
    std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

    if( sc )
        std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
    else
        std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::cout << "Faces: " << msh.faces.size() << std::endl;

    /************** SOLVE **************/
    tc.tic();
#if 0
    SparseLU<SparseMatrix<RealType>>  solver;
    Matrix<RealType, Dynamic, 1> sol;

    if( sc )
    {
        solver.analyzePattern(assembler_sc.LHS);
        solver.factorize(assembler_sc.LHS);
        sol = solver.solve(assembler_sc.RHS);
    }
    else
    {
        solver.analyzePattern(assembler.LHS);
        solver.factorize(assembler.LHS);
        sol = solver.solve(assembler.RHS);
    }
#endif
//#if 0
    Matrix<RealType, Dynamic, 1> sol;
    cg_params<RealType> cgp;
    cgp.histfile = "cuthho_cg_hist.dat";
    cgp.verbose = true;
    cgp.apply_preconditioner = true;
    if( sc )
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
        cgp.max_iter = assembler_sc.LHS.rows();
        conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);
    }
    else
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
        cgp.max_iter = assembler.LHS.rows();
        conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
    }
//#endif
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    /************** POSTPROCESS **************/


    postprocess_output<RealType>  postoutput;

    auto uT_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT.dat");
    auto diff_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_diff.dat");


    std::vector<RealType>   solution_uT;

    tc.tic();
    RealType    H1_error = 0.0;
    RealType    L2_error = 0.0;
    size_t      cell_i   = 0;
    for (auto& cl : msh.cells)
    {
        cell_basis<cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
        auto cbs = cb.size();
        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto fbs = face_basis<cuthho_poly_mesh<RealType>,RealType>::size(hdi.face_degree());
        
        Matrix<RealType, Dynamic, 1> locdata_n, locdata_p, locdata;
        Matrix<RealType, Dynamic, 1> cell_dofs_n, cell_dofs_p, cell_dofs;

        if (location(msh, cl) == element_location::ON_INTERFACE)
        {
            if( sc )
            {
                locdata_n = assembler_sc.take_local_data(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                locdata_p = assembler_sc.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
            {
                locdata_n = assembler.take_local_data(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                locdata_p = assembler.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }

            cell_dofs_n = locdata_n.head(cbs);
            cell_dofs_p = locdata_p.head(cbs);


            auto qps_n = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs_n(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad(qp.first) - grad).dot(sol_grad(qp.first) - grad);

                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs_n.dot(t_phi);
                uT_gp->add_data(qp.first, v);
                
                /* Compute L2-error */
                L2_error += qp.second * (sol_fun(qp.first) - v) * (sol_fun(qp.first) - v);
            }
            
            
            auto qps_p = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs_p(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad(qp.first) - grad).dot(sol_grad(qp.first) - grad);

                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs_p.dot(t_phi);
                uT_gp->add_data(qp.first, v);

                /* Compute L2-error */
                L2_error += qp.second * (sol_fun(qp.first) - v) * (sol_fun(qp.first) - v);
            }
        }
        else
        {
            if( sc )
            {
                locdata = assembler_sc.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
                locdata = assembler.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            cell_dofs = locdata.head(cbs);

            auto qps = integrate(msh, cl, 2*hdi.cell_degree());
            for (auto& qp : qps)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad(qp.first) - grad).dot(sol_grad(qp.first) - grad);

                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs.dot(t_phi);
                uT_gp->add_data(qp.first, v);

                /* Compute L2-error */
                L2_error += qp.second * (sol_fun(qp.first) - v) * (sol_fun(qp.first) - v);
            }
        }

        cell_i++;
    }

    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
    std::cout << bold << green << "L2-norm absolute error:           " << std::sqrt(L2_error) << std::endl;

    postoutput.add_object(uT_gp);
    postoutput.add_object(diff_gp);
    postoutput.write();



    test_info<RealType> TI;
    TI.H1 = std::sqrt(H1_error);
    TI.L2 = std::sqrt(L2_error);

    if (false)
    {
        /////////////// compute condition number
        SparseMatrix<RealType> Mat;
        // Matrix<RealType, Dynamic, Dynamic> Mat;
        if (sc)
            Mat = assembler_sc.LHS;
        else
            Mat = assembler.LHS;


        RealType sigma_max, sigma_min;

        // Construct matrix operation object using the wrapper class SparseSymMatProd
        Spectra::SparseSymMatProd<RealType> op(Mat);
        // Construct eigen solver object, requesting the largest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::LARGEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > max_eigs(&op, 1, 10);
        max_eigs.init();
        max_eigs.compute();
        if(max_eigs.info() == Spectra::SUCCESSFUL)
            sigma_max = max_eigs.eigenvalues()(0);


        // Construct eigen solver object, requesting the smallest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::SMALLEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > min_eigs(&op, 1, 10);

        min_eigs.init();
        min_eigs.compute();
        if(min_eigs.info() == Spectra::SUCCESSFUL)
            sigma_min = min_eigs.eigenvalues()(0);

        // compute condition number
        RealType cond = sigma_max / sigma_min;
        TI.cond = cond;
        std::cout << "sigma_max = " << sigma_max << "   sigma_min = "
                  << sigma_min << "  cond = " << cond
                  << std::endl;
    }
    else
        TI.cond = 0.0;

    tc.toc();
    std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;


    return TI;
}




/////////////////////////   AUTOMATIC TESTS  //////////////////////////


void convergence_test(void)
{
    using T = double;

    std::vector<size_t> mesh_sizes, pol_orders;

    // meshes
    mesh_sizes.push_back(8);
    mesh_sizes.push_back(16);
    mesh_sizes.push_back(32);
    mesh_sizes.push_back(64);
    mesh_sizes.push_back(128);
    // mesh_sizes.push_back(256);

    // polynomial orders
    pol_orders.push_back(0);
    pol_orders.push_back(1);
    pol_orders.push_back(2);
    pol_orders.push_back(3);


    // export to files ...
    std::vector<std::string> files;
    files.push_back("./output/test_k0.txt");
    files.push_back("./output/test_k1.txt");
    files.push_back("./output/test_k2.txt");
    files.push_back("./output/test_k3.txt");

    for (std::vector<size_t>::iterator it = pol_orders.begin(); it != pol_orders.end(); it++)
    {
        size_t k = *it;

        std::cout << "start tests for k = " << k << std::endl;

        // init the file
        std::ofstream output;
        output.open (files.at(*it), std::ios::in | std::ios::trunc);
        if (!output.is_open())
            throw std::logic_error("file not open");

        // output << "N\th\tH1\tordre1\tL2\tordre2" << std::endl;
        output << "N\th\tH1\tordre1\tL2\tordre2\tcond" << std::endl;

        // convergence tests
        T previous_H1 = 0.0;
        T previous_L2 = 0.0;
        T previous_h = 0.0;
        for (std::vector<size_t>::iterator it_msh = mesh_sizes.begin();
             it_msh != mesh_sizes.end(); it_msh++)
        {
            size_t N = *it_msh;

            // init mesh (with agglomeration)
            mesh_init_params<T> mip;
            mip.Nx = N;
            mip.Ny = N;
            cuthho_poly_mesh<T> msh(mip);
            size_t int_refsteps = 1;
            T radius = 1.0/3.0;
            auto circle_level_set_function = circle_level_set<T>(radius, 0.5, 0.5);

            // auto level_set_function = flower_level_set<T>(0.31, 0.5, 0.5, 4, 0.04);
            // auto level_set_function = circle_level_set<T>(radius, 0.5, 0.5);
            // auto level_set_function = square_level_set<T>(1.05, -0.05, -0.05, 1.05);
            // auto level_set_function = square_level_set<T>(1.0, -0.0, -0.0, 1.0);
            // auto level_set_function = square_level_set<T>(0.76, 0.24, 0.24, 0.76);
            auto level_set_function = square_level_set<T>(0.751, 0.249, 0.249, 0.751);
            detect_node_position(msh, level_set_function);
            detect_cut_faces(msh, level_set_function);
            if(1)  // AGGLOMERATION
            {
                detect_cut_cells(msh, level_set_function);
                detect_cell_agglo_set(msh, level_set_function);
                make_neighbors_info_cartesian(msh);
                refine_interface(msh, level_set_function, int_refsteps);
                make_agglomeration(msh, level_set_function);
            }
            else  // NODE DISPLACEMENT
            {
                move_nodes(msh, level_set_function);
                detect_cut_faces(msh, level_set_function); //do it again to update intersection points
                detect_cut_cells(msh, level_set_function);
                refine_interface(msh, level_set_function, int_refsteps);
            }

            // compute solution/errors
            test_info<T> TI;
            // auto TI = run_cuthho_interface(msh, level_set_function, k, 3);

            // auto TI = run_cuthho_interface(msh, level_set_function, k, 3, test_case);
            if(0) // sin(\pi x) * sin(\pi y)
            {
                auto test_case = make_test_case_laplacian_sin_sin(msh, level_set_function);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                TI = run_cuthho_interface(msh, k, meth3, test_case);
                // TI = run_cuthho_fictdom(msh, k, test_case);
            }
            if(0) // 1 + sin(\pi x) * sin(\pi y)
            {
                auto test_case = make_test_case_laplacian_sin_sin_bis(msh, level_set_function);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                // TI = run_cuthho_interface(msh, k, meth3, test_case);
                TI = run_cuthho_fictdom(msh, k, test_case);
            }
            if(0) // sin(\pi (x-a)/(b-a)) * sin(\pi (y-c)/(d-c))
            {
                T a = 0.23;
                T b = 0.77;
                T c = 0.23;
                T d = 0.77;
                auto test_case = make_test_case_laplacian_sin_sin_gen(msh, level_set_function,
                                                                      a, b, c, d);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                // TI = run_cuthho_interface(msh, k, meth3, test_case);
                TI = run_cuthho_fictdom(msh, k, test_case);
            }
            if(0) // exp(x) * cos(y)
            {
                auto test_case = make_test_case_laplacian_exp_cos(msh, level_set_function);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                // TI = run_cuthho_interface(msh, k, meth3, test_case);
                TI = run_cuthho_fictdom(msh, k, test_case);
            }
            if(1) // jumps sin_sin -> exp_cos
            {
                auto test_case = make_test_case_laplacian_jumps_1(msh, level_set_function);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                TI = run_cuthho_interface(msh, k, meth3, test_case);
            }
            if(0) // jumps2 exp_cos -> sin_sin
            {
                auto test_case = make_test_case_laplacian_jumps_2(msh, level_set_function);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                TI = run_cuthho_interface(msh, k, meth3, test_case);
            }
            if(0) // jumps3 sin_sin -> sin_sin + pol
            {
                auto test_case = make_test_case_laplacian_jumps_3(msh, level_set_function);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                TI = run_cuthho_interface(msh, k, meth3, test_case);
            }
            if(0) // contrast deg 2
            {
                auto parms = params<T>();
                parms.kappa_1 = 1.0;
                parms.kappa_2 = 1000.0;

                auto test_case = make_test_case_laplacian_contrast_2(msh, circle_level_set_function, parms);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                TI = run_cuthho_interface(msh, k, meth3, test_case);
            }
            if(0) // contrast deg 6
            {
                auto parms = params<T>();
                parms.kappa_1 = 1.0;
                parms.kappa_2 = 1.0;

                auto test_case = make_test_case_laplacian_contrast_6(msh, circle_level_set_function, parms);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                TI = run_cuthho_interface(msh, k, meth3, test_case);
            }

            if(0) // homogeneous test case on a circle
            {
                auto test_case = make_test_case_laplacian_circle_hom(msh, circle_level_set_function);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                // TI = run_cuthho_interface(msh, k, meth3, test_case);
                TI = run_cuthho_fictdom(msh, k, test_case);
            }

            // auto TI = run_cuthho_fictdom(msh, level_set_function, k);


            // report info in the file
            T h = 1.0/N;
            if (it_msh == mesh_sizes.begin())
            {
                // output << N << "\t" << h << "\t" << TI.H1 << "\t" << "."
                //        << "\t" << TI.L2 << "\t" << "." << "\t" << "0.0"
                //        << std::endl;
                output << N << "\t" << h << "\t" << TI.H1 << "\t" << "."
                       << "\t" << TI.L2 << "\t" << "." << "\t" << TI.cond
                       << std::endl;
            }
            else
            {
                T orderH = log(previous_H1 / TI.H1) / log(previous_h / h);
                T orderL = log(previous_L2 / TI.L2) / log(previous_h / h);
                // output << N << "\t" << h << "\t" << TI.H1 << "\t" << orderH
                //        << "\t" << TI.L2 << "\t" << orderL << "\t" << "0.0"
                //        << std::endl;
                output << N << "\t" << h << "\t" << TI.H1 << "\t" << orderH
                       << "\t" << TI.L2 << "\t" << orderL << "\t" << TI.cond
                       << std::endl;
            }
            previous_H1 = TI.H1;
            previous_L2 = TI.L2;
            previous_h = h;
        }
        // close the file
        output.close();
    }

    // update the gnuplot curves
    system("gnuplot './output/gnuplot_script.txt'");

    // update the .pdf file
    system("pdflatex ./output/autom_tests.tex");

    // open the .pdf file
    system("xdg-open ./autom_tests.pdf");
}

//////////////////////////     MAIN        ////////////////////////////
#if 0
int main(int argc, char **argv)
{
    convergence_test();
    // tests_stabilization();
    // interface_residus();
    return 1;
}
#endif

#if 1
int main(int argc, char **argv)
{
    using RealType = double;
    
    size_t degree           = 0;
    size_t int_refsteps     = 4;

    bool dump_debug         = false;
    bool solve_interface    = false;
    bool solve_fictdom      = false;
    bool agglomeration      = false;

    mesh_init_params<RealType> mip;
    mip.Nx = 8;
    mip.Ny = 8;

    /* k <deg>:     method degree
     * M <num>:     number of cells in x direction
     * N <num>:     number of cells in y direction
     * r <num>:     number of interface refinement steps
     *
     * i:           solve interface problem
     * f:           solve fictitious domain problem
     *
     * D:           use node displacement to solve bad cuts (default)
     * A:           use agglomeration to solve bad cuts 
     *
     * d:           dump debug data
     */

    int ch;
    while ( (ch = getopt(argc, argv, "k:M:N:r:ifDAd")) != -1 )
    {
        switch(ch)
        {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
                break;

            case 'd':
                dump_debug = true;
                break;

            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;


    timecounter tc;

    /************** BUILD MESH **************/
    tc.tic();
    cuthho_poly_mesh<RealType> msh(mip);
    tc.toc();
    std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;
    /************** LEVEL SET FUNCTION **************/
    RealType radius = 1.0/3.0;
    auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);
    // auto level_set_function = line_level_set<RealType>(0.5);
    // auto level_set_function = flower_level_set<RealType>(0.31, 0.5, 0.5, 4, 0.04);
    /************** DO cutHHO MESH PROCESSING **************/

    tc.tic();
    detect_node_position(msh, level_set_function);
    detect_cut_faces(msh, level_set_function);

    if (agglomeration)
    {
        detect_cut_cells(msh, level_set_function);
        detect_cell_agglo_set(msh, level_set_function);
        make_neighbors_info_cartesian(msh);
        // make_neighbors_info(msh);
        refine_interface(msh, level_set_function, int_refsteps);
        make_agglomeration(msh, level_set_function);
    }
    else
    {
        move_nodes(msh, level_set_function);
        detect_cut_faces(msh, level_set_function); //do it again to update intersection points
        detect_cut_cells(msh, level_set_function);
        refine_interface(msh, level_set_function, int_refsteps);
    }


    tc.toc();
    std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

    if (dump_debug)
    {
        dump_mesh(msh);
        output_mesh_info(msh, level_set_function);
        test_projection(msh, level_set_function, degree);
    }

    output_mesh_info(msh, level_set_function);

    // jumps sin_sin -> exp_cos
    // auto test_case = make_test_case_laplacian_jumps_1(msh, level_set_function);
    // jumps3 sin_sin -> sin_sin + pol
    auto test_case = make_test_case_laplacian_jumps_3(msh, level_set_function);

    // auto method = make_Nitsche_interface_method(msh, 1.0, test_case);
    // auto method = make_sym_gradrec_interface_method(msh, 1.0, test_case);
     auto method = make_gradrec_interface_method(msh, 1.0, test_case);
//    auto method = make_Nitsche_interface_method_2(msh, 1.0, test_case);

    if (solve_interface)
        run_cuthho_interface(msh, degree, method, test_case);
    
    if (solve_fictdom)
        run_cuthho_fictdom(msh, degree, test_case);


    return 0;
}
#endif
