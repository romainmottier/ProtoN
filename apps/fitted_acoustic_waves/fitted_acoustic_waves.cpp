/*
 *       /\        Omar Duran 2019
 *      /__\       omar-yesid.duran-triana@enpc.fr
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


////////////////////////// @omar::Begining:: simple fitted implementation /////////////////////////////////////////////

//#define spatial_errors_Q
//#define quadratic_time_solution_Q

int main(int argc, char **argv)
{
    using RealType = double;
    size_t k_degree = 0;
    size_t n_divs   = 0;
    
    size_t nt       = 10;
    RealType dt     = 0.1;
    RealType ti = 0.0;
    
    int opt;
    while ( (opt = getopt(argc, argv, "k:l:n")) != -1 )
    {
        switch(opt)
        {
            case 'k':
            {
                k_degree = atoi(optarg);
            }
                break;
            case 'l':
            {
                n_divs = atoi(optarg);
            }
                break;
            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }
    
    std::cout << bold << red << "k : " << k_degree << reset << std::endl;
    std::cout << bold << red << "l : " << n_divs << reset << std::endl;
    std::cout << bold << red << "nt : " << nt << reset << std::endl;
    std::cout << bold << red << "dt : " << dt << reset << std::endl;

    // The mesh in ProtoN seems like is always 2D
     mesh_init_params<RealType> mip;
     mip.Nx = 1;
     mip.Ny = 1;
    
    for (size_t i = 0; i < n_divs; i++) {
        mip.Nx *= 2;
        mip.Ny *= 2;
    }
    
    timecounter tc;
    
    // Building the cartesian mesh
    tc.tic();
    poly_mesh<RealType> msh(mip);
    tc.toc();

    std::cout << bold << cyan << "Mesh generation: " << tc << " seconds" << reset << std::endl;
    
//    RealType tf = dt*nt;
    Matrix<RealType, Dynamic, 1> alpha_dof_n_m;
    Matrix<RealType, Dynamic, 1> alpha_dof_n;
    
//    // Preliminary Initialization
//    for(size_t it = 0; it < 0; it++){
//
//    std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
//
//    // Manufactured solution
//        RealType t = dt*it+ti;
//        auto rhs_fun = [&t](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
//#ifdef quadratic_time_solution_Q
//            return 2.0 * (M_PI * M_PI * t * t + 1.0) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
//#else
//            return 2.0 * M_PI * M_PI * std::cos(std::sqrt(2.0)*M_PI*t)  * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
//#endif
//        };
//
//        auto exact_sol_fun = [&t](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
//
//#ifdef quadratic_time_solution_Q
//            return t * t * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
//#else
//            return std::cos(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
//#endif
//        };
//    #ifdef spatial_errors_Q
//        auto exact_grad_sol_fun = [&t](const typename poly_mesh<RealType>::point_type& pt) -> Matrix<RealType, 1, 2> {
//            Matrix<RealType, 1, 2> grad;
//            grad(0,0) = std::cos(std::sqrt(2.0)*M_PI*t) * M_PI * std::cos(M_PI*pt.x()) * std::sin(M_PI*pt.y());
//            grad(0,1) = std::cos(std::sqrt(2.0)*M_PI*t) * M_PI * std::sin(M_PI*pt.x()) * std::cos(M_PI*pt.y());
//            return grad;
//        };
//    #endif
//
//        // Creating HHO approximation spaces and corresponding linear operator
//        hho_degree_info hho_di(k_degree,k_degree); // simple stabilization term
//
//        auto assembler = make_assembler(msh, hho_di);
//        tc.tic();
//        for (auto& cell : msh.cells)
//        {
//            auto reconstruction_operator = make_hho_laplacian(msh, cell, hho_di);
//            auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
//
//            Matrix<RealType, Dynamic, Dynamic> laplacian_loc = reconstruction_operator.second + stabilization_operator;
//            Matrix<RealType, Dynamic, 1> f_loc = make_rhs(msh, cell, hho_di.cell_degree(), rhs_fun);
//            assembler.assemble(msh, cell, laplacian_loc, f_loc, exact_sol_fun);
//        }
//        assembler.finalize();
//        tc.toc();
//
//        std::cout << bold << cyan << "Assembly completed: " << tc << " seconds" << reset << std::endl;
//
//        Matrix<RealType, Dynamic, 1> alpha_dof_n_p;
//
//        tc.tic();
//        SparseLU<SparseMatrix<RealType>> analysis;
//        analysis.analyzePattern(assembler.LHS);
//        analysis.factorize(assembler.LHS);
//        alpha_dof_n_p = analysis.solve(assembler.RHS);
//        tc.toc();
//
//        std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
//
//        size_t cell_i = 0;
//    #ifdef spatial_errors_Q
//        tc.tic();
//        RealType l2_error = 0.0;
//        RealType H1_error = 0.0;
//
//        for (auto &cell : msh.cells) {
//
//            if(cell_i == 0){
//                RealType h = diameter(msh, cell);
//                    std::cout << green << "h size = " << std::endl << h << std::endl;
//            }
//
//            cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
//            auto cell_dof = cell_basis.size();
//            Matrix<RealType, Dynamic, 1> cell_alpha_dof_n_p = alpha_dof_n_p.block(cell_i*cell_dof, 0, cell_dof, 1);
//            auto int_rule = integrate(msh, cell, 2.0*hho_di.cell_degree());
//
//            // Error integrals
//            for (auto & point_pair : int_rule) {
//
//                RealType omega = point_pair.second;
//
//                // L2-error
//                auto t_phi = cell_basis.eval_basis( point_pair.first );
//                RealType u_exact = exact_sol_fun(point_pair.first);
//                RealType uh = cell_alpha_dof_n_p.dot( t_phi );
//                RealType error = u_exact - uh;
//                l2_error += omega * error * error;
//
//                // H1-error
//                auto t_dphi = cell_basis.eval_gradients( point_pair.first );
//                Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();
//
//                for (size_t i = 1; i < cell_dof; i++ ){
//                    grad_uh += cell_alpha_dof_n_p(i) * t_dphi.block(i, 0, 1, 2);
//                }
//
//                Matrix<RealType, 1, 2> grad_u_exact = exact_grad_sol_fun(point_pair.first);
//                H1_error += omega * (grad_u_exact - grad_uh).dot(grad_u_exact - grad_uh);
//
//
//            }
//            cell_i++;
//        }
//
//
//        std::cout << green << "L2-norm error = " << std::endl << std::sqrt(l2_error) << std::endl;
//        std::cout << green << "H1-norm error = " << std::endl << std::sqrt(H1_error) << std::endl;
//        std::cout << std::endl;
//        tc.toc();
//
//        std::cout << bold << cyan << "Error completed: " << tc << " seconds" << reset << std::endl;
//
//    #endif
//
//        auto num_cells = msh.cells.size();
//        std::vector<RealType> exact_u, approx_u;
//        exact_u.reserve( num_cells );
//        approx_u.reserve( num_cells );
//        cell_i = 0;
//        for (auto& cell : msh.cells)
//        {
//
//            auto bar = barycenter(msh, cell);
//            exact_u.push_back( exact_sol_fun(bar) );
//
//            cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
//            auto cell_dof = cell_basis.size();
//            Matrix<RealType, Dynamic, 1> cell_alpha_dof_n_p = alpha_dof_n_p.block(cell_i*cell_dof, 0, cell_dof, 1);
//            auto t_phi = cell_basis.eval_basis( bar );
//             RealType uh = cell_alpha_dof_n_p.dot( t_phi );
//            approx_u.push_back(uh);
//            cell_i++;
//        }
//
//        silo_database silo;
//        std::string silo_file_name = "scalar_wave_" + std::to_string(it) + ".silo";
//        silo.create(silo_file_name.c_str());
//        silo.add_mesh(msh, "mesh");
//                silo.add_variable("mesh", "u", exact_u.data(), exact_u.size(), zonal_variable_t);
//        silo.add_mesh(msh, "mesh");
//                silo.add_variable("mesh", "uh", approx_u.data(), approx_u.size(), zonal_variable_t);
//        silo.close();
//
//    }
    
    // Creating HHO approximation spaces and corresponding linear operator
    hho_degree_info hho_di(k_degree,k_degree);
    // Construct Mass matrix
    auto mass_assembler_p = make_assembler(msh, hho_di);
    auto mass_assembler_v = make_assembler(msh, hho_di);
    auto mass_assembler_a = make_assembler(msh, hho_di);
    tc.tic();
    // Projection for acceleration
    {
        RealType t = 0.0;
        
#ifdef quadratic_time_solution_Q
        auto exact_scal_sol_fun = [&t](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
            return t * t * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        };
        auto exact_vel_sol_fun = [&t](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
            return 2.0 * t * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        };
        auto exact_accel_sol_fun = [](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
            return 2.0* std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        };
#else
        auto exact_scal_sol_fun = [&t](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
            return std::cos(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        };
        auto exact_vel_sol_fun = [&t](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
            return -std::sqrt(2.0)*M_PI*std::sin(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        };
        auto exact_accel_sol_fun = [&t](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
            return -2.0*M_PI*M_PI*std::cos(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        };
#endif
        for (auto& cell : msh.cells)
        {
            auto mass_operator = make_mass_matrix(msh, cell, hho_di);
            auto mass_operator_a = make_cell_mass_matrix(msh, cell, hho_di);
            
            Matrix<RealType, Dynamic, 1> f_p = make_rhs(msh, cell, hho_di.cell_degree(), exact_scal_sol_fun);
            Matrix<RealType, Dynamic, 1> f_v = make_rhs(msh, cell, hho_di.cell_degree(), exact_vel_sol_fun);
            Matrix<RealType, Dynamic, 1> f_a = make_rhs(msh, cell, hho_di.cell_degree(), exact_accel_sol_fun);
            
            mass_assembler_p.assemble(msh, cell, mass_operator, f_p, exact_scal_sol_fun);
            mass_assembler_v.assemble(msh, cell, mass_operator, f_v, exact_vel_sol_fun);
            mass_assembler_a.assemble(msh, cell, mass_operator_a, f_a, exact_accel_sol_fun);
        }
        mass_assembler_p.finalize();
        mass_assembler_v.finalize();
        mass_assembler_a.finalize();
    }
    tc.toc();
    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;
    
    // Projecting initial scalar, velocity and acceleration
    Matrix<RealType, Dynamic, 1> p_dof_n, v_dof_n, a_dof_n;
    
    tc.tic();
    SparseLU<SparseMatrix<RealType>> analysis;
    analysis.analyzePattern(mass_assembler_p.LHS);
    analysis.factorize(mass_assembler_p.LHS);
    p_dof_n = analysis.solve(mass_assembler_p.RHS); // Initial scalar
    v_dof_n = analysis.solve(mass_assembler_v.RHS); // Initial velocity
    a_dof_n = analysis.solve(mass_assembler_a.RHS); // Initial acceleration
    tc.toc();
    
    { // rendering projected acceleration
        auto num_cells = msh.cells.size();
        std::vector<RealType> exact_u, approx_u;
        exact_u.reserve( num_cells );
        approx_u.reserve( num_cells );
        size_t cell_i = 0;
        size_t it = 0;
        
        RealType t = 0.0;
        auto exact_sol_fun = [&t](const typename poly_mesh<RealType>::point_type& pt) -> RealType {

#ifdef quadratic_time_solution_Q
            return t * t * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
#else
            return std::cos(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
#endif
        };
        
        for (auto& cell : msh.cells)
        {

            auto bar = barycenter(msh, cell);
            exact_u.push_back( exact_sol_fun(bar) );
            
            cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
            auto cell_dof = cell_basis.size();
            Matrix<RealType, Dynamic, 1> cell_alpha_dof_n_p = p_dof_n.block(cell_i*cell_dof, 0, cell_dof, 1);
            auto t_phi = cell_basis.eval_basis( bar );
             RealType uh = cell_alpha_dof_n_p.dot( t_phi );
            approx_u.push_back(uh);
            cell_i++;
        }
        
        silo_database silo;
        std::string silo_file_name = "scalar_wave_" + std::to_string(it) + ".silo";
        silo.create(silo_file_name.c_str());
        silo.add_mesh(msh, "mesh");
                silo.add_variable("mesh", "u", exact_u.data(), exact_u.size(), zonal_variable_t);
        silo.add_mesh(msh, "mesh");
                silo.add_variable("mesh", "uh", approx_u.data(), approx_u.size(), zonal_variable_t);
        silo.close();
    }
    
    //
    
    // Transient problem
    bool is_implicit_Q = true;
    
    if (is_implicit_Q) {
        
        Matrix<RealType, Dynamic, 1> a_dof_np = a_dof_n;
        
        RealType beta = 0.25;
        RealType gamma = 0.5;
        
        for(size_t it = 1; it <= nt; it++){
                
            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
            
            // Manufactured solution
                RealType t = dt*it+ti;
            auto rhs_fun = [&t](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
#ifdef quadratic_time_solution_Q
                return 2.0 * (M_PI * M_PI * t * t + 1.0) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
#else
                return 0;
#endif
            };

            auto exact_sol_fun = [&t](const typename poly_mesh<RealType>::point_type& pt) -> RealType {

#ifdef quadratic_time_solution_Q
                return t * t * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
#else
                return std::cos(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
#endif
            };
            #ifdef spatial_errors_Q
                auto exact_grad_sol_fun = [&t](const typename poly_mesh<RealType>::point_type& pt) -> Matrix<RealType, 1, 2> {
                    Matrix<RealType, 1, 2> grad;
                    grad(0,0) = std::cos(std::sqrt(2.0)*M_PI*t) * M_PI * std::cos(M_PI*pt.x()) * std::sin(M_PI*pt.y());
                    grad(0,1) = std::cos(std::sqrt(2.0)*M_PI*t) * M_PI * std::sin(M_PI*pt.x()) * std::cos(M_PI*pt.y());
                    return grad;
                };
            #endif
                
                
                auto assembler = make_assembler(msh, hho_di);
                tc.tic();
                for (auto& cell : msh.cells)
                {
                    auto reconstruction_operator = make_hho_laplacian(msh, cell, hho_di);
                    auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
                    Matrix<RealType, Dynamic, Dynamic> laplacian_loc = reconstruction_operator.second + stabilization_operator;

                    Matrix<RealType, Dynamic, 1> f_loc = make_rhs(msh, cell, hho_di.cell_degree(), rhs_fun);
//                    Matrix<RealType, Dynamic, 1> alpha_loc = project_function(msh, cell, hho_di, rhs_fun);
                    
                    assembler.assemble(msh, cell, laplacian_loc, f_loc, exact_sol_fun);
                }
                assembler.finalize();
            
            
            // Compute intermediate state for scalar and rate
            p_dof_n = p_dof_n + dt*v_dof_n + 0.5*dt*dt*(1-2.0*beta)*a_dof_n;
            v_dof_n = v_dof_n + dt*(1-gamma)*a_dof_n;
            Matrix<RealType, Dynamic, 1> res = assembler.LHS*p_dof_n;
            
//            std::cout << bold << cyan << "p_dof_n : " << p_dof_n << reset << std::endl;
//            std::cout << bold << cyan << "v_dof_n : " << v_dof_n << reset << std::endl;
//            std::cout << bold << cyan << "a_dof_n : " << v_dof_n << reset << std::endl;
            
            assembler.LHS *= beta*(dt*dt);
            assembler.LHS += mass_assembler_a.LHS;
            
        //        std::cout << bold << cyan << "Mass matrix : " << mass_assembler.LHS << " seconds" << reset << std::endl;
        //        std::cout << bold << cyan << "DU2DT2 matrix : " << assembler.RHS << " seconds" << reset << std::endl;
            
//                Matrix<RealType, Dynamic, 1> lhs =
//                         + (2.0/(dt*dt))*(alpha_dof_n)
//                         - (1.0/(dt*dt))*(alpha_dof_n_m);
            assembler.RHS -= res;
            
        //        std::cout << bold << cyan << "F matrix : " << assembler.RHS << " seconds" << reset << std::endl;
        //        assembler.RHS =
        //        - (2.0/(dt*dt))*(mass_assembler.LHS * alpha_dof_n)
        //        + (1.0/(dt*dt))*(mass_assembler.LHS * alpha_dof_n_m);
        //        std::cout << bold << red << "F matrix : " << assembler.RHS << " seconds" << reset << std::endl;
                tc.toc();

                std::cout << bold << cyan << "Assembly completed: " << tc << " seconds" << reset << std::endl;
                

                
                tc.tic();
                SparseLU<SparseMatrix<RealType>> analysis;
                analysis.analyzePattern(assembler.LHS);
                analysis.factorize(assembler.LHS);
                a_dof_np = analysis.solve(assembler.RHS); // new acceleration
                tc.toc();

                // update scalar and rate
                p_dof_n += beta*dt*dt*a_dof_np;
                v_dof_n += gamma*dt*a_dof_np;
                a_dof_n  = a_dof_np;
            
                std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
                
                size_t cell_i = 0;
            #ifdef spatial_errors_Q
                tc.tic();
                RealType l2_error = 0.0;
                RealType H1_error = 0.0;

                for (auto &cell : msh.cells) {
                    
                    if(cell_i == 0){
                        RealType h = diameter(msh, cell);
                            std::cout << green << "h size = " << std::endl << h << std::endl;
                    }
                    
                    cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                    auto cell_dof = cell_basis.size();
                    Matrix<RealType, Dynamic, 1> cell_alpha_dof_n_p = p_dof_n.block(cell_i*cell_dof, 0, cell_dof, 1);
                    auto int_rule = integrate(msh, cell, 2.0*hho_di.cell_degree());
                    
                    // Error integrals
                    for (auto & point_pair : int_rule) {
                        
                        RealType omega = point_pair.second;
                        
                        // L2-error
                        auto t_phi = cell_basis.eval_basis( point_pair.first );
                        RealType u_exact = exact_sol_fun(point_pair.first);
                        RealType uh = cell_alpha_dof_n_p.dot( t_phi );
                        RealType error = u_exact - uh;
                        l2_error += omega * error * error;
                        
                        // H1-error
                        auto t_dphi = cell_basis.eval_gradients( point_pair.first );
                        Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();

                        for (size_t i = 1; i < cell_dof; i++ ){
                            grad_uh += cell_alpha_dof_n_p(i) * t_dphi.block(i, 0, 1, 2);
                        }

                        Matrix<RealType, 1, 2> grad_u_exact = exact_grad_sol_fun(point_pair.first);
                        H1_error += omega * (grad_u_exact - grad_uh).dot(grad_u_exact - grad_uh);
                        
                        
                    }
                    cell_i++;
                }
                

                std::cout << green << "L2-norm error = " << std::endl << std::sqrt(l2_error) << std::endl;
                std::cout << green << "H1-norm error = " << std::endl << std::sqrt(H1_error) << std::endl;
                std::cout << std::endl;
                tc.toc();

                std::cout << bold << cyan << "Error completed: " << tc << " seconds" << reset << std::endl;
                
            #endif
                
                auto num_cells = msh.cells.size();
                std::vector<RealType> exact_u, approx_u;
                exact_u.reserve( num_cells );
                approx_u.reserve( num_cells );
                cell_i = 0;
                for (auto& cell : msh.cells)
                {

                    auto bar = barycenter(msh, cell);
                    exact_u.push_back( exact_sol_fun(bar) );
                    
                    cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                    auto cell_dof = cell_basis.size();
                    Matrix<RealType, Dynamic, 1> cell_alpha_dof_n_p = p_dof_n.block(cell_i*cell_dof, 0, cell_dof, 1);
                    auto t_phi = cell_basis.eval_basis( bar );
                     RealType uh = cell_alpha_dof_n_p.dot( t_phi );
                    approx_u.push_back(uh);
                    cell_i++;
                }
                
                silo_database silo;
                std::string silo_file_name = "scalar_wave_" + std::to_string(it) + ".silo";
                silo.create(silo_file_name.c_str());
                silo.add_mesh(msh, "mesh");
                        silo.add_variable("mesh", "u", exact_u.data(), exact_u.size(), zonal_variable_t);
                silo.add_mesh(msh, "mesh");
                        silo.add_variable("mesh", "uh", approx_u.data(), approx_u.size(), zonal_variable_t);
                silo.close();
                
        //        Matrix<RealType, Dynamic, 1> du2dt2 =
        //                - (1.0/(dt*dt))*(alpha_dof_n_p)
        //                + (2.0/(dt*dt))*(alpha_dof_n)
        //                - (1.0/(dt*dt))*(alpha_dof_n_m);
        //
        //                std::cout << bold << cyan << "F approx: " << mass_assembler.LHS*du2dt2 << " seconds" << reset << std::endl;
                
        //        std::cout << yellow << "Solution at n-1 : " << std::endl << alpha_dof_n_m << reset << std::endl;
        //
        //        std::cout << yellow << "Solution at n : " << std::endl << alpha_dof_n << reset << std::endl;
        //
        //        std::cout << yellow << "Solution at n+1 : " << std::endl << alpha_dof_n_p << reset << std::endl;
                
//                // update states
//                alpha_dof_n_m = alpha_dof_n;
//                alpha_dof_n = alpha_dof_n_p;
            }
    }

    return 0;
}



////////////////////////// @omar::Ending:: simple fitted implementation /////////////////////////////////////////////
