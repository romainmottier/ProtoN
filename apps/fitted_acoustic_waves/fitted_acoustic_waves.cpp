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

int main(int argc, char **argv)
{
    size_t k_degree = 0;
    size_t n_divs   = 0;
    
    int opt;
    while ( (opt = getopt(argc, argv, "k:l:")) != -1 )
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
    using RealType = double;

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
    
    size_t nt = 10;
    RealType ti = 0.0;
    RealType tf = 2.0/std::sqrt(2.0);
    RealType dt = tf/nt;

    for(size_t it = 0; it < nt; it++){
        
    std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
        
    // Manufactured solution
        RealType t = dt*it+ti;
        auto rhs_fun = [&t](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
            return 2.0 * M_PI * M_PI * std::cos(std::sqrt(2.0)*M_PI*t)  * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        };

        auto exact_sol_fun = [&t](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
            return std::cos(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        };
    #ifdef spatial_errors_Q
        auto exact_grad_sol_fun = [&t](const typename poly_mesh<RealType>::point_type& pt) -> Matrix<RealType, 1, 2> {
            Matrix<RealType, 1, 2> grad;
            grad(0,0) = std::cos(std::sqrt(2.0)*M_PI*t) * M_PI * std::cos(M_PI*pt.x()) * std::sin(M_PI*pt.y());
            grad(0,1) = std::cos(std::sqrt(2.0)*M_PI*t) * M_PI * std::sin(M_PI*pt.x()) * std::cos(M_PI*pt.y());
            return grad;
        };
    #endif
        
        // Creating HHO approximation spaces and corresponding linear operator
        hho_degree_info hho_di(k_degree,k_degree); // avoiding stabilization term
        
        auto assembler = make_assembler(msh, hho_di);
        tc.tic();
        for (auto& cell : msh.cells)
        {
            auto reconstruction_operator = make_hho_laplacian(msh, cell, hho_di);
            auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
            
            Matrix<RealType, Dynamic, Dynamic> laplacian_loc = reconstruction_operator.second + stabilization_operator;
            Matrix<RealType, Dynamic, 1> f_loc = make_rhs(msh, cell, hho_di.cell_degree(), rhs_fun);
            assembler.assemble(msh, cell, laplacian_loc, f_loc, exact_sol_fun);
        }
        assembler.finalize();
        tc.toc();

        std::cout << bold << cyan << "Assembly completed: " << tc << " seconds" << reset << std::endl;
        
        Matrix<RealType, Dynamic, 1> alpha_dof;
        
        tc.tic();
        SparseLU<SparseMatrix<RealType>> analysis;
        analysis.analyzePattern(assembler.LHS);
        analysis.factorize(assembler.LHS);
        alpha_dof = analysis.solve(assembler.RHS);
        tc.toc();

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
            Matrix<RealType, Dynamic, 1> cell_alpha_dof = alpha_dof.block(cell_i*cell_dof, 0, cell_dof, 1);
            auto int_rule = integrate(msh, cell, 2.0*hho_di.cell_degree());
            
            // Error integrals
            for (auto & point_pair : int_rule) {
                
                RealType omega = point_pair.second;
                
                // L2-error
                auto t_phi = cell_basis.eval_basis( point_pair.first );
                RealType u_exact = exact_sol_fun(point_pair.first);
                RealType uh = cell_alpha_dof.dot( t_phi );
                RealType error = u_exact - uh;
                l2_error += omega * error * error;
                
                // H1-error
                auto t_dphi = cell_basis.eval_gradients( point_pair.first );
                Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cell_dof; i++ ){
                    grad_uh += cell_alpha_dof(i) * t_dphi.block(i, 0, 1, 2);
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
            Matrix<RealType, Dynamic, 1> cell_alpha_dof = alpha_dof.block(cell_i*cell_dof, 0, cell_dof, 1);
            auto t_phi = cell_basis.eval_basis( bar );
             RealType uh = cell_alpha_dof.dot( t_phi );
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

    return 0;
}

////////////////////////// @omar::Ending:: simple fitted implementation /////////////////////////////////////////////
