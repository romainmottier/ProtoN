/*
 *       /\        Matteo Cicuttin (C) 2017,2018; Guillaume Delay 2018,2019
 *      /__\       matteo.cicuttin@enpc.fr        guillaume.delay@enpc.fr
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

#include "tbb/tbb.h"
#define HAVE_INTEL_TBB



template < typename Fonction, typename Mesh, typename Vel_Field , typename FiniteSpace , typename Transport_Method , typename T = typename Mesh::coordinate_type >
void
run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST_NEW_DIRICHLET_COND_NEW_LS(const Mesh & msh, const FiniteSpace& fe_data, Fonction & phi , Vel_Field& u , const Transport_Method& method ,  T& dt )
{
    // Starting time for FE calculation
    std::cout<<"----------- STARTING TRANSPORT PROBLEM LOW ORDER (NEW INLET COND) -----------"<<std::endl;
    
    timecounter tc;
    tc.tic();

    size_t degree = fe_data.order; // finite element order
    size_t dim = fe_data.ndof_FE ;
    size_t local_ndof = fe_data.local_ndof ; // local degrees of freedom
    auto S_i = fe_data.S_i;


    // SAVING PHI AND VELOCITY COEFFS
    auto phi_FEM = phi.sol_FEM ;
    auto u0 = u.sol_FEM.first ;
    auto u1 = u.sol_FEM.second ;
    auto u0_cellwise = u.sol_HHO.first ;
    auto u1_cellwise = u.sol_HHO.second ;
//    auto phi_exact_FEM = phi_exact.sol_FEM ;


    // SAVING OF USEFUL MATRICES
    Matrix<T, Dynamic, 1> global_lumped_mass = method.Global_Mass_Lumped;

    auto global_cij_x = method.Global_c_term_x ;
    auto global_cij_y = method.Global_c_term_y ;
    auto local_vandermonde = method.local_vandermonde ;

    auto cij_norm = method.cij_norm ;
    auto nij0 = method.nij0 ;
    auto nij1 = method.nij1 ;

    auto cji_norm = method.cij_norm ;
    auto nji0 = method.nji0 ;
    auto nji1 = method.nji1 ;





    // VANDERMONDE MATRIX INTERPOLATION
    size_t i_fl = 0 ;
    Matrix<T, Dynamic, 1> flux0_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    Matrix<T, Dynamic, 1> flux1_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;

    Matrix<T, Dynamic, 1> flux0 = Matrix<T, Dynamic, 1>::Zero(dim) ;
    Matrix<T, Dynamic, 1> flux1 = Matrix<T, Dynamic, 1>::Zero(dim) ;



    CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > cod(local_vandermonde);

    for(auto& cl : msh.cells)
    {
        // FLUX TERM : flux is a pair flux0 and flux1
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree );
        for (size_t i = 0; i < local_ndof ; i++)
        {
            flux0_loc(i) = u0_cellwise(i,i_fl) * phi(pts[i] , msh , cl );
            flux1_loc(i) = u1_cellwise(i,i_fl) * phi(pts[i] , msh , cl );

        }

        Matrix<T, Dynamic, 1> sol0 = cod.solve(flux0_loc);
        Matrix<T, Dynamic, 1> sol1 = cod.solve(flux1_loc);
        if (cod.info() != Success)
        {
            std::cout<<"Not positive"<<std::endl;
            assert(0);
        }

        for (size_t i = 0; i < local_ndof ; i++)
        {

            size_t asm_map =  phi.connectivity_matrix[i_fl][i].first ;
            flux0(asm_map) = sol0(i) ;
            flux1(asm_map) = sol1(i) ;

        }


        i_fl++;
    }




    // CONVOLUTION TERM
//    Matrix<T, Dynamic, 1> conv_global = global_cij_x * flux0  + global_cij_y * flux1 ;
    Matrix<T, Dynamic, 1> conv_global = (flux0.transpose()*global_cij_x + flux1.transpose()*global_cij_y).transpose();


    // TERM d_ij
    SparseMatrix<T> dij = SparseMatrix<T>( dim , dim );
    std::vector< Triplet<T> >   triplets_dij;
    Matrix<T, Dynamic, 1> term_dij_no_entropy =  Eigen::Matrix<T,Dynamic,1>::Zero(dim, 1);

    size_t counter_row = 0;
    for(auto& row_i:S_i)
    {
        T sum_row = 0.0 ;
        for(auto& elem:row_i)
        {
            T value0 = std::abs( u0(counter_row) * nij0.coeff(counter_row,elem) + u1(counter_row) * nij1.coeff(counter_row,elem) );
            T value1 = std::abs( u0(elem) * nij0.coeff(counter_row,elem) + u1(elem) * nij1.coeff(counter_row,elem) );
            T value = std::max(value0 , value1);

            T value_adj0 = std::abs( u0(counter_row) * nji0.coeff(counter_row,elem) + u1(counter_row) * nji1.coeff(counter_row,elem) );
            T value_adj1 = std::abs( u0(elem) * nji0.coeff(counter_row,elem) + u1(elem) * nji1.coeff(counter_row,elem) );
            T value_adj = std::max(value_adj0 , value_adj1);

            T lambda_max = value * cij_norm.coeff(counter_row,elem) ;
            T lambda_max_adj = value_adj * cji_norm.coeff(counter_row,elem) ;

            T val_dij = std::max( lambda_max , lambda_max_adj );

            if( counter_row == elem )
                val_dij = 0.0 ;

            sum_row += val_dij ;


            if( counter_row != elem ){
                triplets_dij.push_back( Triplet<T>(counter_row, elem, val_dij ) );
                term_dij_no_entropy(counter_row) += val_dij*(phi_FEM(elem)-phi_FEM(counter_row));
            }


        }
        triplets_dij.push_back( Triplet<T>(counter_row, counter_row, -sum_row ) );
        counter_row++;


    }

    dij.setFromTriplets( triplets_dij.begin(), triplets_dij.end() );
    triplets_dij.clear();



    // CHECK TIME STEP dt
    T CFL_numb = time_step_CFL_L2_velocity_NEW( dij.diagonal() , global_lumped_mass , fe_data.Dirichlet_boundary_inlet , dt );

    
    ///********* RESOLUTION OF THE SYSTEM: **********//


    // RESOLUTION FIRST ORDER
    Matrix<T, Dynamic, 1> phi_L = phi_FEM - dt * conv_global.cwiseQuotient(global_lumped_mass)  + dt * term_dij_no_entropy.cwiseQuotient(global_lumped_mass);


    // IMPOSITION DIRICHLET BOUNDARY CONDITIONS -> NO INLET BDRY CONDITIONS! (IT IS A NEUMANN BDRY PROBLEMS)
    
//    size_t counter_dir = 0 ;
//    for (const auto& dir_elem : fe_data.Dirichlet_boundary_inlet )
//    {
//        if(dir_elem){
//            phi_L(counter_dir) = phi_exact_FEM(counter_dir) ;
//        }
//        counter_dir++ ;
//    }







    // SAVING AND UPLOAD phi_L  INTO CLASS projected_level_set
    phi.sol_FEM = phi_L ;
    phi.converting_into_HHO_formulation(phi_L);



    tc.toc();
   

    std::cout<<"----------- FINE TRANSPORT PROBLEM -----------"<<std::endl;



}
