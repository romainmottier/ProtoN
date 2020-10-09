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


template<typename T, typename Mesh>
bool
pt_in_cell(const Mesh& msh, const point<T,2>& point_to_find, const typename Mesh::cell_type& cl)
{
    auto pts =points(msh,cl);
 
    //std::cout<<"Point to find "<<std::setprecision(15)<<point_to_find.x()<<", "<<point_to_find.y()<<std::endl;

   // std::cout<<"Min x "<<std::setprecision(15)<<pts[0].x()<<", max x "<<pts[1].x()<<std::endl;

    //std::cout<<"Min y "<<std::setprecision(15)<<pts[1].y()<<", max y "<<pts[2].y()<<std::endl;

    T epsilon = 1e-10;
     if( (pts[0].x()-epsilon)<=point_to_find.x() && (pts[1].x()+epsilon)>=point_to_find.x() && (pts[1].y()-epsilon)<=point_to_find.y() && (pts[2].y()+epsilon)>=point_to_find.y() )
         return TRUE;
    else
        return FALSE;
  
}

template<typename T, typename Mesh>
size_t
pt_in_subcell(const Mesh& msh, const point<T,2>& point_to_find, const typename Mesh::cell_type& agglocl)
{
    
    for (auto& offset_subcells : agglocl.user_data.offset_subcells)
    {
        //std::cout<<"OFFSET ORIGINAL CELL "<<offset_subcells<<std::endl;
        auto cl = msh.cells[offset_subcells];
        if( pt_in_cell(msh,point_to_find,cl) )
            return offset_subcells;
    }
    // IF IT ARRIVES HERE, IT DIDN'T FIND THE POINT IN THE CELL.
    std::cout<<"the point did not find is "<<point_to_find<<std::endl;
    std::cout<<"IT DIDN'T FIND THE POINT IN SUBCELL "<<offset(msh,agglocl)<<std::endl;
    throw std::invalid_argument("Invalid point-> NOT IN AGGLO_CELL");

}





template<typename Mesh, typename T = typename Mesh::coordinate_type>
Matrix<T, Dynamic, 1>
make_bernstein_local_mass_matrix_lumped(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree, size_t di = 2)
{
    
    //auto f_neigh = cl.user_data.f_neighbors;
    //auto d_neigh = cl.user_data.d_neighbors;
    
    cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs, 1);
    //Matrix<T, Dynamic, 1> ret2 = Matrix<T, Dynamic, 1>::Zero(cbs, 1);

    auto qps = integrate(msh, cl, (degree)+2); // integration of order 2k

    for (auto& qp : qps)
    {
        auto phi = cb.eval_basis(qp.first);
        ret += qp.second * phi ;
        // phi * phi.transpose is (degree+1)^2 x (degree+1)^2 ; qp.second is a scalar
    }
    
    /*
    Matrix<T, Dynamic, 1> ret2 = Matrix<T, Dynamic, 1>::Zero(cbs, 1);
    //Matrix<T, Dynamic, 1> ret2 = Matrix<T, Dynamic, 1>::Zero(cbs, 1);

    auto qps2 = integrate(msh, cl, (degree)+2); // integration of order 2k

    for (auto& qp : qps2)
    {
        auto phi = cb.eval_basis(qp.first);
        ret2 += qp.second * phi ;
        // phi * phi.transpose is (degree+1)^2 x (degree+1)^2 ; qp.second is a scalar
    }
    
    std::cout<<"CHECKING AMOUNT QUADRATURE POINTS IN LOCAL MASS LUMPED"<<'\n'<<ret-ret2<<'\n'<<std::endl;
    std::cout<<"FINE CHECKING AMOUNT QUADRATURE POINTS IN LOCAL MASS LUMPED"<<std::endl;
    
    
    //ret2  =  ret.rowwise().sum(); // sum row; i.e. mi0 + mi1 + mi2 + mi3 , with i = 0 : 3
    */
    return ret;
}







template<typename Mesh, typename T = typename Mesh::coordinate_type>
Matrix<T, Dynamic, Dynamic>
make_bernstein_local_mass_matrix(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree, size_t di = 1)
{
    
    //auto f_neigh = cl.user_data.f_neighbors;
    //auto d_neigh = cl.user_data.d_neighbors;
    
    cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    //Matrix<T, Dynamic, 1> ret2 = Matrix<T, Dynamic, 1>::Zero(cbs, 1);

    auto qps = integrate(msh, cl, 2*(degree+di)); // integration of order 2k

    for (auto& qp : qps)
    {
        auto phi = cb.eval_basis(qp.first);
        ret += qp.second * phi * phi.transpose();
        // phi * phi.transpose is (degree+1)^2 x (degree+1)^2 ; qp.second is a scalar
    }
    //ret2  =  ret.rowwise().sum(); // sum row; i.e. mi0 + mi1 + mi2 + mi3 , with i = 0 : 3
    
    // CHECK INTEGRATION
    /*
    auto qps2 = integrate(msh, cl, 2*(degree)); // integration of order 2k
    Matrix<T, Dynamic, Dynamic> ret2 = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    for (auto& qp : qps2)
    {
        auto phi = cb.eval_basis(qp.first);
        ret2 += qp.second * phi * phi.transpose();
        // phi * phi.transpose is (degree+1)^2 x (degree+1)^2 ; qp.second is a scalar
    }
    std::cout<<"CHECKING AMOUNT QUADRATURE POINTS IN LOCAL MASS MATRIX"<<'\n'<<ret-ret2<<'\n'<<std::endl;
     std::cout<<"FINE CHECKING AMOUNT QUADRATURE POINTS IN LOCAL MASS MATRIX"<<std::endl;
    */
    return ret;
}


template<typename Mesh, typename T = typename Mesh::coordinate_type>
std::pair< Matrix<T, Dynamic, Dynamic> , Matrix<T, Dynamic, Dynamic> >
make_lagrangian_local_cij_matrix(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree, size_t di = 1)
{
    cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, Dynamic> ret0 = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    Matrix<T, Dynamic, Dynamic> ret1 = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    
    // for Q1 , degree = 1-> integration of order 2
    auto qps = integrate(msh, cl, 2*(degree+di) ); // integration of order 2k

    for (auto& qp : qps)
    {
        auto phi = cb.eval_basis(qp.first);
        std::cout<< "phi "<<'\n'<<phi<<std::endl;
        auto phi_grad = cb.eval_gradients(qp.first);
        std::cout<< "phi_grad "<<'\n'<<phi_grad<<std::endl;
        ret0 += qp.second * phi * ((phi_grad).col(0)).transpose();
    
        ret1 += qp.second * phi * ((phi_grad).col(1)).transpose();
    }
    
    
    return std::make_pair(ret0,ret1);
}

template<typename Mesh,typename Velocity , typename T = typename Mesh::coordinate_type>
std::pair< Matrix<T, Dynamic, Dynamic> , Matrix<T, Dynamic, Dynamic> >
make_bernstein_local_cij_matrix_with_velocity(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree,  Velocity& u ,size_t di = 1 )
{
    cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, Dynamic> ret0 = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    Matrix<T, Dynamic, Dynamic> ret1 = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    
    // for Q1 , degree = 1-> integration of order 2
    auto qps = integrate(msh, cl, 2*(degree+di) ); // integration of order 2k

    for (auto& qp : qps)
    {
        auto phi = cb.eval_basis(qp.first);
        //std::cout<< "phi "<<'\n'<<phi<<std::endl;
        auto phi_grad = cb.eval_gradients(qp.first);
        auto u_val = u(qp.first , msh , cl );
        //std::cout<< "phi_grad "<<'\n'<<phi_grad<<std::endl;
        ret0 += qp.second * u_val.first * phi * ((phi_grad).col(0)).transpose();
    
        ret1 += qp.second * u_val.second *phi * ((phi_grad).col(1)).transpose();
    }
    
    
    return std::make_pair(ret0,ret1);
}

template<typename Mesh, typename T = typename Mesh::coordinate_type>
std::pair< Matrix<T, Dynamic, Dynamic> , Matrix<T, Dynamic, Dynamic> >
make_bernstein_local_cij_matrix(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree, size_t di = 1)
{
    cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, Dynamic> ret0 = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    Matrix<T, Dynamic, Dynamic> ret1 = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    
    // for Q1 , degree = 1-> integration of order 2
    auto qps = integrate(msh, cl, 2*(degree+di) ); // integration of order 2k

    for (auto& qp : qps)
    {
        auto phi = cb.eval_basis(qp.first);
        
        auto phi_grad = cb.eval_gradients(qp.first);
        
        ret0 += qp.second * phi * ((phi_grad).col(0)).transpose();
    
        ret1 += qp.second * phi * ((phi_grad).col(1)).transpose();
    }
    
    /*
    // CHECKING ORDER INTEGRATION
    Matrix<T, Dynamic, Dynamic> ret1_bis = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    
    // for Q1 , degree = 1-> integration of order 2
    auto qps2 = integrate(msh, cl, 2*(degree+di)); // integration of order 2k

    for (auto& qp : qps2)
    {
        auto phi = cb.eval_basis(qp.first);
        
        auto phi_grad = cb.eval_gradients(qp.first);
    
        ret1_bis += qp.second * phi * ((phi_grad).col(1)).transpose();
    }
    
    std::cout<<"THE CHECKING FOR CIJ ORDER IS "<<'\n'<<ret1 - ret1_bis<<std::endl;
    */
    return std::make_pair(ret0,ret1);
}

template<typename Mesh, typename T = typename Mesh::coordinate_type ,  typename Fonction >
std::pair<Matrix<T, Dynamic, 1>,Matrix<T, Dynamic, 1> >
make_bernstein_local_RHS_VEC(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree , const Fonction& f , size_t di = 0)
{
    
    cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    std::pair<Matrix<T, Dynamic, 1>,Matrix<T, Dynamic, 1> > ret = std::make_pair( Matrix<T, Dynamic, 1>::Zero(cbs, 1) , Matrix<T, Dynamic, 1>::Zero(cbs, 1) );
    
    auto qps = integrate(msh, cl, 2*(degree+di)); // integration of order 2k

    for (auto& qp : qps)
    {
        auto b = cb.eval_basis(qp.first);
        auto value0 = f(qp.first,msh,cl).first;
        auto value1 = f(qp.first,msh,cl).second;
        ret.first += qp.second * value0 * b ;
        ret.second += qp.second * value1 * b ;
    }
    
    return ret;
}


template< typename Mesh , typename Fonction , typename FiniteSpace , typename T = typename Mesh::coordinate_type  >
struct L2projected_level_set_high_order: public level_set<T>
{
     
    bool analytic_check = FALSE ;
    
    T phi_max , phi_min ;
    size_t  last_row_init, last_row_end, number_faces_one_row;
    
    T iso_val_interface = 0.0 ;
    Mesh msh; // Original mesh, NOT agglomerated.
    mesh_init_params<T> params; // mesh parameter
       
    size_t degree_FEM ; // FEM degree
    size_t n_cls ; // #cells
    size_t local_dim; // Local Dimension (degree_FEM+1)*(degree_FEM+1)
    size_t n_vertices ; // #vertices
    size_t      Nx, Ny ; // Number of cells in x and y direciton
    
    // connectivity matrix : for each cell, it stores the global numbering
    std::vector< std::vector<std::pair<size_t,bool>>> connectivity_matrix ;
    std::vector<std::set<size_t>> S_i;
    //std::vector<std::vector<size_t>> connectivity_matrix ;
       
    size_t dim_HHO; // Global dimension Discontinuous framework = Local dimension * #cells
    size_t ndof_FE; // Global dimension FE continuous = #nodes
       
    int mapped = 0 ; // = 0 not mapped ; = 1 mapped ; = 2 inverse mapping
    
    SparseMatrix<T>                 Global_Mass; // Global mass, saved for FEM problem
       
    Eigen::Matrix<T, Dynamic, Dynamic> sol_HHO ; // projection saved in HHO format: cell by cell
    Matrix<T, Dynamic, 1> sol_FEM ; // projection saved in Continuos FE format: global nodes
    Eigen::Matrix<T, Dynamic, 1> vertices; // saving level_set on vertices mesh

    // Local Vandermonde Matrix for interpolation
    Matrix<T, Dynamic, Dynamic> local_vandermonde ;
       
       
    // Assembling into global matrix
    SparseMatrix<T>                 Global_c_term_x; // Global mass, saved for FEM problem
    SparseMatrix<T>                 Global_c_term_y; // Global mass, saved for FEM problem
    
    Matrix<T, Dynamic, 1>      Global_Mass_Lumped; // Global mass, saved for FEM problem

    SparseMatrix<T>         cij_norm , nij0 , nij1 ;
    SparseMatrix<T>         cji_norm , nji0 , nji1 ;
    
    
    Eigen::Matrix<T, Dynamic, Dynamic> normal_c_HHO_0 ;
    Eigen::Matrix<T, Dynamic, Dynamic> normal_c_HHO_1 ;
    Eigen::Matrix<T, Dynamic, 1> normal_c_FEM_0 ;
    Eigen::Matrix<T, Dynamic, 1> normal_c_FEM_1 ;
    
    Eigen::Matrix<T, Dynamic, Dynamic> gradient_c_HHO_0 ;
    Eigen::Matrix<T, Dynamic, Dynamic> gradient_c_HHO_1 ;
    Eigen::Matrix<T, Dynamic, 1> gradient_c_FEM_0 ;
    Eigen::Matrix<T, Dynamic, 1> gradient_c_FEM_1 ;
    
    L2projected_level_set_high_order(const FiniteSpace& fe_data , const Fonction & level_set, const Mesh & msh , bool analytic_check = FALSE )
        : degree_FEM(fe_data.order) , local_dim(fe_data.local_ndof), msh(msh), Nx(fe_data.Nx),Ny(fe_data.Ny), params(fe_data.params) , dim_HHO(fe_data.ndof_disc) , n_cls(fe_data.n_cls) ,n_vertices(fe_data.n_vertices) , connectivity_matrix(fe_data.connectivity_matrix) , ndof_FE(fe_data.ndof_FE) , analytic_check(analytic_check)
    {
        if(!analytic_check)
        {
            timecounter tc_level_set;
            tc_level_set.tic();
            Matrix<T, Dynamic, 1>           RHS;    // Known term
            std::vector< Triplet<T> >       triplets; // Position elements: Sparse Matrix Notation
            std::vector< Triplet<T> >       triplets_c_term_x; // Position elements: Sparse Matrix Notation
            std::vector< Triplet<T> >       triplets_c_term_y; // Position elements: Sparse Matrix Notation
            
            
            last_row_init = Ny*(2*Nx+1); // There are 2 faces for each row of cells + Ny
            last_row_end = last_row_init + Nx-1;
            number_faces_one_row = 2*Nx+1; // for each cell I count only the low and sx faces, respectevely 0-1 2-3 4-5 6-7 8 + the last on the right boundary
              
            Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero(local_dim);
             
            Global_Mass = SparseMatrix<T>( ndof_FE, ndof_FE ); //(b_i,b_j)_ij , b_i Lagrange basis fx
            RHS = Matrix<T, Dynamic, 1>::Zero( ndof_FE ); // Known term (f,b_i)_i , b_i Lagrange basis fx
              
            Global_c_term_x = SparseMatrix<T>( ndof_FE, ndof_FE ); //(b_i,b_j)_ij , b_i Lagrange basis fx
            Global_c_term_y = SparseMatrix<T>( ndof_FE, ndof_FE ); //(b_i,b_j)_ij , b_i Lagrange basis fx
               
            Global_Mass_Lumped = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
               
            local_vandermonde = Matrix<T, Dynamic, Dynamic>::Zero ( local_dim,local_dim );
            
            normal_c_HHO_0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
            normal_c_HHO_1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
            normal_c_FEM_0 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            normal_c_FEM_1 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            
            gradient_c_HHO_0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls);
            gradient_c_HHO_1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls);
            gradient_c_FEM_0 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            gradient_c_FEM_1 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            
            
            
            cij_norm = SparseMatrix<T>( ndof_FE, ndof_FE );
            cji_norm = SparseMatrix<T>( ndof_FE, ndof_FE );
            nij0 = SparseMatrix<T>( ndof_FE, ndof_FE );
            nij1 = SparseMatrix<T>( ndof_FE, ndof_FE );
            nji0 = SparseMatrix<T>( ndof_FE, ndof_FE );
            nji1 = SparseMatrix<T>( ndof_FE, ndof_FE );
            /*
            std::vector< Triplet<T> >       triplets_norm;
            std::vector< Triplet<T> >       triplets_norm_adj;
            std::vector< Triplet<T> >       triplets_nij0;
            std::vector< Triplet<T> >       triplets_nij1;
            std::vector< Triplet<T> >       triplets_nji0;
            std::vector< Triplet<T> >       triplets_nji1;
            */
              
            // Saving the projection in HHO discontinuous format (MATRIX NOTATION)
            sol_HHO = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
            sol_FEM = Matrix<T, Dynamic, 1>::Zero(ndof_FE);
            vertices = Eigen::Matrix<T, Dynamic, 1>::Zero( n_vertices , 1 );
              
            //Eigen::Matrix<T, Dynamic, Dynamic> sol_HHO_vandermonde =  Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls ) ;
              
              
            CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > cod;
            //FullPivLU<Matrix<T, Dynamic, Dynamic > > cod_2;
              
              
            std::cout<<"----> In 'L2projected_level_set_high_order': Vandermonde interpolation of the level set with BERNSTEIN basis."<<std::endl;
            
            //std::cout<<"CHECK INVERSION LOCAL VANDERMONDE"<<std::endl;
            for( const auto& cl : msh.cells )
            {
                size_t cell_offset = offset(msh, cl) ;
                
                if(cell_offset == 0)
                {
                    auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree_FEM);
                    cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
                    for (size_t i = 0; i < local_dim; i++)
                    {
                        // Local vandermonde matrix
                        local_vandermonde.block(i,0,1,local_dim) = (cb.eval_basis(pts[i])).transpose() ;
                    }
                      
                    cod.compute(local_vandermonde);
                    //cod_2.compute(local_vandermonde);
                }
                
             
                auto local_mass = make_bernstein_local_mass_matrix( msh, cl , degree_FEM );
                  
                //auto local_RHS = make_bernstein_local_RHS( msh , cl , degree_FEM , level_set );
                // Local c_ij = b_i nabla(b_j) -> USEFUL FOR TRANSPORT PROBLEM
                auto local_cij = make_bernstein_local_cij_matrix (msh, cl, degree_FEM);
                      
                auto local_mass_lumped = make_bernstein_local_mass_matrix_lumped( msh , cl , degree_FEM ) ;
                 
                  
                // Costruction of the coefficients of the Bernstein basis
                auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
                  
                  
                // Assembling triplets for global problem
                for (size_t i = 0; i < local_dim; i++)
                {
                    size_t asm_map_i = connectivity_matrix[cell_offset][i].first ;
                    for (size_t j = 0; j < local_dim; j++)
                    {
                        /*
                        T c_ij0 = local_cij.first(i,j) ;
                        T c_ij1 = local_cij.second(i,j) ;
                        
                        T c_ji0 = local_cij.first(j,i) ;
                        T c_ji1 = local_cij.second(j,i) ;
                        */
                        size_t asm_map_j = connectivity_matrix[cell_offset][j].first ;
                          
                        triplets.push_back(Triplet<T>(asm_map_i,asm_map_j, local_mass(i,j)));
                        triplets_c_term_x.push_back( Triplet<T>(asm_map_i, asm_map_j , local_cij.first(i,j) ) );
                        triplets_c_term_y.push_back( Triplet<T>( asm_map_i , asm_map_j , local_cij.second(i,j) ) );
                        
                        /*
                        T val_norm = sqrt( c_ij0*c_ij0 + c_ij1*c_ij1 );
                        T val_norm_adj = sqrt( c_ji0 *c_ji0 + c_ji1*c_ji1 );
                        T val_nij0 = c_ij0/val_norm ;
                        T val_nij1 = c_ij1/val_norm ;
                        T val_nji0 = c_ji0/val_norm_adj ;
                        T val_nji1 = c_ji1/val_norm_adj ;
                        
                        triplets_norm.push_back(Triplet<T>(asm_map_i,asm_map_j, val_norm));
                        triplets_norm_adj.push_back(Triplet<T>(asm_map_i,asm_map_j, val_norm_adj));
                        triplets_nij0.push_back(Triplet<T>(asm_map_i,asm_map_j, val_nij0));
                        triplets_nij1.push_back(Triplet<T>(asm_map_i,asm_map_j, val_nij1));
                        triplets_nji0.push_back(Triplet<T>(asm_map_i,asm_map_j, val_nji0));
                        triplets_nji1.push_back(Triplet<T>(asm_map_i,asm_map_j, val_nji1));
                        */
                    }
                    Global_Mass_Lumped(asm_map_i) += local_mass_lumped(i);
                    //Global_Mass_Lumped(asm_map[i].first) += local_mass_lumped(i);
                    //RHS(asm_map[i].first) += local_RHS(i) ;
                    RHS_vandermonde(i) = level_set( qps[i]) ;
                }
                  
                  
                auto sol_tmp = cod.solve(RHS_vandermonde) ;
                  
                //auto sol_tmp_2 = cod_2.solve(RHS_vandermonde) ;
                  
                sol_HHO.col(cell_offset) = sol_tmp ;
                for (size_t i = 0; i < local_dim; i++)
                {
                      
                    size_t asm_map =  connectivity_matrix[cell_offset][i].first ;
                    sol_FEM( asm_map ) = sol_HHO(i,cell_offset) ;
                      
                    //if( std::abs( sol_tmp(i) - sol_tmp_2(i) )>1e-14 )
                    //    std::cout<< std::abs(sol_tmp(i) - sol_tmp_2(i) ) <<std::endl;
                      
                }
                size_t i_vertex = cell_offset+floor(cell_offset/Nx);
                vertices(i_vertex) = sol_HHO(0,cell_offset) ;
                vertices(i_vertex+1) = sol_HHO(1,cell_offset) ;
                vertices(i_vertex+Nx+2) = sol_HHO(2,cell_offset) ;
                vertices(i_vertex+Nx+1) = sol_HHO(3,cell_offset) ;
                  
                  
            } // end of cl loop
                
            //std::cout<<"FINE CHECK INVERSION LOCAL VANDERMONDE"<<std::endl;
            // Finalisation global assembling
            Global_Mass.setFromTriplets( triplets.begin(), triplets.end() );
            triplets.clear();
                  
                  
            // Finalisation global assembling
            Global_c_term_x.setFromTriplets( triplets_c_term_x.begin(), triplets_c_term_x.end() );
            triplets_c_term_x.clear();
            Global_c_term_y.setFromTriplets( triplets_c_term_y.begin(), triplets_c_term_y.end() );
            triplets_c_term_y.clear();
                
            /*
            cij_norm.setFromTriplets( triplets_norm.begin(), triplets_norm.end() );
            triplets_norm.clear();
            cji_norm.setFromTriplets( triplets_norm_adj.begin(), triplets_norm_adj.end() );
            triplets_norm_adj.clear();
            
            nij0.setFromTriplets( triplets_nij0.begin(), triplets_nij0.end() );
            triplets_nij0.clear();
            nij1.setFromTriplets( triplets_nij1.begin(), triplets_nij1.end() );
            triplets_nij1.clear();
            
            nji0.setFromTriplets( triplets_nji0.begin(), triplets_nji0.end() );
            triplets_nji0.clear();
            nji1.setFromTriplets( triplets_nji1.begin(), triplets_nji1.end() );
            triplets_nji1.clear();
            */
            
            // NORM of c_ij
            
            cij_norm = ( Global_c_term_x.cwiseProduct(Global_c_term_x) + Global_c_term_y.cwiseProduct(Global_c_term_y) ).cwiseSqrt() ;
            //std::cout<<"cij norm "<<'\n'<<cij_norm<<std::endl;
            
            // MATRIX n_ij
            nij0 = Global_c_term_x.cwiseQuotient( cij_norm );
            nij1 = Global_c_term_y.cwiseQuotient( cij_norm );
            
            //std::cout<<"nij1  "<<'\n'<<nij1<<std::endl;
            

            // MATRIX c_ji
            SparseMatrix<T> cji_x = Global_c_term_x.adjoint() ;
            SparseMatrix<T> cji_y = Global_c_term_y.adjoint() ;
             
            // NORM of c_ji -> i.e. c_ij transposed
            cji_norm = (cji_x.cwiseProduct(cji_x)+cji_y.cwiseProduct(cji_y)).cwiseSqrt();
            
            // MATRIX n_ij (TRANSPOSED)
            nji0 = cji_x.cwiseQuotient( cji_norm );
            nji1 = cji_y.cwiseQuotient( cji_norm );
            

            
              
            //std::cout<<"local_vandermonde"<<'\n'<<local_vandermonde<<std::endl;
                  
              
              // CALCULATION OF THE SIZE + PLOTTING
              /*
              size_t size_supp_nodes = 0;
              std::cout<<"Supporting nodes IN L2:"<<std::endl;
              size_t jjjj = 0;
              for (auto& i: S_i) {
                  size_supp_nodes+=i.size();
                  std::cout <<"Node "<<jjjj<<":";
                  for (auto it=i.begin(); it != i.end(); ++it)
                      std::cout << ' ' << *it;
                      // std::cout<<ii;
                  std::cout<<'\n';
                  jjjj++;
              }
              std::cout<<std::endl;
              std::cout<<"Supporting nodes size:"<<size_supp_nodes<<std::endl;
              */
              
                  
              
                  
              //Matrix<T, Dynamic, 1> sol_FEM_vandermonde  = Matrix<T, Dynamic, 1>::Zero(RHS.rows()); ;
              
            std::cout<<"sol_FEM size "<<sol_FEM.size()<<std::endl;
            std::cout<<"local_dim size "<<local_dim<<std::endl;
            std::cout<<"n_cls "<<n_cls<<std::endl;
                  
              // Global solution saved as discontinuous HHO approach
              // Also saved min & max coefficients + position in HHO notation
              
              /*
              ConjugateGradient<SparseMatrix<T> > solver_global_mass;
              //SparseLU<SparseMatrix<T>, COLAMDOrdering<int> >solver_global_mass;
              //Notice: for this step the numerical values of A are not used
              //solver_global_mass.analyzePattern(Global_Mass);
              //solver_global_mass.factorize(Global_Mass);
               
              solver_global_mass.compute(Global_Mass); // SAVE INVERSE OF GLOBAL MASS
              if(solver_global_mass.info()!=Success) {
                  std::cout<<"FAILED SOLVER 0"<<std::endl;
                  return;
              }
              
              sol_FEM = solver_global_mass.solve(RHS);
              */
              
              /*
              for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
              {
                  for (size_t i = 0; i < local_dim; i++)
                  {
                      size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                      //sol_HHO(i,counter_bis) = sol_FEM( asm_map ) ;
                      sol_FEM( asm_map ) = sol_HHO(i,counter_bis) ;
                  }
              }
              */
              
              
              //std::cout<<"CHECK L2 proj vs Vandermonde interpolation for phi^0"<<'\n'<<sol_FEM-sol_FEM_vandermonde<<std::endl;
              
              
              //sol_FEM = sol_FEM_vandermonde ;
              //sol_HHO = sol_HHO_vandermonde ;
              //std::cout<<"CHECK L2 proj vs Vandermonde interpolation for phi^0"<<'\n'<<sol_FEM-sol_FEM_vandermonde<<std::endl;
              // Set of maximum and minimum
              
            timecounter tcbis ;
                      
            tcbis.tic();
            set_max_min();
            tcbis.toc();
            std::cout << bold << yellow << "--> set_max_min: t = " << tcbis << " seconds" << reset << std::endl;
              
              
              /*
              for( size_t i_global = 0; i_global < n_cls; i_global++)
              {
                  size_t i_vertex = i_global+floor(i_global/Nx);
                  vertices(i_vertex) = sol_HHO(0,i_global) ;
                  vertices(i_vertex+1) = sol_HHO(1,i_global) ;
                  vertices(i_vertex+Nx+2) = sol_HHO(2,i_global) ;
                  vertices(i_vertex+Nx+1) = sol_HHO(3,i_global) ;
              }
              */
             
            tc_level_set.toc();
            std::cout << bold << yellow << "INITIALISATION LEVEL SET: t = " << tc_level_set << " seconds" << reset << std::endl;

        }
        else
        {
            timecounter tc_level_set;
            tc_level_set.tic();
            
            Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero(local_dim);
            local_vandermonde = Matrix<T, Dynamic, Dynamic>::Zero ( local_dim,local_dim );
              
              
            // Saving the projection in HHO discontinuous format (MATRIX NOTATION)
            sol_HHO = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
            normal_c_HHO_0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
            normal_c_HHO_1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
            normal_c_FEM_0 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            normal_c_FEM_1 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            
            gradient_c_HHO_0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls);
            gradient_c_HHO_1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls);
            gradient_c_FEM_0 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            gradient_c_FEM_1 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            
            CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > cod;
            
            std::cout<<"----> FOR ANALYTIC CHECK 'L2projected_level_set_high_order': Vandermonde interpolation of the level set with BERNSTEIN basis."<<std::endl;
            
        
            for( const auto& cl : msh.cells )
            {
                size_t cell_offset = offset(msh, cl) ;
                
                if(cell_offset == 0)
                {
                    auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree_FEM);
                    cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
                    for (size_t i = 0; i < local_dim; i++)
                    {
                        // Local vandermonde matrix
                        local_vandermonde.block(i,0,1,local_dim) = (cb.eval_basis(pts[i])).transpose() ;
                    }
                      
                    cod.compute(local_vandermonde);
                }
                  
            
                
                // Assembling triplets for global problem
                auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
                for (size_t i = 0; i < local_dim; i++)
                    RHS_vandermonde(i) = level_set( qps[i]) ;
            
                  
                auto sol_tmp = cod.solve(RHS_vandermonde) ;
                sol_HHO.col(cell_offset) = sol_tmp ;
                
                
                  
            } // end of cl loop
                
            tc_level_set.toc();
            std::cout << bold << yellow << "INITIALISATION LEVEL SET: t = " << tc_level_set << " seconds" << reset << std::endl;

        }
  
    
    }
            
            

   
    L2projected_level_set_high_order()=default;
    
    
    void
    coefficients_mapping( )
    {

        
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                RHS_vandermonde(ct) =  ( (*this)(pt , msh , cl ) - phi_min )/( phi_max - phi_min );
                ct++;
            }
            
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
         
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map = connectivity_matrix[cell_offset][i].first ;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
        } // end of cl loop
       
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        
        std::cout<<"IN COEFFICIENT_MAPPING, CHECKING phi = 0, after mappin becomes --> "<< (0.0 - phi_min )/( phi_max - phi_min )<<std::endl;
               
        std::cout<<"It should be close to 1/2, error = "<< 1./2. - ( 0.0 - phi_min )/( phi_max - phi_min ) <<std::endl;
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER MAPPING: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
        
    }
    
    void
    coefficients_mapping_MAX_MAX( )
    {

        
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                if( std::abs(phi_max) >= std::abs(phi_min))
                    RHS_vandermonde(ct) =  ( (*this)(pt , msh , cl )+ phi_max) /( 2.0*phi_max );
                else
                    RHS_vandermonde(ct) =  ( (*this)(pt , msh , cl )-phi_min ) /( -2.0*phi_min );
                
                ct++;
            }
            
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
         
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map = connectivity_matrix[cell_offset][i].first ;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
        } // end of cl loop
       
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        iso_val_interface = 1.0/2.0 ;
        
        std::cout<<"IN COEFFICIENT_MAPPING MAX MAX, CHECKING phi = 0, after mappin becomes --> "<< ( 0.0 +phi_max )/( 2.0*phi_max )<<std::endl;
        std::cout<<"It should be close to 1/2, error = "<< 1./2. - ( 0.0 + phi_max )/( 2.0*phi_max  ) <<std::endl;
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER MAPPING_MAX_MAX: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
        
    }
    
    
    
    void
    coefficients_sfasamento( )
    {

        
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        
     
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                RHS_vandermonde(ct) =   (*this)(pt , msh , cl ) + 0.5 ;
                
                ct++;
            }
            
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
            //std::cout<<"--------------->>>>>>>> RHS_vandermonde"<<'\n'<<RHS_vandermonde<<std::endl;
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map = connectivity_matrix[cell_offset][i].first ;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
        } // end of cl loop
       
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        iso_val_interface = 1.0/2.0 ;
        std::cout<<"Isovalue of the interface = "<<iso_val_interface<<std::endl;
        //std::cout<<"IN COEFFICIENT_MAPPING, CHECKING phi = 0, after mappin becomes --> "<< (0.0 - phi_min )/( phi_max - phi_min )<<std::endl;
               
        //std::cout<<"It should be close to 1/2, error = "<< 1./2. - ( 0.0 - phi_min )/( phi_max - phi_min ) <<std::endl;
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER MAPPING: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
        
    }
    
    
    
    
    
    void
    coefficients_mapping_quadratic( )
    {

        
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        
        T a1 = (-1.0/2.0 - phi_min/(phi_max-phi_min))/(pow(phi_min,2));
        T b = 1.0/(phi_max-phi_min);
        T c = 1.0/2.0;
        T a2 = (1.0/2.0 - phi_max/(phi_max-phi_min))/(pow(phi_max,2));
        
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                auto val = (*this)(pt , msh , cl ) ;
                if( val <= 0 )
                    RHS_vandermonde(ct) =   a1 * val * val + b * val + c ;
                else
                    RHS_vandermonde(ct) =   a2 * val * val + b * val + c ;
                    
                ct++;
            }
            
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
         
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map = connectivity_matrix[cell_offset][i].first ;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
        } // end of cl loop
       
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        iso_val_interface = 1.0/2.0 ;
        std::cout<<"Isovalue of the interface = "<<iso_val_interface<<std::endl;
        //std::cout<<"IN COEFFICIENT_MAPPING, CHECKING phi = 0, after mappin becomes --> "<< (0.0 - phi_min )/( phi_max - phi_min )<<std::endl;
               
        //std::cout<<"It should be close to 1/2, error = "<< 1./2. - ( 0.0 - phi_min )/( phi_max - phi_min ) <<std::endl;
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER MAPPING: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
        
    }
    
    void
    coefficients_inverse_mapping_quadratic( )
    {
        
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                RHS_vandermonde(ct) = phi_min + (*this)(pt , msh , cl )*( phi_max - phi_min );
                ct++;
            }
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
            
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map =  connectivity_matrix[cell_offset][i].first;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
            
        } // end of cl loop
        
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER MAPPING: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
           
    }
    
    
    void
    coefficients_inverse_mapping( )
    {
        
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                RHS_vandermonde(ct) = phi_min + (*this)(pt , msh , cl )*( phi_max - phi_min );
                ct++;
            }
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
            
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map =  connectivity_matrix[cell_offset][i].first;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
            
        } // end of cl loop
        
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER MAPPING: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
           
    }
        
    
    void
    coefficients_inverse_mapping_MAX_MAX( )
    {
        
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                if( std::abs(phi_max) >= std::abs(phi_min))
                    RHS_vandermonde(ct) = -phi_max + (*this)(pt , msh , cl )* ( 2.0 * phi_max );
                else
                    RHS_vandermonde(ct) = phi_min - (*this)(pt , msh , cl )* ( 2.0 * phi_min );
                
                ct++;
            }
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
            
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map =  connectivity_matrix[cell_offset][i].first;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
            
        } // end of cl loop
        
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        iso_val_interface = 0 ;
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER INVERSE_MAPPING_MAX_MAX: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
           
    }
        
        
        
    void  set_discrete_points( Eigen::Matrix<T, Dynamic, Dynamic>& values_new)
    {
        sol_HHO = values_new ;
        std::cout<<" --> set_discrete_points: check that sol_FEM already uploaded!"<<std::endl;
        
    }
        
        
    void converting_into_HHO_formulation( const Eigen::Matrix<T, Dynamic, 1>& values_new )
    {
        // SAVE BOTH SOL_HHO AND VERTICES
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                sol_HHO(i,counter_bis) = values_new( asm_map );
                
            }
            size_t i_vertex = counter_bis+floor(counter_bis/Nx);
            vertices(i_vertex) = sol_HHO(0,counter_bis) ;
            vertices(i_vertex+1) = sol_HHO(1,counter_bis) ;
            vertices(i_vertex+Nx+2) = sol_HHO(2,counter_bis) ;
            vertices(i_vertex+Nx+1) = sol_HHO(3,counter_bis) ;
                
        }
        std::cout<<" --> converting_into_HHO_formulation. TO BE CHECKED that sol_FEM already uploaded!"<<std::endl;
        
    }
        
    void converting_into_FE_formulation( const Eigen::Matrix<T, Dynamic, Dynamic>& values_new )
    {
          
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                sol_FEM( asm_map ) = values_new(i,counter_bis) ;
            }
            size_t i_vertex = counter_bis+floor(counter_bis/Nx);
            vertices(i_vertex) = values_new(0,counter_bis) ;
            vertices(i_vertex+1) = values_new(1,counter_bis) ;
            vertices(i_vertex+Nx+2) = values_new(2,counter_bis) ;
            vertices(i_vertex+Nx+1) = values_new(3,counter_bis) ;

        }
        std::cout<<" --> converting_into_FE_formulation. TO BE CHECKED that sol_HHO already uploaded!"<<std::endl;
       
    }
    
    
    
           
    void set_max_min()
    {
        
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        phi_max = ret0;
        phi_min = ret1;
        std::cout<<" --> set_max_min: LEVEL_SET: MAX IS "<<phi_max<< " , MIN IS "<<phi_min<<" . SI PUO TOGLIERE."<<std::endl;
    }
        
       
    // OK FINE, IT WORKS ALSO FOR AGGLOMERATED MESHES
    // It should work also for Bernstein Basis
    T operator()( const typename Mesh::node_type& node ) const
    {
        return vertices(node.ptid);
            
    }
        
     
    // OK FINE, IT WORKS ALSO FOR AGGLOMERATED MESHES -> BUT SLOW
    T operator()(const point<T,2>& pt) const
    {
        //std::cout<<"I AM IN OPERATOR() SLOW !!!!"<<std::endl;
        size_t counter=0;
            
        // It looks for in what cell the point is
        for( const auto& cl:msh.cells)
        {
            if( pt_in_cell<T,Mesh>(msh,pt,cl) )
            {
                //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
                cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
                    
                auto values_cell = (sol_HHO.block(0,counter,local_dim,1)).col(0);
               
                return values_cell.dot( cb.eval_basis(pt) );
                
            }
            counter+=1;
        }
        std::cout<<"IF HERE, THERE IS A PROBLEM IN projected_level_set::operator()!!!"<<std::endl;
        return 1e10; //to check if doesn't enter in the loop
    }

        
    // IT WORKS ONLY FOR NOT-AGGLOMERATED MESHES --> FAST
    T operator()( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
            
        size_t counter = offset(msh,cl) ;
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = (sol_HHO.block(0,counter,local_dim,1)).col(0);
        T tmp = values_cell.dot( cb.eval_basis(pt) );
           
        return tmp;
                
    }
        
    // IT WORKS ONLY FOR NOT-AGGLOMERATED MESHES --> FAST
    T operator()( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::face_type& fc ) const
    {
        auto counter_face = offset(msh,fc);
        size_t counter_cell;
        // ATTENTION, ALL THIS WORKS IN STRUCTURED QUADRANGULAR MESHES
               
        // Check if I am in the last row, upper faces (ordered differently)
        if(counter_face>=last_row_init && counter_face<=last_row_end)
        {
            counter_cell = (Ny-1)*Nx + counter_face%(last_row_init);
        }
        else
        {
            // Find in what row the face is
            auto  num_cell_row = floor(counter_face/(number_faces_one_row));
            if ( counter_face!= ( (2*Nx)*(num_cell_row+1)+num_cell_row ) )
            {
                // Faces not on the right boudary, to know in what cell are, it is sufficient to analyse the low and left face of each quadrangular cell.
                counter_cell = floor( (counter_face-num_cell_row)/2.0 );
            }
            else
            {
                // Face on the right boudary,
                counter_cell = ( num_cell_row+1 )*Nx -1;
            }
            
        }
        //std::cout<<"Face->Cell number "<<counter_cell<<std::endl;
        auto cl = msh.cells.at(counter_cell);
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = (sol_HHO.block(0,counter_cell,local_dim,1)).col(0);
        T tmp = values_cell.dot( cb.eval_basis(pt) );
        return tmp;

               
    }
        
        
    // IT WORKS FOR ALL THE MESHES --> SLOW
    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;

        ret = gradient(pt);
        return ret/ret.norm();
    }
        
         
    // IT WORKS FOR ALL THE MESHES --> SLOW
    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        size_t counter=0;
        //std::cout<<"I AM IN GRADIENT SLOW !!!!"<<std::endl;
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        for( const auto& cl:msh.cells)
        {
            if(pt_in_cell<T,Mesh>(msh,pt,cl))
            {
                cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
                //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
                    
                auto values_cell = sol_HHO.col(counter);
                auto grad_eval =  cb.eval_gradients(pt);
                ret(0) = values_cell.dot( grad_eval.col(0) );
                // std::cout<<"Value of derivative new along x"<<ret(0)<<std::endl;
                ret(1) = values_cell.dot( grad_eval.col(1) );

                return ret;
            }
            counter+=1;

        }
        std::cout<<"Se compare questo problema in gradient()"<<std::endl;
        ret(0)+=1e10;
        ret(1)+=1e10;
        return ret; //to check if doesn't enter in the loop

    }

         
        // IT WORKS FOR NOT-AGGLOMERATED MESHES --> FAST
    Eigen::Matrix<T,2,1> gradient( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        
        // MATRIX NOTATION
        size_t counter = offset(msh,cl);
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = sol_HHO.col(counter);
        auto grad_eval =  cb.eval_gradients(pt);
        ret(0) = values_cell.dot( grad_eval.col(0) );
        // std::cout<<"Value of derivative new along x"<<ret(0)<<std::endl;
        ret(1) = values_cell.dot( grad_eval.col(1) );
        //values_cell.dot( grad_eval.col(1) );
        // std::cout<<"Value of derivative new along y"<<ret(1)<<std::endl;
        return ret;
            
    }

     
    // IT WORKS FOR NOT-AGGLOMERATED MESHES --> FAST
    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt, const Mesh& msh, const typename Mesh::cell_type& cl) const
    {
        Eigen::Matrix<T,2,1> ret;
        ret = gradient(pt,msh,cl);
        return ret/ret.norm();
            
    }
    /*
    T divergence_disc_old( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
            
        size_t counter = offset(msh,cl) ;
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = (sol_HHO.block(0,counter,local_dim,1)).col(0);
        
        T grad_norm = (this->gradient( pt , msh , cl )).norm() ;
    
        //std::cout<<"grad norm is "<<grad_norm<<std::endl;
        //std::cout<<"values_cell is "<<'\n'<<values_cell<<std::endl;
        //std::cout<<"cb.eval_divergence(pt) is "<<'\n'<<cb.eval_divergence(pt)<<std::endl;
        //std::cout<<"( values_cell.dot(cb.eval_divergence(pt)) ) is "<<( values_cell.dot(cb.eval_divergence(pt)) )<<std::endl;
           
        return -( values_cell.dot(cb.eval_divergence(pt)) ) / (2 * grad_norm) ;
                
    }
    */
    
    T divergence( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
            
        size_t counter = offset(msh,cl) ;
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = (sol_HHO.block(0,counter,local_dim,1)).col(0);
        
        T grad_norm = (this->gradient( pt , msh , cl )).norm() ;
        auto grad_eval = cb.eval_gradients(pt) ;
        //std::cout<<"grad norm is "<<grad_norm<<std::endl;
        //std::cout<<"values_cell is "<<'\n'<<values_cell<<std::endl;
        //std::cout<<"cb.eval_divergence(pt) is "<<'\n'<<cb.eval_divergence(pt)<<std::endl;
        //std::cout<<"( values_cell.dot(cb.eval_divergence(pt)) ) is "<<( values_cell.dot(cb.eval_divergence(pt)) )<<std::endl;
           
        T divergence_correction = -1.0/( pow(grad_norm,3) )*( (pow( ( values_cell.dot(grad_eval.col(0)) ) , 2)) * ( values_cell.dot(cb.eval_double_derivative_x(pt)) ) + (pow( ( values_cell.dot(grad_eval.col(1)) ) , 2)) * ( values_cell.dot(cb.eval_double_derivative_y(pt)) ) + 2.0* ( values_cell.dot(grad_eval.col(0)) )  * ( values_cell.dot(grad_eval.col(1)) ) * ( values_cell.dot(cb.eval_derivative_xy(pt)) )
                                                             ) ;
        //std::cout<<"CHECK divergence AND double derivative: in pt = "<< pt <<" error = "<< ( cb.eval_double_derivative_x(pt) + cb.eval_double_derivative_y(pt) - cb.eval_divergence(pt) ) <<std::endl;
        //T divergence_correction = values_cell.dot(cb.eval_gradients(pt).col(0))/pow(grad_norm,3)*values_cell.dot(cb.eval_double_derivative_x(pt)) + values_cell.dot(cb.eval_gradients(pt).col(1))/pow(grad_norm,3)*values_cell.dot(cb.eval_double_derivative_y(pt)) ;
      
        //std::cout<<"Res 0 = "<< values_cell.dot(cb.eval_divergence(pt)) / (grad_norm) + divergence_correction ;
        //std::cout<<"Res 1 = "<< - (values_cell.dot(cb.eval_divergence(pt)) / (grad_norm) + divergence_correction  );
        
        return -( (values_cell.dot(cb.eval_divergence(pt)) ) / (grad_norm) + divergence_correction );
                
    }
   
    
    void normal_continuous_setting()
    {
            
        
        
        
        Matrix<T, Dynamic, 1> ret0 = Matrix<T, Dynamic, 1>::Zero(ndof_FE, 1) ;
        Matrix<T, Dynamic, 1> ret1 = Matrix<T, Dynamic, 1>::Zero(ndof_FE, 1) ;
        timecounter tc ;
        tc.tic();
        
        SimplicialLLT<SparseMatrix<T> >solver_global_mass;
        solver_global_mass.compute(Global_Mass);
        tc.toc();
        std::cout<<"----> TIME: In normal_continuous_setting INVERSIONE MATRIX, time = "<<tc<<std::endl;
        //std::cout<<"sono qua 0"<<std::endl;
      
        tc.tic();
        for(auto& cl : msh.cells)
        {
            timecounter tc2 ;
            //tc2.tic();
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto cbs = cb.size();
            size_t cell_offset = offset(msh,cl);
            Matrix<T, Dynamic, 1> ret0_loc = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
            Matrix<T, Dynamic, 1> ret1_loc = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
            
            auto qps = integrate(msh, cl, 2*(degree_FEM+1)); // integration of order 2k
            //tc2.toc();
            //std::cout<<"----> TIME: pezzo 1, time = "<<tc2<<std::endl;
            //tc2.tic();
            for (auto& qp : qps)
            {
                auto phi = cb.eval_basis(qp.first);
                auto disc_normal = (this->normal( qp.first , msh , cl )) ;
                //auto disc_normal = (this->normal( qp.first , msh , cl )) ;
                ret0_loc += (qp.second * disc_normal(0) * phi.transpose() );
                ret1_loc += (qp.second * disc_normal(1) * phi.transpose() );
                // phi * phi.transpose is (degree+1)^2 x (degree+1)^2 ; qp.second is a scalar
            }
            //tc2.toc();
            //std::cout<<"----> TIME: QPS, time = "<<tc2<<std::endl;
            //std::cout<<"sono qua 1"<<std::endl;
            //tc2.tic();
            for (size_t i = 0; i < local_dim; i++)
            {
                
                size_t asm_map =  connectivity_matrix[cell_offset][i].first ;
                //std::cout<<"i = "<<i<<" , asm_map = "<<asm_map<<" , cell_offset = "<<cell_offset<<" , ret0.size() = "<<ret0.size()<<" , ret1.size() = "<<ret1.size()<<" , ret0_loc.size() = "<<ret0_loc.size()<<" , ret1_loc.size() = "<<ret1_loc.size()<<std::endl;
                ret0( asm_map ) += ret0_loc(i) ;
                ret1( asm_map ) += ret1_loc(i) ;
                  
                //if( std::abs( sol_tmp(i) - sol_tmp_2(i) )>1e-14 )
                //    std::cout<< std::abs(sol_tmp(i) - sol_tmp_2(i) ) <<std::endl;
                  
            }
            //std::cout<<"sono qua 2"<<std::endl;
            //tc2.toc();
            //std::cout<<"----> TIME: pezzo 3, time = "<<tc2<<std::endl;
            
        }
        tc.toc();
        std::cout<<"----> TIME: FEM CREATION, time = "<<tc<<std::endl;
        tc.tic();
        normal_c_FEM_0 = solver_global_mass.solve(ret0);
        normal_c_FEM_1 = solver_global_mass.solve(ret1);
        tc.toc();
        std::cout<<"----> TIME: FEM RESOLUTION, time = "<<tc<<std::endl;
        tc.tic();
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                normal_c_HHO_0(i,counter_bis) = normal_c_FEM_0( asm_map ) ;
                normal_c_HHO_1(i,counter_bis) = normal_c_FEM_1( asm_map ) ;
            }
            
        }
        tc.toc();
               
        std::cout<<"----> TIME: HHO RESOLUTION, time = "<<tc<<std::endl;
        //std::cout<<"normal_c_HHO_0"<<'\n'<<normal_c_HHO_0<<std::endl;
        //std::cout<<"normal_c_HHO_1"<<'\n'<<normal_c_HHO_1<<std::endl;
                
    }
    
    Eigen::Matrix<T,2,1> normal_cont( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        size_t counter = offset(msh,cl);
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell0 = normal_c_HHO_0.col(counter);
        auto values_cell1 = normal_c_HHO_1.col(counter);
        //std::cout<<"normal_c_HHO_0"<<'\n'<<values_cell0<<std::endl;
        //std::cout<<"normal_c_HHO_1"<<'\n'<<values_cell1<<std::endl;
        auto basis_eval =  cb.eval_basis(pt);
        ret(0) = values_cell0.dot( basis_eval );
        ret(1) = values_cell1.dot( basis_eval );
        //std::cout<<"CONTINUOS NORMAL =( "<<ret(0)<<" , "<<ret(1)<<" )  in pt = "<<pt<<std::endl;
        //values_cell.dot( grad_eval.col(1) );
        // std::cout<<"Value of derivative new along y"<<ret(1)<<std::endl;
        return ret;
    
    }
    
    
    T divergence_cont( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
               
        size_t counter = offset(msh,cl) ;
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell0 = normal_c_HHO_0.col(counter);
        auto values_cell1 = normal_c_HHO_1.col(counter);
        auto grad_eval =  cb.eval_gradients(pt);
        //T grad_norm = (this->gradient( pt , msh , cl )).norm() ;
        //std::cout<<"grad norm is "<<grad_norm<<std::endl;
        //std::cout<<"values_cell is "<<'\n'<<values_cell<<std::endl;
        //std::cout<<"cb.eval_divergence(pt) is "<<'\n'<<cb.eval_divergence(pt)<<std::endl;
        //std::cout<<"( values_cell.dot(cb.eval_divergence(pt)) ) is "<<( values_cell.dot(cb.eval_divergence(pt)) )<<std::endl;
        //std::cout<<"CURVATURE( "<< pt <<" ) = "<< values_cell0.dot( grad_eval.col(0)) + values_cell1.dot( grad_eval.col(1))<<std::endl;
        return -(values_cell0.dot( grad_eval.col(0) ) + values_cell1.dot( grad_eval.col(1) ) );
        //  return -( values_cell.dot(cb.eval_divergence(pt)) ) / (2 * grad_norm) ;
                   
    }
    
    void gradient_continuous_setting()
    {
            
        
        
        
        Matrix<T, Dynamic, 1> ret0 = Matrix<T, Dynamic, 1>::Zero(ndof_FE, 1) ;
        Matrix<T, Dynamic, 1> ret1 = Matrix<T, Dynamic, 1>::Zero(ndof_FE, 1) ;
        
        SimplicialLLT<SparseMatrix<T> >solver_global_mass;
        solver_global_mass.compute(Global_Mass);
        //std::cout<<"sono qua 0"<<std::endl;
        for(auto& cl : msh.cells)
        {
            
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto cbs = cb.size();
            size_t cell_offset = offset(msh,cl);
            Matrix<T, Dynamic, 1> ret0_loc = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
            Matrix<T, Dynamic, 1> ret1_loc = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
            
            auto qps = integrate(msh, cl, 2*(degree_FEM+1)); // integration of order 2k

            for (auto& qp : qps)
            {
                auto phi = cb.eval_basis(qp.first);
                auto disc_gradient = (this->gradient( qp.first , msh , cl )) ;
                //auto disc_normal = (this->normal( qp.first , msh , cl )) ;
                ret0_loc += qp.second * disc_gradient(0) * phi.transpose();
                ret1_loc += qp.second * disc_gradient(1) * phi.transpose();
                // phi * phi.transpose is (degree+1)^2 x (degree+1)^2 ; qp.second is a scalar
            }
            //std::cout<<"sono qua 1"<<std::endl;
            for (size_t i = 0; i < local_dim; i++)
            {
                
                size_t asm_map =  connectivity_matrix[cell_offset][i].first ;
                //std::cout<<"i = "<<i<<" , asm_map = "<<asm_map<<" , cell_offset = "<<cell_offset<<" , ret0.size() = "<<ret0.size()<<" , ret1.size() = "<<ret1.size()<<" , ret0_loc.size() = "<<ret0_loc.size()<<" , ret1_loc.size() = "<<ret1_loc.size()<<std::endl;
                ret0( asm_map ) += ret0_loc(i) ;
                ret1( asm_map ) += ret1_loc(i) ;
                  
                //if( std::abs( sol_tmp(i) - sol_tmp_2(i) )>1e-14 )
                //    std::cout<< std::abs(sol_tmp(i) - sol_tmp_2(i) ) <<std::endl;
                  
            }
            //std::cout<<"sono qua 2"<<std::endl;
            
            
        }
        
        gradient_c_FEM_0 = solver_global_mass.solve(ret0);
        gradient_c_FEM_1 = solver_global_mass.solve(ret1);
       
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                gradient_c_HHO_0(i,counter_bis) = gradient_c_FEM_0( asm_map ) ;
                gradient_c_HHO_1(i,counter_bis) = gradient_c_FEM_1( asm_map ) ;
            }
            
        }
        //std::cout<<"normal_c_HHO_0"<<'\n'<<normal_c_HHO_0<<std::endl;
        //std::cout<<"normal_c_HHO_1"<<'\n'<<normal_c_HHO_1<<std::endl;
                
    }
    
    Eigen::Matrix<T,2,1> grad_cont( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        size_t counter = offset(msh,cl);
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell0 = gradient_c_HHO_0.col(counter);
        auto values_cell1 = gradient_c_HHO_1.col(counter);
        //std::cout<<"normal_c_HHO_0"<<'\n'<<values_cell0<<std::endl;
        //std::cout<<"normal_c_HHO_1"<<'\n'<<values_cell1<<std::endl;
        auto basis_eval =  cb.eval_basis(pt);
        ret(0) = values_cell0.dot( basis_eval );
        ret(1) = values_cell1.dot( basis_eval );
        //std::cout<<"CONTINUOS NORMAL =( "<<ret(0)<<" , "<<ret(1)<<" )  in pt = "<<pt<<std::endl;
        //values_cell.dot( grad_eval.col(1) );
        // std::cout<<"Value of derivative new along y"<<ret(1)<<std::endl;
        return ret;
    
    }
    
    
    
    Eigen::Matrix<T,2,1> normal_cont_normalised( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        // Continuous normal noramlised -> obtained via the L2 projection of the discontinuos gradient over the basis B_k.
        size_t counter = offset(msh,cl);
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell0 = gradient_c_HHO_0.col(counter);
        auto values_cell1 = gradient_c_HHO_1.col(counter);
        
        
        
        auto basis_eval =  cb.eval_basis(pt);
        ret(0) = values_cell0.dot( basis_eval );
        ret(1) = values_cell1.dot( basis_eval );
        //std::cout<<"values_cell"<<'\n'<<ret<<std::endl;
        //std::cout<<"values_cell.norm()"<<'\n'<<ret.norm()<<std::endl;
        //std::cout<<"gradient_c_HHO"<<'\n'<<ret/ret.norm()<<std::endl;
        
        return ret/ret.norm();
    
    }
    
    
    T divergence_cont_grad( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
            
        size_t counter = offset(msh,cl) ;
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell0 = gradient_c_HHO_0.col(counter);
        auto values_cell1 = gradient_c_HHO_1.col(counter);
       
        auto grad_eval = cb.eval_gradients(pt) ;
        auto b_eval = cb.eval_basis(pt) ;
        T grad_norm = (this->grad_cont( pt , msh , cl )).norm() ;
    
        //std::cout<<"grad norm is "<<grad_norm<<std::endl;
        //std::cout<<"values_cell is "<<'\n'<<values_cell<<std::endl;
        //std::cout<<"cb.eval_divergence(pt) is "<<'\n'<<cb.eval_divergence(pt)<<std::endl;
        //std::cout<<"( values_cell.dot(cb.eval_divergence(pt)) ) is "<<( values_cell.dot(cb.eval_divergence(pt)) )<<std::endl;
           
        T divergence_correction = -1.0/( pow(grad_norm,3) )*( (pow( ( values_cell0.dot(b_eval) ) , 2)) * ( values_cell0.dot(grad_eval.col(0)) ) +  ( values_cell0.dot(b_eval) ) * ( values_cell1.dot(b_eval) ) * ( values_cell1.dot(grad_eval.col(0)) ) +
                                                             ( values_cell1.dot(b_eval) ) * ( values_cell0.dot(b_eval) ) * ( values_cell0.dot(grad_eval.col(1)) ) +  (pow( ( values_cell1.dot(b_eval) ) , 2)) * ( values_cell1.dot(grad_eval.col(1)) ) );
        
        
        //T divergence_correction = values_cell.dot(cb.eval_gradients(pt).col(0))/pow(grad_norm,3)*values_cell.dot(cb.eval_double_derivative_x(pt)) + values_cell.dot(cb.eval_gradients(pt).col(1))/pow(grad_norm,3)*values_cell.dot(cb.eval_double_derivative_y(pt)) ;
      
        //std::cout<<"Res 0 = "<< values_cell.dot(cb.eval_divergence(pt)) / (grad_norm) + divergence_correction ;
        //std::cout<<"Res 1 = "<< - (values_cell.dot(cb.eval_divergence(pt)) / (grad_norm) + divergence_correction  );
        
        return -( (values_cell0.dot(grad_eval.col(0)) + values_cell1.dot(grad_eval.col(1)) ) / (grad_norm) + divergence_correction );
                
    }
    
    
    
    
    
    void smooth_cut_off( T C , T r0 , T delta , T x_centre , T y_centre , T radius , T radius_a , T radius_b )
    {
        
        Matrix<T, Dynamic,1> local_RHS = Matrix<T, Dynamic, 1>::Zero( local_dim , 1);
        //Matrix<T, Dynamic, 1> RHS = Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
        Matrix<T, Dynamic,1> sol_loc = Matrix<T, Dynamic, 1>::Zero( local_dim , 1);
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        
        
        /*
        T hx = params.hx();
        T hy = params.hy();
        T pos_r0 = 0.5; //std::min(x_centre , 1 - x_centre );
        T r_max = std::max( radius_a , radius_b ) ;
        T h = std::max( hx , hy ) ;
        T r0 = r_max + 2*h*sqrt(2.0);
        C = r0*r0*radius_b*radius_b - radius_a*radius_a*radius_b*radius_b;
         //T dist = pos_r0 - radius + 2.0*0.0625;
         //T dist = pos_r0 - radius + 2.0*0.07;
         //T r0 = radius + dist/2.0;
         
         T delta = r0/8.0; // FIRST CHOICE
        //T delta = r0/20.0;
        */
        
        
        
        
        
        // Lambda function to define smooth function
        auto alfa = [=](const point<T,2>& pt)
        { // sol
            return (1 - tanh( (sqrt( pow((pt.x()-x_centre),2) + pow((pt.y()-y_centre),2) ) - r0 ) / delta ))/2;};
        
        
        
            
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
             
            auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
            for(size_t i = 0 ; i<local_dim ; i++)
            {
                auto pt = qps[i];
                auto asm_map =  connectivity_matrix[cell_offset][i].first;
                local_RHS(i) =  ( 1.0 - alfa(pt)  ) * C + alfa(pt) * (*this)(pt,msh,cl) ;
            }
            sol_loc = vandermonde_interpolant.solve(local_RHS); // SAVE Vandermonde interpolation
            sol_HHO.col(cell_offset) = sol_loc ; // SAVE Vandermonde
            
            if(!analytic_check)
            {
                for (size_t i = 0; i < local_dim; i++)
                {
                    auto asm_map =  connectivity_matrix[cell_offset][i].first;
                    sol_FEM(asm_map) = sol_loc(i) ;
                }
                size_t i_vertex = cell_offset+floor(cell_offset/Nx);
                vertices(i_vertex) = sol_HHO(0,cell_offset) ;
                vertices(i_vertex+1) = sol_HHO(1,cell_offset) ;
                vertices(i_vertex+Nx+2) = sol_HHO(2,cell_offset) ;
                vertices(i_vertex+Nx+1) = sol_HHO(3,cell_offset) ;
            }
            
        } // end of cl loop
             
    
            
        //converting_into_HHO_formulation(sol_FEM);
        if(!analytic_check)
            set_max_min();
        

        /*
        postprocess_output<double> postoutput00;
        typedef typename Mesh::point_type       point_type;
        point<double,2> node;
        auto alfa_values = std::make_shared< gnuplot_output_object<double> >("alfa.dat");
        //auto interface_pos = std::make_shared< gnuplot_output_object<double> >("interface_alfa.dat");
        for(auto& pt:msh.points )
        {
            alfa_values->add_data(pt,alfa(pt));
        }
        postoutput00.add_object(alfa_values);
        postoutput00.write();
        
        */
    }
    
    void smooth_cut_off( T C , T x_centre , T y_centre , T radius , T radius_a , T radius_b )
    {
        
        Matrix<T, Dynamic,1> local_RHS = Matrix<T, Dynamic, 1>::Zero( local_dim , 1);
        //Matrix<T, Dynamic, 1> RHS = Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
        Matrix<T, Dynamic,1> sol_loc = Matrix<T, Dynamic, 1>::Zero( local_dim , 1);
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        
        
        
        T hx = params.hx();
        T hy = params.hy();
        T pos_r0 = 0.5; //std::min(x_centre , 1 - x_centre );
        T r_max = std::max( radius_a , radius_b ) ;
        T h = std::max( hx , hy ) ;
        T r0 = r_max + 2*h*sqrt(2.0);
        C = r0*r0*radius_b*radius_b - radius_a*radius_a*radius_b*radius_b;
         //T dist = pos_r0 - radius + 2.0*0.0625;
         //T dist = pos_r0 - radius + 2.0*0.07;
         //T r0 = radius + dist/2.0;
         
         T delta = r0/8.0; // FIRST CHOICE
        //T delta = r0/20.0;
        
        
        
        
        
        std::cout<<"r_max = "<<r_max<<" , r0 = "<<r0<<" , delta = "<<delta<<" , hx = hy = "<<hx<<std::endl;
        std::cout<<"value in alfa in r_int = "<<(radius-r0)/delta<<std::endl;
        std::cout<<"value in alfa in R = "<<(pos_r0-r0)/delta<<std::endl;
        
        // Lambda function to define smooth function
        auto alfa = [=](const point<T,2>& pt)
        { // sol
            return (1 - tanh( (sqrt( pow((pt.x()-x_centre),2) + pow((pt.y()-y_centre),2) ) - r0 ) / delta ))/2;};
        
        
        
            
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
             
            auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
            for(size_t i = 0 ; i<local_dim ; i++)
            {
                auto pt = qps[i];
                auto asm_map =  connectivity_matrix[cell_offset][i].first;
                local_RHS(i) =  ( 1.0 - alfa(pt)  ) * C + alfa(pt) * (*this)(pt,msh,cl) ;
            }
            sol_loc = vandermonde_interpolant.solve(local_RHS); // SAVE Vandermonde interpolation
            sol_HHO.col(cell_offset) = sol_loc ; // SAVE Vandermonde
            
            if(!analytic_check)
            {
                for (size_t i = 0; i < local_dim; i++)
                {
                    auto asm_map =  connectivity_matrix[cell_offset][i].first;
                    sol_FEM(asm_map) = sol_loc(i) ;
                }
                size_t i_vertex = cell_offset+floor(cell_offset/Nx);
                vertices(i_vertex) = sol_HHO(0,cell_offset) ;
                vertices(i_vertex+1) = sol_HHO(1,cell_offset) ;
                vertices(i_vertex+Nx+2) = sol_HHO(2,cell_offset) ;
                vertices(i_vertex+Nx+1) = sol_HHO(3,cell_offset) ;
            }
            
        } // end of cl loop
             
    
            
        //converting_into_HHO_formulation(sol_FEM);
        if(!analytic_check)
            set_max_min();

    }
    
    
    void cut_off( T d )
    {
        
        Matrix<T, Dynamic,1> local_RHS = Matrix<T, Dynamic, 1>::Zero( local_dim , 1);
        Matrix<T, Dynamic,1> sol_loc = Matrix<T, Dynamic, 1>::Zero( local_dim , 1);
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        
        
        // Known term (f,b_i)_i , b_i Bernstein basis fx
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
              
            auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
            for(size_t i = 0 ; i<local_dim ; i++)
            {
                auto pt = qps[i];
                auto asm_map =  connectivity_matrix[cell_offset][i].first;
                 
                if( (*this)(pt,msh,cl)>= d )
                    local_RHS(i) =  d ;
                else if( (*this)(pt,msh,cl)<= -d )
                    local_RHS(i) =  -d ;
                else
                    local_RHS(i) =  (*this)(pt,msh,cl) ;
        
            }
            
            sol_loc = vandermonde_interpolant.solve(local_RHS); // SAVE Vandermonde interpolation
            sol_HHO.col(cell_offset) = sol_loc ; // SAVE Vandermonde
            if(!analytic_check)
            {
                for (size_t i = 0; i < local_dim; i++)
                {
                    auto asm_map =  connectivity_matrix[cell_offset][i].first;
                    sol_FEM(asm_map) = sol_loc(i) ;
                }
            
                size_t i_vertex = cell_offset+floor(cell_offset/Nx);
                vertices(i_vertex) = sol_HHO(0,cell_offset) ;
                vertices(i_vertex+1) = sol_HHO(1,cell_offset) ;
                vertices(i_vertex+Nx+2) = sol_HHO(2,cell_offset) ;
                vertices(i_vertex+Nx+1) = sol_HHO(3,cell_offset) ;
            }
        } // end of cl loop
              
       
        
        
        //converting_into_HHO_formulation(sol_FEM);
        if(!analytic_check)
            set_max_min();
        
        
    }
    
        
        
};



template< typename Mesh , typename Fonction , typename FiniteSpace , typename T = typename Mesh::coordinate_type  >
struct L2projected_level_set_high_order_grad_cont: public level_set<T>
{
     
    bool analytic_check = FALSE ;
    
    T phi_max , phi_min ;
    size_t  last_row_init, last_row_end, number_faces_one_row;
    
    T iso_val_interface = 0.0 ;
    Mesh msh; // Original mesh, NOT agglomerated.
    mesh_init_params<T> params; // mesh parameter
       
    size_t degree_FEM ; // FEM degree
    size_t n_cls ; // #cells
    size_t local_dim; // Local Dimension (degree_FEM+1)*(degree_FEM+1)
    size_t n_vertices ; // #vertices
    size_t      Nx, Ny ; // Number of cells in x and y direciton
    
    // connectivity matrix : for each cell, it stores the global numbering
    std::vector< std::vector<std::pair<size_t,bool>>> connectivity_matrix ;
    std::vector<std::set<size_t>> S_i;
    //std::vector<std::vector<size_t>> connectivity_matrix ;
       
    size_t dim_HHO; // Global dimension Discontinuous framework = Local dimension * #cells
    size_t ndof_FE; // Global dimension FE continuous = #nodes
       
    int mapped = 0 ; // = 0 not mapped ; = 1 mapped ; = 2 inverse mapping
    
    SparseMatrix<T>                 Global_Mass; // Global mass, saved for FEM problem
       
    Eigen::Matrix<T, Dynamic, Dynamic> sol_HHO ; // projection saved in HHO format: cell by cell
    Matrix<T, Dynamic, 1> sol_FEM ; // projection saved in Continuos FE format: global nodes
    Eigen::Matrix<T, Dynamic, 1> vertices; // saving level_set on vertices mesh

    // Local Vandermonde Matrix for interpolation
    Matrix<T, Dynamic, Dynamic> local_vandermonde ;
       
       
    // Assembling into global matrix
    SparseMatrix<T>                 Global_c_term_x; // Global mass, saved for FEM problem
    SparseMatrix<T>                 Global_c_term_y; // Global mass, saved for FEM problem
    
    Matrix<T, Dynamic, 1>      Global_Mass_Lumped; // Global mass, saved for FEM problem

    SparseMatrix<T>         cij_norm , nij0 , nij1 ;
    SparseMatrix<T>         cji_norm , nji0 , nji1 ;
    
    
    Eigen::Matrix<T, Dynamic, Dynamic> normal_c_HHO_0 ;
    Eigen::Matrix<T, Dynamic, Dynamic> normal_c_HHO_1 ;
    Eigen::Matrix<T, Dynamic, 1> normal_c_FEM_0 ;
    Eigen::Matrix<T, Dynamic, 1> normal_c_FEM_1 ;
    
    Eigen::Matrix<T, Dynamic, Dynamic> gradient_c_HHO_0 ;
    Eigen::Matrix<T, Dynamic, Dynamic> gradient_c_HHO_1 ;
    Eigen::Matrix<T, Dynamic, 1> gradient_c_FEM_0 ;
    Eigen::Matrix<T, Dynamic, 1> gradient_c_FEM_1 ;
    
    L2projected_level_set_high_order_grad_cont(const FiniteSpace& fe_data , const Fonction & level_set, const Mesh & msh , bool analytic_check = FALSE )
        : degree_FEM(fe_data.order) , local_dim(fe_data.local_ndof), msh(msh), Nx(fe_data.Nx),Ny(fe_data.Ny), params(fe_data.params) , dim_HHO(fe_data.ndof_disc) , n_cls(fe_data.n_cls) ,n_vertices(fe_data.n_vertices) , connectivity_matrix(fe_data.connectivity_matrix) , ndof_FE(fe_data.ndof_FE) , analytic_check(analytic_check)
    {
        if(!analytic_check)
        {
            timecounter tc_level_set;
            tc_level_set.tic();
            Matrix<T, Dynamic, 1>           RHS;    // Known term
            std::vector< Triplet<T> >       triplets; // Position elements: Sparse Matrix Notation
            std::vector< Triplet<T> >       triplets_c_term_x; // Position elements: Sparse Matrix Notation
            std::vector< Triplet<T> >       triplets_c_term_y; // Position elements: Sparse Matrix Notation
            
            
            last_row_init = Ny*(2*Nx+1); // There are 2 faces for each row of cells + Ny
            last_row_end = last_row_init + Nx-1;
            number_faces_one_row = 2*Nx+1; // for each cell I count only the low and sx faces, respectevely 0-1 2-3 4-5 6-7 8 + the last on the right boundary
              
            Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero(local_dim);
             
            Global_Mass = SparseMatrix<T>( ndof_FE, ndof_FE ); //(b_i,b_j)_ij , b_i Lagrange basis fx
            RHS = Matrix<T, Dynamic, 1>::Zero( ndof_FE ); // Known term (f,b_i)_i , b_i Lagrange basis fx
              
            Global_c_term_x = SparseMatrix<T>( ndof_FE, ndof_FE ); //(b_i,b_j)_ij , b_i Lagrange basis fx
            Global_c_term_y = SparseMatrix<T>( ndof_FE, ndof_FE ); //(b_i,b_j)_ij , b_i Lagrange basis fx
               
            Global_Mass_Lumped = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
               
            local_vandermonde = Matrix<T, Dynamic, Dynamic>::Zero ( local_dim,local_dim );
            
            normal_c_HHO_0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
            normal_c_HHO_1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
            normal_c_FEM_0 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            normal_c_FEM_1 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            
            gradient_c_HHO_0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls);
            gradient_c_HHO_1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls);
            gradient_c_FEM_0 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            gradient_c_FEM_1 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            
            
            
            cij_norm = SparseMatrix<T>( ndof_FE, ndof_FE );
            cji_norm = SparseMatrix<T>( ndof_FE, ndof_FE );
            nij0 = SparseMatrix<T>( ndof_FE, ndof_FE );
            nij1 = SparseMatrix<T>( ndof_FE, ndof_FE );
            nji0 = SparseMatrix<T>( ndof_FE, ndof_FE );
            nji1 = SparseMatrix<T>( ndof_FE, ndof_FE );
            /*
            std::vector< Triplet<T> >       triplets_norm;
            std::vector< Triplet<T> >       triplets_norm_adj;
            std::vector< Triplet<T> >       triplets_nij0;
            std::vector< Triplet<T> >       triplets_nij1;
            std::vector< Triplet<T> >       triplets_nji0;
            std::vector< Triplet<T> >       triplets_nji1;
            */
              
            // Saving the projection in HHO discontinuous format (MATRIX NOTATION)
            sol_HHO = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
            sol_FEM = Matrix<T, Dynamic, 1>::Zero(ndof_FE);
            vertices = Eigen::Matrix<T, Dynamic, 1>::Zero( n_vertices , 1 );
              
            //Eigen::Matrix<T, Dynamic, Dynamic> sol_HHO_vandermonde =  Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls ) ;
              
              
            CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > cod;
            //FullPivLU<Matrix<T, Dynamic, Dynamic > > cod_2;
              
              
            std::cout<<"----> In 'L2projected_level_set_high_order': Vandermonde interpolation of the level set with BERNSTEIN basis."<<std::endl;
            
            //std::cout<<"CHECK INVERSION LOCAL VANDERMONDE"<<std::endl;
            for( const auto& cl : msh.cells )
            {
                size_t cell_offset = offset(msh, cl) ;
                
                if(cell_offset == 0)
                {
                    auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree_FEM);
                    cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
                    for (size_t i = 0; i < local_dim; i++)
                    {
                        // Local vandermonde matrix
                        local_vandermonde.block(i,0,1,local_dim) = (cb.eval_basis(pts[i])).transpose() ;
                    }
                      
                    cod.compute(local_vandermonde);
                    //cod_2.compute(local_vandermonde);
                }
                
             
                auto local_mass = make_bernstein_local_mass_matrix( msh, cl , degree_FEM );
                  
                //auto local_RHS = make_bernstein_local_RHS( msh , cl , degree_FEM , level_set );
                // Local c_ij = b_i nabla(b_j) -> USEFUL FOR TRANSPORT PROBLEM
                auto local_cij = make_bernstein_local_cij_matrix (msh, cl, degree_FEM);
                      
                auto local_mass_lumped = make_bernstein_local_mass_matrix_lumped( msh , cl , degree_FEM ) ;
                 
                  
                // Costruction of the coefficients of the Bernstein basis
                auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
                  
                  
                // Assembling triplets for global problem
                for (size_t i = 0; i < local_dim; i++)
                {
                    size_t asm_map_i = connectivity_matrix[cell_offset][i].first ;
                    for (size_t j = 0; j < local_dim; j++)
                    {
                        /*
                        T c_ij0 = local_cij.first(i,j) ;
                        T c_ij1 = local_cij.second(i,j) ;
                        
                        T c_ji0 = local_cij.first(j,i) ;
                        T c_ji1 = local_cij.second(j,i) ;
                        */
                        size_t asm_map_j = connectivity_matrix[cell_offset][j].first ;
                          
                        triplets.push_back(Triplet<T>(asm_map_i,asm_map_j, local_mass(i,j)));
                        triplets_c_term_x.push_back( Triplet<T>(asm_map_i, asm_map_j , local_cij.first(i,j) ) );
                        triplets_c_term_y.push_back( Triplet<T>( asm_map_i , asm_map_j , local_cij.second(i,j) ) );
                        
                        /*
                        T val_norm = sqrt( c_ij0*c_ij0 + c_ij1*c_ij1 );
                        T val_norm_adj = sqrt( c_ji0 *c_ji0 + c_ji1*c_ji1 );
                        T val_nij0 = c_ij0/val_norm ;
                        T val_nij1 = c_ij1/val_norm ;
                        T val_nji0 = c_ji0/val_norm_adj ;
                        T val_nji1 = c_ji1/val_norm_adj ;
                        
                        triplets_norm.push_back(Triplet<T>(asm_map_i,asm_map_j, val_norm));
                        triplets_norm_adj.push_back(Triplet<T>(asm_map_i,asm_map_j, val_norm_adj));
                        triplets_nij0.push_back(Triplet<T>(asm_map_i,asm_map_j, val_nij0));
                        triplets_nij1.push_back(Triplet<T>(asm_map_i,asm_map_j, val_nij1));
                        triplets_nji0.push_back(Triplet<T>(asm_map_i,asm_map_j, val_nji0));
                        triplets_nji1.push_back(Triplet<T>(asm_map_i,asm_map_j, val_nji1));
                        */
                    }
                    Global_Mass_Lumped(asm_map_i) += local_mass_lumped(i);
                    //Global_Mass_Lumped(asm_map[i].first) += local_mass_lumped(i);
                    //RHS(asm_map[i].first) += local_RHS(i) ;
                    RHS_vandermonde(i) = level_set( qps[i]) ;
                }
                  
                  
                auto sol_tmp = cod.solve(RHS_vandermonde) ;
                  
                //auto sol_tmp_2 = cod_2.solve(RHS_vandermonde) ;
                  
                sol_HHO.col(cell_offset) = sol_tmp ;
                for (size_t i = 0; i < local_dim; i++)
                {
                      
                    size_t asm_map =  connectivity_matrix[cell_offset][i].first ;
                    sol_FEM( asm_map ) = sol_HHO(i,cell_offset) ;
                      
                    //if( std::abs( sol_tmp(i) - sol_tmp_2(i) )>1e-14 )
                    //    std::cout<< std::abs(sol_tmp(i) - sol_tmp_2(i) ) <<std::endl;
                      
                }
                size_t i_vertex = cell_offset+floor(cell_offset/Nx);
                vertices(i_vertex) = sol_HHO(0,cell_offset) ;
                vertices(i_vertex+1) = sol_HHO(1,cell_offset) ;
                vertices(i_vertex+Nx+2) = sol_HHO(2,cell_offset) ;
                vertices(i_vertex+Nx+1) = sol_HHO(3,cell_offset) ;
                  
                  
            } // end of cl loop
                
            //std::cout<<"FINE CHECK INVERSION LOCAL VANDERMONDE"<<std::endl;
            // Finalisation global assembling
            Global_Mass.setFromTriplets( triplets.begin(), triplets.end() );
            triplets.clear();
                  
                  
            // Finalisation global assembling
            Global_c_term_x.setFromTriplets( triplets_c_term_x.begin(), triplets_c_term_x.end() );
            triplets_c_term_x.clear();
            Global_c_term_y.setFromTriplets( triplets_c_term_y.begin(), triplets_c_term_y.end() );
            triplets_c_term_y.clear();
                
            /*
            cij_norm.setFromTriplets( triplets_norm.begin(), triplets_norm.end() );
            triplets_norm.clear();
            cji_norm.setFromTriplets( triplets_norm_adj.begin(), triplets_norm_adj.end() );
            triplets_norm_adj.clear();
            
            nij0.setFromTriplets( triplets_nij0.begin(), triplets_nij0.end() );
            triplets_nij0.clear();
            nij1.setFromTriplets( triplets_nij1.begin(), triplets_nij1.end() );
            triplets_nij1.clear();
            
            nji0.setFromTriplets( triplets_nji0.begin(), triplets_nji0.end() );
            triplets_nji0.clear();
            nji1.setFromTriplets( triplets_nji1.begin(), triplets_nji1.end() );
            triplets_nji1.clear();
            */
            
            // NORM of c_ij
            
            cij_norm = ( Global_c_term_x.cwiseProduct(Global_c_term_x) + Global_c_term_y.cwiseProduct(Global_c_term_y) ).cwiseSqrt() ;
            //std::cout<<"cij norm "<<'\n'<<cij_norm<<std::endl;
            
            // MATRIX n_ij
            nij0 = Global_c_term_x.cwiseQuotient( cij_norm );
            nij1 = Global_c_term_y.cwiseQuotient( cij_norm );
            
            //std::cout<<"nij1  "<<'\n'<<nij1<<std::endl;
            

            // MATRIX c_ji
            SparseMatrix<T> cji_x = Global_c_term_x.adjoint() ;
            SparseMatrix<T> cji_y = Global_c_term_y.adjoint() ;
             
            // NORM of c_ji -> i.e. c_ij transposed
            cji_norm = (cji_x.cwiseProduct(cji_x)+cji_y.cwiseProduct(cji_y)).cwiseSqrt();
            
            // MATRIX n_ij (TRANSPOSED)
            nji0 = cji_x.cwiseQuotient( cji_norm );
            nji1 = cji_y.cwiseQuotient( cji_norm );
            

            
              
            //std::cout<<"local_vandermonde"<<'\n'<<local_vandermonde<<std::endl;
                  
              
              // CALCULATION OF THE SIZE + PLOTTING
              /*
              size_t size_supp_nodes = 0;
              std::cout<<"Supporting nodes IN L2:"<<std::endl;
              size_t jjjj = 0;
              for (auto& i: S_i) {
                  size_supp_nodes+=i.size();
                  std::cout <<"Node "<<jjjj<<":";
                  for (auto it=i.begin(); it != i.end(); ++it)
                      std::cout << ' ' << *it;
                      // std::cout<<ii;
                  std::cout<<'\n';
                  jjjj++;
              }
              std::cout<<std::endl;
              std::cout<<"Supporting nodes size:"<<size_supp_nodes<<std::endl;
              */
              
                  
              
                  
              //Matrix<T, Dynamic, 1> sol_FEM_vandermonde  = Matrix<T, Dynamic, 1>::Zero(RHS.rows()); ;
              
            std::cout<<"sol_FEM size "<<sol_FEM.size()<<std::endl;
            std::cout<<"local_dim size "<<local_dim<<std::endl;
            std::cout<<"n_cls "<<n_cls<<std::endl;
                  
              // Global solution saved as discontinuous HHO approach
              // Also saved min & max coefficients + position in HHO notation
              
              /*
              ConjugateGradient<SparseMatrix<T> > solver_global_mass;
              //SparseLU<SparseMatrix<T>, COLAMDOrdering<int> >solver_global_mass;
              //Notice: for this step the numerical values of A are not used
              //solver_global_mass.analyzePattern(Global_Mass);
              //solver_global_mass.factorize(Global_Mass);
               
              solver_global_mass.compute(Global_Mass); // SAVE INVERSE OF GLOBAL MASS
              if(solver_global_mass.info()!=Success) {
                  std::cout<<"FAILED SOLVER 0"<<std::endl;
                  return;
              }
              
              sol_FEM = solver_global_mass.solve(RHS);
              */
              
              /*
              for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
              {
                  for (size_t i = 0; i < local_dim; i++)
                  {
                      size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                      //sol_HHO(i,counter_bis) = sol_FEM( asm_map ) ;
                      sol_FEM( asm_map ) = sol_HHO(i,counter_bis) ;
                  }
              }
              */
              
              
              //std::cout<<"CHECK L2 proj vs Vandermonde interpolation for phi^0"<<'\n'<<sol_FEM-sol_FEM_vandermonde<<std::endl;
              
              
              //sol_FEM = sol_FEM_vandermonde ;
              //sol_HHO = sol_HHO_vandermonde ;
              //std::cout<<"CHECK L2 proj vs Vandermonde interpolation for phi^0"<<'\n'<<sol_FEM-sol_FEM_vandermonde<<std::endl;
              // Set of maximum and minimum
              
            timecounter tcbis ;
                      
            tcbis.tic();
            set_max_min();
            tcbis.toc();
            std::cout << bold << yellow << "--> set_max_min: t = " << tcbis << " seconds" << reset << std::endl;
              
              
              /*
              for( size_t i_global = 0; i_global < n_cls; i_global++)
              {
                  size_t i_vertex = i_global+floor(i_global/Nx);
                  vertices(i_vertex) = sol_HHO(0,i_global) ;
                  vertices(i_vertex+1) = sol_HHO(1,i_global) ;
                  vertices(i_vertex+Nx+2) = sol_HHO(2,i_global) ;
                  vertices(i_vertex+Nx+1) = sol_HHO(3,i_global) ;
              }
              */
             
            tc_level_set.toc();
            std::cout << bold << yellow << "INITIALISATION LEVEL SET: t = " << tc_level_set << " seconds" << reset << std::endl;

        }
        else
        {
            timecounter tc_level_set;
            tc_level_set.tic();
            
            Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero(local_dim);
            local_vandermonde = Matrix<T, Dynamic, Dynamic>::Zero ( local_dim,local_dim );
              
              
            // Saving the projection in HHO discontinuous format (MATRIX NOTATION)
            sol_HHO = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
            normal_c_HHO_0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
            normal_c_HHO_1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
            normal_c_FEM_0 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            normal_c_FEM_1 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            
            gradient_c_HHO_0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls);
            gradient_c_HHO_1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls);
            gradient_c_FEM_0 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            gradient_c_FEM_1 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            
            CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > cod;
            
            std::cout<<"----> FOR ANALYTIC CHECK 'L2projected_level_set_high_order': Vandermonde interpolation of the level set with BERNSTEIN basis."<<std::endl;
            
        
            for( const auto& cl : msh.cells )
            {
                size_t cell_offset = offset(msh, cl) ;
                
                if(cell_offset == 0)
                {
                    auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree_FEM);
                    cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
                    for (size_t i = 0; i < local_dim; i++)
                    {
                        // Local vandermonde matrix
                        local_vandermonde.block(i,0,1,local_dim) = (cb.eval_basis(pts[i])).transpose() ;
                    }
                      
                    cod.compute(local_vandermonde);
                }
                  
            
                
                // Assembling triplets for global problem
                auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
                for (size_t i = 0; i < local_dim; i++)
                    RHS_vandermonde(i) = level_set( qps[i]) ;
            
                  
                auto sol_tmp = cod.solve(RHS_vandermonde) ;
                sol_HHO.col(cell_offset) = sol_tmp ;
                
                
                  
            } // end of cl loop
                
            tc_level_set.toc();
            std::cout << bold << yellow << "INITIALISATION LEVEL SET: t = " << tc_level_set << " seconds" << reset << std::endl;

        }
  
    
    }
            
            

   
    L2projected_level_set_high_order_grad_cont()=default;
    
    
    void
    coefficients_mapping( )
    {

        
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                RHS_vandermonde(ct) =  ( (*this)(pt , msh , cl ) - phi_min )/( phi_max - phi_min );
                ct++;
            }
            
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
         
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map = connectivity_matrix[cell_offset][i].first ;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
        } // end of cl loop
       
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        
        std::cout<<"IN COEFFICIENT_MAPPING, CHECKING phi = 0, after mappin becomes --> "<< (0.0 - phi_min )/( phi_max - phi_min )<<std::endl;
               
        std::cout<<"It should be close to 1/2, error = "<< 1./2. - ( 0.0 - phi_min )/( phi_max - phi_min ) <<std::endl;
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER MAPPING: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
        
    }
    
    void
    coefficients_mapping_MAX_MAX( )
    {

        
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                if( std::abs(phi_max) >= std::abs(phi_min))
                    RHS_vandermonde(ct) =  ( (*this)(pt , msh , cl )+ phi_max) /( 2.0*phi_max );
                else
                    RHS_vandermonde(ct) =  ( (*this)(pt , msh , cl )-phi_min ) /( -2.0*phi_min );
                
                ct++;
            }
            
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
         
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map = connectivity_matrix[cell_offset][i].first ;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
        } // end of cl loop
       
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        iso_val_interface = 1.0/2.0 ;
        
        std::cout<<"IN COEFFICIENT_MAPPING MAX MAX, CHECKING phi = 0, after mappin becomes --> "<< ( 0.0 +phi_max )/( 2.0*phi_max )<<std::endl;
        std::cout<<"It should be close to 1/2, error = "<< 1./2. - ( 0.0 + phi_max )/( 2.0*phi_max  ) <<std::endl;
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER MAPPING_MAX_MAX: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
        
    }
    
    
    
    void
    coefficients_sfasamento( )
    {

        
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        
     
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                RHS_vandermonde(ct) =   (*this)(pt , msh , cl ) + 0.5 ;
                
                ct++;
            }
            
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
            //std::cout<<"--------------->>>>>>>> RHS_vandermonde"<<'\n'<<RHS_vandermonde<<std::endl;
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map = connectivity_matrix[cell_offset][i].first ;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
        } // end of cl loop
       
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        iso_val_interface = 1.0/2.0 ;
        std::cout<<"Isovalue of the interface = "<<iso_val_interface<<std::endl;
        //std::cout<<"IN COEFFICIENT_MAPPING, CHECKING phi = 0, after mappin becomes --> "<< (0.0 - phi_min )/( phi_max - phi_min )<<std::endl;
               
        //std::cout<<"It should be close to 1/2, error = "<< 1./2. - ( 0.0 - phi_min )/( phi_max - phi_min ) <<std::endl;
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER MAPPING: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
        
    }
    
    
    
    
    
    void
    coefficients_mapping_quadratic( )
    {

        
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        
        T a1 = (-1.0/2.0 - phi_min/(phi_max-phi_min))/(pow(phi_min,2));
        T b = 1.0/(phi_max-phi_min);
        T c = 1.0/2.0;
        T a2 = (1.0/2.0 - phi_max/(phi_max-phi_min))/(pow(phi_max,2));
        
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                auto val = (*this)(pt , msh , cl ) ;
                if( val <= 0 )
                    RHS_vandermonde(ct) =   a1 * val * val + b * val + c ;
                else
                    RHS_vandermonde(ct) =   a2 * val * val + b * val + c ;
                    
                ct++;
            }
            
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
         
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map = connectivity_matrix[cell_offset][i].first ;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
        } // end of cl loop
       
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        iso_val_interface = 1.0/2.0 ;
        std::cout<<"Isovalue of the interface = "<<iso_val_interface<<std::endl;
        //std::cout<<"IN COEFFICIENT_MAPPING, CHECKING phi = 0, after mappin becomes --> "<< (0.0 - phi_min )/( phi_max - phi_min )<<std::endl;
               
        //std::cout<<"It should be close to 1/2, error = "<< 1./2. - ( 0.0 - phi_min )/( phi_max - phi_min ) <<std::endl;
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER MAPPING: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
        
    }
    
    void
    coefficients_inverse_mapping_quadratic( )
    {
        
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                RHS_vandermonde(ct) = phi_min + (*this)(pt , msh , cl )*( phi_max - phi_min );
                ct++;
            }
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
            
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map =  connectivity_matrix[cell_offset][i].first;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
            
        } // end of cl loop
        
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER MAPPING: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
           
    }
    
    
    void
    coefficients_inverse_mapping( )
    {
        
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                RHS_vandermonde(ct) = phi_min + (*this)(pt , msh , cl )*( phi_max - phi_min );
                ct++;
            }
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
            
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map =  connectivity_matrix[cell_offset][i].first;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
            
        } // end of cl loop
        
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER MAPPING: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
           
    }
        
    
    void
    coefficients_inverse_mapping_MAX_MAX( )
    {
        
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                if( std::abs(phi_max) >= std::abs(phi_min))
                    RHS_vandermonde(ct) = -phi_max + (*this)(pt , msh , cl )* ( 2.0 * phi_max );
                else
                    RHS_vandermonde(ct) = phi_min - (*this)(pt , msh , cl )* ( 2.0 * phi_min );
                
                ct++;
            }
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
            
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map =  connectivity_matrix[cell_offset][i].first;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
            
        } // end of cl loop
        
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        iso_val_interface = 0 ;
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER INVERSE_MAPPING_MAX_MAX: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
           
    }
        
        
        
    void  set_discrete_points( Eigen::Matrix<T, Dynamic, Dynamic>& values_new)
    {
        sol_HHO = values_new ;
        std::cout<<" --> set_discrete_points: check that sol_FEM already uploaded!"<<std::endl;
        
    }
        
        
    void converting_into_HHO_formulation( const Eigen::Matrix<T, Dynamic, 1>& values_new )
    {
        // SAVE BOTH SOL_HHO AND VERTICES
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                sol_HHO(i,counter_bis) = values_new( asm_map );
                
            }
            size_t i_vertex = counter_bis+floor(counter_bis/Nx);
            vertices(i_vertex) = sol_HHO(0,counter_bis) ;
            vertices(i_vertex+1) = sol_HHO(1,counter_bis) ;
            vertices(i_vertex+Nx+2) = sol_HHO(2,counter_bis) ;
            vertices(i_vertex+Nx+1) = sol_HHO(3,counter_bis) ;
                
        }
        std::cout<<" --> converting_into_HHO_formulation. TO BE CHECKED that sol_FEM already uploaded!"<<std::endl;
        
    }
        
    void converting_into_FE_formulation( const Eigen::Matrix<T, Dynamic, Dynamic>& values_new )
    {
          
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                sol_FEM( asm_map ) = values_new(i,counter_bis) ;
            }
            size_t i_vertex = counter_bis+floor(counter_bis/Nx);
            vertices(i_vertex) = values_new(0,counter_bis) ;
            vertices(i_vertex+1) = values_new(1,counter_bis) ;
            vertices(i_vertex+Nx+2) = values_new(2,counter_bis) ;
            vertices(i_vertex+Nx+1) = values_new(3,counter_bis) ;

        }
        std::cout<<" --> converting_into_FE_formulation. TO BE CHECKED that sol_HHO already uploaded!"<<std::endl;
       
    }
    
    
    
           
    void set_max_min()
    {
        
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        phi_max = ret0;
        phi_min = ret1;
        std::cout<<" --> set_max_min: LEVEL_SET: MAX IS "<<phi_max<< " , MIN IS "<<phi_min<<" . SI PUO TOGLIERE."<<std::endl;
    }
        
       
    // OK FINE, IT WORKS ALSO FOR AGGLOMERATED MESHES
    // It should work also for Bernstein Basis
    T operator()( const typename Mesh::node_type& node ) const
    {
        return vertices(node.ptid);
            
    }
        
     
    // OK FINE, IT WORKS ALSO FOR AGGLOMERATED MESHES -> BUT SLOW
    T operator()(const point<T,2>& pt) const
    {
        //std::cout<<"I AM IN OPERATOR() SLOW !!!!"<<std::endl;
        size_t counter=0;
            
        // It looks for in what cell the point is
        for( const auto& cl:msh.cells)
        {
            if( pt_in_cell<T,Mesh>(msh,pt,cl) )
            {
                //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
                cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
                    
                auto values_cell = (sol_HHO.block(0,counter,local_dim,1)).col(0);
               
                return values_cell.dot( cb.eval_basis(pt) );
                
            }
            counter+=1;
        }
        std::cout<<"IF HERE, THERE IS A PROBLEM IN projected_level_set::operator()!!!"<<std::endl;
        return 1e10; //to check if doesn't enter in the loop
    }

        
    // IT WORKS ONLY FOR NOT-AGGLOMERATED MESHES --> FAST
    T operator()( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
            
        size_t counter = offset(msh,cl) ;
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = (sol_HHO.block(0,counter,local_dim,1)).col(0);
        T tmp = values_cell.dot( cb.eval_basis(pt) );
           
        return tmp;
                
    }
        
    // IT WORKS ONLY FOR NOT-AGGLOMERATED MESHES --> FAST
    T operator()( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::face_type& fc ) const
    {
        auto counter_face = offset(msh,fc);
        size_t counter_cell;
        // ATTENTION, ALL THIS WORKS IN STRUCTURED QUADRANGULAR MESHES
               
        // Check if I am in the last row, upper faces (ordered differently)
        if(counter_face>=last_row_init && counter_face<=last_row_end)
        {
            counter_cell = (Ny-1)*Nx + counter_face%(last_row_init);
        }
        else
        {
            // Find in what row the face is
            auto  num_cell_row = floor(counter_face/(number_faces_one_row));
            if ( counter_face!= ( (2*Nx)*(num_cell_row+1)+num_cell_row ) )
            {
                // Faces not on the right boudary, to know in what cell are, it is sufficient to analyse the low and left face of each quadrangular cell.
                counter_cell = floor( (counter_face-num_cell_row)/2.0 );
            }
            else
            {
                // Face on the right boudary,
                counter_cell = ( num_cell_row+1 )*Nx -1;
            }
            
        }
        //std::cout<<"Face->Cell number "<<counter_cell<<std::endl;
        auto cl = msh.cells.at(counter_cell);
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = (sol_HHO.block(0,counter_cell,local_dim,1)).col(0);
        T tmp = values_cell.dot( cb.eval_basis(pt) );
        return tmp;

               
    }
        
        
    // IT WORKS FOR ALL THE MESHES --> SLOW
    /*
    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;

        ret = gradient(pt);
        return ret/ret.norm();
    }
    
         
    // IT WORKS FOR ALL THE MESHES --> SLOW
    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        size_t counter=0;
        //std::cout<<"I AM IN GRADIENT SLOW !!!!"<<std::endl;
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        for( const auto& cl:msh.cells)
        {
            if(pt_in_cell<T,Mesh>(msh,pt,cl))
            {
                cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
                //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
                    
                auto values_cell = sol_HHO.col(counter);
                auto grad_eval =  cb.eval_gradients(pt);
                ret(0) = values_cell.dot( grad_eval.col(0) );
                // std::cout<<"Value of derivative new along x"<<ret(0)<<std::endl;
                ret(1) = values_cell.dot( grad_eval.col(1) );

                return ret;
            }
            counter+=1;

        }
        std::cout<<"Se compare questo problema in gradient()"<<std::endl;
        ret(0)+=1e10;
        ret(1)+=1e10;
        return ret; //to check if doesn't enter in the loop

    }
    */
         
        // IT WORKS FOR NOT-AGGLOMERATED MESHES --> FAST
    Eigen::Matrix<T,2,1> gradient_disc( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        
        // MATRIX NOTATION
        size_t counter = offset(msh,cl);
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = sol_HHO.col(counter);
        auto grad_eval =  cb.eval_gradients(pt);
        ret(0) = values_cell.dot( grad_eval.col(0) );
        // std::cout<<"Value of derivative new along x"<<ret(0)<<std::endl;
        ret(1) = values_cell.dot( grad_eval.col(1) );
        //values_cell.dot( grad_eval.col(1) );
        // std::cout<<"Value of derivative new along y"<<ret(1)<<std::endl;
        return ret;
            
    }

     
    // IT WORKS FOR NOT-AGGLOMERATED MESHES --> FAST
    Eigen::Matrix<T,2,1> normal_disc(const point<T,2>& pt, const Mesh& msh, const typename Mesh::cell_type& cl) const
    {
        Eigen::Matrix<T,2,1> ret;
        ret = gradient_disc(pt,msh,cl);
        return ret/ret.norm();
            
    }
    /*
    T divergence_disc_old( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
            
        size_t counter = offset(msh,cl) ;
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = (sol_HHO.block(0,counter,local_dim,1)).col(0);
        
        T grad_norm = (this->gradient( pt , msh , cl )).norm() ;
    
        //std::cout<<"grad norm is "<<grad_norm<<std::endl;
        //std::cout<<"values_cell is "<<'\n'<<values_cell<<std::endl;
        //std::cout<<"cb.eval_divergence(pt) is "<<'\n'<<cb.eval_divergence(pt)<<std::endl;
        //std::cout<<"( values_cell.dot(cb.eval_divergence(pt)) ) is "<<( values_cell.dot(cb.eval_divergence(pt)) )<<std::endl;
           
        return -( values_cell.dot(cb.eval_divergence(pt)) ) / (2 * grad_norm) ;
                
    }
    */
    
    T divergence_disc( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
            
        size_t counter = offset(msh,cl) ;
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = (sol_HHO.block(0,counter,local_dim,1)).col(0);
        
        T grad_norm = (this->gradient_disc( pt , msh , cl )).norm() ;
        auto grad_eval = cb.eval_gradients(pt) ;
        //std::cout<<"grad norm is "<<grad_norm<<std::endl;
        //std::cout<<"values_cell is "<<'\n'<<values_cell<<std::endl;
        //std::cout<<"cb.eval_divergence(pt) is "<<'\n'<<cb.eval_divergence(pt)<<std::endl;
        //std::cout<<"( values_cell.dot(cb.eval_divergence(pt)) ) is "<<( values_cell.dot(cb.eval_divergence(pt)) )<<std::endl;
           
        T divergence_correction = -1.0/( pow(grad_norm,3) )*( (pow( ( values_cell.dot(grad_eval.col(0)) ) , 2)) * ( values_cell.dot(cb.eval_double_derivative_x(pt)) ) + (pow( ( values_cell.dot(grad_eval.col(1)) ) , 2)) * ( values_cell.dot(cb.eval_double_derivative_y(pt)) ) + 2.0* ( values_cell.dot(grad_eval.col(0)) )  * ( values_cell.dot(grad_eval.col(1)) ) * ( values_cell.dot(cb.eval_derivative_xy(pt)) )
                                                             ) ;
        //std::cout<<"CHECK divergence AND double derivative: in pt = "<< pt <<" error = "<< ( cb.eval_double_derivative_x(pt) + cb.eval_double_derivative_y(pt) - cb.eval_divergence(pt) ) <<std::endl;
        //T divergence_correction = values_cell.dot(cb.eval_gradients(pt).col(0))/pow(grad_norm,3)*values_cell.dot(cb.eval_double_derivative_x(pt)) + values_cell.dot(cb.eval_gradients(pt).col(1))/pow(grad_norm,3)*values_cell.dot(cb.eval_double_derivative_y(pt)) ;
      
        //std::cout<<"Res 0 = "<< values_cell.dot(cb.eval_divergence(pt)) / (grad_norm) + divergence_correction ;
        //std::cout<<"Res 1 = "<< - (values_cell.dot(cb.eval_divergence(pt)) / (grad_norm) + divergence_correction  );
        
        return -( (values_cell.dot(cb.eval_divergence(pt)) ) / (grad_norm) + divergence_correction );
                
    }
   
    
    void normal_continuous_setting()
    {
            
        
        
        
        Matrix<T, Dynamic, 1> ret0 = Matrix<T, Dynamic, 1>::Zero(ndof_FE, 1) ;
        Matrix<T, Dynamic, 1> ret1 = Matrix<T, Dynamic, 1>::Zero(ndof_FE, 1) ;
        timecounter tc ;
        tc.tic();
        
        SimplicialLLT<SparseMatrix<T> >solver_global_mass;
        solver_global_mass.compute(Global_Mass);
        tc.toc();
        std::cout<<"----> TIME: In normal_continuous_setting INVERSIONE MATRIX, time = "<<tc<<std::endl;
        //std::cout<<"sono qua 0"<<std::endl;
      
        tc.tic();
        for(auto& cl : msh.cells)
        {
            timecounter tc2 ;
            //tc2.tic();
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto cbs = cb.size();
            size_t cell_offset = offset(msh,cl);
            Matrix<T, Dynamic, 1> ret0_loc = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
            Matrix<T, Dynamic, 1> ret1_loc = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
            
            auto qps = integrate(msh, cl, 2*(degree_FEM+1)); // integration of order 2k
            //tc2.toc();
            //std::cout<<"----> TIME: pezzo 1, time = "<<tc2<<std::endl;
            //tc2.tic();
            for (auto& qp : qps)
            {
                auto phi = cb.eval_basis(qp.first);
                auto disc_normal = (this->normal_disc( qp.first , msh , cl )) ;
                //auto disc_normal = (this->normal( qp.first , msh , cl )) ;
                ret0_loc += (qp.second * disc_normal(0) * phi.transpose() );
                ret1_loc += (qp.second * disc_normal(1) * phi.transpose() );
                // phi * phi.transpose is (degree+1)^2 x (degree+1)^2 ; qp.second is a scalar
            }
            //tc2.toc();
            //std::cout<<"----> TIME: QPS, time = "<<tc2<<std::endl;
            //std::cout<<"sono qua 1"<<std::endl;
            //tc2.tic();
            for (size_t i = 0; i < local_dim; i++)
            {
                
                size_t asm_map =  connectivity_matrix[cell_offset][i].first ;
                //std::cout<<"i = "<<i<<" , asm_map = "<<asm_map<<" , cell_offset = "<<cell_offset<<" , ret0.size() = "<<ret0.size()<<" , ret1.size() = "<<ret1.size()<<" , ret0_loc.size() = "<<ret0_loc.size()<<" , ret1_loc.size() = "<<ret1_loc.size()<<std::endl;
                ret0( asm_map ) += ret0_loc(i) ;
                ret1( asm_map ) += ret1_loc(i) ;
                  
                //if( std::abs( sol_tmp(i) - sol_tmp_2(i) )>1e-14 )
                //    std::cout<< std::abs(sol_tmp(i) - sol_tmp_2(i) ) <<std::endl;
                  
            }
            //std::cout<<"sono qua 2"<<std::endl;
            //tc2.toc();
            //std::cout<<"----> TIME: pezzo 3, time = "<<tc2<<std::endl;
            
        }
        tc.toc();
        std::cout<<"----> TIME: FEM CREATION, time = "<<tc<<std::endl;
        tc.tic();
        normal_c_FEM_0 = solver_global_mass.solve(ret0);
        normal_c_FEM_1 = solver_global_mass.solve(ret1);
        tc.toc();
        std::cout<<"----> TIME: FEM RESOLUTION, time = "<<tc<<std::endl;
        tc.tic();
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                normal_c_HHO_0(i,counter_bis) = normal_c_FEM_0( asm_map ) ;
                normal_c_HHO_1(i,counter_bis) = normal_c_FEM_1( asm_map ) ;
            }
            
        }
        tc.toc();
               
        std::cout<<"----> TIME: HHO RESOLUTION, time = "<<tc<<std::endl;
        //std::cout<<"normal_c_HHO_0"<<'\n'<<normal_c_HHO_0<<std::endl;
        //std::cout<<"normal_c_HHO_1"<<'\n'<<normal_c_HHO_1<<std::endl;
                
    }
    
    Eigen::Matrix<T,2,1> normal_cont( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        size_t counter = offset(msh,cl);
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell0 = normal_c_HHO_0.col(counter);
        auto values_cell1 = normal_c_HHO_1.col(counter);
        //std::cout<<"normal_c_HHO_0"<<'\n'<<values_cell0<<std::endl;
        //std::cout<<"normal_c_HHO_1"<<'\n'<<values_cell1<<std::endl;
        auto basis_eval =  cb.eval_basis(pt);
        ret(0) = values_cell0.dot( basis_eval );
        ret(1) = values_cell1.dot( basis_eval );
        //std::cout<<"CONTINUOS NORMAL =( "<<ret(0)<<" , "<<ret(1)<<" )  in pt = "<<pt<<std::endl;
        //values_cell.dot( grad_eval.col(1) );
        // std::cout<<"Value of derivative new along y"<<ret(1)<<std::endl;
        return ret;
    
    }
    
    
    T divergence_cont( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
               
        size_t counter = offset(msh,cl) ;
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell0 = normal_c_HHO_0.col(counter);
        auto values_cell1 = normal_c_HHO_1.col(counter);
        auto grad_eval =  cb.eval_gradients(pt);
        //T grad_norm = (this->gradient( pt , msh , cl )).norm() ;
        //std::cout<<"grad norm is "<<grad_norm<<std::endl;
        //std::cout<<"values_cell is "<<'\n'<<values_cell<<std::endl;
        //std::cout<<"cb.eval_divergence(pt) is "<<'\n'<<cb.eval_divergence(pt)<<std::endl;
        //std::cout<<"( values_cell.dot(cb.eval_divergence(pt)) ) is "<<( values_cell.dot(cb.eval_divergence(pt)) )<<std::endl;
        //std::cout<<"CURVATURE( "<< pt <<" ) = "<< values_cell0.dot( grad_eval.col(0)) + values_cell1.dot( grad_eval.col(1))<<std::endl;
        return -(values_cell0.dot( grad_eval.col(0) ) + values_cell1.dot( grad_eval.col(1) ) );
        //  return -( values_cell.dot(cb.eval_divergence(pt)) ) / (2 * grad_norm) ;
                   
    }
    
    void gradient_continuous_setting()
    {
            
        
        
        
        Matrix<T, Dynamic, 1> ret0 = Matrix<T, Dynamic, 1>::Zero(ndof_FE, 1) ;
        Matrix<T, Dynamic, 1> ret1 = Matrix<T, Dynamic, 1>::Zero(ndof_FE, 1) ;
        
        SimplicialLLT<SparseMatrix<T> >solver_global_mass;
        solver_global_mass.compute(Global_Mass);
        //std::cout<<"sono qua 0"<<std::endl;
        for(auto& cl : msh.cells)
        {
            
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto cbs = cb.size();
            size_t cell_offset = offset(msh,cl);
            Matrix<T, Dynamic, 1> ret0_loc = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
            Matrix<T, Dynamic, 1> ret1_loc = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
            
            auto qps = integrate(msh, cl, 2*(degree_FEM+1)); // integration of order 2k

            for (auto& qp : qps)
            {
                auto phi = cb.eval_basis(qp.first);
                auto disc_gradient = (this->gradient_disc( qp.first , msh , cl )) ;
                //auto disc_normal = (this->normal( qp.first , msh , cl )) ;
                ret0_loc += qp.second * disc_gradient(0) * phi.transpose();
                ret1_loc += qp.second * disc_gradient(1) * phi.transpose();
                // phi * phi.transpose is (degree+1)^2 x (degree+1)^2 ; qp.second is a scalar
            }
            //std::cout<<"sono qua 1"<<std::endl;
            for (size_t i = 0; i < local_dim; i++)
            {
                
                size_t asm_map =  connectivity_matrix[cell_offset][i].first ;
                //std::cout<<"i = "<<i<<" , asm_map = "<<asm_map<<" , cell_offset = "<<cell_offset<<" , ret0.size() = "<<ret0.size()<<" , ret1.size() = "<<ret1.size()<<" , ret0_loc.size() = "<<ret0_loc.size()<<" , ret1_loc.size() = "<<ret1_loc.size()<<std::endl;
                ret0( asm_map ) += ret0_loc(i) ;
                ret1( asm_map ) += ret1_loc(i) ;
                  
                //if( std::abs( sol_tmp(i) - sol_tmp_2(i) )>1e-14 )
                //    std::cout<< std::abs(sol_tmp(i) - sol_tmp_2(i) ) <<std::endl;
                  
            }
            //std::cout<<"sono qua 2"<<std::endl;
            
            
        }
        
        gradient_c_FEM_0 = solver_global_mass.solve(ret0);
        gradient_c_FEM_1 = solver_global_mass.solve(ret1);
       
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                gradient_c_HHO_0(i,counter_bis) = gradient_c_FEM_0( asm_map ) ;
                gradient_c_HHO_1(i,counter_bis) = gradient_c_FEM_1( asm_map ) ;
            }
            
        }
        //std::cout<<"normal_c_HHO_0"<<'\n'<<normal_c_HHO_0<<std::endl;
        //std::cout<<"normal_c_HHO_1"<<'\n'<<normal_c_HHO_1<<std::endl;
                
    }
    
    Eigen::Matrix<T,2,1> gradient( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        size_t counter = offset(msh,cl);
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell0 = gradient_c_HHO_0.col(counter);
        auto values_cell1 = gradient_c_HHO_1.col(counter);
        //std::cout<<"normal_c_HHO_0"<<'\n'<<values_cell0<<std::endl;
        //std::cout<<"normal_c_HHO_1"<<'\n'<<values_cell1<<std::endl;
        auto basis_eval =  cb.eval_basis(pt);
        ret(0) = values_cell0.dot( basis_eval );
        ret(1) = values_cell1.dot( basis_eval );
        //std::cout<<"CONTINUOS NORMAL =( "<<ret(0)<<" , "<<ret(1)<<" )  in pt = "<<pt<<std::endl;
        //values_cell.dot( grad_eval.col(1) );
        // std::cout<<"Value of derivative new along y"<<ret(1)<<std::endl;
        return ret;
    
    }
    
    
    
    Eigen::Matrix<T,2,1> normal( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        // Continuous normal noramlised -> obtained via the L2 projection of the discontinuos gradient over the basis B_k.
        size_t counter = offset(msh,cl);
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell0 = gradient_c_HHO_0.col(counter);
        auto values_cell1 = gradient_c_HHO_1.col(counter);
        
        
        
        auto basis_eval =  cb.eval_basis(pt);
        ret(0) = values_cell0.dot( basis_eval );
        ret(1) = values_cell1.dot( basis_eval );
        //std::cout<<"values_cell"<<'\n'<<ret<<std::endl;
        //std::cout<<"values_cell.norm()"<<'\n'<<ret.norm()<<std::endl;
        //std::cout<<"gradient_c_HHO"<<'\n'<<ret/ret.norm()<<std::endl;
        
        return ret/ret.norm();
    
    }
    
    
    T divergence( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
            
        size_t counter = offset(msh,cl) ;
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell0 = gradient_c_HHO_0.col(counter);
        auto values_cell1 = gradient_c_HHO_1.col(counter);
       
        auto grad_eval = cb.eval_gradients(pt) ;
        auto b_eval = cb.eval_basis(pt) ;
        T grad_norm = (this->gradient( pt , msh , cl )).norm() ;
    
        //std::cout<<"grad norm is "<<grad_norm<<std::endl;
        //std::cout<<"values_cell is "<<'\n'<<values_cell<<std::endl;
        //std::cout<<"cb.eval_divergence(pt) is "<<'\n'<<cb.eval_divergence(pt)<<std::endl;
        //std::cout<<"( values_cell.dot(cb.eval_divergence(pt)) ) is "<<( values_cell.dot(cb.eval_divergence(pt)) )<<std::endl;
           
        T divergence_correction = -1.0/( pow(grad_norm,3) )*( (pow( ( values_cell0.dot(b_eval) ) , 2)) * ( values_cell0.dot(grad_eval.col(0)) ) +  ( values_cell0.dot(b_eval) ) * ( values_cell1.dot(b_eval) ) * ( values_cell1.dot(grad_eval.col(0)) ) +
                                                             ( values_cell1.dot(b_eval) ) * ( values_cell0.dot(b_eval) ) * ( values_cell0.dot(grad_eval.col(1)) ) +  (pow( ( values_cell1.dot(b_eval) ) , 2)) * ( values_cell1.dot(grad_eval.col(1)) ) );
        
        
        //T divergence_correction = values_cell.dot(cb.eval_gradients(pt).col(0))/pow(grad_norm,3)*values_cell.dot(cb.eval_double_derivative_x(pt)) + values_cell.dot(cb.eval_gradients(pt).col(1))/pow(grad_norm,3)*values_cell.dot(cb.eval_double_derivative_y(pt)) ;
      
        //std::cout<<"Res 0 = "<< values_cell.dot(cb.eval_divergence(pt)) / (grad_norm) + divergence_correction ;
        //std::cout<<"Res 1 = "<< - (values_cell.dot(cb.eval_divergence(pt)) / (grad_norm) + divergence_correction  );
        
        return -( (values_cell0.dot(grad_eval.col(0)) + values_cell1.dot(grad_eval.col(1)) ) / (grad_norm) + divergence_correction );
                
    }
    
    
    
    
    
    void smooth_cut_off( T C , T r0 , T delta , T x_centre , T y_centre , T radius , T radius_a , T radius_b )
    {
        
        Matrix<T, Dynamic,1> local_RHS = Matrix<T, Dynamic, 1>::Zero( local_dim , 1);
        //Matrix<T, Dynamic, 1> RHS = Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
        Matrix<T, Dynamic,1> sol_loc = Matrix<T, Dynamic, 1>::Zero( local_dim , 1);
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        
        
        /*
        T hx = params.hx();
        T hy = params.hy();
        T pos_r0 = 0.5; //std::min(x_centre , 1 - x_centre );
        T r_max = std::max( radius_a , radius_b ) ;
        T h = std::max( hx , hy ) ;
        T r0 = r_max + 2*h*sqrt(2.0);
        C = r0*r0*radius_b*radius_b - radius_a*radius_a*radius_b*radius_b;
         //T dist = pos_r0 - radius + 2.0*0.0625;
         //T dist = pos_r0 - radius + 2.0*0.07;
         //T r0 = radius + dist/2.0;
         
         T delta = r0/8.0; // FIRST CHOICE
        //T delta = r0/20.0;
        */
        
        
        
        
        
        // Lambda function to define smooth function
        auto alfa = [=](const point<T,2>& pt)
        { // sol
            return (1 - tanh( (sqrt( pow((pt.x()-x_centre),2) + pow((pt.y()-y_centre),2) ) - r0 ) / delta ))/2;};
        
        
        
            
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
             
            auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
            for(size_t i = 0 ; i<local_dim ; i++)
            {
                auto pt = qps[i];
                auto asm_map =  connectivity_matrix[cell_offset][i].first;
                local_RHS(i) =  ( 1.0 - alfa(pt)  ) * C + alfa(pt) * (*this)(pt,msh,cl) ;
            }
            sol_loc = vandermonde_interpolant.solve(local_RHS); // SAVE Vandermonde interpolation
            sol_HHO.col(cell_offset) = sol_loc ; // SAVE Vandermonde
            
            if(!analytic_check)
            {
                for (size_t i = 0; i < local_dim; i++)
                {
                    auto asm_map =  connectivity_matrix[cell_offset][i].first;
                    sol_FEM(asm_map) = sol_loc(i) ;
                }
                size_t i_vertex = cell_offset+floor(cell_offset/Nx);
                vertices(i_vertex) = sol_HHO(0,cell_offset) ;
                vertices(i_vertex+1) = sol_HHO(1,cell_offset) ;
                vertices(i_vertex+Nx+2) = sol_HHO(2,cell_offset) ;
                vertices(i_vertex+Nx+1) = sol_HHO(3,cell_offset) ;
            }
            
        } // end of cl loop
             
    
            
        //converting_into_HHO_formulation(sol_FEM);
        if(!analytic_check)
            set_max_min();
        

        /*
        postprocess_output<double> postoutput00;
        typedef typename Mesh::point_type       point_type;
        point<double,2> node;
        auto alfa_values = std::make_shared< gnuplot_output_object<double> >("alfa.dat");
        //auto interface_pos = std::make_shared< gnuplot_output_object<double> >("interface_alfa.dat");
        for(auto& pt:msh.points )
        {
            alfa_values->add_data(pt,alfa(pt));
        }
        postoutput00.add_object(alfa_values);
        postoutput00.write();
        
        */
    }
    
    void smooth_cut_off( T C , T x_centre , T y_centre , T radius , T radius_a , T radius_b )
    {
        
        Matrix<T, Dynamic,1> local_RHS = Matrix<T, Dynamic, 1>::Zero( local_dim , 1);
        //Matrix<T, Dynamic, 1> RHS = Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
        Matrix<T, Dynamic,1> sol_loc = Matrix<T, Dynamic, 1>::Zero( local_dim , 1);
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        
        
        
        T hx = params.hx();
        T hy = params.hy();
        T pos_r0 = 0.5; //std::min(x_centre , 1 - x_centre );
        T r_max = std::max( radius_a , radius_b ) ;
        T h = std::max( hx , hy ) ;
        T r0 = r_max + 2*h*sqrt(2.0);
        C = r0*r0*radius_b*radius_b - radius_a*radius_a*radius_b*radius_b;
         //T dist = pos_r0 - radius + 2.0*0.0625;
         //T dist = pos_r0 - radius + 2.0*0.07;
         //T r0 = radius + dist/2.0;
         
         T delta = r0/8.0; // FIRST CHOICE
        //T delta = r0/20.0;
        
        
        
        
        
        std::cout<<"r_max = "<<r_max<<" , r0 = "<<r0<<" , delta = "<<delta<<" , hx = hy = "<<hx<<std::endl;
        std::cout<<"value in alfa in r_int = "<<(radius-r0)/delta<<std::endl;
        std::cout<<"value in alfa in R = "<<(pos_r0-r0)/delta<<std::endl;
        
        // Lambda function to define smooth function
        auto alfa = [=](const point<T,2>& pt)
        { // sol
            return (1 - tanh( (sqrt( pow((pt.x()-x_centre),2) + pow((pt.y()-y_centre),2) ) - r0 ) / delta ))/2;};
        
        
        
            
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
             
            auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
            for(size_t i = 0 ; i<local_dim ; i++)
            {
                auto pt = qps[i];
                auto asm_map =  connectivity_matrix[cell_offset][i].first;
                local_RHS(i) =  ( 1.0 - alfa(pt)  ) * C + alfa(pt) * (*this)(pt,msh,cl) ;
            }
            sol_loc = vandermonde_interpolant.solve(local_RHS); // SAVE Vandermonde interpolation
            sol_HHO.col(cell_offset) = sol_loc ; // SAVE Vandermonde
            
            if(!analytic_check)
            {
                for (size_t i = 0; i < local_dim; i++)
                {
                    auto asm_map =  connectivity_matrix[cell_offset][i].first;
                    sol_FEM(asm_map) = sol_loc(i) ;
                }
                size_t i_vertex = cell_offset+floor(cell_offset/Nx);
                vertices(i_vertex) = sol_HHO(0,cell_offset) ;
                vertices(i_vertex+1) = sol_HHO(1,cell_offset) ;
                vertices(i_vertex+Nx+2) = sol_HHO(2,cell_offset) ;
                vertices(i_vertex+Nx+1) = sol_HHO(3,cell_offset) ;
            }
            
        } // end of cl loop
             
    
            
        //converting_into_HHO_formulation(sol_FEM);
        if(!analytic_check)
            set_max_min();

    }
    
    
    void cut_off( T d )
    {
        
        Matrix<T, Dynamic,1> local_RHS = Matrix<T, Dynamic, 1>::Zero( local_dim , 1);
        Matrix<T, Dynamic,1> sol_loc = Matrix<T, Dynamic, 1>::Zero( local_dim , 1);
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        
        
        // Known term (f,b_i)_i , b_i Bernstein basis fx
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
              
            auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
            for(size_t i = 0 ; i<local_dim ; i++)
            {
                auto pt = qps[i];
                auto asm_map =  connectivity_matrix[cell_offset][i].first;
                 
                if( (*this)(pt,msh,cl)>= d )
                    local_RHS(i) =  d ;
                else if( (*this)(pt,msh,cl)<= -d )
                    local_RHS(i) =  -d ;
                else
                    local_RHS(i) =  (*this)(pt,msh,cl) ;
        
            }
            
            sol_loc = vandermonde_interpolant.solve(local_RHS); // SAVE Vandermonde interpolation
            sol_HHO.col(cell_offset) = sol_loc ; // SAVE Vandermonde
            if(!analytic_check)
            {
                for (size_t i = 0; i < local_dim; i++)
                {
                    auto asm_map =  connectivity_matrix[cell_offset][i].first;
                    sol_FEM(asm_map) = sol_loc(i) ;
                }
            
                size_t i_vertex = cell_offset+floor(cell_offset/Nx);
                vertices(i_vertex) = sol_HHO(0,cell_offset) ;
                vertices(i_vertex+1) = sol_HHO(1,cell_offset) ;
                vertices(i_vertex+Nx+2) = sol_HHO(2,cell_offset) ;
                vertices(i_vertex+Nx+1) = sol_HHO(3,cell_offset) ;
            }
        } // end of cl loop
              
       
        
        
        //converting_into_HHO_formulation(sol_FEM);
        if(!analytic_check)
            set_max_min();
        
        
    }
    
        
        
};





template< typename Mesh , typename Fonction , typename FiniteSpace , typename T = typename Mesh::coordinate_type  >
struct L2projected_level_set_high_order_parallelize: public level_set<T>
{
     
    bool analytic_check = FALSE ;
    
    T phi_max , phi_min ;
    size_t  last_row_init, last_row_end, number_faces_one_row;
    
    T iso_val_interface = 0.0 ;
    Mesh msh; // Original mesh, NOT agglomerated.
    mesh_init_params<T> params; // mesh parameter
       
    size_t degree_FEM ; // FEM degree
    size_t n_cls ; // #cells
    size_t local_dim; // Local Dimension (degree_FEM+1)*(degree_FEM+1)
    size_t n_vertices ; // #vertices
    size_t      Nx, Ny ; // Number of cells in x and y direciton
    
    // connectivity matrix : for each cell, it stores the global numbering
    std::vector< std::vector<std::pair<size_t,bool>>> connectivity_matrix ;
    std::vector<std::set<size_t>> S_i;
    //std::vector<std::vector<size_t>> connectivity_matrix ;
       
    size_t dim_HHO; // Global dimension Discontinuous framework = Local dimension * #cells
    size_t ndof_FE; // Global dimension FE continuous = #nodes
       
    int mapped = 0 ; // = 0 not mapped ; = 1 mapped ; = 2 inverse mapping
    
    SparseMatrix<T>                 Global_Mass; // Global mass, saved for FEM problem
       
    Eigen::Matrix<T, Dynamic, Dynamic> sol_HHO ; // projection saved in HHO format: cell by cell
    Matrix<T, Dynamic, 1> sol_FEM ; // projection saved in Continuos FE format: global nodes
    Eigen::Matrix<T, Dynamic, 1> vertices; // saving level_set on vertices mesh

    // Local Vandermonde Matrix for interpolation
    Matrix<T, Dynamic, Dynamic> local_vandermonde ;
       
       
    // Assembling into global matrix
    SparseMatrix<T>                 Global_c_term_x; // Global mass, saved for FEM problem
    SparseMatrix<T>                 Global_c_term_y; // Global mass, saved for FEM problem
    
    Matrix<T, Dynamic, 1>      Global_Mass_Lumped; // Global mass, saved for FEM problem
    //SparseMatrix<T>                 Global_Mass_Lumped_sparse;
    
    SparseMatrix<T>         cij_norm , nij0 , nij1 ;
    SparseMatrix<T>         cji_norm , nji0 , nji1 ;
    
    
    Eigen::Matrix<T, Dynamic, Dynamic> normal_c_HHO_0 ;
    Eigen::Matrix<T, Dynamic, Dynamic> normal_c_HHO_1 ;
    Eigen::Matrix<T, Dynamic, 1> normal_c_FEM_0 ;
    Eigen::Matrix<T, Dynamic, 1> normal_c_FEM_1 ;
    
    Eigen::Matrix<T, Dynamic, Dynamic> gradient_c_HHO_0 ;
    Eigen::Matrix<T, Dynamic, Dynamic> gradient_c_HHO_1 ;
    Eigen::Matrix<T, Dynamic, 1> gradient_c_FEM_0 ;
    Eigen::Matrix<T, Dynamic, 1> gradient_c_FEM_1 ;
    
    L2projected_level_set_high_order_parallelize(const FiniteSpace& fe_data , const Fonction & level_set, const Mesh & msh , bool analytic_check = FALSE )
        : degree_FEM(fe_data.order) , local_dim(fe_data.local_ndof), msh(msh), Nx(fe_data.Nx),Ny(fe_data.Ny), params(fe_data.params) , dim_HHO(fe_data.ndof_disc) , n_cls(fe_data.n_cls) ,n_vertices(fe_data.n_vertices) , connectivity_matrix(fe_data.connectivity_matrix) , ndof_FE(fe_data.ndof_FE) , analytic_check(analytic_check)
    {
        if(!analytic_check)
        {
            timecounter tc_level_set;
            tc_level_set.tic();
            Matrix<T, Dynamic, 1>           RHS;    // Known term
           
            
            
            last_row_init = Ny*(2*Nx+1); // There are 2 faces for each row of cells + Ny
            last_row_end = last_row_init + Nx-1;
            number_faces_one_row = 2*Nx+1; // for each cell I count only the low and sx faces, respectevely 0-1 2-3 4-5 6-7 8 + the last on the right boundary
              
            Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero(local_dim);
             
            Global_Mass = SparseMatrix<T>( ndof_FE, ndof_FE ); //(b_i,b_j)_ij , b_i Lagrange basis fx
            RHS = Matrix<T, Dynamic, 1>::Zero( ndof_FE ); // Known term (f,b_i)_i , b_i Lagrange basis fx
              
            Global_c_term_x = SparseMatrix<T>( ndof_FE, ndof_FE ); //(b_i,b_j)_ij , b_i Lagrange basis fx
            Global_c_term_y = SparseMatrix<T>( ndof_FE, ndof_FE ); //(b_i,b_j)_ij , b_i Lagrange basis fx
               
            Global_Mass_Lumped = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            //Global_Mass_Lumped_sparse = SparseMatrix<T>( ndof_FE, ndof_FE ); //(b_i,b_j)_ij , b_i Lagrange basis fx
            //Global_Mass_Lumped_sparse.reserve(ndof_FE);
            
            local_vandermonde = Matrix<T, Dynamic, Dynamic>::Zero ( local_dim,local_dim );
            
            normal_c_HHO_0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
            normal_c_HHO_1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
            normal_c_FEM_0 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            normal_c_FEM_1 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            
            gradient_c_HHO_0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls);
            gradient_c_HHO_1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls);
            gradient_c_FEM_0 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            gradient_c_FEM_1 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            
            
            
            cij_norm = SparseMatrix<T>( ndof_FE, ndof_FE );
            cji_norm = SparseMatrix<T>( ndof_FE, ndof_FE );
            nij0 = SparseMatrix<T>( ndof_FE, ndof_FE );
            nij1 = SparseMatrix<T>( ndof_FE, ndof_FE );
            nji0 = SparseMatrix<T>( ndof_FE, ndof_FE );
            nji1 = SparseMatrix<T>( ndof_FE, ndof_FE );
            /*
            std::vector< Triplet<T> >       triplets_norm;
            std::vector< Triplet<T> >       triplets_norm_adj;
            std::vector< Triplet<T> >       triplets_nij0;
            std::vector< Triplet<T> >       triplets_nij1;
            std::vector< Triplet<T> >       triplets_nji0;
            std::vector< Triplet<T> >       triplets_nji1;
            */
              
            // Saving the projection in HHO discontinuous format (MATRIX NOTATION)
            sol_HHO = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
            sol_FEM = Matrix<T, Dynamic, 1>::Zero(ndof_FE);
            vertices = Eigen::Matrix<T, Dynamic, 1>::Zero( n_vertices , 1 );
              
            //Eigen::Matrix<T, Dynamic, Dynamic> sol_HHO_vandermonde =  Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls ) ;
              
              
            CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > cod;
            //FullPivLU<Matrix<T, Dynamic, Dynamic > > cod_2;
              
              
            std::cout<<"----> In 'L2projected_level_set_high_order_parallelize': Vandermonde interpolation of the level set with BERNSTEIN basis."<<std::endl;
            
            //std::cout<<"CHECK INVERSION LOCAL VANDERMONDE"<<std::endl;
            timecounter tc_prova ;
            /*
#ifdef HAVE_INTEL_TBB
            //tc_prova.tic();
            //tbb::task_scheduler_init init(1);
            tbb::task_scheduler_init init(tbb::task_scheduler_init::default_num_threads());
            auto first_cl = msh.cells[0] ;
            auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, first_cl , degree_FEM);
            cell_basis_Bernstein<Mesh,T> cb(msh, first_cl , degree_FEM);
            for (size_t ind = 0; ind < local_dim; ind++)
            {
            //tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
            //               [ & ] (size_t & ind){
                local_vandermonde.block(ind,0,1,local_dim) = (cb.eval_basis(pts[ind])).transpose() ;
                
                
            }
              //  );
                
            cod.compute(local_vandermonde);
            //tc_prova.toc();
            //std::cout << bold << yellow << "--> local_vandermonde: t = " << tc_prova << " seconds" << reset << std::endl;
            
            tc_prova.tic();
           
            std::vector< Triplet<T> >       triplets_loc; // Position elements: Sparse Matrix Notation
            std::vector< Triplet<T> >       triplets_c_term_x_loc; // Position elements: Sparse Matrix Notation
            std::vector< Triplet<T> >       triplets_c_term_y_loc; // Position elements: Sparse Matrix Notation
            //std::vector< Triplet<T> >       triplets_lumped_loc; // Position elements: Sparse Matrix Notation
            Matrix<T, Dynamic, 1> RHS_vandermonde_loc = Matrix<T, Dynamic, 1>::Zero(local_dim);
            
            size_t n_cells = msh.cells.size();
            //std::cout<<" I m in parallel zone"<<std::endl;
            timecounter tc_prova2;
            //for( const auto& cl : msh.cells )
            //{
            tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
            [&] (size_t & cell_ind){
                //tc_prova2.tic();
                //size_t cell_ind = offset(msh, cl) ;
                //std::cout<<"----> Pos 0"<<std::endl;
                auto &cl = msh.cells[cell_ind];
                //std::cout<<"---> CELL = "<<cell_ind<<std::endl;
                //std::cout<<"----> Pos 1"<<std::endl;
                //size_t cell_offset = offset(msh, cl) ;
                //std::cout<<"----> Pos 2"<<std::endl;
                auto local_mass = make_bernstein_local_mass_matrix( msh, cl , degree_FEM );
                //std::cout<<"----> Pos 3"<<std::endl;
                //tc_prova2.toc();
                //std::cout << "----> TIME --> local_mass : t = " << tc_prova2 << " seconds" << std::endl;
                //tc_prova2.tic();
                auto local_cij = make_bernstein_local_cij_matrix(msh, cl, degree_FEM);
                //std::cout<<"----> Pos 4"<<std::endl;
                //tc_prova2.toc();
    
                //std::cout << "----> TIME --> local_cij : t = " << tc_prova2 << " seconds" << std::endl;
                //tc_prova2.tic();
                auto local_mass_lumped = make_bernstein_local_mass_matrix_lumped( msh , cl , degree_FEM ) ;
                //std::cout<<"----> Pos 5"<<std::endl;
                //tc_prova2.toc();
                //std::cout << "----> TIME --> local_lumped_mass : t = " << tc_prova2 << " seconds" << std::endl;
                //tc_prova2.tic();
                // Costruction of the coefficients of the Bernstein basis
                auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
                //std::cout<<"-----> Pos 7"<<std::endl;
                //tc_prova2.toc();
                //std::cout << "----> TIME --> qps points : t = " << tc_prova2 << " seconds" << std::endl;
                //tc_prova2.tic();
                for (size_t i = 0; i < local_dim; i++)
                {
               
                    //std::cout<<"----> Pos 6"<<std::endl;
                    size_t asm_map_i = connectivity_matrix[cell_ind][i].first ;
                    for (size_t j = 0; j < local_dim; j++)
                    {
                        size_t asm_map_j = connectivity_matrix[cell_ind][j].first ;
                        //std::cout<<"i = "<<i <<" , j = "<< j <<std::endl;
                        //std::cout<<"asm_map_i = "<<asm_map_i <<" , asm_map_j = "<< asm_map_j <<std::endl;
                        //std::cout<<"-----> Pos 8.0"<<std::endl;
                        //Global_Mass.coeffRef( asm_map_i, asm_map_j ) += local_mass(i,j) ;
                        //std::cout<<"-----> Pos 8.1"<<std::endl;
                        //Global_c_term_x.coeffRef( asm_map_i, asm_map_j ) += local_cij.first(i,j);
                        //std::cout<<"-----> Pos 8.2"<<std::endl;
                        //Global_c_term_y.coeffRef( asm_map_i, asm_map_j ) += local_cij.second(i,j)  ;
                        //std::cout<<"-----> Pos 8.3"<<std::endl;
                        
                      triplets_loc.push_back(Triplet<T>(asm_map_i,asm_map_j, local_mass(i,j)));
                      triplets_c_term_x_loc.push_back( Triplet<T>(asm_map_i, asm_map_j , local_cij.first(i,j) ) );
                      triplets_c_term_y_loc.push_back( Triplet<T>( asm_map_i , asm_map_j , local_cij.second(i,j) ) );
                                         
                    }
                   
                    //std::cout<<"---------> Pos 8.4"<<std::endl;
                    //triplets_lumped_loc.push_back( Triplet<T>( asm_map_i , asm_map_i , local_mass_lumped(i) ) );
                   
                    Global_Mass_Lumped(asm_map_i) += local_mass_lumped(i);
                    RHS_vandermonde_loc(i) = level_set( qps[i]) ;
                    //std::cout<<"----> Pos 8"<<std::endl;
                }
                //tc_prova2.toc();
                //std::cout << "----> TIME --> Double loop : t = " << tc_prova2 << " seconds" << std::endl;
                //tc_prova2.tic();
                //std::cout<<"-----> Pos 9"<<std::endl;
                auto sol_tmp = cod.solve(RHS_vandermonde_loc) ;
                  
                sol_HHO.col(cell_ind) = sol_tmp ;
                for (size_t ind2 = 0; ind2 < local_dim; ind2++)
                {
                    size_t asm_map =  connectivity_matrix[cell_ind][ind2].first ;
                    sol_FEM( asm_map ) = sol_HHO(ind2,cell_ind) ;
                    
                }
                
                
                size_t i_vertex = cell_ind+floor(cell_ind/Nx);
                vertices(i_vertex) = sol_HHO(0,cell_ind) ;
                vertices(i_vertex+1) = sol_HHO(1,cell_ind) ;
                vertices(i_vertex+Nx+2) = sol_HHO(2,cell_ind) ;
                vertices(i_vertex+Nx+1) = sol_HHO(3,cell_ind) ;
                            
                //tc_prova2.toc();
                //std::cout << "----> TIME --> RESOLUTION : t = " << tc_prova2 << " seconds" << std::endl;
            }
            );
            //std::cout<<"----------------------> Pos 10"<<std::endl;
            // Finalisation global assembling
            Global_Mass.setFromTriplets( triplets_loc.begin(), triplets_loc.end() );
            triplets_loc.clear();
                  
                  
            // Finalisation global assembling
            Global_c_term_x.setFromTriplets( triplets_c_term_x_loc.begin(), triplets_c_term_x_loc.end() );
            triplets_c_term_x_loc.clear();
            Global_c_term_y.setFromTriplets( triplets_c_term_y_loc.begin(), triplets_c_term_y_loc.end() );
            triplets_c_term_y_loc.clear();
            //Global_Mass_Lumped_sparse.setFromTriplets( triplets_lumped_loc.begin(), triplets_lumped_loc.end() );
            //triplets_lumped_loc.clear();
            
            tc_prova.toc();
            std::cout << bold << yellow << "--> COSTRUCTION PROJECTION : t = " << tc_prova << " seconds" << reset << std::endl;
             
#else
     */
            std::vector< Triplet<T> >       triplets; // Position elements: Sparse Matrix Notation
            std::vector< Triplet<T> >       triplets_c_term_x; // Position elements: Sparse Matrix Notation
            std::vector< Triplet<T> >       triplets_c_term_y; // Position elements: Sparse Matrix Notation
            
            for( const auto& cl : msh.cells )
            {
                size_t cell_offset = offset(msh, cl) ;
                
                if(cell_offset == 0)
                {
                    auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree_FEM);
                    cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
                    for (size_t i = 0; i < local_dim; i++)
                    {
                        // Local vandermonde matrix
                        local_vandermonde.block(i,0,1,local_dim) = (cb.eval_basis(pts[i])).transpose() ;
                    }
                      
                    cod.compute(local_vandermonde);
                    //cod_2.compute(local_vandermonde);
                }
                
             
                auto local_mass = make_bernstein_local_mass_matrix( msh, cl , degree_FEM );
                  
                //auto local_RHS = make_bernstein_local_RHS( msh , cl , degree_FEM , level_set );
                // Local c_ij = b_i nabla(b_j) -> USEFUL FOR TRANSPORT PROBLEM
                auto local_cij = make_bernstein_local_cij_matrix (msh, cl, degree_FEM);
                      
                auto local_mass_lumped = make_bernstein_local_mass_matrix_lumped( msh , cl , degree_FEM ) ;
                 
                  
                // Costruction of the coefficients of the Bernstein basis
                auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
                  
                  
                // Assembling triplets for global problem
                for (size_t i = 0; i < local_dim; i++)
                {
                    size_t asm_map_i = connectivity_matrix[cell_offset][i].first ;
                    for (size_t j = 0; j < local_dim; j++)
                    {
                        
                        size_t asm_map_j = connectivity_matrix[cell_offset][j].first ;
                          
                        triplets.push_back(Triplet<T>(asm_map_i,asm_map_j, local_mass(i,j)));
                        triplets_c_term_x.push_back( Triplet<T>(asm_map_i, asm_map_j , local_cij.first(i,j) ) );
                        triplets_c_term_y.push_back( Triplet<T>( asm_map_i , asm_map_j , local_cij.second(i,j) ) );
                        
                        /*
                        T val_norm = sqrt( c_ij0*c_ij0 + c_ij1*c_ij1 );
                        T val_norm_adj = sqrt( c_ji0 *c_ji0 + c_ji1*c_ji1 );
                        T val_nij0 = c_ij0/val_norm ;
                        T val_nij1 = c_ij1/val_norm ;
                        T val_nji0 = c_ji0/val_norm_adj ;
                        T val_nji1 = c_ji1/val_norm_adj ;
                        
                        triplets_norm.push_back(Triplet<T>(asm_map_i,asm_map_j, val_norm));
                        triplets_norm_adj.push_back(Triplet<T>(asm_map_i,asm_map_j, val_norm_adj));
                        triplets_nij0.push_back(Triplet<T>(asm_map_i,asm_map_j, val_nij0));
                        triplets_nij1.push_back(Triplet<T>(asm_map_i,asm_map_j, val_nij1));
                        triplets_nji0.push_back(Triplet<T>(asm_map_i,asm_map_j, val_nji0));
                        triplets_nji1.push_back(Triplet<T>(asm_map_i,asm_map_j, val_nji1));
                        */
                    }
                    Global_Mass_Lumped(asm_map_i) += local_mass_lumped(i);
                    //Global_Mass_Lumped(asm_map[i].first) += local_mass_lumped(i);
                    //RHS(asm_map[i].first) += local_RHS(i) ;
                    RHS_vandermonde(i) = level_set( qps[i]) ;
                }
                  
                  
                auto sol_tmp = cod.solve(RHS_vandermonde) ;
                  
                //auto sol_tmp_2 = cod_2.solve(RHS_vandermonde) ;
                  
                sol_HHO.col(cell_offset) = sol_tmp ;
                for (size_t i = 0; i < local_dim; i++)
                {
                      
                    size_t asm_map =  connectivity_matrix[cell_offset][i].first ;
                    sol_FEM( asm_map ) = sol_HHO(i,cell_offset) ;
                      
                    //if( std::abs( sol_tmp(i) - sol_tmp_2(i) )>1e-14 )
                    //    std::cout<< std::abs(sol_tmp(i) - sol_tmp_2(i) ) <<std::endl;
                      
                }
                size_t i_vertex = cell_offset+floor(cell_offset/Nx);
                vertices(i_vertex) = sol_HHO(0,cell_offset) ;
                vertices(i_vertex+1) = sol_HHO(1,cell_offset) ;
                vertices(i_vertex+Nx+2) = sol_HHO(2,cell_offset) ;
                vertices(i_vertex+Nx+1) = sol_HHO(3,cell_offset) ;
                  
                  
            } // end of cl loop
               
            // Finalisation global assembling
            Global_Mass.setFromTriplets( triplets.begin(), triplets.end() );
            triplets.clear();
                  
                  
            // Finalisation global assembling
            Global_c_term_x.setFromTriplets( triplets_c_term_x.begin(), triplets_c_term_x.end() );
            triplets_c_term_x.clear();
            Global_c_term_y.setFromTriplets( triplets_c_term_y.begin(), triplets_c_term_y.end() );
            triplets_c_term_y.clear();
            

          
//#endif
                   
            //std::cout<<"FINE CHECK INVERSION LOCAL VANDERMONDE"<<std::endl;
           
            /*
            cij_norm.setFromTriplets( triplets_norm.begin(), triplets_norm.end() );
            triplets_norm.clear();
            cji_norm.setFromTriplets( triplets_norm_adj.begin(), triplets_norm_adj.end() );
            triplets_norm_adj.clear();
            
            nij0.setFromTriplets( triplets_nij0.begin(), triplets_nij0.end() );
            triplets_nij0.clear();
            nij1.setFromTriplets( triplets_nij1.begin(), triplets_nij1.end() );
            triplets_nij1.clear();
            
            nji0.setFromTriplets( triplets_nji0.begin(), triplets_nji0.end() );
            triplets_nji0.clear();
            nji1.setFromTriplets( triplets_nji1.begin(), triplets_nji1.end() );
            triplets_nji1.clear();
            */
            
            // NORM of c_ij
            
            cij_norm = ( Global_c_term_x.cwiseProduct(Global_c_term_x) + Global_c_term_y.cwiseProduct(Global_c_term_y) ).cwiseSqrt() ;
            //std::cout<<"cij norm "<<'\n'<<cij_norm<<std::endl;
            
            // MATRIX n_ij
            nij0 = Global_c_term_x.cwiseQuotient( cij_norm );
            nij1 = Global_c_term_y.cwiseQuotient( cij_norm );
            
            //std::cout<<"nij1  "<<'\n'<<nij1<<std::endl;
            

            // MATRIX c_ji
            SparseMatrix<T> cji_x = Global_c_term_x.adjoint() ;
            SparseMatrix<T> cji_y = Global_c_term_y.adjoint() ;
             
            // NORM of c_ji -> i.e. c_ij transposed
            cji_norm = (cji_x.cwiseProduct(cji_x)+cji_y.cwiseProduct(cji_y)).cwiseSqrt();
            
            // MATRIX n_ij (TRANSPOSED)
            nji0 = cji_x.cwiseQuotient( cji_norm );
            nji1 = cji_y.cwiseQuotient( cji_norm );
            

            
              
            //std::cout<<"local_vandermonde"<<'\n'<<local_vandermonde<<std::endl;
                  
              
              // CALCULATION OF THE SIZE + PLOTTING
              /*
              size_t size_supp_nodes = 0;
              std::cout<<"Supporting nodes IN L2:"<<std::endl;
              size_t jjjj = 0;
              for (auto& i: S_i) {
                  size_supp_nodes+=i.size();
                  std::cout <<"Node "<<jjjj<<":";
                  for (auto it=i.begin(); it != i.end(); ++it)
                      std::cout << ' ' << *it;
                      // std::cout<<ii;
                  std::cout<<'\n';
                  jjjj++;
              }
              std::cout<<std::endl;
              std::cout<<"Supporting nodes size:"<<size_supp_nodes<<std::endl;
              */
              
                  
              
                  
              //Matrix<T, Dynamic, 1> sol_FEM_vandermonde  = Matrix<T, Dynamic, 1>::Zero(RHS.rows()); ;
              
            std::cout<<"sol_FEM size "<<sol_FEM.size()<<std::endl;
            std::cout<<"local_dim size "<<local_dim<<std::endl;
            std::cout<<"n_cls "<<n_cls<<std::endl;
                  
              // Global solution saved as discontinuous HHO approach
              // Also saved min & max coefficients + position in HHO notation
              
              /*
              ConjugateGradient<SparseMatrix<T> > solver_global_mass;
              //SparseLU<SparseMatrix<T>, COLAMDOrdering<int> >solver_global_mass;
              //Notice: for this step the numerical values of A are not used
              //solver_global_mass.analyzePattern(Global_Mass);
              //solver_global_mass.factorize(Global_Mass);
               
              solver_global_mass.compute(Global_Mass); // SAVE INVERSE OF GLOBAL MASS
              if(solver_global_mass.info()!=Success) {
                  std::cout<<"FAILED SOLVER 0"<<std::endl;
                  return;
              }
              
              sol_FEM = solver_global_mass.solve(RHS);
              */
              
              /*
              for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
              {
                  for (size_t i = 0; i < local_dim; i++)
                  {
                      size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                      //sol_HHO(i,counter_bis) = sol_FEM( asm_map ) ;
                      sol_FEM( asm_map ) = sol_HHO(i,counter_bis) ;
                  }
              }
              */
              
              
              //std::cout<<"CHECK L2 proj vs Vandermonde interpolation for phi^0"<<'\n'<<sol_FEM-sol_FEM_vandermonde<<std::endl;
              
              
              //sol_FEM = sol_FEM_vandermonde ;
              //sol_HHO = sol_HHO_vandermonde ;
              //std::cout<<"CHECK L2 proj vs Vandermonde interpolation for phi^0"<<'\n'<<sol_FEM-sol_FEM_vandermonde<<std::endl;
              // Set of maximum and minimum
            timecounter tcbis ;
            tcbis.tic();
            set_max_min();
            tcbis.toc();
            std::cout << bold << yellow << "--> set_max_min: t = " << tcbis << " seconds" << reset << std::endl;

              
              /*
              for( size_t i_global = 0; i_global < n_cls; i_global++)
              {
                  size_t i_vertex = i_global+floor(i_global/Nx);
                  vertices(i_vertex) = sol_HHO(0,i_global) ;
                  vertices(i_vertex+1) = sol_HHO(1,i_global) ;
                  vertices(i_vertex+Nx+2) = sol_HHO(2,i_global) ;
                  vertices(i_vertex+Nx+1) = sol_HHO(3,i_global) ;
              }
              */
             
            tc_level_set.toc();
            std::cout << bold << yellow << "INITIALISATION LEVEL SET: t = " << tc_level_set << " seconds" << reset << std::endl;

        }
        else
        {
            timecounter tc_level_set;
            tc_level_set.tic();
            
            Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero(local_dim);
            local_vandermonde = Matrix<T, Dynamic, Dynamic>::Zero ( local_dim,local_dim );
              
              
            // Saving the projection in HHO discontinuous format (MATRIX NOTATION)
            sol_HHO = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
            normal_c_HHO_0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
            normal_c_HHO_1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
            normal_c_FEM_0 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            normal_c_FEM_1 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            
            gradient_c_HHO_0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls);
            gradient_c_HHO_1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls);
            gradient_c_FEM_0 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            gradient_c_FEM_1 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
            
            CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > cod;
            
            std::cout<<"----> FOR ANALYTIC CHECK 'L2projected_level_set_high_order_parallelize': Vandermonde interpolation of the level set with BERNSTEIN basis. JUST HHO FORMULATION IMPLEMENTED"<<std::endl;
            
        
           
#ifdef HAVE_INTEL_TBB
            auto first_cl = msh.cells[0] ;
            auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, first_cl , degree_FEM);
            cell_basis_Bernstein<Mesh,T> cb(msh, first_cl , degree_FEM);
            tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
                                      [&] (size_t & i){
                           local_vandermonde.block(i,0,1,local_dim) = (cb.eval_basis(pts[i])).transpose() ;
                           
                           
                           }
                           );
                  
            cod.compute(local_vandermonde);
            
            
            size_t n_cells = msh.cells.size();
            std::cout<<" I m in parallel zone"<<std::endl;
            tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
            [&] (size_t & cell_ind){
                auto& cl = msh.cells[cell_ind];
                size_t cell_offset = offset(msh, cl) ;
                // Assembling triplets for global problem
                auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
                    
            
                tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
                        [&] (size_t & i){
                        RHS_vandermonde(i) = level_set( qps[i]) ;
                    }
                    );
                      
                auto sol_tmp = cod.solve(RHS_vandermonde) ;
                sol_HHO.col(cell_offset) = sol_tmp ;
                
                
            }
            );
                
#else
            
            for( const auto& cl : msh.cells )
            {
                size_t cell_offset = offset(msh, cl) ;
                
                if(cell_offset == 0)
                {
                    auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree_FEM);
                    cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
                    for (size_t i = 0; i < local_dim; i++)
                    {
                        // Local vandermonde matrix
                        local_vandermonde.block(i,0,1,local_dim) = (cb.eval_basis(pts[i])).transpose() ;
                    }
                      
                    cod.compute(local_vandermonde);
                }
                  
            
                
                // Assembling triplets for global problem
                auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
                for (size_t i = 0; i < local_dim; i++)
                    RHS_vandermonde(i) = level_set( qps[i]) ;
            
                  
                auto sol_tmp = cod.solve(RHS_vandermonde) ;
                sol_HHO.col(cell_offset) = sol_tmp ;
                
                
                  
            } // end of cl loop
             
#endif
            tc_level_set.toc();
            std::cout << bold << yellow << "INITIALISATION LEVEL SET: t = " << tc_level_set << " seconds" << reset << std::endl;

        }
  
    
    }
            
            

   
    L2projected_level_set_high_order_parallelize()=default;
    
    
    void
    coefficients_mapping( )
    {
        
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        

#ifdef HAVE_INTEL_TBB
        

        size_t n_cells = msh.cells.size();
        std::cout<<" I m in parallel zone"<<std::endl;
        tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
        [&] (size_t & cell_ind){
            auto& cl = msh.cells[cell_ind];
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
                [&] (size_t & i){
                auto pt =  nodes[i] ;
                RHS_vandermonde(i) =  ( (*this)(pt , msh , cl ) - phi_min )/( phi_max - phi_min );
                                               
                }
                );
            
            
               
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
            tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
                [&] (size_t & i){
                    auto asm_map = connectivity_matrix[cell_offset][i].first ;
                    mapped_phi(asm_map) = sol_vandermonde(i) ;
                              
                                                    
                }
                );
        
            
            
        }
        );
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        
        std::cout<<"IN COEFFICIENT_MAPPING, CHECKING phi = 0, after mappin becomes --> "<< (0.0 - phi_min )/( phi_max - phi_min )<<std::endl;
               
        std::cout<<"It should be close to 1/2, error = "<< 1./2. - ( 0.0 - phi_min )/( phi_max - phi_min ) <<std::endl;
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        std::cout<<" I m in parallel zone"<<std::endl;
        tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
        [&] (size_t & cell_ind){
            auto& cl = msh.cells[cell_ind];
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
            [&] (size_t & i){
                auto nd =  nodes[i] ;
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,cell_offset,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
                                           
            }
            );
            
            }
            );
#else

        
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                RHS_vandermonde(ct) =  ( (*this)(pt , msh , cl ) - phi_min )/( phi_max - phi_min );
                ct++;
            }
            
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
         
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map = connectivity_matrix[cell_offset][i].first ;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
        } // end of cl loop
 

        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        
        std::cout<<"IN COEFFICIENT_MAPPING, CHECKING phi = 0, after mappin becomes --> "<< (0.0 - phi_min )/( phi_max - phi_min )<<std::endl;
               
        std::cout<<"It should be close to 1/2, error = "<< 1./2. - ( 0.0 - phi_min )/( phi_max - phi_min ) <<std::endl;
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
  
#endif
         
         std::cout<<"LEVEL_SET: CHECK VALUES AFTER MAPPING: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;

        
    }
    
    
    void
    coefficients_inverse_mapping( )
    {
        
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        

        
#ifdef HAVE_INTEL_TBB
        

        size_t n_cells = msh.cells.size();
        std::cout<<" I m in parallel zone"<<std::endl;
        tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
        [&] (size_t & cell_ind){
            auto& cl = msh.cells[cell_ind];
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
                [&] (size_t & i){
                auto pt =  nodes[i] ;
                RHS_vandermonde(i) = phi_min + (*this)(pt , msh , cl )*( phi_max - phi_min );
                
                                               
                }
                );
            
            
               
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
            tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
                [&] (size_t & i){
                    auto asm_map = connectivity_matrix[cell_offset][i].first ;
                    mapped_phi(asm_map) = sol_vandermonde(i) ;
                              
                                                    
                }
                );
        
            
            
        }
        );
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        std::cout<<" I m in parallel zone"<<std::endl;
        tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
        [&] (size_t & cell_ind){
            auto& cl = msh.cells[cell_ind];
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
            [&] (size_t & i){
                auto nd =  nodes[i] ;
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,cell_offset,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
                                           
            }
            );
            
        }
        );
        
#else
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                RHS_vandermonde(ct) = phi_min + (*this)(pt , msh , cl )*( phi_max - phi_min );
                ct++;
            }
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
            
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map =  connectivity_matrix[cell_offset][i].first;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
            
        } // end of cl loop
        
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        
#endif
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER MAPPING: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
           
    }
    
    
    
    
    
    
    void
    coefficients_mapping_MAX_MAX( )
    {

        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
               
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
            
#ifdef HAVE_INTEL_TBB
        

        size_t n_cells = msh.cells.size();
        std::cout<<" I m in parallel zone"<<std::endl;
        tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
        [&] (size_t & cell_ind){
            auto& cl = msh.cells[cell_ind];
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
                [&] (size_t & i){
                auto pt =  nodes[i] ;
                if( std::abs(phi_max) >= std::abs(phi_min))
                    RHS_vandermonde(i) =  ( (*this)(pt , msh , cl )+ phi_max) /( 2.0*phi_max );
                else
                    RHS_vandermonde(i) =  ( (*this)(pt , msh , cl )-phi_min ) /( -2.0*phi_min );
                
                                               
                }
                );
            
            
               
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
            tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
                [&] (size_t & i){
                    auto asm_map = connectivity_matrix[cell_offset][i].first ;
                    mapped_phi(asm_map) = sol_vandermonde(i) ;
                              
                                                    
                }
                );
        
            
            
        }
        );
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        iso_val_interface = 1.0/2.0 ;
        
        std::cout<<"IN COEFFICIENT_MAPPING MAX MAX, CHECKING phi = 0, after mappin becomes --> "<< ( 0.0 +phi_max )/( 2.0*phi_max )<<std::endl;
        std::cout<<"It should be close to 1/2, error = "<< 1./2. - ( 0.0 + phi_max )/( 2.0*phi_max  ) <<std::endl;
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        std::cout<<" I m in parallel zone"<<std::endl;
        tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
        [&] (size_t & cell_ind){
            auto& cl = msh.cells[cell_ind];
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
            [&] (size_t & i){
                auto nd =  nodes[i] ;
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,cell_offset,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
                                           
            }
            );
            
            }
            );

        
#else
      
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                if( std::abs(phi_max) >= std::abs(phi_min))
                    RHS_vandermonde(ct) =  ( (*this)(pt , msh , cl )+ phi_max) /( 2.0*phi_max );
                else
                    RHS_vandermonde(ct) =  ( (*this)(pt , msh , cl )-phi_min ) /( -2.0*phi_min );
                
                ct++;
            }
            
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
         
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map = connectivity_matrix[cell_offset][i].first ;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
        } // end of cl loop
 
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        iso_val_interface = 1.0/2.0 ;
        
        std::cout<<"IN COEFFICIENT_MAPPING MAX MAX, CHECKING phi = 0, after mappin becomes --> "<< ( 0.0 +phi_max )/( 2.0*phi_max )<<std::endl;
        std::cout<<"It should be close to 1/2, error = "<< 1./2. - ( 0.0 + phi_max )/( 2.0*phi_max  ) <<std::endl;
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
       
#endif
        
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER MAPPING_MAX_MAX: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
        
        
    }
    
    void
    coefficients_inverse_mapping_MAX_MAX( )
    {
        
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        
        
#ifdef HAVE_INTEL_TBB
        

        size_t n_cells = msh.cells.size();
        std::cout<<" I m in parallel zone"<<std::endl;
        tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
        [&] (size_t & cell_ind){
            auto& cl = msh.cells[cell_ind];
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
                [&] (size_t & i){
                auto pt =  nodes[i] ;
                if( std::abs(phi_max) >= std::abs(phi_min))
                    RHS_vandermonde(i) = -phi_max + (*this)(pt , msh , cl )* ( 2.0 * phi_max );
                else
                    RHS_vandermonde(i) = phi_min - (*this)(pt , msh , cl )* ( 2.0 * phi_min );
                
                                               
                }
                );
            
            
               
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
            tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
                [&] (size_t & i){
                    auto asm_map = connectivity_matrix[cell_offset][i].first ;
                    mapped_phi(asm_map) = sol_vandermonde(i) ;
                              
                                                    
                }
                );
        
            
            
        }
        );
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
      
        tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
        [&] (size_t & cell_ind){
            auto& cl = msh.cells[cell_ind];
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
            [&] (size_t & i){
                auto nd =  nodes[i] ;
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,cell_offset,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
                                           
            }
            );
            
            }
            );

        
        
#else
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                if( std::abs(phi_max) >= std::abs(phi_min))
                    RHS_vandermonde(ct) = -phi_max + (*this)(pt , msh , cl )* ( 2.0 * phi_max );
                else
                    RHS_vandermonde(ct) = phi_min - (*this)(pt , msh , cl )* ( 2.0 * phi_min );
                
                ct++;
            }
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
            
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map =  connectivity_matrix[cell_offset][i].first;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
            
        } // end of cl loop
        
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        
#endif
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER INVERSE_MAPPING_MAX_MAX: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
           
    }
    
    
    void
    coefficients_sfasamento( )
    {

        std::cout<<"---> coefficients_sfasamento is not parallelize for the moment."<<std::endl;
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        
     
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                RHS_vandermonde(ct) =   (*this)(pt , msh , cl ) + 0.5 ;
                
                ct++;
            }
            
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
            //std::cout<<"--------------->>>>>>>> RHS_vandermonde"<<'\n'<<RHS_vandermonde<<std::endl;
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map = connectivity_matrix[cell_offset][i].first ;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
        } // end of cl loop
       
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        iso_val_interface = 1.0/2.0 ;
        std::cout<<"Isovalue of the interface = "<<iso_val_interface<<std::endl;
        //std::cout<<"IN COEFFICIENT_MAPPING, CHECKING phi = 0, after mappin becomes --> "<< (0.0 - phi_min )/( phi_max - phi_min )<<std::endl;
               
        //std::cout<<"It should be close to 1/2, error = "<< 1./2. - ( 0.0 - phi_min )/( phi_max - phi_min ) <<std::endl;
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER MAPPING: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
        
    }
    
    
    
    
    
    void
    coefficients_mapping_quadratic( )
    {
        std::cout<<"---> coefficients_mapping_quadratic is not parallelize for the moment."<<std::endl;
        
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        
        T a1 = (-1.0/2.0 - phi_min/(phi_max-phi_min))/(pow(phi_min,2));
        T b = 1.0/(phi_max-phi_min);
        T c = 1.0/2.0;
        T a2 = (1.0/2.0 - phi_max/(phi_max-phi_min))/(pow(phi_max,2));
        
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                auto val = (*this)(pt , msh , cl ) ;
                if( val <= 0 )
                    RHS_vandermonde(ct) =   a1 * val * val + b * val + c ;
                else
                    RHS_vandermonde(ct) =   a2 * val * val + b * val + c ;
                    
                ct++;
            }
            
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
         
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map = connectivity_matrix[cell_offset][i].first ;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
        } // end of cl loop
       
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        iso_val_interface = 1.0/2.0 ;
        std::cout<<"Isovalue of the interface = "<<iso_val_interface<<std::endl;
        //std::cout<<"IN COEFFICIENT_MAPPING, CHECKING phi = 0, after mappin becomes --> "<< (0.0 - phi_min )/( phi_max - phi_min )<<std::endl;
               
        //std::cout<<"It should be close to 1/2, error = "<< 1./2. - ( 0.0 - phi_min )/( phi_max - phi_min ) <<std::endl;
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER MAPPING: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
        
    }
    
    void
    coefficients_inverse_mapping_quadratic( )
    {
        std::cout<<"---> coefficients_mapping_quadratic is not parallelize for the moment."<<std::endl;
        Matrix<T, Dynamic, 1> mapped_phi = Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        Matrix<T, Dynamic, 1> RHS_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        Matrix<T, Dynamic, 1> sol_vandermonde = Matrix<T, Dynamic, 1>::Zero( local_dim );
        
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
            
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh , cl , degree_FEM);
            size_t ct = 0;
            for (auto& pt : nodes )
            {
                RHS_vandermonde(ct) = phi_min + (*this)(pt , msh , cl )*( phi_max - phi_min );
                ct++;
            }
            sol_vandermonde = vandermonde_interpolant.solve(RHS_vandermonde);
            
            for (size_t i = 0; i < local_dim; i++)
            {
                auto asm_map =  connectivity_matrix[cell_offset][i].first;
                mapped_phi(asm_map) = sol_vandermonde(i) ;
            }
             
            
        } // end of cl loop
        
        sol_FEM = mapped_phi ;
        converting_into_HHO_formulation( sol_FEM );
        
        // CHECK MAX AND MIN AFTER OPERATIONS
        T ret0 = -10.0;
        T ret1 = 10.0;
        size_t counter_ret0 = 0;
        
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        std::cout<<"LEVEL_SET: CHECK VALUES AFTER MAPPING: MAX = "<<ret0<< " , MIN = "<<ret1<<std::endl;
           
    }
    
    
    
        
    
    
        
        
        
    void  set_discrete_points( Eigen::Matrix<T, Dynamic, Dynamic>& values_new)
    {
        sol_HHO = values_new ;
        std::cout<<" --> set_discrete_points: check that sol_FEM already uploaded!"<<std::endl;
        
    }
        
        
    void converting_into_HHO_formulation( const Eigen::Matrix<T, Dynamic, 1>& values_new )
    {
        // SAVE BOTH SOL_HHO AND VERTICES
       
#ifdef HAVE_INTEL_TBB
        

        size_t n_cells = msh.cells.size();
        std::cout<<" I m in parallel zone"<<std::endl;
        tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
        [&] (size_t & cell_ind){
            tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
                            [&] (size_t & i){
                size_t asm_map =  connectivity_matrix[cell_ind][i].first ;
                sol_HHO(i,cell_ind) = values_new( asm_map );
            }
            );
            size_t i_vertex = cell_ind+floor(cell_ind/Nx);
            vertices(i_vertex) = sol_HHO(0,cell_ind) ;
            vertices(i_vertex+1) = sol_HHO(1,cell_ind) ;
            vertices(i_vertex+Nx+2) = sol_HHO(2,cell_ind) ;
            vertices(i_vertex+Nx+1) = sol_HHO(3,cell_ind) ;
            
        }
        );
        
#else
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                sol_HHO(i,counter_bis) = values_new( asm_map );
                
            }
            size_t i_vertex = counter_bis+floor(counter_bis/Nx);
            vertices(i_vertex) = sol_HHO(0,counter_bis) ;
            vertices(i_vertex+1) = sol_HHO(1,counter_bis) ;
            vertices(i_vertex+Nx+2) = sol_HHO(2,counter_bis) ;
            vertices(i_vertex+Nx+1) = sol_HHO(3,counter_bis) ;
                
        }
#endif
            
        std::cout<<" --> converting_into_HHO_formulation. TO BE CHECKED that sol_FEM already uploaded!"<<std::endl;
        
    }
        
    void converting_into_FE_formulation( const Eigen::Matrix<T, Dynamic, Dynamic>& values_new )
    {
          
#ifdef HAVE_INTEL_TBB
        

        size_t n_cells = msh.cells.size();
        std::cout<<" I m in parallel zone"<<std::endl;
        tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
        [&] (size_t & cell_ind){
            tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
                            [&] (size_t & i){
                size_t asm_map =  connectivity_matrix[cell_ind][i].first ;
                sol_FEM( asm_map ) = values_new(i,cell_ind) ;
            }
            );
            size_t i_vertex = cell_ind+floor(cell_ind/Nx);
            vertices(i_vertex) = values_new(0,cell_ind) ;
            vertices(i_vertex+1) = values_new(1,cell_ind) ;
            vertices(i_vertex+Nx+2) = values_new(2,cell_ind) ;
            vertices(i_vertex+Nx+1) = values_new(3,cell_ind) ;
        }
        );
            
#else
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                sol_FEM( asm_map ) = values_new(i,counter_bis) ;
            }
            size_t i_vertex = counter_bis+floor(counter_bis/Nx);
            vertices(i_vertex) = values_new(0,counter_bis) ;
            vertices(i_vertex+1) = values_new(1,counter_bis) ;
            vertices(i_vertex+Nx+2) = values_new(2,counter_bis) ;
            vertices(i_vertex+Nx+1) = values_new(3,counter_bis) ;

        }
         
#endif
        std::cout<<" --> converting_into_FE_formulation. TO BE CHECKED that sol_HHO already uploaded!"<<std::endl;
       
    }
    
    
    
           
    void set_max_min()
    {
        
        T ret0 = -10.0;
        T ret1 = 10.0;
        
#ifdef HAVE_INTEL_TBB
        size_t n_cells = msh.cells.size();
        std::cout<<" I m in parallel zone"<<std::endl;
        tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
            [&] (size_t & cell_ind){
            auto& cl = msh.cells[cell_ind];
                
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            for(auto& nd : nodes ){
            //tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
            //    [&] (size_t & i){
            //    auto nd =  nodes[i] ;
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,cell_ind,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
           // );
        }
        );
        phi_max = ret0;
        phi_min = ret1;
    
#else
        size_t counter_ret0 = 0;
        for(auto& cl:msh.cells)
        {
            cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
            auto nodes = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl , degree_FEM);
            
            for(auto& nd : nodes ){
                auto phi_tmp = cb.eval_basis(nd);
                auto values_cell = (sol_HHO.block(0,counter_ret0,local_dim,1)).col(0);
                auto new_ret = values_cell.dot(phi_tmp) ;
                ret0 = std::max( new_ret , ret0 ) ;
                ret1 = std::min( new_ret , ret1);
            }
            counter_ret0++;
        }
        
        phi_max = ret0;
        phi_min = ret1;
#endif
        std::cout<<" --> set_max_min: LEVEL_SET: MAX IS "<<phi_max<< " , MIN IS "<<phi_min<<" . SI PUO TOGLIERE."<<std::endl;
    }
        
       
    // OK FINE, IT WORKS ALSO FOR AGGLOMERATED MESHES
    // It should work also for Bernstein Basis
    T operator()( const typename Mesh::node_type& node ) const
    {
        return vertices(node.ptid);
            
    }
        
     
    // OK FINE, IT WORKS ALSO FOR AGGLOMERATED MESHES -> BUT SLOW
    T operator()(const point<T,2>& pt) const
    {
        /*
#ifdef HAVE_INTEL_TBB
        size_t n_cells = msh.cells.size();
        std::cout<<" I m in parallel zone"<<std::endl;
        tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
            [&] (size_t & cell_ind){
            auto& cl = msh.cells[cell_ind];
            if( pt_in_cell<T,Mesh>(msh,pt,cl) )
            {
                cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree_FEM);
                auto values_cell = (sol_HHO.block(0,cell_ind,local_dim,1)).col(0);
                
                return values_cell.dot( cb.eval_basis(pt) );
            }
        }
        );
            
        std::cout<<"IF HERE, THERE IS A PROBLEM IN projected_level_set::operator() PARALLEL!!!"<<std::endl;
        return 1e10; //to check if doesn't enter in the loop
            
#else
        */
        //std::cout<<"I AM IN OPERATOR() SLOW !!!!"<<std::endl;
        size_t counter=0;
            
        // It looks for in what cell the point is
        for( const auto& cl:msh.cells)
        {
            if( pt_in_cell<T,Mesh>(msh,pt,cl) )
            {
                //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
                cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
                    
                auto values_cell = (sol_HHO.block(0,counter,local_dim,1)).col(0);
               
                return values_cell.dot( cb.eval_basis(pt) );
                
            }
            counter+=1;
        }
        std::cout<<"IF HERE, THERE IS A PROBLEM IN projected_level_set::operator()!!!"<<std::endl;
        return 1e10; //to check if doesn't enter in the loop
    
//#endif
    
    }

        
    // IT WORKS ONLY FOR NOT-AGGLOMERATED MESHES --> FAST
    T operator()( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
            
        size_t counter = offset(msh,cl) ;
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = (sol_HHO.block(0,counter,local_dim,1)).col(0);
        T tmp = values_cell.dot( cb.eval_basis(pt) );
           
        return tmp;
                
    }
        
    // IT WORKS ONLY FOR NOT-AGGLOMERATED MESHES --> FAST
    T operator()( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::face_type& fc ) const
    {
        auto counter_face = offset(msh,fc);
        size_t counter_cell;
        // ATTENTION, ALL THIS WORKS IN STRUCTURED QUADRANGULAR MESHES
               
        // Check if I am in the last row, upper faces (ordered differently)
        if(counter_face>=last_row_init && counter_face<=last_row_end)
        {
            counter_cell = (Ny-1)*Nx + counter_face%(last_row_init);
        }
        else
        {
            // Find in what row the face is
            auto  num_cell_row = floor(counter_face/(number_faces_one_row));
            if ( counter_face!= ( (2*Nx)*(num_cell_row+1)+num_cell_row ) )
            {
                // Faces not on the right boudary, to know in what cell are, it is sufficient to analyse the low and left face of each quadrangular cell.
                counter_cell = floor( (counter_face-num_cell_row)/2.0 );
            }
            else
            {
                // Face on the right boudary,
                counter_cell = ( num_cell_row+1 )*Nx -1;
            }
            
        }
        //std::cout<<"Face->Cell number "<<counter_cell<<std::endl;
        auto cl = msh.cells.at(counter_cell);
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = (sol_HHO.block(0,counter_cell,local_dim,1)).col(0);
        T tmp = values_cell.dot( cb.eval_basis(pt) );
        return tmp;

               
    }
        
        
    // IT WORKS FOR ALL THE MESHES --> SLOW
    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;

        ret = gradient(pt);
        return ret/ret.norm();
    }
        
         
    // IT WORKS FOR ALL THE MESHES --> SLOW
    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        
         Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
       /*
#ifdef HAVE_INTEL_TBB
               
        size_t n_cells = msh.cells.size();
               
        std::cout<<" I m in parallel zone"<<std::endl;
        tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
            [&] (size_t & cell_ind){
            auto& cl = msh.cells[cell_ind];
           if(pt_in_cell<T,Mesh>(msh,pt,cl))
            {
                cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
                //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
                    
                auto values_cell = sol_HHO.col(cell_ind);
                auto grad_eval =  cb.eval_gradients(pt);
                ret(0) = values_cell.dot( grad_eval.col(0) );
                // std::cout<<"Value of derivative new along x"<<ret(0)<<std::endl;
                ret(1) = values_cell.dot( grad_eval.col(1) );

                return ret;
            }
        }
        );
        std::cout<<"Se compare questo problema in gradient()"<<std::endl;
        ret(0)+=1e10;
        ret(1)+=1e10;
             
        return ret; //to check if doesn't enter in the loop
                    
       
#else
        */
        size_t counter=0;
        //std::cout<<"I AM IN GRADIENT SLOW !!!!"<<std::endl;
        for( const auto& cl:msh.cells)
        {
            if(pt_in_cell<T,Mesh>(msh,pt,cl))
            {
                cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
                //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
                    
                auto values_cell = sol_HHO.col(counter);
                auto grad_eval =  cb.eval_gradients(pt);
                ret(0) = values_cell.dot( grad_eval.col(0) );
                // std::cout<<"Value of derivative new along x"<<ret(0)<<std::endl;
                ret(1) = values_cell.dot( grad_eval.col(1) );

                return ret;
            }
            counter+=1;

        }
        std::cout<<"Se compare questo problema in gradient()"<<std::endl;
        ret(0)+=1e10;
        ret(1)+=1e10;
        return ret; //to check if doesn't enter in the loop
//#endif
        
    }

         
        // IT WORKS FOR NOT-AGGLOMERATED MESHES --> FAST
    Eigen::Matrix<T,2,1> gradient( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        
        // MATRIX NOTATION
        size_t counter = offset(msh,cl);
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = sol_HHO.col(counter);
        auto grad_eval =  cb.eval_gradients(pt);
        ret(0) = values_cell.dot( grad_eval.col(0) );
        // std::cout<<"Value of derivative new along x"<<ret(0)<<std::endl;
        ret(1) = values_cell.dot( grad_eval.col(1) );
        //values_cell.dot( grad_eval.col(1) );
        // std::cout<<"Value of derivative new along y"<<ret(1)<<std::endl;
        return ret;
            
    }

     
    // IT WORKS FOR NOT-AGGLOMERATED MESHES --> FAST
    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt, const Mesh& msh, const typename Mesh::cell_type& cl) const
    {
        Eigen::Matrix<T,2,1> ret;
        ret = gradient(pt,msh,cl);
        return ret/ret.norm();
            
    }
    /*
    T divergence_disc_old( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
            
        size_t counter = offset(msh,cl) ;
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = (sol_HHO.block(0,counter,local_dim,1)).col(0);
        
        T grad_norm = (this->gradient( pt , msh , cl )).norm() ;
    
        //std::cout<<"grad norm is "<<grad_norm<<std::endl;
        //std::cout<<"values_cell is "<<'\n'<<values_cell<<std::endl;
        //std::cout<<"cb.eval_divergence(pt) is "<<'\n'<<cb.eval_divergence(pt)<<std::endl;
        //std::cout<<"( values_cell.dot(cb.eval_divergence(pt)) ) is "<<( values_cell.dot(cb.eval_divergence(pt)) )<<std::endl;
           
        return -( values_cell.dot(cb.eval_divergence(pt)) ) / (2 * grad_norm) ;
                
    }
    */
    
    T divergence( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
            
        size_t counter = offset(msh,cl) ;
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = (sol_HHO.block(0,counter,local_dim,1)).col(0);
        
        T grad_norm = (this->gradient( pt , msh , cl )).norm() ;
        auto grad_eval = cb.eval_gradients(pt) ;
        //std::cout<<"grad norm is "<<grad_norm<<std::endl;
        //std::cout<<"values_cell is "<<'\n'<<values_cell<<std::endl;
        //std::cout<<"cb.eval_divergence(pt) is "<<'\n'<<cb.eval_divergence(pt)<<std::endl;
        //std::cout<<"( values_cell.dot(cb.eval_divergence(pt)) ) is "<<( values_cell.dot(cb.eval_divergence(pt)) )<<std::endl;
           
        T divergence_correction = -1.0/( pow(grad_norm,3) )*( (pow( ( values_cell.dot(grad_eval.col(0)) ) , 2)) * ( values_cell.dot(cb.eval_double_derivative_x(pt)) ) + (pow( ( values_cell.dot(grad_eval.col(1)) ) , 2)) * ( values_cell.dot(cb.eval_double_derivative_y(pt)) ) + 2.0* ( values_cell.dot(grad_eval.col(0)) )  * ( values_cell.dot(grad_eval.col(1)) ) * ( values_cell.dot(cb.eval_derivative_xy(pt)) )
                                                             ) ;
        //std::cout<<"CHECK divergence AND double derivative: in pt = "<< pt <<" error = "<< ( cb.eval_double_derivative_x(pt) + cb.eval_double_derivative_y(pt) - cb.eval_divergence(pt) ) <<std::endl;
        //T divergence_correction = values_cell.dot(cb.eval_gradients(pt).col(0))/pow(grad_norm,3)*values_cell.dot(cb.eval_double_derivative_x(pt)) + values_cell.dot(cb.eval_gradients(pt).col(1))/pow(grad_norm,3)*values_cell.dot(cb.eval_double_derivative_y(pt)) ;
      
        //std::cout<<"Res 0 = "<< values_cell.dot(cb.eval_divergence(pt)) / (grad_norm) + divergence_correction ;
        //std::cout<<"Res 1 = "<< - (values_cell.dot(cb.eval_divergence(pt)) / (grad_norm) + divergence_correction  );
        
        return -( (values_cell.dot(cb.eval_divergence(pt)) ) / (grad_norm) + divergence_correction );
                
    }
   
    
    void normal_continuous_setting()
    {
            
        
        timecounter tc ;
        tc.tic();
        Matrix<T, Dynamic, 1> ret0 = Matrix<T, Dynamic, 1>::Zero(ndof_FE, 1) ;
        Matrix<T, Dynamic, 1> ret1 = Matrix<T, Dynamic, 1>::Zero(ndof_FE, 1) ;
        
        SimplicialLLT<SparseMatrix<T> >solver_global_mass;
        solver_global_mass.compute(Global_Mass);
        
        
        tc.toc();
        std::cout<<"----> TIME: In normal_continuous_setting PARALLEL INVERSIONE MATRIX, time = "<<tc<<std::endl;
        
#ifdef HAVE_INTEL_TBB
        tbb::task_scheduler_init init(1);
        size_t n_cells = msh.cells.size();
                     
        tc.tic();
                            
        std::cout<<" I m in parallel zone"<<std::endl;
        tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
            [&] (size_t & cell_ind){
            
            //timecounter tc2 ;
            //tc2.tic();
            auto& cl = msh.cells[cell_ind];
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto cbs = cb.size();
            //size_t cell_offset = offset(msh,cl);
            Matrix<T, Dynamic, 1> ret0_loc = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
            Matrix<T, Dynamic, 1> ret1_loc = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
                   
            auto qps = integrate(msh, cl, 2*(degree_FEM+1));
            for (auto& qp : qps)
            {
            //size_t qps_size = qps.size();
            //tbb::parallel_for(size_t(0), size_t(qps_size), size_t(1),
            //    [&] (size_t & j){
            //    auto qp =  qps[j] ;
                auto phi = cb.eval_basis(qp.first);
                auto disc_normal = (this->normal( qp.first , msh , cl )) ;
                //auto disc_normal = (this->normal( qp.first , msh , cl )) ;
                ret0_loc += qp.second * disc_normal(0) * phi.transpose();
                ret1_loc += qp.second * disc_normal(1) * phi.transpose();
                       
            }
           // );
            //tc2.toc();
            //std::cout<<"----> TIME: PARALLEL QPS = "<<tc2<<std::endl;
            for (size_t i = 0; i < local_dim; i++)
            {
            //tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
            //    [&] (size_t & i){
                size_t asm_map =  connectivity_matrix[cell_ind][i].first ;
                //std::cout<<"i = "<<i<<" , asm_map = "<<asm_map<<" , cell_offset = "<<cell_offset<<" , ret0.size() = "<<ret0.size()<<" , ret1.size() = "<<ret1.size()<<" , ret0_loc.size() = "<<ret0_loc.size()<<" , ret1_loc.size() = "<<ret1_loc.size()<<std::endl;
                ret0( asm_map ) += ret0_loc(i) ;
                ret1( asm_map ) += ret1_loc(i) ;
                       
            }
            //);
        }
        );
        tc.toc();
                    
        std::cout<<"----> TIME: PARALLEL FEM CONSTRUCTION, time = "<<tc<<std::endl;
        tc.tic();
        normal_c_FEM_0 = solver_global_mass.solve(ret0);
        normal_c_FEM_1 = solver_global_mass.solve(ret1);
        tc.toc();
                    
        std::cout<<"----> TIME: FEM SOLVER, time = "<<tc<<std::endl;
        
        tc.tic();
        tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
        [&] (size_t & cell_ind){
        //    tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
        //    [&] (size_t & i){
             for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[cell_ind][i].first ;
                normal_c_HHO_0(i,cell_ind) = normal_c_FEM_0( asm_map ) ;
                normal_c_HHO_1(i,cell_ind) = normal_c_FEM_1( asm_map ) ;
            }
         //   );
            
        }
        );
        
        tc.toc();
                    
        std::cout<<"----> TIME: PARALLEL HHO, time = "<<tc<<std::endl;
        
#else
        
        
        //std::cout<<"sono qua 0"<<std::endl;
        for(auto& cl : msh.cells)
        {
            
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto cbs = cb.size();
            size_t cell_offset = offset(msh,cl);
            Matrix<T, Dynamic, 1> ret0_loc = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
            Matrix<T, Dynamic, 1> ret1_loc = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
            
            auto qps = integrate(msh, cl, 2*(degree_FEM+1)); // integration of order 2k

            for (auto& qp : qps)
            {
                auto phi = cb.eval_basis(qp.first);
                auto disc_normal = (this->normal( qp.first , msh , cl )) ;
                //auto disc_normal = (this->normal( qp.first , msh , cl )) ;
                ret0_loc += qp.second * disc_normal(0) * phi.transpose();
                ret1_loc += qp.second * disc_normal(1) * phi.transpose();
                // phi * phi.transpose is (degree+1)^2 x (degree+1)^2 ; qp.second is a scalar
            }
            //std::cout<<"sono qua 1"<<std::endl;
            for (size_t i = 0; i < local_dim; i++)
            {
                
                size_t asm_map =  connectivity_matrix[cell_offset][i].first ;
                //std::cout<<"i = "<<i<<" , asm_map = "<<asm_map<<" , cell_offset = "<<cell_offset<<" , ret0.size() = "<<ret0.size()<<" , ret1.size() = "<<ret1.size()<<" , ret0_loc.size() = "<<ret0_loc.size()<<" , ret1_loc.size() = "<<ret1_loc.size()<<std::endl;
                ret0( asm_map ) += ret0_loc(i) ;
                ret1( asm_map ) += ret1_loc(i) ;
                  
                //if( std::abs( sol_tmp(i) - sol_tmp_2(i) )>1e-14 )
                //    std::cout<< std::abs(sol_tmp(i) - sol_tmp_2(i) ) <<std::endl;
                  
            }
            //std::cout<<"sono qua 2"<<std::endl;
            
            
        }
        
        normal_c_FEM_0 = solver_global_mass.solve(ret0);
        normal_c_FEM_1 = solver_global_mass.solve(ret1);
       
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                normal_c_HHO_0(i,counter_bis) = normal_c_FEM_0( asm_map ) ;
                normal_c_HHO_1(i,counter_bis) = normal_c_FEM_1( asm_map ) ;
            }
            
        }
        
#endif
        //std::cout<<"normal_c_HHO_0"<<'\n'<<normal_c_HHO_0<<std::endl;
        //std::cout<<"normal_c_HHO_1"<<'\n'<<normal_c_HHO_1<<std::endl;
                
    }
    
    Eigen::Matrix<T,2,1> normal_cont( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        size_t counter = offset(msh,cl);
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell0 = normal_c_HHO_0.col(counter);
        auto values_cell1 = normal_c_HHO_1.col(counter);
        //std::cout<<"normal_c_HHO_0"<<'\n'<<values_cell0<<std::endl;
        //std::cout<<"normal_c_HHO_1"<<'\n'<<values_cell1<<std::endl;
        auto basis_eval =  cb.eval_basis(pt);
        ret(0) = values_cell0.dot( basis_eval );
        ret(1) = values_cell1.dot( basis_eval );
        //std::cout<<"CONTINUOS NORMAL =( "<<ret(0)<<" , "<<ret(1)<<" )  in pt = "<<pt<<std::endl;
        //values_cell.dot( grad_eval.col(1) );
        // std::cout<<"Value of derivative new along y"<<ret(1)<<std::endl;
        return ret;
    
    }
    
    
    T divergence_cont( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
               
        size_t counter = offset(msh,cl) ;
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell0 = normal_c_HHO_0.col(counter);
        auto values_cell1 = normal_c_HHO_1.col(counter);
        auto grad_eval =  cb.eval_gradients(pt);
        //T grad_norm = (this->gradient( pt , msh , cl )).norm() ;
        //std::cout<<"grad norm is "<<grad_norm<<std::endl;
        //std::cout<<"values_cell is "<<'\n'<<values_cell<<std::endl;
        //std::cout<<"cb.eval_divergence(pt) is "<<'\n'<<cb.eval_divergence(pt)<<std::endl;
        //std::cout<<"( values_cell.dot(cb.eval_divergence(pt)) ) is "<<( values_cell.dot(cb.eval_divergence(pt)) )<<std::endl;
        //std::cout<<"CURVATURE( "<< pt <<" ) = "<< values_cell0.dot( grad_eval.col(0)) + values_cell1.dot( grad_eval.col(1))<<std::endl;
        return -(values_cell0.dot( grad_eval.col(0) ) + values_cell1.dot( grad_eval.col(1) ) );
        //  return -( values_cell.dot(cb.eval_divergence(pt)) ) / (2 * grad_norm) ;
                   
    }
    
    void gradient_continuous_setting()
    {
            
        
        
        
        Matrix<T, Dynamic, 1> ret0 = Matrix<T, Dynamic, 1>::Zero(ndof_FE, 1) ;
        Matrix<T, Dynamic, 1> ret1 = Matrix<T, Dynamic, 1>::Zero(ndof_FE, 1) ;
        
        SimplicialLLT<SparseMatrix<T> >solver_global_mass;
        solver_global_mass.compute(Global_Mass);
        
        
        
              
        
#ifdef HAVE_INTEL_TBB
             
        size_t n_cells = msh.cells.size();
                              
        std::cout<<" I m in parallel zone"<<std::endl;
        tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
            [&] (size_t & cell_ind){
                    
            auto& cl = msh.cells[cell_ind];
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto cbs = cb.size();
            //size_t cell_offset = offset(msh,cl);
            Matrix<T, Dynamic, 1> ret0_loc = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
            Matrix<T, Dynamic, 1> ret1_loc = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
                           
            auto qps = integrate(msh, cl, 2*(degree_FEM+1));
            size_t qps_size = qps.size();
            tbb::parallel_for(size_t(0), size_t(qps_size), size_t(1),
            [&] (size_t & j){
             
                auto qp =  qps[j] ;
                auto phi = cb.eval_basis(qp.first);
                auto disc_gradient = (this->gradient( qp.first , msh , cl )) ;
                //auto disc_normal = (this->normal( qp.first , msh , cl )) ;
                ret0_loc += qp.second * disc_gradient(0) * phi.transpose();
                ret1_loc += qp.second * disc_gradient(1) * phi.transpose();
            }
            );
                          
                   
            tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
                [&] (size_t & i){
                size_t asm_map =  connectivity_matrix[cell_ind][i].first ;
                //std::cout<<"i = "<<i<<" , asm_map = "<<asm_map<<" , cell_offset = "<<cell_offset<<" , ret0.size() = "<<ret0.size()<<" , ret1.size() = "<<ret1.size()<<" , ret0_loc.size() = "<<ret0_loc.size()<<" , ret1_loc.size() = "<<ret1_loc.size()<<std::endl;
                ret0( asm_map ) += ret0_loc(i) ;
                ret1( asm_map ) += ret1_loc(i) ;
                            
            }
            );
        }
        );
                
          
        gradient_c_FEM_0 = solver_global_mass.solve(ret0);
        gradient_c_FEM_1 = solver_global_mass.solve(ret1);
        
        tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
            [&] (size_t & cell_ind){
            tbb::parallel_for(size_t(0), size_t(local_dim), size_t(1),
                [&] (size_t & i){
                size_t asm_map =  connectivity_matrix[cell_ind][i].first ;
                gradient_c_HHO_0(i,cell_ind) = gradient_c_FEM_0( asm_map ) ;
                gradient_c_HHO_1(i,cell_ind) = gradient_c_FEM_1( asm_map ) ;
            }
            );
                    
        }
        );
                
     
#else

        //std::cout<<"sono qua 0"<<std::endl;
        for(auto& cl : msh.cells)
        {
            
            cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
            auto cbs = cb.size();
            size_t cell_offset = offset(msh,cl);
            Matrix<T, Dynamic, 1> ret0_loc = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
            Matrix<T, Dynamic, 1> ret1_loc = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
            
            auto qps = integrate(msh, cl, 2*(degree_FEM+1)); // integration of order 2k

            for (auto& qp : qps)
            {
                auto phi = cb.eval_basis(qp.first);
                auto disc_gradient = (this->gradient( qp.first , msh , cl )) ;
                //auto disc_normal = (this->normal( qp.first , msh , cl )) ;
                ret0_loc += qp.second * disc_gradient(0) * phi.transpose();
                ret1_loc += qp.second * disc_gradient(1) * phi.transpose();
                // phi * phi.transpose is (degree+1)^2 x (degree+1)^2 ; qp.second is a scalar
            }
            //std::cout<<"sono qua 1"<<std::endl;
            for (size_t i = 0; i < local_dim; i++)
            {
                
                size_t asm_map =  connectivity_matrix[cell_offset][i].first ;
                //std::cout<<"i = "<<i<<" , asm_map = "<<asm_map<<" , cell_offset = "<<cell_offset<<" , ret0.size() = "<<ret0.size()<<" , ret1.size() = "<<ret1.size()<<" , ret0_loc.size() = "<<ret0_loc.size()<<" , ret1_loc.size() = "<<ret1_loc.size()<<std::endl;
                ret0( asm_map ) += ret0_loc(i) ;
                ret1( asm_map ) += ret1_loc(i) ;
                  
                //if( std::abs( sol_tmp(i) - sol_tmp_2(i) )>1e-14 )
                //    std::cout<< std::abs(sol_tmp(i) - sol_tmp_2(i) ) <<std::endl;
                  
            }
            //std::cout<<"sono qua 2"<<std::endl;
            
            
        }
        
        gradient_c_FEM_0 = solver_global_mass.solve(ret0);
        gradient_c_FEM_1 = solver_global_mass.solve(ret1);
       
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                gradient_c_HHO_0(i,counter_bis) = gradient_c_FEM_0( asm_map ) ;
                gradient_c_HHO_1(i,counter_bis) = gradient_c_FEM_1( asm_map ) ;
            }
            
        }
      
#endif
        //std::cout<<"normal_c_HHO_0"<<'\n'<<normal_c_HHO_0<<std::endl;
        //std::cout<<"normal_c_HHO_1"<<'\n'<<normal_c_HHO_1<<std::endl;
                
    }
    
    Eigen::Matrix<T,2,1> grad_cont( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        size_t counter = offset(msh,cl);
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell0 = gradient_c_HHO_0.col(counter);
        auto values_cell1 = gradient_c_HHO_1.col(counter);
        //std::cout<<"normal_c_HHO_0"<<'\n'<<values_cell0<<std::endl;
        //std::cout<<"normal_c_HHO_1"<<'\n'<<values_cell1<<std::endl;
        auto basis_eval =  cb.eval_basis(pt);
        ret(0) = values_cell0.dot( basis_eval );
        ret(1) = values_cell1.dot( basis_eval );
        //std::cout<<"CONTINUOS NORMAL =( "<<ret(0)<<" , "<<ret(1)<<" )  in pt = "<<pt<<std::endl;
        //values_cell.dot( grad_eval.col(1) );
        // std::cout<<"Value of derivative new along y"<<ret(1)<<std::endl;
        return ret;
    
    }
    
    
    
    Eigen::Matrix<T,2,1> normal_cont_normalised( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        // Continuous normal noramlised -> obtained via the L2 projection of the discontinuos gradient over the basis B_k.
        size_t counter = offset(msh,cl);
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell0 = gradient_c_HHO_0.col(counter);
        auto values_cell1 = gradient_c_HHO_1.col(counter);
        
        
        
        auto basis_eval =  cb.eval_basis(pt);
        ret(0) = values_cell0.dot( basis_eval );
        ret(1) = values_cell1.dot( basis_eval );
        std::cout<<"gradient_c_HHO"<<'\n'<<ret/ret.norm()<<std::endl;
        
        return ret/ret.norm();
    
    }
    
    
    T divergence_cont_grad( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
            
        size_t counter = offset(msh,cl) ;
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell0 = gradient_c_HHO_0.col(counter);
        auto values_cell1 = gradient_c_HHO_1.col(counter);
       
        auto grad_eval = cb.eval_gradients(pt) ;
        auto b_eval = cb.eval_basis(pt) ;
        T grad_norm = (this->grad_cont( pt , msh , cl )).norm() ;
    
        //std::cout<<"grad norm is "<<grad_norm<<std::endl;
        //std::cout<<"values_cell is "<<'\n'<<values_cell<<std::endl;
        //std::cout<<"cb.eval_divergence(pt) is "<<'\n'<<cb.eval_divergence(pt)<<std::endl;
        //std::cout<<"( values_cell.dot(cb.eval_divergence(pt)) ) is "<<( values_cell.dot(cb.eval_divergence(pt)) )<<std::endl;
           
        T divergence_correction = -1.0/( pow(grad_norm,3) )*( (pow( ( values_cell0.dot(b_eval) ) , 2)) * ( values_cell0.dot(grad_eval.col(0)) ) +  ( values_cell0.dot(b_eval) ) * ( values_cell1.dot(b_eval) ) * ( values_cell1.dot(grad_eval.col(0)) ) + ( values_cell1.dot(b_eval) ) * ( values_cell0.dot(b_eval) ) * ( values_cell0.dot(grad_eval.col(1)) ) +  (pow( ( values_cell1.dot(b_eval) ) , 2)) * ( values_cell1.dot(grad_eval.col(1)) ) );
        
      
        //std::cout<<"Res 0 = "<< values_cell.dot(cb.eval_divergence(pt)) / (grad_norm) + divergence_correction ;
        //std::cout<<"Res 1 = "<< - (values_cell.dot(cb.eval_divergence(pt)) / (grad_norm) + divergence_correction  );
        
        return -( (values_cell0.dot(grad_eval.col(0)) + values_cell1.dot(grad_eval.col(1)) ) / (grad_norm) + divergence_correction );
                
    }
    
    
    
    
    
    void smooth_cut_off( T C , T r0 , T delta , T x_centre , T y_centre , T radius , T radius_a , T radius_b )
    {
        
        Matrix<T, Dynamic,1> local_RHS = Matrix<T, Dynamic, 1>::Zero( local_dim , 1);
        //Matrix<T, Dynamic, 1> RHS = Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
        Matrix<T, Dynamic,1> sol_loc = Matrix<T, Dynamic, 1>::Zero( local_dim , 1);
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        
        
        /*
        T hx = params.hx();
        T hy = params.hy();
        T pos_r0 = 0.5; //std::min(x_centre , 1 - x_centre );
        T r_max = std::max( radius_a , radius_b ) ;
        T h = std::max( hx , hy ) ;
        T r0 = r_max + 2*h*sqrt(2.0);
        C = r0*r0*radius_b*radius_b - radius_a*radius_a*radius_b*radius_b;
         //T dist = pos_r0 - radius + 2.0*0.0625;
         //T dist = pos_r0 - radius + 2.0*0.07;
         //T r0 = radius + dist/2.0;
         
         T delta = r0/8.0; // FIRST CHOICE
        //T delta = r0/20.0;
        */
        
        
        
        
        
        // Lambda function to define smooth function
        auto alfa = [=](const point<T,2>& pt)
        { // sol
            return (1 - tanh( (sqrt( pow((pt.x()-x_centre),2) + pow((pt.y()-y_centre),2) ) - r0 ) / delta ))/2;};
        
        
        
            
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
             
            auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
            for(size_t i = 0 ; i<local_dim ; i++)
            {
                auto pt = qps[i];
                auto asm_map =  connectivity_matrix[cell_offset][i].first;
                local_RHS(i) =  ( 1.0 - alfa(pt)  ) * C + alfa(pt) * (*this)(pt,msh,cl) ;
            }
            sol_loc = vandermonde_interpolant.solve(local_RHS); // SAVE Vandermonde interpolation
            sol_HHO.col(cell_offset) = sol_loc ; // SAVE Vandermonde
            
            if(!analytic_check)
            {
                for (size_t i = 0; i < local_dim; i++)
                {
                    auto asm_map =  connectivity_matrix[cell_offset][i].first;
                    sol_FEM(asm_map) = sol_loc(i) ;
                }
                size_t i_vertex = cell_offset+floor(cell_offset/Nx);
                vertices(i_vertex) = sol_HHO(0,cell_offset) ;
                vertices(i_vertex+1) = sol_HHO(1,cell_offset) ;
                vertices(i_vertex+Nx+2) = sol_HHO(2,cell_offset) ;
                vertices(i_vertex+Nx+1) = sol_HHO(3,cell_offset) ;
            }
            
        } // end of cl loop
             
    
            
        //converting_into_HHO_formulation(sol_FEM);
        if(!analytic_check)
            set_max_min();
        

        /*
        postprocess_output<double> postoutput00;
        typedef typename Mesh::point_type       point_type;
        point<double,2> node;
        auto alfa_values = std::make_shared< gnuplot_output_object<double> >("alfa.dat");
        //auto interface_pos = std::make_shared< gnuplot_output_object<double> >("interface_alfa.dat");
        for(auto& pt:msh.points )
        {
            alfa_values->add_data(pt,alfa(pt));
        }
        postoutput00.add_object(alfa_values);
        postoutput00.write();
        
        */
    }
    
    void smooth_cut_off( T C , T x_centre , T y_centre , T radius , T radius_a , T radius_b )
    {
        
        Matrix<T, Dynamic,1> local_RHS = Matrix<T, Dynamic, 1>::Zero( local_dim , 1);
        //Matrix<T, Dynamic, 1> RHS = Matrix<T, Dynamic, 1>::Zero( ndof_FE , 1 );
        Matrix<T, Dynamic,1> sol_loc = Matrix<T, Dynamic, 1>::Zero( local_dim , 1);
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        
        
        
        T hx = params.hx();
        T hy = params.hy();
        T pos_r0 = 0.5; //std::min(x_centre , 1 - x_centre );
        T r_max = std::max( radius_a , radius_b ) ;
        T h = std::max( hx , hy ) ;
        T r0 = r_max + 2*h*sqrt(2.0);
        C = r0*r0*radius_b*radius_b - radius_a*radius_a*radius_b*radius_b;
         //T dist = pos_r0 - radius + 2.0*0.0625;
         //T dist = pos_r0 - radius + 2.0*0.07;
         //T r0 = radius + dist/2.0;
         
         T delta = r0/8.0; // FIRST CHOICE
        //T delta = r0/20.0;
        
        
        
        
        
        std::cout<<"r_max = "<<r_max<<" , r0 = "<<r0<<" , delta = "<<delta<<" , hx = hy = "<<hx<<std::endl;
        std::cout<<"value in alfa in r_int = "<<(radius-r0)/delta<<std::endl;
        std::cout<<"value in alfa in R = "<<(pos_r0-r0)/delta<<std::endl;
        
        // Lambda function to define smooth function
        auto alfa = [=](const point<T,2>& pt)
        { // sol
            return (1 - tanh( (sqrt( pow((pt.x()-x_centre),2) + pow((pt.y()-y_centre),2) ) - r0 ) / delta ))/2;};
        
        
        
            
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
             
            auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
            for(size_t i = 0 ; i<local_dim ; i++)
            {
                auto pt = qps[i];
                auto asm_map =  connectivity_matrix[cell_offset][i].first;
                local_RHS(i) =  ( 1.0 - alfa(pt)  ) * C + alfa(pt) * (*this)(pt,msh,cl) ;
            }
            sol_loc = vandermonde_interpolant.solve(local_RHS); // SAVE Vandermonde interpolation
            sol_HHO.col(cell_offset) = sol_loc ; // SAVE Vandermonde
            
            if(!analytic_check)
            {
                for (size_t i = 0; i < local_dim; i++)
                {
                    auto asm_map =  connectivity_matrix[cell_offset][i].first;
                    sol_FEM(asm_map) = sol_loc(i) ;
                }
                size_t i_vertex = cell_offset+floor(cell_offset/Nx);
                vertices(i_vertex) = sol_HHO(0,cell_offset) ;
                vertices(i_vertex+1) = sol_HHO(1,cell_offset) ;
                vertices(i_vertex+Nx+2) = sol_HHO(2,cell_offset) ;
                vertices(i_vertex+Nx+1) = sol_HHO(3,cell_offset) ;
            }
            
        } // end of cl loop
             
    
            
        //converting_into_HHO_formulation(sol_FEM);
        if(!analytic_check)
            set_max_min();

    }
    
    
    void cut_off( T d )
    {
        
        Matrix<T, Dynamic,1> local_RHS = Matrix<T, Dynamic, 1>::Zero( local_dim , 1);
        Matrix<T, Dynamic,1> sol_loc = Matrix<T, Dynamic, 1>::Zero( local_dim , 1);
        
        CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > vandermonde_interpolant( local_vandermonde );
        
        
        // Known term (f,b_i)_i , b_i Bernstein basis fx
        for( const auto& cl : msh.cells )
        {
            size_t cell_offset = offset(msh, cl) ;
              
            auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
            for(size_t i = 0 ; i<local_dim ; i++)
            {
                auto pt = qps[i];
                auto asm_map =  connectivity_matrix[cell_offset][i].first;
                 
                if( (*this)(pt,msh,cl)>= d )
                    local_RHS(i) =  d ;
                else if( (*this)(pt,msh,cl)<= -d )
                    local_RHS(i) =  -d ;
                else
                    local_RHS(i) =  (*this)(pt,msh,cl) ;
        
            }
            
            sol_loc = vandermonde_interpolant.solve(local_RHS); // SAVE Vandermonde interpolation
            sol_HHO.col(cell_offset) = sol_loc ; // SAVE Vandermonde
            if(!analytic_check)
            {
                for (size_t i = 0; i < local_dim; i++)
                {
                    auto asm_map =  connectivity_matrix[cell_offset][i].first;
                    sol_FEM(asm_map) = sol_loc(i) ;
                }
            
                size_t i_vertex = cell_offset+floor(cell_offset/Nx);
                vertices(i_vertex) = sol_HHO(0,cell_offset) ;
                vertices(i_vertex+1) = sol_HHO(1,cell_offset) ;
                vertices(i_vertex+Nx+2) = sol_HHO(2,cell_offset) ;
                vertices(i_vertex+Nx+1) = sol_HHO(3,cell_offset) ;
            }
        } // end of cl loop
              
       
        
        
        //converting_into_HHO_formulation(sol_FEM);
        if(!analytic_check)
            set_max_min();
        
        
    }
    
        
        
};



template< typename T , typename Mesh ,typename Level_Set,typename Fonction,typename FiniteSpace >
struct LS_cell_L2proj_high_order: public L2projected_level_set_high_order< Mesh,Fonction,FiniteSpace , T >
{
    
    typedef typename Mesh::cell_type       cell_type;
    cell_type agglo_LS_cl;
    std::vector<cell_type> subcells;
    Mesh agglo_msh;
    Level_Set level_set;
    T iso_val_interface ;
    //LS_cell(const Level_Set & level_set, const Mesh & msh, const typename Mesh::cell_type& cl)
   // : agglo_cl(cl), agglo_msh(msh), level_set(level_set){}
    // I don't know if I have to define a copyconstructor for level_set.. TO BE CHECKED!
    LS_cell_L2proj_high_order(const Level_Set & level_set_, const Mesh & msh)
    : agglo_msh(msh), level_set(level_set_), iso_val_interface(level_set_.iso_val_interface ){}
    //LS_cell(const Level_Set & level_set )
    //: level_set(level_set){}

    //LS_cell()=default;
    
    T operator()(const point<T,2>& pt) const
    {
        
        if (subcells.size()<1)
        {
            //std::cout<<"OPERATOR(): In subcells.size()=0 -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1]<<std::endl;
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = level_set.msh.cells[offset_old];
            return level_set( pt , level_set.msh , cl_old );
        }
        else
        {
            //std::cout<<"OPERATOR(): In subcell AGGLO -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1] <<std::endl;
            auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
            auto subcl = level_set.msh.cells[offset];
            return level_set( pt , level_set.msh , subcl );
        }
               
    }
    
    
    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt) const
    {
        
        if (subcells.size()<1)
        {
            //std::cout<<"NORMAL: In subcells.size()=0 -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1]<<std::endl;
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = level_set.msh.cells[offset_old];
            /// QUI HO MODIFICATO!!!! METTERE NORMAL SE NON USO LA CONTINUA
            //std::cout<<"Ho messo normal_disc anzichè normal!"<<std::endl;
            return level_set.normal( pt , level_set.msh , cl_old );
        }
        else
        {
            //std::cout<<"OPERATOR(): In subcell AGGLO -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1] <<std::endl;
            auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
            auto subcl = level_set.msh.cells[offset];
            //std::cout<<"Ho messo normal_disc anzichè normal!"<<std::endl;
            return level_set.normal( pt , level_set.msh , subcl );
        }
    }
    
    
    Eigen::Matrix<T,2,1> normal_cont(const point<T,2>& pt) const
    {
        
        if (subcells.size()<1)
        {
            //std::cout<<"NORMAL: In subcells.size()=0 -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1]<<std::endl;
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = level_set.msh.cells[offset_old];
            //std::cout<<"Ho messo normal anzichè normal_continuous!"<<std::endl;
            return level_set.normal_cont( pt , level_set.msh , cl_old );
        }
        else
        {
            //std::cout<<"OPERATOR(): In subcell AGGLO -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1] <<std::endl;
            auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
            auto subcl = level_set.msh.cells[offset];
            //std::cout<<"Ho messo normal anzichè normal_continuous!"<<std::endl;
            return level_set.normal_cont( pt , level_set.msh , subcl );
        }
    }
    
    Eigen::Matrix<T,2,1> normal_cont_normalised(const point<T,2>& pt) const
    {
        
        if (subcells.size()<1)
        {
            //std::cout<<"NORMAL: In subcells.size()=0 -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1]<<std::endl;
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = level_set.msh.cells[offset_old];
            //std::cout<<"Ho messo normal anzichè normal_continuous!"<<std::endl;
            return level_set.normal_cont_normalised( pt , level_set.msh , cl_old );
        }
        else
        {
            //std::cout<<"OPERATOR(): In subcell AGGLO -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1] <<std::endl;
            auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
            auto subcl = level_set.msh.cells[offset];
            //std::cout<<"Ho messo normal anzichè normal_continuous!"<<std::endl;
            return level_set.normal_cont_normalised( pt , level_set.msh , subcl );
        }
    }
    
    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        
        if (subcells.size()<1)
        {
           //std::cout<<"GRADIENT: In subcells.size()=0 -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1]<<std::endl;
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = level_set.msh.cells[offset_old];
            return level_set.gradient( pt , level_set.msh , cl_old );
        }
        else
        {
            //std::cout<<"OPERATOR(): In subcell AGGLO -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1] <<std::endl;
            auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
            auto subcl = level_set.msh.cells[offset];
            return level_set.gradient( pt , level_set.msh , subcl );
        }
        
    }
    
    /*
    T divergence_disc_old( const point<T,2>& pt )const
    {
        
        if (subcells.size()<1)
        {
            //std::cout<<"DIVERGENCE: In subcells.size()=0 -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1]<<std::endl;
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = level_set.msh.cells[offset_old];
            return level_set.divergence_disc_old( pt , level_set.msh , cl_old );
        }
        else
        {
            //std::cout<<"OPERATOR(): In subcell AGGLO -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1] <<std::endl;
            auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
            auto subcl = level_set.msh.cells[offset];
            return level_set.divergence_disc_old( pt , level_set.msh , subcl );
        }
    }
    */
    
    T divergence( const point<T,2>& pt )const
    {
        
        if (subcells.size()<1)
        {
            //std::cout<<"DIVERGENCE: In subcells.size()=0 -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1]<<std::endl;
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = level_set.msh.cells[offset_old];
            return level_set.divergence( pt , level_set.msh , cl_old );
        }
        else
        {
            //std::cout<<"OPERATOR(): In subcell AGGLO -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1] <<std::endl;
            auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
            auto subcl = level_set.msh.cells[offset];
            return level_set.divergence( pt , level_set.msh , subcl );
        }
    }
    
    T divergence_cont( const point<T,2>& pt )const
    {
        
        if (subcells.size()<1)
        {
            //std::cout<<"DIVERGENCE: In subcells.size()=0 -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1]<<std::endl;
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = level_set.msh.cells[offset_old];
            return level_set.divergence_cont( pt , level_set.msh , cl_old );
        }
        else
        {
            //std::cout<<"OPERATOR(): In subcell AGGLO -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1] <<std::endl;
            auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
            auto subcl = level_set.msh.cells[offset];
            return level_set.divergence_cont( pt , level_set.msh , subcl );
        }
    }
    
    T divergence_cont_grad( const point<T,2>& pt )const
    {
        
        if (subcells.size()<1)
        {
            //std::cout<<"DIVERGENCE: In subcells.size()=0 -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1]<<std::endl;
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = level_set.msh.cells[offset_old];
            return level_set.divergence_cont_grad( pt , level_set.msh , cl_old );
        }
        else
        {
            //std::cout<<"OPERATOR(): In subcell AGGLO -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1] <<std::endl;
            auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
            auto subcl = level_set.msh.cells[offset];
            return level_set.divergence_cont_grad( pt , level_set.msh , subcl );
        }
    }
    
    
    /*
    void cell_assignment(const cell_type& cl)
    {
        subcells.clear();
        agglo_LS_cl = cl;
        if (agglo_LS_cl.user_data.offset_subcells.size()>1 )
        {

            for (auto& offset_subcells:agglo_LS_cl.user_data.offset_subcells)
            {
                subcells.push_back( level_set.msh.cells[offset_subcells] );
            
            }
        }
            
    }
    */
    // STARE ATTENTI QUA TORNAREEEE
    void cell_assignment(const cell_type& cl)
    {
        subcells.clear();
        agglo_LS_cl = cl;
        
        if( (agglo_LS_cl.user_data.offset_subcells.size()>1) &&  (agglo_LS_cl.user_data.offset_subcells[0] != agglo_LS_cl.user_data.offset_subcells[1] ) )
        {

            for (auto& offset_subcells:agglo_LS_cl.user_data.offset_subcells)
            {
                subcells.push_back( level_set.msh.cells[offset_subcells] );
            
            }
        }
            
    }
    
    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt , const cell_type& cl)
    {
        //agglo_LS_cl = cl;
        cell_assignment(cl);
        std::cout<<"Ho messo normal_disc anzichè normal!"<<std::endl;
        return normal( pt );
    }
    
    T operator()(const point<T,2>& pt,const cell_type& cl )
    {
        //agglo_LS_cl = cl;
        cell_assignment(cl);
        return operator()( pt );
    }
    
    
};


template< typename T , typename Mesh ,typename Level_Set,typename Fonction,typename FiniteSpace >
struct LS_cell_L2proj_high_order_grad_cont: public L2projected_level_set_high_order_grad_cont< Mesh,Fonction,FiniteSpace , T >
{
    
    typedef typename Mesh::cell_type       cell_type;
    cell_type agglo_LS_cl;
    std::vector<cell_type> subcells;
    Mesh agglo_msh;
    Level_Set level_set;
    T iso_val_interface ;
    //LS_cell(const Level_Set & level_set, const Mesh & msh, const typename Mesh::cell_type& cl)
   // : agglo_cl(cl), agglo_msh(msh), level_set(level_set){}
    // I don't know if I have to define a copyconstructor for level_set.. TO BE CHECKED!
    LS_cell_L2proj_high_order_grad_cont(const Level_Set & level_set_, const Mesh & msh)
    : agglo_msh(msh), level_set(level_set_), iso_val_interface(level_set_.iso_val_interface ){}
    //LS_cell(const Level_Set & level_set )
    //: level_set(level_set){}

    //LS_cell()=default;
    
    T operator()(const point<T,2>& pt) const
    {
        
        if (subcells.size()<1)
        {
            //std::cout<<"OPERATOR(): In subcells.size()=0 -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1]<<std::endl;
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = level_set.msh.cells[offset_old];
            return level_set( pt , level_set.msh , cl_old );
        }
        else
        {
            //std::cout<<"OPERATOR(): In subcell AGGLO -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1] <<std::endl;
            auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
            auto subcl = level_set.msh.cells[offset];
            return level_set( pt , level_set.msh , subcl );
        }
               
    }
    
    
    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt) const
    {
        
        if (subcells.size()<1)
        {
            //std::cout<<"NORMAL: In subcells.size()=0 -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1]<<std::endl;
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = level_set.msh.cells[offset_old];
            /// QUI HO MODIFICATO!!!! METTERE NORMAL SE NON USO LA CONTINUA
            //std::cout<<"Ho messo normal_disc anzichè normal!"<<std::endl;
            return level_set.normal( pt , level_set.msh , cl_old );
        }
        else
        {
            //std::cout<<"OPERATOR(): In subcell AGGLO -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1] <<std::endl;
            auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
            auto subcl = level_set.msh.cells[offset];
            //std::cout<<"Ho messo normal_disc anzichè normal!"<<std::endl;
            return level_set.normal( pt , level_set.msh , subcl );
        }
    }
    
    
    Eigen::Matrix<T,2,1> normal_cont(const point<T,2>& pt) const
    {
        
        if (subcells.size()<1)
        {
            //std::cout<<"NORMAL: In subcells.size()=0 -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1]<<std::endl;
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = level_set.msh.cells[offset_old];
            //std::cout<<"Ho messo normal anzichè normal_continuous!"<<std::endl;
            return level_set.normal_cont( pt , level_set.msh , cl_old );
        }
        else
        {
            //std::cout<<"OPERATOR(): In subcell AGGLO -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1] <<std::endl;
            auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
            auto subcl = level_set.msh.cells[offset];
            //std::cout<<"Ho messo normal anzichè normal_continuous!"<<std::endl;
            return level_set.normal_cont( pt , level_set.msh , subcl );
        }
    }
    
    Eigen::Matrix<T,2,1> normal_disc(const point<T,2>& pt) const
    {
        
        if (subcells.size()<1)
        {
            //std::cout<<"NORMAL: In subcells.size()=0 -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1]<<std::endl;
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = level_set.msh.cells[offset_old];
            //std::cout<<"Ho messo normal anzichè normal_continuous!"<<std::endl;
            return level_set.normal_disc( pt , level_set.msh , cl_old );
        }
        else
        {
            //std::cout<<"OPERATOR(): In subcell AGGLO -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1] <<std::endl;
            auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
            auto subcl = level_set.msh.cells[offset];
            //std::cout<<"Ho messo normal anzichè normal_continuous!"<<std::endl;
            return level_set.normal_disc( pt , level_set.msh , subcl );
        }
    }
    
    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        
        if (subcells.size()<1)
        {
           //std::cout<<"GRADIENT: In subcells.size()=0 -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1]<<std::endl;
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = level_set.msh.cells[offset_old];
            return level_set.gradient( pt , level_set.msh , cl_old );
        }
        else
        {
            //std::cout<<"OPERATOR(): In subcell AGGLO -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1] <<std::endl;
            auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
            auto subcl = level_set.msh.cells[offset];
            return level_set.gradient( pt , level_set.msh , subcl );
        }
        
    }
    
    /*
    T divergence_disc_old( const point<T,2>& pt )const
    {
        
        if (subcells.size()<1)
        {
            //std::cout<<"DIVERGENCE: In subcells.size()=0 -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1]<<std::endl;
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = level_set.msh.cells[offset_old];
            return level_set.divergence_disc_old( pt , level_set.msh , cl_old );
        }
        else
        {
            //std::cout<<"OPERATOR(): In subcell AGGLO -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1] <<std::endl;
            auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
            auto subcl = level_set.msh.cells[offset];
            return level_set.divergence_disc_old( pt , level_set.msh , subcl );
        }
    }
    */
    
    T divergence( const point<T,2>& pt )const
    {
        
        if (subcells.size()<1)
        {
            //std::cout<<"DIVERGENCE: In subcells.size()=0 -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1]<<std::endl;
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = level_set.msh.cells[offset_old];
            return level_set.divergence( pt , level_set.msh , cl_old );
        }
        else
        {
            //std::cout<<"OPERATOR(): In subcell AGGLO -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1] <<std::endl;
            auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
            auto subcl = level_set.msh.cells[offset];
            return level_set.divergence( pt , level_set.msh , subcl );
        }
    }
    
    T divergence_cont( const point<T,2>& pt )const
    {
        
        if (subcells.size()<1)
        {
            //std::cout<<"DIVERGENCE: In subcells.size()=0 -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1]<<std::endl;
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = level_set.msh.cells[offset_old];
            return level_set.divergence_cont( pt , level_set.msh , cl_old );
        }
        else
        {
            //std::cout<<"OPERATOR(): In subcell AGGLO -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1] <<std::endl;
            auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
            auto subcl = level_set.msh.cells[offset];
            return level_set.divergence_cont( pt , level_set.msh , subcl );
        }
    }
    
    T divergence_disc( const point<T,2>& pt )const
    {
        
        if (subcells.size()<1)
        {
            //std::cout<<"DIVERGENCE: In subcells.size()=0 -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1]<<std::endl;
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = level_set.msh.cells[offset_old];
            return level_set.divergence_disc( pt , level_set.msh , cl_old );
        }
        else
        {
            //std::cout<<"OPERATOR(): In subcell AGGLO -----> OFFSET AGGLO[0] = "<<agglo_LS_cl.user_data.offset_subcells[0]<<" , OFFSET AGGLO[1] = " << agglo_LS_cl.user_data.offset_subcells[1] <<std::endl;
            auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
            auto subcl = level_set.msh.cells[offset];
            return level_set.divergence_disc( pt , level_set.msh , subcl );
        }
    }
    
    
    /*
    void cell_assignment(const cell_type& cl)
    {
        subcells.clear();
        agglo_LS_cl = cl;
        if (agglo_LS_cl.user_data.offset_subcells.size()>1 )
        {

            for (auto& offset_subcells:agglo_LS_cl.user_data.offset_subcells)
            {
                subcells.push_back( level_set.msh.cells[offset_subcells] );
            
            }
        }
            
    }
    */
    // STARE ATTENTI QUA TORNAREEEE
    void cell_assignment(const cell_type& cl)
    {
        subcells.clear();
        agglo_LS_cl = cl;
        
        if( (agglo_LS_cl.user_data.offset_subcells.size()>1) &&  (agglo_LS_cl.user_data.offset_subcells[0] != agglo_LS_cl.user_data.offset_subcells[1] ) )
        {

            for (auto& offset_subcells:agglo_LS_cl.user_data.offset_subcells)
            {
                subcells.push_back( level_set.msh.cells[offset_subcells] );
            
            }
        }
            
    }
    
    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt , const cell_type& cl)
    {
        //agglo_LS_cl = cl;
        cell_assignment(cl);
        std::cout<<"Ho messo normal_disc anzichè normal!"<<std::endl;
        return normal( pt );
    }
    
    T operator()(const point<T,2>& pt,const cell_type& cl )
    {
        //agglo_LS_cl = cl;
        cell_assignment(cl);
        return operator()( pt );
    }
    
    
};




template<typename T, typename Mesh ,typename Fonction >
struct projected_level_set: public level_set<T>
{
    std::vector< T > values;
    Eigen::Matrix<T, Dynamic, Dynamic> values_bis; // MATRIX NOTATION
    //Eigen::Matrix<T, Dynamic, 1> values_bis1; // VECTOR NOTATION
    Eigen::Matrix<T, Dynamic, 1> vertices; // saving level_set on vertices mesh
    std::vector< size_t > boundary_cells;
    // In the case in which I use FEM with Qk , k>1 I need another vector for all the values in each node
    
    size_t degree_FEM;
    size_t number_elements;
    Mesh msh;
    size_t      Nx, Ny ;
    mesh_init_params<T> params;
    size_t  last_row_init, last_row_end, number_faces_one_row;
  //  size_t counter_cell , counter_face, num_cell_row;
    
    T phi_max , phi_min ;
    T cut_level;
    
    
    projected_level_set(const Fonction & level_set, const Mesh & msh, size_t degree_k , const mesh_init_params<T>& params)
        : number_elements((degree_k+1)*(degree_k+1)), msh(msh),degree_FEM(degree_k),Nx(params.Nx),Ny(params.Ny), params(params)
    {
        
        last_row_init = Ny*(2*Nx+1); // There are 2 faces for each row of cells + Ny
        last_row_end = last_row_init + Nx-1;
        number_faces_one_row = 2*Nx+1; // for each cell I count only the low and sx faces, respectevely 0-1 2-3 4-5 6-7 8 + the last on the right boundary
        vertices = Eigen::Matrix<T, Dynamic, 1>::Zero( ((Nx+1)*(Ny+1)), 1 );
       
    //#ifdef NODES
        // MATRIX NOTATION
        values_bis= Eigen::Matrix<T, Dynamic, Dynamic>::Zero(number_elements, msh.cells.size());
        // VECTOR NOTATION
        //values_bis1= Eigen::Matrix<T, Dynamic, 1>::Zero(number_elements*msh.cells.size(), 1 );
        
        //std::cout<<"Number of cells "<<msh.cells.size()<<std::endl;
        
        // MATRIX NOTATION
        // std::cout<<"Dimension of the basis "<<values_bis.size()<<std::endl;
        // VECTOR NOTATION
        // std::cout<<"Dimension of the basis "<<values_bis1.size()<<std::endl;
        size_t i_global = 0 , i_local=0 , i_vertex=0;
        for(auto& cl:msh.cells)
        {
            /*
            bool boundary_bool = FALSE;
            for (auto& fc:faces(msh,cl)) {
                if (boundary_bool)
                    break;
    
                if(fc.is_boundary && !boundary_bool){
                    boundary_cells.push_back(offset(msh,cl));
                    boundary_bool = TRUE;
                }
            }
            */
            
            auto qps = equidistriduted_nodes<T,Mesh>(msh, cl, degree_FEM);
            i_local = 0;
            for ( const auto & qp : qps)
            {
                
                values.push_back( level_set(qp) ); // I DONT KNOW IF IT IS USEFUL
                
               // if (boundary_bool) {
               //      values_bis(i_local,i_global) = 1 ;  // MATRIX NOTATION
               // }
                values_bis(i_local,i_global) = level_set(qp) ;  // MATRIX NOTATION
                //values_bis1(i_local+i_global) = level_set(qp) ; // VECTOR NOTATION
                i_vertex = i_global+floor(i_global/Nx);
                if( i_local==0 )
                    vertices(i_vertex) = level_set(qp) ;
                
                if( i_local==1 )
                    vertices(i_vertex+1) = level_set(qp) ;
                
                if( i_local==(degree_FEM+2) )
                    vertices(i_vertex+Nx+2) = level_set(qp) ;
                
                if( i_local==(degree_FEM+1) )
                    vertices(i_vertex+Nx+1) = level_set(qp) ;
                i_local++;
            }
             i_global++;  // MATRIX NOTATION
           //  i_global+=number_elements;       // VECTOR NOTATION
        }
    //#endif
        
        
    }
    
    projected_level_set()=default;
    
    
    void  set_discrete_points( Eigen::Matrix<T, Dynamic, Dynamic>& values_new)
    {
        values_bis=values_new;
    }
    
    
    void converting_into_HHO_formulation( const Eigen::Matrix<T, Dynamic, 1>& vertices )
    {
        assert(degree_FEM == 1);
        for (size_t j = 0; j < values_bis.cols() ; j++)
        {

            for (size_t i = 0; i < values_bis.rows() ; i++)
            {
                auto i_vertex = j + floor(j/Nx);
                if ( i == 0 )
                    values_bis(i,j) = vertices(i_vertex);
                
                if( i ==1 )
                    values_bis(i,j) = vertices(i_vertex+1) ;
                
                if( i ==(degree_FEM+2) )
                    values_bis(i,j) = vertices(i_vertex+Nx+2) ;
                
                if( i ==(degree_FEM+1) )
                    values_bis(i,j) = vertices(i_vertex+Nx+1) ;
            }
        }
                 
    }
    
    void converting_into_FE_formulation( const Eigen::Matrix<T, Dynamic, Dynamic>& values )
    {
        assert(degree_FEM == 1);
        for (size_t j = 0; j < values_bis.cols() ; j++)
        {

            for (size_t i = 0; i < values_bis.rows() ; i++)
            {
                auto i_vertex = j + floor(j/Nx);
                if ( i == 0 )
                    vertices(i_vertex) = values(i,j) ;
                
                if( i ==1 )
                    vertices(i_vertex+1) = values(i,j) ;
                
                if( i ==(degree_FEM+2) )
                    vertices(i_vertex+Nx+2) = values(i,j) ;
                
                if( i ==(degree_FEM+1) )
                    vertices(i_vertex+Nx+1) = values(i,j) ;
            }
        }
                 
    }
       
    
    
    // OK FINE, IT WORKS ALSO FOR AGGLOMERATED MESHES
    T operator()( const typename Mesh::node_type& node ) const
    {
        // Optimised to check the value of the level set only in the vertices
        // std::cout<<"Value in vertices "<<vertices(node.ptid)<<", at position "<<node.ptid<<std::endl;
        return vertices(node.ptid);
        
    }
    
 
    // OK FINE, IT WORKS ALSO FOR AGGLOMERATED MESHES
    T operator()(const point<T,2>& pt) const
    {
        //std::cout<<"I AM IN OPERATOR() SLOW !!!!"<<std::endl;
        size_t counter=0;
        
        // It looks for in what cell the point is
        for( const auto& cl:msh.cells)
        {
            //std::cout<<"pt_in_cell operator in slow levelset"<<std::endl;
            if( pt_in_cell<T,Mesh>(msh,pt,cl) )
            {
                //cell_basis_Qk<Mesh,T> cb(msh, cl, degree_FEM);
                cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
               
                auto values_cell = (values_bis.block(0,counter,number_elements,1)).col(0);
                return values_cell.dot( cb.eval_basis(pt) );
                
               // T tmp=0;
               // for(auto i = 0; i<number_elements ; i++)
               // {
               //     tmp += (values.at(i+counter))*(cb.eval_basis(pt))[i];
               // }
               // return tmp;
            }
            //counter+=number_elements; // OLD VERSION OF THE CODE
            counter+=1;
        }
        std::cout<<"IF HERE, THERE IS A PROBLEM IN projected_level_set::operator()!!!"<<std::endl;
        return 1e10; //to check if doesn't enter in the loop
    }

    // IT WORKS ONLY FOR NOT-AGGLOMERATED MESHES
    T operator()( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        
        size_t counter = offset(msh,cl) ;
        // Checking if cell is agglomerated or not
        /*
        size_t cell_points = (points(msh,cl)).size();
        size_t cells_number = msh.cells.size();
        bool agglomeration=FALSE;
        if ( cells_number < (Nx*Ny-1) ) {
            agglomeration = TRUE;
        }
        if ( cells_number > (Nx*Ny-1) ) {
                  // throw std::logic_error("shouldn't have arrived here...");
               }
        */
        
        //if(cell_points<5)
        //{
            // MATRIX NOTATION
         //   if (!agglomeration) {
         //       counter = offset(msh,cl);
         //    }
         //   else{
         //       counter = 1;//cells_counter(msh_origin,msh,cl);
         //   }
        
            //std::cout<<"Value of offset "<<counter<<std::endl;
            //cell_basis_Qk<Mesh,T> cb(msh, cl, degree_FEM);
        cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = (values_bis.block(0,counter,number_elements,1)).col(0);
        T tmp = values_cell.dot( cb.eval_basis(pt) );
        return tmp;
            
            // VECTOR NOTATION
            
            //size_t counter = offset(msh,cl)*number_elements;
            //cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
            //auto values_cell = (values_bis1.segment(counter,number_elements));
            //T tmp = values_cell.dot( cb.eval_basis(pt) );
            //return tmp;
       // }
        /*
        else
        {
            std::vector<size_t> indices = subcell_finder<T,Mesh>(msh, pt , cl , params);
            cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM,indices);
            size_t counter = 1; // DA TROVAREEE!!!!!!!!!!!!!!!
            auto values_cell = (values_bis.block(0,counter,number_elements,1)).col(0);
            T tmp = values_cell.dot( cb.eval_basis(pt) );
            return tmp;
        }
        */
        
        
    }
    
 // IT WORKS ONLY FOR NOT-AGGLOMERATED MESHES
    T operator()( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::face_type& fc ) const
       {
           // MATRIX NOTATION
           auto counter_face = offset(msh,fc);
           size_t counter_cell;
           // da fc devo trovare la cella in cui sono per la base
           //std::cout<<"Face number "<<counter_face<<std::endl;

           // ATTENTION, ALL THIS WORKS IN STRUCTURED QUADRANGULAR MESHES
           
           // Check if I am in the last row, upper faces (ordered differently)
           if(counter_face>=last_row_init && counter_face<=last_row_end)
           {
               counter_cell = (Ny-1)*Nx + counter_face%(last_row_init);
           }
           else
           {
            // Find in what row the face is
             auto  num_cell_row = floor(counter_face/(number_faces_one_row));
               if ( counter_face!= ( (2*Nx)*(num_cell_row+1)+num_cell_row ) )
               {
                // Faces not on the right boudary, to know in what cell are, it is sufficient to analyse the low and left face of each quadrangular cell.
                   counter_cell = floor( (counter_face-num_cell_row)/2.0 );
               }
               else
               {
                // Face on the right boudary,
                   counter_cell = ( num_cell_row+1 )*Nx -1;
               }
           
           }
           //std::cout<<"Face->Cell number "<<counter_cell<<std::endl;
           auto cl = msh.cells.at(counter_cell);
           //cell_basis_Qk<Mesh,T> cb(msh, cl, degree_FEM);
           cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
           auto values_cell = (values_bis.block(0,counter_cell,number_elements,1)).col(0);
           T tmp = values_cell.dot( cb.eval_basis(pt) );
           return tmp;

           
       }
    
    

    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;

        ret = gradient(pt);
        return ret/ret.norm();
    }
    
    
    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        size_t counter=0;
        //std::cout<<"I AM IN GRADIENT SLOW !!!!"<<std::endl;
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        for( const auto& cl:msh.cells)
        {
            // std::cout<<"pt_in_cell gradient in slow levelset"<<std::endl;
            if(pt_in_cell<T,Mesh>(msh,pt,cl))
            {
                //cell_basis_Qk<Mesh,T> cb(msh, cl, degree_FEM);
                cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
                
                auto values_cell = values_bis.col(counter);
                auto grad_eval =  cb.eval_gradients(pt);
                ret(0) = values_cell.dot( grad_eval.col(0) );
                // std::cout<<"Value of derivative new along x"<<ret(0)<<std::endl;
                ret(1) = values_cell.dot( grad_eval.col(1) );

                return ret;
            }
            //counter+=number_elements; // OLD VERSION OF THE CODE
            counter+=1;

        }
        std::cout<<"Se compare questo problema in gradient()"<<std::endl;
        ret(0)+=1e10;
        ret(1)+=1e10;
        return ret; //to check if doesn't enter in the loop

    }

    
    Eigen::Matrix<T,2,1> gradient( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
    
        // MATRIX NOTATION
        size_t counter = offset(msh,cl);
        //std::cout<<"the cell in NEW is the number "<<counter<<std::endl;
        //std::cout<<"Value of offset "<<counter<<std::endl;
        //cell_basis_Qk<Mesh,T> cb(msh, cl, degree_FEM);
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = values_bis.col(counter);
        auto grad_eval =  cb.eval_gradients(pt);
        ret(0) = values_cell.dot( grad_eval.col(0) );
       // std::cout<<"Value of derivative new along x"<<ret(0)<<std::endl;
        ret(1) = values_cell.dot( grad_eval.col(1) );
        //values_cell.dot( grad_eval.col(1) );
       // std::cout<<"Value of derivative new along y"<<ret(1)<<std::endl;
        return ret;
        
        // VECTOR NOTATION
        
        //size_t counter = offset(msh,cl)*number_elements;
        //cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
        //auto values_cell = (values_bis1.segment(counter,number_elements));
        //T tmp = values_cell.dot( cb.eval_basis(pt) );
        //return tmp;
        
        
    }


    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt, const Mesh& msh, const typename Mesh::cell_type& cl) const
    {
        Eigen::Matrix<T,2,1> ret;
        ret = gradient(pt,msh,cl);
        return ret/ret.norm();
        
    }
    
    
    
    
    
    // template<typename T , typename MATRIX , typename VECTOR >
    void smooth_cut_off( T C , T x_centre , T y_centre , T radius )
    {
        T hx = params.hx();
        T hy = params.hy();
        cut_level = C;
        // T r0 = radius + 2*sqrt(pow(hx,2) + pow(hy,2)); // check value
        //T r0 = radius + 2.5*std::max(hx , hy ); // check value
        //T r0 = radius + 2.5*0.0625; // check value
        T pos_r0 = 0.45; //std::min(x_centre , 1 - x_centre );
        // IF I wanna define like that I need to pass the value all'inizio, sennò cambia per valure le exact solution (cambiando x_centre).
        
        //T max_r0 = pos_r0-(pos_r0 - 2*0.0625); //2.5*std::max(hx,hy); // if mesh too much coarse
        //r0 = std::max(r0 , max_r0);
        //r0 = 0.25*max_r0 + 0.75*r0 ;
        //if(r0<radius)
          //  r0 += 0.0625 ;
        T dist = pos_r0 - radius + 2*0.0625;
        T r0 = radius + dist/2;
        T delta = r0/8;
        //T delta = 0.0625 ; // r0/20.0; // or better std::max(hx,hy) ??
        //T delta = sqrt(pow(hx,2) + pow(hy,2));
        //if(std::abs(r0-delta-radius)<2*std::max(hx,hy) )
         //   delta = delta/2;
       
        std::cout<<"radius = "<<radius<<" , r0 = "<<r0<<" , delta = "<<delta<<" , hx = hy = "<<hx<<std::endl;
        std::cout<<"value in alfa in r_int = "<<(radius-r0)/delta<<std::endl;
        std::cout<<"value in alfa in R = "<<(pos_r0-r0)/delta<<std::endl;
        auto alfa = [x_centre , y_centre , delta , r0](const typename Mesh::point_type& pt) { // sol
            return (1 - tanh( (sqrt(pow((pt.x()-x_centre),2) + pow((pt.y()-y_centre),2)) - r0 ) / delta ))/2;};
 
        size_t counter = 0;
       // Eigen::Matrix<T, Dynamic, 1> tmp_vert = Matrix<T, Dynamic, 1>::Zero(vertices.rows(), 1); ;
        for(auto& pt:msh.points )
        {
           // std::cout<<"the point is "<< pt<<" alfa is "<<alfa(pt)<<std::endl;
            vertices(counter) = ( 1 - alfa(pt)  )*C + alfa(pt)*vertices(counter);
            counter++;
        }
       
        converting_into_HHO_formulation(vertices);
        phi_max = vertices.maxCoeff();
        phi_min = vertices.minCoeff(); // DEVO CAMBIARLO O VA BENE COSI?
        
        postprocess_output<double> postoutput00;
        //typedef typename Mesh::point_type       point_type;
        point<double,2> node;
        auto alfa_values = std::make_shared< gnuplot_output_object<double> >("alfa.dat");
        //auto interface_pos = std::make_shared< gnuplot_output_object<double> >("interface_alfa.dat");
        for(auto& pt:msh.points )
               {
                  alfa_values->add_data(pt,alfa(pt));
               }
        postoutput00.add_object(alfa_values);
        postoutput00.write();
    }
    
    
    
    
   // template<typename T , typename MATRIX , typename VECTOR >
    void cut_off( T d )
    {
        //auto vertices_abs = vertices.cwiseAbs();
        //auto phi_max_abs = d*vertices_abs.maxCoeff();
        //std::cout<<"MAX VALUE OF PHI ABS"<<phi_max_abs<<std::endl;
        
        cut_level = d ;
        //std::cout<<"CUTTING AT d = "<<d<<std::endl;
        T level_set_max = vertices.maxCoeff();
        //std::cout<<"MAX VALUE OF PHI BEFORE CUTTING = "<<level_set_max<<std::endl;
        //assert(degree_FEM == 1)
        
        Eigen::Matrix<T, Dynamic, Dynamic> One_mat = Eigen::Matrix<T, Dynamic, Dynamic>::Ones(values_bis.rows(), values_bis.cols() ); // MATRIX NOTATION
           //Eigen::Matrix<T, Dynamic, 1> values_bis1; // VECTOR NOTATION
        Eigen::Matrix<T, Dynamic, 1> One_vec = Eigen::Matrix<T, Dynamic, 1>::Ones( vertices.rows(), 1 ); // saving level_set on vertices mesh
        
        
       auto cut_off_level_vec = d*One_vec ;
       auto cut_off_level_mat = d*One_mat ;
        
        vertices/=(level_set_max);
        values_bis/=(level_set_max);
       // std::cout<<"MAX VALUE OF NORMALISED PHI = "<<vertices.maxCoeff()<<std::endl;
        auto vertices_prova = vertices;
        auto values_bis_prova = values_bis ;
        
        vertices = vertices.cwiseMin(cut_off_level_vec);
        values_bis = values_bis.cwiseMin(cut_off_level_mat);
        // Cut off also in inner domain
        vertices = vertices.cwiseMax(-cut_off_level_vec);
        values_bis = values_bis.cwiseMax(-cut_off_level_mat);
        
        // NOT NORMALISED CUT
        //vertices = vertices.cwiseMin(cut_off_level_vec*level_set_max);
        //values_bis = values_bis.cwiseMin(cut_off_level_mat*level_set_max);
        
        
        phi_max = vertices.maxCoeff();
        phi_min = vertices.minCoeff(); // DEVO CAMBIARLO O VA BENE COSI?
        //std::cout<<"MAX VALUE OF PHI_CUT AND NORMALISED = "<<phi_max<<std::endl;
        std::cout<<"Cut at "<<phi_max<<" , MIN VALUE OF PHI_CUT AND NORMALISED = "<<phi_min<<" (NEGATIVE cut off ACTIVED)."<<std::endl;
        
        /*
        postprocess_output<double> postoutput2;
        typedef typename Mesh::point_type       point_type;
        point<double,2> node;
        auto test_activated_nodes = std::make_shared< gnuplot_output_object<double> >("test_activated_nodes.dat");
        auto test_before_cut_off = std::make_shared< gnuplot_output_object<double> >("test_before_cut_off.dat");
        
        for (size_t i = 0; i<vertices_prova.rows(); i++) {
            node = msh.points.at(i);
            test_before_cut_off->add_data(node,vertices_prova(i));
    
            if( std::abs(vertices_prova(i) - vertices(i))>1e-10 )
                vertices_prova(i) = 1;
            else
                vertices_prova(i) = 0;
            test_activated_nodes->add_data(node,vertices_prova(i));
            
            
        }
        
      
        for (size_t j = 0; j<values_bis.cols(); j++)
        {
            bool active_cell= FALSE;
            size_t i = 0;
            while (i<values_bis.rows() && !active_cell ) {
                if( values_bis_prova(i,j)!= values_bis(i,j) ){
                    active_cell = TRUE;
                //std::cout<<"values_bis_prova(i,j)= "<<values_bis_prova(i,j)<<" , values_bis(i,j)= "<<values_bis(i,j)<<std::endl;
              //  std::cout<<" the cell num "<<j<<" is active."<<std::endl;
                }
                i++;
            }
        }
        
        
        
        postoutput2.add_object(test_before_cut_off);
        postoutput2.add_object(test_activated_nodes);
        postoutput2.write();
        */
        /*
        for (size_t i = 0 ; i<values_bis.rows() ; i++ ) {
            for (size_t j = 0 ; j<values_bis.cols() ; j++ ) {
                
            }
            
        }
            */
        
    }

    void boundary_con( T d )
    {
        T level_set_max = vertices.maxCoeff();
        std::cout<<"MAX VALUE OF PHI"<<level_set_max<<std::endl;
        //assert(degree_FEM == 1)
        
        Eigen::Matrix<T, Dynamic, Dynamic> One_mat = Eigen::Matrix<T, Dynamic, Dynamic>::Ones(values_bis.rows(), values_bis.cols() ); // MATRIX NOTATION
           //Eigen::Matrix<T, Dynamic, 1> values_bis1; // VECTOR NOTATION
        Eigen::Matrix<T, Dynamic, 1> One_vec = Eigen::Matrix<T, Dynamic, 1>::Ones( vertices.rows(), 1 ); // saving level_set on vertices mesh
        
        
       auto cut_off_level_vec = d*One_vec ;
       auto cut_off_level_mat = d*One_mat ;
        
        vertices/=(level_set_max);
        values_bis/=(level_set_max);
        vertices = vertices.cwiseMin(cut_off_level_vec);
        values_bis = values_bis.cwiseMin(cut_off_level_mat);
        /*
        for (size_t i = 0 ; i<values_bis.rows() ; i++ ) {
            for (size_t j = 0 ; j<values_bis.cols() ; j++ ) {
                
            }
            
        }
            */
        
    }
    
    
    
    
};





template<typename T>
class velocity_field
{
   public:
    virtual Eigen::Matrix<T,2,1> operator()(const point<T,2>& pt) const
    {
    }
    
    /*
     virtual Eigen::Matrix<T,2,1> flux(const point<T,2>& pt) const
    {
    }
    
    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;

        ret = gradient(pt);
        return ret/ret.norm();
    }
    */
   
};



template<typename T>
struct rotational_velocity_field: public velocity_field<T>
{
    T u1, xc , yc;

    rotational_velocity_field(T u1 )
    : u1(u1){}

    rotational_velocity_field(T xc , T yc , T u1 )
    : u1(u1) , xc(xc) , yc(yc)
    {}

    Eigen::Matrix<T,2,1> operator()(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        ret(0) = (  u1*pt.y() -xc )/xc ;
        ret(1) = (- u1*pt.x() + yc ) / yc ;
       
        return ret;
    }
    
    
};

template<typename T>
struct linear_velocity_field: public velocity_field<T>
{
    T u1, u2 , u3 , u4;

    linear_velocity_field(T u1 , T u2 , T u3 , T u4 )
        : u1(u1), u2(u2) , u3(u3) , u4(u4)
    {}

    Eigen::Matrix<T,2,1> operator()(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        ret(0) = u1*pt.x() + u2;
        ret(1) = u3*pt.y() + u4;
       
        return ret;
    }
/*
    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        ret(0) = 2*pt.x() - 2*alpha;
        ret(1) = 2*pt.y() - 2*beta;
        return ret;
    }
    */
};




template< typename Mesh , typename Fonction , typename FiniteSpace , typename T = typename Mesh::coordinate_type  >
struct projection_velocity_high_order: public velocity_field<T>
{
  
     size_t  last_row_init, last_row_end, number_faces_one_row;
    
    
     // NEW IMPLEMENTATION STUFF
     Mesh msh; // Original mesh, NOT agglomerated.
     mesh_init_params<T> params; // mesh parameter
     
     size_t degree_FEM ; // FEM degree
     size_t n_cls ; // #cells
     size_t local_dim; // Local Dimension (degree_FEM+1)*(degree_FEM+1)
     size_t n_vertices ; // #vertices
      size_t      Nx, Ny ; // Number of cells in x and y direciton
     // connectivity matrix : for each cell, it stores the global numbering
     std::vector< std::vector<std::pair<size_t,bool>>> connectivity_matrix ;
     //std::vector<std::vector<size_t>> connectivity_matrix ;
     
     size_t dim_HHO; // Global dimension Discontinuous framework = Local dimension * #cells
     size_t ndof_FE; // Global dimension FE continuous = #nodes
    
     T u_max0 , u_max1 ;
     
     
     //SparseMatrix<T>                 Global_Mass; // Global mass, saved for FEM problem
     //Matrix<T, Dynamic, 1>           RHS0;    // Known term 1
     //Matrix<T, Dynamic, 1>           RHS1;    // Known term 1
     //std::vector< Triplet<T> >       triplets; // Position elements: Sparse Matrix Notation
     
     std::pair<Eigen::Matrix<T, Dynamic, Dynamic> , Eigen::Matrix<T, Dynamic, Dynamic>> sol_HHO ; // projection saved in HHO format: cell by cell
     std::pair<Matrix<T, Dynamic, 1>,Matrix<T, Dynamic, 1>> sol_FEM ; // projection saved in Continuos FE format: global nodes
     //std::pair<Eigen::Matrix<T, Dynamic, 1>,Matrix<T, Dynamic, 1>> vertices; // saving level_set on vertices mesh

     
    
     projection_velocity_high_order(const FiniteSpace& fe_data , const Mesh & msh )
           : degree_FEM(fe_data.order) , local_dim(fe_data.local_ndof), msh(msh), Nx(fe_data.Nx),Ny(fe_data.Ny), params(fe_data.params) , dim_HHO(fe_data.ndof_disc) , n_cls(fe_data.n_cls) ,n_vertices(fe_data.n_vertices) , connectivity_matrix(fe_data.connectivity_matrix) , ndof_FE(fe_data.ndof_FE)
       {
           std::cout<<"STO USANDO: equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree)"<<std::endl;
           //last_row_init = Ny*(2*Nx+1); // There are 2 faces for each row of cells + Ny
           //last_row_end = last_row_init + Nx-1;
           //number_faces_one_row = 2*Nx+1; // for each cell I count only the low and sx faces, respectevely 0-1 2-3 4-5 6-7 8 + the last on the right boundary
           
           
           //vertices = std::make_pair( Eigen::Matrix<T, Dynamic, 1>::Zero( ( (Nx+1)*(Ny+1) ), 1 ) , Eigen::Matrix<T, Dynamic, 1>::Zero( ((Nx+1)*(Ny+1)), 1 ) );

           
           //vertices = std::make_pair( Eigen::Matrix<T, Dynamic, 1>::Zero( n_vertices , 1 ),Eigen::Matrix<T, Dynamic, 1>::Zero( n_vertices , 1 ) );
           sol_FEM = std::make_pair( Matrix<T, Dynamic, 1>::Zero(ndof_FE) , Matrix<T, Dynamic, 1>::Zero(ndof_FE) );
          
           sol_HHO = std::make_pair(Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls ) , Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls ) ) ;

           
       }
    
    projection_velocity_high_order(const FiniteSpace& fe_data , const Fonction & u, const Mesh & msh)
        : degree_FEM(fe_data.order) , local_dim(fe_data.local_ndof), msh(msh), Nx(fe_data.Nx),Ny(fe_data.Ny), params(fe_data.params) , dim_HHO(fe_data.ndof_disc) , n_cls(fe_data.n_cls) ,n_vertices(fe_data.n_vertices) , connectivity_matrix(fe_data.connectivity_matrix) , ndof_FE(fe_data.ndof_FE)
    {
        //last_row_init = Ny*(2*Nx+1); // There are 2 faces for each row of cells + Ny
        //last_row_end = last_row_init + Nx-1;
        //number_faces_one_row = 2*Nx+1; // for each cell I count only the low and sx faces, respectevely 0-1 2-3 4-5 6-7 8 + the last on the right boundary
     
        // Saving the projection in HHO discontinuous format (MATRIX NOTATION)
         //vertices = std::make_pair( Eigen::Matrix<T, Dynamic, 1>::Zero( n_vertices , 1 ),Eigen::Matrix<T, Dynamic, 1>::Zero( n_vertices , 1 ) );
         sol_FEM = std::make_pair( Matrix<T, Dynamic, 1>::Zero(ndof_FE) , Matrix<T, Dynamic, 1>::Zero(ndof_FE) );
         
         sol_HHO = std::make_pair(Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls ) , Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls ) ) ;

        
        size_t i_global = 0 , i_local=0 ; //, i_vertex=0;
        for(auto& cl:msh.cells)
        {
            auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
            i_local = 0;
            for ( const auto & qp : qps)
            {
                auto asm_map = connectivity_matrix[i_global][i_local].first ;
                auto u0 = (u(qp))(0) ;
                auto u1 = (u(qp))(1) ;
                
                sol_HHO.first(i_local,i_global) = u0 ;
                sol_HHO.second(i_local,i_global) = u1 ;
                sol_FEM.first(asm_map) = u0 ;
                sol_FEM.second(asm_map) = u1 ;
           /*
                i_vertex = i_global+floor(i_global/Nx);
                if( i_local==0 ){
                    vertices.first(i_vertex) = u0 ;
                    vertices.second(i_vertex) = u1 ;
                }
            
                if( i_local==1 ){
                    vertices.first(i_vertex+1) = u0 ;
                    vertices.second(i_vertex+1) = u1 ;
                }
            
                if( i_local==(degree_FEM+2) ){
                    vertices.first(i_vertex+Nx+2) = u0 ;
                    vertices.second(i_vertex+Nx+2) = u1 ;
                }
                
                if( i_local==(degree_FEM+1) ){
                    vertices.first(i_vertex+Nx+1) = u0 ;
                    vertices.second(i_vertex+Nx+1) = u1 ;
                }
            */
                
                i_local++;
                
            
            }
            i_global++;
            
            
        }
        
    }
    

    template< typename MATRIX >
    void  set_discrete_points( MATRIX& values_new)
    {
        sol_HHO = values_new ;
        std::cout<<" Using set_discrete_points check that sol_FEM already uploaded!"<<std::endl;
       
    }
    
    template< typename VECTOR >
    void converting_into_HHO_formulation( const VECTOR& values_new )
    {
        
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                sol_HHO.first(i,counter_bis) = values_new.first( asm_map );
                sol_HHO.second(i,counter_bis) = values_new.second( asm_map );
            }
            //  size_t i_vertex = counter_bis+floor(counter_bis/Nx);
            //  vertices(i_vertex) = sol_HHO(0,counter_bis) ;
            //  vertices(i_vertex+1) = sol_HHO(1,counter_bis) ;
            // vertices(i_vertex+Nx+2) = sol_HHO(2,counter_bis) ;
            //  vertices(i_vertex+Nx+1) = sol_HHO(3,counter_bis) ;
        }
        std::cout<<" Using converting_into_HHO_formulation check that sol_FEM already uploaded!"<<std::endl;
        //set_max_min();
        //phi_min = sol_HHO.minCoeff() ;
        //phi_max = sol_HHO.maxCoeff() ;
                 
    }
    
    template< typename MATRIX >
    void converting_into_FE_formulation( const MATRIX& values_new )
    {
      
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                sol_FEM.first( asm_map ) = values_new.first(i,counter_bis) ;
                sol_FEM.second( asm_map ) = values_new.second(i,counter_bis) ;
            }
            //  size_t i_vertex = counter_bis+floor(counter_bis/Nx);
            //  vertices(i_vertex) = sol_HHO(0,counter_bis) ;
            //  vertices(i_vertex+1) = sol_HHO(1,counter_bis) ;
            // vertices(i_vertex+Nx+2) = sol_HHO(2,counter_bis) ;
            //  vertices(i_vertex+Nx+1) = sol_HHO(3,counter_bis) ;
        }
        std::cout<<" Using converting_into_FE_formulation check that sol_HHO already uploaded!"<<std::endl;
        //set_max_min();
        //phi_min = sol_FEM.minCoeff() ;
        //phi_max = sol_FEM.maxCoeff() ;
        
    }
       

    
    std::pair<T,T> operator()( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl) const
    {
        size_t counter = offset(msh,cl) ;
        cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        //std::cout<<"cb u proj high order"<<'\n'<<cb.eval_basis(pt)<<'\n'<<std::endl;
        auto values_cell_first = (sol_HHO.first.block(0,counter,local_dim,1)).col(0);
        auto values_cell_second = (sol_HHO.second.block(0,counter,local_dim,1)).col(0);
        //std::cout<<"values_cell_first"<<'\n'<<values_cell_first<<'\n'<<std::endl;
        //std::cout<<"values_cell_second"<<'\n'<<values_cell_second<<'\n'<<std::endl;
        T tmp1 = values_cell_first.dot( cb.eval_basis(pt) );
        T tmp2 = values_cell_second.dot( cb.eval_basis(pt) );
        //std::cout<<"tmp1"<<" "<<tmp1<<'\n'<<"tmp2"<<" "<<tmp2<<std::endl;
        
        return std::make_pair(tmp1,tmp2);
    }
    
    
       
};


template< typename Mesh , typename FiniteSpace , typename T = typename Mesh::coordinate_type  >
struct velocity_high_order
{
  
     size_t  last_row_init, last_row_end, number_faces_one_row;
    
    
     // NEW IMPLEMENTATION STUFF
     Mesh msh; // Original mesh, NOT agglomerated.
     mesh_init_params<T> params; // mesh parameter
     
     size_t degree_FEM ; // FEM degree
     size_t n_cls ; // #cells
     size_t local_dim; // Local Dimension (degree_FEM+1)*(degree_FEM+1)
     size_t n_vertices ; // #vertices
      size_t      Nx, Ny ; // Number of cells in x and y direciton
     // connectivity matrix : for each cell, it stores the global numbering
     std::vector< std::vector<std::pair<size_t,bool>>> connectivity_matrix ;
     //std::vector<std::vector<size_t>> connectivity_matrix ;
     
     size_t dim_HHO; // Global dimension Discontinuous framework = Local dimension * #cells
     size_t ndof_FE; // Global dimension FE continuous = #nodes
    
     T u_max0 , u_max1 ;
     
     
     //SparseMatrix<T>                 Global_Mass; // Global mass, saved for FEM problem
     //Matrix<T, Dynamic, 1>           RHS0;    // Known term 1
     //Matrix<T, Dynamic, 1>           RHS1;    // Known term 1
     //std::vector< Triplet<T> >       triplets; // Position elements: Sparse Matrix Notation
     
     std::pair<Eigen::Matrix<T, Dynamic, Dynamic> , Eigen::Matrix<T, Dynamic, Dynamic>> sol_HHO ; // projection saved in HHO format: cell by cell
     std::pair<Matrix<T, Dynamic, 1>,Matrix<T, Dynamic, 1>> sol_FEM ; // projection saved in Continuos FE format: global nodes
     //std::pair<Eigen::Matrix<T, Dynamic, 1>,Matrix<T, Dynamic, 1>> vertices; // saving level_set on vertices mesh

     
    // AGGLOMERATED MESHES DATA:
    typedef typename Mesh::cell_type       cell_type;
    Mesh agglo_msh;
    std::vector<cell_type> subcells;
    cell_type agglo_LS_cl;
    
     velocity_high_order(const FiniteSpace& fe_data , const Mesh & msh )
           : degree_FEM(fe_data.order) , local_dim(fe_data.local_ndof), msh(msh), Nx(fe_data.Nx),Ny(fe_data.Ny), params(fe_data.params) , dim_HHO(fe_data.ndof_disc) , n_cls(fe_data.n_cls) ,n_vertices(fe_data.n_vertices) , connectivity_matrix(fe_data.connectivity_matrix) , ndof_FE(fe_data.ndof_FE)
       {
           std::cout<<"velocity_high_order: -> implemented with equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree)"<<std::endl;
           //last_row_init = Ny*(2*Nx+1); // There are 2 faces for each row of cells + Ny
           //last_row_end = last_row_init + Nx-1;
           //number_faces_one_row = 2*Nx+1; // for each cell I count only the low and sx faces, respectevely 0-1 2-3 4-5 6-7 8 + the last on the right boundary
           
           
           //vertices = std::make_pair( Eigen::Matrix<T, Dynamic, 1>::Zero( ( (Nx+1)*(Ny+1) ), 1 ) , Eigen::Matrix<T, Dynamic, 1>::Zero( ((Nx+1)*(Ny+1)), 1 ) );

           
           //vertices = std::make_pair( Eigen::Matrix<T, Dynamic, 1>::Zero( n_vertices , 1 ),Eigen::Matrix<T, Dynamic, 1>::Zero( n_vertices , 1 ) );
           sol_FEM = std::make_pair( Matrix<T, Dynamic, 1>::Zero(ndof_FE) , Matrix<T, Dynamic, 1>::Zero(ndof_FE) );
          
           sol_HHO = std::make_pair(Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls ) , Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls ) ) ;

           
       }
    
    void set_agglo_mesh( Mesh & m_agglo_msh )
    {
        agglo_msh = m_agglo_msh;
    }
    
    void cell_assignment(const cell_type& cl)
    {
        subcells.clear();
        agglo_LS_cl = cl;
        
        if( (agglo_LS_cl.user_data.offset_subcells.size()>1) &&  (agglo_LS_cl.user_data.offset_subcells[0] != agglo_LS_cl.user_data.offset_subcells[1] ) )
        {

            for (auto& offset_subcells:agglo_LS_cl.user_data.offset_subcells)
            {
                subcells.push_back( msh.cells[offset_subcells] );
            
            }
        }
            
    }


    template< typename MATRIX >
    void  set_discrete_points( MATRIX& values_new)
    {
        sol_HHO = values_new ;
        std::cout<<" Using set_discrete_points check that sol_FEM already uploaded!"<<std::endl;
       
    }
    
    template< typename VECTOR >
    void converting_into_HHO_formulation( const VECTOR& values_new )
    {
        
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                sol_HHO.first(i,counter_bis) = values_new.first( asm_map );
                sol_HHO.second(i,counter_bis) = values_new.second( asm_map );
            }
            //size_t i_vertex = counter_bis+floor(counter_bis/Nx);
            //vertices(i_vertex) = sol_HHO(0,counter_bis) ;
            //vertices(i_vertex+1) = sol_HHO(1,counter_bis) ;
            //vertices(i_vertex+Nx+2) = sol_HHO(2,counter_bis) ;
            //vertices(i_vertex+Nx+1) = sol_HHO(3,counter_bis) ;
        }
        std::cout<<" Using converting_into_HHO_formulation check that sol_FEM already uploaded!"<<std::endl;
        //set_max_min();
        //phi_min = sol_HHO.minCoeff() ;
        //phi_max = sol_HHO.maxCoeff() ;
                 
    }
    
    template< typename MATRIX >
    void converting_into_FE_formulation( const MATRIX& values_new )
    {
      
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                sol_FEM.first( asm_map ) = values_new.first(i,counter_bis) ;
                sol_FEM.second( asm_map ) = values_new.second(i,counter_bis) ;
            }
           //size_t i_vertex = counter_bis+floor(counter_bis/Nx);
            //vertices(i_vertex) = sol_HHO(0,counter_bis) ;
            //vertices(i_vertex+1) = sol_HHO(1,counter_bis) ;
            //vertices(i_vertex+Nx+2) = sol_HHO(2,counter_bis) ;
            //vertices(i_vertex+Nx+1) = sol_HHO(3,counter_bis) ;
        }
        std::cout<<"In CONVERTING FE -> Using converting_into_FE_formulation check that sol_HHO already uploaded!"<<std::endl;
        //set_max_min();
        //phi_min = sol_FEM.minCoeff() ;
        //phi_max = sol_FEM.maxCoeff() ;
        
    }
       
    template< typename LEVEL_SET >
    void L2_proj_into_FE_formulation( LEVEL_SET& level_set , const Mesh & msh)
    {
        std::cout<<"L2 projection to have CONTINUOUS FE: SimplicialLLT to invert global Mass MAtrix."<<std::endl;
        timecounter tc_vel ;
        tc_vel.tic();
        Matrix<T, Dynamic, 1> RHS1 =  Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        Matrix<T, Dynamic, 1> RHS2 =  Matrix<T, Dynamic, 1>::Zero( ndof_FE );
        SimplicialLLT<SparseMatrix<T> >solver_global_mass;
        solver_global_mass.compute(level_set.Global_Mass);
        
        for(auto cl : msh.cells )
        {
            auto local_RHS = make_bernstein_local_RHS_VEC( msh , cl , degree_FEM , *this );
            size_t cell_offset = offset(msh,cl);
            for (size_t i = 0; i < local_dim; i++)
            {
                size_t asm_map_i = connectivity_matrix[cell_offset][i].first ;
                RHS1(asm_map_i) += local_RHS.first(i) ;
                RHS2(asm_map_i) += local_RHS.second(i) ;
            }
            
        }
        sol_FEM.first = solver_global_mass.solve(RHS1);
        sol_FEM.second = solver_global_mass.solve(RHS2);
        tc_vel.toc();
        std::cout<<"Time to L2 project velocity field : "<<tc_vel<<std::endl;
    }
    
    template< typename MATRIX >
    void smooth_converting_into_FE_formulation( const MATRIX& values_new )
    {
        std::cout<<"SMOOTH CONVERTING INTO CONTINUOUS FE: --> For each point I do the geometric average"<<std::endl;
        Array<T,Dynamic,1> counting_avg = Array<T,Dynamic,1>::Zero(ndof_FE) ;
        Array<T,Dynamic,1> sum_first = Array<T,Dynamic,1>::Zero(ndof_FE) ;
        Array<T,Dynamic,1> sum_second = Array<T,Dynamic,1>::Zero(ndof_FE) ;
        //std::array<T, ndof_FE > counting_avg;
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                sum_first( asm_map ) += values_new.first(i,counter_bis) ;
                sum_second( asm_map ) += values_new.second(i,counter_bis);
                counting_avg(asm_map)++;
            }
           //size_t i_vertex = counter_bis+floor(counter_bis/Nx);
            //vertices(i_vertex) = sol_HHO(0,counter_bis) ;
            //vertices(i_vertex+1) = sol_HHO(1,counter_bis) ;
            //vertices(i_vertex+Nx+2) = sol_HHO(2,counter_bis) ;
            //vertices(i_vertex+Nx+1) = sol_HHO(3,counter_bis) ;
        }
        sol_FEM.first = (sum_first.array()).cwiseQuotient(counting_avg);
        sol_FEM.second = (sum_second.array()).cwiseQuotient(counting_avg);
        //std::cout<<"counting_avg"<<'\n'<<counting_avg<<std::endl;
        std::cout<<"In CONVERTING FE SMOOTH -> Using converting_into_FE_formulation check that sol_HHO already uploaded!"<<std::endl;
        
        //set_max_min();
        //phi_min = sol_FEM.minCoeff() ;
        //phi_max = sol_FEM.maxCoeff() ;
        
    }

    
    std::pair<T,T> operator()( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl) const
    {
        size_t counter = offset(msh,cl) ;
        cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl, degree_FEM);
        //std::cout<<"cb u proj high order"<<'\n'<<cb.eval_basis(pt)<<'\n'<<std::endl;
        auto values_cell_first = (sol_HHO.first.block(0,counter,local_dim,1)).col(0);
        auto values_cell_second = (sol_HHO.second.block(0,counter,local_dim,1)).col(0);
        //std::cout<<"values_cell_first"<<'\n'<<values_cell_first<<'\n'<<std::endl;
        //std::cout<<"values_cell_second"<<'\n'<<values_cell_second<<'\n'<<std::endl;
        T tmp1 = values_cell_first.dot( cb.eval_basis(pt) );
        T tmp2 = values_cell_second.dot( cb.eval_basis(pt) );
        //std::cout<<"tmp1"<<" "<<tmp1<<'\n'<<"tmp2"<<" "<<tmp2<<std::endl;
        
        return std::make_pair(tmp1,tmp2);
    }
    
    
    //  OPERATOR for MESH AGGLOMERATED
    std::pair<T,T> operator()(const point<T,2>& pt) const
    {
        
        if (subcells.size()<1)
        {
            
            assert(agglo_LS_cl.user_data.offset_subcells.size()==2);
            assert( agglo_LS_cl.user_data.offset_subcells[0] == agglo_LS_cl.user_data.offset_subcells[1] );
            auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = msh.cells[offset_old];
           
            cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, cl_old , degree_FEM);
           
            auto values_cell_first = (sol_HHO.first.block(0,offset_old,local_dim,1)).col(0);
            auto values_cell_second = (sol_HHO.second.block(0,offset_old,local_dim,1)).col(0);
           
            T tmp1 = values_cell_first.dot( cb.eval_basis(pt) );
            T tmp2 = values_cell_second.dot( cb.eval_basis(pt) );
            
            return std::make_pair(tmp1,tmp2);
           
        }
        else
        {
            auto offset = pt_in_subcell(msh,pt,agglo_LS_cl);
            auto subcl = msh.cells[offset];
            
            cell_basis_Lagrangian_ordered<Mesh,T> cb(msh, subcl , degree_FEM);
            
            auto values_cell_first = (sol_HHO.first.block(0,offset,local_dim,1)).col(0);
            auto values_cell_second = (sol_HHO.second.block(0,offset,local_dim,1)).col(0);
            
            T tmp1 = values_cell_first.dot( cb.eval_basis(pt) );
            T tmp2 = values_cell_second.dot( cb.eval_basis(pt) );
             
            return std::make_pair(tmp1,tmp2);
            
        }
               
    }
    
       
};





template< typename Mesh>
struct Current_Mesh
{
    Mesh current_mesh ;
    Current_Mesh(const Mesh& msh):current_mesh(msh){}
    
    void set_current_mesh(const Mesh& msh)
    {
        current_mesh = msh;
    }
    
    
};



template< typename T , typename Mesh ,typename Level_Set,typename Fonction >
struct LS_cell: public projected_level_set< T,  Mesh , Fonction>
{
    typedef typename Mesh::cell_type       cell_type;
    cell_type agglo_LS_cl;
    std::vector<cell_type> subcells;
    Mesh agglo_msh;
    Level_Set level_set;
    //LS_cell(const Level_Set & level_set, const Mesh & msh, const typename Mesh::cell_type& cl)
   // : agglo_cl(cl), agglo_msh(msh), level_set(level_set){}
    // I don't know if I have to define a copyconstructor for level_set.. TO BE CHECKED!
    LS_cell(const Level_Set & level_set_, const Mesh & msh)
    : agglo_msh(msh), level_set(level_set_){}
    //LS_cell(const Level_Set & level_set )
    //: level_set(level_set){}

    //LS_cell()=default;
    
    T operator()(const point<T,2>& pt) const
       {
           if (subcells.size()<1)
           {
               assert(agglo_LS_cl.user_data.offset_subcells.size()==1);
               auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
               auto cl_old = level_set.msh.cells[offset_old];
               return level_set( pt , level_set.msh , cl_old );
           }
       
           else
           {
               //std::cout<<"Ls operator"<<std::endl;
               auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
               auto subcl = level_set.msh.cells[offset];
               return level_set( pt , level_set.msh , subcl );
           }
               
       }
    
    
    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt) const
    {
        if (subcells.size()<1)
            {
                assert(agglo_LS_cl.user_data.offset_subcells.size()==1);
                auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
                auto cl_old = level_set.msh.cells[offset_old];
                return level_set.normal( pt , level_set.msh , cl_old );
            }
        
            else
            {
               
                auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
                auto subcl = level_set.msh.cells[offset];
                return level_set.normal( pt , level_set.msh , subcl );
            }
    }
    
    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        if (subcells.size()<1)
            {
                std::cout<<"I m here 2.5 and size is "<<agglo_LS_cl.user_data.offset_subcells.size()<<std::endl;
                assert(agglo_LS_cl.user_data.offset_subcells.size()==1);
                auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
                auto cl_old = level_set.msh.cells[offset_old];
                return level_set.gradient( pt , level_set.msh , cl_old );
            }
        
            else
            {
                std::cout<<"Ls gradient"<<std::endl;
                auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
                auto subcl = level_set.msh.cells[offset];
                return level_set.gradient( pt , level_set.msh , subcl );
            }
        
    }
  
    void cell_assignment(const cell_type& cl)
    {
        subcells.clear();
        agglo_LS_cl = cl;
        if (agglo_LS_cl.user_data.offset_subcells.size()>1)
        {

            for (auto& offset_subcells:agglo_LS_cl.user_data.offset_subcells)
            {
                subcells.push_back( level_set.msh.cells[offset_subcells] );
            
            }
        }
            
    }
    
    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt , const cell_type& cl)
    {
        agglo_LS_cl = cl;
        return normal( pt );
    }
    
    T operator()(const point<T,2>& pt,const cell_type& cl )
    {
        agglo_LS_cl = cl;
        return operator()( pt );
    }
    
    
  //  void get_cell()
  //  {
  //      auto crr_cl = download_current_cell();
  //      std::cout<<"IN get_cell.. size cell "<<crr_cl.user_data.offset_subcells.size()<<std::endl;
  //      this->cell_assignment(crr_cl);
 //   }
    
};





/**************MOVING INTERFACE: LEVEL SET METHOD  **************/

template<typename T, typename Mesh >
struct projection
{
    Eigen::Matrix<T, Dynamic, Dynamic> values_bis;
    Eigen::Matrix<T, Dynamic, 1> vertices; // saving level_set on vertices mesh
    
    size_t degree_FEM;
    size_t number_elements;
    Mesh msh;
    size_t      Nx, Ny ;
    mesh_init_params<T> params;
    //size_t  last_row_init, last_row_end, number_faces_one_row;

    projection( const Mesh & msh, size_t degree_k , const mesh_init_params<T>& params)
        : number_elements((degree_k+1)*(degree_k+1)), msh(msh),degree_FEM(degree_k),Nx(params.Nx),Ny(params.Ny), params(params)
    {
        vertices = Eigen::Matrix<T, Dynamic, 1>::Zero( ((Nx+1)*(Ny+1)), 1 );
        values_bis= Eigen::Matrix<T, Dynamic, Dynamic>::Zero(number_elements, msh.cells.size());
    }
    
    projection()=default;
    
    
    void  set_discrete_points( Eigen::Matrix<T, Dynamic, Dynamic>& values_new)
    {
        values_bis=values_new;
    }
    
    
    void converting_into_HHO_formulation( const Eigen::Matrix<T, Dynamic, 1>& vertices )
    {
        assert(degree_FEM == 1);
        for (size_t j = 0; j < values_bis.cols() ; j++)
        {

            for (size_t i = 0; i < values_bis.rows() ; i++)
            {
                auto i_vertex = j + floor(j/Nx);
                if ( i == 0 )
                    values_bis(i,j) = vertices(i_vertex);
                
                if( i ==1 )
                    values_bis(i,j) = vertices(i_vertex+1) ;
                
                if( i ==(degree_FEM+2) )
                    values_bis(i,j) = vertices(i_vertex+Nx+2) ;
                
                if( i ==(degree_FEM+1) )
                    values_bis(i,j) = vertices(i_vertex+Nx+1) ;
            }
        }
                 
    }
    
    void converting_into_FE_formulation( const Eigen::Matrix<T, Dynamic, Dynamic>& values )
    {
        assert(degree_FEM == 1);
        for (size_t j = 0; j < values_bis.cols() ; j++)
        {

            for (size_t i = 0; i < values_bis.rows() ; i++)
            {
                auto i_vertex = j + floor(j/Nx);
                if ( i == 0 )
                    vertices(i_vertex) = values(i,j) ;
                
                if( i ==1 )
                    vertices(i_vertex+1) = values(i,j) ;
                
                if( i ==(degree_FEM+2) )
                    vertices(i_vertex+Nx+2) = values(i,j) ;
                
                if( i ==(degree_FEM+1) )
                    vertices(i_vertex+Nx+1) = values(i,j) ;
            }
        }
                 
    }
       
    
    
   
    T operator()( const typename Mesh::node_type& node ) const
    {
        return vertices(node.ptid);
    }
    
    T operator()( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        size_t counter = offset(msh,cl) ;
        cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = (values_bis.block(0,counter,number_elements,1)).col(0);
        T tmp = values_cell.dot( cb.eval_basis(pt) );
        return tmp;
    }
  
    Eigen::Matrix<T,2,1> gradient( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        size_t counter = offset(msh,cl);
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = values_bis.col(counter);
        auto grad_eval =  cb.eval_gradients(pt);
        ret(0) = values_cell.dot( grad_eval.col(0) );
        ret(1) = values_cell.dot( grad_eval.col(1) );
        return ret;
    }


    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt, const Mesh& msh, const typename Mesh::cell_type& cl) const
    {
        Eigen::Matrix<T,2,1> ret;
        ret = gradient(pt,msh,cl);
        return ret/ret.norm();
        
    }
    
};




template<typename Mesh, typename T = typename Mesh::coordinate_type>
Matrix<T, Dynamic, Dynamic>
make_lagrange_local_mass_matrix(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree, size_t di = 0)
{
    
    //auto f_neigh = cl.user_data.f_neighbors;
    //auto d_neigh = cl.user_data.d_neighbors;
    
    cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    //Matrix<T, Dynamic, 1> ret2 = Matrix<T, Dynamic, 1>::Zero(cbs, 1);

    auto qps = integrate(msh, cl, 2*(degree+di)); // integration of order 2k

    for (auto& qp : qps)
    {
        auto phi = cb.eval_basis(qp.first);
        ret += qp.second * phi * phi.transpose();
        // phi * phi.transpose is (degree+1)^2 x (degree+1)^2 ; qp.second is a scalar
    }
    //ret2  =  ret.rowwise().sum(); // sum row; i.e. mi0 + mi1 + mi2 + mi3 , with i = 0 : 3
    
    return ret;
}

template<typename Mesh, typename T = typename Mesh::coordinate_type>
Matrix<T, Dynamic, Dynamic>
make_lagrange_lumped_mass_matrix(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree, size_t di = 0)
{
    cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs, 1);

    auto qps = integrate(msh, cl, 2*(degree+di));

    for (auto& qp : qps)
    {
        auto phi = cb.eval_basis(qp.first);
        ret += qp.second * phi;
    }

    return ret;
}

template<typename Mesh, typename T = typename Mesh::coordinate_type>
std::pair< Matrix<T, Dynamic, Dynamic> , Matrix<T, Dynamic, Dynamic> >
make_local_cij_matrix(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree, size_t di = 0)
{
    cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, Dynamic> ret0 = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    Matrix<T, Dynamic, Dynamic> ret1 = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    
    // for Q1 , degree = 1-> integration of order 2
    auto qps = integrate(msh, cl, 2*(degree+di)); // integration of order 2k

    for (auto& qp : qps)
    {
        auto phi = cb.eval_basis(qp.first);
        
        auto phi_grad = cb.eval_gradients(qp.first);
        
        ret0 += qp.second * phi * ((phi_grad).col(0)).transpose();
    
        ret1 += qp.second * phi * ((phi_grad).col(1)).transpose();
    }
    
    return std::make_pair(ret0,ret1);
}





template < typename VECTOR, typename Mesh, typename MATRIX , typename T = typename Mesh::coordinate_type >
Matrix<T, Dynamic, 1>  make_dij_vector(const Mesh & msh, const MATRIX& dij, const VECTOR& phi_old )
{
    Matrix<T, Dynamic, 1> ret = Eigen::Matrix<T, Dynamic, 1>::Zero(dij.rows(), 1);
    
    
    for(size_t i = 0 ; i<dij.rows() ; i++ )
    {
        auto tmp = (dij.row(i)).adjoint();
        ret(i) = tmp.dot( phi_old ) - tmp(i)*( phi_old(i) );
    }
    return ret;
}




template<typename T , typename Mesh >
class Finite_Element
{
public:
    Mesh msh; // Original mesh, not agglomerated
    // number of degree of freedom (HHO setting) , #vertices , #cells
    size_t ndof_disc , n_vertices , n_cls  ;
    size_t ndof_FE ; // number of degree of freedom (Continuous FEM setting)
    size_t num_faces = 4 ; // #faces
    mesh_init_params<T> params; // parameters mesh
    size_t order; // FE order of polynomial B^k
    size_t local_ndof ; // local degree of freedom
    
    size_t Nx , Ny ; // number of cells in x and y direction
    T hx; // step size along x direction
    T hy; // step size along y direction
    
    std::vector<std::set<size_t>> S_i;
    std::vector< std::vector<std::pair<size_t,bool>>> connectivity_matrix ;
    
    std::vector< bool > Dirichlet_boundary; //(ndof_FE , FALSE );
    std::vector< bool > Dirichlet_boundary_inlet; //(ndof_FE , FALSE );
    
    Finite_Element(const Mesh & msh , size_t order , const mesh_init_params<T>& params ): msh(msh) , n_vertices(msh.nodes.size()) , n_cls(msh.cells.size()) ,  order(order) , local_ndof((order+1)*(order+1)), hx(params.hx() ) , hy(params.hy() ) , connectivity_matrix( n_cls , std::vector<std::pair<size_t,bool>>(local_ndof) ) , Nx(params.Nx),Ny(params.Ny) , params(params)
    {
        ndof_disc =  n_vertices*n_cls ;
        
        //std::cout<<"n_vertices "<<n_vertices<<" , n_cls "<<n_cls<<" , ndof_disc "<<ndof_disc<<std::endl;
       
        size_t counter = 0 , i_global = 0 ;
        for(const auto& cl : msh.cells)
        {
            
            std::vector<size_t> loc_bdry_nodes;
            auto fcs = faces(msh,cl);
            for(size_t face_i = 0; face_i < num_faces; face_i++)
            {
                auto fc = fcs[face_i];
                //std::cout<<"fc is "<<fc<<std::endl;
                // In HHO flow problem I can have different types of boundary;
                // For level set evolution, I just focus on Dirichelet boundary
                // The indication of the type of boundary in fc is relative to HHO problem, so useless now
                if( fc.is_boundary )
                {
                    if(face_i == 0){
                        loc_bdry_nodes.push_back(0);
                        loc_bdry_nodes.push_back(1);
                
                        if(order > 1){
                            for(size_t j = 0 ; j <= order - 2 ; j++)
                                loc_bdry_nodes.push_back(4+j);
                        }
                        
                    }
                    else if(face_i == 1){
                        loc_bdry_nodes.push_back(1);
                        loc_bdry_nodes.push_back(2);
                        if(order > 1){
                        for(size_t j = 0 ; j <= order - 2 ; j++)
                            loc_bdry_nodes.push_back(order+3+j);
                        }
                    }
                    else if(face_i == 2){
                        loc_bdry_nodes.push_back(2);
                        loc_bdry_nodes.push_back(3);
                        if(order > 1){
                        for(size_t j = 0 ; j <= order - 2 ; j++)
                            loc_bdry_nodes.push_back(2*order+2+j);
                        }
                    }
                    else if(face_i == 3){
                        loc_bdry_nodes.push_back(0);
                        loc_bdry_nodes.push_back(3);
                        if(order > 1){
                        for(size_t j = 0 ; j <= order - 2 ; j++)
                            loc_bdry_nodes.push_back(3*order+1+j);
                        }
                                       
                    }
                    else
                        exit(1);
                
                }
            
            }
            
            size_t offset_curr = offset(msh,cl);
            sort(loc_bdry_nodes.begin(), loc_bdry_nodes.end());
            loc_bdry_nodes.erase( unique( loc_bdry_nodes.begin(), loc_bdry_nodes.end() ), loc_bdry_nodes.end() );
            
           
        
            
            
            //std::cout<<"loc_bdry_nodes "<<std::endl;
            //for(auto i : loc_bdry_nodes)
            //    std::cout<<i<<" , ";
            //std::cout<<std::endl;
            
            for( size_t i_local = 0 ; i_local < local_ndof ; i_local++)
            {
                // if boundary = TRUE is node on the boudnary , else not
                bool boundary =  binary_search(loc_bdry_nodes.begin(), loc_bdry_nodes.end(), i_local) ;
                
                // Case: first cell
                if( offset_curr == 0 ){
                connectivity_matrix[counter][i_local].first = i_global;
                connectivity_matrix[counter][i_local].second = boundary;
                i_global++;
                }
                
                // Case: First row of cells, a part of cell[0]
                // Left face enumeration for cell i = Right face enumeration for cell i-1
                else if( offset_curr > 0 && offset_curr < Nx )
                {
                    if( i_local == 0  ){ // vertex 1 of cell[counter-1] = vertex 0 of cell[counter]
                        connectivity_matrix[counter][i_local].first = connectivity_matrix[counter-1][1].first ;
                        connectivity_matrix[counter][i_local].second = boundary;
                    }
                    else if(  i_local == 3  ){ // vertex 2 of cell[counter-1] = vertex 3 of cell[counter]
                        connectivity_matrix[counter][i_local].first = connectivity_matrix[counter-1][2].first ;
                        connectivity_matrix[counter][i_local].second = boundary;
                    }
                    else if(  i_local >= 3*order + 1 && i_local <= 4*order - 1   ) // nodes in face 3
                    {
                        // how many points i_local is distant from v3 vertices
                        // Idea is f1 of cell[counter-1] = f2 of cell[counter] (i.e. = means same ordering). It means:
                        //    face 1             face 3
                        // 2*ordering + 1 ---- 3*ordering + 1
                        //    |                      |
                        //    |                      |
                        //    |                      |
                        //  ordering + 3  ---- 4*ordering - 1
                        
                        size_t dist_v3 = i_local - (3*order + 1) ;
                        size_t j = 2*order + 1 - dist_v3 ;
                        connectivity_matrix[counter][i_local].first = connectivity_matrix[counter-1][j].first ;
                        connectivity_matrix[counter][i_local].second = boundary;
                    }
                    else
                    {
                        connectivity_matrix[counter][i_local].first = i_global;
                        connectivity_matrix[counter][i_local].second = boundary;
                        i_global++;
                    }
               
                }
                
                // Case: Left Boundary cells
                // Bottom face enumeration for cell i = Top face enumeration for cell i-Nx
                else if( offset_curr % Nx == 0 && offset_curr > 0 )
                {
                    if( i_local == 0  ){ // vertex 3 of cell[counter-Nx] = vertex 0 of cell[counter]
                        connectivity_matrix[counter][i_local].first = connectivity_matrix[counter-Nx][3].first ;
                        connectivity_matrix[counter][i_local].second = boundary;
                    }
                    else if(  i_local == 1   ){ // vertex 2 of cell[counter-Nx]=vertex 1 of cell[counter]
                        connectivity_matrix[counter][i_local].first = connectivity_matrix[counter-Nx][2].first ;
                        connectivity_matrix[counter][i_local].second = boundary;
                    }
                    else if( i_local >= 4 && i_local <= order + 2  ) // nodes in face 0
                    {
                        // how many points i_local is distant from v0 vertices
                        // Idea is f2 of cell[counter-Nx] = f0 of cell[counter] (i.e. = means same ordering). It means:
                        //         face 0 ; cell[counter]
                        //     4        ----     ordering + 2
                        //     |                      |
                        //         face 2 ; cell[counter-Nx]
                        //     |                      |
                        //  3*ordering  ----    2*ordering +2
                        
                        size_t dist_v0 = i_local - 4 ;
                        size_t j = 3*order - dist_v0 ;
                        connectivity_matrix[counter][i_local].first = connectivity_matrix[counter-Nx][j].first ;
                        connectivity_matrix[counter][i_local].second = boundary;
                    }
                    
                    else
                    {
                        connectivity_matrix[counter][i_local].first = i_global;
                        connectivity_matrix[counter][i_local].second = boundary;
                        i_global++;
                    }
                }
                
                // All the other cells ( i.e. internal and right boundary cells ) are both 2 the cases
                else
                {
                    if( i_local == 0  ){ // vertex 1 of cell[counter-1] = vertex 0 of cell[counter]
                        connectivity_matrix[counter][i_local].first = connectivity_matrix[counter-1][1].first ;
                        connectivity_matrix[counter][i_local].second = boundary;
                    }
                    else if(  i_local == 3  ){ // vertex 2 of cell[counter-1] = vertex 3 of cell[counter]
                        connectivity_matrix[counter][i_local].first = connectivity_matrix[counter-1][2].first ;
                        connectivity_matrix[counter][i_local].second = boundary;
                    }
                    else if(  i_local == 1   ){ // vertex 2 of cell[counter-Nx]=vertex 1 of cell[counter]
                        connectivity_matrix[counter][i_local].first = connectivity_matrix[counter-Nx][2].first ;
                        connectivity_matrix[counter][i_local].second = boundary;
                    }
                    else if(  i_local >= 3*order + 1 && i_local <= 4*order - 1   ) // nodes in face 3
                    {
                        size_t dist_v3 = i_local - (3*order + 1) ;
                        size_t j = 2*order + 1 - dist_v3 ;
                        connectivity_matrix[counter][i_local].first = connectivity_matrix[counter-1][j].first ;
                        connectivity_matrix[counter][i_local].second = boundary;
                    }
                    else if( i_local >= 4 && i_local <= order + 2  ) // nodes in face 0
                    {
                        size_t dist_v0 = i_local - 4 ;
                        size_t j = 3*order - dist_v0 ;
                        connectivity_matrix[counter][i_local].first = connectivity_matrix[counter-Nx][j].first ;
                        connectivity_matrix[counter][i_local].second = boundary;
                    }
                    else
                    {
                        connectivity_matrix[counter][i_local].first = i_global;
                        connectivity_matrix[counter][i_local].second = boundary;
                        i_global++;
                    }
                }
                
            } // End of local loop for each cell
        counter++;
        } // End of loop over the cells.
        ndof_FE = i_global ;
        
        S_i.resize(ndof_FE);
        Dirichlet_boundary.resize(ndof_FE);
        Dirichlet_boundary_inlet.resize(ndof_FE);
        
        
        for(auto&cl : msh.cells)
        {
            auto offset_cl = offset(msh,cl);
            for (size_t i = 0; i < local_ndof; i++)
            {
                auto asm_map_i = connectivity_matrix[offset_cl][i].first;
                for (size_t j = 0; j < local_ndof; j++)
                {
                    auto asm_map_j = connectivity_matrix[offset_cl][j].first;
                    S_i[asm_map_i].insert( asm_map_j );

                }
            }
        }
        
        
        /*
        size_t size_supp_nodes = 0;
        std::cout<<"Supporting nodes IN FE METHOD:"<<std::endl;
        size_t jjjj = 0;
        for (auto& i: S_i) {
            size_supp_nodes+=i.size();
            std::cout <<"Node "<<jjjj<<":";
            for (auto it=i.begin(); it != i.end(); ++it)
                std::cout << ' ' << *it;
               // std::cout<<ii;
            std::cout<<'\n';
            jjjj++;
        }
        std::cout<<std::endl;
        std::cout<<"Supporting nodes size:"<<size_supp_nodes<<std::endl;
        */
        
        //std::cout<<"Beginning of connectivity matrix"<<std::endl;
        
        
        
        // Dirichlet boundary vector
        for(size_t i = 0 ; i< n_cls ; i++)
        {
            for(size_t j = 0 ; j< local_ndof ; j++)
            {
                //std::cout<<"( "<<connectivity_matrix[i][j].first<<" , "<<connectivity_matrix[i][j].second<<" ) , ";
                auto asmap = connectivity_matrix[i][j] ;
                Dirichlet_boundary[asmap.first] = asmap.second ;
                //std::cout<<"Node "<<asmap.first << " is Dirichlet "<<asmap.second<< "  ,  ";
            }
           // std::cout<<std::endl;
        }
        //std::cout<<std::endl;
        //std::cout<<"End of connectivity matrix"<<std::endl;
        
        
        
        
    }
        
        
    //void
    //assembling(SparseMatrix<T>& Global_Mass ,SparseMatrix<T>& Global_c_term_x ,SparseMatrix<T>& Global_c_term_y, DiagonalMatrix<T, Dynamic>& Global_Mass_Lumped , Matrix<T, Dynamic, 1>& RHS){
        
        
   // }
    
    /*
    size_t get_order() const { return order; }
    size_t get_n_nodes() const { return n_nodes; }
    size_t get_n_cells() const { return n_cls; }
    size_t get_hx() const { return hx; }
    size_t get_hy() const { return hy; }
    size_t get_Nx() const { return Nx; }
    size_t get_Ny() const { return Ny; }
    size_t get_local_ndof() const { return local_ndof; }
    size_t get_ndof() const { return ndof; }
    */
};





template<typename T, typename Mesh , typename FiniteSpace>
struct L2_projection
{
    SparseMatrix<T>                 Global_Mass; // Global mass, saved for FEM problem
    Matrix<T, Dynamic, 1>           RHS;    // Known term
    std::vector< Triplet<T> >       triplets; // Position elements: Sparse Matrix Notation
    
    Eigen::Matrix<T, Dynamic, Dynamic> sol_HHO ; // projection saved in HHO format: cell by cell
    Matrix<T, Dynamic, 1> sol_FEM ; // projection saved in Continuos FE format: global nodes
    Eigen::Matrix<T, Dynamic, 1> vertices; // saving level_set on vertices mesh


    size_t number_elements;
    
    size_t n_cls ; // #cells
    size_t local_dim; // Local Dimension (degree_FEM+1)*(degree_FEM+1)
    size_t n_vertices ; // #vertices

    size_t degree_FEM;
    Mesh msh;
    size_t      Nx, Ny ;
    mesh_init_params<T> params;
    
    std::vector< std::vector<std::pair<size_t,bool>>> connectivity_matrix ;
  
    size_t dim_HHO; // Global dimension Discontinuous framework = Local dimension * #cells
    size_t ndof_FE; // Global dimension FE continuous = #nodes
       

    L2_projection(const FiniteSpace& fe_data ,  const Mesh & msh)
           : degree_FEM(fe_data.order) , local_dim(fe_data.local_ndof), msh(msh), Nx(fe_data.Nx),Ny(fe_data.Ny), params(fe_data.params) , dim_HHO(fe_data.ndof_disc) , n_cls(fe_data.n_cls) ,n_vertices(fe_data.n_vertices) , connectivity_matrix(fe_data.connectivity_matrix) , ndof_FE(fe_data.ndof_FE)
       {
           sol_HHO = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( local_dim, n_cls );
           sol_FEM = Matrix<T, Dynamic, 1>::Zero(ndof_FE);
           vertices = Eigen::Matrix<T, Dynamic, 1>::Zero( n_vertices , 1 );
       }
    

    L2_projection()=default;
    
    
    void  set_discrete_points( Eigen::Matrix<T, Dynamic, Dynamic>& values_new)
    {
        std::cout<<"L2 projection for phi_tilde post-resolution."<<std::endl;
        sol_HHO = values_new;
    }
    
    
    void converting_into_HHO_formulation( const Eigen::Matrix<T, Dynamic, 1>& values_new )
    {
        std::cout<<"L2 projection for phi_tilde post-resolution."<<std::endl;
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                sol_HHO(i,counter_bis) = values_new( asm_map );
            }
            size_t i_vertex = counter_bis+floor(counter_bis/Nx);
            vertices(i_vertex) = sol_HHO(0,counter_bis) ;
            vertices(i_vertex+1) = sol_HHO(1,counter_bis) ;
            vertices(i_vertex+Nx+2) = sol_HHO(2,counter_bis) ;
            vertices(i_vertex+Nx+1) = sol_HHO(3,counter_bis) ;
        }
                 
    }
    
    void converting_into_FE_formulation( const Eigen::Matrix<T, Dynamic, Dynamic>& values_new )
    {
        std::cout<<"L2 projection for phi_tilde post-resolution."<<std::endl;
        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
        {
            for (size_t i = 0; i < local_dim; i++){
                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
                sol_FEM( asm_map ) = values_new(i,counter_bis) ;
            }
        }

                 
    }
       
    
    
    T operator()( const typename Mesh::node_type& node ) const
    {
        return vertices(node.ptid);
    }
    
    T operator()( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        
        size_t counter = offset(msh,cl) ;
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        //cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = (sol_HHO.block(0,counter,local_dim,1)).col(0);
        T tmp = values_cell.dot( cb.eval_basis(pt) );
        return tmp;
    }
    
    Eigen::Matrix<T,2,1> gradient( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
       {
       
           // MATRIX NOTATION
           size_t counter = offset(msh,cl);
           Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
           cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
           //cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
           auto values_cell = sol_HHO.col(counter);
           auto grad_eval =  cb.eval_gradients(pt);
           ret(0) = values_cell.dot( grad_eval.col(0) );
          // std::cout<<"Value of derivative new along x"<<ret(0)<<std::endl;
           ret(1) = values_cell.dot( grad_eval.col(1) );
           //values_cell.dot( grad_eval.col(1) );
          // std::cout<<"Value of derivative new along y"<<ret(1)<<std::endl;
           return ret;
           
       }

    
       // IT WORKS FOR NOT-AGGLOMERATED MESHES --> FAST
       Eigen::Matrix<T,2,1> normal(const point<T,2>& pt, const Mesh& msh, const typename Mesh::cell_type& cl) const
       {
           Eigen::Matrix<T,2,1> ret;
           ret = gradient(pt,msh,cl);
           return ret/ret.norm();
           
       }
       
    
};






template< typename FiniteSpace , typename Mesh , typename T >
void check_inlet( const Mesh& msh , FiniteSpace& fe_data , bool bdry_bottom , bool bdry_right , bool bdry_up , bool bdry_left , T eps )
{
    std::cout<<"Checking inlet boundary condition for analytical flows."<<std::endl;
    std::vector< std::vector<std::pair<size_t,bool>>> connectivity_matrix = fe_data.connectivity_matrix ;
    
    
    for( const auto& cl : msh.cells )
    {
        size_t cell_offset = offset(msh, cl) ;
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, fe_data.order);
        for (size_t i = 0; i < fe_data.local_ndof; i++)
        {
            auto pt = pts[i];
            size_t asm_map = connectivity_matrix[cell_offset][i].first ;
            if( connectivity_matrix[cell_offset][i].second )
            {
                if(bdry_bottom && ( std::abs(pt.y()) < eps) )
                    fe_data.Dirichlet_boundary_inlet[asm_map] = TRUE ;
                if(bdry_right && (std::abs(pt.x()- 1.0) < eps) )
                    fe_data.Dirichlet_boundary_inlet[asm_map] = TRUE ;
                if(bdry_up && (std::abs(pt.y()- 1.0) < eps) )
                    fe_data.Dirichlet_boundary_inlet[asm_map] = TRUE ;
                if(bdry_left && (std::abs(pt.x()) < eps) )
                    fe_data.Dirichlet_boundary_inlet[asm_map] = TRUE ;
                else
                    fe_data.Dirichlet_boundary_inlet[asm_map] = FALSE ;
                
            }
            else
                fe_data.Dirichlet_boundary_inlet[asm_map] = FALSE ;
            
            
            
        }
    }
    
}

template< typename FiniteSpace , typename Mesh ,typename Vel_Field , typename T>
void check_inlet( const Mesh& msh , FiniteSpace& fe_data , const Vel_Field& u , T eps )
{
    std::cout<<"Checking inlet boundary condition for numerical flows."<<std::endl;
    std::vector< std::vector<std::pair<size_t,bool>>> connectivity_matrix = fe_data.connectivity_matrix ;
    
    
    for( const auto& cl : msh.cells )
    {
        size_t cell_offset = offset(msh, cl) ;
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, fe_data.order);
        for (size_t i = 0; i < fe_data.local_ndof; i++)
        {
            auto pt = pts[i];
            size_t asm_map = connectivity_matrix[cell_offset][i].first ;
            if( connectivity_matrix[cell_offset][i].second )
            {
                if( ( u(pt,msh,cl).second > eps ) && (pt.y() == 0.0) )
                    fe_data.Dirichlet_boundary_inlet[asm_map] = TRUE ;
                if( ( u(pt,msh,cl).first < -eps ) && (pt.x() == 1.0) )
                    fe_data.Dirichlet_boundary_inlet[asm_map] = TRUE ;
                if( ( u(pt,msh,cl).second < -eps ) && (pt.y() == 1.0) )
                    fe_data.Dirichlet_boundary_inlet[asm_map] = TRUE ;
                if( ( u(pt,msh,cl).first > eps ) && (pt.x() == 0.0) )
                    fe_data.Dirichlet_boundary_inlet[asm_map] = TRUE ;
                else
                    fe_data.Dirichlet_boundary_inlet[asm_map] = FALSE ;
                
            }
            else
                fe_data.Dirichlet_boundary_inlet[asm_map] = FALSE ;
            
            
            
        }
    }
    
}



template<typename T>
struct taylor_green_vortex: public velocity_field<T>
{
    T viscosity = 2.0 ;
    T time = 0.0 ;
    T F = std::exp( -2*viscosity * time ) ;
    bool old_tgv = true;
    
    taylor_green_vortex(T a , bool old_tgv): viscosity(a), old_tgv(old_tgv){};
    taylor_green_vortex(){};
    taylor_green_vortex(T a ): viscosity(a){};
    
    Eigen::Matrix<T,2,1> operator()(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        
        if( old_tgv )
        {
            ret(0) =  std::cos( 2*M_PI*pt.x() ) * std::sin( 2*M_PI*pt.y() ) * F ;
            ret(1) = -std::sin( 2*M_PI*pt.x() ) * std::cos( 2*M_PI*pt.y() ) * F ;
        }
        else
        {
            ret(0) =  std::sin( 2*M_PI*pt.x() ) * std::cos( 2*M_PI*pt.y() ) * F ;
            ret(1) = -std::cos( 2*M_PI*pt.x() ) * std::sin( 2*M_PI*pt.y() ) * F ;
        }
        return ret;
    }
    
    void
    set_time( T new_time )
    {
        time = new_time ;
        F = std::exp( -2*viscosity * time ) ;
        
    }
    
    
};


