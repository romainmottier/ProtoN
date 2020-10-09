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





template <typename T>
void positive_part( Eigen::Matrix<T, Dynamic, Dynamic>& mat) {
    for (size_t i = 0; i<mat.rows();i++) {
        for (size_t j = 0; j<mat.cols();j++) {
            if( mat(i,j) < 0 )
                mat(i,j) = 0.;
        }
    }

}





/*

template <typename T>
Eigen::Matrix<T, Dynamic, Dynamic> positive_part(const Eigen::Matrix<T, Dynamic, Dynamic>& mat) {
    //Eigen::Matrix<T, Dynamic, Dynamic> ret =mat;
    Eigen::Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(mat.rows(), mat.cols());
    auto positive = mat.cwiseSign();
    std::cout<<"positive: "<<'\n'<<positive<<std::endl;
    for (size_t i = 0; i<mat.rows();i++) {
        for (size_t j = 0; j<mat.cols();j++) {
            if (positive(i,j)==-1) {
                ret(i,j) = 0;
            }
            //if( mat(i,j) < 0 )
            //    ret(i,j) = 0;
        }
    }
    return ret;
}
*/


template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template<typename T, typename Mesh>
class entropy
{
public:
    virtual T operator()(const point<T,2>& pt, const typename Mesh::cell_type& cl) const
    {
    }
    
    virtual T derivative(const point<T,2>& pt, const typename Mesh::cell_type& cl) const
    {
    }
   
};



template<typename Entropy_func , typename Fonction , typename Mesh , typename Vel_Field,  typename T = typename Mesh::coordinate_type >
struct entropy_flux
{
    T eps;
    Fonction phi;
    Mesh msh;
    Vel_Field u ;
    Entropy_func E ;
    
    Eigen::Matrix<T, Dynamic, 1 > values0 , values1 ;
    
    entropy_flux(Entropy_func& E_entropy , const Fonction& phi , const Vel_Field& u , const Mesh& msh): E(E_entropy), phi(phi) , u(u) , msh(msh){
        eps = E.eps;
        values0 =  Matrix<T, Dynamic, 1 >::Zero(phi.ndof_FE , 1 );
        values1 =  Matrix<T, Dynamic, 1 >::Zero(phi.ndof_FE , 1 );
        Eigen::Matrix<T, Dynamic, 1 > One = Matrix<T, Dynamic, 1 >::Ones(phi.ndof_FE , 1 );
        auto ret = E.E_values + std::log10(eps)*One ;
        values0 = u.sol_FEM.first.cwiseProduct(ret);
        values1 = u.sol_FEM.second.cwiseProduct(ret);
    }
    
    std::pair<T,T> operator()(const point<T,2>& pt, const typename Mesh::cell_type& cl) const
       {
           T ret = E(pt,cl) + std::log10(eps) ;
           return make_pair(u.first(msh,pt,cl)*ret , u.second(msh,pt,cl)*ret );
       }
};

template<typename T , typename Fonction , typename Mesh>
struct non_linear_entropy_new: public entropy<T,Mesh>
{
    T eps;
    Fonction phi;
    T phi_max , phi_min ;
    Mesh msh;
    Eigen::Matrix<T, Dynamic, 1 > E_values , E_der ;
    
    non_linear_entropy_new(T eps , const Fonction& phi ,const Mesh& msh): eps(eps),phi(phi),msh(msh)
    {
        phi_max = phi.phi_max;
        phi_min = phi.phi_min;
        E_values = Matrix<T, Dynamic, 1 >::Zero(phi.ndof_FE , 1 );
        Eigen::Matrix<T, Dynamic, 1 > One = Matrix<T, Dynamic, 1 >::Ones(phi.ndof_FE , 1 );
        E_values = -(( ( (phi.sol_FEM).cwiseProduct( One - phi.sol_FEM ) ).cwiseAbs() + eps*One ).array().log10() );
        
        E_der = Matrix<T, Dynamic, 1 >::Zero(phi.ndof_FE , 1 );
        E_der = ((-One.cwiseQuotient( ( (phi.sol_FEM).cwiseProduct( One - phi.sol_FEM ) ).cwiseAbs() + eps*One ) ) ) ;//.cwiseProduct(  ( (phi.sol_FEM).cwiseProduct( One - phi.sol_FEM ) ).array().sign() )).cwiseProduct( One -2*phi.sol_FEM ) ;
        E_der = (E_der.array().cwiseProduct(  ( (phi.sol_FEM).cwiseProduct( One - phi.sol_FEM ) ).array().sign() )) ;
        E_der = (E_der.cwiseProduct(  One -2*phi.sol_FEM)) ;
        //E_der = -1./( ( std::abs( (1-phi(pt,msh,cl))*phi(pt,msh,cl)  ) +eps )*std::log(10) ) * sgn( (1-phi(pt,msh,cl))*phi(pt,msh,cl)  ) * ( 1 - 2*phi(pt,msh,cl)  ) ;
        
    }
    
    T operator()(const Fonction& f, const point<T,2>& pt, const typename Mesh::cell_type& cl) const
    {
        return -std::log10( std::abs( f(pt,msh,cl)*(1-f(pt,msh,cl)) ) + eps );
    }
    
    T operator()(const point<T,2>& pt , const typename Mesh::cell_type& cl) const
    {
        return -std::log10( std::abs( phi(pt,msh,cl)*(1-phi(pt,msh,cl)) ) + eps );
    }
    
    T derivative(const point<T,2>& pt, const typename Mesh::cell_type& cl) const
    {
       // std::cout<<"il segno di "<<((phi_max-phi(pt,msh,cl))*(phi(pt,msh,cl)-phi_min ))<<" is "<< sgn( (phi_max-phi(pt,msh,cl))*(phi(pt,msh,cl)-phi_min ) )<<std::endl;
        return -1./( ( std::abs( (1-phi(pt,msh,cl))*phi(pt,msh,cl)  ) +eps )*std::log(10) ) * sgn( (1-phi(pt,msh,cl))*phi(pt,msh,cl)  ) * ( 1 - 2*phi(pt,msh,cl)  ) ;
    }
   
};




template<typename T , typename Fonction , typename Mesh>
struct non_linear_entropy: public entropy<T,Mesh>
{
    const T eps;
    Fonction phi;
    T phi_max , phi_min ;
    Mesh msh;
    
    non_linear_entropy(const T eps , const Fonction& phi ,const Mesh& msh): eps(eps),phi(phi),msh(msh)
    {
        phi_max = phi.phi_max;
        phi_min = phi.phi_min;
        
    }
    
    /// THIS WAS FOR A GENERAL PHI, RE-WRITTEN FOR PHI BETWEEN 0 AND 1
    /*
    T operator()(const Fonction& f, const point<T,2>& pt, const typename Mesh::cell_type& cl) const
    {
        return -std::log10( std::abs( (phi_max-f(pt,msh,cl))*(f(pt,msh,cl)-phi_min) ) + eps );
    }
    
    T operator()(const point<T,2>& pt , const typename Mesh::cell_type& cl) const
    {
        return -std::log10( std::abs( (phi_max-phi(pt,msh,cl))*(phi(pt,msh,cl)-phi_min) ) + eps );
    }
    
    T derivative(const point<T,2>& pt, const typename Mesh::cell_type& cl) const
    {
       // std::cout<<"il segno di "<<((phi_max-phi(pt,msh,cl))*(phi(pt,msh,cl)-phi_min ))<<" is "<< sgn( (phi_max-phi(pt,msh,cl))*(phi(pt,msh,cl)-phi_min ) )<<std::endl;
        return -1./(( std::abs( (phi_max-phi(pt,msh,cl))*(phi(pt,msh,cl)-phi_min ) ) +eps)*std::log(10)) * sgn( (phi_max-phi(pt,msh,cl))*(phi(pt,msh,cl)-phi_min ) ) * ( phi_max - 2*phi(pt,msh,cl) + phi_min ) ;
    }
    */
    
    T operator()(const Fonction& f, const point<T,2>& pt, const typename Mesh::cell_type& cl) const
    {
        return -std::log10( std::abs( f(pt,msh,cl)*(1-f(pt,msh,cl)) ) + eps );
    }
    
    T operator()(const point<T,2>& pt , const typename Mesh::cell_type& cl) const
    {
        return -std::log10( std::abs( phi(pt,msh,cl)*(1-phi(pt,msh,cl)) ) + eps );
    }
    
    T derivative(const point<T,2>& pt, const typename Mesh::cell_type& cl) const
    {
       // std::cout<<"il segno di "<<((phi_max-phi(pt,msh,cl))*(phi(pt,msh,cl)-phi_min ))<<" is "<< sgn( (phi_max-phi(pt,msh,cl))*(phi(pt,msh,cl)-phi_min ) )<<std::endl;
        return -1./( ( std::abs( (1-phi(pt,msh,cl))*phi(pt,msh,cl)  ) +eps )*std::log(10) ) * sgn( (1-phi(pt,msh,cl))*phi(pt,msh,cl)  ) * ( 1 - 2*phi(pt,msh,cl)  ) ;
    }
   
};




template < typename VECTOR, typename T >
Matrix<T, Dynamic, 1> solveFEM( const VECTOR& lumped_mass , const VECTOR& conv , const VECTOR& last_term, const VECTOR& phi_old , T dt , const std::vector<size_t>& bdry_nodes)
{
    /*
    Matrix<T, Dynamic, 1> sol =  Eigen::Matrix<T, Dynamic, 1>::Zero(lumped_mass.rows(), 1);
    size_t j = 0;
    for (size_t i = 0 ; i < phi_old.rows() ; i++ ) {
        if( i == bdry_nodes.at(j) ){
            sol(i) = phi_old(i);
            j++; // I can do since bdry_nodes is ordered
        }
        else
           sol(i) = phi_old(i) - dt * conv(i)/( lumped_mass(i) ) + dt * last_term(i)/( lumped_mass(i) );
    }
    return sol;
    */
    // if I dont consider boundary condition here, but just solve for all
    
    
    return phi_old - dt * conv.cwiseQuotient(lumped_mass) + dt * last_term.cwiseQuotient(lumped_mass);
    
}






template < typename Fonction, typename VECTOR, typename MATRIX , typename T >
Matrix<T, Dynamic, 1> solveFEM_Entropic( const MATRIX& mass,const VECTOR& conv, const VECTOR& last_term, const Fonction& phi_old , T dt )
{
    // if I dont consider boundary condition here, but just solve for all-> look to solveFEM
    auto b = phi_old - dt * conv + dt * last_term;
    
    return mass.completeOrthogonalDecomposition().solve(b);
    
    
}

template < typename Fonction, typename VECTOR, typename MATRIX , typename T >
Matrix<T, Dynamic, 1> solveFEM_Entropic_FAST( const MATRIX& llt,const VECTOR& conv, const VECTOR& last_term, const Fonction& phi_old , T dt )
{
    // if I dont consider boundary condition here, but just solve for all-> look to solveFEM
    auto b = phi_old - dt * conv + dt * last_term;
    
    return llt.solve(b);
    
    
}



template< typename VECTOR_LOC , typename VECTOR_GLOB , typename NODES>
void global_update_vector( const VECTOR_LOC& local , VECTOR_GLOB& global , const NODES& nodes_position )
{
    size_t i = 0 ;
   // std::cout<<"numero nodi "<<nodes_position.size()<<std::endl;
    for(auto & k : nodes_position)
    {
        size_t kk = k.ptid;
        //std::cout<<"riga "<<kk<<std::endl;
        // std::cout<<"i "<<i<<std::endl;
        global( kk ) += local ( i ) ;
        i++;
    }

}


template< typename VECTOR_LOC , typename VECTOR_GLOB , typename NODES>
void global_update_vector_NOSUM( const VECTOR_LOC& local , VECTOR_GLOB& global , const NODES& nodes_position )
{
    size_t i = 0 ;
   // std::cout<<"numero nodi "<<nodes_position.size()<<std::endl;
    for(auto & k : nodes_position)
    {
        size_t kk = k.ptid;
        //std::cout<<"riga "<<kk<<std::endl;
        // std::cout<<"i "<<i<<std::endl;
        global( kk ) = local ( i ) ;
        i++;
    }

}


template< typename VECTOR_LOC , typename VECTOR_GLOB , typename NODES>
void global_update_vector_MAX( const VECTOR_LOC& local , VECTOR_GLOB& global , const NODES& nodes_position )
{
    size_t i = 0 ;
   // std::cout<<"numero nodi "<<nodes_position.size()<<std::endl;
    for(auto & k : nodes_position)
    {
        size_t kk = k.ptid;
        //std::cout<<"riga "<<kk<<std::endl;
        // std::cout<<"i "<<i<<std::endl;
        auto max_glob = global( kk );
        global( kk ) = std::max( local ( i ) , max_glob ) ;
        i++;
    }

}

template< typename VECTOR_LOC , typename VECTOR_GLOB , typename NODES>
void global_update_vector_MIN( const VECTOR_LOC& local , VECTOR_GLOB& global , const NODES& nodes_position )
{
    size_t i = 0 ;
   // std::cout<<"numero nodi "<<nodes_position.size()<<std::endl;
    for(auto & k : nodes_position)
    {
        size_t kk = k.ptid;
        //std::cout<<"riga "<<kk<<std::endl;
        // std::cout<<"i "<<i<<std::endl;
        auto min_glob = global( kk );
        global( kk ) = std::min( local ( i ) , min_glob ) ;
        i++;
    }

}




template< typename MATRIX_LOC , typename MATRIX_GLOB , typename NODES>
void global_update_NOSUM( const MATRIX_LOC& local , MATRIX_GLOB& global , const NODES& nodes_position )
{
    size_t i = 0 ;
   // std::cout<<"numero nodi "<<nodes_position.size()<<std::endl;
    for(auto & k : nodes_position)
    {
        size_t j = 0;
        size_t kk = k.ptid;
        //std::cout<<"riga "<<kk<<std::endl;
       // std::cout<<"i "<<i<<std::endl;
        
        for(auto & l : nodes_position)
        {
            size_t ll = l.ptid;
            
            //std::cout<<"colonna "<<ll<<std::endl;
            //std::cout<<"j "<<j<<std::endl;
            global( kk  , ll  ) = local ( i , j ) ;
            j++;
        }
        i++;
        
    }
    
}



template< typename MATRIX_LOC , typename MATRIX_GLOB , typename NODES>
void global_update( const MATRIX_LOC& local , MATRIX_GLOB& global , const NODES& nodes_position )
{
    size_t i = 0 ;
   // std::cout<<"numero nodi "<<nodes_position.size()<<std::endl;
    for(auto & k : nodes_position)
    {
        size_t j = 0;
        size_t kk = k.ptid;
        //std::cout<<"riga "<<kk<<std::endl;
       // std::cout<<"i "<<i<<std::endl;
        
        for(auto & l : nodes_position)
        {
            size_t ll = l.ptid;
            
            //std::cout<<"colonna "<<ll<<std::endl;
            //std::cout<<"j "<<j<<std::endl;
            global( kk  , ll  ) += local ( i , j ) ;
            j++;
        }
        i++;
        
    }
    
}



/*

// OLD IMPLEMENTATION
template< typename MATRIX_LOC , typename MATRIX_GLOB , typename NODES>
void global_update2( const MATRIX_LOC& local , MATRIX_GLOB& global , const NODES& nodes_position )
{
    size_t i = 0 ;
    
   std::cout<<"LOCAL "<<std::endl;
   std::cout<<local<<std::endl;
    
    
    for(auto & k : nodes_position)
    {
        size_t j = 0;
        //size_t kk = k.ptid;
        std::cout<<"riga "<<k<<std::endl;
        std::cout<<"i "<<i<<std::endl;
        
        for(auto & l : nodes_position)
        {
            //size_t ll = l.ptid;
            
            std::cout<<"colonna "<<l<<std::endl;
            std::cout<<"j "<<j<<std::endl;
            
            global( k  , l  ) += local ( i , j ) ;
            j++;
        }
        i++;
        
    }
    std::cout<<"GLOBAL "<<std::endl;
    std::cout<<global<<std::endl;
    
}
*/



std::vector<size_t> boundary_nodes_function( size_t Nx , size_t Ny )
{
    // IT WORKS ONLY FOR Q1!!!
    std::vector<size_t> bdry_nodes ;
    for (size_t j = 1 ; j<=Ny ; j++) {
        bdry_nodes.push_back( j*(Nx+1) -1 ) ;
        bdry_nodes.push_back( j*(Nx+1) ) ;
    }
    for( size_t j = 1 ; j< Nx ; j++){
        bdry_nodes.push_back( j ) ;
        bdry_nodes.push_back( Ny*(Nx+1)+j ) ;
    }
    bdry_nodes.push_back( 0 ) ;
    bdry_nodes.push_back( (Ny+1)*(Nx+1)-1 ) ;
    std::sort(bdry_nodes.begin(), bdry_nodes.end() );
    
    return bdry_nodes;
}


template<typename Node , typename SUPP_NODES>
void supporting_nodes( SUPP_NODES& ret ,  const Node& nodes_position )
{
    
    for( auto & i : nodes_position)
    {
        for( auto & j : nodes_position)
        {
            (ret.at(i.ptid)).insert(j.ptid);
         //    std::cout<<"ret("<<i.ptid<<") is "<<j<<std::endl;
        }
         //std::sort(ret(i.ptid).begin(), ret(i.ptid).end() ); // set already increasing ordered
    }
   
}


template<typename Node , typename MATRIX >
void division_Si( const Node& S_i , const MATRIX& mat1 , const MATRIX& mat2 , MATRIX& sol )
{
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            sol(counter,elem) = mat1(counter,elem)/mat2(counter,elem);
        }
        counter++;
    }
}

template<typename Node , typename MATRIX >
void division_Si_T( const Node& S_i , const MATRIX& mat1 , const MATRIX& mat2 , MATRIX& sol )
{
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            sol(elem,counter) = mat1(elem,counter)/mat2(elem,counter);
        }
        counter++;
    }
}

template<typename Node , typename MATRIX >
void moltiplication_Si_T( const Node& S_i , const MATRIX& mat1 , const MATRIX& mat2 , MATRIX& sol )
{
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            sol(elem,counter) += ( mat1(elem,counter) *mat2(elem,counter) );
        }
        counter++;
    }
}


template<typename Node , typename MATRIX >
void moltiplication_Si( const Node& S_i , const MATRIX& mat1 , const MATRIX& mat2 , MATRIX& sol )
{
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            sol(counter,elem) += ( mat1(counter,elem)*mat2(counter,elem) );
        }
        counter++;
    }
}

template<typename Node , typename MATRIX , typename VECTOR>
void sum_Si( const Node& S_i , const MATRIX& mat , const VECTOR& vec , VECTOR& sol )
{
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            sol(counter) += mat(counter,elem)*vec(elem);
        }
        counter++;
    }
}

template<typename Node , typename MATRIX , typename VECTOR>
void averaged_sum_Si( const Node& S_i , const MATRIX& mat , const VECTOR& vec , VECTOR& sol )
{
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            sol(counter) += mat(counter,elem)*(vec(elem)-vec(counter));
        }
        counter++;
    }
}


/*
template < typename VECTOR, typename MATRIX , typename T , typename POS >
void checking_phi_l( const VECTOR& lumped_mass, const MATRIX& cij_x , const MATRIX& cij_y , const MATRIX& d_ij, const VECTOR& phi_old , const VECTOR& phi_l , T dt , const POS& S_i )
{
 
    
    VECTOR K_term =  Eigen::Matrix<T, Dynamic, 1>::Zero(lumped_mass.rows(), 1);
    VECTOR Dl_term =  Eigen::Matrix<T, Dynamic, 1>::Zero(lumped_mass.rows(), 1);
    // SBAGLIATO NON è PHI_OLD MA IL FLUSSO CHE SERVE QUI DENTRO!
    averaged_sum_Si( S_i , cij_x , phi_old , K_term );
    averaged_sum_Si( S_i , cij_y , phi_old , K_term );
    sum_Si( S_i , d_ij , phi_old , Dl_term );
    //std::cout<<'\n'<<"check Dij "<<'\n'<<Dl_term-d_ij*phi_old<<std::endl;
    VECTOR tmp =  lumped_mass.cwiseProduct( (phi_l - phi_old)/dt ) + K_term - Dl_term;
    std::cout<<'\n'<<"check phi_l vec is "<<'\n'<<tmp<<std::endl;
}
*/
template < typename VECTOR , typename T  >
void checking_phi_lBIS( const VECTOR& lumped_mass, const VECTOR& cij_term , const VECTOR& dij_term, const VECTOR& phi_old , const VECTOR& phi_l , T dt )
{
 
    VECTOR tmp =  lumped_mass.cwiseProduct( (phi_l - phi_old)/dt ) + cij_term - dij_term;
    std::cout<<'\n'<<"CHECK BIS phi_l vec is "<<'\n'<<tmp<<std::endl;
}


/*
template < typename VECTOR , typename MATRIX , typename T , typename POS >
void checking_phi_h( const MATRIX& mass, const MATRIX& cij_x , const MATRIX& cij_y , const MATRIX& dC_ij , const VECTOR& phi_old , const VECTOR& phi_h , T dt , const POS& S_i)
{
 
    VECTOR K_term =  Eigen::Matrix<T, Dynamic, 1>::Zero(mass.rows(), 1);
    VECTOR Dc_term =  Eigen::Matrix<T, Dynamic, 1>::Zero(mass.rows(), 1);
    // SBAGLIATO NON è PHI_OLD MA IL FLUSSO CHE SERVE QUI DENTRO!
    averaged_sum_Si( S_i , cij_x , phi_old , K_term );
    averaged_sum_Si( S_i , cij_y , phi_old , K_term );
    sum_Si( S_i , dC_ij , phi_old , Dc_term );
    VECTOR tmp =  mass*( (phi_h - phi_old)/dt ) + K_term  - dC_ij*phi_old;
   // std::cout<<'\n'<<"check DCij "<<'\n'<<Dc_term-dC_ij*phi_old<<std::endl;
    std::cout<<'\n'<<"check phi_h vec is "<<'\n'<<tmp<<std::endl;
}
*/
template < typename VECTOR , typename T , typename MATRIX  >
void checking_phi_hBIS( const MATRIX& mass, const VECTOR& cij_term , const VECTOR& dij_term, const VECTOR& phi_old , const VECTOR& phi_h , T dt )
{
 
    VECTOR tmp =  mass*( (phi_h - phi_old)/dt ) + cij_term - dij_term;
    std::cout<<'\n'<<"CHECK BIS phi_H vec is "<<'\n'<<tmp<<std::endl;
}




template < typename Entropy , typename Fonction, typename Fonction_TILDE , typename Mesh, typename Vel_Field , typename T , typename POSITION , typename VECTOR >
  void r_i_calculator(const Mesh& msh, const typename Mesh::cell_type& cl , const Entropy& E , const Fonction_TILDE& phi_tilde , const Fonction& phi , T dt , const Vel_Field& u , const POSITION& S_i , VECTOR& Emax_global , VECTOR& Emin_global , VECTOR& R_i  )
{
    size_t di = 0;
    cell_basis_Lagrangian<Mesh,T> cb(msh, cl, phi.degree_FEM);
    auto cbs = cb.size();
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
    auto qps = integrate(msh, cl, 2*(phi.degree_FEM+di));
    
    
    auto nds = nodes(msh,cl);
    iter_swap(nds.end()-1 , nds.end()-2 );
    
    auto pts = points(msh,cl);
    iter_swap(pts.end()-1 , pts.end()-2 );
   
    
    
    size_t counter = 0;
    /// FARE FUNZIONE max_min_entropy PER RENDERE IL CODICE PULITO
    Eigen::Matrix<T, Dynamic, 1> Emax =  Eigen::Matrix<T, Dynamic, 1>::Ones(pts.size(), 1);
    Eigen::Matrix<T, Dynamic, 1> Emin =  Eigen::Matrix<T, Dynamic, 1>::Ones(pts.size(), 1);
    for(auto& nd : nds )
    {
        auto i = nd.ptid ;
        T max_loc = -1e20;
        T min_loc = 1e20;
        for(auto& pt_j : pts )
        {
            Emax(counter) = std::max ( E(pt_j , cl) , max_loc );
            Emin(counter) = std::min ( E(pt_j , cl) , min_loc );
            max_loc = std::max ( Emax(counter) , max_loc );
            min_loc = std::min ( Emin(counter) , min_loc );
        }
        
        counter++;
    }
    global_update_vector_MAX( Emax , Emax_global , nds );
    global_update_vector_MIN( Emin , Emin_global , nds );
    // std::cout<<"Emax_global: "<<'\n'<<Emax_global<<std::endl;
    //  std::cout<<"Emin_global: "<<'\n'<<Emin_global<<std::endl;
    
    for (auto& qp : qps)
    {
        auto bi = cb.eval_basis(qp.first);
        auto phi_grad0 = phi.gradient(qp.first,msh,cl)(0);
        auto phi_grad1 = phi.gradient(qp.first,msh,cl)(1);
        
        auto f = ( ( ( phi_tilde(qp.first,msh,cl)-phi(qp.first,msh,cl) )/dt + u(qp.first,msh,cl).first*phi_grad0 + u(qp.first,msh,cl).second*phi_grad1 ) *
                  E.derivative(qp.first,cl) ); //.cwiseQuotient( Emax - Emin ) ;
        ret += qp.second * bi * f;
    }
    global_update_vector( ret , R_i , nds );
    
}



template < typename T , typename MATRIX , typename VECTOR ,typename POSITION >
VECTOR alfaf_ij_creator( const VECTOR& lumped_mass , const MATRIX& mass , const VECTOR& delta_phi , const VECTOR& phi_L , T dt , const MATRIX& D_ij , const MATRIX& Dc_ij , const VECTOR& phi_old , const POSITION& S_i)
{
    
    MATRIX new_mass = - mass;
    for(size_t i = 0 ; i < new_mass.rows() ; i++){
        new_mass(i,i) += lumped_mass(i);
    }
    
    MATRIX new_D = (Dc_ij - D_ij); //(D_ij - Dc_ij);
    
    MATRIX f_ij = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( mass.rows(), mass.cols() );
    MATRIX alpha_ij = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( mass.rows(), mass.cols() );
    
    VECTOR ret = Eigen::Matrix<T, Dynamic, 1>::Zero( mass.rows(), 1 );
    
    VECTOR P_plus = Eigen::Matrix<T, Dynamic, 1>::Zero(mass.rows(), 1);
    VECTOR P_minus = Eigen::Matrix<T, Dynamic, 1>::Zero(mass.rows(), 1);
    VECTOR Q_plus = Eigen::Matrix<T, Dynamic, 1>::Zero(mass.rows(), 1);
    VECTOR Q_minus = Eigen::Matrix<T, Dynamic, 1>::Zero(mass.rows(), 1);
    
    VECTOR R_plus = Eigen::Matrix<T, Dynamic, 1>::Ones(mass.rows(), 1);
    VECTOR R_minus = Eigen::Matrix<T, Dynamic, 1>::Ones(mass.rows(), 1);
    VECTOR phi_max = phi_old;
    VECTOR phi_min = phi_old;
    
               
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            phi_max(counter) = std::max(phi_old(elem),phi_max(counter));
            phi_min(counter) = std::min(phi_old(elem),phi_min(counter));
        }
        counter++;
    }
    
    for(size_t i = 0; i< f_ij.rows() ; i++)
    {
        for(size_t j = 0; j< f_ij.cols() ; j++)
        {
            f_ij(i, j) = new_mass(i,j)*(delta_phi(j)-delta_phi(i)) +dt*new_D(i,j)*(phi_old(j)-phi_old(i)) ;
            T f_tmp = f_ij(i, j);
            P_plus(i) += std::max( 0. , f_tmp );
            P_minus(i) += std::min( 0. , f_tmp );
        }
        Q_plus(i) = lumped_mass(i)*(phi_max(i)-phi_L(i));
        Q_minus(i) = lumped_mass(i)*(phi_min(i)-phi_L(i));
        if( std::abs(P_plus(i)) > 1e-20 ){
            T Q_P_plus = Q_plus(i)/P_plus(i);
            R_plus(i) = std::min( 1. , Q_P_plus );
        }
        if( std::abs(P_minus(i)) > 1e-20 ){
            T Q_P_minus = Q_minus(i)/P_minus(i);
            R_minus(i) = std::min( 1. , Q_P_minus );
        }
    }
    
    for(size_t i = 0; i< f_ij.rows() ; i++)
    {
        for(size_t j = 0; j< f_ij.cols() ; j++)
        {
            if( f_ij(i, j) > 0 )
                alpha_ij(i,j) = std::min( R_plus(i) , R_minus(j) );
            else
                alpha_ij(i,j) = std::min( R_plus(j) , R_minus(i) );
        }
       
    }
    
    size_t counter2 = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            ret(counter2) += ( alpha_ij(counter2,elem)*f_ij(counter2,elem) );
        }
        counter2++;
    }

    
    // CHECKING F and Alpha properties
    /*
    for(size_t i = 0; i< f_ij.rows() ; i++)
    {
        for(size_t j = 0; j< f_ij.cols() ; j++)
        {
            std::cout<<f_ij(i,j)+f_ij(j,i)<<" , ";
        }
       std::cout<<'\n'<<std::endl;
    }
    */
    
    /*
    std::cout<<"alpha_ij checking symmetry"<<'\n'<<std::endl;
    for(size_t i = 0; i< f_ij.rows() ; i++)
    {
        for(size_t j = 0; j< f_ij.cols() ; j++)
        {
            std::cout<<alpha_ij(i,j)-alpha_ij(j,i)<<" , ";
        }
       std::cout<<'\n'<<std::endl;
    }
    */
    
    /*
    size_t counter3 = 0;
    
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            std::cout<<f_ij(counter3,elem)+f_ij(elem,counter3)<<" , ";
        }
        counter3++;
        std::cout<<'\n';
    }
    std::cout<<std::endl;
     
    counter3 = 0;
     std::cout<<"alpha_ij checking symmetry: METODO 2"<<'\n'<<std::endl;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            std::cout<<alpha_ij(counter3,elem)-alpha_ij(elem,counter3)<<" , ";
        }
        counter3++;
        std::cout<<'\n';
    }
    std::cout<<std::endl;
    */
    //  CHECKING F_IJ
    VECTOR ret0 = Eigen::Matrix<T, Dynamic, 1>::Zero(new_mass.rows(), 1);
    VECTOR ret1 = Eigen::Matrix<T, Dynamic, 1>::Zero(new_mass.rows(), 1);
    averaged_sum_Si( S_i , new_mass , delta_phi , ret0 );
    averaged_sum_Si( S_i , new_D , phi_old , ret1 );
    VECTOR f_i = (ret0 + dt*ret1);
    
    VECTOR f_i_NEW = Eigen::Matrix<T, Dynamic, 1>::Zero(new_mass.rows(), 1);
    VECTOR One_Vec = Eigen::Matrix<T, Dynamic, 1>::Ones(new_mass.rows(), 1);
    sum_Si( S_i , f_ij , One_Vec , f_i_NEW );
    
    //std::cout<<'\n'<<"Diff f_i 2 meths is:"<<'\n'<< f_i_NEW - f_i<<std::endl;
    
    return  ret ;
    
}




template < typename T , typename MATRIX , typename VECTOR ,typename POSITION >
VECTOR f_ij_creator( const VECTOR& lumped_mass , const MATRIX& mass , const VECTOR& delta_phi , T dt , const MATRIX& D_ij , const MATRIX& Dc_ij , const VECTOR& phi_old , const POSITION& S_i)
{
    
    MATRIX new_mass = - mass;
    MATRIX new_D = (Dc_ij - D_ij); //(D_ij - Dc_ij);
    for(size_t i = 0 ; i < new_mass.rows() ; i++){
        new_mass(i,i) += lumped_mass(i);
    }
    
    /*
    std::cout<<"new_mass sum check: "<<std::endl;
    for(size_t i = 0 ; i < new_mass.rows() ; i++){
        new_mass(i,i) += lumped_mass(i);
        std::cout<<(new_mass.row(i)).sum()<<std::endl;
    }
    std::cout<<"new_D check: "<<std::endl;
    for(size_t i = 0 ; i < new_D.rows() ; i++){
       std::cout<<(new_D.row(i).sum() )<<std::endl;
    }
    */
    
    VECTOR ret0 = Eigen::Matrix<T, Dynamic, 1>::Zero(new_mass.rows(), 1);
    VECTOR ret1 = Eigen::Matrix<T, Dynamic, 1>::Zero(new_mass.rows(), 1);
    averaged_sum_Si( S_i , new_mass , delta_phi , ret0 );
    averaged_sum_Si( S_i , new_D , phi_old , ret1 );
    /*
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            sol(counter) += mat(counter,elem)*(vec(elem)-vec(counter));
        }
        counter++;
    }
    */
    //  std::cout<<'\n'<<"dt: "<<'\n'<<dt<<std::endl;
   
    //    std::cout<<'\n'<<"ret0: "<<'\n'<<ret0<<std::endl;
    //    std::cout<<'\n'<<"ret1: "<<'\n'<<dt*ret1<<std::endl;
  
    return (ret0 + dt*ret1) ;
    
}
template<typename FUNCTION, typename T>
void mapping_phi(FUNCTION& phi , T phi_max , T phi_min)
{
    Matrix<T, Dynamic, 1> Vec_One = Eigen::Matrix<T, Dynamic, 1>::Ones( phi.vertices.rows(), 1 );
    Matrix<T, Dynamic, Dynamic> Mat_One = Eigen::Matrix<T, Dynamic, Dynamic>::Ones( phi.values_bis.rows(), phi.values_bis.cols() );
    // mapping between 0 and 1
    phi.vertices = (phi.vertices-phi_min*Vec_One)/( phi_max - phi_min );
    phi.values_bis = (phi.values_bis-phi_min*Mat_One)/( phi_max - phi_min );
}

template<typename FUNCTION, typename T>
void inverse_mapping_phi(FUNCTION& phi, T phi_max , T phi_min)
{
    Matrix<T, Dynamic, 1> Vec_One = Eigen::Matrix<T, Dynamic, 1>::Ones( phi.vertices.rows() , 1 );
    Matrix<T, Dynamic, Dynamic> Mat_One = Eigen::Matrix<T, Dynamic, Dynamic>::Ones( phi.values_bis.rows(), phi.values_bis.cols() );
    phi.vertices = phi_min*Vec_One + phi.vertices*( phi_max - phi_min );
    phi.values_bis = phi_min*Mat_One + phi.values_bis*( phi_max - phi_min );
}






template< typename VEC , typename DIAG , typename T >
T time_step_CFL_L2_velocity_NEW( const DIAG& dii , const VEC& lumped_mass , const std::vector< bool >& Dirichlet_boundary , T& dt )
{
    T eps = 0.0;
    T tau_min  = dt;
    T CFL_numb = 10.0;
    size_t i = 0 ;
    for (const auto& dir_elem : Dirichlet_boundary )
    {
        if(!dir_elem){
            if( (1 + 2*tau_min*dii.coeff(i)/lumped_mass(i) ) < eps ){
                std::cout<<"tau_min PRE modification is "<<tau_min;
                tau_min = (eps-1.0)/2.0*lumped_mass(i)/dii.coeff(i);
                std::cout<<" and tau_min POST modification is "<<tau_min<<std::endl;
            }
            CFL_numb = std::min( CFL_numb ,(eps-1.0)/2.0*lumped_mass(i)/dii.coeff(i) );
        }
        i++ ;
    }
    std::cout<<"CFL_numb ---> "<<CFL_numb<<std::endl;
    std::cout<<yellow<<bold<<"dt is "<<dt<<" and tau min is "<<tau_min<<reset<<std::endl;
    dt = std::min(dt,tau_min);
    return CFL_numb;
}



template < typename Entropy , typename Fonction, typename Fonction_TILDE , typename Mesh, typename Vel_Field , typename T , typename VECTOR >
  void r_i_calculator_Bernstein(const Mesh& msh, const typename Mesh::cell_type& cl , const Entropy& E , const Fonction_TILDE& phi_tilde , const Fonction& phi , T dt , const Vel_Field& u , VECTOR& Emax_global , VECTOR& Emin_global , VECTOR& R_i  )
{
    size_t di = 1;
    cell_basis_Bernstein<Mesh,T> cb(msh, cl, phi.degree_FEM);
    auto cbs = cb.size();
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
    auto qps = integrate(msh, cl, 2*(phi.degree_FEM+di));
    
    auto nds = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, phi.degree_FEM );
    
    size_t offset_cell = offset( msh , cl ) ;
    
    T max_loc = -1e20;
    T min_loc = 1e20;
    
    for(auto& ndj : nds )
    {
        max_loc = std::max ( E(ndj , cl) , max_loc );
        min_loc = std::min ( E(ndj , cl) , min_loc );
    }
    
    for (size_t i = 0; i < phi.local_dim; i++)
    {
        size_t asm_map =  phi.connectivity_matrix[offset_cell][i].first ;
        Emax_global(asm_map) = std::max(Emax_global(asm_map) , max_loc );
        Emin_global(asm_map) = std::min(Emin_global(asm_map) , min_loc ) ;
    }
    
    //std::cout<<"Emax_global: "<<'\n'<<Emax_global<<std::endl;
    //std::cout<<"Emin_global: "<<'\n'<<Emin_global<<std::endl;
    
    
    for (auto& qp : qps)
    {
        auto bi = cb.eval_basis(qp.first);
        auto phi_grad0 = phi.gradient(qp.first,msh,cl)(0);
        auto phi_grad1 = phi.gradient(qp.first,msh,cl)(1);
        
        auto f = ( ( ( phi_tilde(qp.first,msh,cl)-phi(qp.first,msh,cl) )/dt + u(qp.first,msh,cl).first * phi_grad0 + u(qp.first,msh,cl).second * phi_grad1 ) *
                  E.derivative(qp.first,cl) ); //.cwiseQuotient( Emax - Emin ) ;
        ret += qp.second * bi * f;
    }
    
    for (size_t i = 0; i < phi.local_dim; i++)
    {
        size_t asm_map =  phi.connectivity_matrix[offset_cell][i].first ;
        R_i(asm_map) += ret(i) ;
    }
   
    
}


template <typename T>
void positive_part_SPARSE( SparseMatrix<T>& mat) {
    for (size_t i = 0; i<mat.rows();i++) {
        for (size_t j = 0; j<mat.cols();j++) {
            if( mat.coeff(i,j) < 0. )
                mat.coeffRef(i,j) = 0.;
        }
    }

}


template < typename T , typename MATRIX , typename VECTOR ,typename POSITION >
VECTOR alfaf_ij_creator_SPARSE( const VECTOR& lumped_mass , const MATRIX& mass , const VECTOR& delta_phi , const VECTOR& phi_L , T dt , const MATRIX& D_ij , const MATRIX& Dc_ij , const VECTOR& phi_old , const POSITION& S_i)
{
    
    MATRIX new_mass = - mass;
    for(size_t i = 0 ; i < new_mass.rows() ; i++){
        new_mass.coeffRef(i,i) += lumped_mass(i);
    }
    
    MATRIX new_D = (Dc_ij - D_ij); //(D_ij - Dc_ij);
    
    SparseMatrix<T> f_ij =  SparseMatrix<T>( mass.rows() , mass.cols() );
    std::vector< Triplet<T> >   triplets0;
    SparseMatrix<T> alpha_ij =  SparseMatrix<T>( mass.rows() , mass.cols() );
    std::vector< Triplet<T> >   triplets1;
    
    
    //MATRIX f_ij = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( mass.rows(), mass.cols() );
    //MATRIX alpha_ij = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( mass.rows(), mass.cols() );
    
    VECTOR ret = Eigen::Matrix<T, Dynamic, 1>::Zero( mass.rows(), 1 );
    
    VECTOR P_plus = Eigen::Matrix<T, Dynamic, 1>::Zero(mass.rows(), 1);
    VECTOR P_minus = Eigen::Matrix<T, Dynamic, 1>::Zero(mass.rows(), 1);
    VECTOR Q_plus = Eigen::Matrix<T, Dynamic, 1>::Zero(mass.rows(), 1);
    VECTOR Q_minus = Eigen::Matrix<T, Dynamic, 1>::Zero(mass.rows(), 1);
    
    VECTOR R_plus = Eigen::Matrix<T, Dynamic, 1>::Ones(mass.rows(), 1);
    VECTOR R_minus = Eigen::Matrix<T, Dynamic, 1>::Ones(mass.rows(), 1);
    VECTOR phi_max = phi_old;
    VECTOR phi_min = phi_old;
    
               
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            phi_max(counter) = std::max(phi_old(elem),phi_max(counter));
            phi_min(counter) = std::min(phi_old(elem),phi_min(counter));
        }
        counter++;
    }
    
    counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            auto value = new_mass.coeff(counter,elem)*(delta_phi(elem)-delta_phi(counter)) + dt*new_D.coeff(counter,elem)*(phi_old(elem)-phi_old(counter)) ;
            
            triplets0.push_back( Triplet<T>(counter, elem , value ));
            P_plus(counter) += std::max( 0. , value );
            P_minus(counter) += std::min( 0. , value );
                                
        }
        
        Q_plus(counter) = lumped_mass(counter)*(phi_max(counter)-phi_L(counter));
        Q_minus(counter) = lumped_mass(counter)*(phi_min(counter)-phi_L(counter));
        if( std::abs(P_plus(counter)) > 1e-20 ){
            T Q_P_plus = Q_plus(counter)/P_plus(counter);
            R_plus(counter) = std::min( 1.0 , Q_P_plus );
        }
        if( std::abs(P_minus(counter)) > 1e-20 ){
            T Q_P_minus = Q_minus(counter)/P_minus(counter);
            R_minus(counter) = std::min( 1.0 , Q_P_minus );
        }
        
        
        counter++;
    }
    

    f_ij.setFromTriplets( triplets0.begin(), triplets0.end() );
    triplets0.clear();
    
    size_t i = 0 ;
    for(auto& row_i:S_i)
    {
        for(auto& j:row_i)
        {
            if( f_ij.coeff(i, j) > 0 ){
                auto value = std::min( R_plus(i) , R_minus(j) );
                triplets1.push_back( Triplet<T>(i, j , value ));
                
            }
            else{
                auto value = std::min( R_plus(j) , R_minus(i) );
                triplets1.push_back( Triplet<T>(i, j , value ));
                
            }
        }
        i++;
    }
    
    alpha_ij.setFromTriplets( triplets1.begin(), triplets1.end() );
    triplets1.clear();
    
    
    size_t counter2 = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            ret(counter2) += ( alpha_ij.coeff(counter2,elem)*f_ij.coeff(counter2,elem) );
        }
        counter2++;
    }

    
    // CHECKING F and Alpha properties
    
    /*
    std::cout<<"f_ij checking symmetry"<<'\n'<<std::endl;
    for(size_t i = 0; i< f_ij.rows() ; i++)
    {
        for(size_t j = 0; j< f_ij.cols() ; j++)
        {
            std::cout<<f_ij.coeff(i,j)+f_ij.coeff(j,i)<<" , ";
        }
       std::cout<<'\n'<<std::endl;
    }
    
    
    
    std::cout<<"alpha_ij checking symmetry"<<'\n'<<std::endl;
    for(size_t i = 0; i< f_ij.rows() ; i++)
    {
        for(size_t j = 0; j< f_ij.cols() ; j++)
        {
            std::cout<<alpha_ij.coeff(i,j)-alpha_ij.coeff(j,i)<<" , ";
        }
       std::cout<<'\n'<<std::endl;
    }
    */
    
    
    
    /*
    std::cout<<"f_ij checking symmetry"<<'\n'<<std::endl;
    size_t counter3 = 0;
    
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            std::cout<<f_ij.coeff(counter3,elem)+f_ij.coeff(elem,counter3)<<" , ";
        }
        counter3++;
        std::cout<<'\n';
    }
    std::cout<<std::endl;
     
    counter3 = 0;
     std::cout<<"alpha_ij checking symmetry: METODO 2"<<'\n'<<std::endl;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            std::cout<<alpha_ij.coeff(counter3,elem)-alpha_ij.coeff(elem,counter3)<<" , ";
        }
        counter3++;
        std::cout<<'\n';
    }
    std::cout<<std::endl;
    
    //  CHECKING F_IJ
    
    VECTOR ret0 = Eigen::Matrix<T, Dynamic, 1>::Zero(new_mass.rows(), 1);
    VECTOR ret1 = Eigen::Matrix<T, Dynamic, 1>::Zero(new_mass.rows(), 1);
    size_t counter4 = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            ret0(counter4) += new_mass.coeff(counter4,elem)*(delta_phi(elem)-delta_phi(counter4));
            ret1(counter4) += new_D.coeff(counter4,elem)*(phi_old(elem)-phi_old(counter4));
        }
        counter++;
    }
    
    VECTOR f_i = (ret0 + dt*ret1);
    
    VECTOR f_i_NEW = Eigen::Matrix<T, Dynamic, 1>::Zero(new_mass.rows(), 1);
    VECTOR One_Vec = Eigen::Matrix<T, Dynamic, 1>::Ones(new_mass.rows(), 1);
    //sum_Si( S_i , f_ij , One_Vec , f_i_NEW );
    size_t counter5 = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            f_i_NEW(counter5) += f_ij.coeff(counter5,elem)*(One_Vec(elem));
        }
        counter5++;
    }
    
    std::cout<<'\n'<<"Diff f_i 2 meths is:"<<'\n'<< f_i_NEW - f_i<<std::endl;
    */
    return  ret ;
    
}



template < typename T , typename MATRIX , typename VECTOR ,typename POSITION >
VECTOR f_ij_creator_SPARSE( const VECTOR& lumped_mass , const MATRIX& mass , const VECTOR& delta_phi , T dt , const MATRIX& D_ij , const MATRIX& Dc_ij , const VECTOR& phi_old , const POSITION& S_i)
{
    
    MATRIX new_mass = - mass;
    MATRIX new_D = (Dc_ij - D_ij); //(D_ij - Dc_ij);
    for(size_t i = 0 ; i < new_mass.rows() ; i++){
        new_mass.coeffRef(i,i) += lumped_mass(i);
    }
    
    /*
    std::cout<<"new_mass sum check: "<<std::endl;
    for(size_t i = 0 ; i < new_mass.rows() ; i++){
        new_mass(i,i) += lumped_mass(i);
        std::cout<<(new_mass.row(i)).sum()<<std::endl;
    }
    std::cout<<"new_D check: "<<std::endl;
    for(size_t i = 0 ; i < new_D.rows() ; i++){
       std::cout<<(new_D.row(i).sum() )<<std::endl;
    }
    */
    
    VECTOR ret0 = Eigen::Matrix<T, Dynamic, 1>::Zero(new_mass.rows(), 1);
    VECTOR ret1 = Eigen::Matrix<T, Dynamic, 1>::Zero(new_mass.rows(), 1);
    //averaged_sum_Si( S_i , new_mass , delta_phi , ret0 );
    //averaged_sum_Si( S_i , new_D , phi_old , ret1 );
    
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            ret0(counter) += new_mass.coeff(counter,elem)*(delta_phi(elem)-delta_phi(counter));
            ret1(counter) += new_D.coeff(counter,elem)*(phi_old(elem)-phi_old(counter));
            
            
        }
        counter++;
    }
    
    //  std::cout<<'\n'<<"dt: "<<'\n'<<dt<<std::endl;
   
    //    std::cout<<'\n'<<"ret0: "<<'\n'<<ret0<<std::endl;
    //    std::cout<<'\n'<<"ret1: "<<'\n'<<dt*ret1<<std::endl;
  
    return (ret0 + dt*ret1) ;
    
}

template < typename Fonction, typename Mesh, typename Vel_Field , typename FiniteSpace , typename T = typename Mesh::coordinate_type >
void
run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D_NEW_DIRICHLET_COND(const Mesh & msh, const FiniteSpace& fe_data, Fonction & phi , Vel_Field& u , T& dt , bool mapping )
{
    // Starting time for FE calculation
    std::cout<<yellow<<bold<<"----------- STARTING TRANSPORT PROBLEM -----------"<<reset<<std::endl;
    //std::cout<<yellow<<bold<<"PROVA--- USO MIXC LAGRANGE- BERNSTEIN"<<reset<<std::endl;
    
    timecounter tc;
    tc.tic();
    
    size_t degree = fe_data.order; // finite element order
    size_t dim = fe_data.ndof_FE ;
    //size_t n_cls = fe_data.n_cls ;
    size_t local_ndof = fe_data.local_ndof ; // local degrees of freedom
    auto S_i = fe_data.S_i;
    
   
    if(!mapping)
        phi.coefficients_mapping_MAX_MAX( );
    
    //phi.coefficients_mapping(); // mapping of phi to have a phi between 0 and 1
    //phi_exact.coefficients_mapping(); // mapping of phi to have a phi between 0 and 1
   
    // SAVING PHI AND VELOCITY COEFFS
    auto phi_FEM = phi.sol_FEM ;
    auto u0 = u.sol_FEM.first ;
    auto u1 = u.sol_FEM.second ;
    auto u0_cellwise = u.sol_HHO.first ;
    auto u1_cellwise = u.sol_HHO.second ;
    
  
    
    // NON LINEAR ENTROPY INITIALISATION
    T eps = 1e-14 ; //constant into entropy
    non_linear_entropy_new<T,Fonction,Mesh> E(eps , phi ,msh );
    typedef non_linear_entropy_new<T,Fonction,Mesh> Entropy_func;
    entropy_flux<Entropy_func,Fonction,Mesh,Vel_Field,T> q_entropy(E , phi , u , msh );
    
    // PHI TILDE INITIALISATION --> (FOR HIGH ORDER METHOD)
    auto phi_tilde = L2_projection< T, Mesh , FiniteSpace> ( fe_data , msh );
    
    
    // SAVING OF USEFUL MATRICES
    auto global_mass = phi.Global_Mass ;
    Matrix<T, Dynamic, 1> global_lumped_mass = phi.Global_Mass_Lumped;
    
    auto global_cij_x = phi.Global_c_term_x ;
    auto global_cij_y = phi.Global_c_term_y ;
    auto local_vandermonde = phi.local_vandermonde ;
    
    auto cij_norm = phi.cij_norm ;
    auto nij0 = phi.nij0 ;
    auto nij1 = phi.nij1 ;
    
    auto cji_norm = phi.cij_norm ;
    auto nji0 = phi.nji0 ;
    auto nji1 = phi.nji1 ;
    
    
    
    // INITIALISATION OF THE SOLVER (CONJUGATE GRADIENT)
    /*
    ConjugateGradient<SparseMatrix<T> > solver_prova;
    solver_prova.compute(global_mass);
    if(solver_prova.info()!=Success){
           std::cout<<"FAILED SOLVER PROVA."<<std::endl;
           exit(1);
       }
    */
    timecounter tc_solver;
    tc_solver.tic();
    
    SimplicialLLT<SparseMatrix<T> >solver_global_mass;
    solver_global_mass.compute(global_mass); // use solver_global_mass to solve M^-1
    
    if(solver_global_mass.info()!=Success){
        std::cout<<"FAILED SOLVER LLT."<<std::endl;
        exit(1);
    }
    
    tc_solver.toc();
    std::cout << bold << yellow << "INVERSION WITH CHOLESKY METHOD, t = " << tc_solver << " seconds" << reset << std::endl;
       
    
    /*
    timecounter tc_solver1_bis;
    tc_solver1_bis.tic();
    ConjugateGradient<SparseMatrix<double>, Lower|Upper> solver2_bis;
    solver2_bis.compute(global_mass);
    if(solver2_bis.info()!=Success) {
        std::cout<<"FAILED SOLVER2 PROVA ->phi_tilde"<<std::endl;
        return;
    }
       
    tc_solver1_bis.toc();
    std::cout << bold << yellow << "INVERSION WITH ITERATIVE CG METHOD, t = " << tc_solver1_bis << " seconds" << reset << std::endl;
    
    */
    
   
    
    
    // ALTERNATIVE VANDERMONDE MATRIX
    size_t i_fl = 0 ;
    Matrix<T, Dynamic, 1> flux0_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    Matrix<T, Dynamic, 1> flux1_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    
    Matrix<T, Dynamic, 1> flux0 = Matrix<T, Dynamic, 1>::Zero(dim) ;
    Matrix<T, Dynamic, 1> flux1 = Matrix<T, Dynamic, 1>::Zero(dim) ;

    timecounter tc_solver2;
    tc_solver2.tic();
    
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
    

    tc_solver2.toc();
    std::cout << bold << yellow << "DIRECT INVERSION OF VANDERMONDE MATRIX LOCAL, t = " << tc_solver2 << " seconds" << reset << std::endl;
    
    
    
    timecounter tc_case00;
    tc_case00.tic();
    
    // RESOLUTION OF phi_tilde (GLOBALLY)    ( with cij(phi_j) )
    Matrix<T, Dynamic, 1> mass_phi_old = global_mass * phi_FEM ;
    //std::cout<<"vec1  "<<'\n'<<mass_phi_old<<std::endl;
    
    // CONVOLUTION TERM
    Matrix<T, Dynamic, 1> conv_global = global_cij_x * flux0  + global_cij_y * flux1 ;
    
    
    
    // TERM d_ij + CALCULATION OF MAX AND MIN OF THE ENTROPY
    SparseMatrix<T> dij = SparseMatrix<T>( dim , dim );
    std::vector< Triplet<T> >   triplets_dij;
    
    // TERM R_i^n
    //Matrix<T, Dynamic, 1> Emax_global = Eigen::Matrix<T, Dynamic, 1>::Ones(dim, 1);
    //Matrix<T, Dynamic, 1> Emin_global = Eigen::Matrix<T, Dynamic, 1>::Ones(dim, 1);
    //Emax_global *= -1e20;
    //Emin_global *= 1e20;
    
    // TERM R_i^n
    Matrix<T, Dynamic, 1> R_i = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    
    
    size_t counter_row = 0;
    for(auto& row_i:S_i)
    {
        T sum_row = 0.0 ;
        T N_i_entropic = 0.0;
        T D_i_entropic0 = 0.0;
        T D_i_entropic1 = 0.0;
        for(auto& elem:row_i)
        {
            
            //Emax_global(counter_row) = std::max ( E.E_values(elem) , Emax_global(counter_row) );
            //Emin_global(counter_row) = std::min ( E.E_values(elem) , Emin_global(counter_row) );
            N_i_entropic += ( (q_entropy.values0(elem) - E.E_der(counter_row)*flux0(elem) )*global_cij_x.coeff(counter_row,elem) + (q_entropy.values1(elem) - E.E_der(counter_row)*flux1(elem) )*global_cij_y.coeff(counter_row,elem) );
            D_i_entropic0 += ( q_entropy.values0(elem) * global_cij_x.coeff(counter_row,elem) + q_entropy.values1(elem) * global_cij_y.coeff(counter_row,elem) );
            D_i_entropic1 += ( flux0(elem)*global_cij_x.coeff(counter_row,elem) + flux1(elem)*global_cij_y.coeff(counter_row,elem) ) ;
            
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
            
            if( counter_row != elem )
                triplets_dij.push_back( Triplet<T>(counter_row, elem, val_dij ) );
      
            
        }
        triplets_dij.push_back( Triplet<T>(counter_row, counter_row, -sum_row ) );
        
        R_i(counter_row) = std::abs(N_i_entropic)/( std::abs(D_i_entropic0) + std::abs(D_i_entropic1)*std::abs(E.E_der(counter_row)) ) ;
        
        
        counter_row++;
        
    }
    
    dij.setFromTriplets( triplets_dij.begin(), triplets_dij.end() );
    triplets_dij.clear();
    
    
    
    tc_case00.toc();
    std::cout << bold << yellow << "RESOLUTION OF LOW ORDER TRANSPORT, t = " << tc_case00 << " seconds" << reset << std::endl;
    
    timecounter tc_case01;
    tc_case01.tic();
    
    // CHECK TIME STEP dt
    T dt_old = dt ;
    std::cout<<bold<<yellow<<"---> COND IN TEMPO CFL, ALEXANDRE BOOK"<<reset<<std::endl;
    T CFL_numb = time_step_CFL_L2_velocity_NEW( dij.diagonal() , global_lumped_mass , fe_data.Dirichlet_boundary , dt );
    
    T nu_max0 = CFL_numb/fe_data.hx;
    T nu0 = dt_old/fe_data.hx;
    T nu1 = dt/fe_data.hx;
    
    std::cout<<"VALID FOR u = (1,0). nu_max VERO = "<<nu_max0<<" , nu max con dt assegnato = "<<nu0<< " and with dt appeared by CFL COND "<<nu1<<std::endl;
    
    if(dt_old != dt )
    {
        std::cout<<"dt is "<<dt_old<<" and dt CFL is "<<dt<<" . STOP!"<<std::endl;
        exit(10);
    }
    
    tc_case01.toc();
    std::cout << bold << yellow << "TIME CHECKING, t = " << tc_case01 << " seconds" << reset << std::endl;
    
    
    
    
    // CONSTANT TERM (PHI TILDE PROBLEM)
    //Matrix<T, Dynamic, 1> b = mass_phi_old - dt*conv_global.cwiseQuotient(global_lumped_mass);
    //std::cout<<"TERMINE NOTO:b  "<<'\n'<<b<<std::endl;
    
  /*
    timecounter tc_solver3;
    tc_solver3.tic();
    
    // RESOLUTION OF PHI_TILDE
    phi_tilde.sol_FEM = solver_global_mass.solve(b); // SAVE THE L2 projection
    //auto prova0 = solver_prova.solve(b); // SAVE THE L2 projection
    // norm() is L2 norm
    T relative_error0 = (global_mass*phi_tilde.sol_FEM - b).norm() / b.norm();
    std::cout << "The relative error is: " << relative_error0 << std::endl;
    //std::cout << "b norm =  " << b.norm() << " , (global_mass*phi_tilde.sol_FEM - b).norm() =  "<< (global_mass*phi_tilde.sol_FEM - b).norm() << std::endl;
    
    //T error0_prova = (global_mass*prova0 - b).norm() / b.norm();
    
    //std::cout << "The PROVA error is: " << error0_prova << std::endl;
    //std::cout << "b norm =  " << b.norm() << " , (global_mass*prova0 - b).norm() =  "<< (global_mass*prova0 - b).norm() << std::endl;
    
    tc_solver3.toc();
    std::cout << bold << yellow << "INVERSION OF phi_tilde, t = " << tc_solver3 << " seconds" << reset << std::endl;
    
    // SAVING BOTH SOL_HHO AND VERTICES OF PHI_TILDE
    std::cout<<"CONVERTING phi_tilde"<<std::endl;
    timecounter tc_solver4;
    tc_solver4.tic();
    phi_tilde.converting_into_HHO_formulation( phi_tilde.sol_FEM );
    tc_solver4.toc();
    std::cout << bold << yellow << "CONVERTING phi_tilde, t = " << tc_solver4 << " seconds" << reset << std::endl;
    
    
  */
    
    timecounter tc_case02;
    tc_case02.tic();
    
 /*   // TERM R_i^n
    Matrix<T, Dynamic, 1> R_i_bis = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    
    for( auto& cl: msh.cells )
    {
        size_t di = 1;
        size_t offset_cell = offset(msh,cl) ;
        cell_basis_Bernstein<Mesh,T> cb(msh, cl, phi.degree_FEM);
        auto cbs = cb.size();
        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
        //auto qps = integrate(msh, cl, 2*(phi.degree_FEM+di));
        //auto qps = integrate(msh, cl, (phi.degree_FEM)+di);
        auto qps = integrate(msh, cl, di);
        
        for (auto& qp : qps)
        {
            auto bi = cb.eval_basis(qp.first);
            auto phi_grad0 = phi.gradient(qp.first,msh,cl)(0);
            auto phi_grad1 = phi.gradient(qp.first,msh,cl)(1);
            
            auto f = ( ( ( phi_tilde(qp.first,msh,cl)-phi(qp.first,msh,cl) )/dt + u(qp.first,msh,cl).first * phi_grad0 + u(qp.first,msh,cl).second * phi_grad1 ) * E.derivative(qp.first,cl) );
            ret += qp.second * bi * f;
        }
        for (size_t i = 0; i < phi.local_dim; i++)
        {
            size_t asm_map =  phi.connectivity_matrix[offset_cell][i].first ;
            R_i_bis(asm_map) += ret(i) ;
        }

    }
    
  
    // R_i FINALISATION:
    //std::cout<<"Emax_global: "<<'\n'<<Emax_global<<std::endl;
    //std::cout<<"Emin_global: "<<'\n'<<Emin_global<<std::endl;
    R_i_bis = R_i_bis.cwiseQuotient( Emax_global - Emin_global );
    //std::cout<<"Emax_global - Emin_global = "<<'\n'<<Emax_global-Emin_global<<std::endl;
    std::cout<<"Ri - R_i_bis = "<<'\n'<<R_i- R_i_bis <<std::endl;
 
    tc_case02.toc();
    std::cout << bold << yellow << "R_i PROCESS, t = " << tc_case02 << " seconds" << reset << std::endl;
 */
    timecounter tc_case03;
    tc_case03.tic();
    
    // ENTROPIC SOLUTION: MINIMUM BETWEEN d_ij AND R_i --> d^E_ij MATRIX
    T c_e = 1.0;
    T c_comp = 1.0;
    
    
    SparseMatrix<T> dC_ij =  SparseMatrix<T>( dim , dim );
    std::vector< Triplet<T> >   triplets_dC_ij;
    
    Matrix<T, Dynamic, 1> term_dij_no_entropy =  Eigen::Matrix<T,Dynamic,1>::Zero(dim, 1);
    Matrix<T, Dynamic, 1> term_dij            =  Eigen::Matrix<T,Dynamic,1>::Zero(dim, 1);
    
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            
            if(elem!=counter)
            {
                
                auto R_i_j = c_e * std::max( std::abs(R_i(counter)),std::abs(R_i(elem)) );
                //auto value_E = std::min( dij.coeff( counter , elem ) , R_i_j );
                auto value_E = dij.coeff( counter , elem ) * R_i_j ;
                //triplets_dE_ij.push_back( Triplet<T>(counter, elem, value_E ) );
                
                auto value_1 = 0.5*( phi_FEM(counter) + phi_FEM(elem) );
                auto value_2 = std::max( value_1*(1.0 -value_1) , 0.0 );
                //triplets_phi_ij.push_back( Triplet<T>(counter, elem, value_bis ) );
                //T value = 0.0;
                bool check_else = FALSE;
                T value = value_2/( std::abs(phi_FEM(counter) - phi_FEM(elem)) );
                /*
                if( (std::abs(phi_FEM(counter) - phi_FEM(elem))>1e-15) && (std::abs(value_2)>1e-15) ){
                    value = value_2/( std::abs(phi_FEM(counter) - phi_FEM(elem)) );
                }
                */
                if( (std::abs(phi_FEM(counter) - phi_FEM(elem))<1e-20) && (std::abs(value_2)<1e-20) ){
                    //std::cout<<"SONO IN ELSE: std::abs(phi_FEM(counter) - phi_FEM(elem)) = "<<std::abs(phi_FEM(counter) - phi_FEM(elem))<< " and value_2 = "<<value_2<<std::endl;
                    //std::cout<<"elem = "<<elem << " and counter = "<<counter<<std::endl;
                    //auto value_C = std::max( 1.0 - c_comp * value , 0.0 ) ;
                    //std::cout<<"1.0 - c_comp * value = "<<(1.0 - c_comp * value) << " and value_C = "<<value_C<<std::endl;
                    check_else = TRUE;
                }
                /*
                else{
                    std::cout<<"SONO IN ELSE: std::abs(phi_FEM(counter) - phi_FEM(elem)) = "<<std::abs(phi_FEM(counter) - phi_FEM(elem))<< " and value_2 = "<<value_2<<std::endl;
                    std::cout<<"elem = "<<elem << " and counter = "<<counter<<std::endl;
                    auto value_prova = value_2/( std::abs(phi_FEM(counter) - phi_FEM(elem)) ) ;
                    auto value_C = std::max( 1.0 - c_comp * value_prova , 0.0 ) ;
                    std::cout<<"1.0 - c_comp * value = "<<(1.0 - c_comp * value_prova) << " and value_C = "<<value_C<<std::endl;
                    check_else = TRUE;
                }
                */
                auto value_C = std::max( 1.0 - c_comp * value , 0.0 ) ;
                if(check_else){
                    //std::cout<<"value_C GIUSTO = "<<value_C <<std::endl;
                    value = (value_2 )/( std::abs(phi_FEM(counter) - phi_FEM(elem))+ 1e-18 );
                    value_C = std::max( 1.0 - c_comp * value , 0.0 ) ;
                    //value_C = 1.0;
                    //std::cout<<"Se NaN-> metto dC = 0!!! -> value_C CORRETTO = "<<value_C<<'\n' <<std::endl;
                } // CONTROLLA QUAAAAA
                
                auto value_dC_ij = value_E * value_C ;
                triplets_dC_ij.push_back( Triplet<T>(counter, elem, value_dC_ij ) );
                
                term_dij_no_entropy(counter) += dij.coeff(counter,elem)*(phi_FEM(elem)-phi_FEM(counter));
                
                term_dij(counter) += value_dC_ij*(phi_FEM(elem)-phi_FEM(counter));
            
        
            }
            
            
        }
        counter++;
    }
    
    dC_ij.setFromTriplets( triplets_dC_ij.begin(), triplets_dC_ij.end() );
    triplets_dC_ij.clear();

    
    tc_case03.toc();
    std::cout << bold << yellow << "ENTROPIC and HO PROCESS, t = " << tc_case03 << " seconds" << reset << std::endl;
    
  
    
    ///********* RESOLUTION OF THE SYSTEM: **********//
   
    
    // RESOLUTION FIRST ORDER
    Matrix<T, Dynamic, 1> phi_L = phi_FEM - dt * conv_global.cwiseQuotient(global_lumped_mass)  + dt * term_dij_no_entropy.cwiseQuotient(global_lumped_mass);
  

    timecounter tc_solver5;
    tc_solver5.tic();
    // RESOLUTION HIGH ORDER -> NO MIN MAX PRINCIPLE PRESERVING
    Matrix<T, Dynamic, 1> b_phiH = mass_phi_old - dt * conv_global + dt * term_dij ;
    Matrix<T, Dynamic, 1> phi_H = solver_global_mass.solve(b_phiH);
    
    //auto prova1 = solver_prova.solve(b_phiH); // SAVE THE L2 projection
    
    tc_solver5.toc();
    std::cout << bold << yellow << "SOLUTION phi_H, t = " << tc_solver5 << " seconds" << reset << std::endl;
    
    //std::cout << "mass_phi_old =  " << mass_phi_old << " , conv_global =  "<< conv_global << " , term_dij = "<< term_dij << " , dt = "<< dt << std::endl;
    
    T relative_error0 = (global_mass*phi_H - b_phiH).norm() / b_phiH.norm();
    std::cout << "The relative error is: " << relative_error0 << std::endl;
    //std::cout << "b_phiH.norm() =  " <<b_phiH.norm() << " , (global_mass*phi_H - b_phiH).norm() =  "<< (global_mass*phi_H - b_phiH).norm() << std::endl;
    
    //T error1_prova = (global_mass*prova1 - b_phiH).norm() / b_phiH.norm();
    
    //std::cout << "The PROVA error is: " << error1_prova << std::endl;
    //std::cout << "b_phiH norm =  " << b_phiH.norm() << " , (global_mass*prova1 - b_phiH).norm() =  "<< (global_mass*prova1 - b_phiH).norm() << std::endl;
    
    /*
    auto phi_H_prova2 = solver2_bis.solve(b_phiH);
    relative_error2 = (global_mass*phi_H_prova2 - b_phiH).norm() / b_phiH.norm();
    std::cout << "The relative error is: " << relative_error2 << std::endl;
   */
    
    timecounter tc_case06;
    tc_case06.tic();
    
    // EXTENSION: MAXIMUM PRINCIPLE PRESERVING
    
    Matrix<T, Dynamic, 1> delta_phi = phi_H - phi_FEM;
   
    Matrix<T, Dynamic, 1> f_i = f_ij_creator_SPARSE( global_lumped_mass , global_mass , delta_phi , dt , dij , dC_ij , phi_FEM , S_i );
    
    // CORRECTION TERM
    Matrix<T, Dynamic, 1> correction_fi = alfaf_ij_creator_SPARSE( global_lumped_mass , global_mass , delta_phi , phi_L , dt , dij , dC_ij , phi_FEM , S_i );
    Matrix<T, Dynamic, 1>  phi_new = phi_L + correction_fi.cwiseQuotient(global_lumped_mass);
    
    
    // IMPOSITION DIRICHLET BOUNDARY CONDITIONS
    size_t counter_dir = 0 ;
    
    for ( const auto& dir_elem : fe_data.Dirichlet_boundary_inlet )
    {
        if(dir_elem){
            //phi_L(counter_dir) = phi_FEM(counter_dir) ;
            //phi_H(counter_dir) = phi_FEM(counter_dir) ;
            phi_new(counter_dir) = phi_FEM(counter_dir) ;
        }
        counter_dir++ ;
    }

    
    // SAVING AND UPLOAD phi_L  INTO CLASS projected_level_set
    phi.sol_FEM = phi_new ;
    phi.converting_into_HHO_formulation(phi_new);
    
    if(!mapping)
        phi.coefficients_inverse_mapping_MAX_MAX( );
    //phi.coefficients_inverse_mapping();
    //phi_exact.coefficients_inverse_mapping();
  
    
    tc_case06.toc();
    std::cout << bold << yellow << "EXTENSION HO, t = " << tc_case06 << " seconds" << reset << std::endl;
    
    tc.toc();
    std::cout << bold << yellow << "FEM method, time resolution: " << tc << " seconds" << reset << std::endl;
       
       
    
    /// PLOTTING SOLUTION (GNUPLOT) + SAVING FOR HHO (MISCHIATO PER POTERE PLOTTARE ENTRAMBE).
    //postprocess_output<double> postoutput5;
    //auto test_phi_new = std::make_shared< gnuplot_output_object<double> >("phi_new.dat");
    
    
    
    /*
    for(auto& cl :msh.cells)
    {
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree );
        for (auto pt : pts){
            //std::cout<<pt<<std::endl;
            //test_phi_h->add_data( pt , phi.sol_HHO(iii , counter_cl ) );
            test_phi_new->add_data( pt , phi(pt, msh , cl ) );
        }
    }
    postoutput5.add_object(test_phi_new);
    postoutput5.write();
    */
  
    
    std::cout<<yellow<<bold<<"----------- FINE TRANSPORT PROBLEM -----------"<<reset<<std::endl;
    
    

    //return phi_tilde;
    
}



template < typename Fonction, typename Mesh, typename Vel_Field , typename FiniteSpace , typename T = typename Mesh::coordinate_type >
void
run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D(const Mesh & msh, const FiniteSpace& fe_data, Fonction & phi , Vel_Field& u , T& dt )
{
    // Starting time for FE calculation
    std::cout<<yellow<<bold<<"----------- STARTING TRANSPORT PROBLEM NEW D -----------"<<reset<<std::endl;
    //std::cout<<yellow<<bold<<"PROVA--- USO MIXC LAGRANGE- BERNSTEIN"<<reset<<std::endl;
    
    timecounter tc;
    tc.tic();
    
    size_t degree = fe_data.order; // finite element order
    size_t dim = fe_data.ndof_FE ;
    //size_t n_cls = fe_data.n_cls ;
    size_t local_ndof = fe_data.local_ndof ; // local degrees of freedom
    auto S_i = fe_data.S_i;
    
    
    phi.coefficients_mapping(); // mapping of phi to have a phi between 0 and 1
   
    // SAVING PHI AND VELOCITY COEFFS
    auto phi_FEM = phi.sol_FEM ;
    auto u0 = u.sol_FEM.first ;
    auto u1 = u.sol_FEM.second ;
    auto u0_cellwise = u.sol_HHO.first ;
    auto u1_cellwise = u.sol_HHO.second ;
  
    
    // NON LINEAR ENTROPY INITIALISATION
    T eps = 1e-14 ; //constant into entropy
    non_linear_entropy_new<T,Fonction,Mesh> E(eps , phi ,msh );
    typedef non_linear_entropy_new<T,Fonction,Mesh> Entropy_func;
    entropy_flux<Entropy_func,Fonction,Mesh,Vel_Field,T> q_entropy(E , phi , u , msh );
    
    // PHI TILDE INITIALISATION --> (FOR HIGH ORDER METHOD)
    auto phi_tilde = L2_projection< T, Mesh , FiniteSpace> ( fe_data , msh );
    
    
    // SAVING OF USEFUL MATRICES
    auto global_mass = phi.Global_Mass ;
    Matrix<T, Dynamic, 1> global_lumped_mass = phi.Global_Mass_Lumped;
    
    auto global_cij_x = phi.Global_c_term_x ;
    auto global_cij_y = phi.Global_c_term_y ;
    auto local_vandermonde = phi.local_vandermonde ;
    
    auto cij_norm = phi.cij_norm ;
    auto nij0 = phi.nij0 ;
    auto nij1 = phi.nij1 ;
    
    auto cji_norm = phi.cij_norm ;
    auto nji0 = phi.nji0 ;
    auto nji1 = phi.nji1 ;
    
    
    
    // INITIALISATION OF THE SOLVER (CONJUGATE GRADIENT)
    /*
    ConjugateGradient<SparseMatrix<T> > solver_prova;
    solver_prova.compute(global_mass);
    if(solver_prova.info()!=Success){
           std::cout<<"FAILED SOLVER PROVA."<<std::endl;
           exit(1);
       }
    */
    timecounter tc_solver;
    tc_solver.tic();
    
    SimplicialLLT<SparseMatrix<T> >solver_global_mass;
    solver_global_mass.compute(global_mass); // use solver_global_mass to solve M^-1
    
    if(solver_global_mass.info()!=Success){
        std::cout<<"FAILED SOLVER LLT."<<std::endl;
        exit(1);
    }
    
    tc_solver.toc();
    std::cout << bold << yellow << "INVERSION WITH CHOLESKY METHOD, t = " << tc_solver << " seconds" << reset << std::endl;
       
    
    /*
    timecounter tc_solver1_bis;
    tc_solver1_bis.tic();
    ConjugateGradient<SparseMatrix<double>, Lower|Upper> solver2_bis;
    solver2_bis.compute(global_mass);
    if(solver2_bis.info()!=Success) {
        std::cout<<"FAILED SOLVER2 PROVA ->phi_tilde"<<std::endl;
        return;
    }
       
    tc_solver1_bis.toc();
    std::cout << bold << yellow << "INVERSION WITH ITERATIVE CG METHOD, t = " << tc_solver1_bis << " seconds" << reset << std::endl;
    
    */
    
   
    
    
    // ALTERNATIVE VANDERMONDE MATRIX
    size_t i_fl = 0 ;
    Matrix<T, Dynamic, 1> flux0_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    Matrix<T, Dynamic, 1> flux1_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    
    Matrix<T, Dynamic, 1> flux0 = Matrix<T, Dynamic, 1>::Zero(dim) ;
    Matrix<T, Dynamic, 1> flux1 = Matrix<T, Dynamic, 1>::Zero(dim) ;

    timecounter tc_solver2;
    tc_solver2.tic();
    
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
    

    tc_solver2.toc();
    std::cout << bold << yellow << "DIRECT INVERSION OF VANDERMONDE MATRIX LOCAL, t = " << tc_solver2 << " seconds" << reset << std::endl;
    
    
    
    timecounter tc_case00;
    tc_case00.tic();
    
    // RESOLUTION OF phi_tilde (GLOBALLY)    ( with cij(phi_j) )
    Matrix<T, Dynamic, 1> mass_phi_old = global_mass * phi_FEM ;
    //std::cout<<"vec1  "<<'\n'<<mass_phi_old<<std::endl;
    
    // CONVOLUTION TERM
    Matrix<T, Dynamic, 1> conv_global = global_cij_x * flux0  + global_cij_y * flux1 ;
    
    
    
    // TERM d_ij + CALCULATION OF MAX AND MIN OF THE ENTROPY
    SparseMatrix<T> dij = SparseMatrix<T>( dim , dim );
    std::vector< Triplet<T> >   triplets_dij;
    
    // TERM R_i^n
    //Matrix<T, Dynamic, 1> Emax_global = Eigen::Matrix<T, Dynamic, 1>::Ones(dim, 1);
    //Matrix<T, Dynamic, 1> Emin_global = Eigen::Matrix<T, Dynamic, 1>::Ones(dim, 1);
    //Emax_global *= -1e20;
    //Emin_global *= 1e20;
    
    // TERM R_i^n
    Matrix<T, Dynamic, 1> R_i = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    
    //Matrix<T, Dynamic, 1> R_i_prova = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    
    
    size_t counter_row = 0;
    for(auto& row_i:S_i)
    {
        T sum_row = 0.0 ;
        T N_i_entropic = 0.0;
        T D_i_entropic0 = 0.0;
        T D_i_entropic1 = 0.0;
        for(auto& elem:row_i)
        {
            
            //Emax_global(counter_row) = std::max ( E.E_values(elem) , Emax_global(counter_row) );
            //Emin_global(counter_row) = std::min ( E.E_values(elem) , Emin_global(counter_row) );
            N_i_entropic += ( (q_entropy.values0(elem) - E.E_der(counter_row)*flux0(elem) )*global_cij_x.coeff(counter_row,elem) + (q_entropy.values1(elem) - E.E_der(counter_row)*flux1(elem) )*global_cij_y.coeff(counter_row,elem) );
            D_i_entropic0 += ( q_entropy.values0(elem) * global_cij_x.coeff(counter_row,elem) + q_entropy.values1(elem) * global_cij_y.coeff(counter_row,elem) );
            D_i_entropic1 += ( flux0(elem)*global_cij_x.coeff(counter_row,elem) + flux1(elem)*global_cij_y.coeff(counter_row,elem) ) ;
            
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
            
            if( counter_row != elem )
                triplets_dij.push_back( Triplet<T>(counter_row, elem, val_dij ) );
      
            
        }
        triplets_dij.push_back( Triplet<T>(counter_row, counter_row, -sum_row ) );
        
        R_i(counter_row) = std::abs(N_i_entropic)/( std::abs(D_i_entropic0) + std::abs(D_i_entropic1)*std::abs(E.E_der(counter_row)) ) ;
        //R_i_prova(counter_row) = std::abs(N_i_entropic) ;
        
        counter_row++;
        
    }
    
    dij.setFromTriplets( triplets_dij.begin(), triplets_dij.end() );
    triplets_dij.clear();
    
    
    
    tc_case00.toc();
    std::cout << bold << yellow << "RESOLUTION OF LOW ORDER TRANSPORT, t = " << tc_case00 << " seconds" << reset << std::endl;
    
    timecounter tc_case01;
    tc_case01.tic();
    
    // CHECK TIME STEP dt
    T dt_old = dt ;
    std::cout<<bold<<yellow<<"---> COND IN TEMPO CFL, ALEXANDRE BOOK"<<reset<<std::endl;
    T CFL_numb = time_step_CFL_L2_velocity_NEW( dij.diagonal() , global_lumped_mass , fe_data.Dirichlet_boundary , dt );
    
    T nu_max0 = CFL_numb/fe_data.hx;
    T nu0 = dt_old/fe_data.hx;
    T nu1 = dt/fe_data.hx;
    
    std::cout<<"VALID FOR u = (1,0). nu_max VERO = "<<nu_max0<<" , nu max con dt assegnato = "<<nu0<< " and with dt appeared by CFL COND "<<nu1<<std::endl;
    
    if(dt_old != dt )
    {
        std::cout<<"dt is "<<dt_old<<" and dt CFL is "<<dt<<" . STOP!"<<std::endl;
        exit(10);
    }
    
    tc_case01.toc();
    std::cout << bold << yellow << "TIME CHECKING, t = " << tc_case01 << " seconds" << reset << std::endl;
    
    
  /*
    
    // CONSTANT TERM (PHI TILDE PROBLEM)
    Matrix<T, Dynamic, 1> b = mass_phi_old - dt*conv_global.cwiseQuotient(global_lumped_mass);
    //std::cout<<"TERMINE NOTO:b  "<<'\n'<<b<<std::endl;
    
    
    timecounter tc_solver3;
    tc_solver3.tic();
    
    // RESOLUTION OF PHI_TILDE
    phi_tilde.sol_FEM = solver_global_mass.solve(b); // SAVE THE L2 projection
    //auto prova0 = solver_prova.solve(b); // SAVE THE L2 projection
    // norm() is L2 norm
    T relative_error0 = (global_mass*phi_tilde.sol_FEM - b).norm() / b.norm();
    std::cout << "The relative error is: " << relative_error0 << std::endl;
    //std::cout << "b norm =  " << b.norm() << " , (global_mass*phi_tilde.sol_FEM - b).norm() =  "<< (global_mass*phi_tilde.sol_FEM - b).norm() << std::endl;
    
    //T error0_prova = (global_mass*prova0 - b).norm() / b.norm();
    
    //std::cout << "The PROVA error is: " << error0_prova << std::endl;
    //std::cout << "b norm =  " << b.norm() << " , (global_mass*prova0 - b).norm() =  "<< (global_mass*prova0 - b).norm() << std::endl;
    
    tc_solver3.toc();
    std::cout << bold << yellow << "INVERSION OF phi_tilde, t = " << tc_solver3 << " seconds" << reset << std::endl;
    
    // SAVING BOTH SOL_HHO AND VERTICES OF PHI_TILDE
    std::cout<<"CONVERTING phi_tilde"<<std::endl;
    timecounter tc_solver4;
    tc_solver4.tic();
    phi_tilde.converting_into_HHO_formulation( phi_tilde.sol_FEM );
    tc_solver4.toc();
    std::cout << bold << yellow << "CONVERTING phi_tilde, t = " << tc_solver4 << " seconds" << reset << std::endl;
    
*/
    
/*
    timecounter tc_case02;
    tc_case02.tic();
    
    // TERM R_i^n
    
    Matrix<T, Dynamic, 1> R_i_bis = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    
    for( auto& cl: msh.cells )
    {
        size_t di = 1;
        size_t offset_cell = offset(msh,cl) ;
        cell_basis_Bernstein<Mesh,T> cb(msh, cl, phi.degree_FEM);
        auto cbs = cb.size();
        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
        auto qps = integrate(msh, cl, 2*(phi.degree_FEM+di));
        //auto qps = integrate(msh, cl, (phi.degree_FEM)+di);
        //auto qps = integrate(msh, cl, di);
        
        for (auto& qp : qps)
        {
            auto bi = cb.eval_basis(qp.first);
            auto phi_grad0 = phi.gradient(qp.first,msh,cl)(0);
            auto phi_grad1 = phi.gradient(qp.first,msh,cl)(1);
            
            auto f = ( ( ( phi_tilde(qp.first,msh,cl)-phi(qp.first,msh,cl) )/dt + u(qp.first,msh,cl).first * phi_grad0 + u(qp.first,msh,cl).second * phi_grad1 ) * E.derivative(qp.first,cl) );
            ret += qp.second * bi * f;
        }
        for (size_t i = 0; i < phi.local_dim; i++)
        {
            size_t asm_map =  phi.connectivity_matrix[offset_cell][i].first ;
            R_i_bis(asm_map) += ret(i) ;
        }

    }
    
    
    // R_i FINALISATION:
    //std::cout<<"Emax_global: "<<'\n'<<Emax_global<<std::endl;
    //std::cout<<"Emin_global: "<<'\n'<<Emin_global<<std::endl;
    //R_i_bis = R_i_bis.cwiseQuotient( Emax_global - Emin_global );
    //std::cout<<"Emax_global - Emin_global = "<<'\n'<<Emax_global-Emin_global<<std::endl;
    std::cout<<"Ri - R_i_bis = "<<'\n'<<R_i_prova<<" , "<<R_i_bis<<" , DIFF -> "<<R_i_prova- R_i_bis <<std::endl;
    
    tc_case02.toc();
    std::cout << bold << yellow << "R_i PROCESS, t = " << tc_case02 << " seconds" << reset << std::endl;
*/
    timecounter tc_case03;
    tc_case03.tic();
    
    // ENTROPIC SOLUTION: MINIMUM BETWEEN d_ij AND R_i --> d^E_ij MATRIX
    T c_e = 1.0;
    T c_comp = 1.0;
    
    
    SparseMatrix<T> dC_ij =  SparseMatrix<T>( dim , dim );
    std::vector< Triplet<T> >   triplets_dC_ij;
    
    Matrix<T, Dynamic, 1> term_dij_no_entropy =  Eigen::Matrix<T,Dynamic,1>::Zero(dim, 1);
    Matrix<T, Dynamic, 1> term_dij            =  Eigen::Matrix<T,Dynamic,1>::Zero(dim, 1);
    
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            
            if(elem!=counter)
            {
                
                auto R_i_j = c_e * std::max( std::abs(R_i(counter)),std::abs(R_i(elem)) );
                //auto value_E = std::min( dij.coeff( counter , elem ) , R_i_j );
                auto value_E = dij.coeff( counter , elem ) * R_i_j ;
                //triplets_dE_ij.push_back( Triplet<T>(counter, elem, value_E ) );
                
                auto value_1 = 0.5*( phi_FEM(counter) + phi_FEM(elem) );
                auto value_2 = std::max( value_1*(1.0 -value_1) , 0.0 );
                //triplets_phi_ij.push_back( Triplet<T>(counter, elem, value_bis ) );
                //T value = 0.0;
                bool check_else = FALSE;
                T value = value_2/( std::abs(phi_FEM(counter) - phi_FEM(elem)) );
                /*
                if( (std::abs(phi_FEM(counter) - phi_FEM(elem))>1e-15) && (std::abs(value_2)>1e-15) ){
                    value = value_2/( std::abs(phi_FEM(counter) - phi_FEM(elem)) );
                }
                */
                if( (std::abs(phi_FEM(counter) - phi_FEM(elem))<1e-20) && (std::abs(value_2)<1e-20) ){
                    std::cout<<"SONO IN ELSE: std::abs(phi_FEM(counter) - phi_FEM(elem)) = "<<std::abs(phi_FEM(counter) - phi_FEM(elem))<< " and value_2 = "<<value_2<<std::endl;
                    std::cout<<"elem = "<<elem << " and counter = "<<counter<<std::endl;
                    auto value_C = std::max( 1.0 - c_comp * value , 0.0 ) ;
                    std::cout<<"1.0 - c_comp * value = "<<(1.0 - c_comp * value) << " and value_C = "<<value_C<<std::endl;
                    check_else = TRUE;
                }
                /*
                else{
                    std::cout<<"SONO IN ELSE: std::abs(phi_FEM(counter) - phi_FEM(elem)) = "<<std::abs(phi_FEM(counter) - phi_FEM(elem))<< " and value_2 = "<<value_2<<std::endl;
                    std::cout<<"elem = "<<elem << " and counter = "<<counter<<std::endl;
                    auto value_prova = value_2/( std::abs(phi_FEM(counter) - phi_FEM(elem)) ) ;
                    auto value_C = std::max( 1.0 - c_comp * value_prova , 0.0 ) ;
                    std::cout<<"1.0 - c_comp * value = "<<(1.0 - c_comp * value_prova) << " and value_C = "<<value_C<<std::endl;
                    check_else = TRUE;
                }
                */
                auto value_C = std::max( 1.0 - c_comp * value , 0.0 ) ;
                if(check_else){
                    std::cout<<"value_C GIUSTO = "<<value_C <<std::endl;
                    value = (value_2 )/( std::abs(phi_FEM(counter) - phi_FEM(elem))+ 1e-18 );
                    value_C = std::max( 1.0 - c_comp * value , 0.0 ) ;
                    //value_C = 1.0;
                    std::cout<<"Se NaN-> metto dC = 0!!! -> value_C CORRETTO = "<<value_C<<'\n' <<std::endl;
                } // CONTROLLA QUAAAAA
                
                auto value_dC_ij = value_E * value_C ;
                triplets_dC_ij.push_back( Triplet<T>(counter, elem, value_dC_ij ) );
                
                term_dij_no_entropy(counter) += dij.coeff(counter,elem)*(phi_FEM(elem)-phi_FEM(counter));
                
                term_dij(counter) += value_dC_ij*(phi_FEM(elem)-phi_FEM(counter));
            
        
            }
            
            
        }
        counter++;
    }
    
    dC_ij.setFromTriplets( triplets_dC_ij.begin(), triplets_dC_ij.end() );
    triplets_dC_ij.clear();

    
    tc_case03.toc();
    std::cout << bold << yellow << "ENTROPIC and HO PROCESS, t = " << tc_case03 << " seconds" << reset << std::endl;
    
  
    
    ///********* RESOLUTION OF THE SYSTEM: **********//
   
    
    // RESOLUTION FIRST ORDER
    Matrix<T, Dynamic, 1> phi_L = phi_FEM - dt * conv_global.cwiseQuotient(global_lumped_mass)  + dt * term_dij_no_entropy.cwiseQuotient(global_lumped_mass);
  

    timecounter tc_solver5;
    tc_solver5.tic();
    // RESOLUTION HIGH ORDER -> NO MIN MAX PRINCIPLE PRESERVING
    Matrix<T, Dynamic, 1> b_phiH = mass_phi_old - dt * conv_global + dt * term_dij ;
    Matrix<T, Dynamic, 1> phi_H = solver_global_mass.solve(b_phiH);
    
    //auto prova1 = solver_prova.solve(b_phiH); // SAVE THE L2 projection
    
    tc_solver5.toc();
    std::cout << bold << yellow << "SOLUTION phi_H, t = " << tc_solver5 << " seconds" << reset << std::endl;
    
    //std::cout << "mass_phi_old =  " << mass_phi_old << " , conv_global =  "<< conv_global << " , term_dij = "<< term_dij << " , dt = "<< dt << std::endl;
    
    T relative_error0 = (global_mass*phi_H - b_phiH).norm() / b_phiH.norm();
    std::cout << "The relative error is: " << relative_error0 << std::endl;
    //std::cout << "b_phiH.norm() =  " <<b_phiH.norm() << " , (global_mass*phi_H - b_phiH).norm() =  "<< (global_mass*phi_H - b_phiH).norm() << std::endl;
    
    //T error1_prova = (global_mass*prova1 - b_phiH).norm() / b_phiH.norm();
    
    //std::cout << "The PROVA error is: " << error1_prova << std::endl;
    //std::cout << "b_phiH norm =  " << b_phiH.norm() << " , (global_mass*prova1 - b_phiH).norm() =  "<< (global_mass*prova1 - b_phiH).norm() << std::endl;
    
    /*
    auto phi_H_prova2 = solver2_bis.solve(b_phiH);
    relative_error2 = (global_mass*phi_H_prova2 - b_phiH).norm() / b_phiH.norm();
    std::cout << "The relative error is: " << relative_error2 << std::endl;
   */
    
    timecounter tc_case06;
    tc_case06.tic();
    
    // EXTENSION: MAXIMUM PRINCIPLE PRESERVING
    
    Matrix<T, Dynamic, 1> delta_phi = phi_H - phi_FEM;
   
    Matrix<T, Dynamic, 1> f_i = f_ij_creator_SPARSE( global_lumped_mass , global_mass , delta_phi , dt , dij , dC_ij , phi_FEM , S_i );
    
    // CORRECTION TERM
    Matrix<T, Dynamic, 1> correction_fi = alfaf_ij_creator_SPARSE( global_lumped_mass , global_mass , delta_phi , phi_L , dt , dij , dC_ij , phi_FEM , S_i );
    Matrix<T, Dynamic, 1>  phi_new = phi_L + correction_fi.cwiseQuotient(global_lumped_mass);
    
    
    
    
    // IMPOSITION DIRICHLET BOUNDARY CONDITIONS
    size_t counter_dir = 0 ;
    for (const auto& dir_elem : fe_data.Dirichlet_boundary )
    {
        if(dir_elem){
            phi_L(counter_dir) = phi_FEM(counter_dir) ;
            phi_H(counter_dir) = phi_FEM(counter_dir) ;
            phi_new(counter_dir) = phi_FEM(counter_dir) ;
        }
        counter_dir++ ;
    }
     
    tc_case06.toc();
    std::cout << bold << yellow << "EXTENSION HO, t = " << tc_case06 << " seconds" << reset << std::endl;
    
    
    
   
    
    // SAVING AND UPLOAD phi_L  INTO CLASS projected_level_set
    phi.sol_FEM = phi_new ;
    phi.converting_into_HHO_formulation(phi_new);
    phi.coefficients_inverse_mapping();
    
    tc.toc();
    std::cout << bold << yellow << "FEM method, time resolution: " << tc << " seconds" << reset << std::endl;
       
       
    
    /// PLOTTING SOLUTION (GNUPLOT) + SAVING FOR HHO (MISCHIATO PER POTERE PLOTTARE ENTRAMBE).
    //postprocess_output<double> postoutput5;
    //auto test_phi_new = std::make_shared< gnuplot_output_object<double> >("phi_new.dat");
    
    
    
    /*
    for(auto& cl :msh.cells)
    {
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree );
        for (auto pt : pts){
            //std::cout<<pt<<std::endl;
            //test_phi_h->add_data( pt , phi.sol_HHO(iii , counter_cl ) );
            test_phi_new->add_data( pt , phi(pt, msh , cl ) );
        }
    }
    postoutput5.add_object(test_phi_new);
    postoutput5.write();
    */
  
    
    std::cout<<yellow<<bold<<"----------- FINE TRANSPORT PROBLEM -----------"<<reset<<std::endl;
    
    

    //return phi_tilde;
    
}



template < typename Fonction, typename Mesh, typename Vel_Field , typename FiniteSpace , typename T = typename Mesh::coordinate_type >
void
run_FEM_BERNSTEIN_CORRECT_FAST(const Mesh & msh, const FiniteSpace& fe_data, Fonction & phi , Vel_Field& u , T& dt )
{
    // Starting time for FE calculation
    std::cout<<yellow<<bold<<"----------- STARTING TRANSPORT PROBLEM -----------"<<reset<<std::endl;
    //std::cout<<yellow<<bold<<"PROVA--- USO MIXC LAGRANGE- BERNSTEIN"<<reset<<std::endl;
    
    timecounter tc;
    tc.tic();
    
    size_t degree = fe_data.order; // finite element order
    size_t dim = fe_data.ndof_FE ;
    //size_t n_cls = fe_data.n_cls ;
    size_t local_ndof = fe_data.local_ndof ; // local degrees of freedom
    auto S_i = fe_data.S_i;
    
    
    phi.coefficients_mapping(); // mapping of phi to have a phi between 0 and 1
   
    // SAVING PHI AND VELOCITY COEFFS
    auto phi_FEM = phi.sol_FEM ;
    auto u0 = u.sol_FEM.first ;
    auto u1 = u.sol_FEM.second ;
    auto u0_cellwise = u.sol_HHO.first ;
    auto u1_cellwise = u.sol_HHO.second ;
  
    
    // NON LINEAR ENTROPY INITIALISATION
    const T eps = 1e-14 ; //constant into entropy
    non_linear_entropy_new<T,Fonction,Mesh> E(eps , phi ,msh );
   
    
    // PHI TILDE INITIALISATION --> (FOR HIGH ORDER METHOD)
    auto phi_tilde = L2_projection< T, Mesh , FiniteSpace> ( fe_data , msh );
    
    
    // SAVING OF USEFUL MATRICES
    auto global_mass = phi.Global_Mass ;
    Matrix<T, Dynamic, 1> global_lumped_mass = phi.Global_Mass_Lumped;
    
    auto global_cij_x = phi.Global_c_term_x ;
    auto global_cij_y = phi.Global_c_term_y ;
    auto local_vandermonde = phi.local_vandermonde ;
    
    auto cij_norm = phi.cij_norm ;
    auto nij0 = phi.nij0 ;
    auto nij1 = phi.nij1 ;
    
    auto cji_norm = phi.cij_norm ;
    auto nji0 = phi.nji0 ;
    auto nji1 = phi.nji1 ;
    
    
    
    // INITIALISATION OF THE SOLVER (CONJUGATE GRADIENT)
    //ConjugateGradient<SparseMatrix<T> > solver_global_mass;
    
    timecounter tc_solver;
    tc_solver.tic();
    
    SimplicialLLT<SparseMatrix<T> >solver_global_mass;
    solver_global_mass.compute(global_mass); // use solver_global_mass to solve M^-1
    
    if(solver_global_mass.info()!=Success){
        std::cout<<"FAILED SOLVER 0->phi_tilde"<<std::endl;
        exit(1);
    }
    
    tc_solver.toc();
    std::cout << bold << yellow << "INVERSION WITH CHOLESKY METHOD, t = " << tc_solver << " seconds" << reset << std::endl;
       
    
    /*
    timecounter tc_solver1_bis;
    tc_solver1_bis.tic();
    ConjugateGradient<SparseMatrix<double>, Lower|Upper> solver2_bis;
    solver2_bis.compute(global_mass);
    if(solver2_bis.info()!=Success) {
        std::cout<<"FAILED SOLVER2 PROVA ->phi_tilde"<<std::endl;
        return;
    }
       
    tc_solver1_bis.toc();
    std::cout << bold << yellow << "INVERSION WITH ITERATIVE CG METHOD, t = " << tc_solver1_bis << " seconds" << reset << std::endl;
    
    */
    
   
    
    
    // ALTERNATIVE VANDERMONDE MATRIX
    size_t i_fl = 0 ;
    Matrix<T, Dynamic, 1> flux0_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    Matrix<T, Dynamic, 1> flux1_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    
    Matrix<T, Dynamic, 1> flux0 = Matrix<T, Dynamic, 1>::Zero(dim) ;
    Matrix<T, Dynamic, 1> flux1 = Matrix<T, Dynamic, 1>::Zero(dim) ;

    timecounter tc_solver2;
    tc_solver2.tic();
    
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
    

    tc_solver2.toc();
    std::cout  << "DIRECT INVERSION OF VANDERMONDE MATRIX LOCAL, t = " << tc_solver2 << " seconds"  << std::endl;
    
    
    
    timecounter tc_case00;
    tc_case00.tic();
    
    // RESOLUTION OF phi_tilde (GLOBALLY)    ( with cij(phi_j) )
    Matrix<T, Dynamic, 1> mass_phi_old = global_mass * phi_FEM ;
    //std::cout<<"vec1  "<<'\n'<<mass_phi_old<<std::endl;
    
    // CONVOLUTION TERM
    Matrix<T, Dynamic, 1> conv_global = global_cij_x * flux0  + global_cij_y * flux1 ;
    
    
    
    // TERM d_ij + CALCULATION OF MAX AND MIN OF THE ENTROPY
    SparseMatrix<T> dij = SparseMatrix<T>( dim , dim );
    std::vector< Triplet<T> >   triplets_dij;
    
    // TERM R_i^n
    Matrix<T, Dynamic, 1> Emax_global = Eigen::Matrix<T, Dynamic, 1>::Ones(dim, 1);
    Matrix<T, Dynamic, 1> Emin_global = Eigen::Matrix<T, Dynamic, 1>::Ones(dim, 1);
    Emax_global *= -1e20;
    Emin_global *= 1e20;
    
    
    
    
    size_t counter_row = 0;
    for(auto& row_i:S_i)
    {
        T sum_row = 0.0 ;
        for(auto& elem:row_i)
        {
            
            Emax_global(counter_row) = std::max ( E.E_values(elem) , Emax_global(counter_row) );
            Emin_global(counter_row) = std::min ( E.E_values(elem) , Emin_global(counter_row) );
            
            
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
            
            if( counter_row != elem )
                triplets_dij.push_back( Triplet<T>(counter_row, elem, val_dij ) );
      
            
        }
        triplets_dij.push_back( Triplet<T>(counter_row, counter_row, -sum_row ) );
        counter_row++;
        
    }
    
    dij.setFromTriplets( triplets_dij.begin(), triplets_dij.end() );
    triplets_dij.clear();
    
    
    
    tc_case00.toc();
    std::cout << "RESOLUTION OF LOW ORDER TRANSPORT, t = " << tc_case00 << " seconds" << std::endl;
    
    timecounter tc_case01;
    tc_case01.tic();
    
    // CHECK TIME STEP dt
    T dt_old = dt ;
    std::cout<<"---> COND IN TEMPO CFL, ALEXANDRE BOOK"<<std::endl;
    T CFL_numb = time_step_CFL_L2_velocity_NEW( dij.diagonal() , global_lumped_mass , fe_data.Dirichlet_boundary , dt );
    
    //T nu_max0 = CFL_numb/fe_data.hx;
    //T nu0 = dt_old/fe_data.hx;
    //T nu1 = dt/fe_data.hx;
    
    //std::cout<<"VALID FOR u = (1,0). nu_max VERO = "<<nu_max0<<" , nu max con dt assegnato = "<<nu0<< " and with dt appeared by CFL COND "<<nu1<<std::endl;
    
    if(dt_old != dt )
    {
        std::cout<<"dt is "<<dt_old<<" and dt CFL is "<<dt<<" . STOP!"<<std::endl;
        exit(10);
    }
    
    tc_case01.toc();
    std::cout << "TIME CHECKING, t = " << tc_case01 << " seconds" << std::endl;
    
    
    
    
    // CONSTANT TERM (PHI TILDE PROBLEM)
    Matrix<T, Dynamic, 1> b = mass_phi_old - dt*conv_global.cwiseQuotient(global_lumped_mass);
    //std::cout<<"TERMINE NOTO:b  "<<'\n'<<b<<std::endl;
    
    
    timecounter tc_solver3;
    tc_solver3.tic();
    
    // RESOLUTION OF PHI_TILDE
    phi_tilde.sol_FEM = solver_global_mass.solve(b); // SAVE THE L2 projection
    
    // norm() is L2 norm
    T relative_error0 = (global_mass*phi_tilde.sol_FEM - b).norm() / b.norm();
    std::cout << "The relative error is: " << relative_error0 << std::endl;
    
    /*
    auto prova_phi_tilde_2 = solver2_bis.solve(b);
    T relative_error2 = (global_mass*prova_phi_tilde_2 - b).norm() / b.norm();
    std::cout << "The relative error is: " << relative_error2 << std::endl;
    
    
    if(solver_global_mass.info()!=Success) {
        std::cout<<"FAILED SOLVER 1->phi_tilde"<<std::endl;
        exit(1);
    }
    */
    tc_solver3.toc();
    std::cout << bold << yellow << "INVERSION OF phi_tilde, t = " << tc_solver3 << " seconds" << reset << std::endl;
    
    // SAVING BOTH SOL_HHO AND VERTICES OF PHI_TILDE
    std::cout<<"CONVERTING phi_tilde"<<std::endl;
    timecounter tc_solver4;
    tc_solver4.tic();
    phi_tilde.converting_into_HHO_formulation( phi_tilde.sol_FEM );
    tc_solver4.toc();
    std::cout << bold << yellow << "CONVERTING phi_tilde, t = " << tc_solver4 << " seconds" << reset << std::endl;
    
    
    
    
    timecounter tc_case02;
    tc_case02.tic();
    
    // TERM R_i^n
    Matrix<T, Dynamic, 1> R_i = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    
    for( auto& cl: msh.cells )
    {
        size_t di = 1;
        size_t offset_cell = offset(msh,cl) ;
        cell_basis_Bernstein<Mesh,T> cb(msh, cl, phi.degree_FEM);
        auto cbs = cb.size();
        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
        //auto qps = integrate(msh, cl, 2*(phi.degree_FEM+di));
        //auto qps = integrate(msh, cl, (phi.degree_FEM)+di);
        auto qps = integrate(msh, cl, di);
        
        for (auto& qp : qps)
        {
            auto bi = cb.eval_basis(qp.first);
            auto phi_grad0 = phi.gradient(qp.first,msh,cl)(0);
            auto phi_grad1 = phi.gradient(qp.first,msh,cl)(1);
            
            auto f = ( ( ( phi_tilde(qp.first,msh,cl)-phi(qp.first,msh,cl) )/dt + u(qp.first,msh,cl).first * phi_grad0 + u(qp.first,msh,cl).second * phi_grad1 ) * E.derivative(qp.first,cl) );
            ret += qp.second * bi * f;
        }
        for (size_t i = 0; i < phi.local_dim; i++)
        {
            size_t asm_map =  phi.connectivity_matrix[offset_cell][i].first ;
            R_i(asm_map) += ret(i) ;
        }

    }
    
    
    // R_i FINALISATION:
    //std::cout<<"Emax_global: "<<'\n'<<Emax_global<<std::endl;
    //std::cout<<"Emin_global: "<<'\n'<<Emin_global<<std::endl;
    //R_i = R_i.cwiseQuotient( Emax_global - Emin_global );
    //std::cout<<"Emax_global - Emin_global = "<<'\n'<<Emax_global-Emin_global<<std::endl;
    //std::cout<<"Ri = "<<'\n'<<R_i<<std::endl;
    
    tc_case02.toc();
    std::cout << bold << yellow << "R_i PROCESS, t = " << tc_case02 << " seconds" << reset << std::endl;
    
    timecounter tc_case03;
    tc_case03.tic();
    
    // ENTROPIC SOLUTION: MINIMUM BETWEEN d_ij AND R_i --> d^E_ij MATRIX
    T c_e = 1.0;
    T c_comp = 1.0;
    
    
    SparseMatrix<T> dC_ij =  SparseMatrix<T>( dim , dim );
    std::vector< Triplet<T> >   triplets_dC_ij;
    
    Matrix<T, Dynamic, 1> term_dij_no_entropy =  Eigen::Matrix<T,Dynamic,1>::Zero(dim, 1);
    Matrix<T, Dynamic, 1> term_dij            =  Eigen::Matrix<T,Dynamic,1>::Zero(dim, 1);
    
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            
            if(elem!=counter)
            {
                
                auto R_i_j = c_e * std::max( std::abs(R_i(counter)),std::abs(R_i(elem)) );
                auto value_E = std::min( dij.coeff( counter , elem ) , R_i_j );
                //triplets_dE_ij.push_back( Triplet<T>(counter, elem, value_E ) );
                
                auto value_1 = 0.5*( phi_FEM(counter) + phi_FEM(elem) );
                auto value_2 = std::max( value_1*(1.0 -value_1) , 0.0 );
                
                bool check_else = FALSE;
                //triplets_phi_ij.push_back( Triplet<T>(counter, elem, value_bis ) );
                //if( std::abs(phi_FEM(counter_tmp) - phi_FEM(elem))>1e-15 ){
                auto value = value_2/( std::abs(phi_FEM(counter) - phi_FEM(elem)) );
                
                if( (std::abs(phi_FEM(counter) - phi_FEM(elem))<1e-20) && (std::abs(value_2)<1e-20) ){
                    std::cout<<"SONO IN ELSE: std::abs(phi_FEM(counter) - phi_FEM(elem)) = "<<std::abs(phi_FEM(counter) - phi_FEM(elem))<< " and value_2 = "<<value_2<<std::endl;
                    std::cout<<"elem = "<<elem << " and counter = "<<counter<<std::endl;
    
                    auto value_C = std::max( 1.0 - c_comp * value , 0.0 ) ;
                    std::cout<<"1.0 - c_comp * value = "<<(1.0 - c_comp * value) << " and value_C = "<<value_C<<std::endl;
                    check_else = TRUE;
                }
                
                
                
                auto value_C = std::max( 1.0 - c_comp * value , 0.0 ) ;
                if(check_else){
                    std::cout<<"value_C GIUSTO = "<<value_C <<std::endl;
                    value = (value_2+ 1e-18)/( std::abs(phi_FEM(counter) - phi_FEM(elem))  );
                    value_C = std::max( 1.0 - c_comp * value , 0.0 ) ;
                    //value_C = 1.0;
                    std::cout<<"Se NaN-> metto dC = 0!!! -> value_C CORRETTO = "<<value_C <<std::endl;
                }
                
                auto value_dC_ij = value_E * value_C ;
                triplets_dC_ij.push_back( Triplet<T>(counter, elem, value_dC_ij ) );
                
                term_dij_no_entropy(counter) += dij.coeff(counter,elem)*(phi_FEM(elem)-phi_FEM(counter));
                
                term_dij(counter) += value_dC_ij*(phi_FEM(elem)-phi_FEM(counter));
            
        
            }
            
            
        }
        counter++;
    }
    
    dC_ij.setFromTriplets( triplets_dC_ij.begin(), triplets_dC_ij.end() );
    triplets_dC_ij.clear();

    
    tc_case03.toc();
    std::cout << bold << yellow << "ENTROPIC and HO PROCESS, t = " << tc_case03 << " seconds" << reset << std::endl;
    
  
    
    ///********* RESOLUTION OF THE SYSTEM: **********//
   
    
    // RESOLUTION FIRST ORDER
    Matrix<T, Dynamic, 1> phi_L = phi_FEM - dt * conv_global.cwiseQuotient(global_lumped_mass)  + dt * term_dij_no_entropy.cwiseQuotient(global_lumped_mass);
  

    timecounter tc_solver5;
    tc_solver5.tic();
    // RESOLUTION HIGH ORDER -> NO MIN MAX PRINCIPLE PRESERVING
    Matrix<T, Dynamic, 1> b_phiH = mass_phi_old - dt * conv_global + dt * term_dij ;
    Matrix<T, Dynamic, 1> phi_H = solver_global_mass.solve(b_phiH);
    
    tc_solver5.toc();
    std::cout << bold << yellow << "SOLUTION phi_H, t = " << tc_solver5 << " seconds" << reset << std::endl;
    
    relative_error0 = (global_mass*phi_H - b_phiH).norm() / b_phiH.norm();
    std::cout << "The relative error is: " << relative_error0 << std::endl;
    /*
    auto phi_H_prova2 = solver2_bis.solve(b_phiH);
    relative_error2 = (global_mass*phi_H_prova2 - b_phiH).norm() / b_phiH.norm();
    std::cout << "The relative error is: " << relative_error2 << std::endl;
   */
    
    timecounter tc_case06;
    tc_case06.tic();
    
    // EXTENSION: MAXIMUM PRINCIPLE PRESERVING
    
    Matrix<T, Dynamic, 1> delta_phi = phi_H - phi_FEM;
   
    Matrix<T, Dynamic, 1> f_i = f_ij_creator_SPARSE( global_lumped_mass , global_mass , delta_phi , dt , dij , dC_ij , phi_FEM , S_i );
    
    // CORRECTION TERM
    Matrix<T, Dynamic, 1> correction_fi = alfaf_ij_creator_SPARSE( global_lumped_mass , global_mass , delta_phi , phi_L , dt , dij , dC_ij , phi_FEM , S_i );
    Matrix<T, Dynamic, 1>  phi_new = phi_L + correction_fi.cwiseQuotient(global_lumped_mass);
    
    
    
    
    // IMPOSITION DIRICHLET BOUNDARY CONDITIONS
    size_t counter_dir = 0 ;
    for (const auto& dir_elem : fe_data.Dirichlet_boundary )
    {
        if(dir_elem){
            phi_L(counter_dir) = phi_FEM(counter_dir) ;
            phi_H(counter_dir) = phi_FEM(counter_dir) ;
            phi_new(counter_dir) = phi_FEM(counter_dir) ;
        }
        counter_dir++ ;
    }
     
    tc_case06.toc();
    std::cout << bold << yellow << "EXTENSION HO, t = " << tc_case06 << " seconds" << reset << std::endl;
    
    
    
   
    
    // SAVING AND UPLOAD phi_L  INTO CLASS projected_level_set
    phi.sol_FEM = phi_new ;
    phi.converting_into_HHO_formulation(phi_new);
    phi.coefficients_inverse_mapping();
    
    tc.toc();
    std::cout << bold << yellow << "FEM method, time resolution: " << tc << " seconds" << reset << std::endl;
       
       
    
    /// PLOTTING SOLUTION (GNUPLOT) + SAVING FOR HHO (MISCHIATO PER POTERE PLOTTARE ENTRAMBE).
    //postprocess_output<double> postoutput5;
    //auto test_phi_new = std::make_shared< gnuplot_output_object<double> >("phi_new.dat");
    
    
    
    /*
    for(auto& cl :msh.cells)
    {
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree );
        for (auto pt : pts){
            //std::cout<<pt<<std::endl;
            //test_phi_h->add_data( pt , phi.sol_HHO(iii , counter_cl ) );
            test_phi_new->add_data( pt , phi(pt, msh , cl ) );
        }
    }
    postoutput5.add_object(test_phi_new);
    postoutput5.write();
    */
  
    
    std::cout<<yellow<<bold<<"----------- FINE TRANSPORT PROBLEM -----------"<<reset<<std::endl;
    
    

    //return phi_tilde;
    
}

template < typename Fonction, typename Mesh, typename Vel_Field , typename FiniteSpace , typename T = typename Mesh::coordinate_type >
void
run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST(const Mesh & msh, const FiniteSpace& fe_data, Fonction & phi , Vel_Field& u , T& dt )
{
    // Starting time for FE calculation
    std::cout<<yellow<<bold<<"----------- STARTING TRANSPORT PROBLEM LOW ORDER FAST -----------"<<reset<<std::endl;
    //std::cout<<yellow<<bold<<"PROVA--- USO MIXC LAGRANGE- BERNSTEIN"<<reset<<std::endl;
    
    timecounter tc;
    tc.tic();
    
    size_t degree = fe_data.order; // finite element order
    size_t dim = fe_data.ndof_FE ;
    //size_t n_cls = fe_data.n_cls ;
    size_t local_ndof = fe_data.local_ndof ; // local degrees of freedom
    auto S_i = fe_data.S_i;
    
   
    // SAVING PHI AND VELOCITY COEFFS
    auto phi_FEM = phi.sol_FEM ;
    auto u0 = u.sol_FEM.first ;
    auto u1 = u.sol_FEM.second ;
    auto u0_cellwise = u.sol_HHO.first ;
    auto u1_cellwise = u.sol_HHO.second ;
  
    
  
    // SAVING OF USEFUL MATRICES
    Matrix<T, Dynamic, 1> global_lumped_mass = phi.Global_Mass_Lumped;
    
    auto global_cij_x = phi.Global_c_term_x ;
    auto global_cij_y = phi.Global_c_term_y ;
    auto local_vandermonde = phi.local_vandermonde ;
    
    auto cij_norm = phi.cij_norm ;
    auto nij0 = phi.nij0 ;
    auto nij1 = phi.nij1 ;
    
    auto cji_norm = phi.cij_norm ;
    auto nji0 = phi.nji0 ;
    auto nji1 = phi.nji1 ;
    
    
    
    
    
    // VANDERMONDE MATRIX INTERPOLATION
    size_t i_fl = 0 ;
    Matrix<T, Dynamic, 1> flux0_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    Matrix<T, Dynamic, 1> flux1_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    
    Matrix<T, Dynamic, 1> flux0 = Matrix<T, Dynamic, 1>::Zero(dim) ;
    Matrix<T, Dynamic, 1> flux1 = Matrix<T, Dynamic, 1>::Zero(dim) ;

    timecounter tc_solver2;
    tc_solver2.tic();
    
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
    

    tc_solver2.toc();
    std::cout  << "DIRECT INVERSION OF VANDERMONDE MATRIX LOCAL, t = " << tc_solver2 << " seconds"  << std::endl;
    
    
    
    timecounter tc_case00;
    tc_case00.tic();
    
    
    
    // CONVOLUTION TERM
    Matrix<T, Dynamic, 1> conv_global = global_cij_x * flux0  + global_cij_y * flux1 ;
    
    
    
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
    
    
    
    tc_case00.toc();
    std::cout  << "RESOLUTION OF LOW ORDER TRANSPORT, t = " << tc_case00 << " seconds" << std::endl;
    
    timecounter tc_case01;
    tc_case01.tic();
    
    // CHECK TIME STEP dt
    T dt_old = dt ;
    std::cout<<"---> COND IN TEMPO CFL, ALEXANDRE BOOK"<<std::endl;
    T CFL_numb = time_step_CFL_L2_velocity_NEW( dij.diagonal() , global_lumped_mass , fe_data.Dirichlet_boundary , dt );
    
    //T nu_max0 = CFL_numb/fe_data.hx;
    //T nu0 = dt_old/fe_data.hx;
    //T nu1 = dt/fe_data.hx;
    
    //std::cout<<"VALID FOR u = (1,0). nu_max VERO = "<<nu_max0<<" , nu max con dt assegnato = "<<nu0<< " and with dt appeared by CFL COND "<<nu1<<std::endl;
    
    if(dt_old != dt )
    {
        dt = floorf(CFL_numb*0.98 * 1000) / 1000;
        std::cout<<yellow<<bold<<"ATTENTION CHANGIN dt!!! ----> "<<reset<<"dt is "<<dt_old<<" and dt CFL is "<<dt<<" !"<<std::endl;
        //exit(10);
        
    }
    
    tc_case01.toc();
    std::cout << "TIME CHECKING, t = " << tc_case01 << " seconds" << std::endl;
    
 
  
    
    ///********* RESOLUTION OF THE SYSTEM: **********//
   
    
    // RESOLUTION FIRST ORDER
    Matrix<T, Dynamic, 1> phi_L = phi_FEM - dt * conv_global.cwiseQuotient(global_lumped_mass)  + dt * term_dij_no_entropy.cwiseQuotient(global_lumped_mass);
  
    
    // IMPOSITION DIRICHLET BOUNDARY CONDITIONS
    size_t counter_dir = 0 ;
    for (const auto& dir_elem : fe_data.Dirichlet_boundary )
    {
        if(dir_elem){
            phi_L(counter_dir) = phi_FEM(counter_dir) ;
        }
        counter_dir++ ;
    }
     
    
    
    
    
   
    
    // SAVING AND UPLOAD phi_L  INTO CLASS projected_level_set
    phi.sol_FEM = phi_L ;
    phi.converting_into_HHO_formulation(phi_L);
   
    
    
    tc.toc();
    std::cout << bold << yellow << "FEM method, time resolution: " << tc << " seconds" << reset << std::endl;
       
       
    
    /// PLOTTING SOLUTION (GNUPLOT) + SAVING FOR HHO (MISCHIATO PER POTERE PLOTTARE ENTRAMBE).
    //postprocess_output<double> postoutput5;
    //auto test_phi_L = std::make_shared< gnuplot_output_object<double> >("phi_L.dat");
    
    
    
    /*
    for(auto& cl :msh.cells)
    {
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree );
        for (auto pt : pts){
            test_phi_L->add_data( pt , phi(pt, msh , cl ) );
        }
    }
    postoutput5.add_object(test_phi_L);
    postoutput5.write();
    */
  
    
    std::cout<<yellow<<bold<<"----------- FINE TRANSPORT PROBLEM -----------"<<reset<<std::endl;
    
    

    //return phi_tilde;
    
}




template < typename Fonction, typename Mesh, typename Vel_Field , typename FiniteSpace , typename T = typename Mesh::coordinate_type >
void
run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST_to_check_C_NEW(const Mesh & msh, const FiniteSpace& fe_data, Fonction & phi , Vel_Field& u , T& dt , Matrix<T, Dynamic, 1>& phi2 , Matrix<T, Dynamic, 1>& phi3 , int i_prova , Matrix<T, Dynamic, 1>& vecprova0, Matrix<T, Dynamic, 1>& vecprova1, Matrix<T, Dynamic, 1>& vecprova2 , Matrix<T, Dynamic, 1>& vecprovabis0 , Matrix<T, Dynamic, 1>& vecprovabis1 ,Matrix<T, Dynamic, Dynamic>& vecprovabis2 , Matrix<T, Dynamic, Dynamic>& vecprovabis3 , Matrix<T, Dynamic, Dynamic>& vecprovabis4 ,Matrix<T, Dynamic, Dynamic>& vecprovabis5 , Matrix<T, Dynamic, 1>& vecprovabis6 , Matrix<T, Dynamic, 1>& vecprovabis7 )
{
    // Starting time for FE calculation
    std::cout<<yellow<<bold<<"----------- CHECKING ---> TRANSPORT PROBLEM LOW ORDER FAST -----------"<<reset<<std::endl;
    //std::cout<<yellow<<bold<<"PROVA--- USO MIXC LAGRANGE- BERNSTEIN"<<reset<<std::endl;
    
    timecounter tc;
    tc.tic();
    
    size_t degree = fe_data.order; // finite element order
    size_t dim = fe_data.ndof_FE ;
    //size_t n_cls = fe_data.n_cls ;
    size_t local_ndof = fe_data.local_ndof ; // local degrees of freedom
    auto S_i = fe_data.S_i;
    
   
    // SAVING PHI AND VELOCITY COEFFS
    auto phi_FEM = phi.sol_FEM ;
    auto u0 = u.sol_FEM.first ;
    auto u1 = u.sol_FEM.second ;
    auto u0_cellwise = u.sol_HHO.first ;
    auto u1_cellwise = u.sol_HHO.second ;
  
    std::cout<<"IMPORTANTEEEEE --------------------------_> i = "<<i_prova <<std::endl;
  
    // SAVING OF USEFUL MATRICES
    Matrix<T, Dynamic, 1> global_lumped_mass = phi.Global_Mass_Lumped;
    
    //auto global_cij_x = phi.Global_c_term_x ;
    //auto global_cij_y = phi.Global_c_term_y ;
    auto local_vandermonde = phi.local_vandermonde ;
    
    
    //std::cout<<"local_vandermonde"<<'\n'<<local_vandermonde<<std::endl;
    SparseMatrix<T>                 global_cij_x; // Global mass, saved for FEM problem
    SparseMatrix<T>                 global_cij_y;
    global_cij_x = SparseMatrix<T>( dim, dim ); //(b_i,b_j)_ij , b_i Lagrange basis fx
    global_cij_y = SparseMatrix<T>( dim, dim ); //(b_i,b_j)_ij , b_i Lagrange basis fx
    std::vector< Triplet<T> >       triplets_c_term_x_l; // Position elements: Sparse Matrix Notation
    std::vector< Triplet<T> >       triplets_c_term_y_l; // Position elements: Sparse Matrix Notation
    for( const auto& cl : msh.cells )
    {
        size_t cell_offset = offset(msh, cl) ;
        //auto local_cij_lagrangian = make_lagrangian_local_cij_matrix (msh, cl, degree);
        auto local_cij_lagrangian = make_bernstein_local_cij_matrix_with_velocity (msh, cl, degree , u );
        auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree);
        //std::cout<< "local_cij_lagrangian.first "<<'\n'<<local_cij_lagrangian.first<<std::endl;
    
        
        // Assembling triplets for global problem
        for (size_t i = 0; i < local_ndof; i++)
        {
            size_t asm_map_i = phi.connectivity_matrix[cell_offset][i].first ;
            for (size_t j = 0; j < local_ndof; j++)
            {
                size_t asm_map_j = phi.connectivity_matrix[cell_offset][j].first ;
                triplets_c_term_x_l.push_back( Triplet<T>(asm_map_i, asm_map_j , local_cij_lagrangian.first(i,j) ) );
                triplets_c_term_y_l.push_back( Triplet<T>( asm_map_i , asm_map_j , local_cij_lagrangian.second(i,j) ) );
            
            }
        }

    }

    global_cij_x.setFromTriplets( triplets_c_term_x_l.begin(), triplets_c_term_x_l.end() );
    triplets_c_term_x_l.clear();
    
    global_cij_y.setFromTriplets( triplets_c_term_y_l.begin(), triplets_c_term_y_l.end() );
    triplets_c_term_y_l.clear();
    /*
    auto cij_norm = phi.cij_norm ;
    auto nij0 = phi.nij0 ;
    auto nij1 = phi.nij1 ;
    
    auto cji_norm = phi.cij_norm ;
    auto nji0 = phi.nji0 ;
    auto nji1 = phi.nji1 ;
    */
    SparseMatrix<T> cij_norm = ( global_cij_x.cwiseProduct(global_cij_x) + global_cij_y.cwiseProduct(global_cij_y) ).cwiseSqrt() ;
    //std::cout<<"cij norm "<<'\n'<<cij_norm<<std::endl;
    
    // MATRIX n_ij
    SparseMatrix<T> nij0 = global_cij_x.cwiseQuotient( cij_norm );
    SparseMatrix<T> nij1 = global_cij_y.cwiseQuotient( cij_norm );
    
    //std::cout<<"nij1  "<<'\n'<<nij1<<std::endl;
    

    // MATRIX c_ji
    SparseMatrix<T> cji_x = global_cij_x.adjoint() ;
    SparseMatrix<T> cji_y = global_cij_y.adjoint() ;
     
    // NORM of c_ji -> i.e. c_ij transposed
    SparseMatrix<T> cji_norm = (cji_x.cwiseProduct(cji_x)+cji_y.cwiseProduct(cji_y)).cwiseSqrt();
    
    // MATRIX n_ij (TRANSPOSED)
    SparseMatrix<T> nji0 = cji_x.cwiseQuotient( cji_norm );
    SparseMatrix<T> nji1 = cji_y.cwiseQuotient( cji_norm );
    
    
    
    Matrix<T, Dynamic, 1> sum0 = Eigen::Matrix<T, Dynamic, 1>::Zero(global_cij_x.rows(), 1);
       
    Matrix<T, Dynamic, 1> sum1 = Eigen::Matrix<T, Dynamic, 1>::Zero(global_cij_y.rows(), 1);
       
    for (size_t k = 0; k<global_cij_x.cols(); k++)
    {
        sum0 += (global_cij_x).col(k);
        sum1 += (global_cij_y).col(k);
    }
       
    for (size_t i=0; i<global_cij_x.rows(); i++) {
        std::cout<<"The sum is "<<sum1(i)<<" and "<<sum0(i)<<std::endl;
    }
       
       
       
    for(size_t counter_sum = 0 ; counter_sum < global_cij_x.rows() ; counter_sum++ )
    {
        for(size_t counter_col = 0 ; counter_col < global_cij_x.cols() ; counter_col++)
        {
       
            if(counter_col==counter_sum)
                std::cout<<"In ("<<counter_sum<<" , "<<counter_col<<" ), c^0 = "<<global_cij_x.coeff( counter_sum , counter_col )<<" and c1 = "<<global_cij_y.coeff( counter_sum , counter_col )<<std::endl;
            else
                std::cout<<"In ("<<counter_sum<<" , "<<counter_col<<" ), c^0_ij + c^0_ji = "<<global_cij_x.coeff( counter_sum , counter_col ) + global_cij_x.coeff( counter_col , counter_sum )<<" and c^1_ij + c^1_ji = "<<global_cij_y.coeff( counter_sum , counter_col ) + global_cij_y.coeff( counter_col , counter_sum ) <<std::endl;
        }
    }
    
    
    // VANDERMONDE MATRIX INTERPOLATION
    //size_t i_fl = 0 ;
    Matrix<T, Dynamic, 1> flux0_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    Matrix<T, Dynamic, 1> flux1_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    
    Matrix<T, Dynamic, 1> flux0 = Matrix<T, Dynamic, 1>::Zero(dim) ;
    Matrix<T, Dynamic, 1> flux1 = Matrix<T, Dynamic, 1>::Zero(dim) ;
    
    Matrix<T, Dynamic, 1> flux0_prova = Matrix<T, Dynamic, 1>::Zero(dim) ;
    Matrix<T, Dynamic, 1> flux1_prova = Matrix<T, Dynamic, 1>::Zero(dim) ;
    Matrix<T, Dynamic, 1> flux0_prova_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    Matrix<T, Dynamic, 1> flux1_prova_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    Matrix<T, Dynamic, 1> vel2_loc0 = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    Matrix<T, Dynamic, 1> sol_vel_gl0 = Matrix<T, Dynamic, 1>::Zero(dim) ;
    Matrix<T, Dynamic, 1> vel2_loc1 = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    Matrix<T, Dynamic, 1> sol_vel_gl1 = Matrix<T, Dynamic, 1>::Zero(dim) ;
    
    /*
    Matrix<T, Dynamic, Dynamic> sol_vel_gl_cellwise = Matrix<T, Dynamic, Dynamic>::Zero(local_ndof,msh.cells.size()) ;
    Matrix<T, Dynamic, Dynamic> one_loc = Matrix<T, Dynamic, Dynamic>::Zero(local_ndof,local_ndof) ;
    
    timecounter tc_solver2;
    tc_solver2.tic();
    
    CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > cod(local_vandermonde);
    ColPivHouseholderQR<Matrix<T, Dynamic, Dynamic > > cod_bis(local_vandermonde);

    
    for(auto& cl : msh.cells)
    {
        // FLUX TERM : flux is a pair flux0 and flux1
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree );
        for (size_t i = 0; i < local_ndof ; i++)
        {
            flux0_loc(i) = u0_cellwise(i,i_fl) * phi(pts[i] , msh , cl );
            flux1_loc(i) = u1_cellwise(i,i_fl) * phi(pts[i] , msh , cl );
            //std::cout<<"pts[i] = "<<pts[i] << " HHO_u0 = "<<u0_cellwise(i,i_fl)<<std::endl;
            flux0_prova_loc(i) = u0_cellwise(i,i_fl) * (phi(pts[i] , msh , cl ) + 0.5);
            flux1_prova_loc(i) = u1_cellwise(i,i_fl) * (phi(pts[i] , msh , cl ) + 0.5);
            vel2_loc0(i) = u0_cellwise(i,i_fl) ;
            vel2_loc1(i) = u1_cellwise(i,i_fl) ;
            one_loc(i,i) = 1.0;
            if(i_prova == 0){
                //std::cout<<"phi(" << pts[i] <<", msh , cl ) = "<<phi(pts[i] , msh , cl )<<std::endl;
                vecprovabis4(i,offset(msh,cl)) = ( phi(pts[i] , msh , cl ) ) ;
            }
            else
                vecprovabis5(i,offset(msh,cl)) = ( phi(pts[i] , msh , cl ) ) ;
        }
        
        Matrix<T, Dynamic, 1> sol0_bis = cod.solve(flux0_loc);
        Matrix<T, Dynamic, 1> sol1_bis = cod.solve(flux1_loc);
        Matrix<T, Dynamic, 1> sol0 = cod_bis.solve(flux0_loc);
        Matrix<T, Dynamic, 1> sol1 = cod_bis.solve(flux1_loc);
        Matrix<T, Dynamic, 1> sol_vel0 = cod_bis.solve(vel2_loc0);
        Matrix<T, Dynamic, 1> sol_vel1 = cod_bis.solve(vel2_loc1);
        Matrix<T, Dynamic, Dynamic> check_inversion = cod_bis.solve(one_loc);
        
        //T relative_error01 = (local_vandermonde*sol_vel- vel2_loc).norm() / vel2_loc.norm();
        //std::cout << "The relative error0 of VELOCITY is: " << relative_error01 << std::endl;
        
        
        //std::cout<<"Cell = "<<offset(msh,cl)<<", check_inversion_Vandermonde = "<<'\n'<<check_inversion <<std::endl;
        //std::cout<<"Cell = "<<offset(msh,cl)<<", vel2_loc = "<<'\n'<<vel2_loc <<std::endl;
        //std::cout<<"Cell = "<<offset(msh,cl)<<", sol_vel LOC = "<<'\n'<<sol_vel <<std::endl;
        //std::cout<<"Cell = "<<offset(msh,cl)<<", sol_vel LOC - u0 LOC = "<<'\n'<<sol_vel - u0_cellwise.col(i_fl)  <<std::endl;
        
        if(i_prova == 0){
        //std::cout<<"-------------------> sol0 - sol0_bis "<<'\n'<< sol0 - sol0_bis<<std::endl;
        //std::cout<<"-------------------> sol1 - sol1_bis "<<'\n'<< sol1 - sol1_bis<<std::endl;
            vecprovabis2.col(offset(msh,cl)) = flux0_loc  ; // è quello +0.5
            vecprovabis3.col(offset(msh,cl)) = flux1_loc  ; // è quello +0.5
            
        }
        if(i_prova == 1)
        {
            //std::cout<<"-------------------> sol0 - vecprovabis2 "<<'\n'<< flux0_loc - vecprovabis2.col(offset(msh,cl))<<std::endl;
            //std::cout<<"-------------------> sol1 - vecprovabis3 "<<'\n'<< flux1_loc - vecprovabis3.col(offset(msh,cl))<<std::endl;
            std::cout<<"-------------------> FLUX THEORETICAL 0 LOC"<<'\n'<< flux0_prova_loc - vecprovabis2.col(offset(msh,cl))<<std::endl;
            std::cout<<"-------------------> FLUX THEORETICAL 1 LOC"<<'\n'<< flux1_prova_loc - vecprovabis3.col(offset(msh,cl))<<std::endl;
            
            //std::cout<<"-------------------> CHECK phi VAL LOC ----------"<<'\n'<< vecprovabis5.col(offset(msh,cl)) - vecprovabis4.col(offset(msh,cl))<<std::endl;
            //std::cout<<"-------------------> CHECK phi VAL SOL +0.5 ----------"<<'\n'<<  vecprovabis4.col(offset(msh,cl))<<std::endl;
            //std::cout<<"-------------------> CHECK phi VAL SOL OLD ----------"<<'\n'<<  vecprovabis5.col(offset(msh,cl))<<std::endl;
            
            
        }
       
        Matrix<T, Dynamic, 1>     sol0_prova = cod.solve(flux0_prova_loc);
        Matrix<T, Dynamic, 1>     sol1_prova = cod.solve(flux1_prova_loc);
    
        
            
        if (cod.info() != Success)
        {
            std::cout<<"Not positive"<<std::endl;
            assert(0);
        }
        //T relative_error0 = (local_vandermonde*sol0_prova - flux0_prova_loc).norm() / flux0_prova_loc.norm();
        //T relative_error1 = (local_vandermonde*sol1_prova - flux1_prova_loc).norm() / flux1_prova_loc.norm();
        //T relative_error2 = (local_vandermonde*sol0 - flux0_loc).norm() / flux0_loc.norm();
        //T relative_error3 = (local_vandermonde*sol1- flux1_loc).norm() / flux1_loc.norm();
        //std::cout << "The relative error0 is: " << relative_error0 << std::endl;
        //std::cout << "The relative error1 is: " << relative_error1<< std::endl;
        //std::cout << "The relative error2 is: " << relative_error2 << std::endl;
        //std::cout << "The relative error3 is: " << relative_error3<< std::endl;
        cell_basis_Bernstein<Mesh,T> cb(msh, cl, phi.degree_FEM);
        auto cbs = cb.size();
        
        for (size_t i = 0; i < local_ndof ; i++)
        {
            
            size_t asm_map =  phi.connectivity_matrix[i_fl][i].first ;
            flux0(asm_map) = sol0(i) ;
            flux1(asm_map) = sol1(i) ;
            flux0_prova(asm_map) = sol0_prova(i) ;
            flux1_prova(asm_map) = sol1_prova(i) ;
            sol_vel_gl0(asm_map) = sol_vel0(i) ;
            sol_vel_gl1(asm_map) = sol_vel1(i) ;
            sol_vel_gl_cellwise(i,i_fl) = sol_vel0.dot( cb.eval_basis(pts[i]) );
        }
        
        
        i_fl++;
    }
    
    */
    
    
   // auto check_div_u = 1.0/2.0 * global_cij_x *sol_vel_gl0   + global_cij_y * sol_vel_gl1 ;
   // auto check_div_u_2 = 1.0/2.0 * Global_c_term_x_lagrangian *u0   + Global_c_term_y_lagrangian * u1 ;
    
    //std::cout<<"----------CHECK DIV Bernstein -> "<<'\n'<<check_div_u<<std::endl;
    //std::cout<<"----------CHECK DIV Lagrangian -> "<<'\n'<<check_div_u_2<<std::endl;
    
    /*
    for(auto& cl:msh.cells)
    {
        size_t cell_offset = offset(msh, cl) ;
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, fe_data.order);
        for (size_t i = 0; i < fe_data.local_ndof; i++)
        {
            auto pt = pts[i];
            std::cout<<"In pt = "<<pt<<" --> u0(pt) = "<<u(pt,msh,cl).first<<" and ERROR = "<<  (u(pt,msh,cl).first - sol_vel_gl_cellwise(i,cell_offset))<<std::endl;
        }
        
    }
    
    //std::cout<<"-------------------> sol_vel - u0 "<<'\n'<< sol_vel_gl - u0<<std::endl;
    
    
    tc_solver2.toc();
    std::cout  << "DIRECT INVERSION OF VANDERMONDE MATRIX LOCAL, t = " << tc_solver2 << " seconds"  << std::endl;
    
    
    
    timecounter tc_case00;
    tc_case00.tic();
    
    if(i_prova == 1){
        //std::cout<<"----->CHECK PROJECTION flux0 = "<<'\n'<<flux0 +0.5*u0 - vecprovabis0 <<std::endl;
        //std::cout<<"----->CHECK PROJECTION flux1 = "<<'\n'<<flux1 +0.5*u1 - vecprovabis1 <<std::endl;
        
        std::cout<<"----->CHECK PROJECTION flux0 TEORIC = "<<'\n'<<vecprovabis0 - flux0_prova<<std::endl;
              
        std::cout<<"----->CHECK PROJECTION flux1 TEORIC = "<<'\n'<<vecprovabis1 - flux1_prova <<std::endl;
        
        
        //std::cout<<"---> flux0 ERROR = "<<'\n'<<flux0 - vecprovabis0 <<std::endl;
        //std::cout<<"---> flux1 ERROR = "<<'\n'<<flux1 - vecprovabis1 <<std::endl;
        //std::cout<<"---> flux0  = "<<'\n'<<flux0  <<std::endl;
        //std::cout<<"---> flux1  = "<<'\n'<<flux1  <<std::endl;
        //std::cout<<"---> u0_cellwise = "<<'\n'<<u0_cellwise - vecprovabis2 <<std::endl;
        //std::cout<<"---> u1_cellwise = "<<'\n'<<u1_cellwise - vecprovabis3 <<std::endl;
        //std::cout<<"---> phi_FEM = "<<'\n'<<phi_FEM - vecprovabis4 <<std::endl;
        //std::cout<<"---> global_cij_x = "<<'\n'<<global_cij_x - vecprovabis5 <<std::endl;
        //std::cout<<"---> global_cij_y = "<<'\n'<<global_cij_y - vecprovabis6 <<std::endl;
        
    }
    if(i_prova == 0){
        vecprovabis0 = flux0 ;
        vecprovabis1 = flux1 ;
       
        
    }
    */
    // CONVOLUTION TERM
    Matrix<T, Dynamic, 1> conv_global = global_cij_x * phi_FEM  + global_cij_y * phi_FEM ;
    //Matrix<T, Dynamic, 1> prova0_fl =  0.5*(global_cij_x * u0  + global_cij_y * u1) ;
    Matrix<T, Dynamic, 1> prova0_fl1 =  global_cij_x * flux0_prova  + global_cij_y * flux1_prova ;
    
    if(i_prova == 0)
        vecprova2 = conv_global ;
    
    if(i_prova == 1){
        //std::cout<<"---> prova0_fl = "<<'\n'<<prova0_fl - vecprovabis7 <<std::endl;
        //std::cout<<"------> prova0_fl_somma phi_tilde convolution = "<<'\n'<<vecprova2 - conv_global - vecprovabis7  <<std::endl;
        std::cout<<"------> CHECK CONVOLUTION +0.5 = "<<'\n'<<prova0_fl1 - vecprova2   <<std::endl;
    }
    //if(i_prova == 0)//{
    //vecprovabis7 = prova0_fl ;
    //}
    //if(i_prova == 1)
        //std::cout<<"--------->   conv_global = "<<'\n'<<conv_global - vecprova2 <<std::endl;
    
    
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
    
    
    
    //tc_case00.toc();
    //std::cout  << "RESOLUTION OF LOW ORDER TRANSPORT, t = " << tc_case00 << " seconds" << std::endl;
    
    timecounter tc_case01;
    tc_case01.tic();
    
    // CHECK TIME STEP dt
    T dt_old = dt ;
    std::cout<<"---> COND IN TEMPO CFL, ALEXANDRE BOOK"<<std::endl;
    T CFL_numb = time_step_CFL_L2_velocity_NEW( dij.diagonal() , global_lumped_mass , fe_data.Dirichlet_boundary , dt );
    
    //T nu_max0 = CFL_numb/fe_data.hx;
    //T nu0 = dt_old/fe_data.hx;
    //T nu1 = dt/fe_data.hx;
    
    //std::cout<<"VALID FOR u = (1,0). nu_max VERO = "<<nu_max0<<" , nu max con dt assegnato = "<<nu0<< " and with dt appeared by CFL COND "<<nu1<<std::endl;
    
    if(dt_old != dt )
    {
        dt = floorf(CFL_numb*0.98 * 1000) / 1000;
        std::cout<<yellow<<bold<<"ATTENTION CHANGIN dt!!! ----> "<<reset<<"dt is "<<dt_old<<" and dt CFL is "<<dt<<" !"<<std::endl;
        //exit(10);
        
    }
    
    tc_case01.toc();
    std::cout << "TIME CHECKING, t = " << tc_case01 << " seconds" << std::endl;
    
 
  
    
    ///********* RESOLUTION OF THE SYSTEM: **********//
   
    
    // RESOLUTION FIRST ORDER
    Matrix<T, Dynamic, 1> phi_L = phi_FEM - dt * conv_global.cwiseQuotient(global_lumped_mass)  + dt * term_dij_no_entropy.cwiseQuotient(global_lumped_mass);
  
    if(i_prova == 1)
      //  std::cout<<"global_lumped_mass = "<<'\n'<<global_lumped_mass - vecprova0 <<std::endl;
    if(i_prova == 0)
        vecprova0 = global_lumped_mass ;
    
    if(i_prova == 1)
      //  std::cout<<"term_dij_no_entropy = "<<'\n'<<term_dij_no_entropy - vecprova1 <<std::endl;
    if(i_prova == 0)
        vecprova1 = term_dij_no_entropy ;
    
    
    
    // IMPOSITION DIRICHLET BOUNDARY CONDITIONS
    size_t counter_dir = 0 ;
    for (const auto& dir_elem : fe_data.Dirichlet_boundary )
    {
        if(dir_elem){
            phi_L(counter_dir) = phi_FEM(counter_dir) ;
        }
        counter_dir++ ;
    }
     
    
    if(i_prova == 1)
     //   std::cout<<"phi_FEM = "<<'\n'<<phi_FEM - phi2 <<std::endl;
    
    if(i_prova == 0)
        phi2 = phi_FEM ;
   
    
    // SAVING AND UPLOAD phi_L  INTO CLASS projected_level_set
    phi.sol_FEM = phi_L ;
    phi.converting_into_HHO_formulation(phi_L);
   
    
    
    tc.toc();
    std::cout << bold << yellow << "FEM method, time resolution: " << tc << " seconds" << reset << std::endl;
       
       
    
    /// PLOTTING SOLUTION (GNUPLOT) + SAVING FOR HHO (MISCHIATO PER POTERE PLOTTARE ENTRAMBE).
    //postprocess_output<double> postoutput5;
    //auto test_phi_L = std::make_shared< gnuplot_output_object<double> >("phi_L.dat");
    
    
    
    /*
    for(auto& cl :msh.cells)
    {
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree );
        for (auto pt : pts){
            test_phi_L->add_data( pt , phi(pt, msh , cl ) );
        }
    }
    postoutput5.add_object(test_phi_L);
    postoutput5.write();
    */
  
    
    std::cout<<yellow<<bold<<"----------- FINE TRANSPORT PROBLEM -----------"<<reset<<std::endl;
    
    

    //return phi_tilde;
    
}


template < typename Fonction, typename Mesh, typename Vel_Field , typename FiniteSpace , typename T = typename Mesh::coordinate_type >
void
run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST_to_check(const Mesh & msh, const FiniteSpace& fe_data, Fonction & phi , Vel_Field& u , T& dt , Matrix<T, Dynamic, 1>& phi2 , Matrix<T, Dynamic, 1>& phi3 , int i_prova , Matrix<T, Dynamic, 1>& vecprova0, Matrix<T, Dynamic, 1>& vecprova1, Matrix<T, Dynamic, 1>& vecprova2 , Matrix<T, Dynamic, 1>& vecprovabis0 , Matrix<T, Dynamic, 1>& vecprovabis1 ,Matrix<T, Dynamic, Dynamic>& vecprovabis2 , Matrix<T, Dynamic, Dynamic>& vecprovabis3 , Matrix<T, Dynamic, Dynamic>& vecprovabis4 ,Matrix<T, Dynamic, Dynamic>& vecprovabis5 , Matrix<T, Dynamic, 1>& vecprovabis6 , Matrix<T, Dynamic, 1>& vecprovabis7 )
{
    // Starting time for FE calculation
    std::cout<<yellow<<bold<<"----------- CHECKING ---> TRANSPORT PROBLEM LOW ORDER FAST -----------"<<reset<<std::endl;
    //std::cout<<yellow<<bold<<"PROVA--- USO MIXC LAGRANGE- BERNSTEIN"<<reset<<std::endl;
    
    timecounter tc;
    tc.tic();
    
    size_t degree = fe_data.order; // finite element order
    size_t dim = fe_data.ndof_FE ;
    //size_t n_cls = fe_data.n_cls ;
    size_t local_ndof = fe_data.local_ndof ; // local degrees of freedom
    auto S_i = fe_data.S_i;
    
   
    // SAVING PHI AND VELOCITY COEFFS
    auto phi_FEM = phi.sol_FEM ;
    auto u0 = u.sol_FEM.first ;
    auto u1 = u.sol_FEM.second ;
    auto u0_cellwise = u.sol_HHO.first ;
    auto u1_cellwise = u.sol_HHO.second ;
  
    std::cout<<"IMPORTANTEEEEE --------------------------_> i = "<<i_prova <<std::endl;
  
    // SAVING OF USEFUL MATRICES
    Matrix<T, Dynamic, 1> global_lumped_mass = phi.Global_Mass_Lumped;
    
    auto global_cij_x = phi.Global_c_term_x ;
    auto global_cij_y = phi.Global_c_term_y ;
    auto local_vandermonde = phi.local_vandermonde ;
    
    auto cij_norm = phi.cij_norm ;
    auto nij0 = phi.nij0 ;
    auto nij1 = phi.nij1 ;
    
    auto cji_norm = phi.cij_norm ;
    auto nji0 = phi.nji0 ;
    auto nji1 = phi.nji1 ;
    
    //std::cout<<"local_vandermonde"<<'\n'<<local_vandermonde<<std::endl;
    SparseMatrix<T>                 Global_c_term_x_lagrangian; // Global mass, saved for FEM problem
    SparseMatrix<T>                 Global_c_term_y_lagrangian;
    Global_c_term_x_lagrangian = SparseMatrix<T>( dim, dim ); //(b_i,b_j)_ij , b_i Lagrange basis fx
    Global_c_term_y_lagrangian = SparseMatrix<T>( dim, dim ); //(b_i,b_j)_ij , b_i Lagrange basis fx
    std::vector< Triplet<T> >       triplets_c_term_x_l; // Position elements: Sparse Matrix Notation
    std::vector< Triplet<T> >       triplets_c_term_y_l; // Position elements: Sparse Matrix Notation
    for( const auto& cl : msh.cells )
    {
        size_t cell_offset = offset(msh, cl) ;
        //auto local_cij_lagrangian = make_lagrangian_local_cij_matrix (msh, cl, degree);
        auto local_cij_lagrangian = make_bernstein_local_cij_matrix_with_velocity (msh, cl, degree , u );
        auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree);
        //std::cout<< "local_cij_lagrangian.first "<<'\n'<<local_cij_lagrangian.first<<std::endl;
    
        
        // Assembling triplets for global problem
        for (size_t i = 0; i < local_ndof; i++)
        {
            size_t asm_map_i = phi.connectivity_matrix[cell_offset][i].first ;
            for (size_t j = 0; j < local_ndof; j++)
            {
                size_t asm_map_j = phi.connectivity_matrix[cell_offset][j].first ;
                triplets_c_term_x_l.push_back( Triplet<T>(asm_map_i, asm_map_j , local_cij_lagrangian.first(i,j) ) );
                triplets_c_term_y_l.push_back( Triplet<T>( asm_map_i , asm_map_j , local_cij_lagrangian.second(i,j) ) );
            
            }
        }

    }

    Global_c_term_x_lagrangian.setFromTriplets( triplets_c_term_x_l.begin(), triplets_c_term_x_l.end() );
    triplets_c_term_x_l.clear();
    
    Global_c_term_y_lagrangian.setFromTriplets( triplets_c_term_y_l.begin(), triplets_c_term_y_l.end() );
    triplets_c_term_y_l.clear();
    
    Matrix<T, Dynamic, 1> sum0 = Eigen::Matrix<T, Dynamic, 1>::Zero(Global_c_term_x_lagrangian.rows(), 1);
       
    Matrix<T, Dynamic, 1> sum1 = Eigen::Matrix<T, Dynamic, 1>::Zero(Global_c_term_y_lagrangian.rows(), 1);
       
    for (size_t k = 0; k<global_cij_x.cols(); k++)
    {
        sum0 += (Global_c_term_x_lagrangian).col(k);
        sum1 += (Global_c_term_y_lagrangian).col(k);
    }
       
    for (size_t i=0; i<Global_c_term_x_lagrangian.rows(); i++) {
        std::cout<<"The sum is "<<sum1(i)<<" and "<<sum0(i)<<std::endl;
    }
       
       
       
    for(size_t counter_sum = 0 ; counter_sum < Global_c_term_x_lagrangian.rows() ; counter_sum++ )
    {
        for(size_t counter_col = 0 ; counter_col < Global_c_term_x_lagrangian.cols() ; counter_col++)
        {
       
            if(counter_col==counter_sum)
                std::cout<<"In ("<<counter_sum<<" , "<<counter_col<<" ), c^0 = "<<Global_c_term_x_lagrangian.coeff( counter_sum , counter_col )<<" and c1 = "<<Global_c_term_y_lagrangian.coeff( counter_sum , counter_col )<<std::endl;
            else
                std::cout<<"In ("<<counter_sum<<" , "<<counter_col<<" ), c^0_ij + c^0_ji = "<<Global_c_term_x_lagrangian.coeff( counter_sum , counter_col ) + Global_c_term_x_lagrangian.coeff( counter_col , counter_sum )<<" and c^1_ij + c^1_ji = "<<Global_c_term_y_lagrangian.coeff( counter_sum , counter_col ) + Global_c_term_y_lagrangian.coeff( counter_col , counter_sum ) <<std::endl;
        }
    }
    
    
    // VANDERMONDE MATRIX INTERPOLATION
    size_t i_fl = 0 ;
    Matrix<T, Dynamic, 1> flux0_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    Matrix<T, Dynamic, 1> flux1_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    
    Matrix<T, Dynamic, 1> flux0 = Matrix<T, Dynamic, 1>::Zero(dim) ;
    Matrix<T, Dynamic, 1> flux1 = Matrix<T, Dynamic, 1>::Zero(dim) ;
    
    Matrix<T, Dynamic, 1> flux0_prova = Matrix<T, Dynamic, 1>::Zero(dim) ;
    Matrix<T, Dynamic, 1> flux1_prova = Matrix<T, Dynamic, 1>::Zero(dim) ;
    Matrix<T, Dynamic, 1> flux0_prova_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    Matrix<T, Dynamic, 1> flux1_prova_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    Matrix<T, Dynamic, 1> vel2_loc0 = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    Matrix<T, Dynamic, 1> sol_vel_gl0 = Matrix<T, Dynamic, 1>::Zero(dim) ;
    Matrix<T, Dynamic, 1> vel2_loc1 = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    Matrix<T, Dynamic, 1> sol_vel_gl1 = Matrix<T, Dynamic, 1>::Zero(dim) ;
    
    
    Matrix<T, Dynamic, Dynamic> sol_vel_gl_cellwise = Matrix<T, Dynamic, Dynamic>::Zero(local_ndof,msh.cells.size()) ;
    Matrix<T, Dynamic, Dynamic> one_loc = Matrix<T, Dynamic, Dynamic>::Zero(local_ndof,local_ndof) ;
    
    timecounter tc_solver2;
    tc_solver2.tic();
    
    CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > cod(local_vandermonde);
    ColPivHouseholderQR<Matrix<T, Dynamic, Dynamic > > cod_bis(local_vandermonde);

    
    for(auto& cl : msh.cells)
    {
        // FLUX TERM : flux is a pair flux0 and flux1
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree );
        for (size_t i = 0; i < local_ndof ; i++)
        {
            flux0_loc(i) = u0_cellwise(i,i_fl) * phi(pts[i] , msh , cl );
            flux1_loc(i) = u1_cellwise(i,i_fl) * phi(pts[i] , msh , cl );
            //std::cout<<"pts[i] = "<<pts[i] << " HHO_u0 = "<<u0_cellwise(i,i_fl)<<std::endl;
            flux0_prova_loc(i) = u0_cellwise(i,i_fl) * (phi(pts[i] , msh , cl ) + 0.5);
            flux1_prova_loc(i) = u1_cellwise(i,i_fl) * (phi(pts[i] , msh , cl ) + 0.5);
            vel2_loc0(i) = u0_cellwise(i,i_fl) ;
            vel2_loc1(i) = u1_cellwise(i,i_fl) ;
            one_loc(i,i) = 1.0;
            if(i_prova == 0){
                //std::cout<<"phi(" << pts[i] <<", msh , cl ) = "<<phi(pts[i] , msh , cl )<<std::endl;
                vecprovabis4(i,offset(msh,cl)) = ( phi(pts[i] , msh , cl ) ) ;
            }
            else
                vecprovabis5(i,offset(msh,cl)) = ( phi(pts[i] , msh , cl ) ) ;
        }
        
        Matrix<T, Dynamic, 1> sol0_bis = cod.solve(flux0_loc);
        Matrix<T, Dynamic, 1> sol1_bis = cod.solve(flux1_loc);
        Matrix<T, Dynamic, 1> sol0 = cod_bis.solve(flux0_loc);
        Matrix<T, Dynamic, 1> sol1 = cod_bis.solve(flux1_loc);
        Matrix<T, Dynamic, 1> sol_vel0 = cod_bis.solve(vel2_loc0);
        Matrix<T, Dynamic, 1> sol_vel1 = cod_bis.solve(vel2_loc1);
        Matrix<T, Dynamic, Dynamic> check_inversion = cod_bis.solve(one_loc);
        
        //T relative_error01 = (local_vandermonde*sol_vel- vel2_loc).norm() / vel2_loc.norm();
        //std::cout << "The relative error0 of VELOCITY is: " << relative_error01 << std::endl;
        
        
        //std::cout<<"Cell = "<<offset(msh,cl)<<", check_inversion_Vandermonde = "<<'\n'<<check_inversion <<std::endl;
        //std::cout<<"Cell = "<<offset(msh,cl)<<", vel2_loc = "<<'\n'<<vel2_loc <<std::endl;
        //std::cout<<"Cell = "<<offset(msh,cl)<<", sol_vel LOC = "<<'\n'<<sol_vel <<std::endl;
        //std::cout<<"Cell = "<<offset(msh,cl)<<", sol_vel LOC - u0 LOC = "<<'\n'<<sol_vel - u0_cellwise.col(i_fl)  <<std::endl;
        
        if(i_prova == 0){
        //std::cout<<"-------------------> sol0 - sol0_bis "<<'\n'<< sol0 - sol0_bis<<std::endl;
        //std::cout<<"-------------------> sol1 - sol1_bis "<<'\n'<< sol1 - sol1_bis<<std::endl;
            vecprovabis2.col(offset(msh,cl)) = flux0_loc  ; // è quello +0.5
            vecprovabis3.col(offset(msh,cl)) = flux1_loc  ; // è quello +0.5
            
        }
        if(i_prova == 1)
        {
            //std::cout<<"-------------------> sol0 - vecprovabis2 "<<'\n'<< flux0_loc - vecprovabis2.col(offset(msh,cl))<<std::endl;
            //std::cout<<"-------------------> sol1 - vecprovabis3 "<<'\n'<< flux1_loc - vecprovabis3.col(offset(msh,cl))<<std::endl;
            std::cout<<"-------------------> FLUX THEORETICAL 0 LOC"<<'\n'<< flux0_prova_loc - vecprovabis2.col(offset(msh,cl))<<std::endl;
            std::cout<<"-------------------> FLUX THEORETICAL 1 LOC"<<'\n'<< flux1_prova_loc - vecprovabis3.col(offset(msh,cl))<<std::endl;
            
            //std::cout<<"-------------------> CHECK phi VAL LOC ----------"<<'\n'<< vecprovabis5.col(offset(msh,cl)) - vecprovabis4.col(offset(msh,cl))<<std::endl;
            //std::cout<<"-------------------> CHECK phi VAL SOL +0.5 ----------"<<'\n'<<  vecprovabis4.col(offset(msh,cl))<<std::endl;
            //std::cout<<"-------------------> CHECK phi VAL SOL OLD ----------"<<'\n'<<  vecprovabis5.col(offset(msh,cl))<<std::endl;
            
            
        }
       
        Matrix<T, Dynamic, 1>     sol0_prova = cod.solve(flux0_prova_loc);
        Matrix<T, Dynamic, 1>     sol1_prova = cod.solve(flux1_prova_loc);
    
        
            
        if (cod.info() != Success)
        {
            std::cout<<"Not positive"<<std::endl;
            assert(0);
        }
        //T relative_error0 = (local_vandermonde*sol0_prova - flux0_prova_loc).norm() / flux0_prova_loc.norm();
        //T relative_error1 = (local_vandermonde*sol1_prova - flux1_prova_loc).norm() / flux1_prova_loc.norm();
        //T relative_error2 = (local_vandermonde*sol0 - flux0_loc).norm() / flux0_loc.norm();
        //T relative_error3 = (local_vandermonde*sol1- flux1_loc).norm() / flux1_loc.norm();
        //std::cout << "The relative error0 is: " << relative_error0 << std::endl;
        //std::cout << "The relative error1 is: " << relative_error1<< std::endl;
        //std::cout << "The relative error2 is: " << relative_error2 << std::endl;
        //std::cout << "The relative error3 is: " << relative_error3<< std::endl;
        cell_basis_Bernstein<Mesh,T> cb(msh, cl, phi.degree_FEM);
        auto cbs = cb.size();
        
        for (size_t i = 0; i < local_ndof ; i++)
        {
            
            size_t asm_map =  phi.connectivity_matrix[i_fl][i].first ;
            flux0(asm_map) = sol0(i) ;
            flux1(asm_map) = sol1(i) ;
            flux0_prova(asm_map) = sol0_prova(i) ;
            flux1_prova(asm_map) = sol1_prova(i) ;
            sol_vel_gl0(asm_map) = sol_vel0(i) ;
            sol_vel_gl1(asm_map) = sol_vel1(i) ;
            sol_vel_gl_cellwise(i,i_fl) = sol_vel0.dot( cb.eval_basis(pts[i]) );
        }
        
        
        i_fl++;
    }
    
    
    auto check_div_u = 1.0/2.0 * global_cij_x *sol_vel_gl0   + global_cij_y * sol_vel_gl1 ;
    auto check_div_u_2 = 1.0/2.0 * Global_c_term_x_lagrangian *u0   + Global_c_term_y_lagrangian * u1 ;
    
    std::cout<<"----------CHECK DIV Bernstein -> "<<'\n'<<check_div_u<<std::endl;
    std::cout<<"----------CHECK DIV Lagrangian -> "<<'\n'<<check_div_u_2<<std::endl;
    
    
    for(auto& cl:msh.cells)
    {
        size_t cell_offset = offset(msh, cl) ;
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, fe_data.order);
        for (size_t i = 0; i < fe_data.local_ndof; i++)
        {
            auto pt = pts[i];
            std::cout<<"In pt = "<<pt<<" --> u0(pt) = "<<u(pt,msh,cl).first<<" and ERROR = "<<  (u(pt,msh,cl).first - sol_vel_gl_cellwise(i,cell_offset))<<std::endl;
        }
        
    }
    
    //std::cout<<"-------------------> sol_vel - u0 "<<'\n'<< sol_vel_gl - u0<<std::endl;
    
    
    tc_solver2.toc();
    std::cout  << "DIRECT INVERSION OF VANDERMONDE MATRIX LOCAL, t = " << tc_solver2 << " seconds"  << std::endl;
    
    
    
    timecounter tc_case00;
    tc_case00.tic();
    
    if(i_prova == 1){
        //std::cout<<"----->CHECK PROJECTION flux0 = "<<'\n'<<flux0 +0.5*u0 - vecprovabis0 <<std::endl;
        //std::cout<<"----->CHECK PROJECTION flux1 = "<<'\n'<<flux1 +0.5*u1 - vecprovabis1 <<std::endl;
        
        std::cout<<"----->CHECK PROJECTION flux0 TEORIC = "<<'\n'<<vecprovabis0 - flux0_prova<<std::endl;
              
        std::cout<<"----->CHECK PROJECTION flux1 TEORIC = "<<'\n'<<vecprovabis1 - flux1_prova <<std::endl;
        
        
        //std::cout<<"---> flux0 ERROR = "<<'\n'<<flux0 - vecprovabis0 <<std::endl;
        //std::cout<<"---> flux1 ERROR = "<<'\n'<<flux1 - vecprovabis1 <<std::endl;
        //std::cout<<"---> flux0  = "<<'\n'<<flux0  <<std::endl;
        //std::cout<<"---> flux1  = "<<'\n'<<flux1  <<std::endl;
        //std::cout<<"---> u0_cellwise = "<<'\n'<<u0_cellwise - vecprovabis2 <<std::endl;
        //std::cout<<"---> u1_cellwise = "<<'\n'<<u1_cellwise - vecprovabis3 <<std::endl;
        //std::cout<<"---> phi_FEM = "<<'\n'<<phi_FEM - vecprovabis4 <<std::endl;
        //std::cout<<"---> global_cij_x = "<<'\n'<<global_cij_x - vecprovabis5 <<std::endl;
        //std::cout<<"---> global_cij_y = "<<'\n'<<global_cij_y - vecprovabis6 <<std::endl;
        
    }
    if(i_prova == 0){
        vecprovabis0 = flux0 ;
        vecprovabis1 = flux1 ;
       
        
    }
    
    // CONVOLUTION TERM
    Matrix<T, Dynamic, 1> conv_global = global_cij_x * flux0  + global_cij_y * flux1 ;
    //Matrix<T, Dynamic, 1> prova0_fl =  0.5*(global_cij_x * u0  + global_cij_y * u1) ;
    Matrix<T, Dynamic, 1> prova0_fl1 =  global_cij_x * flux0_prova  + global_cij_y * flux1_prova ;
    
    if(i_prova == 0)
        vecprova2 = conv_global ;
    
    if(i_prova == 1){
        //std::cout<<"---> prova0_fl = "<<'\n'<<prova0_fl - vecprovabis7 <<std::endl;
        //std::cout<<"------> prova0_fl_somma phi_tilde convolution = "<<'\n'<<vecprova2 - conv_global - vecprovabis7  <<std::endl;
        std::cout<<"------> CHECK CONVOLUTION +0.5 = "<<'\n'<<prova0_fl1 - vecprova2   <<std::endl;
    }
    //if(i_prova == 0)//{
    //vecprovabis7 = prova0_fl ;
    //}
    //if(i_prova == 1)
        //std::cout<<"--------->   conv_global = "<<'\n'<<conv_global - vecprova2 <<std::endl;
    
    
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
    
    
    
    tc_case00.toc();
    std::cout  << "RESOLUTION OF LOW ORDER TRANSPORT, t = " << tc_case00 << " seconds" << std::endl;
    
    timecounter tc_case01;
    tc_case01.tic();
    
    // CHECK TIME STEP dt
    T dt_old = dt ;
    std::cout<<"---> COND IN TEMPO CFL, ALEXANDRE BOOK"<<std::endl;
    T CFL_numb = time_step_CFL_L2_velocity_NEW( dij.diagonal() , global_lumped_mass , fe_data.Dirichlet_boundary , dt );
    
    //T nu_max0 = CFL_numb/fe_data.hx;
    //T nu0 = dt_old/fe_data.hx;
    //T nu1 = dt/fe_data.hx;
    
    //std::cout<<"VALID FOR u = (1,0). nu_max VERO = "<<nu_max0<<" , nu max con dt assegnato = "<<nu0<< " and with dt appeared by CFL COND "<<nu1<<std::endl;
    
    if(dt_old != dt )
    {
        dt = floorf(CFL_numb*0.98 * 1000) / 1000;
        std::cout<<yellow<<bold<<"ATTENTION CHANGIN dt!!! ----> "<<reset<<"dt is "<<dt_old<<" and dt CFL is "<<dt<<" !"<<std::endl;
        //exit(10);
        
    }
    
    tc_case01.toc();
    std::cout << "TIME CHECKING, t = " << tc_case01 << " seconds" << std::endl;
    
 
  
    
    ///********* RESOLUTION OF THE SYSTEM: **********//
   
    
    // RESOLUTION FIRST ORDER
    Matrix<T, Dynamic, 1> phi_L = phi_FEM - dt * conv_global.cwiseQuotient(global_lumped_mass)  + dt * term_dij_no_entropy.cwiseQuotient(global_lumped_mass);
  
    if(i_prova == 1)
      //  std::cout<<"global_lumped_mass = "<<'\n'<<global_lumped_mass - vecprova0 <<std::endl;
    if(i_prova == 0)
        vecprova0 = global_lumped_mass ;
    
    if(i_prova == 1)
      //  std::cout<<"term_dij_no_entropy = "<<'\n'<<term_dij_no_entropy - vecprova1 <<std::endl;
    if(i_prova == 0)
        vecprova1 = term_dij_no_entropy ;
    
    
    
    // IMPOSITION DIRICHLET BOUNDARY CONDITIONS
    size_t counter_dir = 0 ;
    for (const auto& dir_elem : fe_data.Dirichlet_boundary )
    {
        if(dir_elem){
            phi_L(counter_dir) = phi_FEM(counter_dir) ;
        }
        counter_dir++ ;
    }
     
    
    if(i_prova == 1)
     //   std::cout<<"phi_FEM = "<<'\n'<<phi_FEM - phi2 <<std::endl;
    
    if(i_prova == 0)
        phi2 = phi_FEM ;
   
    
    // SAVING AND UPLOAD phi_L  INTO CLASS projected_level_set
    phi.sol_FEM = phi_L ;
    phi.converting_into_HHO_formulation(phi_L);
   
    
    
    tc.toc();
    std::cout << bold << yellow << "FEM method, time resolution: " << tc << " seconds" << reset << std::endl;
       
       
    
    /// PLOTTING SOLUTION (GNUPLOT) + SAVING FOR HHO (MISCHIATO PER POTERE PLOTTARE ENTRAMBE).
    //postprocess_output<double> postoutput5;
    //auto test_phi_L = std::make_shared< gnuplot_output_object<double> >("phi_L.dat");
    
    
    
    /*
    for(auto& cl :msh.cells)
    {
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree );
        for (auto pt : pts){
            test_phi_L->add_data( pt , phi(pt, msh , cl ) );
        }
    }
    postoutput5.add_object(test_phi_L);
    postoutput5.write();
    */
  
    
    std::cout<<yellow<<bold<<"----------- FINE TRANSPORT PROBLEM -----------"<<reset<<std::endl;
    
    

    //return phi_tilde;
    
}


template < typename Fonction, typename Mesh, typename Vel_Field , typename FiniteSpace , typename T = typename Mesh::coordinate_type >
void
run_FEM_BERNSTEIN_CORRECT(const Mesh & msh, const FiniteSpace& fe_data, Fonction & phi , Vel_Field& u , T& dt )
{
    // Starting time for FE calculation
    std::cout<<yellow<<bold<<"----------- STARTING TRANSPORT PROBLEM -----------"<<reset<<std::endl;
    //std::cout<<yellow<<bold<<"PROVA--- USO MIXC LAGRANGE- BERNSTEIN"<<reset<<std::endl;
    
    timecounter tc;
    tc.tic();
    
    size_t degree = fe_data.order; // finite element order
    size_t dim = fe_data.ndof_FE ;
    //size_t n_cls = fe_data.n_cls ;
    size_t local_ndof = fe_data.local_ndof ; // local degrees of freedom
    auto S_i = fe_data.S_i;
    
    // TOLTO PER ORA IL MAPPING NON SERVE NEL CASO BASE
    
    phi.coefficients_mapping(); // mapping of phi to have a phi between 0 and 1
   
    
    /// PLOT ---> PHI MAPPED INTO 0-1
    postprocess_output<double> postoutput4;
    auto test_phi_mapped = std::make_shared< gnuplot_output_object<double> >("phi_mapped.dat");
    for (auto cl:msh.cells) {
        auto pts = points(msh,cl) ;
        for (auto pt:pts) {
            T value = phi(pt,msh,cl);
            test_phi_mapped->add_data(pt,value);
        }
    }
    postoutput4.add_object(test_phi_mapped);
    postoutput4.write();
    
    
    // SAVING PHI AND VELOCITY COEFFS
    auto phi_FEM = phi.sol_FEM ;
    auto u0 = u.sol_FEM.first ;
    auto u1 = u.sol_FEM.second ;
    //u.set_max_vel(); // UPLOADING MAX VELOCITY OF u -> Bernstein Basis
    
    //std::cout<<"CONSTANT VELOCITY: u0 is "<<u0<<" and u1 is "<<u1<<std::endl;
    
    // NON LINEAR ENTROPY INITIALISATION
    const T eps = 1e-14 ; //constant into entropy
    non_linear_entropy<T,Fonction,Mesh> E(eps , phi ,msh );
    //SONOQUA1
    // PHI TILDE INITIALISATION --> (FOR HIGH ORDER METHOD)
    auto phi_tilde = L2_projection< T, Mesh , FiniteSpace> ( fe_data , msh );
    
    //L2_projection< T, Mesh , FiniteSpace> ( fe_data , msh );
    
    /// ENTROPY PLOT
    /*
    postprocess_output<double> postoutput6;
    auto cut_entropy = std::make_shared< gnuplot_output_object<double> >("entropy.dat");
    for (auto& cl:msh.cells)
    {
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree );
        for (auto& pt:pts)
        {
            T value = E(pt,cl);
            cut_entropy->add_data(pt,value);
        }
    }
    postoutput6.add_object(cut_entropy);
    postoutput6.write();
    */
    
   
    
    // SAVING OF USEFUL MATRICES
    auto global_mass = phi.Global_Mass ;
    Matrix<T, Dynamic, 1> global_lumped_mass = phi.Global_Mass_Lumped;
    
    auto global_cij_x = phi.Global_c_term_x ;
    auto global_cij_y = phi.Global_c_term_y ;
    auto local_vandermonde = phi.local_vandermonde ;
    
    // THE GLOBAL MATRIX c_ij varies each time: dependes on u -> c_ij = int(u*b_i*grad(b_j))
    /*
    SparseMatrix<T>                 global_cij_x = SparseMatrix<T>( dim, dim );
    SparseMatrix<T>                 global_cij_y = SparseMatrix<T>( dim, dim );
    std::vector< Triplet<T> >       triplets_c_term_x;
    std::vector< Triplet<T> >       triplets_c_term_y;
    for( const auto& cl : msh.cells )
    {
        size_t cell_offset = offset(msh, cl) ;
    
        
        // Local c_ij = u(x)*b_i* grad(b_j)
        auto local_cij = make_bernstein_local_cij_matrix_with_velocity(msh, cl, degree, u );
        
        for (size_t i = 0; i < local_ndof; i++)
        {
            auto asm_map_i = phi.connectivity_matrix[cell_offset][i].first ;
            for (size_t j = 0; j < local_ndof; j++)
            {
                auto asm_map_j = phi.connectivity_matrix[cell_offset][j].first ;
                triplets_c_term_x.push_back( Triplet<T>(asm_map_i, asm_map_j, local_cij.first(i,j) ) );
                triplets_c_term_y.push_back( Triplet<T>(asm_map_i, asm_map_j, local_cij.second(i,j) ) );
           }
            
        }
    }
    // FINALISATION c_ij GLOBAL ASSEMBLING
    global_cij_x.setFromTriplets( triplets_c_term_x.begin(), triplets_c_term_x.end() );
    triplets_c_term_x.clear();
    global_cij_y.setFromTriplets( triplets_c_term_y.begin(), triplets_c_term_y.end() );
    triplets_c_term_y.clear();
    */
    
    
    
    // INITIALISATION OF THE SOLVER (CONJUGATE GRADIENT)
    //ConjugateGradient<SparseMatrix<T> > solver_global_mass;
    
    timecounter tc1;
    tc1.tic();
    
    SimplicialLLT<SparseMatrix<T> >solver_global_mass;
    solver_global_mass.compute(global_mass); // use solver_global_mass to solve M^-1
    
    if(solver_global_mass.info()!=Success){
        std::cout<<"FAILED SOLVER 0->phi_tilde"<<std::endl;
        exit(1);
    }
    
    tc1.toc();
    std::cout << bold << yellow << "Inversion of the mass matrix: " << tc1 << " seconds" << reset << std::endl;
    
    /*
    SparseLU<SparseMatrix<T>, AMDOrdering<int> > solver2 ;
    solver2.compute(global_mass);
    if(solver2.info()!=Success) {
        std::cout<<"FAILED SOLVER2 PROVA ->phi_tilde"<<std::endl;
        return;
    }
    */
    
    /// CHECKING OF MASS MATRICES' PROPERTIES
    /*
    std::cout<<"First checking: Lumped Mass"<<std::endl;
    for(size_t i = 0; i<global_lumped_mass.rows();i++)
        if(global_lumped_mass(i)<0)
            std::cout<<"PROBLEM into lumped mass"<<std::endl;
    
    Matrix<T, Dynamic, Dynamic> mass_check = - global_mass;
    for(size_t i = 0 ; i < mass_check.rows() ; i++){
        mass_check(i,i) += global_lumped_mass(i);
        std::cout<<(mass_check.row(i)).sum()<<std::endl;
    }

    /// CHECK OF THE SUM PROPERTY OF CIJ
     Matrix<T, Dynamic, 1> sum0 = Eigen::Matrix<T, Dynamic, 1>::Zero(global_cij_x.rows(), 1);
     Matrix<T, Dynamic, 1> sum1 = Eigen::Matrix<T, Dynamic, 1>::Zero(global_cij_x.rows(), 1);
    
    for (size_t k = 0; k<global_cij_x.cols(); k++)
    {
        sum0 += (global_cij_x).col(k);
        sum1 += (global_cij_y).col(k);
    }
    
    for (size_t i=0; i<global_cij_y.rows(); i++) {
         std::cout<<"The sum is "<<sum1(i)<<" and "<<sum0(i)<<std::endl;
    }
    
    
    
    for(size_t counter_sum = 0 ; counter_sum < global_cij_x.rows() ; counter_sum++ )
    {
        for(size_t counter_col = 0 ; counter_col < global_cij_x.cols() ; counter_col++)
        {
    
           if(counter_col==counter_sum)
               std::cout<<"In ("<<counter_sum<<" , "<<counter_col<<" ), c^0 = "<<global_cij_x.coeff( counter_sum , counter_col )<<" and c1 = "<<global_cij_y.coeff( counter_sum , counter_col )<<std::endl;
           else
                std::cout<<"In ("<<counter_sum<<" , "<<counter_col<<" ), c^0_ij + c^0_ji = "<<global_cij_x.coeff( counter_sum , counter_col ) + global_cij_x.coeff( counter_col , counter_sum )<<" and c^1_ij + c^1_ji = "<<global_cij_y.coeff( counter_sum , counter_col ) + global_cij_y.coeff( counter_col , counter_sum ) <<std::endl;
        }
    }
    
 */
    
    // FLUX TERM
    /*
    Matrix<T, Dynamic, 1>  RHS_flux0 =  Matrix<T, Dynamic, 1>::Zero(dim) ;
    Matrix<T, Dynamic, 1>  RHS_flux1 =  Matrix<T, Dynamic, 1>::Zero(dim) ;
    size_t counter_flux = 0 ;
    for(auto& cl : msh.cells)
    {
        // FLUX TERM : flux is a pair flux0 and flux1
        auto local_flux = make_bernstein_local_RHS_FLUX( msh , cl , degree , u , phi ) ;
        //std::cout<<"flux_loc0  "<<'\n'<<flux_loc.first<<std::endl;
        //std::cout<<"flux_loc1  "<<'\n'<<flux_loc.second<<std::endl;
        for (size_t i = 0; i < local_ndof ; i++)
        {
            size_t asm_map =  phi.connectivity_matrix[counter_flux][i].first ;
            RHS_flux0(asm_map) += local_flux.first(i) ;
            RHS_flux1(asm_map) += local_flux.second(i) ;
        }
        counter_flux++;
    }
    Matrix<T, Dynamic, 1> flux0 = solver_global_mass.solve(RHS_flux0);
    Matrix<T, Dynamic, 1> flux1 = solver_global_mass.solve(RHS_flux1);
     */
    
    
    // ALTERNATIVE VANDERMONDE MATRIX
    
    //Matrix<T, Dynamic, 1> flux0 = u0.cwiseProduct( phi_FEM ) ;
    //Matrix<T, Dynamic, 1> flux1 = u1.cwiseProduct( phi_FEM );
    
    size_t i_fl = 0 ;
    Matrix<T, Dynamic, 1> flux0_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    Matrix<T, Dynamic, 1> flux1_loc = Matrix<T, Dynamic, 1>::Zero(local_ndof) ;
    
    Matrix<T, Dynamic, 1> flux0 = Matrix<T, Dynamic, 1>::Zero(dim) ;
    Matrix<T, Dynamic, 1> flux1 = Matrix<T, Dynamic, 1>::Zero(dim) ;
    
    //Matrix<T, Dynamic, Dynamic> local_vandermonde = Matrix<T, Dynamic, Dynamic>::Zero ( local_ndof , local_ndof) ;

    
    //T error0_L2 = 0.0 , error1_L2 = 0.0 ;
    
    CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > cod( local_vandermonde );
    
    
    for(auto& cl : msh.cells)
    {
        // FLUX TERM : flux is a pair flux0 and flux1
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree );
        //cell_basis_Bernstein<Mesh,T> cb(msh, cl, degree);
        for (size_t i = 0; i < local_ndof ; i++)
        {
            flux0_loc(i) = u(pts[i] , msh , cl ).first * phi(pts[i] , msh , cl );
            flux1_loc(i) = u(pts[i] , msh , cl ).second * phi(pts[i] , msh , cl );
            // LOCAL VANDERMONDE MATRIX -> è uguale penso basti farla tra [0,1] e riportarla qua
            //local_vandermonde.block(i,0,1,local_ndof) = (cb.eval_basis(pts[i])).transpose() ;
            
            //size_t asm_map =  phi.connectivity_matrix[i_fl][i].first ;
            
        }
        
        //std::cout<<'\n'<<'\n'<<"local_vandermonde "<<local_vandermonde<<'\n'<<std::endl;
        //CompleteOrthogonalDecomposition<Matrix<T, Dynamic, Dynamic > > cod( local_vandermonde );
        Matrix<T, Dynamic, 1> sol0 = cod.solve(flux0_loc);
        Matrix<T, Dynamic, 1> sol1 = cod.solve(flux1_loc);
        //std::cout<<"flux0_loc -sol0"<<'\n'<<flux0_loc-sol0<<'\n'<<std::endl;
        //if (cod.info() != Success)
        //{
        //    std::cout<<"Not positive"<<std::endl;
        //    assert(0);
        //}

        
        
        for (size_t i = 0; i < local_ndof ; i++)
        {
            
            size_t asm_map =  phi.connectivity_matrix[i_fl][i].first ;
            flux0(asm_map) = sol0(i) ;
            flux1(asm_map) = sol1(i) ;
            
        }
        
        
        i_fl++;
    }
    
    
    
    
    //std::cout<<"flux0_bis:"<<'\n'<<flux0_bis <<'\n' <<std::endl;
    //std::cout<<"flux0:"<<'\n'<< flux0<<'\n' <<std::endl;
/*
    T sum0 = 0.0 , sum1 = 0.0 ;
    for (int ii = 0 ; ii < dim ; ii++ )
    {
        sum0 += pow( (flux0_bis(ii) - flux0(ii)) , 2 );
        sum1 += pow( (flux1_bis(ii) - flux1(ii)) , 2 );
    }
    
    error0_L2 = sqrt( 1.0/dim * sum0 );
    error1_L2 = sqrt( 1.0/dim * sum1 );
    std::cout<<"ERRORE L2 (flux0_bis - flux0):"<<'\n'<<error0_L2<<'\n' <<std::endl;
    std::cout<<"ERRORE L2 (flux1_bis - flux1):"<<'\n'<<error1_L2<<'\n' <<std::endl;
*/
    
    
    // RESOLUTION OF phi_tilde (GLOBALLY)    ( with cij(phi_j) )
    Matrix<T, Dynamic, 1> mass_phi_old = global_mass * phi_FEM ; // USEFUL ALSO FOR HO PHI_H
    //std::cout<<"vec1  "<<'\n'<<mass_phi_old<<std::endl;
    
    // CONVOLUTION TERM
    Matrix<T, Dynamic, 1> conv_global = global_cij_x * flux0  + global_cij_y * flux1 ;
    //Matrix<T, Dynamic, 1> conv_global = global_cij_x * phi_FEM  + global_cij_y * phi_FEM ;
    /*
    Matrix<T, Dynamic, 1> flux0 = 10 * phi_FEM ;
    Matrix<T, Dynamic, 1> flux1 = 0 * phi_FEM ;
    Matrix<T, Dynamic, 1> conv_global2 = phi.Global_c_term_x * flux0  + phi.Global_c_term_y * flux1 ;
    std::cout<<"CHECK cij conv_global  "<<'\n'<<conv_global-conv_global2<<std::endl;
    */
    
    
    

       
    
    // NORM of c_ij
    SparseMatrix<T> cij_norm = ( global_cij_x.cwiseProduct(global_cij_x) + global_cij_y.cwiseProduct(global_cij_y) ).cwiseSqrt() ;
    //std::cout<<"cij norm "<<'\n'<<cij_norm<<std::endl;
    
    // MATRIX n_ij
    SparseMatrix<T> nij0 = global_cij_x.cwiseQuotient( cij_norm );
    SparseMatrix<T> nij1 = global_cij_y.cwiseQuotient( cij_norm );
    
    //std::cout<<"nij1  "<<'\n'<<nij1<<std::endl;
    

    // MATRIX c_ji
    SparseMatrix<T> cji_x = global_cij_x.adjoint() ;
    SparseMatrix<T> cji_y = global_cij_y.adjoint() ;
     
    // NORM of c_ji -> i.e. c_ij transposed
    SparseMatrix<T> cji_norm = (cji_x.cwiseProduct(cji_x)+cji_y.cwiseProduct(cji_y)).cwiseSqrt();
    
    // MATRIX n_ij (TRANSPOSED)
    SparseMatrix<T> nji0 = cji_x.cwiseQuotient( cji_norm );
    SparseMatrix<T> nji1 = cji_y.cwiseQuotient( cji_norm );
    
    
    // NORMAL VELOCITY
    SparseMatrix<T> normal_vel = SparseMatrix<T>( dim , dim );
    std::vector< Triplet<T> >   triplets;
    
    // NORMAL VELOCITY TRANSPOSED
    SparseMatrix<T> normal_vel_adj = SparseMatrix<T>( dim , dim );
    std::vector< Triplet<T> >   triplets_adj ;
    /*
    // NORMAL VELOCITY
    SparseMatrix<T> normal_vel_bis = SparseMatrix<T>( dim , dim );
    std::vector< Triplet<T> >   triplets0_bis;
    
    // NORMAL VELOCITY TRANSPOSED
    SparseMatrix<T> normal_vel_bis_adj = SparseMatrix<T>( dim , dim );
    std::vector< Triplet<T> >   triplets1_bis ;
    */
    size_t counter_row = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            auto value0 = std::abs( u0(counter_row) * nij0.coeff(counter_row,elem) + u1(counter_row) * nij1.coeff(counter_row,elem) );
            auto value1 = std::abs( u0(elem) * nij0.coeff(counter_row,elem) + u1(elem) * nij1.coeff(counter_row,elem) );
            auto value = std::max(value0 , value1);
            
            auto value_adj0 = std::abs( u0(counter_row) * nji0.coeff(counter_row,elem) + u1(counter_row) * nji1.coeff(counter_row,elem) );
            auto value_adj1 = std::abs( u0(elem) * nji0.coeff(counter_row,elem) + u1(elem) * nji1.coeff(counter_row,elem) );
            auto value_adj = std::max(value_adj0 , value_adj1);
               
            triplets.push_back( Triplet<T>(counter_row, elem, value ) );
            triplets_adj.push_back( Triplet<T>(counter_row, elem, value_adj ) );
            
            
            // CHECK IT
            /*
            auto value0_bis = std::max( u0(counter_row) , u0(elem) );
            auto value1_bis = std::max( u1(counter_row) , u1(elem) );
            auto val = value0_bis * nij0.coeff(counter_row,elem) + value1_bis * nij1.coeff(counter_row,elem) ;
            auto val_adj = value0_bis * nji0.coeff(counter_row,elem) + value1_bis * nji1.coeff(counter_row,elem) ;
            
            triplets0_bis.push_back( Triplet<T>(counter_row, elem, val ) );
            triplets1_bis.push_back( Triplet<T>(counter_row, elem, val_adj ) );
            */
            
               
        }
        counter_row++;
    }
    
    normal_vel.setFromTriplets( triplets.begin(), triplets.end() );
    triplets.clear();
    normal_vel_adj.setFromTriplets( triplets_adj.begin(), triplets_adj.end() );
    triplets_adj.clear();
    
    /*
    normal_vel_bis.setFromTriplets( triplets0_bis.begin(), triplets0_bis.end() );
    triplets0_bis.clear();
    normal_vel_bis_adj.setFromTriplets( triplets1_bis.begin(), triplets1_bis.end() );
    triplets1_bis.clear();
    SparseMatrix<T> normal_vel_abs_bis = normal_vel_bis.cwiseAbs() ;  // o lambda_max=normal_vel ?
    SparseMatrix<T> normal_vel_abs_adj_bis = normal_vel_bis_adj.cwiseAbs();
    */
    // CHECKING NEW METHOD NORMAL VEL
    /*
    for(size_t i = 0 ; i < normal_vel_abs_bis.rows() ; i++)
    {
        for(size_t j = 0 ; j < normal_vel_abs_bis.cols() ; j++){
            if(std::abs(normal_vel.coeff(i,j)-normal_vel_abs_bis.coeff(i,j))>1e-5)
                std::cout<<"In ("<<i<<" , "<<j<<" ), normal vel OLD is "<<normal_vel_abs_bis.coeff(i,j)<<" , normal vel NEW is "<<normal_vel.coeff(i,j)<<std::endl;
            
            if(std::abs(normal_vel_adj.coeff(i,j)-normal_vel_abs_adj_bis.coeff(i,j))>1e-5)
                std::cout<<"In ("<<i<<" , "<<j<<" ), normal vel ADJ OLD is "<<normal_vel_abs_adj_bis.coeff(i,j)<<" , normal vel  ADJ NEW is "<<normal_vel_adj.coeff(i,j)<<std::endl;
            
        }
    }
    */
    //SparseMatrix<T> normal_vel2 = u0 * nij0 + u1 * nij1 ;
    //SparseMatrix<T> normal_vel_adj2 = u0 * nji0 + u1 * nji1 ;
    
    //std::cout<<"normal_vel - nij0"<<'\n'<<normal_vel - nij0<<std::endl;
    //std::cout<<"normal_vel_adj - nij0"<<'\n'<<normal_vel_adj - nji0<<std::endl;
    //std::cout<<"CHECK normal_vel2  : "<<'\n'<<normal_vel-normal_vel2<<'\n'<<std::endl;
    //std::cout<<"CHECK normal_vel_adj2 : "<<'\n'<<normal_vel_adj-normal_vel_adj2<<'\n'<<std::endl;
   
    // MATRIX Dij CALCULATION:
    // normval_vel and the adjoint one are already absolute value
    
    //SparseMatrix<T> normal_vel_abs = normal_vel.cwiseAbs() ;  // o lambda_max=normal_vel ?
    //SparseMatrix<T> normal_vel_abs_adj = normal_vel_adj.cwiseAbs();// o lambda_max=normal_vel?
 
    
    /// POSSO SCEGLIERE SE METTERE normal_vel_abs_bis O normal_vel !!!!!!!!!!!!!!!!
    //std::cout<<bold<<yellow<<"--->  sto usando normal_vel_abs_bis, i.e. con max su u(pt)!!"<<reset<<std::endl;
     std::cout<<bold<<yellow<<"--->  sto usando normal_vel, i.e. con max su u*n !!" <<reset<<std::endl;
    
    
    //SparseMatrix<T> lambda_max = normal_vel_abs_bis.cwiseProduct( cij_norm );
    //SparseMatrix<T> lambda_max_adj = normal_vel_abs_adj_bis.cwiseProduct( cji_norm );
    SparseMatrix<T> lambda_max = normal_vel.cwiseProduct( cij_norm );
    SparseMatrix<T> lambda_max_adj = normal_vel_adj.cwiseProduct( cji_norm );
    
    
    SparseMatrix<T> dij = lambda_max.cwiseMax(lambda_max_adj);
    //SparseMatrix<T> dij2 = lambda_max.cwiseMax(lambda_max_adj);
    
    for(size_t i = 0 ; i < dij.rows() ; i++)
    {
        dij.coeffRef(i,i) = 0;
        dij.coeffRef(i,i) = -dij.row(i).sum();
    }
    //std::cout<<"dij-dij2"<<'\n'<<dij-dij2<<std::endl;
    
    // CHECK TIME STEP dt
    T dt_old = dt ;
    std::cout<<bold<<yellow<<"---> COND IN TEMPO CFL, ALEXANDRE BOOK"<<reset<<std::endl;
    T CFL_numb = time_step_CFL_L2_velocity_NEW( dij.diagonal() , global_lumped_mass , fe_data.Dirichlet_boundary , dt );
    
    //T nu_max0 = CFL_numb/fe_data.hx;
    //T nu0 = dt_old/fe_data.hx;
    //T nu1 = dt/fe_data.hx;
    
    //std::cout<<"VALID FOR u = (1,0). nu_max VERO = "<<nu_max0<<" , nu max con dt assegnato = "<<nu0<< " and with dt appeared by CFL COND "<<nu1<<std::endl;
    //T nu_max = dt/fe_data.hx;
    //std::cout<<"VALID FOR u = (1,0). nu_max = "<<nu_max<<std::endl;
    if(dt_old != dt )
    {
        std::cout<<"dt is "<<dt_old<<" and dt CFL is "<<dt<<" . STOP!"<<std::endl;
        exit(10);
    }
    
    
    
    
    // CONSTANT TERM (PHI TILDE PROBLEM)
    //Matrix<T,Dynamic,1> vec2 = Matrix<T, Dynamic, 1>::Zero(dim) ;
    //averaged_sum_Si_SPARSE(S_i , global_cij_x , flux0 , vec2);
    //averaged_sum_Si_SPARSE(S_i , global_cij_y , flux1 , vec2);
    
    //std::cout<<"CHECK phi TILDE "<<'\n'<<conv_global-vec2<<'\n'<<"FINE PHI TILDE CHECKING"<<std::endl;
    Matrix<T, Dynamic, 1> b = mass_phi_old - dt*conv_global.cwiseQuotient(global_lumped_mass);
    //std::cout<<"TERMINE NOTO:b  "<<'\n'<<b<<std::endl;
    
    // RESOLUTION OF PHI_TILDE
    phi_tilde.sol_FEM = solver_global_mass.solve(b); // SAVE THE L2 projection
    
    //auto prova_phi_tilde = solver2.solve(b);
    //std::cout<<"phi_tilde.sol_FEM  "<<'\n'<<phi_tilde.sol_FEM <<std::endl;
    
    //std::cout<<"phi_tilde.sol_FEM - prova_phi_tilde "<<'\n'<<phi_tilde.sol_FEM - prova_phi_tilde <<std::endl;
    
    
    if(solver_global_mass.info()!=Success) {
        std::cout<<"FAILED SOLVER 1->phi_tilde"<<std::endl;
        exit(1);
    }
    
    // SAVING BOTH SOL_HHO AND VERTICES OF PHI_TILDE
    std::cout<<"CONVERTING phi_tilde"<<std::endl;
    phi_tilde.converting_into_HHO_formulation( phi_tilde.sol_FEM );
    
    
    
    // TERM R_i^n
    Matrix<T, Dynamic, 1> Emax_global = Eigen::Matrix<T, Dynamic, 1>::Ones(dim, 1);
    Matrix<T, Dynamic, 1> Emin_global = Eigen::Matrix<T, Dynamic, 1>::Ones(dim, 1);
    Emax_global *= -1e20;
    Emin_global *= 1e20;
    Matrix<T, Dynamic, 1> R_i = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    
    for( auto& cl: msh.cells )
        r_i_calculator_Bernstein( msh , cl , E , phi_tilde , phi , dt , u , Emax_global , Emin_global , R_i);
    
    // R_i FINALISATION:
    //std::cout<<"Emax_global: "<<'\n'<<Emax_global<<std::endl;
    //std::cout<<"Emin_global: "<<'\n'<<Emin_global<<std::endl;
    R_i = R_i.cwiseQuotient( Emax_global - Emin_global );
    //std::cout<<"Emax_global - Emin_global = "<<'\n'<<Emax_global-Emin_global<<std::endl;
    //std::cout<<"Ri = "<<'\n'<<R_i<<std::endl;
    
    
    
    // ENTROPIC SOLUTION: MINIMUM BETWEEN d_ij AND R_i --> d^E_ij MATRIX
    T c_e = 1.0;
    SparseMatrix<T> dE_ij  = SparseMatrix<T>( dim , dim );
    std::vector< Triplet<T> >   triplets_dE_ij;
    SparseMatrix<T> phi_ij = SparseMatrix<T>( dim , dim );
    std::vector< Triplet<T> >   triplets_phi_ij;
    
    
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            
            if(elem!=counter)
            {
                auto R_i_j = c_e * std::max(std::abs( R_i(counter) ) , std::abs( R_i(elem) ));
                auto value = std::min( dij.coeff( counter , elem ), R_i_j );
                triplets_dE_ij.push_back( Triplet<T>(counter, elem, value ) );
            //dE_ij( counter_row , elem ) = std::min( dij( counter_row , elem ), R_i_j );
            //std::cout<<"R_i_j "<<R_i_j<<" and dij"<<dij( counter_row , elem )<<std::endl;
            }
            auto value_bis = 0.5*( phi_FEM(counter) + phi_FEM(elem) );
            triplets_phi_ij.push_back( Triplet<T>(counter, elem, value_bis ) );
            
            
        }
        counter++;
    }
    
    //std::cout<<"term_dij_no_entropy : "<<'\n'<<term_dij_no_entropy<<std::endl;
    
    dE_ij.setFromTriplets( triplets_dE_ij.begin(), triplets_dE_ij.end() );
    triplets_dE_ij.clear();
    phi_ij.setFromTriplets( triplets_phi_ij.begin(), triplets_phi_ij.end() );
    triplets_phi_ij.clear();
    
    //std::cout<<"dE_ij: "<<'\n'<<dE_ij<<std::endl;
    
    // MATRIX d^C_ij
    
    T c_comp = 1.0;
    SparseMatrix<T> tmp0 = phi_ij - phi_ij.cwiseProduct(phi_ij);
    //std::cout<<"tmp0 "<<'\n'<<tmp0<<std::endl;
    positive_part_SPARSE( tmp0 );
    //std::cout<<"tmp0 POSITIVE "<<'\n'<<tmp0<<std::endl;
    
    SparseMatrix<T> tmp_dC1 =  SparseMatrix<T>( dim , dim );
    std::vector< Triplet<T> >   triplets_tmp_dC1;
    SparseMatrix<T> Mat_One =  SparseMatrix<T>( dim , dim );
    //Matrix<T, Dynamic, Dynamic> Mat_One = Eigen::Matrix<T, Dynamic, Dynamic>::Ones( dim, dim );
    std::vector< Triplet<T> >   triplets_1;
    
    size_t counter_tmp = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
           // std::cout<<"In ("<<counter<<" , "<<elem<<") tmp_dC0 = "<<tmp_dC0(counter,elem)<<" and phi.ver diff "<<std::abs(phi.vertices(counter) - phi.vertices(elem))<<std::endl;
            if( std::abs(phi_FEM(counter_tmp) - phi_FEM(elem))>1e-15 ){
                auto value = tmp0.coeff(counter_tmp,elem)/( std::abs(phi_FEM(counter_tmp) - phi_FEM(elem)) );
                triplets_tmp_dC1.push_back( Triplet<T>(counter_tmp, elem, value ) );
                triplets_1.push_back( Triplet<T>(counter_tmp, elem, 1.0 ) );
            }
            //tmp_dC1(counter_tmp,elem) = tmp0(counter_tmp,elem)/( std::abs(sol_FEM(counter_tmp) - sol_FEM(elem)) );
        }
        counter_tmp++;
    }
    tmp_dC1.setFromTriplets( triplets_tmp_dC1.begin(), triplets_tmp_dC1.end() );
    triplets_tmp_dC1.clear();
    
    
    Mat_One.setFromTriplets( triplets_1.begin(), triplets_1.end() );
    
    
    Matrix<T , Dynamic , Dynamic > Mat1 = Eigen::Matrix<T, Dynamic, Dynamic>::Ones(dim, dim);
    Matrix<T , Dynamic , Dynamic > tmp1_bis = Mat1 - c_comp*tmp_dC1;
    positive_part( tmp1_bis );
    Matrix<T , Dynamic , Dynamic > dC_ij_bis = dE_ij.cwiseProduct(tmp1_bis) ;


    
    //std::cout<<'\n'<<"tmp_dC1: "<<'\n'<<tmp_dC1<<std::endl;
    SparseMatrix<T> tmp1 = Mat_One - c_comp*tmp_dC1;
    
    //Matrix<T, Dynamic, Dynamic> tmp1 = Mat_One - c_comp*tmp_dC1;
    //positive_part( tmp1  );
    //SparseMatrix<T> dC_ij = SparseMatrix<T>( dim , dim );
    //size_t counter_prova = 0;
    //for(auto& row_i:S_i)
    //{
     //   for(auto& elem:row_i)
     //   {
     //       dC_ij.coeffRef(counter_prova,elem) += ( dE_ij.coeff(counter_prova,elem)*tmp1(counter_prova,elem) );
     //   }
      //  counter_prova++;
    //}
    
   
    
    //std::cout<<'\n'<<"tmp1: "<<'\n'<<tmp1<<std::endl;
    positive_part_SPARSE( tmp1  );
    
    SparseMatrix<T> dC_ij = dE_ij.cwiseProduct(tmp1) ;
    
    /*
    SparseMatrix<T> dC_ij_prova = dC_ij ;
    // NON SONO SICURO SERVA STA COSA
    std::cout<<"CHECK IF SERVE L'IMPOSIZIONE PER L'ELEMENTO DIAGONALE = SOMMA RIGA PER VARI d^I_ij!!!!"<<std::endl;
    for(size_t i = 0 ; i < dC_ij.rows() ; i++)
    {
        dC_ij.coeffRef(i,i) = 0.;
        dij.coeffRef(i,i) = 0.;
        dC_ij.coeffRef(i,i) = -dC_ij.row(i).sum();
        dij.coeffRef(i,i) = -dij.row(i).sum();
    }
    
    std::cout<<'\n'<<"dC_ij_prova - dC_ij: "<<'\n'<<dC_ij_prova - dC_ij<<std::endl;
    */
    
    // ARTIFICIAL VISCOSITY TERM d_ij * (phi_j-phi_i)
    
    Matrix<T, Dynamic, 1> term_dij_no_entropy =  Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    Matrix<T, Dynamic, 1> term_dij_E            =  Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    Matrix<T, Dynamic, 1> term_dij            =  Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    
    //Matrix<T, Dynamic, 1> term_dij_bis =  Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    
    size_t counter_dij = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
             // TERM d_ij * (phi_j-phi_i)
            term_dij_no_entropy(counter_dij) += dij.coeff(counter_dij,elem)*(phi_FEM(elem)-phi_FEM(counter_dij));
             // TERM dC_ij * (phi_j-phi_i)
            term_dij(counter_dij) += dC_ij.coeff(counter_dij,elem)*(phi_FEM(elem)-phi_FEM(counter_dij));
            term_dij_E(counter_dij) += dE_ij.coeff(counter_dij,elem)*(phi_FEM(elem)-phi_FEM(counter_dij));
            
            //term_dij_bis(counter_dij) += dC_ij_bis.coeff(counter_dij,elem)*(phi_FEM(elem)-phi_FEM(counter_dij));
        }
        counter_dij++;
    }
    
    //std::cout<<"term_dij_bis - term_dij"<<'\n'<<term_dij_bis - term_dij<<std::endl;
    
    // FOR CHECKING
    //std::cout<<"CHECK term_dij_no_entropy : "<<'\n'<<term_dij_no_entropy - term_dij<<std::endl;
    
    //std::cout<<"CHECK term_dij_no_entropy : "<<'\n'<<term_dij_no_entropy - term_dij_no_entropy2<<std::endl;
    
    //Matrix<T, Dynamic, 1> term_dij_no_entropy2 = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1 );
    //term_dij_no_entropy2 = dij * phi_FEM ;
    
    //Matrix<T, Dynamic, 1> term_dij2 = dC_ij * phi_FEM ;
    //std::cout<<"CHECK term_dij2 : "<<'\n'<<term_dij2 - term_dij<<std::endl;
    
    //std::cout<<"CHECK term_dij_no_entropy : "<<'\n'<<term_dij_no_entropy - term_dij_no_entropy2<<std::endl;
    
    
    
    /// MATRIX D_IJ CHECKING
    /*
     //The  error is < 1e-15, at max
    std::cout<<"D_ij check: "<<std::endl;
    for(size_t i = 0; i<lambda_max.rows();i++){
        for(size_t j = i+1; j<lambda_max.rows();j++){
            if(std::abs(dij.coeff(i,j)-dij.coeff(j,i))>1e-15)
            std::cout<<dij.coeff(i,j)-dij.coeff(j,i);
        }
        std::cout<<'\n';
    }
    std::cout<<std::endl;
    
    
     std::cout<<'\n'<<"dij: "<<'\n'<<dij<<std::endl;
    
    std::cout<<"D_ij check symmetry "<<std::endl;
    for(size_t counter_row = 0 ; counter_row < global_cij_x.rows() ; counter_row++ )
    {
        
        for(size_t elem = 0 ; elem < global_cij_x.cols() ; elem++)
        {
            std::cout<<dij.coeff(counter_row,elem) - dij.coeff(elem,counter_row)<<std::endl;
        }
        std::cout<<'\n';
    }
    std::cout<<std::endl;
    */
    
    /// TERM D_IJ*(PHI_J-PHI_I)  CHECKING
    /*
    Matrix<T, Dynamic, 1> term_dij_no_entropy = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    for(size_t i = 0 ; i < dij.rows() ; i++)
    {
        for(size_t j = 0 ; j < dij.cols() ; j++)
        {
            term_dij_no_entropy.coeffRef(i) += dij.coeff(i,j) * ( phi_FEM(j)-phi_FEM(i) ) ;
        }
    }
    */
    
    ///********* RESOLUTION OF THE SYSTEM: **********//
   
    
    // RESOLUTION FIRST ORDER
    Matrix<T, Dynamic, 1> phi_L = phi_FEM - dt * conv_global.cwiseQuotient(global_lumped_mass)  + dt * term_dij_no_entropy.cwiseQuotient(global_lumped_mass);
  
    //std::cout<<'\n'<<"phi_L : "<<'\n'<<phi_L<<std::endl;
    
    
    // RESOLUTION HIGH ORDER -> NO MIN MAX PRINCIPLE PRESERVING
    Matrix<T, Dynamic, 1> b_phiH = mass_phi_old - dt * conv_global + dt * term_dij ;
    Matrix<T, Dynamic, 1> phi_H = solver_global_mass.solve(b_phiH);
    
    Matrix<T, Dynamic, 1> b_phiE = mass_phi_old - dt * conv_global + dt * term_dij_E ;
    Matrix<T, Dynamic, 1> phi_E = solver_global_mass.solve(b_phiE);
    
    
    // CHECKING OF phi_h AND phi_l
    //Matrix<T, Dynamic, 1> check0 = global_lumped_mass.cwiseProduct( (phi_L - phi_FEM)/dt ) + conv_global - term_dij_no_entropy;
    //std::cout<<'\n'<<"CHECK phi_L is "<<'\n'<<check0<<std::endl;
    
    //Matrix<T, Dynamic, 1> check1 = global_mass*( (phi_H - phi_FEM)/dt ) + conv_global - term_dij;
    //std::cout<<'\n'<<"CHECK phi_H is "<<'\n'<<check1<<std::endl;
    
    
    // EXTENSION: MAXIMUM PRINCIPLE PRESERVING
    
    Matrix<T, Dynamic, 1> delta_phi = phi_H - phi_FEM;
    //std::cout<<'\n'<<"delta_phi "<<'\n'<<delta_phi<<std::endl;
    
    Matrix<T, Dynamic, 1> f_i = f_ij_creator_SPARSE( global_lumped_mass , global_mass , delta_phi , dt , dij , dC_ij , phi_FEM , S_i );
    
    // CHECKING PHI
    /*
    auto check_phi = phi_H - phi_L - f_i.cwiseQuotient(global_lumped_mass) ;//+dt*Vec_One ;
    
    Matrix<T, Dynamic, 1>  phi_H2 = phi_L + f_i.cwiseQuotient(global_lumped_mass);
    
    std::cout<<'\n'<<"check phi "<<'\n'<<check_phi<<std::endl;
    std::cout<<'\n'<<"check phi_h: "<<'\n'<<phi_H - phi_H2<<std::endl;
    */
    
    
    // CORRECTION TERM
    Matrix<T, Dynamic, 1> correction_fi = alfaf_ij_creator_SPARSE( global_lumped_mass , global_mass , delta_phi , phi_L , dt , dij , dC_ij , phi_FEM , S_i );
    Matrix<T, Dynamic, 1>  phi_new = phi_L + correction_fi.cwiseQuotient(global_lumped_mass);
    
    
    
    
    // IMPOSITION DIRICHLET BOUNDARY CONDITIONS
    size_t counter_dir = 0 ;
    for (const auto& dir_elem : fe_data.Dirichlet_boundary )
    {
        if(dir_elem){
            phi_L(counter_dir) = phi_FEM(counter_dir) ;
            phi_H(counter_dir) = phi_FEM(counter_dir) ;
            phi_new(counter_dir) = phi_FEM(counter_dir) ;
            phi_E(counter_dir) = phi_FEM(counter_dir) ;
        }
        counter_dir++ ;
    }
     
    
    //std::cout<<"phi_L - phi_H"<<'\n'<<phi_L - phi_H<<'\n'<<'\n'<<'\n'<<std::endl;
    //std::cout<<"phi_L - phi_new"<<'\n'<<phi_L - phi_new<<'\n'<<'\n'<<'\n'<<std::endl;
    //std::cout<<"phi_H - phi_new"<<'\n'<<phi_H - phi_new<<'\n'<<'\n'<<'\n'<<std::endl;
    
    
    
    tc.toc();
    std::cout << bold << yellow << "FEM method, time resolution: " << tc << " seconds" << reset << std::endl;
    
    
    
    
    
    /// PLOTTING SOLUTION (GNUPLOT) + SAVING FOR HHO (MISCHIATO PER POTERE PLOTTARE ENTRAMBE).
    postprocess_output<double> postoutput5;
    auto test_phi_h = std::make_shared< gnuplot_output_object<double> >("phi_h.dat");
    auto test_phi_l = std::make_shared< gnuplot_output_object<double> >("phi_l.dat");
    auto test_phi_new = std::make_shared< gnuplot_output_object<double> >("phi_new.dat");
    
    // SAVING AND UPLOAD phi_L  INTO CLASS projected_level_set
    
   /*
   

    phi.sol_FEM = phi_L ;
    
    std::cout<<"USING LOW ORDER SOLUTION PHI_L. Also vertices uploading."<<std::endl;
    //phi.converting_into_HHO_formulation(phi_L);
    phi.converting_into_HHO_formulation(phi_L);
    //std::cout<<"phi max "<<phi.sol_FEM.maxCoeff()<<'\n';
    //std::cout<<"phi min "<<phi.sol_FEM.minCoeff()<<'\n';
    //std::cout<<"SAVED phi max "<<phi.phi_max<<'\n';
    //std::cout<<"SAVED phi min "<<phi.phi_min<<std::endl;
    
    phi.coefficients_inverse_mapping();
    //std::cout<<"phi max "<<phi.sol_FEM.maxCoeff()<<'\n';
    //std::cout<<"phi min "<<phi.sol_FEM.minCoeff()<<'\n';
    //std::cout<<"SAVED phi max "<<phi.phi_max<<'\n';
    //std::cout<<"SAVED phi min "<<phi.phi_min<<std::endl;
    
    //phi.set_max_min(); // SETTING OF THE NEW MAX AND MIN FOR NEXT TRANSPORT PROBLEM
    //std::cout<<"phi max "<<phi.sol_FEM.maxCoeff()<<'\n';
    //std::cout<<"phi min "<<phi.sol_FEM.minCoeff()<<'\n';
    //std::cout<<"SAVED phi max "<<phi.phi_max<<'\n';
    //std::cout<<"SAVED phi min "<<phi.phi_min<<std::endl;
    
    for(auto& cl :msh.cells)
    {
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree );
        for (auto pt : pts){
            test_phi_l->add_data( pt ,phi(pt, msh , cl ) );
        }
    }
    
     */
    
    /*
   
 
     
    phi.sol_FEM = phi_E ;
       
    std::cout<<"CONVERTING phi_E in HHO formulation. Also vertices uploading."<<std::endl;
    phi.converting_into_HHO_formulation(phi_E);
       
       
    phi.coefficients_inverse_mapping();
    //phi.set_max_min(); // SETTING OF THE NEW MAX AND MIN FOR NEXT TRANSPORT PROBLEM
       
    
   
    
  
    // SAVING AND UPLOAD phi_L  INTO CLASS projected_level_set
    phi.sol_FEM = phi_H ;
       
    
    std::cout<<"CONVERTING phi_L in HHO formulation. Also vertices uploading."<<std::endl;
    //phi.converting_into_HHO_formulation(phi_L);
    phi.converting_into_HHO_formulation(phi_H);
       
       
    phi.coefficients_inverse_mapping();
    //phi.set_max_min(); // SETTING OF THE NEW MAX AND MIN FOR NEXT TRANSPORT PROBLEM

    for(auto& cl :msh.cells)
    {
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree );
        for (auto pt : pts){
            test_phi_h->add_data( pt ,phi(pt, msh , cl ) );
        }
    }
    
    
     */
 
   
    // SAVING AND UPLOAD phi_L  INTO CLASS projected_level_set
    phi.sol_FEM = phi_new ;
    
    std::cout<<"USING HIGH ORDER SOLUTION PHI_HP. Also vertices uploading."<<std::endl;
    //phi.converting_into_HHO_formulation(phi_L);
    phi.converting_into_HHO_formulation(phi_new);
    
    
    phi.coefficients_inverse_mapping();
    //phi.set_max_min(); // SETTING OF THE NEW MAX AND MIN FOR NEXT TRANSPORT PROBLEM
    
    
    
    
    for(auto& cl :msh.cells)
    {
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree );
        for (auto pt : pts){
            //std::cout<<pt<<std::endl;
            //test_phi_h->add_data( pt , phi.sol_HHO(iii , counter_cl ) );
            test_phi_new->add_data( pt , phi(pt, msh , cl ) );
        }
    }
    
  
    
    /// PLOTTING SOLUTION (GNUPLOT)
    /*
    postprocess_output<double> postoutput5;
    auto test_phi_h = std::make_shared< gnuplot_output_object<double> >("phi_h.dat");
    auto test_phi_l = std::make_shared< gnuplot_output_object<double> >("phi_l.dat");
    //auto test_phi_new = std::make_shared< gnuplot_output_object<double> >("phi_new.dat");
   
    size_t counter_cl = 0;
    for(auto& cl :msh.cells)
    {
        auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, degree );
        size_t iii = 0;
        for (auto pt : pts){
            //std::cout<<pt<<std::endl;
            test_phi_h->add_data( pt , phi.sol_HHO(iii , counter_cl ) );
            test_phi_l->add_data( pt , phi.sol_HHO(iii , counter_cl ) );
            iii++;
        }
        counter_cl++;
    }
     */
    postoutput5.add_object(test_phi_h);
    postoutput5.add_object(test_phi_l);
    postoutput5.add_object(test_phi_new);
    postoutput5.write();
    
    std::cout<<yellow<<bold<<"----------- FINE TRANSPORT PROBLEM -----------"<<reset<<std::endl;

    //return phi_tilde;
    
}







template < typename Fonction, typename Mesh, typename Vel_Field , typename T = typename Mesh::coordinate_type >
void run_FEM_levelset(const Mesh & msh, size_t degree, Fonction & phi, const Vel_Field & u , const T& dt , const mesh_init_params<T>& params)
{
    timecounter tc;
    tc.tic();
    //size_t ndof = ( degree+1 ) * ( degree+1 ) ; // degrees of freedom
    const T eps = 1e-14; //constant into entropy
 
    // LOCAL MASS MATRIX ==> CHECK DEGREE == 1!!!
    assert(degree==1);
    
    
    mapping_phi(phi , phi.phi_max , phi.phi_min ); // mapping of phi to  have a phi between 0 and 1
    /// PHI MAPPED INTO 0-1 PLOT
    /*
    postprocess_output<double> postoutput4;
    auto test_phi_mapped = std::make_shared< gnuplot_output_object<double> >("phi_mapped.dat");
    for (auto cl:msh.cells) {
        auto pts = points(msh,cl) ;
        for (auto pt:pts) {
            T value = phi(pt,msh,cl);
            test_phi_mapped->add_data(pt,value);
        }
    }
    postoutput4.add_object(test_phi_mapped);
    postoutput4.write();
    */
    
    // ENTROPY -> for phi with codomain in (0,1)
    non_linear_entropy<T,Fonction,Mesh> E(eps , phi ,msh );
    auto phi_tilde = projection<T,Mesh> ( msh, degree , params );
    
    /// ENTROPY PLOT
    /*
    postprocess_output<double> postoutput6;
    auto cut_entropy = std::make_shared< gnuplot_output_object<double> >("entropy.dat");
    for (auto cl:msh.cells) {
        auto pts = points(msh,cl) ;
        for (auto pt:pts) {
            T value = E(pt,cl);
            cut_entropy->add_data(pt,value);
        }
    }
    postoutput6.add_object(cut_entropy);
    postoutput6.write();
    */

    // Global Matrix and Global Vector Definitions:
    size_t dim = msh.nodes.size();
    // Lumped mass vector
    Matrix<T, Dynamic, 1> global_lumped_mass = Eigen::Matrix<T, Dynamic, 1>::Zero( dim, 1 );
    // Mass matrix
    Matrix<T, Dynamic, Dynamic> global_mass = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim );
    // C_ij,1 matrix
    Matrix<T, Dynamic, Dynamic> global_cij_x = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim );
    // C_ij,2 matrix
    Matrix<T, Dynamic, Dynamic> global_cij_y = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim );
    // Convection term (vector)
    Matrix<T, Dynamic, 1> conv_global = Eigen::Matrix<T, Dynamic, 1>::Zero( dim, 1 );
    // D_ij term (vector)
    Matrix<T, Dynamic, 1> term_dij = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1 );
    // N_ij matrix
    std::pair<Matrix<T, Dynamic, Dynamic>,Matrix<T, Dynamic, Dynamic>> nij = std::make_pair(Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim ) , Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim ) ) ;
    // N_ji matrix
    std::pair<Matrix<T, Dynamic, Dynamic>,Matrix<T, Dynamic, Dynamic>> nij_transpose = std::make_pair(Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim ) , Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim ) ) ;
    // ||c_ij||^2
    Matrix<T, Dynamic, Dynamic> cij_square = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    // ||c_ji||^2
    Matrix<T, Dynamic, Dynamic> cji_square = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    
    // One vector
    Matrix<T, Dynamic, 1> Vec_One = Eigen::Matrix<T, Dynamic, 1>::Ones( dim, 1 );
    // One matrix
    Matrix<T, Dynamic, Dynamic> Mat_One = Eigen::Matrix<T, Dynamic, Dynamic>::Ones( dim, dim );
    
    
    // Boundary Nodes = Vertices (in Q1)
    auto bdry_nodes = boundary_nodes_function( phi.Nx , phi.Ny );

    
    // The Support of Global Shape Function: list of neighborood nodes for each node i
    std::vector<std::set<size_t>> S_i(dim);
    timecounter tc1;
    tc1.tic();
    for (auto cl : msh.cells)
    {
        
        
        auto nodes_position = nodes(msh, cl); // node_pos.ptid to have just the number
        
        // In basis (mesh creation) the order is
        /// 2 - - 3
        /// |       |
        /// 0 - - 1
        
        // In nodes position instead the order is
        /// 3 - - 2
        /// |       |
        /// 0 - - 1
        // I swap last two elements to have the first ordering
        iter_swap(nodes_position.end()-1 , nodes_position.end()-2 );
       
        // Support of global shape function: list of close nodes for each node i
        supporting_nodes( S_i , nodes_position );
        
        /*
        // CALCULATION OF THE SIZE + PLOTTING
        size_t size_supp_nodes = 0;
        std::cout<<"Supporting nodes:"<<std::endl;
        size_t jjjj = 0;
        for (auto& i:S_i) {
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
        
        
        // Local mass matrix
        Matrix<T, Dynamic, Dynamic> mass = make_lagrange_local_mass_matrix(msh, cl, degree);
        
        // Assembling into global matrix
        //global_update2( mass , global_mass , nodes_position2 );
        global_update( mass , global_mass , nodes_position );
        
        // Local c_ij = b_i nabla(b_j)
        auto cij = make_local_cij_matrix(msh, cl, degree);
        
        // Assembling into global matrix
        global_update( cij.first , global_cij_x , nodes_position );
        global_update( cij.second , global_cij_y , nodes_position );
        
    // USEFUL COMMAND FOR NORMS of a Eigen::matrix A:
    // A.template lpNorm<2>()
    // A.norm()
        
    }
    
    // LUMPED MASS VECTOR (i.e. vector = diagonal of a matrix)
    sum_Si(S_i, global_mass , Vec_One , global_lumped_mass );
    
    
    
    // CHECKING OF MASS MATRICES' PROPERTIES-> OK, perfect precision 1e-19
    /*
    std::cout<<"First checking: Lumped Mass"<<std::endl;
    for(size_t i = 0; i<global_lumped_mass.rows();i++)
        if(global_lumped_mass(i)<0)
            std::cout<<"PROBLEM into lumped mass"<<std::endl;
    
    Matrix<T, Dynamic, Dynamic> mass_check = - global_mass;
    for(size_t i = 0 ; i < mass_check.rows() ; i++){
        mass_check(i,i) += global_lumped_mass(i);
        std::cout<<(mass_check.row(i)).sum()<<std::endl;
    }
    */
    
    
    
    /// CHECK OF THE SUM PROPERTY OF CIJ
    /*
     Matrix<T, Dynamic, 1> sum0 = Eigen::Matrix<T, Dynamic, 1>::Zero(global_cij_x.rows(), 1);
     Matrix<T, Dynamic, 1> sum1 = Eigen::Matrix<T, Dynamic, 1>::Zero(global_cij_x.rows(), 1);
    
    for (size_t k = 0; k<global_cij_x.cols(); k++)
    {
        sum0 += (global_cij_x).col(k);
        sum1 += (global_cij_y).col(k);
    }
    
    for (size_t i=0; i<global_cij_y.rows(); i++) {
         std::cout<<"The sum is "<<sum1(i)<<" and "<<sum0(i)<<std::endl;
    }
    
    size_t counter_sum = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
           sum0( counter_sum ) += global_cij_x( counter_sum , elem );
           sum1( counter_sum ) += global_cij_y( counter_sum , elem );
           if(elem==counter_sum)
               std::cout<<"In ("<<counter_sum<<" , "<<elem<<" ), c^0 = "<<global_cij_x( counter_sum , elem )<<" and c1 = "<<global_cij_y( counter_sum , elem )<<std::endl;
           else
                std::cout<<"In ("<<counter_sum<<" , "<<elem<<" ), c^0_ij + c^0_ji = "<<global_cij_x( counter_sum , elem ) + global_cij_x( elem , counter_sum )<<" and c^1_ij + c^1_ji = "<<global_cij_y( counter_sum , elem ) + global_cij_y( elem , counter_sum ) <<std::endl;
        }
        counter_sum++;
    }
    //std::cout<<"The sum0 is "<<'\n'<<sum0<<" and sum1 "<<'\n'<<sum1<<std::endl;
    */
     
   

    // FLUX TERM
    Matrix<T, Dynamic, 1> flux0 = (u.vertices.first).cwiseProduct(phi.vertices) ;
    Matrix<T, Dynamic, 1> flux1 = (u.vertices.second).cwiseProduct(phi.vertices) ;
    
    
    
    // RESOLUTION OF phi_tilde GLOBALE     ( with cij(phi_j - phi_i) )
    
    Matrix<T, Dynamic, 1> vec1 = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    sum_Si(S_i , global_mass , phi.vertices , vec1);
    
    Matrix<T, Dynamic, 1> vec2 = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    //Matrix<T, Dynamic, 1> vec2_bis = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    averaged_sum_Si(S_i , global_cij_x , flux0 , vec2);
    averaged_sum_Si(S_i , global_cij_y , flux1 , vec2);
    //sum_Si(S_i , global_cij_x , flux0 , vec2_bis);
    //sum_Si(S_i , global_cij_y , flux1 , vec2_bis);
    //std::cout<<"diff vec 2 = "<<'\n'<<vec2-vec2_bis<<std::endl;
    /// THERE IS A DIFFERENCE OF  <1E-18 DEPENDING IF I CONSIDER OR NOT THE AVERAGED SUM
    
    Matrix<T, Dynamic, 1> b = vec1 - dt*vec2.cwiseQuotient(global_lumped_mass);
    
    /// OLD IMPLEMENTATION
    //Matrix<T, Dynamic, 1> prova_FAST = global_mass.completeOrthogonalDecomposition().solve(b);
    //phi_tilde.vertices = global_mass.HouseholderQR().solve(b);
    timecounter tc00;
    tc00.tic();
    LLT<Matrix<T, Dynamic, Dynamic > > llt( global_mass );
    phi_tilde.vertices = llt.solve(b);
    tc00.toc();
    
    
    if (llt.info() != Success)
    {
        std::cout<<"not positive"<<std::endl;
        assert(0);
    }
    //std::cout<<"ERRORE FAST= "<<'\n'<<phi_tilde.vertices - prova_FAST <<std::endl;
   
    phi_tilde.converting_into_HHO_formulation( phi_tilde.vertices );
    std::cout << bold << yellow << "LLT RESOLUTION TIME: " << tc00 << " seconds" << reset << std::endl;
    
    
    //std::cout<<"phi_tilde= "<<'\n'<<phi_tilde.vertices<<std::endl;
    /*
    postprocess_output<double> postoutput3;
    typedef typename Mesh::point_type       point_type;
    point<double,2> node;
    auto test_phi_tilde = std::make_shared< gnuplot_output_object<double> >("phi_tilde.dat");
    for (size_t i = 0; i<phi_tilde.vertices.rows(); i++) {
        node = msh.points.at(i);
        test_phi_tilde->add_data(node,phi_tilde.vertices(i));
    }
    postoutput3.add_object(test_phi_tilde);
    postoutput3.write();
    */
    
    
    // Term R_i^n
    Matrix<T, Dynamic, 1> Emax_global = Eigen::Matrix<T, Dynamic, 1>::Ones(dim, 1);
    Matrix<T, Dynamic, 1> Emin_global = Eigen::Matrix<T, Dynamic, 1>::Ones(dim, 1);
    Emax_global *= -1e20;
    Emin_global *= 1e20;
    Matrix<T, Dynamic, 1> R_i = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    for( auto& cl: msh.cells )
    {
        r_i_calculator( msh , cl , E , phi_tilde , phi , dt , u ,S_i , Emax_global , Emin_global , R_i);
    }
    //std::cout<<"Emax_global - Emin_global = "<<'\n'<<Emax_global-Emin_global<<std::endl;
    R_i = R_i.cwiseQuotient( Emax_global - Emin_global );
    //std::cout<<"Ri = "<<'\n'<<R_i<<std::endl;
    
    
    
    // CONVOLUTION TERM
    //sum_Si(S_i, global_cij_x , flux0 , conv_global );
    //sum_Si(S_i, global_cij_y , flux1 , conv_global );
    averaged_sum_Si(S_i, global_cij_x , flux0 , conv_global );
    averaged_sum_Si(S_i, global_cij_y , flux1 , conv_global );
    // OLD IMPLEMENTATION
    //auto conv_global2 = global_cij_x*( flux0 ) + global_cij_y*( flux1 ) ;
    
    // Norm of c_ij
    moltiplication_Si( S_i , global_cij_x , global_cij_x , cij_square );
    moltiplication_Si( S_i , global_cij_y , global_cij_y , cij_square );
    Matrix<T, Dynamic, Dynamic> cij_norm = cij_square.cwiseSqrt();
    
    // Matrix n_ij
    division_Si( S_i , global_cij_x , cij_norm , nij.first );
    division_Si( S_i , global_cij_y , cij_norm , nij.second );
    // OLD IMPLEMENTATION
    //auto nij2 = std::make_pair( global_cij_x.cwiseQuotient(cij_norm) , (global_cij_y).cwiseQuotient( cij_norm ) ) ;
    
    // NOTICE:
    /// Normal Velocity calculation IS EQUIVALENT TO f', that is EQUIVALENT to lambda_r , lambda_l.  Moreover I don't need the calculation of the normal flux. Since then it has to be derived, it is enough to calculate f' = u dot n
     
    // Normal velocity
    Matrix<T, Dynamic, Dynamic> normal_vel = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    size_t counter_row = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            auto u_max_0 = std::max( u.vertices.first(counter_row),u.vertices.first(elem) );
            auto u_max_1 = std::max( u.vertices.second(counter_row),u.vertices.second(elem) );
            normal_vel(counter_row,elem) = u_max_0 * nij.first(counter_row,elem) + u_max_1*nij.second(counter_row,elem);
        }
        counter_row++;
    }
     
    
    
   
    // C_ji matrix
    std::pair<Matrix<T, Dynamic, Dynamic>,Matrix<T, Dynamic, Dynamic>> cji_global = std::make_pair(global_cij_x.adjoint() , global_cij_y.adjoint() );
    
    // Norm of c_ji -> i.e. c_ij transposed
    moltiplication_Si( S_i , cji_global.first , cji_global.first , cji_square );
    moltiplication_Si( S_i , cji_global.second , cji_global.second , cji_square );
    // CHECK IF WORKS BETTER NOW (  NO TRANSPOSE )
    //moltiplication_Si_T( S_i , cji_global.first , cji_global.first , cji_square );
    //moltiplication_Si_T( S_i , cji_global.second , cji_global.second , cji_square );
    
    //C_ji Norm
    Matrix<T, Dynamic, Dynamic> cji_norm = cji_square.cwiseSqrt();
    //auto cji_square2 = (cji_global.first).cwiseProduct(cji_global.first) + (cji_global.second).cwiseProduct(cji_global.second);
    //Matrix<T, Dynamic, Dynamic> cji_norm2 = cji_square2.cwiseSqrt();
    
   
    // Matrix n_ij (TRANSPOSED)
    division_Si( S_i , cji_global.first , cji_norm , nij_transpose.first );
    division_Si( S_i , cji_global.second , cji_norm , nij_transpose.second );
     // CHECK IF WORKS BETTER NOW (  NO TRANSPOSE )
    //division_Si_T( S_i , cji_global.first , cji_norm , nij_transpose.first );
    //division_Si_T( S_i , cji_global.second , cji_norm , nij_transpose.second );
    //auto  nij_transposex2 = cji_global.first.cwiseQuotient( cji_norm2 );
    //auto  nij_transposey2 = cji_global.second.cwiseQuotient( cji_norm2 );
    
    // Normal velocity (TRANSPOSED)
    Matrix<T, Dynamic, Dynamic> normal_vel_adj = Eigen::Matrix<T,Dynamic,Dynamic>::Zero(dim, dim);
    //Matrix<T, Dynamic, Dynamic> normal_vel_adj2 = Eigen::Matrix<T,Dynamic,Dynamic>::Zero(dim, dim);
    /*
    for(size_t i = 0 ; i< normal_vel_adj2.rows() ; i++)
    {
        for(size_t j = 0 ; j< normal_vel_adj2.cols() ; j++)
        {
            auto u_max_0 = std::max( u.vertices.first(i),u.vertices.first(j) );
            auto u_max_1 = std::max( u.vertices.second(i),u.vertices.second(j) );
            //normal_vel(counter_row,elem) = u_max_0 * nij.first(counter_row,elem) + u_max_1*nij.second(counter_row,elem);
            // CHECK IF WORKS BETTER NOW (  NO TRANSPOSE )
            normal_vel_adj2(i,j) = u_max_0 * nij_transposex2(i,j) + u_max_1*nij_transposey2(i,j);
        }
    }
    
    */

    counter_row = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            auto u_max_0 = std::max( u.vertices.first(elem),u.vertices.first(counter_row) );
            auto u_max_1 = std::max( u.vertices.second(elem),u.vertices.second(counter_row) );
            //normal_vel_adj(elem,counter_row) = u_max_0 * nij.first(elem,counter_row) + u_max_1*nij.second(elem,counter_row);
            // CHECK IF WORKS BETTER NOW (  NO TRANSPOSE )
            normal_vel_adj(counter_row , elem) = u_max_0 * nij_transpose.first(counter_row,elem) + u_max_1*nij_transpose.second(counter_row,elem);
        }
        counter_row++;
    }
    
    
    // Matrix Dij calculation
    Matrix<T, Dynamic, Dynamic> lambda_max = normal_vel.cwiseAbs() ;  // o lambda_max=normal_vel ?
    Matrix<T, Dynamic, Dynamic> tmp = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    moltiplication_Si( S_i , lambda_max , cij_norm , tmp );
    
    Matrix<T, Dynamic, Dynamic> lambda_max_adj=normal_vel_adj.cwiseAbs();// o lambda_max=normal_vel?
    Matrix<T, Dynamic, Dynamic> tmp_adj = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    moltiplication_Si( S_i , lambda_max_adj , cji_norm , tmp_adj );
    // CHECK IF WORKS BETTER NOW (  NO TRANSPOSE )
    //moltiplication_Si_T( S_i , lambda_max_adj , cji_norm , tmp_adj );
    //Matrix<T, Dynamic, Dynamic> lambda_max_adj2=normal_vel_adj2.cwiseAbs();
    //auto tmp_adj2 = lambda_max_adj.cwiseProduct( cji_norm2 ) ;
    
    Matrix<T, Dynamic, Dynamic> dij = tmp.cwiseMax(tmp_adj);
    
    //Matrix<T, Dynamic, Dynamic> dij2 = tmp.cwiseMax(tmp_adj2);
    
    //std::cout<<"Dij: "<<'\n'<<dij<<std::endl;
    
    /*
      /// The  error is < 1e-15, at max
    std::cout<<"D_ij check: "<<std::endl;
    for(size_t i = 0; i<tmp.rows();i++){
        for(size_t j = i+1; j<tmp.rows();j++){
            if(std::abs(dij(i,j)-dij(j,i))>1e-15)
            std::cout<<dij(i,j)-dij(j,i);
        }
        std::cout<<'\n';
    }
    std::cout<<std::endl;
    */
   
    
    tc1.toc();
    std::cout << bold << yellow << "Solving time method original: " << tc1 << " seconds" << reset << std::endl;
    
    timecounter tc2;
    tc2.tic();
       
    

    
    // Minimum between d_ij and R_i
    T c_e = 1.;
    Matrix<T, Dynamic, Dynamic> dE_ij = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    Matrix<T, Dynamic, Dynamic> phi_ij = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    counter_row = 0;
   
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            if(elem!=counter_row)
            {
            auto R_i_j = c_e * std::max( std::abs( R_i(counter_row) ) , std::abs( R_i(elem) ) );
            dE_ij( counter_row , elem ) = std::min( dij( counter_row , elem ), R_i_j );
            //std::cout<<"R_i_j "<<R_i_j<<" and dij"<<dij( counter_row , elem )<<std::endl;
            }
            phi_ij( counter_row , elem ) = 0.5*( phi.vertices(counter_row) + phi.vertices(elem) );
        }
        counter_row++;
    }
    //std::cout<<"dE_ij: "<<'\n'<<dE_ij<<std::endl;
    //std::cout<<"phi_ij"<<'\n'<<phi_ij<<std::endl;
    T c_comp = 1.;
    Matrix<T, Dynamic, Dynamic> dC_ij = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    Matrix<T, Dynamic, Dynamic> tmp0 = phi_ij - phi_ij.cwiseProduct(phi_ij);
    //Matrix<T, Dynamic, Dynamic> tmp0 = (phi_ij-phi.phi_min*Mat_One).cwiseProduct(phi.phi_max*Mat_One-phi_ij);
    //std::cout<<"tmp0: "<<'\n'<<tmp0<<std::endl;
    positive_part( tmp0 );
    
    
   // Matrix<T, Dynamic, Dynamic> tmp_dC0 = positive_part( tmp0 );

   
    Matrix<T, Dynamic, Dynamic> tmp_dC1 =  Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
           // std::cout<<"In ("<<counter<<" , "<<elem<<") tmp_dC0 = "<<tmp_dC0(counter,elem)<<" and phi.ver diff "<<std::abs(phi.vertices(counter) - phi.vertices(elem))<<std::endl;
            if( std::abs(phi.vertices(counter) - phi.vertices(elem))>1e-15 )
            tmp_dC1(counter,elem) = tmp0(counter,elem)/( std::abs(phi.vertices(counter) - phi.vertices(elem)) );
        }
        counter++;
    }
    //std::cout<<'\n'<<"tmp_dC1: "<<'\n'<<tmp_dC1<<std::endl;
    Matrix<T, Dynamic, Dynamic> tmp1 = Mat_One-c_comp*tmp_dC1;
    
    //std::cout<<'\n'<<"tmp1: "<<'\n'<<tmp1<<std::endl;
    positive_part( tmp1  );
    //std::cout<<'\n'<<"tmp1 POSTIVIVE: "<<'\n'<<tmp1<<std::endl;
    
    moltiplication_Si( S_i , dE_ij , tmp1 , dC_ij );
   
    
    for(size_t i = 0 ; i < dC_ij.rows() ; i++)
    {
        dC_ij(i,i) = 0;
        dij(i,i) = 0;
        dC_ij(i,i) = -dC_ij.row(i).sum();
        dij(i,i) = -dij.row(i).sum();
    }
    //dC_ij = dE_ij.cwiseProduct( positive_part( tmp1  ) );
    //   std::cout<<'\n'<<"dC_ij: "<<'\n'<<dC_ij<<std::endl;
    //   std::cout<<'\n'<<"dij: "<<'\n'<<dij<<std::endl;
    
   
    //std::cout<<"dC_ij: "<<'\n'<<dC_ij<<std::endl;
    
    
    /*
    counter_row = 0;
    std::cout<<"DC_ij check symmetry: "<<std::endl;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            std::cout<<dC_ij(counter_row,elem) - dC_ij(elem,counter_row)<<std::endl;
        }
        counter_row++;
        std::cout<<'\n';
    }
    std::cout<<std::endl;
    
    
    counter_row = 0;
    std::cout<<"D_ij check symmetry "<<std::endl;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            std::cout<<dij(counter_row,elem) - dij(elem,counter_row)<<std::endl;
        }
        counter_row++;
        std::cout<<'\n';
    }
    std::cout<<std::endl;
    */
    
    
    // Term D_ij(phi_j - phi_i )
    averaged_sum_Si(S_i, dC_ij , phi.vertices , term_dij );
    //std::cout<<"term_dij : "<<'\n'<<term_dij<<std::endl;
    
    
    //averaged_sum_Si(S_i, dE_ij , phi.vertices , term_dij );
    Matrix<T, Dynamic, 1> term_dij_no_entropy = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1 );
    // VERSION WITHOUT ENTROPY CORRECTION
    averaged_sum_Si(S_i, dij , phi.vertices , term_dij_no_entropy );
    
    //std::cout<<"term_dij_no_entropy : "<<'\n'<<term_dij_no_entropy<<std::endl;
    
    // M*phi^n
    Matrix<T, Dynamic, 1> mass_phi_old = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    sum_Si( S_i , global_mass , phi.vertices , mass_phi_old );
    
    
    
  /*
    // **************** OLD IMPLEMENTATION **************** //
    
   for (size_t i = 0; i<global_lumped_mass.rows(); i++) {
          global_lumped_mass(i) = global_mass.row(i).sum() ;
      }
   
   
    Matrix<T, Dynamic, Dynamic> normal_vel2 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim1, dim2);
    Matrix<T, Dynamic, Dynamic> normal_vel_adj2 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim2, dim1);
    
    //normal velocity
    for (size_t i = 0; i < dim1; i++)
    {
        for (size_t j = 0; j < dim2; j++)
        {
            auto u_max_0 = std::max( u.vertices.first(i),u.vertices.first(j) );
            auto u_max_1 = std::max( u.vertices.second(i),u.vertices.second(j) );
            normal_vel2( i , j ) = u_max_0 * nij.first(i,j) + u_max_1*nij.second(i,j);
            //}
        }
           
    }
    
     //normal velocity (transposed)
    auto nij_transpose2 = std::make_pair( ( global_cij_x.transpose() ).cwiseQuotient(cji_norm) ,                                    ( global_cij_y. transpose() ).cwiseQuotient( cji_norm ) ) ;
    
    for (size_t i = 0; i < dim2; i++)
    {
        for (size_t j = 0; j < dim1; j++)
        {
            auto u_max_0 = std::max( u.vertices.first(i),u.vertices.first(j) );
            auto u_max_1 = std::max( u.vertices.second(i),u.vertices.second(j) );
            normal_vel_adj2( i , j ) = u_max_0 * nij_transpose2.first(i,j) + u_max_1*nij_transpose2.second(i,j);
        }
    }

    Matrix<T, Dynamic, Dynamic> lambda_max2 = normal_vel2.cwiseAbs() ;  // o lambda_max=normal_vel ?
    auto tmp2 = lambda_max2.cwiseProduct( cij_norm ) ;
    
    Matrix<T, Dynamic, Dynamic> lambda_max_adj2 = normal_vel_adj2.cwiseAbs() ;
    auto tmp_adj2 = lambda_max_adj2.cwiseProduct( cji_norm ) ;
  
    auto dij2 = tmp2.cwiseMax(tmp_adj2);
    Matrix<T, Dynamic, 1> term_dij2 = Eigen::Matrix<T, Dynamic, 1>::Zero(dim2, 1);
    /// CHECK OUT THE FUNCTION: make_dij_vector
    // auto phi_old = (phi.values_bis).col(counter);
    // auto last_term = make_dij_vector(msh , dij, phi_old );

    for (size_t i = 0; i<dij.rows(); i++)
    {
        auto diff =  phi.vertices - phi.vertices(i)*Vec_One;
        term_dij2(i) = ( dij2.row(i) ).dot( diff )  ;
    }
    std::cout<<"diff term dij: "<<term_dij-term_dij2<<std::endl;
   
    // ****************END OF OLD IMPLEMENTATION **************** //
    
 */
    
    tc2.toc();
    std::cout << bold << yellow << "Solving time method2: " << tc2 << " seconds" << reset << std::endl;
    
    
    timecounter tc3;
    tc3.tic();
       
    ///********* RESOLUTION OF THE SYSTEM: **********//
    //tc.tic();
    Matrix<T, Dynamic, 1> phi_L = solveFEM(global_lumped_mass , conv_global , term_dij_no_entropy , phi.vertices , dt , bdry_nodes); // VERSION WITHOUT ENTROPY CORRECTION
    tc3.toc();
    std::cout << bold << yellow << "Solving time ORIGINAL METHOD: " << tc3 << " seconds" << reset << std::endl;
    
    timecounter tc4;
    tc4.tic();

     Matrix<T, Dynamic, 1> phi_H = solveFEM_Entropic_FAST(llt , conv_global , term_dij , mass_phi_old , dt );
    
    //Matrix<T, Dynamic, 1> phi_H = solveFEM_Entropic(global_mass , conv_global , term_dij , mass_phi_old , dt );
     //std::cout<<'\n'<<"phi_H -phi_H_fast : "<<'\n'<<phi_H-phi_H_prova<<std::endl;
    tc4.toc();
    
    std::cout << bold << yellow << "Solving time CORRECTED METHOD: " << tc4 << " seconds" << reset << std::endl;
    
    
    
    // std::cout<<'\n'<<"phi_L : "<<'\n'<<phi_L<<std::endl;
    
    
    // CHECKING phi_h and phi_l
    Matrix<T, Dynamic, 1> zero_vec = Eigen::Matrix<T, Dynamic, 1>::Zero(phi.vertices.rows(), 1);
   // checking_phi_l( global_lumped_mass , global_cij_x , global_cij_y ,dij , phi.vertices ,phi_L,dt, S_i); // NO, DA MODIFICARE
    
    //checking_phi_lBIS( global_lumped_mass , conv_global , term_dij_no_entropy , phi.vertices , phi_L , dt );
    
    //checking_phi_h( global_mass , global_cij_x , global_cij_y , dC_ij , phi.vertices  , phi_H ,dt,S_i); // NO, DA MODIFICARE
    
    //checking_phi_hBIS( global_mass , conv_global , term_dij , phi.vertices , phi_H , dt );
    
   
    
    
    

    // EXTENSION: MAXIMUM PRINCIPLE PRESERVING
    
    Matrix<T, Dynamic, 1> delta_phi = phi_H - phi.vertices;
    //std::cout<<'\n'<<"delta_phi "<<'\n'<<delta_phi<<std::endl;
    
    Matrix<T, Dynamic, 1> f_i = f_ij_creator( global_lumped_mass , global_mass , delta_phi , dt , dij , dC_ij , phi.vertices , S_i );
    
    // CHECKING PHI
    auto check_phi = phi_H - phi_L - f_i.cwiseQuotient(global_lumped_mass) ;//+dt*Vec_One ;
    Matrix<T, Dynamic, 1>  phi_H2 = phi_L + f_i.cwiseQuotient(global_lumped_mass);
    //std::cout<<'\n'<<"check phi "<<'\n'<<check_phi<<std::endl;
   // std::cout<<'\n'<<"check phi_h: "<<'\n'<<phi_H - phi_H2<<std::endl;
    
    // CORRECTION TERM
    Matrix<T, Dynamic, 1> correction_fi = alfaf_ij_creator( global_lumped_mass , global_mass , delta_phi , phi_L , dt , dij , dC_ij , phi.vertices , S_i );
    Matrix<T, Dynamic, 1>  phi_new = phi_L + correction_fi.cwiseQuotient(global_lumped_mass);
    
    //T check_phi2 = ((phi_new - phi_L).cwiseProduct(global_lumped_mass)).sum();
   // std::cout<<'\n'<<"check PHI_NEW - PHI_L = "<<check_phi2<<std::endl;
    
    //T check_phi3 = ((phi_new - phi.vertices).cwiseProduct(global_lumped_mass)).sum();
   // std::cout<<'\n'<<"check PHI_NEW - PHI_OLD = "<<check_phi3<<std::endl;
    
   // T check_phi4 = ((phi_L - phi.vertices).cwiseProduct(global_lumped_mass)).sum();
   // std::cout<<'\n'<<"check PHI_L - PHI_OLD = "<<check_phi4<<std::endl;
    
    //T check_phi5 = ((phi_H - phi.vertices).cwiseProduct(global_lumped_mass)).sum();
    //std::cout<<'\n'<<"check PHI_H - PHI_OLD = "<<check_phi5<<std::endl;
    
    //T check_phi6 = ((phi_H - phi_L).cwiseProduct(global_lumped_mass)).sum();
    //std::cout<<'\n'<<"check PHI_H - PHI_L = "<<check_phi6<<std::endl;
    
    

    // BOUNDARY CONDITION IMPOSITION -->   NO ENTROPY
    for (auto& j : bdry_nodes )
        phi_L(j) = phi.vertices(j) ;
    
    // BOUNDARY CONDITION IMPOSITION -> ENTROPIC SOLUTION
    for (auto& j : bdry_nodes )
        phi_H(j) = phi.vertices(j) ;
    
    // BOUNDARY CONDITION IMPOSITION -> ENTROPIC AND MAX PRESERVING SOLUTION
    for (auto& j : bdry_nodes )
        phi_new(j) = phi.vertices(j) ;
    
    
    // SAVING AND UPLOAD phi_new  INTO CLASS projected_level_set
    phi.converting_into_HHO_formulation(phi_new);
    phi.vertices = phi_new;
    
    // INVERSE MAPPING BACK [0,1] --> [PHI_MIN,PHI_MAX]
    inverse_mapping_phi( phi , phi.phi_max , phi.phi_min );
    tc.toc();
    std::cout << bold << yellow << "FEM method, time resolution: " << tc << " seconds" << reset << std::endl;
    
    /// PLOTTING SOLUTION (GNUPLOT)
    /*
    postprocess_output<double> postoutput5;
    // auto test_phi_h = std::make_shared< gnuplot_output_object<double> >("phi_h.dat");
    // auto test_phi_l = std::make_shared< gnuplot_output_object<double> >("phi_l.dat");
    auto test_phi_new = std::make_shared< gnuplot_output_object<double> >("phi_new.dat");
    
    size_t iii = 0;
    for (auto pt : msh.points) {
      //  test_phi_h->add_data(pt,phi_H(iii) );
      //  test_phi_l->add_data(pt,phi_L(iii) );
        test_phi_new->add_data(pt,phi_new(iii) );
        iii++;
    }
    //   postoutput5.add_object(test_phi_h);
    //  postoutput5.add_object(test_phi_l);
    postoutput5.add_object(test_phi_new);
    postoutput5.write();
    */
    
    std::cout << bold << yellow << "!!ATTENCTION: I'M COMPUTING BOTH phi AND phi_entropic!!" << reset <<std::endl;
    
    
    
    /// COUT OF PHI IN CELLWISE NOTATION
/*
    for (size_t i = 0; i<phi.values_bis.rows(); i++) {
        for (size_t j = 0; j<phi.values_bis.cols(); j++) {
            std::cout<<phi.values_bis(i,j);
        }
        std::cout<<'\n';
    }
    std::cout<<std::endl;
*/
  
    

}






template< typename Mesh , typename Fonction , typename T = typename Mesh::coordinate_type ,typename VEC >
void Lp_space_Tfin_error_FEM( const Fonction& level_set_final , const Fonction& level_set_initial , const Mesh& msh , size_t degree ,double p , VEC& error )
{
    T errorLp = 0.;
    for(auto& cl : msh.cells)
    {
        auto qps = integrate(msh, cl, 2*degree + 2 );
            for (auto& qp : qps )
            {
                auto diff_val = std::abs( level_set_final( qp.first , msh , cl ) - level_set_initial( qp.first , msh , cl ) );
                errorLp += qp.second * pow(diff_val , p) ;
            }
        //std::cout<<"The L^2 error squared in cell "<<offset(msh,cl)<<" is "<< errorL2 <<std::endl;
    }
    //std::cout<<"The L^"<<p<<" error is "<< pow (errorLp , 1.0/p )<<std::endl;
    error.push_back(pow (errorLp , 1.0/p ));
}



template< typename Mesh , typename Fonction , typename T = typename Mesh::coordinate_type >
void Lp_space_Tfin_error_FEM( const Fonction& level_set_final , const Fonction& level_set_initial , const Mesh& msh , size_t degree ,double p )
{
    T errorLp = 0.;
    for(auto& cl : msh.cells)
    {
        auto qps = integrate(msh, cl, 2*degree + 2 );
            for (auto& qp : qps )
            {
                auto diff_val = std::abs( level_set_final( qp.first , msh , cl ) - level_set_initial( qp.first , msh , cl ) );
                errorLp += qp.second * pow(diff_val , p) ;
            }
        //std::cout<<"The L^2 error squared in cell "<<offset(msh,cl)<<" is "<< errorL2 <<std::endl;
    }
    std::cout<<"The L^"<<p<<" error is "<< pow (errorLp , 1.0/p )<<std::endl;
}

template< typename Mesh , typename Fonction , typename T = typename Mesh::coordinate_type >
T Lp_space_error_FEM( const Fonction& level_set_final , const Fonction& level_set_initial , const Mesh& msh , size_t degree , double p )
{
    // L^p in space ; l^q in time
    T errorLp = 0.;
    for(auto& cl : msh.cells)
    {
        auto qps = integrate(msh, cl, 2*degree + 2 ); // what orders?
            for (auto& qp : qps )
            {
                auto diff_val = std::abs( level_set_final( qp.first , msh , cl ) - level_set_initial( qp.first , msh , cl ) );
                errorLp += qp.second * pow(diff_val,p) ;
            }
        //std::cout<<"The L^2 error squared in cell "<<offset(msh,cl)<<" is "<< errorL2 <<std::endl;
    }
    errorLp = pow( errorLp , 1.0/ p );
    //std::cout<<"The L^2 error is "<<sqrt( errorL2 )<<std::endl;
    return errorLp ;
}

template< typename Mesh , typename Fonction , typename T = typename Mesh::coordinate_type >
T W1p_error_FEM( const Fonction& level_set_final , const Fonction& level_set_initial , const Mesh& msh , size_t degree , double p )
{
    T errorH1 = 0.;
    for(auto& cl : msh.cells)
    {
        auto qps = integrate(msh, cl, 2*degree + 2 ); // what orders?
            for (auto& qp : qps )
            {
                auto diff_val0 = std::abs( level_set_final.gradient( qp.first , msh , cl )(0) - level_set_initial.gradient( qp.first , msh , cl )(0) );
                auto diff_val1 = std::abs( level_set_final.gradient( qp.first , msh , cl )(1) - level_set_initial.gradient( qp.first , msh , cl )(1) );
                errorH1 += qp.second * (pow(diff_val0,p) + pow(diff_val1,p)) ;
            }
        //std::cout<<"The L^2 error squared in cell "<<offset(msh,cl)<<" is "<< errorL2 <<std::endl;
    }
    errorH1 = pow( errorH1 , 1.0/p );
    //std::cout<<"The L^2 error is "<<sqrt( errorL2 )<<std::endl;
    return errorH1 ;
}



template< typename Mesh , typename Fonction , typename T = typename Mesh::coordinate_type >
T Linf_error_FEM( const Fonction& level_set_final , const Fonction& level_set_initial , const Mesh& msh , size_t degree )
{
    T errorLinf = ((level_set_final.vertices - level_set_initial.vertices).cwiseAbs()).maxCoeff();
    return errorLinf;

}

template< typename Mesh , typename Fonction , typename T = typename Mesh::coordinate_type >
T W1inf_error_FEM( const Fonction& level_set_final , const Fonction& level_set_initial , const Mesh& msh , size_t degree )
{
    T errorLinf0 = 0 , errorLinf1 = 0;
    
    for(auto& cl : msh.cells)
    {
        auto pts = points(msh,cl);
        for (auto& pt : pts )
        {
            auto diff_val0 = std::abs( level_set_final.gradient( pt , msh , cl )(0) - level_set_initial.gradient( pt , msh , cl )(0) );
            errorLinf0 = std::max(errorLinf0 , diff_val0);
            auto diff_val1 = std::abs( level_set_final.gradient( pt , msh , cl )(1) - level_set_initial.gradient( pt , msh , cl )(1) );
            errorLinf1 = std::max(errorLinf1 , diff_val1);
        }
        
    }
    return errorLinf0 + errorLinf1;

}


template< typename T , typename VeloField>
T time_step_CFL( const VeloField& u , const mesh_init_params<T>& mip , T eps ){
    
    auto h_max = std::max(mip.hx() , mip.hy() );
    auto u_max = u.values_bis.first.template lpNorm<Infinity>() + u.values_bis.second.template lpNorm<Infinity>() ;
    if( std::abs(u_max) < 1e-15 )
        return 1e-8;
    else
        return eps*h_max/u_max ;
}

template< typename T , typename VeloField>
T time_step_CFL_new( const VeloField& u , const mesh_init_params<T>& mip ,T eps)
{
    
    auto h_max = std::max(mip.hx() , mip.hy() );
    auto u_max = u.sol_FEM.first.template lpNorm<Infinity>() + u.sol_FEM.second.template lpNorm<Infinity>() ;
    if( std::abs(u_max) < 1e-15 )
        return 1e-8;
    else
        return eps*h_max/(std::abs(u_max)) ;
}






