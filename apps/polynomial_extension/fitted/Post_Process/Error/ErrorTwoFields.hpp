
#ifndef Error_hpp
#define Error_hpp

void ComputeL2ErrorTwoFields(poly_mesh<double> & msh, hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
std::function<double(const typename poly_mesh<double>::point_type& )> scal_fun,
std::function<std::vector<double>(const typename poly_mesh<double>::point_type& )> flux_fun);


void ComputeL2ErrorTwoFields(poly_mesh<double> & msh, hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
std::function<double(const typename poly_mesh<double>::point_type& )> scal_fun,
                             std::function<std::vector<double>(const typename poly_mesh<double>::point_type& )> flux_fun){
    
    timecounter tc;
    tc.tic();
    
    using RealType = double;
    RealType scalar_l2_error = 0.0;
    RealType flux_l2_error = 0.0;
    size_t cell_i = 0;
    for (auto& cell : msh.cells)
    {
        if(cell_i == 0){
            RealType h = diameter(msh, cell);
            std::cout << green << "h size = " << std::endl << h << std::endl;
        }
        
        size_t cell_scal_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree());
        size_t cell_flux_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree()+1)-1;
        size_t cell_dof = cell_scal_dof+cell_flux_dof;
        
        // scalar evaluation
        {
            Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof+cell_flux_dof, 0, cell_scal_dof, 1);
            Matrix<RealType, Dynamic, Dynamic> mass = make_mass_matrix(msh, cell, hho_di.cell_degree());
            Matrix<RealType, Dynamic, 1> rhs = make_rhs(msh, cell, hho_di.cell_degree(), scal_fun, 2);
            Matrix<RealType, Dynamic, 1> real_dofs = mass.llt().solve(rhs);
            Matrix<RealType, Dynamic, 1> diff = real_dofs - scalar_cell_dof;
            scalar_l2_error += diff.dot(mass*diff);
        }
        
        // flux evaluation
        auto int_rule = integrate(msh, cell, 2*(hho_di.cell_degree()+1));
        cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree()+1);
        Matrix<RealType, Dynamic, 1> flux_cell_dof = x_dof.block(cell_i*cell_dof, 0, cell_flux_dof, 1);
        for (auto & point_pair : int_rule) {
            
            RealType omega = point_pair.second;
            
            auto t_dphi = cell_basis.eval_gradients(point_pair.first);
            Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();
            for (size_t i = 1; i < t_dphi.rows(); i++){
              grad_uh = grad_uh + flux_cell_dof(i-1)*t_dphi.block(i, 0, 1, 2);
            }

            Matrix<RealType, 1, 2> grad_u_exact = Matrix<RealType, 1, 2>::Zero();
            grad_u_exact(0,0) =  flux_fun(point_pair.first)[0];
            grad_u_exact(0,1) =  flux_fun(point_pair.first)[1];
            flux_l2_error += omega * (grad_u_exact - grad_uh).dot(grad_u_exact - grad_uh);
        }
        
        cell_i++;
    }
    
    std::cout << green << "scalar L2-norm error = " << std::endl << std::sqrt(scalar_l2_error) << std::endl;
    std::cout << green << "flux L2-norm error = " << std::endl << std::sqrt(flux_l2_error) << std::endl;
    std::cout << std::endl;
    tc.toc();
    std::cout << bold << cyan << "Error completed: " << tc << " seconds" << reset << std::endl;
    
    
}

#endif
