
#ifndef ComputeEnergyFirstOrder_hpp
#define ComputeEnergyFirstOrder_hpp

double ComputeEnergyFirstOrder(poly_mesh<double> & msh, hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
std::function<double(const typename poly_mesh<double>::point_type& )> scal_fun,
                               std::function<std::vector<double>(const typename poly_mesh<double>::point_type& )> flux_fun);


double ComputeEnergyFirstOrder(poly_mesh<double> & msh, hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
std::function<double(const typename poly_mesh<double>::point_type& )> scal_fun,
                             std::function<std::vector<double>(const typename poly_mesh<double>::point_type& )> flux_fun){
    
    timecounter tc;
    tc.tic();
    
    using RealType = double;
    RealType energy_h = 0.0;
    RealType scalar_l2_energy = 0.0;
    RealType flux_l2_energy = 0.0;
    size_t cell_i = 0;
    for (auto& cell : msh.cells)
    {
        if(cell_i == 0){
            RealType h = diameter(msh, cell);
            std::cout << green << "h size = " << std::endl << h << std::endl;
        }
        
        double c = 1.0;
        auto bar = barycenter(msh, cell);
        double x = bar.x();
        if (x < 0.5) {
            c *= contrast;
        }
        
        size_t cell_scal_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree());
        size_t cell_flux_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree()+1)-1;
        size_t cell_dof = cell_scal_dof+cell_flux_dof;
        
        // scalar evaluation
        {
            Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof+cell_flux_dof, 0, cell_scal_dof, 1);
            Matrix<RealType, Dynamic, Dynamic> mass = make_mass_matrix(msh, cell, hho_di.cell_degree());
            Matrix<RealType, Dynamic, 1> rhs = make_rhs(msh, cell, hho_di.cell_degree(), scal_fun);
            Matrix<RealType, Dynamic, 1> real_dofs = mass.llt().solve(rhs);
            Matrix<RealType, Dynamic, 1> diff = 0.0*real_dofs - scalar_cell_dof;

            if (x > 0.5) {
                scalar_l2_energy += ((1.0/(contrast*contrast))) * (1.0/(c*c))*diff.dot(mass*diff);
             }else{
                 scalar_l2_energy += (1.0/(c*c))*diff.dot(mass*diff);
             }
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
            if (x > 0.5) {
                flux_l2_energy += omega * ((1.0/(contrast*contrast))) * (0.0*grad_u_exact - grad_uh).dot(0.0*grad_u_exact - grad_uh);
            }else{
                flux_l2_energy += omega * (0.0*grad_u_exact - grad_uh).dot(0.0*grad_u_exact - grad_uh);
            }
            
        }

        
        cell_i++;
    }
    
    energy_h = (scalar_l2_energy + flux_l2_energy)/2.0;
    std::cout << green << "Energy_h = " << std::endl << energy_h << std::endl;
    std::cout << bold << cyan << "Energy completed: " << tc << " seconds" << reset << std::endl;
    tc.toc();
    
    return energy_h;
    
}

#endif
