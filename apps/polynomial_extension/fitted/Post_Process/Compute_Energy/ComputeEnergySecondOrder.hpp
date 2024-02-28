
#ifndef ComputeEnergySecondOrder_hpp
#define ComputeEnergySecondOrder_hpp

double ComputeEnergySecondOrder(poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, Matrix<double, Dynamic, 1> & p, Matrix<double, Dynamic, 1> & v);

double ComputeEnergySecondOrder(poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, Matrix<double, Dynamic, 1> & p_dof_n, Matrix<double, Dynamic, 1> & v_dof_n){
    using RealType = double;
    RealType energy_h = 0.0;
    size_t cell_i = 0;
    
    double t = 0.0
    ;    auto exact_sol_fun = [&t](const typename poly_mesh<double>::point_type& pt) -> double {

#ifdef quadratic_time_solution_Q
                return t * t * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
#else
#ifdef quadratic_space_solution_Q
                return std::cos(std::sqrt(2.0)*M_PI*t) * (1.0-pt.x())*pt.x() * (1.0-pt.y())*pt.y();
#else
                return (1.0/(std::sqrt(2.0)*M_PI))*std::sin(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
#endif
#endif
    };
    
    for (auto &cell : msh.cells) {
        
            cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
            auto cell_dof = cell_basis.size();

            Matrix<RealType, Dynamic, Dynamic> mass = make_cell_mass_matrix(msh, cell, hho_di);

            Matrix<RealType, Dynamic, Dynamic> cell_mass = mass.block(0, 0, cell_dof, cell_dof);
            Matrix<RealType, Dynamic, 1> cell_alpha_dof_n_v = v_dof_n.block(cell_i*cell_dof, 0, cell_dof, 1);
            
            Matrix<RealType, Dynamic, 1> cell_mass_tested = cell_mass * cell_alpha_dof_n_v;
            Matrix<RealType, 1, 1> term_1 = cell_alpha_dof_n_v.transpose() * cell_mass_tested;
            energy_h += term_1(0,0);
            
            auto reconstruction_operator = make_hho_laplacian(msh, cell, hho_di);
#ifdef fancy_stabilization_Q
            auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
#else
            auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
#endif
            Matrix<RealType, Dynamic, Dynamic> laplacian_loc = reconstruction_operator.second + stabilization_operator;
            Matrix<RealType, Dynamic, 1> cell_p_dofs = assembler.take_local_data(msh, cell, p_dof_n, exact_sol_fun);
            Matrix<RealType, Dynamic, 1> cell_stiff_tested = laplacian_loc * cell_p_dofs;
            Matrix<RealType, 1, 1> term_2 = cell_p_dofs.transpose() * cell_stiff_tested;
            energy_h += term_2(0,0);
        cell_i++;
    }

    energy_h *= 0.5;
    std::cout << green << "Energy_h = " << std::endl << energy_h << std::endl;
    std::cout << std::endl;
    
    return energy_h;
}

#endif
