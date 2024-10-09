
#ifndef create_kg_and_mg_cuthho_interface_hpp
#define create_kg_and_mg_cuthho_interface_hpp

template<typename Mesh, typename testType, typename meth>
std::vector<std::pair<size_t,size_t>>
create_kg_and_mg_cuthho_interface(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg);

template<typename Mesh, typename testType, typename meth>
std::vector<std::pair<size_t,size_t>>
create_kg_and_mg_cuthho_interface(const Mesh& msh, hho_degree_info & hdi, meth &method, testType & test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg){
    
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
    
    tc.tic();
    auto assembler = make_one_field_interface_assembler(msh, bcs_fun, hdi);
    std::vector<std::pair<size_t,size_t>> cell_basis_data = assembler.compute_cell_basis_data(msh);
    for (auto& cell : msh.cells) {
      auto contrib = method.make_contrib(msh, cell, test_case, hdi);
      auto lc = contrib.first;
      auto f = contrib.second;
      auto cell_mass = method.make_contrib_mass(msh, cell, test_case, hdi);
      size_t n_dof = assembler.n_dof(msh,cell);
      Matrix<RealType, Dynamic, Dynamic> mass = Matrix<RealType, Dynamic, Dynamic>::Zero(n_dof,n_dof);
      mass.block(0,0,cell_mass.rows(),cell_mass.cols()) = cell_mass;
      assembler.assemble(msh, cell, lc, f);
      assembler.assemble_mass(msh, cell, mass);
    }
    assembler.finalize();
    tc.toc();
    std::cout << bold << yellow << "         Matrix assembly: " << tc << " seconds" << reset << std::endl;
    
    Kg = assembler.LHS; // A DEBUG
    Mg = assembler.MASS;
    return cell_basis_data;
}



template<typename Mesh, typename testType, typename meth>
std::vector<std::pair<size_t,size_t>>
test_operators(Mesh& msh, hho_degree_info & hdi, meth &method, testType & test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg){
    
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
    
    tc.tic();
    auto assembler = make_one_field_interface_assembler(msh, bcs_fun, hdi);
    std::vector<std::pair<size_t,size_t>> cell_basis_data = assembler.compute_cell_basis_data(msh);
    size_t system_size = assembler.compute_dofs_data(msh, hdi);
    // auto dofs_proj = assembler.make_projection_operator(msh, hdi, system_size, sol_fun);
    // for (auto& cell : msh.cells) {
    //   auto contrib = method.make_contrib(msh, cell, test_case, hdi);
    //   auto lc = contrib.first;
    //   auto f = contrib.second;
    //   auto cell_mass = method.make_contrib_mass(msh, cell, test_case, hdi);
    //   size_t n_dof = assembler.n_dof(msh,cell);
    //   Matrix<RealType, Dynamic, Dynamic> mass = Matrix<RealType, Dynamic, Dynamic>::Zero(n_dof,n_dof);
    //   mass.block(0,0,cell_mass.rows(),cell_mass.cols()) = cell_mass;
    //   assembler.assemble_extended(msh, cell, lc, f);  
    //   assembler.assemble_mass(msh, cell, mass);
    // }

    
    return cell_basis_data;
}

#endif
