
#ifndef create_mixed_kg_and_mg_cuthho_interface_hpp
#define create_mixed_kg_and_mg_cuthho_interface_hpp

template<typename Mesh, typename testType, typename meth>
std::vector<std::pair<size_t,size_t>>
create_mixed_kg_and_mg_cuthho_interface(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg, bool add_scalar_mass_Q = true, size_t *n_faces = 0);

template<typename Mesh, typename testType, typename meth>
std::vector<std::pair<size_t,size_t>>
create_mixed_kg_and_mg_cuthho_interface(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg, bool add_scalar_mass_Q, size_t *n_faces){
    
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
    auto assembler = make_two_fields_interface_assembler(msh, bcs_fun, hdi);
    std::vector<std::pair<size_t,size_t>> cell_basis_data = assembler.compute_cell_basis_data(msh);
    if(n_faces) *n_faces = assembler.get_n_faces();
    size_t cell_ind = 0;
    for (auto& cl : msh.cells) {
        auto contrib = method.make_contrib(msh, cl, test_case, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;

        auto cell_mass = method.make_contrib_mass(msh, cl, test_case, hdi, add_scalar_mass_Q);
        size_t n_dof = assembler.n_dof(msh,cl);
        Matrix<RealType, Dynamic, Dynamic> mass = Matrix<RealType, Dynamic, Dynamic>::Zero(n_dof,n_dof);
        mass.block(0,0,cell_mass.rows(),cell_mass.cols()) = cell_mass;
        assembler.assemble(msh, cl, lc, f);
        assembler.assemble_mass(msh, cl, mass);
        cell_ind++;
    }
    assembler.finalize();

    tc.toc();
    std::cout << bold << yellow << "         Matrix assembly: " << tc << " seconds" << reset << std::endl;
    
    Kg = assembler.LHS;
    Mg = assembler.MASS;
    return cell_basis_data;
}

#endif
