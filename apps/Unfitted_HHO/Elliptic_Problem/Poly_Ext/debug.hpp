
// TEST GRADIENT RECONSTRUCTION
template<typename Mesh, typename testType, typename meth>
Matrix<RealType, Dynamic, 1> test_gradient_on_proj(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case);

template<typename Mesh, typename testType, typename meth>
Matrix<RealType, Dynamic, 1> test_gradient_on_proj(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case){
    
    using RealType = typename Mesh::coordinate_type;
    using VecTuple = std::vector<std::tuple<double,element_location,std::vector<double>>>;
    using T = typename Mesh::coordinate_type;

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

    auto assembler = make_interface_assembler(msh, bcs_fun, hdi);
    std::pair<VecTuple,VecTuple> Pairs = make_pair_KO_pair_OK(msh);
    SparseMatrix<RealType> grad;

    // Loop on POK subcells 
    for (auto& P_OK : Pairs.first) { 
        // CELL INFOS 
        auto cell_index = std::get<0>(P_OK);
        auto loc = std::get<1>(P_OK);
        auto cl = msh.cells[cell_index];
        // MATERIAL PROPERTIES
        double kappa;
        if (loc == element_location::IN_NEGATIVE_SIDE)
            kappa = 1.0/test_case.parms.kappa_1;
        else
            kappa = 1.0/test_case.parms.kappa_2;
        auto stab_parms = test_case.parms;
        stab_parms.kappa_1 = 1.0/(test_case.parms.kappa_1); 
        stab_parms.kappa_2 = 1.0/(test_case.parms.kappa_2); 
        auto coeff = 0.0;
        if (stab_parms.kappa_1 < stab_parms.kappa_2) {
            if (loc == element_location::IN_POSITIVE_SIDE)
                coeff = 1.0;
        }
        else {
            if (loc == element_location::IN_NEGATIVE_SIDE)
                coeff = 1.0;
        }
        // GRADIENT 
        auto gr = make_hho_gradrec_vector_POK(msh, P_OK, hdi, level_set_function, coeff);
        auto dofs = assembler.gather_proj(msh, P_OK, hdi, sol_fun);
        auto grad_dofs = gr.first * dofs;
        assembler.assemble_grad_ext(msh, P_OK, grad_dofs);  
        assembler.grad_contrib_assembly(msh, P_OK, gr.first);
    } 
    
    // Loop on PKO subcells 
    for (auto& P_KO : Pairs.second) { 
        // CELL INFOS 
        auto cell_index = std::get<0>(P_KO);
        auto loc = std::get<1>(P_KO);
        auto cl = msh.cells[cell_index];
        // GRADIENT
        auto gr = make_hho_gradrec_vector_PKO(msh, P_KO, hdi, level_set_function);
        auto dofs = assembler.gather_proj(msh, P_KO, hdi, sol_fun);
        auto grad_dofs = gr.first * dofs;
        assembler.assemble_grad_ext(msh, P_KO, grad_dofs); 
        assembler.grad_contrib_assembly(msh, P_KO, gr.first);
    } 
    assembler.finalize();
    tc.toc();
    std::cout << bold << yellow << "         Test Gradient: " << tc << " seconds" << reset << std::endl;
    
    return assembler.GRAD;

}

// TEST STABILIZAIONS
template<typename Mesh, typename testType, typename meth>
void test_stab_on_proj(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, std::ostream & stab_file, std::ostream & stab_usual_file, std::ostream & stab_cut_file, std::ostream & stab_ill_dofs_file);

template<typename Mesh, typename testType, typename meth>
void test_stab_on_proj(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, std::ostream & stab_file, std::ostream & stab_usual_file, std::ostream & stab_cut_file, std::ostream & stab_ill_dofs_file){
    
    using RealType = typename Mesh::coordinate_type;
    using VecTuple = std::vector<std::tuple<double,element_location,std::vector<double>>>;
    using T = typename Mesh::coordinate_type;

    auto level_set_function = test_case.level_set_;
    auto rhs_fun = test_case.rhs_fun;
    auto sol_fun = test_case.sol_fun;   
    auto sol_grad = test_case.sol_grad;
    auto bcs_fun = test_case.bcs_fun;
    auto dirichlet_jump = test_case.dirichlet_jump;
    auto neumann_jump = test_case.neumann_jump;
    struct params<RealType> parms = test_case.parms;
    
    auto stab_error = 0.0;
    auto stab_usual_error = 0.0;
    auto stab_cut_error = 0.0;
    auto stab_illdofs_error = 0.0;

    timecounter tc;
    tc.tic();

    auto assembler = make_interface_assembler(msh, bcs_fun, hdi);
    std::pair<VecTuple,VecTuple> Pairs = make_pair_KO_pair_OK(msh);

    RealType h = 10.0;
    // Loop on POK subcells 
    for (auto& P_OK : Pairs.first) { 
        // CELL INFOS 
        auto cell_index = std::get<0>(P_OK);
        auto loc = std::get<1>(P_OK);
        auto cl = msh.cells[cell_index];
        // MATERIAL PROPERTIES
        double kappa;
        if (loc == element_location::IN_NEGATIVE_SIDE)
            kappa = 1.0/test_case.parms.kappa_1;
        else
            kappa = 1.0/test_case.parms.kappa_2;
        auto stab_parms = test_case.parms;
        stab_parms.kappa_1 = 1.0/(test_case.parms.kappa_1); 
        stab_parms.kappa_2 = 1.0/(test_case.parms.kappa_2); 
        auto coeff = 0.0;
        if (stab_parms.kappa_1 < stab_parms.kappa_2) {
            if (loc == element_location::IN_POSITIVE_SIDE)
                coeff = 1.0;
        }
        else {
            if (loc == element_location::IN_NEGATIVE_SIDE)
                coeff = 1.0;
        }
        RealType h_l = diameter(msh, cl);
        if (h_l < h)
            h = h_l;
        // STABILIZATION 
        auto stab_usual = make_hho_stabilization(msh, P_OK, hdi);                               // s° 
        auto stab_cut = make_hho_stabilization_penalty_term(msh, P_OK, hdi, kappa, 1.0, coeff); // s^\Gamma
        auto stab_ill_dofs = make_hho_ill_dofs_stabilization(msh, P_OK, hdi, 1.0);              // s^N
        auto stab = stab_usual + stab_cut + stab_ill_dofs;
        auto dofs = assembler.gather_proj(msh, P_OK, hdi, sol_fun);
        stab_error += dofs.transpose() * stab * dofs;
        stab_usual_error += dofs.transpose() * stab_usual * dofs;
        stab_cut_error += dofs.transpose() * stab_cut * dofs;
        stab_illdofs_error += dofs.transpose() * stab_ill_dofs * dofs;
    } 
    
    // Loop on PKO subcells 
    for (auto& P_KO : Pairs.second) { 
        // CELL INFOS 
        auto cell_index = std::get<0>(P_KO);
        auto loc = std::get<1>(P_KO);
        auto cl = msh.cells[cell_index];
        // MATERIAL PROPERTIES
        double kappa;
        if (loc == element_location::IN_NEGATIVE_SIDE)
            kappa = 1.0/test_case.parms.kappa_1;
        else
            kappa = 1.0/test_case.parms.kappa_2;
        auto stab_parms = test_case.parms;
        stab_parms.kappa_1 = 1.0/(test_case.parms.kappa_1); 
        stab_parms.kappa_2 = 1.0/(test_case.parms.kappa_2); 
        auto coeff = 0.0;
        if (stab_parms.kappa_1 < stab_parms.kappa_2) {
            if (loc == element_location::IN_POSITIVE_SIDE)
                coeff = 1.0;
        }
        else {
            if (loc == element_location::IN_NEGATIVE_SIDE)
                coeff = 1.0;
        }
        RealType h_l = diameter(msh, cl);
        if (h_l < h)
            h = h_l;
        // STABILIZATION
        auto stab_usual = make_hho_stabilization(msh, P_KO, hdi);                               // s° 
        auto stab_cut = make_hho_stabilization_penalty_term(msh, P_KO, hdi, kappa, 1.0, coeff); // s^\Gamma
        auto stab = stab_usual + stab_cut;
        auto dofs = assembler.gather_proj(msh, P_KO, hdi, sol_fun);
        stab_error += dofs.transpose() * stab * dofs; 
        stab_usual_error += dofs.transpose() * stab_usual * dofs;
        stab_cut_error += dofs.transpose() * stab_cut * dofs;
    } 
    assembler.finalize();
    tc.toc();
    stab_file << "Characteristic h size = " << std::setprecision(64) << h << std::endl;
    stab_file << "L2-norm grad error = " << std::setprecision(64) << std::sqrt(stab_error) << std::endl;
    stab_usual_file << "Characteristic h size = " << std::setprecision(64) << h << std::endl;
    stab_usual_file << "L2-norm grad error = " << std::setprecision(64) << std::sqrt(stab_usual_error) << std::endl;
    stab_cut_file << "Characteristic h size = " << std::setprecision(64) << h << std::endl;
    stab_cut_file << "L2-norm grad error = " << std::setprecision(64) << std::sqrt(stab_cut_error) << std::endl;
    stab_ill_dofs_file << "Characteristic h size = " << std::setprecision(64) << h << std::endl;
    stab_ill_dofs_file << "L2-norm grad error = " << std::setprecision(64) << std::sqrt(stab_illdofs_error) << std::endl;

    std::string stab_error_file_txt = "stab_proj_error_file_centered.txt";
    std::string stab_usual_error_file_txt = "stab_proj_usual_file_centered.txt";
    std::string stab_cut_error_file_txt = "stab_proj_cut_file_centered.txt";
    std::string stab_illdofs_error_file_txt = "stab_proj_ill_dofs_file_centered.txt";
    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(stab_error_file_txt);
    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(stab_usual_error_file_txt);
    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(stab_cut_error_file_txt);
    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(stab_illdofs_error_file_txt);

    std::cout << bold << yellow << "         Test Stab: " << tc << " seconds" << reset << std::endl;

}

// TEST CONDITIONING 
template<typename T>
auto cond(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A) {
    using MT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::JacobiSVD<MT> svd(A);
    auto lmax = svd.singularValues()(0);
    auto lmin = svd.singularValues()(svd.singularValues().size()-1);
    return lmax/lmin; 
}
