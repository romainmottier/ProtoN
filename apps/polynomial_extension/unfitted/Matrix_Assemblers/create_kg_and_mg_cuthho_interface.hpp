
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
    using VecTuple = std::vector<std::tuple<double,element_location,std::vector<double>>>;
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
    std::pair<VecTuple,VecTuple> Pairs = make_pair_KO_pair_OK(msh);
    // auto dofs_proj = assembler.make_projection_operator(msh, hdi, system_size, sol_fun);
    // Loop over POK
    for (auto& pair : Pairs.first) {
        auto contrib = method.make_contrib_POK(msh, pair, hdi, test_case);
        auto lc = contrib.first;
        auto f = contrib.second;
        auto cell_mass = method.make_contrib_mass(msh, pair, test_case, hdi);
        assembler.assemble_extended(msh, pair, lc, f);  
        assembler.assemble_mass_extended(msh, pair, cell_mass);
    } 
    std::cout << "POK DONE" << std::endl;
    // Loop over PKO
    for (auto& pair : Pairs.second) {
        auto contrib = method.make_contrib_PKO(msh, pair, hdi, test_case);
        auto lc = contrib.first;
        auto f = contrib.second;
        auto cell_mass = method.make_contrib_mass(msh, pair, test_case, hdi);
        assembler.assemble_extended(msh, pair, lc, f);  
        assembler.assemble_mass_extended(msh, pair, cell_mass);
    } 
    std::cout << "PKO DONE" << std::endl;
    // Loop over cut cells 
    for (auto& cl : msh.cells) {
        if (is_cut(msh,cl)) {
            method.make_contrib_cut(msh, cl, test_case, hdi);         
        }
    }
    std::cout << "PENALTY DONE" << std::endl;
        
    assembler.finalize();
    tc.toc();
    std::cout << bold << yellow << "         Matrix assembly: " << tc << " seconds" << reset << std::endl;
    
    Kg = assembler.LHS; 
    Mg = assembler.MASS;
    
    return cell_basis_data;
}

template<typename Mesh>
std::pair<std::vector< std::tuple<double,element_location,std::vector<double>>>,
          std::vector< std::tuple<double,element_location,std::vector<double>>>> 
make_pair_KO_pair_OK(Mesh& msh) {

    std::vector< std::tuple<double,element_location,std::vector<double>>> PairOK;
    std::vector< std::tuple<double,element_location,std::vector<double>>> PairKO;

    for (auto &cl : msh.cells) {
        auto offset_cl = offset(msh,cl);
        if (cl.user_data.location != element_location::ON_INTERFACE) { // UNCUT CELL
            std::vector<double> dp_cells;
            element_location loc;
            if (cl.user_data.location == element_location::IN_NEGATIVE_SIDE) {
                loc = element_location::IN_NEGATIVE_SIDE;
                for (auto& dp_cl: cl.user_data.dependent_cells_neg) {
                    dp_cells.push_back(dp_cl);
                }
            }
            else {
                loc = element_location::IN_POSITIVE_SIDE;
                for (auto& dp_cl: cl.user_data.dependent_cells_pos) {
                    dp_cells.push_back(dp_cl);
                }   
            }
            auto tuple = std::make_tuple(offset_cl, loc, dp_cells);
            cl.user_data.Pair_OK.push_back(tuple);
        }
        else if (cl.user_data.agglo_set == cell_agglo_set::T_OK) { // CUT CELL TOK
            std::vector<double> dp_cells_neg;
            std::vector<double> dp_cells_pos;
            for (auto& dp_cl: cl.user_data.dependent_cells_neg) {
                dp_cells_neg.push_back(dp_cl);
            }
            for (auto& dp_cl: cl.user_data.dependent_cells_pos) {
                dp_cells_pos.push_back(dp_cl);
            }
            auto tuple_neg = std::make_tuple(offset_cl, element_location::IN_NEGATIVE_SIDE, dp_cells_neg);
            auto tuple_pos = std::make_tuple(offset_cl, element_location::IN_POSITIVE_SIDE, dp_cells_pos);
            cl.user_data.Pair_OK.push_back(tuple_neg);
            cl.user_data.Pair_OK.push_back(tuple_pos);
        }
        else if (cl.user_data.agglo_set == cell_agglo_set::T_KO_NEG) { // CUT CELL TKONEG
            std::vector<double> dp_cells_neg;
            std::vector<double> dp_cells_pos;
            for (auto& dp_cl: cl.user_data.dependent_cells_pos) {
                dp_cells_pos.push_back(dp_cl);
            }
            auto tuple_neg = std::make_tuple(offset_cl, element_location::IN_NEGATIVE_SIDE, dp_cells_neg);
            auto tuple_pos = std::make_tuple(offset_cl, element_location::IN_POSITIVE_SIDE, dp_cells_pos);
            cl.user_data.Pair_KO.push_back(tuple_neg);
            cl.user_data.Pair_OK.push_back(tuple_pos);
        }
        else if (cl.user_data.agglo_set == cell_agglo_set::T_KO_POS) { // CUT CELL TKONEG
            std::vector<double> dp_cells_neg;
            std::vector<double> dp_cells_pos;
            for (auto& dp_cl: cl.user_data.dependent_cells_neg) {
                dp_cells_neg.push_back(dp_cl);
            }
            auto tuple_neg = std::make_tuple(offset_cl, element_location::IN_NEGATIVE_SIDE, dp_cells_neg);
            auto tuple_pos = std::make_tuple(offset_cl, element_location::IN_POSITIVE_SIDE, dp_cells_pos);
            cl.user_data.Pair_OK.push_back(tuple_neg);
            cl.user_data.Pair_KO.push_back(tuple_pos);
        }

    PairOK.insert(PairOK.end(), cl.user_data.Pair_OK.begin(), cl.user_data.Pair_OK.end());
    PairKO.insert(PairKO.end(), cl.user_data.Pair_KO.begin(), cl.user_data.Pair_KO.end());

    }   

    // // Debug
    // for (auto &cl : msh.cells) {
    //     auto offset_cl = offset(msh,cl);
    //     std::cout << "Cell: " << offset_cl << std::endl;
    //     std::cout << "Negative dependent cells:   ";
    //     for (auto& cells : cl.user_data.dependent_cells_neg) {
    //         std::cout << cells << "   ";
    //     }
    //     std::cout << std::endl << "Positive dependent cells:   ";
    //     for (auto& cells : cl.user_data.dependent_cells_pos) {
    //         std::cout << cells << "   ";
    //     }
    //     std::cout << std::endl << "TUPLE PAIROK:   ";
    //     for (auto& cells : cl.user_data.Pair_OK) {
    //         if (std::get<1>(cells) == element_location::IN_NEGATIVE_SIDE) {
    //             std::cout << "(" << std::get<0>(cells) << ", " << "NEGATIVE SIDE";
    //             for (auto& dp_cl : std::get<2>(cells)) {
    //                 std::cout << ", " << dp_cl;
    //             }
    //             std::cout << ")";
    //         }
    //         if (std::get<1>(cells) == element_location::IN_POSITIVE_SIDE) {
    //             std::cout << "(" << std::get<0>(cells) << ", " << "POSITIVE SIDE";
    //             for (auto& dp_cl : std::get<2>(cells)) {
    //                 std::cout << ", " << dp_cl;
    //             }
    //             std::cout << ")";
    //         }
    //     }
    //     std::cout << std::endl << "TUPLE PAIRKO:   ";
    //     for (auto& cells : cl.user_data.Pair_KO) {
    //         if (std::get<1>(cells) == element_location::IN_NEGATIVE_SIDE) {
    //             std::cout << "(" << std::get<0>(cells) << ", " << "NEGATIVE SIDE" << ") ";
    //         }
    //         if (std::get<1>(cells) == element_location::IN_POSITIVE_SIDE) {
    //             std::cout << "(" << std::get<0>(cells) << ", " << "POSITIVE SIDE" << ") ";
    //         }
    //     }  
    //     std::cout << std::endl << std::endl;
    // }
    // std::cout << "PAIRES OK: " << std::endl;
    // for (auto& cells : PairOK) {
    //     if (std::get<1>(cells) == element_location::IN_NEGATIVE_SIDE) {
    //         std::cout << "(" << std::get<0>(cells) << ", " << "NEGATIVE SIDE";
    //         for (auto& dp_cl : std::get<2>(cells)) {
    //             std::cout << ", " << dp_cl;
    //         }
    //         std::cout << ")" << std::endl;
    //     }
    //     if (std::get<1>(cells) == element_location::IN_POSITIVE_SIDE) {
    //         std::cout << "(" << std::get<0>(cells) << ", " << "POSITIVE SIDE";
    //         for (auto& dp_cl : std::get<2>(cells)) {
    //             std::cout << ", " << dp_cl;
    //         }
    //         std::cout << ")" << std::endl;
    //     }
    // }
    // std::cout << std::endl << "PAIRES KO: " << std::endl;
    // for (auto& cells : PairKO) {
    //     if (std::get<1>(cells) == element_location::IN_NEGATIVE_SIDE) {
    //         std::cout << "(" << std::get<0>(cells) << ", " << "NEGATIVE SIDE";
    //         for (auto& dp_cl : std::get<2>(cells)) {
    //             std::cout << ", " << dp_cl;
    //         }
    //         std::cout << ")" << std::endl;
    //     }
    //     if (std::get<1>(cells) == element_location::IN_POSITIVE_SIDE) {
    //         std::cout << "(" << std::get<0>(cells) << ", " << "POSITIVE SIDE";
    //         for (auto& dp_cl : std::get<2>(cells)) {
    //             std::cout << ", " << dp_cl;
    //         }
    //         std::cout << ")" << std::endl;
    //     }
    // }

    auto Pair_OK_KO = std::make_pair(PairOK,PairKO);
    return Pair_OK_KO;

}


#endif
