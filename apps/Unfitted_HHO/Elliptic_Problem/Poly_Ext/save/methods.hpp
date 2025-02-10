
#ifndef methods_hpp
#define methods_hpp

#include "cut_methods.hpp"
#include "uncut_methods.hpp"
#include "../Agglo/methods.hpp"

template<typename T>
auto cond(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A) {
    using MT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::JacobiSVD<MT> svd(A);
    auto lmax = svd.singularValues()(0);
    auto lmin = svd.singularValues()(svd.singularValues().size()-1);
    return lmax/lmin; 
}

void writeMatrixToCSV(const std::string& filename, const Eigen::MatrixXd& matrix) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                file << matrix(i, j);
                if (j < matrix.cols() - 1) 
                file << ",";
            }
            file << "\n"; 
        }
        file.close();
    } 
    else 
    std::cerr << "Impossible d'ouvrir le fichier pour écriture." << std::endl;
}

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

    auto assembler = make_one_field_interface_assembler(msh, bcs_fun, hdi);
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
        stab_error += dofs.transpose() * stab * dofs; // prendre la valeure absolue 
        // y_proj = Deltax_proj 
        // 
        stab_usual_error += dofs.transpose() * stab_usual * dofs;
        stab_cut_error += dofs.transpose() * stab_cut * dofs;
    } 
    assembler.finalize();
    tc.toc();
    stab_file << "Characteristic h size = " << std::setprecision(16) << h << std::endl;
    stab_file << "L2-norm grad error = " << std::setprecision(16) << std::sqrt(stab_error) << std::endl;
    stab_usual_file << "Characteristic h size = " << std::setprecision(16) << h << std::endl;
    stab_usual_file << "L2-norm grad error = " << std::setprecision(16) << std::sqrt(stab_usual_error) << std::endl;
    stab_cut_file << "Characteristic h size = " << std::setprecision(16) << h << std::endl;
    stab_cut_file << "L2-norm grad error = " << std::setprecision(16) << std::sqrt(stab_cut_error) << std::endl;
    stab_ill_dofs_file << "Characteristic h size = " << std::setprecision(16) << h << std::endl;
    stab_ill_dofs_file << "L2-norm grad error = " << std::setprecision(16) << std::sqrt(stab_illdofs_error) << std::endl;

    std::string stab_error_file_txt = "stab_error_file.txt";
    std::string stab_usual_error_file_txt = "stab_usual_file.txt";
    std::string stab_cut_error_file_txt = "stab_cut_file.txt";
    std::string stab_illdofs_error_file_txt = "stab_ill_dofs_file.txt";
    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(stab_error_file_txt);
    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(stab_usual_error_file_txt);
    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(stab_cut_error_file_txt);
    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(stab_illdofs_error_file_txt);

    std::cout << bold << yellow << "         Test Stab: " << tc << " seconds" << reset << std::endl;

}

template<typename Mesh, typename testType, typename meth>
void test_stab_on_proj_centered(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, std::ostream & stab_file, std::ostream & stab_usual_file, std::ostream & stab_cut_file, std::ostream & stab_ill_dofs_file);

template<typename Mesh, typename testType, typename meth>
void test_stab_on_proj_centered(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, std::ostream & stab_file, std::ostream & stab_usual_file, std::ostream & stab_cut_file, std::ostream & stab_ill_dofs_file){
    
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

    auto assembler = make_one_field_interface_assembler(msh, bcs_fun, hdi);
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
        auto stab_usual = make_hho_stabilization_centered(msh, P_OK, hdi);                               // s° 
        auto stab_cut = make_hho_stabilization_penalty_term_centered(msh, P_OK, hdi, kappa, 1.0, coeff); // s^\Gamma
        auto stab_ill_dofs = make_hho_ill_dofs_stabilization_centered(msh, P_OK, hdi, 1.0);              // s^N
        auto stab = stab_usual + stab_cut + stab_ill_dofs;
        auto dofs = assembler.gather_proj_centered(msh, P_OK, hdi, sol_fun);
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
        auto stab_usual = make_hho_stabilization_centered(msh, P_KO, hdi);                               // s° 
        auto stab_cut = make_hho_stabilization_penalty_term_centered(msh, P_KO, hdi, kappa, 1.0, coeff); // s^\Gamma
        auto stab = stab_usual + stab_cut;
        auto dofs = assembler.gather_proj_centered(msh, P_KO, hdi, sol_fun);
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

template<typename Mesh, typename testType, typename meth>
Matrix<RealType, Dynamic, 1> test_gradient_on_proj_centered(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case);

template<typename Mesh, typename testType, typename meth>
Matrix<RealType, Dynamic, 1> test_gradient_on_proj_centered(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case){
    
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

    auto assembler = make_one_field_interface_assembler(msh, bcs_fun, hdi);
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
        // basis_evaluation(msh, P_OK, hdi, level_set_function, coeff);
        auto gr = make_hho_gradrec_vector_POK_centered(msh, P_OK, hdi, level_set_function, coeff);
        auto dofs = assembler.gather_proj_centered(msh, P_OK, hdi, sol_fun);
        auto grad_dofs = gr.first * dofs;
        assembler.assemble_grad_bis_extended(msh, P_OK, grad_dofs);  
        assembler.grad_contrib_assembly(msh, P_OK, gr.first);
    } 
    
    // Loop on PKO subcells 
    for (auto& P_KO : Pairs.second) { 
        // CELL INFOS 
        auto cell_index = std::get<0>(P_KO);
        auto loc = std::get<1>(P_KO);
        auto cl = msh.cells[cell_index];
        // GRADIENT
        auto gr = make_hho_gradrec_vector_PKO_centered(msh, P_KO, hdi, level_set_function);
        auto dofs = assembler.gather_proj_centered(msh, P_KO, hdi, sol_fun);
        auto grad_dofs = gr.first * dofs;
        assembler.assemble_grad_bis_extended(msh, P_KO, grad_dofs); 
        assembler.grad_contrib_assembly(msh, P_KO, gr.first);
    } 
    assembler.finalize();
    tc.toc();
    std::cout << bold << yellow << "         Test Gradient: " << tc << " seconds" << reset << std::endl;
    
    // auto Grad_zip = assembler.condensed_GLOBAL_GRAD(msh, assembler.GLOBAL_GRAD);
    // writeMatrixToCSV("Grad_zip.csv", Grad_zip); 

    return assembler.GRAD;

}

template<typename Mesh, typename testType, typename meth>
void test_stab_on_dofs_centered(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<RealType, Dynamic, 1> x_dof, std::ostream & stab_file, std::ostream & stab_usual_file, std::ostream & stab_cut_file, std::ostream & stab_ill_dofs_file);

template<typename Mesh, typename testType, typename meth>
void test_stab_on_dofs_centered(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<RealType, Dynamic, 1> x_dof, std::ostream & stab_file, std::ostream & stab_usual_file, std::ostream & stab_cut_file, std::ostream & stab_ill_dofs_file){
    
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

    auto assembler = make_one_field_interface_assembler(msh, bcs_fun, hdi);
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
        auto stab_usual = make_hho_stabilization_centered(msh, P_OK, hdi);                               // s° 
        auto stab_cut = make_hho_stabilization_penalty_term_centered(msh, P_OK, hdi, kappa, 1.0, coeff); // s^\Gamma
        auto stab_ill_dofs = make_hho_ill_dofs_stabilization_centered(msh, P_OK, hdi, 1.0);              // s^N
        auto stab = stab_usual + stab_cut + stab_ill_dofs;
        auto dofs = assembler.gather_dof(msh, P_OK, x_dof);
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
        auto stab_usual = make_hho_stabilization_centered(msh, P_KO, hdi);                               // s° 
        auto stab_cut = make_hho_stabilization_penalty_term_centered(msh, P_KO, hdi, kappa, 1.0, coeff); // s^\Gamma
        auto stab = stab_usual + stab_cut;
        auto dofs = assembler.gather_dof(msh, P_KO, x_dof);
        stab_error += dofs.transpose() * stab * dofs; 
        stab_usual_error += dofs.transpose() * stab_usual * dofs;
        stab_cut_error += dofs.transpose() * stab_cut * dofs;
    } 
    //     // Loop on PKO subcells 
    // for (auto& P_KO : Pairs.second) { 
    //     // CELL INFOS 
    //     auto cell_index = std::get<0>(P_KO);
    //     auto loc = std::get<1>(P_KO);
    //     auto cl = msh.cells[cell_index];
    //     // GRADIENT
    //     auto gr = make_hho_gradrec_vector_PKO_centered(msh, P_KO, hdi, level_set_function);
    //     auto dofs = assembler.gather_dof(msh, P_KO, x_dof);
    //     auto grad_dofs = gr.first * dofs;
    //     assembler.assemble_grad_bis_extended(msh, P_KO, grad_dofs); 
    //     assembler.grad_contrib_assembly(msh, P_KO, gr.first);
    // } 
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

    std::string stab_error_file_txt = "stab_dofs_error_file_centered.txt";
    std::string stab_usual_error_file_txt = "stab_dofs_usual_file_centered.txt";
    std::string stab_cut_error_file_txt = "stab_dofs_cut_file_centered.txt";
    std::string stab_illdofs_error_file_txt = "stab_dofs_ill_dofs_file_centered.txt";
    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(stab_error_file_txt);
    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(stab_usual_error_file_txt);
    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(stab_cut_error_file_txt);
    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(stab_illdofs_error_file_txt);

    std::cout << bold << yellow << "         Test Stab: " << tc << " seconds" << reset << std::endl;

}

template<typename Mesh, typename testType, typename meth>
Matrix<RealType, Dynamic, 1> test_gradient_on_dofs_centered(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<RealType, Dynamic, 1> x_dof);

template<typename Mesh, typename testType, typename meth>
Matrix<RealType, Dynamic, 1> test_gradient_on_dofs_centered(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<RealType, Dynamic, 1> x_dof){
    
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

    auto assembler = make_one_field_interface_assembler(msh, bcs_fun, hdi);
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
        auto gr = make_hho_gradrec_vector_POK_centered(msh, P_OK, hdi, level_set_function, coeff);
        auto dofs = assembler.gather_dof(msh, P_OK, x_dof);
        auto grad_dofs = gr.first * dofs;
        assembler.assemble_grad_bis_extended(msh, P_OK, grad_dofs);  
        assembler.grad_contrib_assembly(msh, P_OK, gr.first);
    } 

    // Loop on PKO subcells 
    for (auto& P_KO : Pairs.second) { 
        // CELL INFOS 
        auto cell_index = std::get<0>(P_KO);
        auto loc = std::get<1>(P_KO);
        auto cl = msh.cells[cell_index];
        // GRADIENT
        auto gr = make_hho_gradrec_vector_PKO_centered(msh, P_KO, hdi, level_set_function);
        auto dofs = assembler.gather_dof(msh, P_KO, x_dof);
        auto grad_dofs = gr.first * dofs;
        assembler.assemble_grad_bis_extended(msh, P_KO, grad_dofs); 
        assembler.grad_contrib_assembly(msh, P_KO, gr.first);
    } 
    
    assembler.finalize();
    tc.toc();
    std::cout << bold << yellow << "         Test Gradient: " << tc << " seconds" << reset << std::endl;
    
    // auto Grad_zip = assembler.condensed_GLOBAL_GRAD(msh, assembler.GLOBAL_GRAD);
    // writeMatrixToCSV("Grad_zip.csv", Grad_zip); 

    return assembler.GRAD;

}

template<typename Mesh, typename testType, typename meth>
Matrix<RealType, Dynamic, 1> test_gradient_on_dofs(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<RealType, Dynamic, 1> x_dof);

template<typename Mesh, typename testType, typename meth>
Matrix<RealType, Dynamic, 1> test_gradient_on_dofs(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<RealType, Dynamic, 1> x_dof){
    
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

    auto assembler = make_one_field_interface_assembler(msh, bcs_fun, hdi);
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
        auto dofs = assembler.gather_dof(msh, P_OK, x_dof);
        auto grad_dofs = gr.first * dofs;
        assembler.assemble_grad_bis_extended(msh, P_OK, grad_dofs);  
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
        auto dofs = assembler.gather_dof(msh, P_KO, x_dof);
        auto grad_dofs = gr.first * dofs;
        assembler.assemble_grad_bis_extended(msh, P_KO, grad_dofs); 
        assembler.grad_contrib_assembly(msh, P_KO, gr.first);
    } 
    
    assembler.finalize();
    tc.toc();
    std::cout << bold << yellow << "         Test Gradient: " << tc << " seconds" << reset << std::endl;
    
    // auto Grad_zip = assembler.condensed_GLOBAL_GRAD(msh, assembler.GLOBAL_GRAD);
    // writeMatrixToCSV("Grad_zip.csv", Grad_zip); 

    return assembler.GRAD;

}

template<typename Mesh, typename testType, typename meth>
SparseMatrix<typename Mesh::coordinate_type>  test_grad_grad(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case);

template<typename Mesh, typename testType, typename meth>
SparseMatrix<typename Mesh::coordinate_type> test_grad_grad(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case){
    
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

    auto assembler = make_one_field_interface_assembler(msh, bcs_fun, hdi);
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
        assembler.assemble_grad_grad_bis_extended(msh, P_OK, gr.second);  
    } 
    
    // Loop on PKO subcells 
    for (auto& P_KO : Pairs.second) { 
        // CELL INFOS 
        auto cell_index = std::get<0>(P_KO);
        auto loc = std::get<1>(P_KO);
        auto cl = msh.cells[cell_index];
        // GRADIENT
        auto gr = make_hho_gradrec_vector_PKO(msh, P_KO, hdi, level_set_function);
        assembler.assemble_grad_grad_bis_extended(msh, P_KO, gr.second); 
    } 
    assembler.finalize();
    tc.toc();
    std::cout << bold << yellow << "         Test Gradient: " << tc << " seconds" << reset << std::endl;
    
    // auto Grad_zip = assembler.condensed_GLOBAL_GRAD(msh, assembler.GLOBAL_GRAD);
    // writeMatrixToCSV("Grad_zip.csv", Grad_zip); 

    return assembler.GLOBAL_GRAD_GRAD;

}

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

    auto assembler = make_one_field_interface_assembler(msh, bcs_fun, hdi);
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
        assembler.assemble_grad_bis_extended(msh, P_OK, grad_dofs);  
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
        assembler.assemble_grad_bis_extended(msh, P_KO, grad_dofs); 
        assembler.grad_contrib_assembly(msh, P_KO, gr.first);
    } 
    assembler.finalize();
    tc.toc();
    std::cout << bold << yellow << "         Test Gradient: " << tc << " seconds" << reset << std::endl;
    
    // auto Grad_zip = assembler.condensed_GLOBAL_GRAD(msh, assembler.GLOBAL_GRAD);
    // writeMatrixToCSV("Grad_zip.csv", Grad_zip); 

    return assembler.GRAD;

}

template<typename Mesh, typename testType, typename meth>
std::vector<std::pair<size_t,size_t>> assembly_poly_extension_centered(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg);

template<typename Mesh, typename testType, typename meth>
std::vector<std::pair<size_t,size_t>> assembly_poly_extension_centered(const Mesh& msh, hho_degree_info & hdi, meth &method, testType & test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg) {
    
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
    std::pair<VecTuple,VecTuple> Pairs = make_pair_KO_pair_OK(msh);

    // Loop on POK subcells 
    for (auto& pair : Pairs.first) { 
        auto cl = msh.cells[std::get<0>(pair)];
        auto contrib = method.make_contrib_POK_centered(msh, pair, test_case, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;
        // auto cell_mass = method.make_contrib_mass(msh, pair, test_case, hdi);      
        // size_t n_dof = assembler.n_dof(msh, cl);
        // Matrix<RealType, Dynamic, Dynamic> mass = Matrix<RealType, Dynamic, Dynamic>::Zero(n_dof, n_dof);
        // mass.block(0,0,cell_mass.rows(), cell_mass.cols()) = cell_mass;
        assembler.assemble_extended(msh, pair, lc, f);  
        // assembler.assemble_mass(msh, cl, mass);
    } 
    // Loop on PKO subcells 
    for (auto& pair : Pairs.second) { 
        auto cl = msh.cells[std::get<0>(pair)];
        auto contrib = method.make_contrib_PKO_centered(msh, pair, test_case, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;
        assembler.assemble_extended(msh, pair, lc, f);  
    } 
    assembler.finalize();
    
    tc.toc();
    std::cout << bold << yellow << "         Matrix assembly: " << tc << " seconds" << reset << std::endl;

    Kg = assembler.LHS;
    Mg = assembler.MASS;
    
    return cell_basis_data;

}

template<typename Mesh, typename testType, typename meth>
std::vector<std::pair<size_t,size_t>> assembly_poly_extension(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg);

template<typename Mesh, typename testType, typename meth>
std::vector<std::pair<size_t,size_t>> assembly_poly_extension(const Mesh& msh, hho_degree_info & hdi, meth &method, testType & test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg) {
    
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
    std::pair<VecTuple,VecTuple> Pairs = make_pair_KO_pair_OK(msh);
    // Loop on POK subcells 
    for (auto& pair : Pairs.first) { 
        auto cl = msh.cells[std::get<0>(pair)];
        auto contrib = method.make_contrib_POK(msh, pair, test_case, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;
        assembler.assemble_extended(msh, pair, lc, f);  
    } 

    // Loop on PKO subcells 
    for (auto& pair : Pairs.second) { 
        auto cl = msh.cells[std::get<0>(pair)];
        auto contrib = method.make_contrib_PKO(msh, pair, test_case, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;
        assembler.assemble_extended(msh, pair, lc, f);  
    } 
    assembler.finalize();
    
    tc.toc();
    std::cout << bold << yellow << "         Matrix assembly: " << tc << " seconds" << reset << std::endl;

    Kg = assembler.LHS;
    Mg = assembler.MASS;
    
    return cell_basis_data;

}

template<typename Mesh, typename testType, typename meth>
Matrix<RealType, Dynamic, 1> test_conditioning(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case);

template<typename Mesh, typename testType, typename meth>
Matrix<RealType, Dynamic, 1> test_conditioning(const Mesh& msh, hho_degree_info & hdi, meth &method, testType & test_case) {
    
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
    std::pair<VecTuple,VecTuple> Pairs = make_pair_KO_pair_OK(msh);
    bool matrix_conditioning = true;
    // Loop on POK subcells 
    for (auto& pair : Pairs.first) { 
        auto cl = msh.cells[std::get<0>(pair)];
        auto contrib = method.make_contrib_POK(msh, pair, test_case, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;
        assembler.assemble_extended(msh, pair, lc, f);  
        if (matrix_conditioning) {
            auto n = lc.rows() - 1;
            auto condensedlc = lc.bottomRightCorner(n,n).eval();
            auto condlc = cond(condensedlc);
            assembler.assemble_conditioning(msh, pair, condlc);  
        }
    } 

    // Loop on PKO subcells 
    for (auto& pair : Pairs.second) { 
        auto cl = msh.cells[std::get<0>(pair)];
        auto contrib = method.make_contrib_PKO(msh, pair, test_case, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;
        assembler.assemble_extended(msh, pair, lc, f);  
        if (matrix_conditioning) {
            auto n = lc.rows() - 1;
            auto condensedlc = lc.bottomRightCorner(n,n).eval();
            auto condlc = cond(condensedlc);
            assembler.assemble_conditioning(msh, pair, condlc);  
        }
    } 
    
    tc.toc();
    std::cout << bold << yellow << "         Test conditioning: " << tc << " seconds" << reset << std::endl;
    
    return assembler.CONDITIONING;

}

template<typename T, size_t ET, typename testType>
class call_methods : public uncut_method<T, ET, testType>, cut_method<T, ET, testType> {

    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;
    using Tuple = std::tuple<double,element_location,std::vector<double>>;

public:

    T eta;

    call_methods(T eta_) : uncut_method<T,ET,testType>(), cut_method<T,ET,testType>(eta_), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_POK_centered(const Mesh& msh, Tuple P, const testType &test_case, const hho_degree_info hdi) {
            return cut_method<T, ET, testType>::make_contrib_POK_centered(msh, P, test_case, hdi);
    }
    
    std::pair<Mat, Vect>
    make_contrib_PKO_centered(const Mesh& msh, Tuple P, const testType &test_case, const hho_degree_info hdi) {
            return cut_method<T, ET, testType>::make_contrib_PKO_centered(msh, P, test_case, hdi);
    }

    std::pair<Mat, Vect>
    make_contrib_POK(const Mesh& msh, Tuple P, const testType &test_case, const hho_degree_info hdi) {
            return cut_method<T, ET, testType>::make_contrib_POK(msh, P, test_case, hdi);
    }
    
    std::pair<Mat, Vect>
    make_contrib_PKO(const Mesh& msh, Tuple P, const testType &test_case, const hho_degree_info hdi) {
            return cut_method<T, ET, testType>::make_contrib_PKO(msh, P, test_case, hdi);
    }

    Vect
    make_contrib_rhs(const Mesh& msh, const typename Mesh::cell_type& cl, const testType &test_case, const hho_degree_info hdi) {
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return uncut_method<T, ET, testType>::make_contrib_rhs_uncut(msh, cl, hdi, test_case);
        else 
            return cut_method<T, ET, testType>::make_contrib_rhs_cut(msh, cl, test_case, hdi);
    }

    Vect
    make_contrib_rhs_centered(const Mesh& msh, const typename Mesh::cell_type& cl, const testType &test_case, const hho_degree_info hdi) {
        
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return uncut_method<T, ET, testType>::make_contrib_rhs_uncut(msh, cl, hdi, test_case);
        else 
            return cut_method<T, ET, testType>::make_contrib_rhs_cut_centered(msh, cl, test_case, hdi);
    }

    Mat
    make_contrib_mass(const Mesh& msh, Tuple P, const testType &test_case, const hho_degree_info hdi) {
        return cut_method<T, ET, testType>::make_contrib_cut_mass(msh, P, hdi, test_case);
    }

};

template<typename T, size_t ET, typename testType>
auto make_call_methods(const cuthho_mesh<T, ET>& msh, const T eta_, testType test_case) {
    
    return call_methods<T, ET, testType>(eta_);

}

#endif
