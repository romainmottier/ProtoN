
#ifndef cut_interface_method_hpp
#define cut_interface_method_hpp

template<typename T, size_t ET, typename testType>
class cut_interface_method : public interface_method<T, ET, testType> {

    using Mat  = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;
    using Tuple = std::tuple<double,element_location,std::vector<double>>;

public:
    
    T eta;

    cut_interface_method(T eta_)
        : interface_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_POK(const Mesh& msh, Tuple P_OK, const hho_degree_info hdi, const testType &test_case) {

        // CELL INFOS & PARAMETERS
        T kappa;
        auto stab_parms = test_case.parms;
        if (std::get<1>(P_OK) == element_location::IN_NEGATIVE_SIDE)
            kappa = stab_parms.kappa_1;
        else 
            kappa = stab_parms.kappa_2; 
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        // SUB-CELL INFOS
        auto cell_index = std::get<0>(P_OK);
        auto cl = msh.cells[cell_index];
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        // OPERATORS
        auto gr = make_hho_gradrec_vector_POK(msh, P_OK, hdi, level_set_function, kappa, stab_parms);  // G       
        auto stab_usual = make_hho_stabilization(msh, P_OK, hdi, stab_parms);                          // s° 
        auto stab_cut = make_hho_stabilization_penalty_term(msh, P_OK, hdi, kappa, eta, stab_parms);   // s^\Gamma
        auto stab_ill_dofs = make_hho_ill_dofs_stabilization(msh, P_OK, hdi, eta, stab_parms);         // s^N

        Mat lc = kappa*(gr.second + stab_usual) + stab_cut + stab_ill_dofs; 
        lc.setZero(); 
        Mat f  = make_rhs_pair(msh, P_OK, hdi, test_case); // A DEBUG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        if (std::get<0>(P_OK) == 13)
            std::cout << "Matrice lc :\n" << lc.format(Eigen::IOFormat(4, 0, ", ", "\n", "[", "]")) << std::endl;

        return std::make_pair(lc, f);
    }
    
    std::pair<Mat, Vect>
    make_contrib_PKO(const Mesh& msh, Tuple P_KO, const hho_degree_info hdi, const testType &test_case) {

        // CELL INFOS & PARAMETERS
        auto cell_index = std::get<0>(P_KO);
        auto cl = msh.cells[cell_index];
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        T kappa;
        auto stab_parms = test_case.parms;
        if (std::get<1>(P_KO) == element_location::IN_NEGATIVE_SIDE)
            kappa = stab_parms.kappa_1;
        else 
            kappa = stab_parms.kappa_2; 
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        // OPERATORS
        auto gr = make_hho_gradrec_vector_PKO(msh, P_KO, hdi, level_set_function);
        auto stab = make_hho_stabilization(msh, P_KO, hdi, stab_parms);

        Mat lc = kappa * (gr.second + stab);  
        lc.setZero(); 
        Mat f  = make_rhs_pair(msh, P_KO, hdi, test_case);

        return std::make_pair(lc, f);
    }

    Vect
    make_rhs_pair(const Mesh& msh, Tuple P, const hho_degree_info hdi, const testType &test_case) {
        
        // CELL INFOS 
        auto cell_index = std::get<0>(P);
        auto loc = std::get<1>(P);
        auto cl = msh.cells[cell_index];
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto local_dofs = cbs;
        if (is_cut(msh,cl)) 
            local_dofs += cbs;
    
        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        Matrix<T, Dynamic, 1> f = Matrix<T, Dynamic, 1>::Zero(local_dofs);
        
        if (loc == element_location::IN_NEGATIVE_SIDE)
            f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_NEGATIVE_SIDE);
        else if (!is_cut(msh,cl) && loc == element_location::IN_POSITIVE_SIDE) 
            f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_POSITIVE_SIDE);
        else if (is_cut(msh,cl) && loc == element_location::IN_POSITIVE_SIDE) 
            f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_POSITIVE_SIDE);      
        
        return f;
    }


    Vect
    make_contrib_rhs_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType &test_case, const hho_degree_info hdi) {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        ///////////////    RHS
        Vect f = Vect::Zero(cbs*2);
        // neg part
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_NEGATIVE_SIDE);
        // pos part
        f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_POSITIVE_SIDE);

        return f;
    }

};

template<typename T, size_t ET, typename testType>
auto make_cut_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                   testType test_case) {
    return cut_interface_method<T, ET, testType>(eta_);
}

#endif
