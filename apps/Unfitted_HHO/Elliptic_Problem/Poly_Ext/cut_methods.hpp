
template<typename T, size_t ET, typename testType>
class cut_method {

    using Mat  = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;
    using Tuple = std::tuple<double,element_location,std::vector<double>>;

protected:
    T eta; 

    cut_method(T eta_) : eta(eta_) {} 

public:

    std::pair<Mat, Vect>
    make_contrib_POK_centered(const Mesh& msh, Tuple P_OK, const testType &test_case, const hho_degree_info hdi) {
        
        // CELL INFOS 
        auto cell_index = std::get<0>(P_OK);
        auto loc = std::get<1>(P_OK);
        auto cl = msh.cells[cell_index];

        // MATERIAL PROPERTIES
        T kappa;
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

        // LEVEL SET FUNCTION
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        // HHO OPERATORS
        auto gr = make_hho_gradrec_vector_POK_centered(msh, P_OK, hdi, level_set_function, coeff);       // G     
        auto stab_usual = make_hho_stabilization_centered(msh, P_OK, hdi);                               // s° 
        auto stab_cut = make_hho_stabilization_penalty_term_centered(msh, P_OK, hdi, kappa, eta, coeff); // s^\Gamma
        auto stab_ill_dofs = make_hho_ill_dofs_stabilization_centered(msh, P_OK, hdi, eta);              // s^N
        auto stab = stab_usual + stab_cut + stab_ill_dofs;
        Mat lc = kappa*(gr.second + stab); 

        // RHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        Vect f = Vect::Zero(lc.rows());

        return std::make_pair(lc, f);

    }

    std::pair<Mat, Vect>
    make_contrib_PKO_centered(const Mesh& msh, Tuple P_KO, const testType &test_case, const hho_degree_info hdi) {

        // CELL INFOS & PARAMETERS
        auto cell_index = std::get<0>(P_KO);
        auto loc = std::get<1>(P_KO);
        auto cl = msh.cells[cell_index];

        // MATERIAL PROPERTIES
        T kappa;
        auto stab_parms = test_case.parms;
        if (std::get<1>(P_KO) == element_location::IN_NEGATIVE_SIDE)
            kappa = 1.0/test_case.parms.kappa_1;
        else 
            kappa = 1.0/test_case.parms.kappa_2;  
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

        // LEVEL SET FUNCTION
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        // HHO OPERATORS
        auto gr = make_hho_gradrec_vector_PKO_centered(msh, P_KO, hdi, level_set_function);
        auto stab_usual = make_hho_stabilization_centered(msh, P_KO, hdi);
        auto stab_cut = make_hho_stabilization_penalty_term_centered(msh, P_KO, hdi, eta, coeff); // s^\Gamma
        auto stab = stab_usual + stab_cut;

        Mat lc = kappa * (gr.second + stab);  

        Vect f = Vect::Zero(lc.rows());

        return std::make_pair(lc, f);

    }

    std::pair<Mat, Vect>
    make_contrib_POK(const Mesh& msh, Tuple P_OK, const testType &test_case, const hho_degree_info hdi) {
        
        // CELL INFOS 
        auto cell_index = std::get<0>(P_OK);
        auto loc = std::get<1>(P_OK);
        auto cl = msh.cells[cell_index];

        // MATERIAL PROPERTIES
        T kappa;
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

        // LEVEL SET FUNCTION
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        // HHO OPERATORS
        auto gr = make_hho_gradrec_vector_POK(msh, P_OK, hdi, level_set_function, coeff);       // G     
        auto stab_usual = make_hho_stabilization(msh, P_OK, hdi);                               // s° 
        auto stab_cut = make_hho_stabilization_penalty_term(msh, P_OK, hdi, kappa, eta, coeff); // s^\Gamma
        auto stab_ill_dofs = make_hho_ill_dofs_stabilization(msh, P_OK, hdi, eta);              // s^N
        auto stab = stab_usual + stab_cut + stab_ill_dofs;
        Mat lc = kappa*(gr.second + stab); 

        // RHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        Vect f = Vect::Zero(lc.rows());

        return std::make_pair(lc, f);

    }

    std::pair<Mat, Vect>
    make_contrib_PKO(const Mesh& msh, Tuple P_KO, const testType &test_case, const hho_degree_info hdi) {

        // CELL INFOS & PARAMETERS
        auto cell_index = std::get<0>(P_KO);
        auto loc = std::get<1>(P_KO);
        auto cl = msh.cells[cell_index];

        // MATERIAL PROPERTIES
        T kappa;
        auto stab_parms = test_case.parms;
        if (std::get<1>(P_KO) == element_location::IN_NEGATIVE_SIDE)
            kappa = 1.0/test_case.parms.kappa_1;
        else 
            kappa = 1.0/test_case.parms.kappa_2;  
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

        // LEVEL SET FUNCTION
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        // HHO OPERATORS
        auto gr = make_hho_gradrec_vector_PKO(msh, P_KO, hdi, level_set_function);
        auto stab_usual = make_hho_stabilization(msh, P_KO, hdi);
        auto stab_cut = make_hho_stabilization_penalty_term(msh, P_KO, hdi, eta, coeff); // s^\Gamma
        auto stab = stab_usual + stab_cut;

        Mat lc = kappa * (gr.second + stab);  
        Vect f = Vect::Zero(lc.rows());
        return std::make_pair(lc, f);

    }

    Vect
    make_contrib_rhs_cut(const Mesh& msh, const typename Mesh::cell_type& cl, const testType &test_case, const hho_degree_info hdi)
    {
        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh, T>::size(celdeg);

        // RHS
        Vect f = Vect::Zero(cbs * 2);
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_NEGATIVE_SIDE);   // Neg part
        f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_POSITIVE_SIDE); // Pos part

        return f;
    }

    Vect
    make_contrib_rhs_cut_centered(const Mesh& msh, const typename Mesh::cell_type& cl, const testType &test_case, const hho_degree_info hdi) {
        
        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        auto celdeg = hdi.cell_degree();
        auto cbs = cut_cell_basis<Mesh, T>::size(celdeg);

        // RHS
        Vect f = Vect::Zero(cbs * 2);
        f.block(0, 0, cbs, 1) += make_rhs_centered(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_NEGATIVE_SIDE);   // Neg part
        f.block(cbs, 0, cbs, 1) += make_rhs_centered(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_POSITIVE_SIDE); // Pos part

        return f;
    }
    
    Mat
    make_contrib_cut_mass(const Mesh& msh, const typename Mesh::cell_type& cl, const hho_degree_info hdi, const testType &test_case) {
        
        Mat mass_neg = make_mass_matrix(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
        Mat mass_pos = make_mass_matrix(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
        mass_neg *= (1.0 / (test_case.parms.c_1 * test_case.parms.c_1 * test_case.parms.kappa_1));
        mass_pos *= (1.0 / (test_case.parms.c_2 * test_case.parms.c_2 * test_case.parms.kappa_2));

        size_t n_data_neg = mass_neg.rows();
        size_t n_data_pos = mass_pos.rows();
        size_t n_data = n_data_neg + n_data_pos;

        Mat mass = Mat::Zero(n_data, n_data);
        mass.block(0, 0, n_data_neg, n_data_neg) = mass_neg;
        mass.block(n_data_neg, n_data_neg, n_data_pos, n_data_pos) = mass_pos;

        return mass;
    }

    Mat
    make_contrib_cut_mass(const Mesh& msh, Tuple P, const hho_degree_info hdi, const testType &test_case) {
        
        // CELL INFOS & PARAMETERS
        auto cell_index = std::get<0>(P);
        auto loc = std::get<1>(P);
        auto cl = msh.cells[cell_index];

        Mat mass_neg = make_mass_matrix(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
        Mat mass_pos = make_mass_matrix(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
        mass_neg *= (1.0 / (test_case.parms.c_1 * test_case.parms.c_1 * test_case.parms.kappa_1));
        mass_pos *= (1.0 / (test_case.parms.c_2 * test_case.parms.c_2 * test_case.parms.kappa_2));

        size_t n_data_neg = mass_neg.rows();
        size_t n_data_pos = mass_pos.rows();
        size_t n_data = n_data_neg + n_data_pos;

        Mat mass = Mat::Zero(n_data, n_data);
        if (loc == element_location::IN_NEGATIVE_SIDE)
            mass.block(0, 0, n_data_neg, n_data_neg) = mass_neg;
        if (loc == element_location::IN_POSITIVE_SIDE)
            mass.block(n_data_neg, n_data_neg, n_data_pos, n_data_pos) = mass_pos;

        return mass;
    }

};
