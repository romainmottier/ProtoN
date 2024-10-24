
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
    make_contrib_POK(const Mesh& msh, Tuple P_OK, const testType &test_case, const hho_degree_info hdi) {
        
        // CELL INFOS 
        auto cell_index = std::get<0>(P_OK);
        auto loc = std::get<1>(P_OK);
        auto cl = msh.cells[cell_index];

        // MATERIAL PROPERTIES
        T kappa;
        if (loc == element_location::IN_NEGATIVE_SIDE)
            kappa = test_case.parms.kappa_1;
        else
            kappa = test_case.parms.kappa_2;
        auto coeff = 0.0;
        if (test_case.parms.kappa_1 < test_case.parms.kappa_2) {
            if (loc == element_location::IN_NEGATIVE_SIDE)
                coeff = 1.0;
        }
        else {
            if (loc == element_location::IN_POSITIVE_SIDE)
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
        Mat lc = kappa*(gr.second + stab_usual + stab_cut) + stab_ill_dofs; 

        // RHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        Vect f = Vect::Zero(lc.rows());
        // if (is_cut(msh,cl)) {
        //     if (loc == element_location::IN_NEGATIVE_SIDE) {
        //         f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_NEGATIVE_SIDE);
        //         f.head(cbs) -= ((1.0/stab_parms.kappa_1)) * make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,level_set_function, dir_jump, eta);
        //     }
        //     if (loc == element_location::IN_POSITIVE_SIDE) {
        //         f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_POSITIVE_SIDE);
        //         f.block(cbs, 0, cbs, 1) += (1.0/stab_parms.kappa_1) * make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,level_set_function, dir_jump, eta);
        //         f.block(cbs, 0, cbs, 1) += make_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE, test_case.neumann_jump);
        //     }
        // }
        // else 
        //     f = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);

        return std::make_pair(lc, f);
    }

    std::pair<Mat, Vect>
    make_blabla(const Mesh& msh, Tuple P_OK, const testType &test_case, const hho_degree_info hdi) {

        // CELL INFOS & PARAMETERS
        auto cell_index = std::get<0>(P_OK);
        auto loc = std::get<1>(P_OK);
        auto cl = msh.cells[cell_index];

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        // LHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        // GR
        T factor = 0.0;
        if (1.0/(parms.kappa_1) < 1.0/(parms.kappa_2)) {
            factor = 1.0;
        }
        auto gr_n = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_NEGATIVE_SIDE, 1.0-factor);
        auto gr_p = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_POSITIVE_SIDE, factor);

        // stab
        auto stab_parms = test_case.parms;
        stab_parms.kappa_1 = 1.0/(parms.kappa_1); // rho_1 = kappa_1
        stab_parms.kappa_2 = 1.0/(parms.kappa_2); // rho_2 = kappa_2
        Mat stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, stab_parms);
        
        T penalty_scale = std::min(1.0/(parms.kappa_1), 1.0/(parms.kappa_2));
        Mat penalty = make_hho_cut_interface_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
        stab.block(0, 0, cbs, cbs) += penalty_scale * penalty;
        stab.block(0, cbs, cbs, cbs) -= penalty_scale * penalty;
        stab.block(cbs, 0, cbs, cbs) -= penalty_scale * penalty;
        stab.block(cbs, cbs, cbs, cbs) += penalty_scale * penalty;

        Mat lc = stab + stab_parms.kappa_1 * gr_n.second + stab_parms.kappa_2 * gr_p.second;

        // RHS
        Vect f = Vect::Zero(lc.rows());
        // neg part
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                          element_location::IN_NEGATIVE_SIDE);
        f.head(cbs) -= parms.kappa_1 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);

        // pos part
        f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                           element_location::IN_POSITIVE_SIDE);
        f.block(cbs, 0, cbs, 1) += parms.kappa_1 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);
        f.block(cbs, 0, cbs, 1) += make_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                    test_case.neumann_jump);

        // rhs term with GR
        auto gbs = vector_cell_basis<cuthho_poly_mesh<T>,T>::size(hdi.grad_degree());
        vector_cell_basis<cuthho_poly_mesh<T>, T> gb(msh, cl, hdi.grad_degree());
        Matrix<T, Dynamic, 1> F_bis = Matrix<T, Dynamic, 1>::Zero(gbs);
        auto iqps = integrate_interface(msh, cl, 2 * hdi.grad_degree(),
                                        element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : iqps)
        {
            const auto g_phi = gb.eval_basis(qp.first);
            const Matrix<T, 2, 1> n = level_set_function.normal(qp.first);

            F_bis += qp.second * dir_jump(qp.first) * g_phi * n;
        }
        f -= F_bis.transpose() * (parms.kappa_1 * gr_n.first);

        return std::make_pair(lc, f);
    }

    std::pair<Mat, Vect>
    make_contrib_cut_PKO(const Mesh& msh, Tuple P_KO, const testType &test_case, const hho_degree_info hdi) {

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
        auto stab = make_hho_stabilization(msh, P_KO, hdi);
        auto stab_cut = make_hho_stabilization_penalty_term(msh, P_KO, hdi, eta, coeff); // s^\Gamma

        Mat lc = kappa * (gr.second + stab + stab_cut);  
        
        // RHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        Vect f = Vect::Zero(lc.rows());
        if (loc == element_location::IN_NEGATIVE_SIDE) {
            f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_NEGATIVE_SIDE);
            f.head(cbs) -= ((1.0/stab_parms.kappa_1)) * make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,level_set_function, dir_jump, eta);
        }
        if (loc == element_location::IN_POSITIVE_SIDE) {
            f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_POSITIVE_SIDE);
            f.block(cbs, 0, cbs, 1) += (1.0/stab_parms.kappa_1) * make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,level_set_function, dir_jump, eta);
            f.block(cbs, 0, cbs, 1) += make_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE, test_case.neumann_jump);
        }

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
        // neg part
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                          element_location::IN_NEGATIVE_SIDE);

        // pos part
        f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                           element_location::IN_POSITIVE_SIDE);

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

};
