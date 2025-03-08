
template<typename T, size_t ET, typename testType>
class interface_method {

    using Mat  = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

protected:

    interface_method(){}

    virtual std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl, const testType test_case, const hho_degree_info hdi) {
    }

public:

    std::pair<Mat, Vect>
    make_contrib_uncut(const Mesh& msh, const typename Mesh::cell_type& cl, const hho_degree_info hdi, const testType test_case) {
        
        T kappa;

        if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
            kappa = test_case.parms.kappa_1;
        else
            kappa = test_case.parms.kappa_2;

        auto gr = make_hho_gradrec_vector(msh, cl, hdi);
        Mat stab = make_hho_naive_stabilization(msh, cl, hdi);
        Mat lc = kappa * (gr.second + stab);
        Mat f = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);
        return std::make_pair(lc, f);

    }
    
};

template<typename T, size_t ET, typename testType>
class gradrec_interface_method : public interface_method<T, ET, testType> {

    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;
    using Tuple = std::tuple<double,element_location,std::vector<double>>;

public:

    T eta;

    gradrec_interface_method(T eta_) : interface_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl, const testType test_case, const hho_degree_info hdi) {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        ///////////////    LHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        // GR
        auto gr_n = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE, 1.0);
        auto gr_p = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE, 0.0);

        // stab
        Mat stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, parms);

        Mat penalty = make_hho_cut_interface_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
        stab.block(0, 0, cbs, cbs) += parms.kappa_1 * penalty;
        stab.block(0, cbs, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, 0, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, cbs, cbs, cbs) += parms.kappa_1 * penalty;

        Mat lc = stab + parms.kappa_1 * gr_n.second + parms.kappa_2 * gr_p.second;

        ///////////////    RHS
        Vect f = Vect::Zero(lc.rows());
        // neg part
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_NEGATIVE_SIDE);
        f.head(cbs) -= parms.kappa_1 * make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE, level_set_function, dir_jump, eta);
        // pos part
        f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_POSITIVE_SIDE);
        f.block(cbs, 0, cbs, 1) += parms.kappa_1 * make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE, level_set_function, dir_jump, eta);
        f.block(cbs, 0, cbs, 1) += make_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE, test_case.neumann_jump);

        // rhs term with GR
        auto gbs = vector_cell_basis<cuthho_poly_mesh<T>,T>::size(hdi.grad_degree());
        vector_cell_basis<cuthho_poly_mesh<T>, T> gb( msh, cl, hdi.grad_degree() );
        Matrix<T, Dynamic, 1> F_bis = Matrix<T, Dynamic, 1>::Zero( gbs );
        auto iqps = integrate_interface(msh, cl, 2*hdi.grad_degree(), element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : iqps) {
            const auto g_phi = gb.eval_basis(qp.first);
            const Matrix<T,2,1> n = level_set_function.normal(qp.first);
            F_bis += qp.second * dir_jump(qp.first) * g_phi * n;
        }
        f -= F_bis.transpose() * (parms.kappa_1 * gr_n.first );

        return std::make_pair(lc, f);

    }

    std::pair<Mat, Vect>
    make_contrib_POK(const Mesh& msh, Tuple P_OK, const testType &test_case, const hho_degree_info hdi) {
        
        // CELL INFOS 
        auto cell_index = std::get<0>(P_OK);
        auto loc = std::get<1>(P_OK);
        auto cl = msh.cells[cell_index];
        bool POK = true;

        // MATERIAL PROPERTIES
        T kappa, coeff;
        T kappa_1 = test_case.parms.kappa_1;
        if (loc == element_location::IN_NEGATIVE_SIDE) {
            kappa = test_case.parms.kappa_1;
            coeff = 1.0;
        }
        else {
            kappa = test_case.parms.kappa_2;
            coeff = 0.0;
        }
        
        // HHO OPERATORS
        auto gr = make_hho_gradrec_vector_POK(msh, P_OK, hdi, test_case.level_set_, coeff);     // G     
        auto stab_usual = make_hho_stabilization(msh, P_OK, hdi);                               // s° 
        auto stab_cut = make_hho_stabilization_penalty_term(msh, P_OK, hdi, eta, coeff); // s^\Gamma
        auto stab_ill_dofs = make_hho_ill_dofs_stabilization(msh, P_OK, hdi, eta);              // s^N
        Mat lc = kappa*(gr.second + stab_usual + stab_ill_dofs) + kappa_1*stab_cut; 

        // RHS
        auto f = make_rhs_jumps(msh, P_OK, hdi, gr.first, POK, test_case, eta);

        return std::make_pair(lc, f);

    }

    std::pair<Mat, Vect>
    make_contrib_PKO(const Mesh& msh, Tuple P_KO, const testType &test_case, const hho_degree_info hdi) {

        // CELL INFOS & PARAMETERS
        auto cell_index = std::get<0>(P_KO);
        auto loc = std::get<1>(P_KO);
        auto cl = msh.cells[cell_index];
        bool POK = false;

        // MATERIAL PROPERTIES
        T kappa, coeff;
        T kappa_1 = test_case.parms.kappa_1;
        if (loc == element_location::IN_NEGATIVE_SIDE) {
            kappa = test_case.parms.kappa_1;
            coeff = 1.0;
        }
        else {
            kappa = test_case.parms.kappa_2;
            coeff = 0.0;
        }

        // LEVEL SET FUNCTION
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        // HHO OPERATORS
        auto gr = make_hho_gradrec_vector_PKO(msh, P_KO, hdi, level_set_function);
        auto stab_usual = make_hho_stabilization(msh, P_KO, hdi);
        auto stab_cut = make_hho_stabilization_penalty_term(msh, P_KO, hdi, eta, coeff); // s^\Gamma
 
        Mat lc = 0.0*(kappa * (gr.second + stab_usual) + kappa_1*stab_cut); 

        // RHS
        auto f = 0.0*make_rhs_jumps(msh, P_KO, hdi, gr.first, POK, test_case, eta);

        return std::make_pair(lc, f);

    }

};

template<typename T, size_t ET, typename testType>
auto make_gradrec_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_, testType test_case) {
    return gradrec_interface_method<T, ET, testType>(eta_);
}

