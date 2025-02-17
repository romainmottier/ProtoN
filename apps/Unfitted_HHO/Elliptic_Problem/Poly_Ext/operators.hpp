
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

        // TEST CASE
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
        Vect f = Vect::Zero(lc.rows());
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        size_t offset = 0.0;
        if (is_cut(msh,cl) && loc == element_location::IN_POSITIVE_SIDE)
            offset = cbs;
        f.block(offset, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, loc);

        // // JUMP TERMS 
        // if (is_cut(msh, cl)) {
        //     if (loc == element_location::IN_NEGATIVE_SIDE) {
        //         f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_NEGATIVE_SIDE);
        //         f.head(cbs) -= stab_parms.kappa_1 * make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE, level_set_function, dir_jump, eta);
        //     }
        //     if (loc == element_location::IN_POSITIVE_SIDE) {
        //         f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_POSITIVE_SIDE);
        //         f.block(cbs, 0, cbs, 1) += stab_parms.kappa_1 * make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE, level_set_function, dir_jump, eta);
        //         f.block(cbs, 0, cbs, 1) += make_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE, test_case.neumann_jump);
        //     }
        // }
        // #if(!centering_bases)
        // auto gbs = vector_cell_basis<cuthho_poly_mesh<T>,T>::size(hdi.grad_degree());
        // vector_cell_basis<cuthho_poly_mesh<T>, T> gb( msh, cl, hdi.grad_degree());
        // #else
        // auto gbs = cut_vector_cell_basis<cuthho_poly_mesh<T>,T>::size(hdi.grad_degree());
        // cut_vector_cell_basis<cuthho_poly_mesh<T>, T> gb( msh, cl, hdi.grad_degree(), loc);
        // #endif
        // Matrix<T, Dynamic, 1> F_bis = Matrix<T, Dynamic, 1>::Zero( gbs );
        // auto iqps = integrate_interface(msh, cl, 2*hdi.grad_degree(), element_location::IN_NEGATIVE_SIDE);
        // for (auto& qp : iqps) {
        //     const auto g_phi = gb.eval_basis(qp.first);
        //     const Matrix<T,2,1> n = level_set_function.normal(qp.first);
        //     F_bis += qp.second * dir_jump(qp.first) * g_phi * n;
        // }
        // f -= coeff * F_bis.transpose() * (stab_parms.kappa_1 * gr.first );

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

        // RHS
        Vect f = Vect::Zero(lc.rows());
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        size_t offset = 0.0;
        if (is_cut(msh,cl) && loc == element_location::IN_POSITIVE_SIDE)
            offset = cbs;
        f.block(offset, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, loc);

        return std::make_pair(lc, f);

    }

};

template<typename T, size_t ET, typename testType>
auto make_gradrec_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_, testType test_case) {
    return gradrec_interface_method<T, ET, testType>(eta_);
}

