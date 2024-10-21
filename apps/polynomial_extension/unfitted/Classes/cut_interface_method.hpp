
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
        auto gr = make_hho_gradrec_vector_POK(msh, P_OK, hdi, level_set_function, kappa, stab_parms);  
        auto stab_usual = make_hho_stabilization(msh, P_OK, hdi, stab_parms);                     // s° 
        auto stab_cut = make_hho_stabilization_penalty_term(msh, P_OK, hdi, kappa, eta, stab_parms);   // s^\Gamma
        auto stab_ill_dofs = make_hho_ill_dofs_stabilization(msh, P_OK, hdi, eta, stab_parms);         // s^N

        Mat lc = kappa*(gr.second + stab_usual) + stab_cut + stab_ill_dofs; 
        Mat f  = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun); // A DEBUG

        // ///////////////    RHS
        // Vect f = Vect::Zero(lc.rows());
        // // neg part
        // f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
        //                                   element_location::IN_NEGATIVE_SIDE);
        // // we use element_location::IN_POSITIVE_SIDE to get rid of the Nitsche term
        // // (see definition of make_Dirichlet_jump)
        // f.head(cbs) -= parms.kappa_1 *
        //     make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
        //                         level_set_function, dir_jump, eta);

        // // pos part
        // f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
        //                                    element_location::IN_POSITIVE_SIDE);
        // f.block(cbs, 0, cbs, 1) += parms.kappa_1 *
        //     make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
        //                         level_set_function, dir_jump, eta);
        // f.block(cbs, 0, cbs, 1)
        //     += make_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
        //                             test_case.neumann_jump);


        // // rhs term with GR
        // auto gbs = vector_cell_basis<cuthho_poly_mesh<T>,T>::size(hdi.grad_degree());
        // vector_cell_basis<cuthho_poly_mesh<T>, T> gb( msh, cl, hdi.grad_degree() );
        // Matrix<T, Dynamic, 1> F_bis = Matrix<T, Dynamic, 1>::Zero( gbs );
        // auto iqps = integrate_interface(msh, cl, 2*hdi.grad_degree(),
        //                                 element_location::IN_NEGATIVE_SIDE);
        // for (auto& qp : iqps)
        // {
        //     const auto g_phi    = gb.eval_basis(qp.first);
        //     const Matrix<T,2,1> n      = level_set_function.normal(qp.first);

        //     F_bis += qp.second * dir_jump(qp.first) * g_phi * n;
        // }
        // f -= F_bis.transpose() * (parms.kappa_1 * gr_n.first );

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
          
        // AJOUTER SECOND MEMBRE CORRECTEMENT 
        Mat f  = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);

        return std::make_pair(lc, f);
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
        // // // f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_NEGATIVE_SIDE);
        // // // // pos part
        // // // f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_POSITIVE_SIDE);

        return f;
    }

};

template<typename T, size_t ET, typename testType>
auto make_cut_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                   testType test_case) {
    return cut_interface_method<T, ET, testType>(eta_);
}

#endif
