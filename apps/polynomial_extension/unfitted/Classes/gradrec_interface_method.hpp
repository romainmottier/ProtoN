
#ifndef gradrec_interface_method_hpp
#define gradrec_interface_method_hpp

template<typename T, size_t ET, typename testType>
class gradrec_interface_method : public interface_method<T, ET, testType>
{
    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:
    T eta;

    gradrec_interface_method(T eta_)
        : interface_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType &test_case, const hho_degree_info hdi) {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        ///////////////    LHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        // Gradient Reconstruction : initialisation for TOK
        auto gr_n = make_hho_gradrec_vector_interface_TOK(msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE);
        auto gr_p = make_hho_gradrec_vector_interface_TOK(msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE);
        if (cl.user_data.agglo_set == cell_agglo_set::T_KO_NEG) {// remplacer element_location par i ou bar{i}
            gr_n = make_hho_gradrec_vector_interface_TKOi(msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE);
            gr_p = make_hho_gradrec_vector_interface_extended(msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE);
        }
        if (cl.user_data.agglo_set == cell_agglo_set::T_KO_POS) {
            gr_n = make_hho_gradrec_vector_interface_extended(msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE);
            gr_p = make_hho_gradrec_vector_interface_extended(msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE);
        }


        // stab
        auto stab_parms = test_case.parms;
        stab_parms.kappa_1 = 1.0/(parms.kappa_1);// rho_1 = kappa_1
        stab_parms.kappa_2 = 1.0/(parms.kappa_2);// rho_2 = kappa_2
        Mat stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, stab_parms);
        
        T penalty_scale = std::min(1.0/(parms.kappa_1), 1.0/(parms.kappa_2));
        Mat penalty = make_hho_cut_interface_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
        stab.block(0, 0, cbs, cbs)     += penalty_scale* penalty;
        stab.block(0, cbs, cbs, cbs)   -= penalty_scale * penalty;
        stab.block(cbs, 0, cbs, cbs)   -= penalty_scale * penalty;
        stab.block(cbs, cbs, cbs, cbs) += penalty_scale * penalty;

        Mat lc = stab + stab_parms.kappa_1 * gr_n.second + stab_parms.kappa_2 * gr_p.second;
        
        ///////////////    RHS
        Vect f = Vect::Zero(lc.rows());
        // neg part
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                          element_location::IN_NEGATIVE_SIDE);
        // we use element_location::IN_POSITIVE_SIDE to get rid of the Nitsche term
        // (see definition of make_Dirichlet_jump)
        f.head(cbs) -= parms.kappa_1 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);

        // pos part
        f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                           element_location::IN_POSITIVE_SIDE);
        f.block(cbs, 0, cbs, 1) += parms.kappa_1 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);
        f.block(cbs, 0, cbs, 1)
            += make_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                    test_case.neumann_jump);


        // rhs term with GR
        auto gbs = vector_cell_basis<cuthho_poly_mesh<T>,T>::size(hdi.grad_degree());
        vector_cell_basis<cuthho_poly_mesh<T>, T> gb( msh, cl, hdi.grad_degree() );
        Matrix<T, Dynamic, 1> F_bis = Matrix<T, Dynamic, 1>::Zero( gbs );
        auto iqps = integrate_interface(msh, cl, 2*hdi.grad_degree(),
                                        element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : iqps)
        {
            const auto g_phi    = gb.eval_basis(qp.first);
            const Matrix<T,2,1> n      = level_set_function.normal(qp.first);

            F_bis += qp.second * dir_jump(qp.first) * g_phi * n;
        }
        f -= F_bis.transpose() * (parms.kappa_1 * gr_n.first );

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
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                          element_location::IN_NEGATIVE_SIDE);
//        // we use element_location::IN_POSITIVE_SIDE to get rid of the Nitsche term
//        // (see definition of make_Dirichlet_jump)
//        f.head(cbs) -= parms.kappa_1 *
//            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
//                                level_set_function, dir_jump, eta);

        // pos part
        f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                           element_location::IN_POSITIVE_SIDE);
//        f.block(cbs, 0, cbs, 1) += parms.kappa_1 *
//            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
//                                level_set_function, dir_jump, eta);
//        f.block(cbs, 0, cbs, 1)
//            += make_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
//                                    test_case.neumann_jump);

        return f;
    }
};



template<typename T, size_t ET, typename testType>
auto make_gradrec_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                   testType test_case) {
    return gradrec_interface_method<T, ET, testType>(eta_);
}




#endif
