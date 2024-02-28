
#ifndef gradrec_mixed_interface_method_hpp
#define gradrec_mixed_interface_method_hpp

template<typename T, size_t ET, typename testType>
class gradrec_mixed_interface_method : public mixed_interface_method<T, ET, testType>
{
    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:
    T eta;

    gradrec_mixed_interface_method(T eta_)
        : mixed_interface_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType & test_case, const hho_degree_info hdi) {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        ///////////////    LHS
        const auto celdeg  = hdi.cell_degree();
        const auto facdeg  = hdi.face_degree();
        const auto graddeg = hdi.grad_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto rbs = vector_cell_basis<Mesh,T>::size(graddeg);

        // GRADIENT RECONSTRUCTION
        auto gr_n = make_hho_gradrec_mixed_vector_interface(msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE, 1.0);
        auto gr_p = make_hho_gradrec_mixed_vector_interface(msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE, 0.0);

        // STABILIZATION
        auto stab_parms = test_case.parms;
        stab_parms.kappa_1 = 1.0/(parms.c_1*parms.kappa_1);// rho_1 = kappa_1
        stab_parms.kappa_2 = 1.0/(parms.c_2*parms.kappa_2);// rho_2 = kappa_2
        Mat stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, stab_parms, scaled_stab_Q);
        
        T penalty_scale = std::min(1.0/(parms.c_1*parms.kappa_1), 1.0/(parms.c_2*parms.kappa_2));
        Mat penalty = make_hho_cut_interface_penalty(msh, cl, hdi, eta, scaled_stab_Q).block(0, 0, cbs, cbs);
        stab.block(0, 0, cbs, cbs) += penalty_scale* penalty;
        stab.block(0, cbs, cbs, cbs) -= penalty_scale * penalty;
        stab.block(cbs, 0, cbs, cbs) -= penalty_scale * penalty;
        stab.block(cbs, cbs, cbs, cbs) += penalty_scale * penalty;

        Matrix<T, Dynamic, Dynamic> R_operator = gr_n.second + gr_p.second;
        auto n_rows = R_operator.rows();
        auto n_cols = R_operator.cols();
        
        Matrix<T, Dynamic, Dynamic> S_operator = Matrix<T, Dynamic, Dynamic>::Zero(n_rows, n_cols);
        auto stabilization_operator    = stab;
        auto n_s_rows = stabilization_operator.rows();
        auto n_s_cols = stabilization_operator.cols();
        S_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, n_s_rows, n_s_cols) = stabilization_operator;

        // GRADREC + STAB
        Mat lc = R_operator + S_operator;

        /////////////// RHS
        Vect f = Vect::Zero((cbs+rbs)*2);
        std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << rbs << std::endl;
        std::cout << "rbs = " << rbs << std::endl;
        // NEGATIVE PART
        f.block(2*rbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_NEGATIVE_SIDE);
        
        // POSITIVE PART
        f.block(2*rbs+cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_POSITIVE_SIDE);


        return std::make_pair(lc, f);
    }

    Vect
    make_contrib_rhs_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType &test_case, const hho_degree_info hdi)
    {

        const auto celdeg  = hdi.cell_degree();
        const auto facdeg  = hdi.face_degree();
        const auto graddeg = hdi.grad_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto rbs = vector_cell_basis<Mesh,T>::size(graddeg);

        ///////////////    RHS
        Vect f = Vect::Zero((cbs+rbs)*2);
        // neg part
        f.block(2*rbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                          element_location::IN_NEGATIVE_SIDE);
        // pos part
        f.block(2*rbs+cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                           element_location::IN_POSITIVE_SIDE);

        return f;
    }
};

template<typename T, size_t ET, typename testType>
auto make_gradrec_mixed_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                   testType & test_case) {
    return gradrec_mixed_interface_method<T, ET, testType>(eta_);
}

#endif
