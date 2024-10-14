
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

        // Gradient Reconstruction 
        std::pair<Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
                  Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>> gr_n;        
        std::pair<Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
                  Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>> gr_p;     

        // EXTENDED GRADIENT RECONSTRUCTIONS
        if (cl.user_data.agglo_set == cell_agglo_set::T_OK) {
            // Gradient reconstruction
            gr_n = make_hho_gradrec_vector_interface_TOK(msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE);
            gr_p = make_hho_gradrec_vector_interface_TOK(msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE);
        }
        if (cl.user_data.agglo_set == cell_agglo_set::T_KO_NEG) { 
            // Gradient reconstruction
            gr_n = make_hho_gradrec_vector_interface_TKOi(msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE);
            gr_p = make_hho_gradrec_vector_interface_TKOibar(msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE, 0.0);
        }
        if (cl.user_data.agglo_set == cell_agglo_set::T_KO_POS) {
            // Gradient reconstruction 
            gr_n = make_hho_gradrec_vector_interface_TKOibar(msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE, 1.0);
            gr_p = make_hho_gradrec_vector_interface_TKOi(msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE);
        } 

        // STABILIZATION PARAMETERS
        auto stab_parms = test_case.parms;
        stab_parms.kappa_1 = 1.0/(parms.kappa_1); // rho_1 = kappa_1
        stab_parms.kappa_2 = 1.0/(parms.kappa_2); // rho_2 = kappa_2  

        // EXTENDED STABILIZATION
        Mat stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, stab_parms);

        // EXTENDED PENALTY
        // // // auto penalty_extended = make_hho_cut_interface_penalty_extended(msh, cl, hdi, eta, stab_parms);
        // // // stab += penalty_extended;

        // STAB + RECONSTRUCTION     
        // std::cout << "DIMENSION STAB: " << stab.size() << std::endl;
        // std::cout << "DIMENSION GRADREC NEG: " << gr_n.second.size() << std::endl;
        // std::cout << "DIMENSION GRADREC POS: " << gr_p.second.size() << std::endl;
        Mat lc = stab_parms.kappa_1*gr_n.second + stab_parms.kappa_2*gr_p.second + stab;
        
        // RIGHT HAND SIDE (PROBABLEMENT A MODIFIER)
        Vect f = Vect::Zero(lc.rows());
        // // // // NEGATIVE SIDE
        // // // f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_NEGATIVE_SIDE);
        // // // f.head(cbs)           -= parms.kappa_1*make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE, level_set_function, dir_jump, eta);
        // // // // POSITIVE SIDE
        // // // f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_POSITIVE_SIDE);
        // // // f.block(cbs, 0, cbs, 1) += parms.kappa_1*make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE, level_set_function, dir_jump, eta);
        // // // f.block(cbs, 0, cbs, 1) += make_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE, test_case.neumann_jump);

        // // // // RHS TERM WITH GRADIENT RECONSTRUCTION
        // // // auto gbs = vector_cell_basis<cuthho_poly_mesh<T>,T>::size(hdi.grad_degree());
        // // // vector_cell_basis<cuthho_poly_mesh<T>, T> gb( msh, cl, hdi.grad_degree() );
        // // // Matrix<T, Dynamic, 1> F_bis = Matrix<T, Dynamic, 1>::Zero(gbs);
        // // // auto iqps = integrate_interface(msh, cl, 2*hdi.grad_degree(), element_location::IN_NEGATIVE_SIDE);
        // // // for (auto& qp : iqps) {
        // // //     const auto g_phi = gb.eval_basis(qp.first);
        // // //     const Matrix<T,2,1> n = level_set_function.normal(qp.first);
        // // //     F_bis += qp.second * dir_jump(qp.first) * g_phi * n;
        // // // }
        // // // f -= F_bis.transpose() * (parms.kappa_1*gr_n.first );

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
auto make_gradrec_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                   testType test_case) {
    return gradrec_interface_method<T, ET, testType>(eta_);
}




#endif
