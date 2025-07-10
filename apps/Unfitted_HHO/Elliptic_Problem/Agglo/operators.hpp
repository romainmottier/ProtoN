
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
        auto k = hdi.cell_degree();

        if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
            kappa = test_case.parms.kappa_1;
        else
            kappa = test_case.parms.kappa_2;

        auto gr = make_hho_gradrec_vector(msh, cl, hdi);
        Mat stab = make_hho_naive_stabilization(msh, cl, hdi);
        Mat lc = kappa * (gr.second + k*stab);
        Mat f = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);
        return std::make_pair(lc, f);

    }


    std::pair<Mat, Vect>
    make_contrib(const Mesh& msh, const typename Mesh::cell_type& cl, const testType test_case, const hho_degree_info hdi) {
        
        if (location(msh, cl) != element_location::ON_INTERFACE)
            return make_contrib_uncut(msh, cl, hdi, test_case);
        else 
            return make_contrib_cut(msh, cl, test_case, hdi);
    }
    
};

template<typename T, size_t ET, typename testType>
class gradrec_interface_method : public interface_method<T, ET, testType> {

    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:

    T eta;

    gradrec_interface_method(T eta_) : interface_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl, const testType test_case, const hho_degree_info hdi) {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;
        auto k = hdi.cell_degree();

        ///////////////    LHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        // GR
        auto gr_n = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE, 1.0);
        auto gr_p = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE, 0.0);

        // stab
        Mat stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, parms);
 
        #ifndef subcell_centering
        Mat penalty = make_hho_cut_interface_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
        stab.block(0, 0, cbs, cbs) += parms.kappa_1 * penalty;
        stab.block(0, cbs, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, 0, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, cbs, cbs, cbs) += parms.kappa_1 * penalty;
        #else
        Mat penalty = make_hho_cut_interface_penalty(msh, cl, hdi, eta).block(0, 0, 2*cbs, 2*cbs);
        stab.block(0, 0, 2*cbs, 2*cbs) += parms.kappa_1 * penalty;
        #endif


        Mat lc = k*stab + parms.kappa_1 * gr_n.second + parms.kappa_2 * gr_p.second;

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
        #ifndef subcell_centering        
        vector_cell_basis<cuthho_poly_mesh<T>, T> gb(msh, cl, hdi.grad_degree());
        #else
        cut_vector_cell_basis<cuthho_poly_mesh<T>, T> gb(msh, cl, hdi.grad_degree(), element_location::IN_NEGATIVE_SIDE);
        #endif
        size_t cpt = 0;
        auto dn = get_discrete_normal(msh, cl, 2*hdi.grad_degree(), element_location::IN_NEGATIVE_SIDE); 
        Matrix<T, Dynamic, 1> F_bis = Matrix<T, Dynamic, 1>::Zero(gbs);
        auto iqps = integrate_interface(msh, cl, 2*hdi.grad_degree(), element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : iqps) {
            const auto g_phi = gb.eval_basis(qp.first);
            Matrix<T,2,1> n = level_set_function.normal(qp.first);
            n = dn[cpt];
            F_bis += qp.second * dir_jump(qp.first) * g_phi * n;
            cpt++;
        }
        f -= F_bis.transpose() * (parms.kappa_1 * gr_n.first );

        return std::make_pair(lc, f);

    }

};

template<typename T, size_t ET, typename testType>
auto make_gradrec_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_, testType test_case) {
    return gradrec_interface_method<T, ET, testType>(eta_);
}

