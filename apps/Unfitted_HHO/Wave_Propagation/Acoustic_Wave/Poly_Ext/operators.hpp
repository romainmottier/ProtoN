
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

    Mat
    make_contrib_mass(const Mesh& msh, const typename Mesh::cell_type& cl, const testType &test_case, const hho_degree_info hdi) {

        if( location(msh, cl) != element_location::ON_INTERFACE )
            return make_contrib_uncut_mass(msh, cl, hdi, test_case);
        else 
            return make_contrib_cut_mass(msh, cl, hdi, test_case);

    }

    Mat
    make_contrib_uncut_mass(const Mesh& msh, const typename Mesh::cell_type& cl, const hho_degree_info hdi, const testType &test_case) {

        T c;
        if (location(msh, cl) == element_location::IN_NEGATIVE_SIDE)
            c = test_case.parms.c_1;
        else
            c = test_case.parms.c_2;

        Mat mass = make_mass_matrix(msh, cl, hdi.cell_degree());
        mass *= (1.0/(c*c*test_case.parms.kappa_1));

        return mass;

    }
    
    Mat
    make_contrib_cut_mass(const Mesh& msh, const typename Mesh::cell_type& cl, const hho_degree_info hdi, const testType &test_case) {

        Mat mass_neg = make_mass_matrix(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
        Mat mass_pos = make_mass_matrix(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
        mass_neg *= (1.0/(test_case.parms.c_1*test_case.parms.c_1*test_case.parms.kappa_1));
        mass_pos *= (1.0/(test_case.parms.c_2*test_case.parms.c_2*test_case.parms.kappa_2));
        
        size_t n_data_neg = mass_neg.rows();
        size_t n_data_pos = mass_pos.rows();
        size_t n_data = n_data_neg + n_data_pos;
        
        Mat mass = Mat::Zero(n_data,n_data);
        mass.block(0,0,n_data_neg,n_data_neg) = mass_neg;
        mass.block(n_data_neg,n_data_neg,n_data_pos,n_data_pos) = mass_pos;

        return mass;

    }

    Vect
    make_contrib_rhs(const Mesh& msh, const typename Mesh::cell_type& cl, const testType &test_case, const hho_degree_info hdi) {

        if (location(msh, cl) != element_location::ON_INTERFACE)
            return make_contrib_rhs_uncut(msh, cl, hdi, test_case);
        else 
            return make_contrib_rhs_cut(msh, cl, test_case, hdi);

    }
    
    Vect
    make_contrib_rhs_uncut(const Mesh& msh, const typename Mesh::cell_type& cl, const hho_degree_info hdi, const testType &test_case) {
        
        Mat f = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);
        
        return f;
    
    }

    Vect
    make_contrib_rhs_cut(const Mesh& msh, const typename Mesh::cell_type& cl, const testType &test_case, const hho_degree_info hdi) {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        ///////////////    RHS
        Vect f = Vect::Zero(cbs*2);
        // neg part
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_NEGATIVE_SIDE);
//        // we use element_location::IN_POSITIVE_SIDE to get rid of the Nitsche term
//        // (see definition of make_Dirichlet_jump)
//        f.head(cbs) -= parms.kappa_1 *
//            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
//                                level_set_function, dir_jump, eta);

        // pos part
        f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_POSITIVE_SIDE);
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
class gradrec_interface_method : public interface_method<T, ET, testType> {

    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;
    using Tuple = std::tuple<double,element_location,std::vector<double>>;

public:

    T eta;

    gradrec_interface_method(T eta_) : interface_method<T,ET,testType>(), eta(eta_) {}

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
        auto gr = make_hho_gradrec_vector_POK(msh, P_OK, hdi, test_case.level_set_, coeff); // G     
        auto stab_usual = make_hho_stabilization(msh, P_OK, hdi);                           // s° 
        auto stab_cut = make_hho_stabilization_penalty_term(msh, P_OK, hdi, eta, coeff);    // s^\Gamma
        auto stab_ill_dofs = make_hho_ill_dofs_stabilization(msh, P_OK, hdi, eta);          // s^N
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
 
        Mat lc = kappa * (gr.second + stab_usual) + kappa_1*stab_cut; 

        // RHS
        auto f = make_rhs_jumps(msh, P_KO, hdi, gr.first, POK, test_case, eta);

        return std::make_pair(lc, f);

    }

};

template<typename T, size_t ET, typename testType>
auto make_gradrec_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_, testType test_case) {
    return gradrec_interface_method<T, ET, testType>(eta_);
}
