
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
        auto gr = make_hho_gradrec_vector_PKO(msh, P_KO, hdi, level_set_function);       // G
        // auto stab_usual = make_hho_stabilization(msh, P_KO, hdi);                        // s°
        auto stab_cut = make_hho_stabilization_penalty_term(msh, P_KO, hdi, eta, coeff); // s^\Gamma
        // Mat lc = kappa * (gr.second + stab_usual) + kappa_1*stab_cut; 
        Mat lc = kappa * (gr.second) + kappa_1*stab_cut; 

        // RHS
        auto f = make_rhs_jumps(msh, P_KO, hdi, gr.first, POK, test_case, eta);

        return std::make_pair(lc, f);

    }

};

template<typename T, size_t ET, typename testType>
auto make_gradrec_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_, testType test_case) {
    return gradrec_interface_method<T, ET, testType>(eta_);
}

